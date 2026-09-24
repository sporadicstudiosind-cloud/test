"""Macro-router, focus head and ponder controller.

What this decides, per token, from the control core's mid-stack state:

* **which** superstacks see it (``top_k`` of ``n_stacks``),
* **how deep** into them it goes (the focus ladder),
* **whether** it goes round again (the ponder loop).

Three corrections to the source plan are implemented here, and each one is a
property a test can fail on.

**C-1 · The focus statistic must be causal.**
The plan computed focus from ``h_norm.mean(dim=1)`` — a mean over the whole
chunk — and then used that scalar to set the execution depth. During teacher
forcing that mean contains future tokens, so the depth chosen for token ``t``
depends on tokens ``t+1..T`` which do not exist at sampling time. The model
would learn a policy it cannot execute, and training-time and serving-time
behaviour would silently diverge. Here focus is computed from the token's own
state, optionally blended with a *prefix* (cumulative) mean, which is causal by
construction. ``test_router.py::test_focus_is_causal`` perturbs the future and
asserts the decision is unchanged.

**C-2 · Top-k *over experts* is causal; top-k *over the sequence* is not.**
This router selects k of N stacks for each token independently, so it has no
cross-token dependence. It deliberately does **not** implement capacity-based
token dropping by default: a capacity cutoff ranks tokens against each other
within a chunk and reintroduces exactly the non-causality that Mixture-of-Depths
(Raposo et al. 2024) has to patch with an auxiliary predictor. If capacity is
enabled, ``capacity_factor > 0``, the class refuses to run in causal mode.

**C-3 · Load balancing needs the coefficient and all k choices.**
The Switch Transformer objective is ``alpha * N * sum_i f_i * P_i`` with ``f_i``
the *dispatch* fraction. The plan dropped ``alpha`` and computed ``f_i`` from
the top-1 choice only while dispatching top-2, so the loss was blind to half of
the traffic it was supposed to balance.

The gating weights are a softmax over the *selected* logits without the Gumbel
noise (noise is used for the selection, not the weighting); mixing the two
biases the combination weights toward whatever the noise favoured.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import RouterConfig
from .layers import RMSNorm, at_least_fp32


@dataclass
class RoutingDecision:
    """Everything stage II and the dispatcher need, plus what the loss needs."""

    stack_index: torch.Tensor      # [B, T, k] long
    stack_weight: torch.Tensor     # [B, T, k] float, sums to 1 over k
    focus: torch.Tensor            # [B, T] in (0, 1)
    target_depth: torch.Tensor     # [B, T] long, in [min_depth, n_layers]
    halt_logit: torch.Tensor       # [B, T] raw logit for the ponder loop
    balance_loss: torch.Tensor     # scalar
    z_loss: torch.Tensor           # scalar
    gate_probs: torch.Tensor       # [B, T, N] full softmax, for diagnostics

    valid: torch.Tensor | None = None

    def entropy(self) -> torch.Tensor:
        p = self.gate_probs.clamp_min(1e-9)
        return -(p * p.log()).sum(-1).mean()

    def dispatch_fraction(self, n_stacks: int) -> torch.Tensor:
        one_hot = F.one_hot(self.stack_index, n_stacks).sum(2).float()
        return one_hot.mean(dim=(0, 1))


class MacroRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_stacks: int,
        cfg: RouterConfig,
        min_depth: int,
        max_depth: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_stacks = n_stacks
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.norm = RMSNorm(d_model)
        self.gate = nn.Linear(d_model, n_stacks, bias=False)
        self.focus_head = nn.Linear(d_model, 1)
        self.halt_head = nn.Linear(d_model, 1)
        self.loop_embed = nn.Embedding(cfg.max_loops + 1, d_model)
        nn.init.zeros_(self.loop_embed.weight)
        nn.init.zeros_(self.focus_head.bias)
        nn.init.zeros_(self.halt_head.bias)

    # -- focus -----------------------------------------------------------

    def normalize(self, h: torch.Tensor, loop_index: int) -> torch.Tensor:
        b, t, _ = h.shape
        loop = torch.full((b, t), loop_index, dtype=torch.long, device=h.device)
        return self.norm(h + self.loop_embed(loop))

    def prefix_summary(
        self, h: torch.Tensor, loop_index: int = 0, cache: dict | None = None
    ) -> torch.Tensor:
        """Prefix mean ``s_t = mean(hn_1..hn_t)`` over the *whole stream*.

        The running sum has to live in the cache. Recomputing a cumulative mean
        from the current chunk alone restarts the statistic at every chunk
        boundary, so a token decoded one-at-a-time sees a different focus
        summary than the same token under teacher forcing — a parity failure
        that is invisible in any single-chunk test and that this method exists
        to prevent.
        """
        hn = self.normalize(h, loop_index)
        b, t, _ = hn.shape
        steps = torch.arange(1, t + 1, device=hn.device, dtype=hn.dtype).view(1, -1, 1)
        running = hn.cumsum(dim=1)
        if cache is None:
            return running / steps
        prev_sum = cache.get(("router", "prefix_sum"))
        prev_n = cache.get(("router", "prefix_n"), 0)
        if prev_sum is not None:
            running = running + prev_sum
        cache[("router", "prefix_sum")] = running[:, -1:, :]
        cache[("router", "prefix_n")] = prev_n + t
        return running / (steps + prev_n)

    def forward(
        self,
        h: torch.Tensor,
        loop_index: int = 0,
        training: bool | None = None,
        gumbel_noise: bool = True,
        span_id: torch.Tensor | None = None,
        summary: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> RoutingDecision:
        if training is None:
            training = self.training
        if self.cfg.capacity_factor > 0:
            raise NotImplementedError(
                "capacity-based token dropping ranks tokens against each other "
                "inside a chunk and is not causally executable at sampling "
                "time; see C-2 in this module's docstring"
            )

        b, t, _ = h.shape
        hn = self.normalize(h, loop_index)

        logits = at_least_fp32(self.gate(hn)) / self.cfg.router_temperature
        if span_id is not None:
            logits = pool_over_spans(logits, span_id)
        probs = torch.softmax(logits, dim=-1)

        if training and gumbel_noise:
            u = torch.rand_like(logits).clamp_(1e-9, 1 - 1e-9)
            selection_scores = logits - torch.log(-torch.log(u))
            if span_id is not None:
                # The *noise* must be shared across an atomic span too.
                # Pooling only the logits leaves per-token noise free to break
                # the tie differently for each patch of a field, which shreds
                # the grid exactly as if there had been no pooling at all.
                selection_scores = pool_over_spans(selection_scores, span_id)
        else:
            selection_scores = logits

        k = min(self.cfg.top_k, self.n_stacks)
        _, index = torch.topk(selection_scores, k, dim=-1)
        # Weight by the *clean* probabilities of the chosen stacks, renormalized.
        chosen = torch.gather(probs, -1, index)
        weight = chosen / chosen.sum(-1, keepdim=True).clamp_min(1e-9)

        # Focus: own state + causal prefix summary. Never a chunk mean (C-1).
        if summary is None:
            summary = self.prefix_summary(h, loop_index, None)
        focus_in = hn + summary
        focus = torch.sigmoid(at_least_fp32(self.focus_head(focus_in))).squeeze(-1)
        span = self.max_depth - self.min_depth
        target_depth = (
            self.min_depth + torch.round(focus.detach() * span)
        ).long().clamp_(self.min_depth, self.max_depth)

        halt_logit = self.halt_head(hn).squeeze(-1)

        balance = self._balance_loss(probs, index, valid)
        z_tokens = torch.logsumexp(at_least_fp32(logits), dim=-1).pow(2)
        z = self.cfg.z_alpha * (z_tokens[valid].mean() if valid is not None else z_tokens.mean())
        return RoutingDecision(
            stack_index=index,
            stack_weight=weight,
            focus=focus,
            target_depth=target_depth,
            halt_logit=halt_logit,
            balance_loss=balance,
            z_loss=z,
            gate_probs=probs,
            valid=valid,
        )

    def _balance_loss(
        self, probs: torch.Tensor, index: torch.Tensor, valid=None
    ) -> torch.Tensor:
        """``alpha * N * sum_i f_i * P_i`` (Fedus et al. 2021, eq. 4).

        ``f_i`` counts *every* dispatch, not only the top-1 choice, and is
        normalized so that ``sum_i f_i == 1`` — otherwise the minimum of the
        objective moves with ``k`` and the coefficient stops meaning anything.

        What this term actually penalises is the *correlation* between dispatch
        frequency and router probability, not unevenness as such. When ``P`` is
        uniform the value is exactly ``alpha`` whatever ``f`` does, because
        ``sum_i f_i P_i = (1/N) sum_i f_i``. That is the intended behaviour —
        ``f`` comes from a ``topk`` and carries no gradient, so the only thing
        the optimizer can move is ``P``, and the term should bite exactly when
        the router is both confident and concentrated.
        """
        n = self.n_stacks
        # ``probs`` keeps its own precision: this is a loss term in an fp64
        # model when the caller asked for fp64, and `.float()` would round it
        # for no reason the loss benefits from.
        dispatched = F.one_hot(index, n).sum(2).to(probs.dtype)  # [B, T, N]
        if valid is not None:
            dispatched, probs = dispatched[valid], probs[valid]
            f, p = dispatched.mean(dim=0), at_least_fp32(probs).mean(dim=0)
        else:
            f, p = dispatched.mean(dim=(0, 1)), at_least_fp32(probs).mean(dim=(0, 1))
        f = f / f.sum().clamp_min(1e-9)
        return self.cfg.balance_alpha * n * torch.sum(f * p)


def pool_over_spans(logits: torch.Tensor, span_id: torch.Tensor) -> torch.Tensor:
    """Average routing logits within each atomic span, leaving ``-1`` alone.

    A field is a grid, and a spectral operator needs all of it. Routing its
    patches token-by-token scatters them across stacks and the operator then
    has a fragment, which it cannot use. Pooling over the span keeps it whole.

    This reads positions later than the token being routed, so it is only
    applied to spans the batcher marked ``observed`` — see the guard in
    ``codecs/spans.py``. ``test_router.py::test_span_pooling_requires_observed``
    holds that line.
    """
    b, t, n = logits.shape
    flat = (span_id + 1).clamp_min(0)                     # -1 -> 0 (the "none" bin)
    n_ids = int(flat.max().item()) + 1
    gid = (
        torch.arange(b, device=logits.device).unsqueeze(1) * n_ids + flat
    ).reshape(-1)
    sums = logits.new_zeros(b * n_ids, n).index_add_(0, gid, logits.reshape(-1, n))
    counts = logits.new_zeros(b * n_ids).index_add_(
        0, gid, torch.ones_like(gid, dtype=logits.dtype)
    )
    means = (sums / counts.clamp_min(1).unsqueeze(-1))[gid].view(b, t, n)
    return torch.where((span_id >= 0).unsqueeze(-1), means, logits)


def stopping_distribution(halt_prob: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """PonderNet stopping distribution, differentiable, along ``dim``.

    ``p_r = lambda_r * prod_{j<r}(1 - lambda_j)`` with the final step absorbing
    the remaining mass so the distribution sums to exactly one. This mirrors
    ``iridium/model/halting.py``; ``test_ponder.py`` asserts the two agree to
    float64 precision so the numpy contract and the torch implementation cannot
    drift apart.
    """
    # cumprod is promoted by AMP; its zero-handling backward must see a
    # matching input/output dtype. Retain float64 reference calculations.
    work = halt_prob if halt_prob.dtype == torch.float64 else halt_prob.float()
    lam = work.movedim(dim, -1).clamp(0.0, 1.0)
    keep = (1.0 - lam).clamp_min(0.0)
    cum = torch.cumprod(
        torch.cat([torch.ones_like(keep[..., :1]), keep[..., :-1]], dim=-1), dim=-1, dtype=lam.dtype
    )
    p = lam * cum
    last = cum[..., -1:] * keep[..., -1:]
    p = torch.cat([p[..., :-1], p[..., -1:] + last], dim=-1)
    return p.movedim(-1, dim)


def geometric_prior(
    n_steps: int, p_stop: float, device=None, dtype=torch.float32
) -> torch.Tensor:
    """Geometric stop prior with the last budgeted step absorbing the tail."""
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if not 0.0 < p_stop <= 1.0:
        raise ValueError("p_stop must lie in (0, 1]")
    r = torch.arange(n_steps, device=device, dtype=dtype)
    prior = p_stop * (1.0 - p_stop) ** r
    prior[-1] = (1.0 - p_stop) ** (n_steps - 1)
    return prior


def ponder_kl(p: torch.Tensor, prior: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """``KL(p || prior)`` averaged over everything but ``dim``."""
    p = (p if p.dtype == torch.float64 else p.float()).clamp_min(1e-9)
    shape = [1] * p.dim()
    shape[dim] = -1
    q = prior.view(shape).clamp_min(1e-9)
    return (p * (p.log() - q.log())).sum(dim=dim).mean()
