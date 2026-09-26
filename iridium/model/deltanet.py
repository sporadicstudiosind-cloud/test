"""Gated DeltaNet: a linear-attention layer with an O(1)-in-length decode state.

Yang, Kautz & Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta
Rule", arXiv:2412.06464, ICLR 2025 (verified against the abstract and the
ICLR listing, not transcribed from memory). Qwen3-Next and Kimi Linear both
put it (or KDA, its finer-gated cousin) in most of their layers, at a 3:1
linear:full ratio, and cite the same reason this file exists: attention's
``[B,H,T,T]`` score matrix and its KV cache are what a million-token context
cannot afford, on any number of accelerators, because both grow with T. This
layer never builds that matrix. Per head it keeps one ``d_head x d_head``
matrix, updated by

    S_t = S_{t-1} @ (alpha_t * (I - beta_t * k_t k_t^T)) + beta_t * v_t k_t^T
    o_t = S_t @ q_t

``alpha_t in (0,1)`` (a per-head, per-token forget gate) and ``beta_t in
(0,1)`` (a per-head, per-token write strength) are both linear functions of
the input, squashed with ``sigmoid``. Q and K are L2-normalised before this
runs (unbounded dot products would make ``beta_t`` -- a *learning rate* on
the state -- scale-dependent on whatever the previous layer happened to
output). The obvious wrong alternative is plain linear attention's cumulative
sum ``S_t = S_{t-1} + v_t k_t^T``: it never forgets, so a state that has
absorbed a million tokens weighs a stale key from token 3 exactly as heavily
as one from token 999,999, and gradient signal for "this key mattered" has
nowhere to go but up. The delta rule's ``(I - beta_t k_t k_t^T)`` term first
*erases* whatever the state currently returns for this exact key before
writing the new value -- it is a targeted update, not an accumulation -- and
the gate lets that erasure/write happen on a *decaying* substrate instead of
a permanent one, which is what actually gives this architecture a chance at
retrieval quality Mamba2 (delta rule, no gate) and Gated Linear Attention
(gate, no delta rule) each fall short of on their own; that comparison is the
paper's whole point, not a claim invented here.

Two computation paths compute this exact function:

* :meth:`GatedDeltaNet.forward` with no ``cache`` runs
  :func:`chunkwise_forward`, which starts every sequence from a zero state and
  is meant for training on a full sequence at once.
* :meth:`GatedDeltaNet.forward` with a ``cache``/``cache_key`` runs
  :func:`recurrent_forward` against whatever ``(S, conv state)`` sits in the
  cache, one token (or a short chunk) at a time, exactly the way
  ``GroupedQueryAttention`` reads and writes its own cache entry.

``chunkwise_forward`` is not the WY/UT fused-kernel algorithm the paper's own
training code uses; that algorithm inverts a lower-triangular intra-chunk
system to get real *tensor-parallel* work inside a chunk, and this file does
not implement it -- doing so correctly without a GPU to profile it against
would be exactly the kind of unverified performance claim this repository
asks not to be made. What is implemented instead is a **two-pass exact**
reformulation, chosen because it is genuinely checkable pure-torch and still
buys something real: the state-transition map "start with a chunk's tokens,
end with a state" is *affine* in the incoming state (S_t = S_0 @ P_t + Z_t,
where P_t is the accumulated D×D transition matrix and Z_t is what the
recurrence produces if it starts, hypothetically, from zero). So each chunk
can be run once from a zero state -- fully parallel across chunks, since
none of them yet depend on each other -- and the chunks are then stitched
together with one sequential pass over only the *chunk count* (T/C matrix
combines) rather than one over every token. Whether that is worth doing on
this codebase's CPU test rungs is not something this file claims either way;
what it buys, honestly, is a second, structurally different implementation of
the same recurrence to check the first one against (see
``tests/unit/test_deltanet.py``), which is exactly the role the module's
ground rules ask a "chunkwise-parallel form" to play here.

Positions are accepted (to match ``GroupedQueryAttention``'s signature) and
ignored. There is nothing to rotate: RoPE encodes relative offset into a
*fixed* attention score between two positions, but this layer has no
score -- order is encoded procedurally, by which token's key/value updated
the state before which token's query read it. Every published Gated DeltaNet
(and Mamba2, and GLA) omits positional embeddings for exactly this reason.

``keep`` semantics this layer HONOURS, and the (large) subset it does not
------------------------------------------------------------------------
A ``GroupedQueryAttention`` ``keep`` mask can express arbitrary per-query,
per-key visibility -- causal, a sliding window, a superstack's sparse
stack-local gather, ``BridgeCrossAttention``'s query-to-external-KV mask,
anything a boolean tensor can encode. A recurrent layer's ``keep`` cannot: the
state is a single running summary, so "query i sees key j" is not a
per-(i, j) decision this layer is capable of making -- it can only decide,
per *token*, whether that token is allowed to write into the state at all,
and whether the layer's job for that token is to run at all.

Concretely, this module accepts exactly the convention this codebase's own
plain-causal layers use (:func:`~iridium.model.layers.causal_keep` and
``TransformerControlCore.stream_keep``: causal AND-ed with a per-key-position
padding mask, identical for every query row beyond the causal cutoff) and
extracts a per-position validity vector from it via the *last* query row,
``keep[..., -1, :]`` -- which, for that convention and only that convention,
equals "is key j real (not padding)". A padding position neither reads from
nor writes into the state (its gate is forced to alpha=1, beta=0, a pure
pass-through) and its own output row is zeroed, mirroring
``GroupedQueryAttention``'s "a dead row returns zero, never NaN" rule.

It is **not** a drop-in replacement for:

* a *sliding-window* layer (``window=`` on ``GroupedQueryAttention``): the
  gate ``alpha_t`` is a *soft*, learned forget rate, not a hard cutoff at
  exactly ``window`` tokens back. Nothing here guarantees a token's influence
  is exactly zero after ``window`` steps the way ``sliding_window_keep``
  does; it only guarantees it is *learned to decay*. A hybrid schedule that
  wants an exact bound has to periodically reset the state (as
  ``iridium.runtime.long_context.LongContextSession`` already does for its
  own compressed memory) rather than trust the gate alone.
* ``BridgeCrossAttention``: that mask is a superstack token attending to the
  *control core's* KV, a different sequence than the one producing the
  query. There is no self-recurrence to run there; a fixed-size state built
  by walking the superstack's own tokens has nothing to do with the core's.
* any ``keep`` that varies the set of visible keys *per query row* beyond
  plain causality -- a superstack's stack-local self-attention gather
  happens to fit (its ``self_keep`` is exactly the causal+padding convention
  above, applied to the tokens already routed to that stack, in the order
  they were routed -- the recurrence needs only that order, not the
  original stream positions), but the router's own soft gating, any capacity
  drop, or a mask that admits a non-contiguous, per-row-different key set
  does not, and this layer raises rather than silently computing the wrong
  thing if the extracted validity vector is inconsistent with that
  convention (see ``_validity_from_keep``).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, at_least_fp32


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalise the last dim to unit norm, fp32 internally (see ``head_rms``).

    Unnormalised Q/K would make ``beta_t`` -- read directly off the raw input
    below -- implicitly rescale with whatever norm the previous layer's
    output happens to have, which turns "how strongly does this token
    overwrite the state" into a function of activation scale rather than of
    what the gate actually learned. The gated delta rule paper and every
    public Gated DeltaNet implementation normalise for this reason.

    Accumulates in fp32 for bf16/fp16 input and never *downcasts* fp32 or
    fp64 input to fp32: this module's own equivalence test runs the model in
    fp64 for a tight numeric tolerance, and a hardcoded ``.float()`` here
    would quietly reintroduce fp32 rounding (observed: 1.19e-7, exactly fp32
    eps) into a computation the caller explicitly asked to run in fp64.

    This module reached that conclusion first and worked around it locally.
    ``layers.at_least_fp32`` is the same rule applied everywhere it was
    missing -- ``head_rms``, both ``RMSNorm``s, the dynamic-tanh norm and
    both masked softmaxes all downcast fp64 until it was, which is why the
    parity gate could only ever be as exact as fp32.
    """
    work = at_least_fp32(x)
    return (work * work.square().sum(-1, keepdim=True).clamp_min(eps).rsqrt()).to(x.dtype)


def _validity_from_keep(keep: torch.Tensor, t: int) -> torch.Tensor:
    """Extract a per-key-position validity vector for the ``t`` newest tokens.

    ``keep`` must be broadcastable to ``[B, H, Tq, Tk]`` and equal
    ``causal_keep(Tq, Tk) & valid[:, None, None, :]`` for some per-batch,
    per-position boolean ``valid`` -- the convention every plain-causal layer
    in this codebase uses (see the module docstring for which layers that is
    and is not). Under that convention the *last* query row is never
    causally excluded from any key (its absolute position is the newest key's
    position), so ``keep[..., -1, j]`` equals ``valid[j]`` for every key
    column ``j`` -- in particular for the last ``t`` columns, which are the
    keys this call's own ``t`` tokens just produced (whether or not ``keep``
    also carries older, cached key positions this layer never needs to see
    directly, since the recurrent state already absorbed them).
    """
    if keep.shape[-1] < t or keep.shape[-2] < 1:
        raise ValueError(
            f"GatedDeltaNet needs a [.., >=1, >={t}] causal-style keep mask; "
            f"got trailing dims {tuple(keep.shape[-2:])} for t={t}"
        )
    row = keep[..., -1, -t:]  # [..., H_or_1, t], broadcastable over batch and heads
    while row.dim() > 2:
        # Collapse every dim between batch and the trailing key axis (head,
        # or nothing) with ``any``: this convention's validity is the same
        # for every head, so an ``any`` here is exact, not an approximation.
        row = row.any(dim=-2)
    return row  # [B, t] (or [1, t] if keep carried no batch dim of its own)


def _causal_conv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    conv_size: int, state: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal conv over the time axis, with an explicit carry state.

    ``x`` is ``[B, C, T]``. ``state`` is the previous ``conv_size - 1`` input
    columns (zeros for a fresh sequence, which reproduces left-zero-padding
    exactly -- there is no separate "first call" branch to keep in sync with
    the cached one). Returns the conv output (``[B, C, T]``, causal: output
    step ``t`` depends only on inputs ``<= t``) and the new carry state, so a
    training-time full-sequence call and a token-at-a-time decode call run
    the identical function.
    """
    b, c, t = x.shape
    if state is None:
        state = x.new_zeros(b, c, max(conv_size - 1, 0))
    xp = torch.cat([state, x], dim=2)
    out = F.conv1d(xp, weight, bias, groups=c)
    new_state = xp[:, :, xp.shape[2] - max(conv_size - 1, 0):]
    return out, new_state


def _delta_step(
    S: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    alpha: torch.Tensor, beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One gated-delta-rule update, batched over any leading dims ``G``.

    ``S``: ``[G, Dv, Dk]``. ``q, k, v``: ``[G, D]``. ``alpha, beta``: ``[G]``.
    Returns the *updated* state and ``o_t = S_t @ q_t`` (the rule reads out
    the post-update state, per the module docstring's formula).
    """
    Sk = torch.einsum("gvk,gk->gv", S, k)
    delta = v - alpha.unsqueeze(-1) * Sk
    S = alpha.view(-1, 1, 1) * S + beta.view(-1, 1, 1) * torch.einsum("gv,gk->gvk", delta, k)
    o = torch.einsum("gvk,gk->gv", S, q)
    return S, o


def recurrent_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    alpha: torch.Tensor, beta: torch.Tensor, state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token reference path. ``q,k,v``: ``[B, H, T, D]``; ``alpha,
    beta``: ``[B, H, T]``. Returns ``(outputs [B,H,T,D], final_state [B,H,D,D])``.

    This is the ground truth :func:`chunkwise_forward` is checked against,
    and it is also exactly what decoding runs (one token, ``T == 1``, per
    call) -- not a separate "fast decode kernel" reimplementing the same
    arithmetic, which is the usual place a cache and a training path quietly
    drift apart.
    """
    b, h, t, d = q.shape
    g = b * h
    S = q.new_zeros(g, d, d) if state is None else state.reshape(g, d, d)
    qf = q.reshape(g, t, d)
    kf = k.reshape(g, t, d)
    vf = v.reshape(g, t, d)
    af = alpha.reshape(g, t)
    bf = beta.reshape(g, t)
    outs = []
    for i in range(t):
        S, o = _delta_step(S, qf[:, i], kf[:, i], vf[:, i], af[:, i], bf[:, i])
        outs.append(o)
    out = torch.stack(outs, dim=1).reshape(b, h, t, d) if outs else q.new_zeros(b, h, 0, d)
    return out, S.reshape(b, h, d, d)


def chunkwise_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    alpha: torch.Tensor, beta: torch.Tensor, chunk_size: int,
) -> torch.Tensor:
    """Two-pass exact reformulation described in the module docstring.

    Always starts from a zero state (training-from-scratch over a full
    sequence); see the module docstring for why this is not the fused
    WY/UT kernel and what it buys instead. ``chunk_size`` need not divide
    ``T`` -- the remainder is run as one final, shorter "chunk" through the
    same two-pass machinery, not specially cased, which is what lets the
    equivalence test in ``tests/unit/test_deltanet.py`` sweep sizes that
    don't divide ``T`` without a second code path to keep correct.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    b, h, t, d = q.shape
    g = b * h
    device, dtype = q.device, q.dtype
    qf, kf, vf = (x.reshape(g, t, d) for x in (q, k, v))
    af, bf = alpha.reshape(g, t), beta.reshape(g, t)

    S = torch.zeros(g, d, d, device=device, dtype=dtype)
    outputs = torch.empty(g, t, d, device=device, dtype=dtype)
    start = 0
    while start < t:
        end = min(t, start + chunk_size)
        length = end - start
        # Pass 1: per-step transition matrices P (D x D, this chunk's own
        # steps only) and the zero-state trajectory Z, both starting fresh.
        P = torch.eye(d, device=device, dtype=dtype).expand(g, d, d).clone()
        Z = torch.zeros(g, d, d, device=device, dtype=dtype)
        p_at_t, z_out_at_t = [], []
        for i in range(start, end):
            ai, bi, ki = af[:, i], bf[:, i], kf[:, i]
            Pk = torch.einsum("gij,gj->gi", P, ki)
            P = ai.view(-1, 1, 1) * P - (ai * bi).view(-1, 1, 1) * torch.einsum("gi,gj->gij", Pk, ki)
            p_at_t.append(P)
            Z, zo = _delta_step(Z, qf[:, i], ki, vf[:, i], ai, bi)
            z_out_at_t.append(zo)
        # Pass 2: fold in the state this chunk actually inherited, S (the
        # combination of every earlier chunk). o_t = S0 @ (P_t q_t) + z_out_t.
        for local, i in enumerate(range(start, end)):
            p_q = torch.einsum("gij,gj->gi", p_at_t[local], qf[:, i])
            outputs[:, i] = torch.einsum("gvk,gk->gv", S, p_q) + z_out_at_t[local]
        S = torch.einsum("gvk,gkj->gvj", S, p_at_t[-1]) + Z
        start = end
    return outputs.reshape(b, h, t, d)


class GatedDeltaNet(nn.Module):
    """Drop-in alternative to ``GroupedQueryAttention`` with an O(1) KV cache.

    Same ``forward(x, positions, keep, cache, cache_key)`` signature and the
    same cache-dict protocol (one entry per ``cache_key``, holding whatever
    that layer needs to resume). Here that entry is ``(state, conv_q, conv_k,
    conv_v)`` rather than growing ``(k, v)`` tensors: see
    :meth:`state_bytes`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        conv_size: int = 4,
        chunk_size: int = 64,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if conv_size < 1:
            raise ValueError("conv_size must be >= 1")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.n_heads = n_heads
        self.d_head = d_head
        self.conv_size = conv_size
        self.chunk_size = chunk_size
        self.eps = eps
        d_qkv = n_heads * d_head
        self.d_qkv = d_qkv

        self.wq = nn.Linear(d_model, d_qkv, bias=False)
        self.wk = nn.Linear(d_model, d_qkv, bias=False)
        self.wv = nn.Linear(d_model, d_qkv, bias=False)
        self.w_gate = nn.Linear(d_model, d_qkv, bias=False)
        self.wo = nn.Linear(d_qkv, d_model, bias=False)

        # Per-head scalar gates. Bias initialised so a freshly initialised
        # layer starts close to "remember everything, write cautiously"
        # (alpha ~ sigmoid(4) = 0.982, beta ~ sigmoid(-2) = 0.12): a layer
        # that forgets aggressively before it has learned anything useful to
        # forget destroys signal the rest of training can never recover,
        # since there is no attention fallback path to fall back to.
        self.w_alpha = nn.Linear(d_model, n_heads, bias=True)
        self.w_beta = nn.Linear(d_model, n_heads, bias=True)
        nn.init.zeros_(self.w_alpha.weight)
        nn.init.constant_(self.w_alpha.bias, 4.0)
        nn.init.zeros_(self.w_beta.weight)
        nn.init.constant_(self.w_beta.bias, -2.0)

        # Depthwise causal convs, one per projection (Mamba2 / Gated
        # DeltaNet's short local mixing before the recurrence sees anything).
        # Kept as three separate convs rather than one over the concatenated
        # width: q/k/v are semantically different signals and grouping (via
        # ``groups=d_qkv``) already stops any cross-channel mixing within
        # each, so concatenating them would buy nothing but a single kernel
        # launch, and this is CPU code where that launch isn't the cost.
        self.conv_q = nn.Conv1d(d_qkv, d_qkv, conv_size, groups=d_qkv, bias=True)
        self.conv_k = nn.Conv1d(d_qkv, d_qkv, conv_size, groups=d_qkv, bias=True)
        self.conv_v = nn.Conv1d(d_qkv, d_qkv, conv_size, groups=d_qkv, bias=True)

        self.out_norm = RMSNorm(d_qkv)

    @staticmethod
    def param_count(d_model: int, n_heads: int, d_head: int, conv_size: int = 4) -> int:
        """Exact parameter count, matched against ``sum(p.numel())`` in the test.

        ``wq``/``wk``/``wv``/``w_gate``/``wo``: five ``d_model x d_qkv``
        matrices, no bias. ``w_alpha``/``w_beta``: ``d_model x n_heads`` plus
        an ``n_heads`` bias each. Three depthwise convs: ``d_qkv * conv_size``
        weights (one ``[1, conv_size]`` filter per channel, ``groups=d_qkv``)
        plus a ``d_qkv`` bias, each. One RMSNorm gain of width ``d_qkv``.
        """
        d_qkv = n_heads * d_head
        linear = 5 * d_model * d_qkv
        gates = 2 * (d_model * n_heads + n_heads)
        convs = 3 * (d_qkv * conv_size + d_qkv)
        norm = d_qkv
        return linear + gates + convs + norm

    @staticmethod
    def state_bytes(n_heads: int, d_head: int, conv_size: int = 4, bytes_per_element: int = 2) -> int:
        """Decode-time cache size for one stream, one layer -- independent of T.

        This is the number a KV cache cannot report for any attention layer:
        ``GroupedQueryAttention.kv_bytes_per_token`` (see ``config.py``) is a
        *per-token* rate that a serving system multiplies by however long the
        stream has gotten; there is nothing analogous to multiply here.
        """
        matrix = n_heads * d_head * d_head
        conv = 3 * n_heads * d_head * max(conv_size - 1, 0)
        return (matrix + conv) * bytes_per_element

    def _project(self, x: torch.Tensor, cache: Optional[dict], cache_key: Optional[tuple]):
        b, t, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        conv_state = cache.get(cache_key) if (cache is not None and cache_key is not None) else None
        cq, ck, cv = conv_state[1:] if conv_state is not None else (None, None, None)
        q, cq = _causal_conv(q.transpose(1, 2), self.conv_q.weight, self.conv_q.bias, self.conv_size, cq)
        k, ck = _causal_conv(k.transpose(1, 2), self.conv_k.weight, self.conv_k.bias, self.conv_size, ck)
        v, cv = _causal_conv(v.transpose(1, 2), self.conv_v.weight, self.conv_v.bias, self.conv_size, cv)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        q = q.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        q, k = _l2_normalize(q, self.eps), _l2_normalize(k, self.eps)
        return q, k, v, (cq, ck, cv)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,  # noqa: ARG002 -- see module docstring: unused by design
        keep: torch.Tensor,
        cache: Optional[dict] = None,
        cache_key: Optional[tuple] = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        prior_state = None
        if cache is not None and cache_key is not None:
            entry = cache.get(cache_key)
            if entry is not None:
                prior_state = entry[0]

        q, k, v, (cq, ck, cv) = self._project(x, cache, cache_key)

        log_alpha = F.logsigmoid(self.w_alpha(x))  # stable log(sigmoid(.)); see module docstring
        alpha = log_alpha.exp().transpose(1, 2)     # [B, H, T]
        beta = torch.sigmoid(self.w_beta(x)).transpose(1, 2)

        valid = _validity_from_keep(keep, t)          # [B, T] or [1, T]
        valid = valid.expand(b, t).reshape(b, 1, t).expand(b, self.n_heads, t)
        alpha = torch.where(valid, alpha, torch.ones_like(alpha))
        beta = torch.where(valid, beta, torch.zeros_like(beta))

        if cache is not None and cache_key is not None:
            out, state = recurrent_forward(q, k, v, alpha, beta, prior_state)
            cache[cache_key] = (state, cq, ck, cv)
        else:
            out = chunkwise_forward(q, k, v, alpha, beta, self.chunk_size)

        out = out.masked_fill(~valid.unsqueeze(-1), 0.0)
        out = out.transpose(1, 2).reshape(b, t, self.d_qkv)
        out = self.out_norm(out) * F.silu(self.w_gate(x))
        return self.wo(out)
