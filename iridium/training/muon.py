"""Muon: momentum-orthogonalized SGD for hidden matrices, AdamW for everything else.

Muon (Keller Jordan, "Muon: An optimizer for hidden layers in neural networks",
2024, https://kellerjordan.github.io/posts/muon/) replaces AdamW's per-element
second moment with a structural constraint: take the Nesterov-momentum
gradient, then replace it with the nearest matrix of orthonormal singular
vectors before applying it. An orthogonalized update spreads its energy
evenly across every direction in the weight's row/column space instead of
letting one already-well-conditioned direction dominate, which is a norm
constraint on the update's *spectral* norm the way AdamW's per-element
normalization is a norm constraint on each element's magnitude. For matrix
parameters -- the thing a Transformer is mostly made of -- the spectral
picture is the one that matches how the weight is actually used (as an
operator on activations), which is why Jordan reports faster small-scale
convergence than AdamW at equal step count.

That result did not scale up as given. Moonshot's "Muon is Scalable for LLM
Training" (arXiv:2502.16982, Moonlight; verified from the paper PDF, section
2.2) found two problems with the small-scale recipe once model and token
count grow, and fixed both:

* **No weight decay.** Left undecayed, weight and layer-output RMS grow past
  bf16's precise range over long enough training. Section 2.2 adds standard
  decoupled decay, giving update rule (their eq. 3/4)::

      W_t = W_{t-1} - lr * (0.2 * O_t * sqrt(max(A, B)) + weight_decay * W_{t-1})

  where ``O_t`` is the orthogonalized, momentum'd gradient for a weight of
  shape ``[A, B]``.
* **Update RMS depends on shape.** Lemma 1 of that paper: a full-rank
  orthogonalized update for a ``[A, B]`` matrix has RMS ``sqrt(1 / max(A,
  B))`` -- i.e. a fatter matrix (larger `max(A, B)`, e.g. an MLP's up-
  projection) gets a *smaller* per-element update than a squarer one, purely
  as an artifact of shape, not of what that matrix needs. Left alone this
  starves wide matrices of signal and over-moves narrow ones. Multiplying by
  ``sqrt(max(A, B))`` cancels the shape dependence so every matrix in the
  model gets the same update RMS regardless of its aspect ratio. The paper
  then also matches that RMS to AdamW's typical ~0.2-0.4 (the ``0.2`` factor
  above) specifically so one lr/weight_decay pair, already tuned for AdamW,
  can be reused for Muon groups and AdamW groups in the same run without a
  second hyperparameter sweep -- which is the point of sharing ``group["lr"]``
  across both paths below rather than giving each its own schedule.

Both papers are explicit, and this file follows them, that **Muon is only
correct for hidden 2-D weight matrices**. Applying it to an embedding table
is a specific, documented mistake, not a missed optimization: an embedding
row is a *lookup*, updated only for the tokens seen in a batch, so most rows
get a zero-momentum, zero-orthogonalized update on most steps while a few
rows get the whole thing -- there is no shared "spectral structure" across
rows for Newton-Schulz to preserve, and empirically (Jordan's writeup,
restated in Moonlight sec 2.2) embeddings, the output head, and any
norm/bias/scalar param are handed to AdamW instead. :func:`muon_param_groups`
enforces that split so a caller cannot silently point Muon at an embedding.

Kimi K2 (arXiv:2507.20534) trained a 1T-parameter MoE on Muon at scale and
found a further failure mode past Moonlight's fixes: attention logits could
still blow up mid-run. Their MuonClip adds a post-step "QK-clip" that
rescales a head's query/key projections down whenever that head's max
attention logit exceeds a threshold, directly rather than via gradient
clipping. That is an attention-specific stability patch orthogonal to the
orthogonalization/scaling math here, needs hooks into attention logits this
module has no access to, and was not asked for in this file's scope -- it is
called out here only so nobody reads "Muon" in the trainer and assumes
QK-clip is included.

**Newton-Schulz, precision.** The 5-step quintic iteration below uses
Jordan's tuned coefficients ``(3.4445, -4.7750, 2.0315)``, which were chosen
by gradient search to maximize the slope at zero rather than to converge
exactly to the orthogonal matrix; the result lands singular values in roughly
``[0.7, 1.2]`` instead of exactly 1, deliberately, because Jordan's ablations
found the exact answer costs more steps for no measured training benefit.
Moonlight sec 2.2 confirms 10 steps orthogonalizes more precisely than 5 but
does not train better, and keeps 5 for speed; we do the same. The iteration
is run in bf16 on CUDA (that is the whole reason Newton-Schulz was chosen
over an SVD -- it is a few small matmuls, stable in low precision, versus a
serial algorithm SVD has no fast low-precision kernel for) and in fp32 on
CPU, where bf16 has no speed advantage and only adds rounding noise on the
CPU reference path these tests run against.

**Conv and other >2-D weights.** Jordan's implementation collapses a conv
filter's trailing dimensions to treat it as 2-D and orthogonalizes that. We
deliberately do *not* do this here: collapsing ``(out, in, kh, kw)`` to
``(out, in*kh*kw)`` treats spatially adjacent taps as interchangeable
directions in the same vector space Newton-Schulz is meant to spread energy
across evenly, which is a much weaker structural argument for a conv filter
than for a Linear's weight (an MLP or attention projection actually is
"apply this matrix to a vector"). Absent a convolutional layer in this
codebase to validate that reshape against, routing every non-``nn.Linear``
weight of rank >= 2 to the AdamW path is the conservative default; it costs
nothing but the (small, well understood) AdamW update on a tensor Muon was
never validated against, is exactly what Jordan's own writeup does for
"anything that isn't a hidden weight layer", and is trivial to revisit if a
conv-heavy model shows up later.

Same call contract as :class:`iridium.training.eager_adamw.EagerAdamW`,
because the trainer's wiring code needs to treat every optimizer the same
way: a ``param_groups`` list of dicts (the trainer sets ``group["lr"]`` every
step), ``zero_grad(set_to_none)``, ``step()``, ``state_dict()``,
``load_state_dict()``. Unlike EagerAdamW this class does not add a foreach
fusion path: EagerAdamW's foreach path exists because a routed model has
thousands of small AdamW tensors and the CUDA launch overhead of stepping
them one at a time is the bottleneck. Muon's dominant cost is the opposite
shape -- a handful of large matmuls (the Newton-Schulz iterations) per hidden
weight -- so there is no launch-bound elementwise loop here for foreach to
fix, and the AdamW remainder group (embeddings, norms, biases) is small by
construction. If that remainder group ever grows large enough for its
per-tensor loop to matter, lifting ``EagerAdamW``'s bucket-by-``(step,
device, dtype)`` routine over is straightforward; adding it pre-emptively
here would be unused complexity.
"""
from __future__ import annotations

import torch

try:
    import torch.nn as nn
except ImportError:  # pragma: no cover - torch always ships nn
    nn = None


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5,
                                eps: float = 1e-7) -> torch.Tensor:
    """Orthogonalize a 2-D matrix via the quintic Newton-Schulz iteration.

    Returns something close to ``U @ V.T`` where ``G = U @ S @ V.T`` is the
    SVD, but not exactly: Jordan's coefficients were tuned to maximize
    convergence speed rather than exactness, so the singular values of the
    result land in roughly ``[0.7, 1.2]`` rather than exactly at 1 (see
    module docstring). That is intentional and is what the test suite checks
    for, not a bug to "fix" by adding steps.

    Runs in bf16 on CUDA and fp32 on CPU/other devices. bf16 is safe here
    specifically because every step is a small matmul followed by a rescale,
    unlike e.g. an eigen-decomposition, which has no such low-precision
    fallback; it is also pointless on CPU, where bf16 matmuls are not
    accelerated and only add rounding error, so the CPU path stays in fp32
    (and this is the path this repository's tests exercise, since it targets
    CPU-only PyTorch).
    """
    if G.ndim != 2:
        raise ValueError("zeropower_via_newtonschulz5 expects a 2-D matrix; "
                         "flatten or route higher-rank tensors to the AdamW path first")
    a, b, c = 3.4445, -4.7750, 2.0315
    work_dtype = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(work_dtype)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def _muon_update(grad: torch.Tensor, momentum_buffer: torch.Tensor, beta: float,
                 ns_steps: int, nesterov: bool) -> torch.Tensor:
    """Nesterov momentum, then orthogonalize. Mutates ``momentum_buffer`` in place."""
    momentum_buffer.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum_buffer, beta) if nesterov else momentum_buffer
    return zeropower_via_newtonschulz5(update, steps=ns_steps)


class Muon:
    """Muon for 2-D hidden matrices, decoupled-decay AdamW for everything else.

    Does not subclass ``torch.optim.Optimizer`` for the same reason
    :class:`EagerAdamW` does not: no Dynamo import lurking in the base
    constructor, no assumptions about a single flat parameter list.

    ``param_groups`` must already be the torch.optim-style list of group
    dicts produced by :func:`muon_param_groups` (or built the same way by
    hand): every group needs ``"params"`` and a boolean ``"use_muon"`` that
    says which update rule that group's tensors get. This is required, not
    inferred from shape, because Muon-vs-AdamW is a *semantic* distinction
    (is this tensor a hidden weight matrix?) that shape alone gets wrong --
    a tied embedding/output-head weight is exactly 2-D and would pass any
    rank test, which is the known mistake this class refuses to make
    silently. A group missing the key is a construction bug, not a default
    to paper over.

    Every Muon group's tensors must be 2-D; anything else raises at
    construction, before a single step corrupts state.
    """

    def __init__(self, param_groups, lr: float = 3e-4, weight_decay: float = 0.01,
                momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
                adamw_betas: tuple[float, float] = (0.9, 0.95), adamw_eps: float = 1e-8):
        if lr < 0 or weight_decay < 0 or adamw_eps <= 0:
            raise ValueError("invalid Muon parameters")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must lie in [0, 1)")
        if any(not 0 <= b < 1 for b in adamw_betas):
            raise ValueError("adamw_betas must lie in [0, 1)")
        raw_groups = list(param_groups)
        if not raw_groups or not isinstance(raw_groups[0], dict):
            raise ValueError("Muon requires pre-split torch.optim-style param groups; "
                             "build them with muon_param_groups(), not a bare parameter list")
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        self.param_groups = []
        for raw in raw_groups:
            if 'params' not in raw or 'use_muon' not in raw:
                raise ValueError("every Muon param group needs 'params' and 'use_muon'; "
                                 "use muon_param_groups() to build the split")
            group = dict(defaults)
            group.update({k: v for k, v in raw.items() if k != 'params'})
            group['params'] = list(raw['params'])
            self.param_groups.append(group)
        flat = [p for g in self.param_groups for p in g['params']]
        if not flat:
            raise ValueError("invalid Muon parameters")
        if any(p.dtype not in (torch.float32, torch.float64) for p in flat):
            raise ValueError("Muon needs fp32/fp64 master weights; use autocast for the forward pass")
        if len({id(p) for p in flat}) != len(flat):
            raise ValueError("a parameter appears in more than one group")
        for group in self.param_groups:
            if group['use_muon']:
                bad = [p for p in group['params'] if p.ndim != 2]
                if bad:
                    raise ValueError(
                        "Muon is only defined for 2-D hidden matrices; route embeddings, "
                        "output heads, norms, biases, scalars and any other-rank tensor "
                        "through the adamw path instead (use_muon=False)")
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.adamw_betas = adamw_betas
        self.adamw_eps = adamw_eps
        self.state: dict = {}

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group['params']:
                if set_to_none:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.detach_().zero_()

    @torch.no_grad()
    def step(self):
        # Allocate all missing state before mutating any weight, same
        # reasoning as EagerAdamW: an allocation failure must never leave
        # half the model updated by one optimizer's rule and half untouched.
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise ValueError("Muon requires dense gradients")
                if p in self.state:
                    continue
                if group['use_muon']:
                    self.state[p] = dict(momentum_buffer=torch.zeros_like(p))
                else:
                    self.state[p] = dict(step=0, exp_avg=torch.zeros_like(p),
                                         exp_avg_sq=torch.zeros_like(p))
        for group in self.param_groups:
            live = [p for p in group['params'] if p.grad is not None]
            if not live:
                continue
            if group['use_muon']:
                self._step_muon_group(group, live)
            else:
                self._step_adamw_group(group, live)

    def _step_muon_group(self, group, live):
        lr, wd, momentum = group['lr'], group['weight_decay'], group['momentum']
        for p in live:
            buf = self.state[p]['momentum_buffer']
            update = _muon_update(p.grad, buf, momentum, self.ns_steps, self.nesterov)
            m, n = p.shape
            # Moonlight eq. 4: 0.2 * sqrt(max(m, n)) cancels the shape-dependent
            # RMS of Lemma 1 and lands Muon's update RMS in AdamW's ~0.2-0.4
            # range, so the same lr/weight_decay tuned for AdamW applies here.
            scale = 0.2 * (max(m, n) ** 0.5)
            p.mul_(1 - lr * wd)
            p.add_(update, alpha=-lr * scale)

    def _step_adamw_group(self, group, live):
        # Same decoupled-AdamW math as EagerAdamW._step_group_eager. Kept
        # inline rather than imported: EagerAdamW is a standalone class this
        # file must not depend on, and the AdamW remainder groups here are
        # small enough that duplicating ~6 lines of arithmetic is cheaper to
        # read than adding a cross-module coupling for it.
        b1, b2 = self.adamw_betas
        eps = self.adamw_eps
        lr, wd = group['lr'], group['weight_decay']
        for p in live:
            s = self.state[p]
            s['step'] += 1
            m, v = s['exp_avg'], s['exp_avg_sq']
            m.lerp_(p.grad, 1 - b1)
            v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            denom = (v.sqrt() / (1 - b2 ** s['step']) ** 0.5).add_(eps)
            p.mul_(1 - lr * wd)
            p.addcdiv_(m, denom, value=-lr / (1 - b1 ** s['step']))

    def state_dict(self):
        params = [p for g in self.param_groups for p in g['params']]
        indices = {p: i for i, p in enumerate(params)}
        return {'state': {indices[p]: s for p, s in self.state.items()},
                'param_groups': [{**g, 'params': [indices[p] for p in g['params']]}
                                 for g in self.param_groups]}

    def load_state_dict(self, data):
        params = [p for g in self.param_groups for p in g['params']]
        saved = data['param_groups']
        if len(saved) != len(self.param_groups) or sum(
                len(g['params']) for g in saved) != len(params):
            raise ValueError("checkpoint optimizer parameter layout differs")
        for group, stored in zip(self.param_groups, saved):
            if len(stored['params']) != len(group['params']):
                raise ValueError("checkpoint optimizer parameter layout differs")
            group.update({k: v for k, v in stored.items() if k != 'params'})
        self.state = {params[int(i)]: {k: v.to(params[int(i)]) if torch.is_tensor(v) else v
                                      for k, v in s.items()}
                      for i, s in data['state'].items()}


def muon_param_groups(model_or_named_params, lr: float = 3e-4, weight_decay: float = 0.01,
                      momentum: float = 0.95):
    """Split a model's parameters into a Muon group and one or two AdamW groups.

    Accepts either an ``nn.Module`` or an iterable of ``(name, param)`` pairs
    (as ``model.named_parameters()`` yields). Passing the module, when one is
    available, is preferred and is what "by module type, not only rank"
    means: a tensor only goes to Muon when it is the ``weight`` of an
    ``nn.Linear`` *and* is 2-D *and* its qualified name does not contain
    "embedding" or "text_head" (a tied output head is frequently implemented
    as an ``nn.Linear`` reusing the embedding weight, or named for what it
    is; either way its role, not its shape, is what disqualifies it). Given
    only named parameters with no module structure, the same name-substring
    exclusion applies but "is this a Linear weight" degrades to "is this
    2-D", so a same-shaped non-Linear matrix (e.g. a hand-rolled routed
    expert weight not wrapped in ``nn.Linear``) would be misclassified as
    Muon-eligible in that path; prefer the module form whenever the caller
    has a live model.

    Everything else -- ``nn.Embedding`` weights, biases, norm gains, scalar
    gates, conv filters, and any tensor under an excluded name -- is handed
    to :func:`iridium.runtime.memory.decay_groups`, which is reused rather
    than reimplemented so the AdamW remainder here decays exactly the
    tensors the rest of this codebase already agrees should be decayed
    (rank >= 2: embeddings, output heads, conv filters) and exempts the rest
    (biases, norm gains, scalars) for the reasons documented there. This
    means an embedding table *does* get decoupled weight decay through the
    AdamW path -- decay_groups' own rule -- even though the same file's
    docstring separately flags embedding decay as a judgment call for tied
    heads; that tension is decay_groups' to resolve, not this function's,
    and reusing it keeps one project-wide answer to "what gets decayed"
    instead of a second, silently different one living here.

    Returns a list of 1-3 torch.optim-style group dicts (empty groups are
    omitted, unlike ``decay_groups``, since ``Muon.__init__`` requires every
    group to have at least one live tensor... actually it does not, but an
    empty group is dead weight in every downstream loop), each carrying
    ``"lr"``, ``"weight_decay"``, ``"use_muon"`` and, for the Muon group,
    ``"momentum"`` -- ready to hand straight to ``Muon(...)``.
    """
    from ..runtime.memory import decay_groups

    exclude = ("embedding", "text_head")

    def is_excluded(name: str) -> bool:
        lowered = name.lower()
        return any(s in lowered for s in exclude)

    muon_params: list = []
    rest: list = []

    if nn is not None and isinstance(model_or_named_params, nn.Module):
        model = model_or_named_params
        linear_weight_ids = {id(m.weight) for m in model.modules()
                            if isinstance(m, nn.Linear) and m.weight is not None}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in linear_weight_ids and param.ndim == 2 and not is_excluded(name):
                muon_params.append(param)
            else:
                rest.append((name, param))
    else:
        for name, param in model_or_named_params:
            if not param.requires_grad:
                continue
            if param.ndim == 2 and not is_excluded(name):
                muon_params.append(param)
            else:
                rest.append((name, param))

    groups = []
    if muon_params:
        groups.append({"params": muon_params, "lr": lr, "weight_decay": weight_decay,
                       "momentum": momentum, "use_muon": True, "group": "muon"})
    for adamw_group in decay_groups(rest, weight_decay=weight_decay):
        if not adamw_group["params"]:
            continue
        adamw_group = dict(adamw_group)
        adamw_group["lr"] = lr
        adamw_group["use_muon"] = False
        adamw_group["group"] = f"adamw_{adamw_group['group']}"
        groups.append(adamw_group)
    return groups
