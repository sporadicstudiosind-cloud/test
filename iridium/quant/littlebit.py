"""Sub-1-bit weight compression: LittleBit with LittleBit-2's Joint-ITQ start.

Written from the papers' equations, not from Samsung's code (which is
CC BY-NC 4.0): LittleBit, Lee et al., NeurIPS 2025, arXiv 2506.13771, and
LittleBit-2, arXiv 2603.00042 (ICML 2026).

A weight ``W`` (``out x in``) becomes a rank-``r`` product of two *binary*
matrices with three real scale vectors::

    W ~= diag(h) . sign(U) . diag(l) . sign(V)^T . diag(g)

so ``y = ((((x * g) @ sign(V)) * l) @ sign(U)^T) * h``. The signs cost one bit
each and the scales 16, so a layer costs ``r(out + in) + 16(out + in + r)``
bits. A second, identical "residual" path fitted to what the first misses
doubles the sign bits. Rank, not bit width, sets the budget, which is how
the method reaches 0.1 bits per weight.

The start matters more than anything else at these budgets:

* **Dual-SVID** (LittleBit): take the rank-``r`` SVD factors, keep their signs
  as the binary matrices, and fit each factor's magnitudes with a rank-one
  outer product, which gives ``h``, ``g`` and ``l``.
* **Joint-ITQ** (LittleBit-2): before that, rotate both factors by the same
  orthogonal ``R`` (``U R (V R)^T = U V^T``, so the product is unchanged)
  chosen by iterative quantisation to bring the factors close to their own
  signs. Binarising a factor whose rows sit near the corners of the
  hypercube loses less. It costs nothing at inference: only the start changes.

Training is quantisation-aware with distillation from the uncompressed model
(:func:`distill`); ``sign`` passes gradients through the derivative of
``tanh(k x)`` (their "SmoothSign", ``k = 100``).

**Read this before using it on iridium-50m.** Every result in both papers is
on 1.3B-70B models. The quantisation literature finds small and over-trained
models degrade most (Ouyang et al. 2024, arXiv 2411.17691), and iridium-50m is
both. On a 2M-parameter test model, see ``docs/littlebit.md`` for what this
module actually measured.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -- bit budget ---------------------------------------------------------------

def layer_bits(out_features: int, in_features: int, rank: int, residual: bool) -> int:
    """Storage of one compressed layer: sign bits plus fp16 scales (Eq. 16)."""
    paths = 2 if residual else 1
    return paths * (rank * (out_features + in_features) + 16 * (out_features + in_features + rank))


def rank_for_bpw(out_features: int, in_features: int, bpw: float, residual: bool) -> int:
    """The largest rank whose storage fits ``bpw`` bits per original weight
    (Eq. 17). At least 1: a layer cannot be compressed to nothing."""
    paths = 2 if residual else 1
    budget = bpw * out_features * in_features / paths - 16 * (out_features + in_features)
    return max(1, int(budget // (out_features + in_features + 16)))


# -- sign with a smooth gradient ---------------------------------------------

class _SmoothSign(torch.autograd.Function):
    """``sign(x)`` forward; ``d/dx tanh(k x)`` backward (LittleBit sec. 4.1).

    A plain straight-through estimator passes the gradient unchanged however
    far ``x`` is from zero; this one passes it only near the sign flip, where
    a step can actually change the binary value."""

    @staticmethod
    def forward(ctx, x, k):
        ctx.save_for_backward(x)
        ctx.k = k
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        t = torch.tanh(ctx.k * x)
        return grad * ctx.k * (1 - t * t), None


def smooth_sign(x: torch.Tensor, k: float = 100.0) -> torch.Tensor:
    return _SmoothSign.apply(x, k)


# -- initialisation -------------------------------------------------------------

def joint_itq(u: torch.Tensor, v: torch.Tensor, iterations: int = 50,
              generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """LittleBit-2's Joint-ITQ (Alg. 1): an orthogonal ``R`` (``r x r``) that
    brings both factors close to their signs at once.

    Stacks ``Z = [u; v]`` and alternates ``B = sign(Z R)`` with the orthogonal
    Procrustes solution ``R = argmin ||B - Z R||`` (from the SVD of ``Z^T B``).
    Applying the same ``R`` to both keeps ``u v^T`` exact."""
    z = torch.cat([u, v], dim=0).double()
    r = z.shape[1]
    q, _ = torch.linalg.qr(torch.randn(r, r, generator=generator, dtype=torch.float64))
    for _ in range(iterations):
        b = torch.where(z @ q >= 0, 1.0, -1.0).to(z.dtype)
        left, _, right_t = torch.linalg.svd(z.T @ b)
        q = left @ right_t
    return q.to(u.dtype)


def _rank_one(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nonnegative ``a, b`` with ``m ~= a b^T`` for a nonnegative ``m``."""
    left, s, right_t = torch.linalg.svd(m.double(), full_matrices=False)
    a = left[:, 0].abs() * s[0].sqrt()
    b = right_t[0].abs() * s[0].sqrt()
    return a.to(m.dtype), b.to(m.dtype)


def dual_svid(w: torch.Tensor, rank: int, itq: bool = True, iterations: int = 50,
              generator: Optional[torch.Generator] = None):
    """Latent factors and scales for one path (LittleBit's Dual-SVID, with an
    optional Joint-ITQ rotation first). Returns ``u, v, h, g, l``."""
    w = w.detach().float()
    left, s, right_t = torch.linalg.svd(w.double(), full_matrices=False)
    rank = min(rank, s.numel())
    root = s[:rank].sqrt()
    u = (left[:, :rank] * root).float()
    v = (right_t[:rank].T * root).float()
    if itq and rank > 1:
        rot = joint_itq(u, v, iterations, generator)
        u, v = u @ rot, v @ rot
    h, l_u = _rank_one(u.abs())
    g, l_v = _rank_one(v.abs())
    return u, v, h, g, l_u * l_v


# -- the layer ------------------------------------------------------------------

class LittleBitLinear(nn.Module):
    """A drop-in ``nn.Linear`` replacement holding binary factors.

    ``u_latent`` and ``v_latent`` are the real-valued shadows that training
    updates; only their signs are used. :meth:`packed` returns what would be
    stored."""

    def __init__(self, in_features: int, out_features: int, rank: int,
                 residual: bool = True, bias: bool = False, k: float = 100.0) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.rank, self.residual, self.k = rank, residual, k
        paths = 2 if residual else 1
        self.u_latent = nn.Parameter(torch.zeros(paths, out_features, rank))
        self.v_latent = nn.Parameter(torch.zeros(paths, in_features, rank))
        self.h = nn.Parameter(torch.ones(paths, out_features))
        self.g = nn.Parameter(torch.ones(paths, in_features))
        self.l = nn.Parameter(torch.ones(paths, rank))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, residual: bool = True,
                    itq: bool = True, iterations: int = 50,
                    generator: Optional[torch.Generator] = None) -> "LittleBitLinear":
        layer = cls(linear.in_features, linear.out_features, rank, residual,
                    bias=linear.bias is not None)
        target = linear.weight.detach().float()
        with torch.no_grad():
            for p in range(2 if residual else 1):
                u, v, h, g, l = dual_svid(target, rank, itq, iterations, generator)
                r = u.shape[1]
                layer.u_latent[p, :, :r] = u
                layer.v_latent[p, :, :r] = v
                layer.h[p], layer.g[p] = h, g
                layer.l[p, :r] = l
                layer.l[p, r:] = 0.0
                target = target - layer._path_weight(p)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer.to(linear.weight.device)

    def _signs(self, p: int):
        if torch.is_grad_enabled() and self.training:
            return smooth_sign(self.u_latent[p], self.k), smooth_sign(self.v_latent[p], self.k)
        return (torch.where(self.u_latent[p] >= 0, 1.0, -1.0).to(self.h.dtype),
                torch.where(self.v_latent[p] >= 0, 1.0, -1.0).to(self.h.dtype))

    def _path_weight(self, p: int) -> torch.Tensor:
        su, sv = self._signs(p)
        return (self.h[p, :, None] * su * self.l[p]) @ (sv * self.g[p, :, None]).T

    def weight(self) -> torch.Tensor:
        """The dense equivalent (for inspection and tests)."""
        return sum(self._path_weight(p) for p in range(self.u_latent.shape[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        for p in range(self.u_latent.shape[0]):
            su, sv = self._signs(p)
            y = (((x * self.g[p].to(x.dtype)) @ sv.to(x.dtype)) * self.l[p].to(x.dtype))
            y = (y @ su.to(x.dtype).T) * self.h[p].to(x.dtype)
            out = y if out is None else out + y
        return out if self.bias is None else out + self.bias.to(out.dtype)

    def bits(self) -> int:
        return layer_bits(self.out_features, self.in_features, self.rank, self.residual)

    def packed(self) -> dict:
        """Sign bits packed eight to a byte, scales in fp16: the stored form."""
        import numpy as np
        out = {"shape": (self.out_features, self.in_features), "rank": self.rank}
        for p in range(self.u_latent.shape[0]):
            for name, latent in (("u", self.u_latent[p]), ("v", self.v_latent[p])):
                out[f"{name}{p}"] = np.packbits((latent.detach().cpu().numpy() >= 0).ravel())
            for name in ("h", "g", "l"):
                out[f"{name}{p}"] = getattr(self, name)[p].detach().cpu().half().numpy()
        return out


# -- whole models -------------------------------------------------------------

def default_eligible(name: str, module: nn.Module) -> bool:
    """Hidden matrices only, as for Muon: embeddings, heads, routers and narrow
    layers stay in full precision (they are small and their errors compound)."""
    lowered = name.lower()
    if any(s in lowered for s in ("embedding", "head", "router", "halt", "gate")):
        return False
    return isinstance(module, nn.Linear) and min(module.weight.shape) >= 32


def compress(model: nn.Module, bpw: float, residual: bool = True, itq: bool = True,
             eligible: Callable[[str, nn.Module], bool] = default_eligible,
             seed: int = 0) -> dict:
    """Replace every eligible ``nn.Linear`` in ``model`` in place. Returns a
    report: layers replaced, original and compressed bits for them."""
    generator = torch.Generator().manual_seed(seed)
    targets = [(name, m) for name, m in model.named_modules() if eligible(name, m)]
    original = compressed = floored = 0
    for name, linear in targets:
        rank = rank_for_bpw(linear.out_features, linear.in_features, bpw, residual)
        # Below a size-dependent floor the fp16 scales alone exceed the budget
        # (about 0.115 bpw for a 1024 x 384 matrix with the residual path), and
        # rank 1 is still over it. Counted, so the report cannot overstate.
        floored += layer_bits(linear.out_features, linear.in_features, rank, residual) > \
            bpw * linear.weight.numel()
        rank = min(rank, min(linear.out_features, linear.in_features))
        new = LittleBitLinear.from_linear(linear, rank, residual, itq, generator=generator)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], new)
        original += 16 * linear.weight.numel()
        compressed += new.bits()
    return {"layers": len(targets), "over_budget_layers": floored,
            "fp16_bits": original, "compressed_bits": compressed,
            "bits_per_weight": compressed / max(original / 16, 1)}


def distill(student: nn.Module, teacher: nn.Module, batches: Iterable,
            logits_of: Callable[[nn.Module, object], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            steps: int, lr: float = 1e-4, hidden_weight: float = 10.0,
            temperature: float = 1.0) -> list[float]:
    """Quantisation-aware training by distillation (LittleBit Eq. 11):
    ``KL(teacher || student)`` on the output distribution plus
    ``hidden_weight`` times the MSE between final hidden states.

    ``logits_of(model, batch)`` returns ``(logits, hidden, mask)`` for the
    positions to score. Adam with cosine decay and 2% warmup, as the paper."""
    teacher.eval()
    student.train()
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    warm = max(1, int(0.02 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    history = []
    it = iter(batches)
    for _ in range(steps):
        batch = next(it)
        with torch.no_grad():
            t_logits, t_hidden, mask = logits_of(teacher, batch)
        s_logits, s_hidden, _ = logits_of(student, batch)
        m = mask.float()
        kl = F.kl_div(F.log_softmax(s_logits.float() / temperature, -1),
                      F.log_softmax(t_logits.float() / temperature, -1),
                      log_target=True, reduction="none").sum(-1)
        kl = (kl * m).sum() / m.sum().clamp_min(1) * temperature ** 2
        mse = (((s_hidden.float() - t_hidden.float()) ** 2).mean(-1) * m).sum() / m.sum().clamp_min(1)
        loss = kl + hidden_weight * mse
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        history.append(float(loss))
    student.eval()
    return history
