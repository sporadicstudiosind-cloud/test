""""Dynamic weights", done at the only rank where they are affordable.

The naive reading of "dynamic weights" is a hypernetwork that emits a full
``d_out x d_in`` matrix per token (Ha, Dai & Le, *HyperNetworks*, 2016,
arXiv:1609.09106 -- the paper that established generating one network's
weights from another). That is not a knob, it is a different asymptotic
cost: a batch of ``B`` tokens at ``d`` in ``d`` out now needs
``B * d_in * d_out`` numbers materialised, versus ``d_in * d_out`` for a
plain shared matrix, on top of ``B * d_in * d_out`` multiply-adds to *apply*
each token's private matrix once. At ``d = 2048`` and a 4k-token batch that
is tens of gigabytes of transient weight tensors for one layer -- HyperLSTM,
the paper's own sequel work, hits exactly this wall and is why it generates a
per-timestep *scaling* vector for a shared weight rather than the weight
itself.

The tractable form is low rank: keep one shared matrix ``W`` and add a
per-token correction that factors through a rank-``r`` bottleneck gated by
something the token itself produces. This is the same shape as LoRA
(Hu et al. 2021) made *input-conditioned* rather than fixed after training --
sometimes called "dynamic LoRA" or framed as a token-level mixture of LoRA
experts, and it is also, structurally, HyperLSTM's per-step scaling vector
applied to a feedforward matrix instead of a recurrent one: a per-token gate
in a rank-``r`` space, never a per-token matrix.

``DynamicLowRankLinear`` costs ``O(r * (in + out + cond))`` extra parameters
and ``O(r * (in + out))`` extra multiply-adds per token over a plain
``Linear`` -- both linear in ``r``, not quadratic in ``d`` -- and is
zero-initialised so that switching it on for an existing checkpoint is a
no-op at that instant, the same contract ``PerLayerEmbedding`` uses.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicLowRankLinear(nn.Module):
    """``y = W x + U(diag(g(c)) (V x))``, zero-init so ``y == W x`` at init.

    ``g(c) = sigmoid(gate(c))`` is a per-token gate vector in the rank-``r``
    space, predicted from a conditioning vector ``c`` that defaults to the
    layer's own input ``x`` (``cond_dim`` then must equal ``in_features``) --
    the cheapest available signal, and the one that makes this a drop-in
    replacement for ``nn.Linear`` with no extra input needed at the call
    site. Passing an explicit ``cond`` lets a caller condition the gate on
    something else instead (a routing decision, an external control vector).

    Only ``U``'s weight is zero-initialised. That alone is sufficient for
    exact identity at init regardless of how ``V`` or ``gate`` are
    initialised, because ``U`` sits last in the correction term -- and it is
    the right parameter to zero rather than ``V`` or the gate, because
    leaving ``V`` and ``gate`` at their normal init means their gradients are
    already informative on step one; zeroing ``V`` instead would make the
    correction's first gradient with respect to ``U`` vanish too (``V x`` is
    the input ``U`` differentiates through), stalling both halves of the
    low-rank path together.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, cond_dim: Optional[int] = None) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.cond_dim = cond_dim if cond_dim is not None else in_features

        self.base = nn.Linear(in_features, out_features, bias=False)
        self.V = nn.Linear(in_features, rank, bias=False)
        self.U = nn.Linear(rank, out_features, bias=False)
        nn.init.zeros_(self.U.weight)
        self.gate = nn.Linear(self.cond_dim, rank)

    @staticmethod
    def param_count(in_features: int, out_features: int, rank: int, cond_dim: Optional[int] = None) -> int:
        cond_dim = cond_dim if cond_dim is not None else in_features
        base = in_features * out_features
        v = in_features * rank
        u = rank * out_features
        gate = cond_dim * rank + rank  # gate has a bias, unlike the others
        return base + v + u + gate

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = x if cond is None else cond
        g = torch.sigmoid(self.gate(c))
        return self.base(x) + self.U(g * self.V(x))


class DynamicSwiGLU(nn.Module):
    """Drop-in for ``layers.SwiGLU``, with each of gate/up/down made dynamic.

    Same forward signature as ``layers.SwiGLU`` (``forward(x) -> tensor``) and
    the same expression, ``down(silu(gate(x)) * up(x))``; only the three
    linear maps are replaced. ``down``'s conditioning defaults to its own
    input -- the ``d_ff``-wide mixed activation -- which is a different
    vector from the block's ``d_model``-wide residual input, so its gate
    reacts to the *result* of the gate/up nonlinearity rather than replaying
    the same conditioning signal three matmuls in a row for free.

    At init every ``U`` is zero (see ``DynamicLowRankLinear``), so this
    computes exactly ``down_base(silu(gate_base(x)) * up_base(x))`` -- a
    plain SwiGLU built from ``gate.base``/``up.base``/``down.base`` --
    and ``tests/unit/test_dynamic.py`` checks that against a real
    ``layers.SwiGLU`` sharing those weights to 1e-6.
    """

    def __init__(self, d_model: int, d_ff: int, rank: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.rank = rank
        self.gate = DynamicLowRankLinear(d_model, d_ff, rank, cond_dim=d_model)
        self.up = DynamicLowRankLinear(d_model, d_ff, rank, cond_dim=d_model)
        self.down = DynamicLowRankLinear(d_ff, d_model, rank, cond_dim=d_ff)

    @staticmethod
    def param_count(d_model: int, d_ff: int, rank: int) -> int:
        return (
            DynamicLowRankLinear.param_count(d_model, d_ff, rank, cond_dim=d_model)  # gate
            + DynamicLowRankLinear.param_count(d_model, d_ff, rank, cond_dim=d_model)  # up
            + DynamicLowRankLinear.param_count(d_ff, d_model, rank, cond_dim=d_ff)     # down
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


__all__ = ["DynamicLowRankLinear", "DynamicSwiGLU"]
