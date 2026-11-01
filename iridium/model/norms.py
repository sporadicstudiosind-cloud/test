""""Continuous dynamic layer norms": normalisation-free elementwise norms and
per-token adaptive normalisation, plus the factory that lets a config select
among them and ``layers.RMSNorm`` without a special case.

Two different senses of "dynamic" live here, and conflating them is the
mistake this docstring exists to head off:

* **Elementwise, statistics-free "dynamic":** :class:`DyT` and :class:`Derf`
  replace RMSNorm/LayerNorm's *data-dependent statistic* (a mean or a root-
  mean-square computed over the feature dimension, recomputed every forward
  pass) with a *fixed, learned, elementwise function* that only ever sees one
  scalar at a time. Nothing here is a function of the batch or of the other
  features in the same token; "dynamic" names the shape of the function
  (Zhu et al. found trained normalisation layers already look like a
  saturating S-curve of their input, hence a literal ``tanh``), not any
  runtime adaptivity. ``DyT`` and ``Derf`` cost strictly less compute than
  RMSNorm, since there is no reduction to perform.
* **Per-token, input-conditioned "dynamic":** :class:`AdaptiveRMSNorm` keeps
  RMSNorm's actual normalising statistic and instead makes the *affine
  correction after it* (gain and shift) a function of a conditioning vector,
  predicted fresh per token. This is "dynamic" in the ordinary adaptive-norm
  sense (AdaLN/FiLM), and it is the one the "self-conditioned" mode below is
  actually risky, precisely because it goes back to reading the statistics of
  the very thing it is about to modulate.

Dynamic Tanh
------------
``DyT(x) = gamma * tanh(alpha * x) + beta`` (Zhu, Chen, He, LeCun, Liu,
"Transformers without Normalization", arXiv:2503.10622, CVPR 2025). ``gamma``
and ``beta`` are per-channel, as in RMSNorm/LayerNorm; ``alpha`` is a single
learned scalar controlling how sharply the tanh saturates. The paper is
explicit that ``alpha``'s *initial* value matters far more for language
models than for vision transformers -- their vision recipe uses
``alpha_init=0.5`` uniformly, but their LLM experiments needed per-model
tuning of the initial value (they report sweeping it, with results sensitive
enough that they call it out as the one hyperparameter DyT reintroduces).
This module therefore takes ``alpha_init`` as a required-in-spirit keyword
(default 0.5, the vision default) rather than silently assuming a vision
workload, and the caller integrating this into a language config should
treat it as swept, not fixed.

Derf: the December-2025 follow-up
----------------------------------
Chen, Lu, Zhu, Sun, Liu, "Stronger Normalization-Free Transformers"
(arXiv:2512.10938) run a search over elementwise replacement functions and
report ``Derf(x) = erf(alpha * x + s)`` -- the rescaled Gaussian CDF, with
``alpha`` and ``s`` both learned scalars -- as the strongest, beating LayerNorm,
RMSNorm and DyT across vision, speech and DNA-sequence tasks. The paper's
own contribution is exactly that core function; it does not restate whether
the per-channel ``gamma``/``beta`` affine wrap is kept, because DyT already
established that convention for this family and Derf is presented as a
replacement for DyT's inner ``tanh(alpha * x)``, not for its outer affine
step. :class:`Derf` here follows that lineage -- ``gamma * erf(alpha * x + s)
+ beta`` -- which is this file's extrapolation from the confirmed core
formula, not a claim that the paper states the wrapped form verbatim; anyone
reproducing their exact numbers should check the paper's appendix or
reference code before relying on the affine wrapper matching.

Adaptive RMSNorm
----------------
:class:`AdaptiveRMSNorm` is RMSNorm (the same statistic, same epsilon
placement as ``layers.RMSNorm``) whose gain and shift are predicted from a
conditioning vector by two zero-initialised linear heads, so the module
starts (before any training) as *exactly* plain RMSNorm with a weight of all
ones -- ``gain = 1 + tanh(0) = 1``, ``shift = tanh(0) = 0``. Gain and shift
stay bounded to ``+-bound`` around their identity values via ``tanh``, for
the reason given next.

``self_conditioned=True`` uses the token's own pre-normalisation state as the
conditioning vector, rather than an externally supplied one. Be clear-eyed
about what that risks: the entire reason RMSNorm exists is to remove the
token's own scale from the residual stream, and a norm that reads that same
unremoved scale to decide how hard to re-scale its own output is, in the
worst case, undoing its own job -- a large-magnitude token could learn to
request a large gain and reproduce (or amplify) exactly the scale RMSNorm
was supposed to have normalised away, and because the read and the modulate
happen in the same forward call there is no obvious signal in the loss that
distinguishes "the norm is helping" from "the norm has become a no-op with
extra parameters". The zero-init (so this behaviour has to be *learned into*
rather than starting active) and the ``tanh`` bound (so however hard training
pushes, the correction cannot exceed ``+-bound`` around identity) are the two
mitigations this module has; they bound the damage, they do not prove the
self-conditioned mode is safe to enable, and that risk cannot be exercised or
disproved by a unit test on random weights -- only by training.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .layers import RMSNorm, at_least_fp32


class DyT(nn.Module):
    """``gamma * tanh(alpha * x) + beta``. See the module docstring for the paper."""

    def __init__(self, d: int, alpha_init: float = 0.5) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.tanh(self.alpha * x) + self.beta

    @staticmethod
    def param_count(d: int) -> int:
        return 2 * d + 1


class Derf(nn.Module):
    """``gamma * erf(alpha * x + s) + beta``. See the module docstring for the paper
    and the caveat about the affine wrapper being this file's extrapolation."""

    def __init__(self, d: int, alpha_init: float = 1.0, s_init: float = 0.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.s = nn.Parameter(torch.tensor(float(s_init)))
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.erf(self.alpha * x + self.s) + self.beta

    @staticmethod
    def param_count(d: int) -> int:
        return 2 * d + 2


class AdaptiveRMSNorm(nn.Module):
    """RMSNorm with per-token gain/shift predicted from a conditioning vector.

    ``cond_dim`` is ignored when ``self_conditioned=True`` (the conditioning
    vector is then ``x`` itself, width ``d``); see the module docstring for
    what that trades away. ``bound`` caps how far ``tanh`` lets the predicted
    gain/shift move from the identity values (1 and 0) at any input.
    """

    def __init__(self, d: int, cond_dim: Optional[int] = None, eps: float = 1e-5,
                 self_conditioned: bool = False, bound: float = 1.0) -> None:
        super().__init__()
        if not self_conditioned and cond_dim is None:
            raise ValueError("cond_dim is required unless self_conditioned=True")
        self.eps = eps
        self.self_conditioned = self_conditioned
        self.bound = bound
        c_dim = d if self_conditioned else cond_dim
        self.to_gain = nn.Linear(c_dim, d)
        self.to_shift = nn.Linear(c_dim, d)
        nn.init.zeros_(self.to_gain.weight)
        nn.init.zeros_(self.to_gain.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.self_conditioned:
            cond = x
        elif cond is None:
            raise ValueError("cond is required unless self_conditioned=True")
        dtype = x.dtype
        x32 = at_least_fp32(x)
        rms = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        normed = (x32 * rms).to(dtype)
        gain = 1.0 + self.bound * torch.tanh(self.to_gain(cond))
        shift = self.bound * torch.tanh(self.to_shift(cond))
        return normed * gain + shift

    @staticmethod
    def param_count(d: int, cond_dim: int) -> int:
        """``cond_dim`` here is the conditioning width actually used -- pass
        ``d`` itself for a self-conditioned instance."""
        return 2 * (cond_dim * d + d)


def norm_param_count(kind: str, d: int, cond_dim: Optional[int] = None) -> int:
    """The parameter count :func:`make_norm` would build, without building it.

    Exists so callers costing a config (``blocks.ParallelBlock.param_count``,
    eventually ``config.py``) can account for a norm choice without
    instantiating torch modules, matching how ``config.py`` costs everything
    else from shapes alone.
    """
    kind = kind.lower()
    if kind == "rms":
        return d
    if kind == "dyt":
        return DyT.param_count(d)
    if kind in ("derf",):
        return Derf.param_count(d)
    if kind == "adaptive":
        if cond_dim is None:
            raise ValueError("cond_dim is required to cost an adaptive norm")
        return AdaptiveRMSNorm.param_count(d, cond_dim)
    raise ValueError(f"unknown norm kind {kind!r}")


def make_norm(kind: str, d: int, eps: float = 1e-5, **kwargs) -> nn.Module:
    """Build a norm module by name.

    ``kind="rms"`` returns ``iridium.model.layers.RMSNorm`` itself -- not a
    reimplementation -- so selecting it through this factory is bit-identical
    to constructing ``RMSNorm`` directly; that is the whole point of routing
    the default path through here rather than adding a parallel "kind" enum
    only the new norms understand. ``kwargs`` are forwarded to the chosen
    norm's constructor (``alpha_init`` for ``dyt``, ``s_init`` for ``derf``,
    ``cond_dim``/``self_conditioned``/``bound`` for ``adaptive``).
    """
    kind = kind.lower()
    if kind == "rms":
        return RMSNorm(d, eps)
    if kind == "dyt":
        return DyT(d, **kwargs)
    if kind in ("derf",):
        return Derf(d, **kwargs)
    if kind == "adaptive":
        return AdaptiveRMSNorm(d, eps=eps, **kwargs)
    raise ValueError(f"unknown norm kind {kind!r}; choose rms, dyt, derf or adaptive")
