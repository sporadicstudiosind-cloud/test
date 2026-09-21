"""Loss aggregation, with every term named and separately reported.

A single scalar hides which objective is actually moving. Every term is kept
addressable so the training log can show that, for example, the router balance
term went to zero while the field term did nothing — which is a collapsed
router, not progress.

Default weights are engineering choices, not results. They are recorded in the
run manifest so a later comparison is against a known mixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import torch


@dataclass(frozen=True)
class LossWeights:
    text: float = 1.0
    #: Output-softmax z-loss. 1e-4 is PaLM's value and has become the default
    #: everywhere it is used, because the term is not trying to shape the
    #: distribution — it only removes cross-entropy's invariance to a constant
    #: shift of every logit. Large enough to pin the drift, small enough that it
    #: contributes almost nothing to the gradient once the drift is pinned.
    #: Raising it does not make the model more stable, it makes it underconfident.
    text_z: float = 1e-4
    action_op: float = 1.0
    action_scalar: float = 0.5
    image: float = 0.5
    video: float = 0.5
    audio: float = 0.5
    field: float = 1.0
    geometry: float = 0.5
    # The quantity head carries the answer on the numeric families, so it is
    # weighted above the modelling losses that only have to predict context.
    quantity: float = 3.0
    slot_type: float = 0.2
    router_balance: float = 1.0
    router_z: float = 1.0
    depth_kl: float = 1.0
    loop_kl: float = 1.0
    confidence: float = 0.2

    def get(self, name: str, default: float = 1.0) -> float:
        return float(getattr(self, name, default))

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


def combine(
    losses: Mapping[str, torch.Tensor], weights: Optional[LossWeights] = None
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum plus a detached report of every term."""
    weights = weights or LossWeights()
    total: Optional[torch.Tensor] = None
    report: dict[str, float] = {}
    for name, value in losses.items():
        w = weights.get(name)
        report[name] = float(value.detach())
        if w == 0.0:
            continue
        term = w * value
        total = term if total is None else total + term
    if total is None:
        raise ValueError("no loss terms were active")
    report["total"] = float(total.detach())
    return total, report


def grad_global_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5
