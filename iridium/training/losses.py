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


#: Terms a balancer may rescale: the per-modality *task* losses. Router,
#: halting and z-loss terms are regularisers whose weights are deliberate
#: strengths, not scales to be equalised, and are never touched.
TASK_TERMS = frozenset({"text", "action_op", "action_scalar", "image", "video", "audio",
                        "field", "geometry", "quantity"})


class LossBalancer:
    """Equalise the scale of per-modality task losses with a running estimate.

    The problem this addresses is specific to an omnimodal objective. Text is a
    cross-entropy near ``ln(vocab)`` early in training and a few nats later; a
    flow-matching term is a velocity MSE whose natural scale depends on patch
    width and data variance; an action scalar is a squared error in whatever
    units the operands have. Summed with fixed weights, whichever term happens
    to be numerically largest dominates the gradient, and the others learn at a
    rate set by an accident of units rather than by the weights anyone chose.

    ``mode="ema"`` divides each task term by a running mean of its own
    (detached) magnitude, so every term enters the sum at order one and the
    configured :class:`LossWeights` become *relative* priorities, which is what
    they read like. It is a heuristic in the same family as GradNorm (Chen et
    al. 2018) and uncertainty weighting (Kendall et al. 2018), much cheaper
    than either, and it is **unvalidated here**: it changes the optimisation
    problem, so it ships off and should be compared against ``"none"`` on a
    real run before being trusted. Its known failure mode is equalising a term
    that *should* stay small -- a nearly solved modality keeps getting
    gradient at full strength -- which is why the running estimate is floored.
    """

    def __init__(self, mode: str = "none", momentum: float = 0.99, floor: float = 1e-3) -> None:
        if mode not in ("none", "ema"):
            raise ValueError("loss balance mode must be 'none' or 'ema'")
        if not 0.0 < momentum < 1.0:
            raise ValueError("momentum must lie in (0, 1)")
        self.mode, self.momentum, self.floor = mode, momentum, floor
        self.scale: dict[str, float] = {}

    def __call__(self, losses: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.mode == "none":
            return dict(losses)
        out: dict[str, torch.Tensor] = {}
        for name, value in losses.items():
            if name not in TASK_TERMS:
                out[name] = value
                continue
            magnitude = float(value.detach())
            if magnitude == 0.0:          # modality absent from this batch
                out[name] = value
                continue
            prev = self.scale.get(name)
            est = magnitude if prev is None else self.momentum * prev + (1 - self.momentum) * magnitude
            self.scale[name] = est
            out[name] = value / max(est, self.floor)
        return out

    def state_dict(self) -> dict:
        return {"mode": self.mode, "momentum": self.momentum, "floor": self.floor,
                "scale": dict(self.scale)}

    def load_state_dict(self, data: Mapping) -> None:
        self.scale = dict(data.get("scale", {}))
