"""The continual-learning flywheel: LoRA + differential privacy + EWC.

The originating vision: *the model should not share data between users' chats,
but should learn from all of them in the background.* Those two requirements
are compatible only through a specific pipeline, and each stage of it is a
real mechanism, not a label:

1. **Isolation at serving time.** Nothing crosses between live streams.
   ``runtime/registry.py`` makes that structural.
2. **Verified traces only.** A trace is eligible if its outcome was *checked* —
   a solver agreed, a sandbox exited zero, a goal predicate passed. Learning
   from unverified interactions teaches the model its own errors.
3. **Differential privacy on the gradients.** Per-example gradient clipping to
   norm ``C`` plus Gaussian noise of scale ``sigma * C``. The privacy of the
   result is a property of ``(sigma, sampling rate, steps)``, and
   ``dp_epsilon_bound`` states which bound is being used and what it does and
   does not cover.
4. **Low-rank adapters.** Updates land on LoRA matrices, not the base weights,
   so a bad batch is revertible by deleting a file.
5. **Elastic Weight Consolidation on promotion.** When an adapter is merged,
   a Fisher-weighted penalty holds the parameters that mattered for previous
   capabilities in place.

What this does *not* provide: DP on the *data selection*, membership-inference
resistance for the base checkpoint, or any guarantee about a trace whose text
the filter failed to recognise as identifying. Those are open, and are listed
in the capability register as such.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# 1 · Low-rank adapters
# --------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """``y = W x + (alpha / r) * B A x`` with ``W`` frozen."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        # B starts at zero so the adapted model is *exactly* the base model
        # before any training. A randomly initialized B silently changes the
        # deployed behaviour the moment an adapter is attached.
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.a), self.b
        )

    @torch.no_grad()
    def merge(self) -> None:
        self.base.weight.add_(self.scaling * (self.b @ self.a))
        nn.init.zeros_(self.b)


def attach_lora(
    model: nn.Module, pattern: str = r"bank\..*\.ffn\.(gate|up|down)",
    rank: int = 8, alpha: float = 16.0,
) -> dict[str, int]:
    """Wrap matching ``nn.Linear`` modules. Returns what was adapted."""
    regex = re.compile(pattern)
    targets = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and regex.search(name)
    ]
    for name in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, LoRALinear(getattr(parent, attr), rank, alpha))
    trainable = sum(
        p.numel() for n, p in model.named_parameters() if p.requires_grad and (".a" in n or ".b" in n)
    )
    return {"adapted_modules": len(targets), "adapter_parameters": trainable}


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [
        p for module in model.modules() if isinstance(module, LoRALinear)
        for p in (module.a, module.b)
    ]


def merge_all(model: nn.Module) -> int:
    merged = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            merged += 1
    return merged


# --------------------------------------------------------------------------
# 2 · Differentially private updates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DPConfig:
    clip_norm: float = 1.0
    noise_multiplier: float = 1.0
    lot_size: int = 8
    delta: float = 1e-6


def clip_and_accumulate(
    parameters: Sequence[nn.Parameter], accumulator: list[torch.Tensor],
    clip_norm: float,
) -> float:
    """Clip *this example's* gradient to ``clip_norm`` and add it to the lot."""
    total = torch.sqrt(
        sum((p.grad.detach().pow(2).sum() for p in parameters if p.grad is not None),
            torch.zeros(()))
    )
    scale = float(min(1.0, clip_norm / float(total + 1e-12)))
    for slot, p in zip(accumulator, parameters):
        if p.grad is not None:
            slot.add_(p.grad.detach() * scale)
    return float(total)


def add_noise_and_apply(
    parameters: Sequence[nn.Parameter], accumulator: list[torch.Tensor],
    cfg: DPConfig, generator: Optional[torch.Generator] = None,
) -> None:
    sigma = cfg.noise_multiplier * cfg.clip_norm
    for slot, p in zip(accumulator, parameters):
        noise = torch.normal(
            mean=0.0, std=sigma, size=slot.shape, generator=generator
        )
        p.grad = (slot + noise) / cfg.lot_size
        slot.zero_()


def dp_epsilon_bound(
    cfg: DPConfig, steps: int, dataset_size: int
) -> dict[str, float | str]:
    """A *stated* bound, with its assumptions attached.

    This is the classical strong-composition bound for the subsampled Gaussian
    mechanism. It is loose — a Renyi or PLD accountant gives a smaller epsilon
    for the same noise — and it is reported as an upper bound, not as "the"
    privacy of the run. Quoting a single epsilon with no accountant named,
    as the source plan did, is not a privacy claim that can be checked.
    """
    q = cfg.lot_size / max(dataset_size, 1)
    sigma = cfg.noise_multiplier
    if sigma <= 0:
        return {"epsilon": float("inf"), "accountant": "none",
                "note": "zero noise provides no differential privacy"}
    eps_step = q * math.sqrt(2.0 * math.log(1.25 / cfg.delta)) / sigma
    eps_total = eps_step * math.sqrt(2.0 * steps * math.log(1.0 / cfg.delta))
    return {
        "epsilon_upper_bound": eps_total,
        "delta": cfg.delta,
        "sampling_rate": q,
        "noise_multiplier": sigma,
        "steps": float(steps),
        "accountant": "strong composition, subsampled Gaussian (loose)",
        "covers": "gradient release only; not data selection or trace filtering",
    }


# --------------------------------------------------------------------------
# 3 · Elastic Weight Consolidation
# --------------------------------------------------------------------------


class ElasticWeightConsolidation:
    """Fisher-weighted anchor on the parameters an old task depended on."""

    def __init__(self, model: nn.Module, lam: float = 100.0) -> None:
        self.model = model
        self.lam = lam
        self.fisher: dict[str, torch.Tensor] = {}
        self.anchor: dict[str, torch.Tensor] = {}

    def estimate(
        self, loss_fn: Callable[[], torch.Tensor], n_batches: int = 16
    ) -> None:
        """Diagonal empirical Fisher: ``E[(d log p / d theta)^2]``."""
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters() if p.requires_grad
        }
        for _ in range(n_batches):
            self.model.zero_grad(set_to_none=True)
            loss_fn().backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)
        self.fisher = {n: v / n_batches for n, v in fisher.items()}
        self.anchor = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters() if p.requires_grad
        }
        self.model.zero_grad(set_to_none=True)

    def penalty(self) -> torch.Tensor:
        if not self.fisher:
            return torch.zeros(())
        total = torch.zeros(())
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                total = total + (
                    self.fisher[n] * (p - self.anchor[n]).pow(2)
                ).sum()
        return 0.5 * self.lam * total

    def state(self) -> dict[str, float]:
        if not self.fisher:
            return {"parameters": 0.0, "fisher_mass": 0.0}
        mass = float(sum(float(v.sum()) for v in self.fisher.values()))
        return {"parameters": float(len(self.fisher)), "fisher_mass": mass,
                "lambda": self.lam}


# --------------------------------------------------------------------------
# 4 · Trace eligibility
# --------------------------------------------------------------------------

_IDENTIFIER_PATTERNS = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),                 # email
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),                  # long digit runs
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),             # IPv4
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\b"),
)


@dataclass
class Trace:
    text: str
    verified: bool
    evidence: str = "predicted"
    owner: str = ""


def eligible(trace: Trace) -> tuple[bool, str]:
    """A trace may be learned from only if it was checked and is clean."""
    if not trace.verified:
        return False, "outcome was never verified"
    if trace.evidence in ("illustrative", "predicted"):
        return False, f"evidence level {trace.evidence!r} is below 'checked'"
    for pattern in _IDENTIFIER_PATTERNS:
        if pattern.search(trace.text):
            return False, "contains an identifier-shaped token"
    return True, "verified and clean"


def filter_traces(traces: Iterable[Trace]) -> tuple[list[Trace], dict[str, int]]:
    kept: list[Trace] = []
    reasons: dict[str, int] = {}
    for trace in traces:
        ok, why = eligible(trace)
        reasons[why] = reasons.get(why, 0) + 1
        if ok:
            kept.append(trace)
    return kept, reasons
