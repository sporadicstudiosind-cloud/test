"""The dual-system controller: learned operator first, exact solver on demand.

System 1 is the model's spectral/neural prediction: fast, differentiable,
approximate, and with no conservation guarantee whatsoever. System 2 is a
deterministic solver: slow, exact to its discretization, and incapable of
hallucinating.

The controller's only job is to decide, per request, which one the answer is
allowed to come from — and to *label* the answer with which one it came from.
That label is the whole point. A number that looks the same either way is not
the same claim, and an architecture that cannot tell a user which of the two
produced a result has no physical grounding, only physical vocabulary.

Escalation criteria, and one correction
---------------------------------------
The source plan escalated when ``||div u||_2 > 1e-4``. That is a dimensional
quantity compared against a bare constant: the same flow expressed in cm/s
instead of m/s has a hundred times the divergence and would escalate or not
depending on the user's choice of units. The criterion here is the
dimensionless ``||div u|| * L / U`` from ``operators.divergence_norm``, so it
means the same thing in every unit system.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .operators import divergence_norm


class Evidence(enum.IntEnum):
    """How much a number is worth. Ordered; higher is stronger."""

    ILLUSTRATIVE = 0   # generated to look right; no numerical claim
    PREDICTED = 1      # a learned prediction, unchecked
    CHECKED = 2        # a learned prediction whose invariants were measured
    VERIFIED = 3       # produced or corrected by a deterministic solver
    VALIDATED = 4      # agreed with an independent measurement

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class Tolerance:
    """Named, dimensionless thresholds. Every one is a documented choice."""

    divergence: float = 1e-3          # ||div u|| L / U
    mass_budget: float = 1e-6         # relative, against a declared floor
    bounds: float = 0.0               # any violation of a physical range
    energy_growth: float = 1e-2       # relative energy gain in a decaying flow

    def as_dict(self) -> dict[str, float]:
        return {
            "divergence": self.divergence,
            "mass_budget": self.mass_budget,
            "bounds": self.bounds,
            "energy_growth": self.energy_growth,
        }


@dataclass
class Residuals:
    divergence: float = 0.0
    mass_budget: float = 0.0
    bounds: float = 0.0
    energy_growth: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    def worst_violation(self, tol: Tolerance) -> tuple[str, float]:
        """The invariant furthest past its threshold, as a ratio."""
        ratios = {
            "divergence": self.divergence / max(tol.divergence, 1e-30),
            "mass_budget": self.mass_budget / max(tol.mass_budget, 1e-30),
            "bounds": self.bounds / max(tol.bounds, 1e-30) if tol.bounds > 0
            else (float("inf") if self.bounds > 0 else 0.0),
            "energy_growth": self.energy_growth / max(tol.energy_growth, 1e-30),
        }
        name = max(ratios, key=ratios.get)
        return name, ratios[name]

    def passes(self, tol: Tolerance) -> bool:
        return self.worst_violation(tol)[1] <= 1.0

    def as_dict(self) -> dict[str, float]:
        out = {
            "divergence": self.divergence,
            "mass_budget": self.mass_budget,
            "bounds": self.bounds,
            "energy_growth": self.energy_growth,
        }
        out.update(self.extra)
        return out


@dataclass
class VerificationRecord:
    """What was claimed, how it was obtained, and what was checked."""

    evidence: Evidence
    residuals: Residuals
    tolerance: Tolerance
    escalated: bool
    reason: str
    solver: Optional[str] = None
    cost_seconds: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        worst, ratio = self.residuals.worst_violation(self.tolerance)
        return (
            f"evidence={self.evidence.label} escalated={self.escalated} "
            f"worst={worst}({ratio:.2f}x) reason={self.reason}"
        )


def velocity_residuals(
    u: np.ndarray,
    v: np.ndarray,
    length: float = 2.0 * np.pi,
    reference_energy: Optional[float] = None,
) -> Residuals:
    res = Residuals(divergence=divergence_norm(u, v, length))
    if reference_energy is not None and reference_energy > 0:
        energy = float(0.5 * np.mean(u ** 2 + v ** 2))
        res.energy_growth = max(0.0, energy - reference_energy) / reference_energy
    return res


def scalar_budget_residual(
    stored_start: float, stored_end: float, inflow: float, outflow: float,
    sources: float = 0.0, floor: float = 1e-9,
) -> float:
    """Open-system balance error, relative to a declared normalization floor.

    An open domain does not conserve total mass — a filling reservoir is not a
    solver bug — so the check is that the *budget closes*, not that the total
    is constant. See ``physics/fluxes.py::Budget``.
    """
    expected = inflow - outflow + sources
    actual = stored_end - stored_start
    return abs(actual - expected) / max(abs(expected), floor)


class DualSystemController:
    """Runs System 1, checks it, escalates to System 2 when the check fails."""

    def __init__(
        self,
        system1: Callable[[dict], dict],
        system2: Callable[[dict], dict],
        tolerance: Tolerance | None = None,
        residual_fn: Optional[Callable[[dict, dict], Residuals]] = None,
        always_verify: bool = False,
        solver_name: str = "deterministic",
    ) -> None:
        self.system1 = system1
        self.system2 = system2
        self.tolerance = tolerance or Tolerance()
        self.residual_fn = residual_fn
        self.always_verify = always_verify
        self.solver_name = solver_name

    def resolve(
        self, request: dict, required: Evidence = Evidence.CHECKED
    ) -> tuple[dict, VerificationRecord]:
        import time

        started = time.monotonic()
        if self.always_verify or required >= Evidence.VERIFIED:
            result = self.system2(request)
            residuals = self._residuals(request, result)
            return result, VerificationRecord(
                evidence=Evidence.VERIFIED,
                residuals=residuals,
                tolerance=self.tolerance,
                escalated=True,
                reason="caller required solver-grade evidence",
                solver=self.solver_name,
                cost_seconds=time.monotonic() - started,
            )

        prediction = self.system1(request)
        residuals = self._residuals(request, prediction)
        if residuals.passes(self.tolerance):
            return prediction, VerificationRecord(
                evidence=Evidence.CHECKED,
                residuals=residuals,
                tolerance=self.tolerance,
                escalated=False,
                reason="learned prediction satisfied every declared invariant",
                cost_seconds=time.monotonic() - started,
            )

        worst, ratio = residuals.worst_violation(self.tolerance)
        result = self.system2(request)
        verified_residuals = self._residuals(request, result)
        return result, VerificationRecord(
            evidence=Evidence.VERIFIED,
            residuals=verified_residuals,
            tolerance=self.tolerance,
            escalated=True,
            reason=f"{worst} exceeded tolerance by {ratio:.1f}x",
            solver=self.solver_name,
            cost_seconds=time.monotonic() - started,
            payload={"rejected_residuals": residuals.as_dict()},
        )

    def _residuals(self, request: dict, result: dict) -> Residuals:
        if self.residual_fn is not None:
            return self.residual_fn(request, result)
        if "u" in result and "v" in result:
            return velocity_residuals(
                result["u"], result["v"], request.get("length", 2.0 * np.pi),
                request.get("reference_energy"),
            )
        return Residuals()
