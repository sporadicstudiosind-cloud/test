"""Conservative finite-volume updates with learned flux corrections.

Architecture §9.3 corrects the archived ORRERY claim that "a coarse conservative
solve plus a learned correction cannot invent mass". A learned *per-cell* update
absolutely can invent mass. What cannot is a learned correction applied to
**oriented shared faces**, because each internal face contributes equally and
oppositely to the two cells that share it and therefore cancels in the global
budget.

This module implements both and the test suite demonstrates the difference: the
face-form correction conserves to machine precision in float64, the cell-form
correction does not. That is the whole argument, made executable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ConservationError(ValueError):
    pass


@dataclass(frozen=True)
class Mesh1D:
    """A uniform 1-D finite-volume mesh.

    Faces are indexed ``0..n``; face ``i`` sits between cell ``i-1`` and cell
    ``i``. Faces ``0`` and ``n`` are boundary faces.
    """

    n_cells: int
    dx: float

    @property
    def volumes(self) -> np.ndarray:
        return np.full(self.n_cells, self.dx, dtype=np.float64)

    @property
    def n_faces(self) -> int:
        return self.n_cells + 1


def upwind_flux(u: np.ndarray, velocity: float, periodic: bool = True) -> np.ndarray:
    """First-order upwind face fluxes for linear advection ``u_t + c u_x = 0``."""
    u = np.asarray(u, dtype=np.float64)
    n = u.size
    if periodic:
        left = np.concatenate([u[-1:], u])          # value on the left of each face
        right = np.concatenate([u, u[:1]])          # value on the right of each face
    else:
        left = np.concatenate([u[:1], u])
        right = np.concatenate([u, u[-1:]])
    return velocity * (left if velocity >= 0 else right)


def apply_face_update(
    u: np.ndarray,
    face_flux: np.ndarray,
    mesh: Mesh1D,
    dt: float,
    sources: np.ndarray | None = None,
) -> np.ndarray:
    """Flux-form update. Internal face contributions telescope exactly.

    ``U_i^{n+1} = U_i^n - dt/V_i * (F_{i+1/2} - F_{i-1/2}) + dt * S_i``

    Because the *same* ``F`` value is subtracted from one cell and added to its
    neighbour, the interior sum is unchanged by construction, to the precision
    of the arithmetic.
    """
    u = np.asarray(u, dtype=np.float64)
    face_flux = np.asarray(face_flux, dtype=np.float64)
    if face_flux.size != mesh.n_faces:
        raise ConservationError(
            f"expected {mesh.n_faces} face fluxes, got {face_flux.size}"
        )
    divergence = (face_flux[1:] - face_flux[:-1]) / mesh.dx
    out = u - dt * divergence
    if sources is not None:
        out = out + dt * np.asarray(sources, dtype=np.float64)
    return out


def apply_cell_update(
    u: np.ndarray, cell_correction: np.ndarray, dt: float
) -> np.ndarray:
    """Non-conservative per-cell update, kept for the contrast test only.

    This is the shape of correction the archived specification implicitly
    allowed. It is *not* conservative: nothing ties one cell's correction to its
    neighbour's.
    """
    return np.asarray(u, dtype=np.float64) + dt * np.asarray(
        cell_correction, dtype=np.float64
    )


def total_mass(u: np.ndarray, mesh: Mesh1D) -> float:
    return float(np.sum(np.asarray(u, dtype=np.float64) * mesh.volumes))


@dataclass(frozen=True)
class Budget:
    """An open-system balance over a fixed control volume.

    Architecture §9.4: stored mass in a waterfall domain may rise while inflow
    exceeds outflow. That is not a conservation failure, and demanding constant
    total mass on an open domain — as the archived spec's evaluation section did
    — is simply the wrong check.
    """

    stored_start: float
    stored_end: float
    inflow: float
    outflow: float
    sources: float

    @property
    def expected_change(self) -> float:
        return self.inflow - self.outflow + self.sources

    @property
    def actual_change(self) -> float:
        return self.stored_end - self.stored_start

    @property
    def absolute_error(self) -> float:
        return abs(self.actual_change - self.expected_change)

    def relative_error(self, floor: float) -> float:
        """Relative error against a declared normalization floor.

        §9.4 requires the floor: a near-zero reference quantity otherwise
        produces a meaningless relative error.
        """
        if floor <= 0:
            raise ConservationError("normalization floor must be positive")
        denominator = max(abs(self.expected_change), floor)
        return self.absolute_error / denominator

    def closes(self, tolerance: float, floor: float) -> bool:
        return self.relative_error(floor) <= tolerance


def integrate_budget(
    u0: np.ndarray,
    u1: np.ndarray,
    mesh: Mesh1D,
    inflow_flux: float,
    outflow_flux: float,
    dt: float,
    sources: np.ndarray | None = None,
) -> Budget:
    """Assemble an open-system budget from a single step."""
    source_total = (
        0.0
        if sources is None
        else float(np.sum(np.asarray(sources, dtype=np.float64) * mesh.volumes) * dt)
    )
    return Budget(
        stored_start=total_mass(u0, mesh),
        stored_end=total_mass(u1, mesh),
        inflow=inflow_flux * dt,
        outflow=outflow_flux * dt,
        sources=source_total,
    )


def cfl_number(velocity: float, dt: float, dx: float) -> float:
    return abs(velocity) * dt / dx


def check_admissible(
    u: np.ndarray,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[bool, float]:
    """Bound check for quantities with a physical range (phase fraction, density).

    Returns ``(ok, worst_violation)``. §9.3: if constraints cannot be satisfied,
    reject the step or reduce the timestep rather than clipping away a large
    error silently. The caller decides; this reports.
    """
    u = np.asarray(u, dtype=np.float64)
    worst = 0.0
    if lower is not None:
        worst = max(worst, float(np.max(np.maximum(lower - u, 0.0))))
    if upper is not None:
        worst = max(worst, float(np.max(np.maximum(u - upper, 0.0))))
    return (worst == 0.0, worst)
