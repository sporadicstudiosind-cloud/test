"""Pseudo-spectral incompressible Navier-Stokes in two dimensions.

    du/dt + (u . grad) u = -grad p + nu lap u,    div u = 0

on a doubly periodic square. Nonlinear terms are evaluated in physical space
with 2/3 dealiasing; viscosity is integrated exactly with an integrating
factor; incompressibility is imposed by Leray projection rather than by a
penalty, so ``div u`` is zero to round-off at every step and not merely small.

This is System 2 of the dual-system physics design: the deterministic reference
that the learned operator is checked against and, when the check fails, is
replaced by. It is validated against the **Taylor-Green vortex**, an exact
closed-form solution of the 2-D equations, so the solver's own error is
measurable rather than assumed. ``tests/scientific/test_fluid2d.py`` reports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .operators import (
    divergence_norm,
    grid,
    kinetic_energy,
    project_divergence_free,
    vorticity,
    wavenumbers,
)


@dataclass(frozen=True)
class FluidState:
    u: np.ndarray
    v: np.ndarray
    t: float

    def copy(self) -> "FluidState":
        return FluidState(self.u.copy(), self.v.copy(), self.t)


def taylor_green(n: int, t: float = 0.0, nu: float = 0.0, length: float = 2.0 * np.pi
                 ) -> FluidState:
    """Exact solution: ``u = cos x sin y e^{-2 nu t}``, ``v = -sin x cos y e^{-2 nu t}``.

    The nonlinear term is balanced exactly by the pressure gradient, so this
    decays as a pure diffusion problem while remaining a solution of the full
    nonlinear equations. That is precisely what makes it a sharp test: a solver
    with a broken advection term still gets the right answer only if the
    balance is reproduced numerically as well as analytically.
    """
    x, y = grid(n, length)
    decay = np.exp(-2.0 * nu * t)
    return FluidState(np.cos(x) * np.sin(y) * decay,
                      -np.sin(x) * np.cos(y) * decay, t)


class NavierStokes2D:
    def __init__(self, n: int, nu: float = 1e-3, length: float = 2.0 * np.pi,
                 dealias: bool = True) -> None:
        self.n = n
        self.nu = nu
        self.length = length
        kx = wavenumbers(n, length).reshape(-1, 1)
        ky = wavenumbers(n, length).reshape(1, -1)
        self.kx = np.broadcast_to(kx, (n, n)).copy()
        self.ky = np.broadcast_to(ky, (n, n)).copy()
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.k2_safe = self.k2.copy()
        self.k2_safe[0, 0] = 1.0
        cutoff = (2.0 / 3.0) * np.max(np.abs(wavenumbers(n, length)))
        self.mask = (
            (np.abs(self.kx) <= cutoff) & (np.abs(self.ky) <= cutoff)
            if dealias else np.ones_like(self.k2, dtype=bool)
        )

    # -- spectral helpers -------------------------------------------------

    def _project(self, uh: np.ndarray, vh: np.ndarray):
        dot = self.kx * uh + self.ky * vh
        return uh - self.kx * dot / self.k2_safe, vh - self.ky * dot / self.k2_safe

    def _nonlinear(self, uh: np.ndarray, vh: np.ndarray):
        """``-(u . grad) u`` with dealiasing, returned in spectral space."""
        u = np.real(np.fft.ifft2(uh))
        v = np.real(np.fft.ifft2(vh))
        ux = np.real(np.fft.ifft2(1j * self.kx * uh))
        uy = np.real(np.fft.ifft2(1j * self.ky * uh))
        vx = np.real(np.fft.ifft2(1j * self.kx * vh))
        vy = np.real(np.fft.ifft2(1j * self.ky * vh))
        nu_ = -(u * ux + v * uy)
        nv_ = -(u * vx + v * vy)
        return np.fft.fft2(nu_) * self.mask, np.fft.fft2(nv_) * self.mask

    # -- stepping ---------------------------------------------------------

    def step(self, state: FluidState, dt: float,
             force: Optional[Callable[[float], tuple[np.ndarray, np.ndarray]]] = None
             ) -> FluidState:
        """One Heun (explicit trapezoid) step with exact viscous integration."""
        uh = np.fft.fft2(state.u)
        vh = np.fft.fft2(state.v)
        decay = np.exp(-self.nu * self.k2 * dt)

        nu1, nv1 = self._nonlinear(uh, vh)
        if force is not None:
            fu, fv = force(state.t)
            nu1 = nu1 + np.fft.fft2(fu) * self.mask
            nv1 = nv1 + np.fft.fft2(fv) * self.mask
        nu1, nv1 = self._project(nu1, nv1)

        uh_p = (uh + dt * nu1) * decay
        vh_p = (vh + dt * nv1) * decay
        uh_p, vh_p = self._project(uh_p, vh_p)

        nu2, nv2 = self._nonlinear(uh_p, vh_p)
        if force is not None:
            fu, fv = force(state.t + dt)
            nu2 = nu2 + np.fft.fft2(fu) * self.mask
            nv2 = nv2 + np.fft.fft2(fv) * self.mask
        nu2, nv2 = self._project(nu2, nv2)

        uh_new = (uh + 0.5 * dt * nu1) * decay + 0.5 * dt * nu2
        vh_new = (vh + 0.5 * dt * nv1) * decay + 0.5 * dt * nv2
        uh_new, vh_new = self._project(uh_new, vh_new)
        return FluidState(np.real(np.fft.ifft2(uh_new)),
                          np.real(np.fft.ifft2(vh_new)), state.t + dt)

    def run(self, state: FluidState, dt: float, steps: int,
            record_every: int = 0) -> tuple[FluidState, list[FluidState]]:
        frames: list[FluidState] = []
        for i in range(steps):
            state = self.step(state, dt)
            if record_every and (i + 1) % record_every == 0:
                frames.append(state.copy())
        return state, frames

    def cfl_dt(self, state: FluidState, safety: float = 0.4) -> float:
        dx = self.length / self.n
        speed = float(np.max(np.abs(state.u)) + np.max(np.abs(state.v))) + 1e-12
        advective = safety * dx / speed
        viscous = safety * dx * dx / (4.0 * self.nu + 1e-30)
        return float(min(advective, viscous))

    # -- diagnostics ------------------------------------------------------

    def diagnostics(self, state: FluidState) -> dict[str, float]:
        return {
            "kinetic_energy": kinetic_energy(state.u, state.v),
            "divergence_norm": divergence_norm(state.u, state.v, self.length),
            "max_vorticity": float(np.max(np.abs(vorticity(state.u, state.v, self.length)))),
            "t": float(state.t),
        }


def project(state: FluidState, length: float = 2.0 * np.pi) -> FluidState:
    u, v = project_divergence_free(state.u, state.v, length)
    return FluidState(u, v, state.t)
