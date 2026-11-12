"""One-dimensional shallow-water flow: the waterfall scenario, made checkable.

    d(h)/dt    + d(hu)/dx           = 0
    d(hu)/dt   + d(hu^2 + g h^2/2)/dx = -g h dz/dx - g n^2 u|u| / h^(1/3)

Finite volume, Rusanov (local Lax-Friedrichs) flux, explicit in time with a
semi-implicit friction update so that friction cannot reverse the flow at small
depth. Bed slope and Manning friction are real source terms, not decoration.

Why this file exists
--------------------
The originating request was: *given a simulation of fluid going down a
waterfall, what happens if I double the input water?* That question has an
exact answer for steady open-channel flow, and two different exact answers
depending on which depth you mean:

* **normal depth** (uniform flow, friction balancing gravity), from Manning:
  ``h_n = (q n / sqrt(S0))^(3/5)``, so doubling ``q`` multiplies it by
  ``2^(3/5) = 1.5157``;
* **critical depth** at the brink, ``h_c = (q^2/g)^(1/3)``, so doubling ``q``
  multiplies it by ``2^(2/3) = 1.5874``.

Neither is ``2``. A system that answers "the water doubles" is wrong, and a
system that answers with one of these without saying which is unfalsifiable.
The solver reaches both regimes and ``tests/scientific/test_shallow_water.py``
checks the solver against the analytic laws, not against itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

G = 9.80665           # standard gravity, m/s^2
DRY = 1e-8            # depth below which a cell is treated as dry


@dataclass(frozen=True)
class Channel:
    """A rectangular prismatic channel."""

    length: float = 100.0      # m
    n_cells: int = 400
    slope: float = 0.002       # S0, dimensionless (bed drop per unit length)
    manning: float = 0.030     # n, s/m^(1/3)
    brink_at: Optional[float] = None   # x of a free overfall, or None

    @property
    def dx(self) -> float:
        return self.length / self.n_cells

    @property
    def x(self) -> np.ndarray:
        return (np.arange(self.n_cells) + 0.5) * self.dx

    @property
    def bed(self) -> np.ndarray:
        """Bed elevation. Drops linearly; a brink adds a sharp fall."""
        z = (self.length - self.x) * self.slope
        if self.brink_at is not None:
            past = self.x > self.brink_at
            z = z - past * 5.0 * np.tanh((self.x - self.brink_at) / (2 * self.dx))
        return z


def normal_depth(q: float, slope: float, manning: float) -> float:
    """Manning uniform-flow depth for a wide rectangular channel, per unit width."""
    if slope <= 0:
        raise ValueError("normal depth is undefined for a non-positive slope")
    return float((q * manning / np.sqrt(slope)) ** 0.6)


def critical_depth(q: float, g: float = G) -> float:
    """``h_c = (q^2/g)^(1/3)``, the depth at which the Froude number is 1."""
    return float((q * q / g) ** (1.0 / 3.0))


def froude(h: np.ndarray, u: np.ndarray, g: float = G) -> np.ndarray:
    return np.abs(u) / np.sqrt(np.maximum(g * h, 1e-30))


@dataclass
class ChannelState:
    h: np.ndarray
    hu: np.ndarray
    t: float = 0.0

    @property
    def u(self) -> np.ndarray:
        return np.where(self.h > DRY, self.hu / np.maximum(self.h, DRY), 0.0)

    @property
    def discharge(self) -> np.ndarray:
        return self.hu

    def copy(self) -> "ChannelState":
        return ChannelState(self.h.copy(), self.hu.copy(), self.t)


class ShallowWater1D:
    def __init__(self, channel: Channel, g: float = G) -> None:
        self.ch = channel
        self.g = g
        self._bed = channel.bed
        self._dzdx = np.gradient(self._bed, channel.dx)

    # -- initial conditions ----------------------------------------------

    def uniform_state(self, q: float, depth: Optional[float] = None) -> ChannelState:
        """Uniform initial depth. Defaults to normal depth.

        Tests that want to show the solver *finds* the equilibrium must pass a
        ``depth`` away from it; starting at the answer only demonstrates that
        the equilibrium is a fixed point, which is a much weaker claim.
        """
        h0 = depth if depth is not None else normal_depth(q, self.ch.slope, self.ch.manning)
        h = np.full(self.ch.n_cells, float(h0))
        return ChannelState(h, np.full(self.ch.n_cells, q))

    # -- numerics ---------------------------------------------------------

    def _flux(self, h: np.ndarray, hu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.where(h > DRY, hu / np.maximum(h, DRY), 0.0)
        return hu, hu * u + 0.5 * self.g * h * h

    def _rusanov(self, hL, huL, hR, huR):
        fL = self._flux(hL, huL)
        fR = self._flux(hR, huR)
        uL = np.where(hL > DRY, huL / np.maximum(hL, DRY), 0.0)
        uR = np.where(hR > DRY, huR / np.maximum(hR, DRY), 0.0)
        a = np.maximum(
            np.abs(uL) + np.sqrt(self.g * np.maximum(hL, 0.0)),
            np.abs(uR) + np.sqrt(self.g * np.maximum(hR, 0.0)),
        )
        f0 = 0.5 * (fL[0] + fR[0]) - 0.5 * a * (hR - hL)
        f1 = 0.5 * (fL[1] + fR[1]) - 0.5 * a * (huR - huL)
        return f0, f1, a

    def max_wave_speed(self, state: ChannelState) -> float:
        return float(
            np.max(np.abs(state.u) + np.sqrt(self.g * np.maximum(state.h, 0.0)))
        )

    def cfl_dt(self, state: ChannelState, cfl: float = 0.45) -> float:
        return cfl * self.ch.dx / max(self.max_wave_speed(state), 1e-12)

    def step(self, state: ChannelState, dt: float, q_in: float) -> ChannelState:
        h, hu = state.h, state.hu
        dx = self.ch.dx

        # Ghost cells: specified discharge upstream, transmissive downstream.
        hL = np.concatenate([[h[0]], h])
        huL = np.concatenate([[q_in], hu])
        hR = np.concatenate([h, [h[-1]]])
        huR = np.concatenate([hu, [hu[-1]]])
        f0, f1, _ = self._rusanov(hL[:-1], huL[:-1], hL[1:], huL[1:])
        g0, g1, _ = self._rusanov(hR[:-1], huR[:-1], hR[1:], huR[1:])
        # Interior faces are shared; assemble the full face vector once.
        face_h = np.empty(self.ch.n_cells + 1)
        face_hu = np.empty(self.ch.n_cells + 1)
        face_h[:-1], face_hu[:-1] = f0, f1
        face_h[-1], face_hu[-1] = g0[-1], g1[-1]
        face_h[0] = q_in                       # exact specified inflow
        face_hu[0] = q_in * q_in / max(h[0], DRY) + 0.5 * self.g * h[0] ** 2

        h_new = h - dt * (face_h[1:] - face_h[:-1]) / dx
        hu_new = hu - dt * (face_hu[1:] - face_hu[:-1]) / dx
        h_new = np.maximum(h_new, 0.0)

        # Bed slope source (non-well-balanced form; adequate for mild slopes).
        hu_new = hu_new - dt * self.g * h_new * self._dzdx

        # Manning friction, semi-implicit: hu <- hu / (1 + dt * k * |u| / h^(4/3))
        wet = h_new > DRY
        u = np.where(wet, hu_new / np.maximum(h_new, DRY), 0.0)
        k = self.g * self.ch.manning ** 2 * np.abs(u) / np.maximum(h_new, DRY) ** (4.0 / 3.0)
        hu_new = np.where(wet, hu_new / (1.0 + dt * k), 0.0)

        return ChannelState(h_new, hu_new, state.t + dt)

    # -- driving ----------------------------------------------------------

    def run_to_steady(
        self,
        q_in: float,
        state: Optional[ChannelState] = None,
        max_time: float = 400.0,
        tol: float = 1e-7,
        record_every: float = 0.0,
        initial_depth: Optional[float] = None,
    ) -> tuple[ChannelState, list[ChannelState], dict]:
        """Advance until the depth field stops changing, or ``max_time``."""
        state = state or self.uniform_state(q_in, initial_depth)
        frames: list[ChannelState] = []
        next_record = record_every
        t = 0.0
        residual = float("inf")
        steps = 0
        h_scale = max(float(np.mean(state.h)), 1e-6)
        q_scale = max(abs(q_in), 1e-6)
        while t < max_time:
            dt = self.cfl_dt(state)
            prev_h, prev_hu = state.h, state.hu
            state = self.step(state, dt, q_in)
            t = state.t
            steps += 1
            # The residual must watch *both* conserved variables. A uniform
            # initial depth has zero depth tendency at the first step while its
            # momentum is still far from balance, so a depth-only criterion
            # declares convergence immediately and returns the initial
            # condition as the answer. That is how this was found.
            residual = max(
                float(np.max(np.abs(state.h - prev_h))) / (dt * h_scale),
                float(np.max(np.abs(state.hu - prev_hu))) / (dt * q_scale),
            )
            if record_every and t >= next_record:
                frames.append(state.copy())
                next_record += record_every
            if residual < tol and steps > 1:
                break
        info = {
            "steps": steps,
            "time": t,
            "residual": residual,
            "converged": residual < tol,
        }
        return state, frames, info

    # -- diagnostics ------------------------------------------------------

    def diagnostics(self, state: ChannelState, q_in: float) -> dict[str, float]:
        fr = froude(state.h, state.u, self.g)
        interior = slice(self.ch.n_cells // 10, -self.ch.n_cells // 10)
        return {
            "q_in": float(q_in),
            "h_upstream": float(np.mean(state.h[interior][:5])),
            "h_normal_analytic": normal_depth(q_in, self.ch.slope, self.ch.manning),
            "h_critical_analytic": critical_depth(q_in, self.g),
            "h_outlet": float(state.h[-1]),
            "froude_upstream": float(np.mean(fr[interior][:5])),
            "froude_max": float(np.max(fr)),
            "discharge_error": float(
                np.max(np.abs(state.hu[interior] - q_in)) / max(q_in, 1e-12)
            ),
            "mass_flux_in": float(q_in),
            "mass_flux_out": float(state.hu[-1]),
        }


def intervention(
    channel: Channel, q_before: float, q_after: float, **kwargs
) -> dict[str, object]:
    """Run the *same* channel at two discharges and report what actually changed.

    This is the executable form of the originating question. It returns the
    measured ratios alongside the analytic ones so that a report generated from
    it can be checked rather than believed.
    """
    solver = ShallowWater1D(channel)
    before, _, info_b = solver.run_to_steady(q_before, **kwargs)
    after, _, info_a = solver.run_to_steady(q_after, **kwargs)
    d_before = solver.diagnostics(before, q_before)
    d_after = solver.diagnostics(after, q_after)
    ratio = q_after / q_before
    return {
        "before": d_before,
        "after": d_after,
        "info_before": info_b,
        "info_after": info_a,
        "discharge_ratio": ratio,
        "depth_ratio_measured": d_after["h_upstream"] / d_before["h_upstream"],
        "depth_ratio_normal_law": ratio ** 0.6,
        "depth_ratio_critical_law": ratio ** (2.0 / 3.0),
        "velocity_ratio_measured": (
            (d_after["q_in"] / d_after["h_upstream"])
            / (d_before["q_in"] / d_before["h_upstream"])
        ),
    }
