"""Episode generator for the first implementation slice (§20.3).

One exact-state episode binds:

* a symbolic instruction (byte-level text) naming the intervention,
* an exact initial field on a periodic 1-D mesh,
* an exact target field at a horizon, from a reference solution,
* a diagnostic space-time image derived from the exact fields,
* the scalars needed to build a metric table.

The governing equation is linear advection-diffusion

    u_t + c u_x = nu u_xx

on a periodic domain, which has an exact spectral solution. Using the exact
solution as the reference removes discretization error from the comparison, so
a learned model's error is its own and not the reference solver's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_CELLS = 64
DOMAIN_LENGTH = 1.0
DX = DOMAIN_LENGTH / N_CELLS
N_FRAMES = 16          # frames in the diagnostic space-time image
HORIZON = 0.25         # physical seconds from t0 to target

# Training covers this band of wave speeds; the evaluation splits below sit
# inside it (interpolation) and outside it (extrapolation).
TRAIN_SPEED_RANGE = (0.2, 0.8)
EXTRAPOLATION_SPEED_RANGE = (1.0, 1.4)
DIFFUSIVITY_RANGE = (5e-4, 4e-3)


@dataclass(frozen=True)
class Episode:
    instruction: str
    u0: np.ndarray            # (N_CELLS,) exact initial field
    u_target: np.ndarray      # (N_CELLS,) exact field at HORIZON
    diagnostic: np.ndarray    # (N_FRAMES, N_CELLS) exact space-time image
    speed: float
    diffusivity: float
    horizon: float

    @property
    def mass_initial(self) -> float:
        return float(np.sum(self.u0) * DX)

    @property
    def mass_target(self) -> float:
        return float(np.sum(self.u_target) * DX)


def _wavenumbers() -> np.ndarray:
    return 2.0 * np.pi * np.fft.fftfreq(N_CELLS, d=DX)


def exact_solution(u0: np.ndarray, speed: float, diffusivity: float, t: float) -> np.ndarray:
    """Spectral exact solution of periodic advection-diffusion at time ``t``."""
    k = _wavenumbers()
    uhat = np.fft.fft(u0)
    propagator = np.exp(-1j * k * speed * t - diffusivity * k**2 * t)
    return np.real(np.fft.ifft(uhat * propagator))


def random_initial_field(rng: np.random.Generator, n_modes: int = 4) -> np.ndarray:
    """A smooth periodic field with a controlled, resolvable spectrum."""
    x = (np.arange(N_CELLS) + 0.5) * DX
    field = np.zeros(N_CELLS)
    for mode in range(1, n_modes + 1):
        amplitude = rng.normal(scale=1.0 / mode)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        field += amplitude * np.sin(2.0 * np.pi * mode * x + phase)
    # A non-zero mean makes mass conservation a meaningful, testable property:
    # with zero mean, total mass is trivially zero and any scheme "conserves".
    return field + rng.uniform(0.8, 1.2)


def make_episode(
    rng: np.random.Generator,
    *,
    speed_range: tuple[float, float] = TRAIN_SPEED_RANGE,
    horizon: float = HORIZON,
) -> Episode:
    speed = float(rng.uniform(*speed_range))
    diffusivity = float(rng.uniform(*DIFFUSIVITY_RANGE))
    u0 = random_initial_field(rng)

    times = np.linspace(0.0, horizon, N_FRAMES)
    diagnostic = np.stack(
        [exact_solution(u0, speed, diffusivity, t) for t in times]
    )
    u_target = diagnostic[-1].copy()

    instruction = (
        f"advect field for {horizon:.3f} s at speed {speed:.3f} m/s "
        f"with diffusivity {diffusivity:.5f} m2/s; report the final field"
    )
    return Episode(instruction, u0, u_target, diagnostic, speed, diffusivity, horizon)


def make_split(
    n: int,
    seed: int,
    *,
    speed_range: tuple[float, float] = TRAIN_SPEED_RANGE,
) -> list[Episode]:
    rng = np.random.default_rng(seed)
    return [make_episode(rng, speed_range=speed_range) for _ in range(n)]


MAX_INSTRUCTION_BYTES = 96


def encode_instruction(text: str) -> np.ndarray:
    """Byte-level encoding with a reversible padding convention.

    §5.2 uses byte-fallback text. Byte level does not make arithmetic exact; it
    avoids tokenizer-induced fragmentation of the numerals in the instruction.
    """
    raw = text.encode("utf-8")[:MAX_INSTRUCTION_BYTES]
    out = np.zeros(MAX_INSTRUCTION_BYTES, dtype=np.int64)
    out[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return out


def decode_instruction(codes: np.ndarray) -> str:
    trimmed = bytes(int(c) for c in codes if int(c) != 0)
    return trimmed.decode("utf-8", errors="replace")


def persistence_baseline(u0: np.ndarray) -> np.ndarray:
    """The honest floor: predict no change at all."""
    return u0.copy()


def upwind_reference(
    u0: np.ndarray, speed: float, diffusivity: float, t_end: float, cfl: float = 0.4
) -> np.ndarray:
    """First-order upwind + explicit diffusion, for a numerical-solver control.

    This is a *discretized* reference, deliberately separate from the exact
    spectral solution, so the evaluation can distinguish "the learned model beat
    a cheap solver" from "the learned model matched the truth".
    """
    dt_adv = cfl * DX / max(abs(speed), 1e-12)
    dt_dif = 0.4 * DX**2 / max(diffusivity, 1e-12)
    dt = min(dt_adv, dt_dif)
    steps = max(1, int(np.ceil(t_end / dt)))
    dt = t_end / steps

    u = u0.copy()
    for _ in range(steps):
        if speed >= 0:
            flux = speed * np.roll(u, 1)
        else:
            flux = speed * u
        flux_faces = np.concatenate([flux, flux[:1]])
        du_adv = -(flux_faces[1:] - flux_faces[:-1]) / DX
        du_dif = diffusivity * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / DX**2
        u = u + dt * (du_adv + du_dif)
    return u
