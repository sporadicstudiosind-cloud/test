"""Spectral differential operators and field diagnostics.

Everything here is float64 and deterministic. These are the quantities the
verifier compares a learned prediction against, so if they were approximate the
whole evidence chain would be.
"""

from __future__ import annotations

import numpy as np


def wavenumbers(
    n: int, length: float = 2.0 * np.pi, zero_nyquist: bool = True
) -> np.ndarray:
    """Angular wavenumbers, with the Nyquist mode zeroed by default.

    **Why the Nyquist mode must be zeroed for odd-order operators.** On an even
    grid, ``fftfreq`` assigns ``-n/2`` to the bin at the Nyquist frequency, but
    that bin is its own conjugate partner: ``+n/2`` and ``-n/2`` are the same
    coefficient. A multiplier that is *odd* in ``k`` - which every first
    derivative is - therefore cannot be Hermitian there, whichever sign is
    chosen. The transform of a real field then comes back with a large
    imaginary part, and ``np.real(...)`` silently discards it.

    The damage is not subtle. Leray projection is exact in spectral space
    (residual 2e-13) and then loses it entirely on the way back to physical
    space: measured divergence of a projected random field went from 2e-13 to
    443 purely through that discarded component, and the projection stopped
    being idempotent. Taylor-Green hides it completely, because a two-mode
    analytic flow has no Nyquist content - which is exactly why validating
    only against smooth analytic solutions is not enough.

    Zeroing the Nyquist multiplier is the standard treatment: the first
    derivative of a real field at that mode is not representable on the grid,
    so the honest value is zero rather than an arbitrary sign.
    """
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    if zero_nyquist and n % 2 == 0:
        k[n // 2] = 0.0
    return k


def grid(n: int, length: float = 2.0 * np.pi) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, length, n, endpoint=False)
    return np.meshgrid(x, x, indexing="ij")


def ddx(f: np.ndarray, length: float = 2.0 * np.pi) -> np.ndarray:
    kx = wavenumbers(f.shape[0], length).reshape(-1, 1)
    return np.real(np.fft.ifft2(1j * kx * np.fft.fft2(f)))


def ddy(f: np.ndarray, length: float = 2.0 * np.pi) -> np.ndarray:
    ky = wavenumbers(f.shape[1], length).reshape(1, -1)
    return np.real(np.fft.ifft2(1j * ky * np.fft.fft2(f)))


def divergence(u: np.ndarray, v: np.ndarray, length: float = 2.0 * np.pi) -> np.ndarray:
    return ddx(u, length) + ddy(v, length)


def vorticity(u: np.ndarray, v: np.ndarray, length: float = 2.0 * np.pi) -> np.ndarray:
    return ddx(v, length) - ddy(u, length)


def kinetic_energy(u: np.ndarray, v: np.ndarray) -> float:
    return float(0.5 * np.mean(u ** 2 + v ** 2))


def enstrophy(u: np.ndarray, v: np.ndarray, length: float = 2.0 * np.pi) -> float:
    w = vorticity(u, v, length)
    return float(0.5 * np.mean(w ** 2))


def divergence_norm(u: np.ndarray, v: np.ndarray, length: float = 2.0 * np.pi) -> float:
    """``||div u||_2`` normalized by the velocity scale.

    The absolute divergence of a fast flow on a fine grid is numerically larger
    than that of a slow flow on a coarse one without being any less
    incompressible. A bare threshold on ``||div u||`` — as the source plan
    specified, at ``1e-4`` — is therefore not a scale-free criterion, and the
    same model would pass or fail it depending on the units the user chose.
    """
    d = divergence(u, v, length)
    scale = np.sqrt(np.mean(u ** 2 + v ** 2)) + 1e-30
    return float(np.sqrt(np.mean(d ** 2)) * length / scale)


def project_divergence_free(
    u: np.ndarray, v: np.ndarray, length: float = 2.0 * np.pi
) -> tuple[np.ndarray, np.ndarray]:
    """Leray projection: remove the curl-free part in Fourier space.

    ``u_hat <- u_hat - k (k . u_hat) / |k|^2``. Exact to round-off, which is
    what makes it usable as a hard constraint rather than a penalty.
    """
    n = u.shape[0]
    kx = wavenumbers(n, length).reshape(-1, 1)
    ky = wavenumbers(u.shape[1], length).reshape(1, -1)
    k2 = kx ** 2 + ky ** 2
    # Zeroing the Nyquist multiplier makes k2 vanish on the whole Nyquist row
    # and column, not only at the mean mode. Guarding [0, 0] alone leaves
    # divisions by zero on an even grid.
    k2 = np.where(k2 == 0.0, 1.0, k2)
    uh, vh = np.fft.fft2(u), np.fft.fft2(v)
    dot = kx * uh + ky * vh
    uh -= kx * dot / k2
    vh -= ky * dot / k2
    return np.real(np.fft.ifft2(uh)), np.real(np.fft.ifft2(vh))
