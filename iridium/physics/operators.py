"""Spectral differential operators and field diagnostics.

Everything here is float64 and deterministic. These are the quantities the
verifier compares a learned prediction against, so if they were approximate the
whole evidence chain would be.
"""

from __future__ import annotations

import numpy as np


def wavenumbers(n: int, length: float = 2.0 * np.pi) -> np.ndarray:
    return 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)


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
    k2[0, 0] = 1.0
    uh, vh = np.fft.fft2(u), np.fft.fft2(v)
    dot = kx * uh + ky * vh
    uh -= kx * dot / k2
    vh -= ky * dot / k2
    return np.real(np.fft.ifft2(uh)), np.real(np.fft.ifft2(vh))
