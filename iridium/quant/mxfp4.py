"""Microscaling FP4 (MXFP4) and FP8, implemented rather than cited.

OCP Microscaling Formats (MX) v1.0: a block of **32** elements shares one
**E8M0** scale — 8 exponent bits, no sign, no mantissa, a pure power of two —
and each element is **E2M1**: 1 sign bit, 2 exponent bits, 1 mantissa bit.

**The correction this module exists to make concrete.** The source plan costed
MXFP4 at 0.5 bytes per parameter and derived 4.39 TB for the 8.79 T
configuration. The shared scale is not free: 8 bits per 32 elements is 0.25
bits per element, so the real cost is **4.25 bits per parameter**, 6.25% more.
``bits_per_param`` computes it from the block size rather than asserting it,
and ``tests/unit/test_quant.py`` checks the packed size against the formula.

Everything here is a faithful *simulation* of the numerics — values are
quantized to exactly the representable set and stored unpacked. That is
deliberate: the purpose is to measure the error a 4-bit weight path introduces,
which no amount of bit-packing changes, and a packed representation that no
kernel consumes would only obscure the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch

#: E2M1 magnitudes: exponent 2 bits (bias 1), mantissa 1 bit, plus zero.
E2M1_VALUES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0
E2M1_MAX_EXP = 2           # largest binary exponent an E2M1 element can carry
E8M0_BIAS = 127
E8M0_MIN_EXP, E8M0_MAX_EXP = -127, 127

#: E4M3 (OCP FP8) magnitudes cap.
E4M3_MAX = 448.0
E4M3_MIN_NORMAL = 2.0 ** -6


def bits_per_param(block_size: int = 32, element_bits: int = 4,
                   scale_bits: int = 8) -> float:
    """Storage cost per element *including* the shared scale."""
    if block_size <= 0:
        raise ValueError("block size must be positive")
    return element_bits + scale_bits / block_size


@dataclass(frozen=True)
class QuantStats:
    sqnr_db: float
    max_abs_error: float
    relative_l2: float
    bits_per_param: float
    n_blocks: int

    def as_dict(self) -> dict[str, float]:
        return {
            "sqnr_db": self.sqnr_db,
            "max_abs_error": self.max_abs_error,
            "relative_l2": self.relative_l2,
            "bits_per_param": self.bits_per_param,
            "n_blocks": float(self.n_blocks),
        }


def _round_to_set(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Round each magnitude to the nearest representable value, ties to even.

    Implemented as a midpoint search rather than a nearest-neighbour scan so
    that the tie rule is explicit: MX rounding is round-half-to-even, and a
    naive ``argmin(|x - v|)`` silently rounds halves away from zero, which
    biases every block whose values sit on a midpoint (a common case for
    weights initialized on a symmetric grid).
    """
    mid = (values[:-1] + values[1:]) / 2.0
    idx = torch.bucketize(x, mid)
    lower = values[idx.clamp_max(len(values) - 1)]
    # Resolve exact midpoints toward the even-indexed representable value.
    on_mid = torch.isin(x, mid)
    if bool(on_mid.any()):
        upper_idx = (idx + 1).clamp_max(len(values) - 1)
        pick_upper = (upper_idx % 2 == 0)
        lower = torch.where(on_mid & pick_upper, values[upper_idx], lower)
    return lower


def quantize_mxfp4(
    tensor: torch.Tensor, block_size: int = 32, axis: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(dequantized, shared_exponents)`` for an MXFP4 round trip."""
    if block_size <= 0:
        raise ValueError("block size must be positive")
    x = tensor.detach().to(torch.float32).movedim(axis, -1)
    shape = x.shape
    n = shape[-1]
    pad = (-n) % block_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    blocks = x.reshape(-1, block_size)

    amax = blocks.abs().amax(dim=-1)
    # Shared exponent: place the block maximum at the top of the E2M1 range.
    exp = torch.where(
        amax > 0,
        torch.floor(torch.log2(amax.clamp_min(1e-38))) - E2M1_MAX_EXP,
        torch.zeros_like(amax),
    ).clamp(E8M0_MIN_EXP, E8M0_MAX_EXP)
    scale = torch.pow(torch.tensor(2.0), exp).unsqueeze(-1)

    values = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    scaled = (blocks / scale).abs().clamp(max=E2M1_MAX)
    q = _round_to_set(scaled, values) * torch.sign(blocks)
    out = (q * scale).reshape(shape)
    if pad:
        out = out[..., :n]
    return out.movedim(-1, axis).to(tensor.dtype), exp


def quantize_fp8_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    """Round to the E4M3 grid: 3 mantissa bits, exponents down to 2^-6."""
    x = tensor.detach().to(torch.float32)
    sign = torch.sign(x)
    mag = x.abs().clamp(max=E4M3_MAX)
    exp = torch.floor(torch.log2(mag.clamp_min(E4M3_MIN_NORMAL / 8)))
    exp = exp.clamp_min(-9)
    step = torch.pow(torch.tensor(2.0), exp - 3)        # 3 mantissa bits
    q = torch.round(mag / step) * step
    return (sign * q).to(tensor.dtype)


def measure(
    tensor: torch.Tensor, block_size: int = 32, mode: str = "mxfp4"
) -> QuantStats:
    original = tensor.detach().to(torch.float32)
    if mode == "mxfp4":
        recovered, exp = quantize_mxfp4(tensor, block_size)
        n_blocks = int(exp.numel())
        bits = bits_per_param(block_size)
    elif mode == "fp8":
        recovered = quantize_fp8_e4m3(tensor)
        n_blocks = 0
        bits = 8.0
    elif mode == "bf16":
        recovered = tensor.detach().to(torch.bfloat16).to(torch.float32)
        n_blocks = 0
        bits = 16.0
    else:
        raise ValueError(f"unknown mode {mode!r}")
    recovered = recovered.to(torch.float32)
    err = recovered - original
    signal = float(original.pow(2).mean())
    noise = float(err.pow(2).mean()) + 1e-30
    return QuantStats(
        sqnr_db=10.0 * float(np.log10(max(signal, 1e-30) / noise)),
        max_abs_error=float(err.abs().max()),
        relative_l2=float(err.pow(2).sum().sqrt() / original.pow(2).sum().sqrt().clamp_min(1e-30)),
        bits_per_param=bits,
        n_blocks=n_blocks,
    )


def quantize_module(
    module: torch.nn.Module,
    block_size: int = 32,
    skip: Iterable[str] = ("norm", "embedding", "halt", "gate", "focus"),
    mode: str = "mxfp4",
) -> dict[str, float]:
    """Quantize weight matrices in place; report what was and was not touched.

    Norms, embeddings and the routing heads are left alone. They are a
    negligible fraction of the parameters and a large fraction of the damage:
    a 4-bit router gate changes *which* stack a token visits, which is a
    discrete decision that does not degrade gracefully.
    """
    skip = tuple(skip)
    touched = 0
    left = 0
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param.dim() < 2 or any(s in name.lower() for s in skip):
                left += param.numel()
                continue
            if mode == "mxfp4":
                param.copy_(quantize_mxfp4(param, block_size)[0])
            elif mode == "fp8":
                param.copy_(quantize_fp8_e4m3(param))
            else:
                raise ValueError(f"unknown mode {mode!r}")
            touched += param.numel()
    total = touched + left
    return {
        "quantized_params": touched,
        "kept_params": left,
        "fraction_quantized": touched / max(total, 1),
        "effective_bits_per_param": (
            touched * bits_per_param(block_size) + left * 16.0
        ) / max(total, 1),
    }
