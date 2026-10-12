"""One place that answers "what hardware am I actually running on."

``device.py`` answers "which device, and is bf16 available" for the model's own
numerics — that module is the source of truth for the router's dtype decision
and stays that way. This module answers the questions *around* it that the rest
of the codebase (serving, Docker healthchecks, a user's own sanity check) needs
and would otherwise reinvent per-caller, each slightly differently: is this
actually HIP or CUDA under the ``torch.cuda`` name, what silicon, does
``torch.compile`` have a working backend here, which SDPA kernels will the
scheduler actually pick between, what is the TF32-equivalent knob, and which
environment variables change any of that. ``capabilities()`` is the one
function meant to be called from outside this package; everything else is
detail feeding it.

**Why this is not just more code in ``device.py``.** ``device.py``'s contract
is "name the backend and the model dtype, correctly, with no side effects on
import." Adding `torch.compile` probing, SDPA backend enumeration and an
environment-variable audit there would make importing it slower and would mix
"what dtype should the model use" with "what should an operator print on a
dashboard" — two different callers with different tolerance for extra work at
import time. Keeping them separate means ``device.py`` stays cheap enough to
call from a hot path (it already is, `generate()` calls `device_of()` every
step) while this module can afford to actually *try* things.

**The ROCm-specific traps this exists to name, not silently work around:**

* ``torch.cuda.get_device_capability()`` returns HIP's notion of compute
  capability on a ROCm build, which is *not* an NVIDIA SM number and is not
  guaranteed comparable across generations the way `(major, minor) >= (8, 0)`
  assumes for CUDA. Code that reads it as an SM number gate (grep the codebase
  for `>= 8` — `iridium/runtime/placement.py:native_bf16` does exactly this) is
  wrong on AMD by construction, not by omission. This module never uses that
  API to decide bf16; it defers to ``device.py``'s architecture-string match,
  which is correct for both vendors, and reports the raw capability tuple
  separately, labelled as vendor-specific and not a support signal.
* ``torch.cuda.is_bf16_supported()`` on a HIP build asks the *current* device,
  so it is one card's answer, not the vendor's. Fine for a single-process
  server; a multi-card fleet needs the per-device architecture check
  ``device.py`` does, not this flag alone.
* TF32 is an NVIDIA Ampere+ tensor-core mode; there is no equivalent knob to
  set on ROCm because AMD's matmul precision is controlled differently (MIOpen
  find-mode, not a global fp32-matmul toggle), so `matmul_precision` here is
  reported as "n/a (ROCm)" rather than silently mapped onto the CUDA setting,
  which would look like it did something and would not.
* `HIP_VISIBLE_DEVICES` is ROCm's device mask; `CUDA_VISIBLE_DEVICES` is also
  honored by ROCm builds of PyTorch for CUDA-script compatibility, but if both
  are set and disagree, behavior is build-dependent. This module reports both
  rather than picking one, so a misconfigured launcher is visible instead of
  silently using whichever one PyTorch happened to prefer.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .device import DeviceInfo, detect

#: Environment variables that change ROCm/CUDA behavior in ways worth seeing
#: together, rather than checked one at a time scattered across launch scripts.
ENV_VARS = (
    "HSA_OVERRIDE_GFX_VERSION",  # force an unsupported consumer arch to report as a supported one
    "PYTORCH_ROCM_ARCH",         # which gfx targets a source build compiles kernels for
    "MIOPEN_FIND_MODE",          # MIOpen's autotune vs. cached-heuristic tradeoff (ROCm's cuDNN)
    "MIOPEN_USER_DB_PATH",       # where MIOpen caches the autotune results FIND_MODE produces
    "HIP_VISIBLE_DEVICES",       # ROCm's device mask
    "CUDA_VISIBLE_DEVICES",      # also honored by ROCm PyTorch builds, for script compatibility
    "PYTORCH_CUDA_ALLOC_CONF",   # same allocator env var on both backends (name is a CUDA-ism)
    "PYTORCH_HIP_ALLOC_CONF",    # the HIP-named alias some ROCm builds prefer; either can apply
    "ROCR_VISIBLE_DEVICES",      # ROCr runtime's own device mask, older / lower-level than HIP_VISIBLE_DEVICES
    "NCCL_SOCKET_IFNAME",        # multi-GPU: RCCL reads the same NCCL_* variable names, no RCCL_* prefix
)


def is_hip() -> bool:
    """True on a ROCm PyTorch build, regardless of whether a GPU is present.

    Distinct from "a ROCm GPU is available now" — this is about which *wheel*
    is installed, which is what decides whether RCCL vs NCCL is in play, and
    it is knowable with zero devices attached (CI, this container).
    """
    return getattr(torch.version, "hip", None) is not None


def sdpa_backends(device_type: str = "cuda") -> dict[str, bool]:
    """Which SDPA kernels PyTorch will actually consider on this device type.

    FlashAttention's upstream package (``flash_attn`` on PyPI) is CUDA-only and
    is not imported anywhere in this codebase (checked: no import of
    ``flash_attn`` exists here) — attention goes exclusively through
    ``torch.nn.functional.scaled_dot_product_attention`` in
    ``iridium/model/layers.py``, which is the right call for portability, since
    SDPA dispatches to *whatever fused kernel the backend provides* rather than
    requiring a specific one. On ROCm that means PyTorch's own composable-kernel
    (CK) flash-attention backend when the build includes it, math fallback
    otherwise; on CPU, math only. This reports which of SDPA's own backend
    flags are enabled, not which kernel a given call will pick — that is
    shape- and dtype-dependent and only known by running it.
    """
    out = {"math": bool(torch.backends.cuda.math_sdp_enabled())
           if hasattr(torch.backends, "cuda") else True}
    if hasattr(torch.backends, "cuda"):
        out["flash"] = bool(torch.backends.cuda.flash_sdp_enabled())
        out["efficient"] = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
        out["cudnn"] = bool(getattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: False)())
    return out


def compile_available() -> tuple[bool, str]:
    """Whether ``torch.compile`` has a usable backend here, found out by trying.

    Inductor works on ROCm — it compiles through the same Triton-based codegen
    — but Triton's ROCm support lags CUDA's and varies by PyTorch/ROCm version
    pairing, and a missing or mismatched Triton produces a compile-time failure
    the first time a decorated function is actually called, not at import.
    Asserting "inductor works on ROCm" without running anything would be
    exactly the kind of unverified claim this audit exists to avoid, so this
    compiles and runs one trivial function and reports what happened.
    """
    try:
        @torch.compile(fullgraph=True)
        def _f(x: torch.Tensor) -> torch.Tensor:
            return x * 2 + 1

        out = _f(torch.ones(4))
        if not torch.allclose(out, torch.full((4,), 3.0)):
            return False, "compiled function returned wrong values"
        return True, "ok"
    except Exception as exc:                          # pragma: no cover - env-dependent
        return False, f"{type(exc).__name__}: {exc}"


def matmul_precision(info: DeviceInfo) -> str:
    """The TF32-equivalent knob for this backend, or the honest absence of one.

    ``torch.backends.cuda.matmul.allow_tf32`` and
    ``torch.set_float32_matmul_precision`` are real, settable knobs on a ROCm
    build too (they are generic PyTorch API, not CUDA-only despite the name),
    but they do not do anything on AMD hardware the way they do on Ampere+:
    there is no TF32 tensor-core mode to opt into. Reporting a value here
    without saying that would read as "this is tuned," when the honest
    statement is "this knob is a no-op on this vendor."
    """
    if info.backend == "cuda":
        return "tf32 available (allow_tf32 / set_float32_matmul_precision)"
    if info.backend == "rocm":
        return "n/a on ROCm — no TF32 tensor-core mode; fp32 matmul is fp32"
    return "n/a"


def capabilities(prefer: Optional[str] = None) -> dict:
    """Everything the rest of the codebase, or a human, needs to know at once.

    Deliberately a plain dict of JSON-safe values (no tensors, no dataclasses
    that need a custom encoder) because its two call sites are a health-check
    HTTP response and a printed report — both want to serialize or print it
    without ceremony.
    """
    info = detect(prefer)
    out: dict[str, object] = {
        "backend": info.backend,
        "device": info.device,
        "name": info.name,
        "bf16": info.bf16,
        "dtype": "bfloat16" if info.bf16 else "float32",
        "total_memory_gb": info.total_memory_gb,
        "detail": info.detail,
        "is_hip_build": is_hip(),
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "matmul_precision": matmul_precision(info),
        "env": {name: os.environ.get(name) for name in ENV_VARS
                if os.environ.get(name) is not None},
    }
    if info.device == "cuda":
        try:
            cap = torch.cuda.get_device_capability(0)
            out["device_capability_raw"] = list(cap)
            out["device_capability_note"] = (
                "HIP-reported tuple, not an NVIDIA SM number — do not gate features "
                "on it for ROCm" if info.backend == "rocm" else
                f"sm_{cap[0]}{cap[1]}"
            )
        except Exception as exc:                       # pragma: no cover
            out["device_capability_raw"] = f"unavailable: {exc}"
        out["sdpa_backends"] = sdpa_backends()
        ok, detail = compile_available()
        out["torch_compile"] = {"available": ok, "detail": detail}
    else:
        out["sdpa_backends"] = {"math": True}
        out["torch_compile"] = {"available": False, "detail": "no CUDA/HIP device"}
    return out


def report() -> str:
    """The printed form of ``capabilities()``, for a build-time or shell check."""
    import json
    return json.dumps(capabilities(), indent=2, default=str)


if __name__ == "__main__":                              # pragma: no cover
    print(report())
