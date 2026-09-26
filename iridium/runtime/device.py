"""Device selection, including AMD GPUs through ROCm.

ROCm needs no separate code path. PyTorch's ROCm build exposes AMD hardware
through the *same* ``torch.cuda`` API via HIP, so ``.to("cuda")`` lands on a
Radeon or Instinct card exactly as it would on an NVIDIA one. The only honest
differences worth handling are these:

* **Telling them apart.** ``torch.version.hip`` is set on a ROCm build and
  ``torch.version.cuda`` on an NVIDIA one. Reporting "cuda" on an AMD card is
  technically true and operationally confusing, so this module names it.
* **bf16.** Available on CDNA (MI100+) and RDNA3 (gfx1100+), absent on older
  RDNA. On NVIDIA it needs compute capability 8.0+. fp16 is *not* an automatic
  fallback here: the router's gating softmax and the attention logits are
  exactly where fp16 overflows, and a collapsed router looks like a bad run
  rather than a numerics failure. When bf16 is unavailable this reports fp32
  and lets the caller decide.
* **Generators are per-device.** ``torch.Generator()`` is a CPU generator and
  ``torch.randn(..., device="cuda", generator=cpu_generator)`` raises. The
  flow-matching head samples noise on the target's device, so the trainer's
  generator has to be created on the same device — this is the one thing that
  breaks a GPU run immediately, on ROCm and CUDA alike.
* **FFT.** The spectral blocks call ``torch.fft.rfftn``; ROCm routes it through
  rocFFT. It works, and ``verify()`` below exercises it rather than assuming.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager, Optional

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    backend: str            # "rocm" | "cuda" | "cpu" | "mps" | "xla"
    name: str
    bf16: bool
    total_memory_gb: float
    detail: str
    precision: str = "fp32"  # fp32 | bf16 | fp16; selected for inference

    @property
    def dtype(self) -> torch.dtype:
        return {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[self.precision]

    def describe(self) -> str:
        mem = f", {self.total_memory_gb:.1f} GB" if self.total_memory_gb else ""
        return (f"{self.backend}: {self.name}{mem}, "
                f"inference dtype {self.precision}"
                + (f" ({self.detail})" if self.detail else ""))


def detect(prefer: Optional[str] = None, precision: Optional[str] = None) -> DeviceInfo:
    """Select a real torch device and a conservative inference precision.

    ``torch.device('rocm')`` is invalid even in a ROCm build: HIP deliberately
    uses ``cuda`` device strings.  Accepting ``rocm`` here is only a friendly
    spelling for an environment variable, not a different PyTorch backend.
    An explicit unavailable accelerator raises with diagnostics instead of
    quietly falling back to CPU and making a GPU deployment look successful.
    """
    preferred = (prefer or "auto").strip().lower()
    if preferred in ("", "auto"):
        preferred = "auto"
    if preferred in ("rocm", "hip"):
        preferred = "cuda"
    if preferred in ("tpu", "xla") or preferred.startswith("xla:"):
        return _xla(precision)
    if preferred == "cpu":
        if precision not in (None, "auto", "fp32", "float32"):
            raise RuntimeError("CPU inference supports fp32 only")
        return _cpu()
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "IRIDIUM_DEVICE requested CUDA/ROCm but torch.cuda.is_available() is false. "
            "Install a matching ROCm PyTorch wheel, expose /dev/kfd and /dev/dri "
            "to the container, then run `python -m iridium.runtime.device`."
        )
    if torch.cuda.is_available() and (preferred in ("auto", "cuda") or preferred.startswith("cuda:")):
        try:
            index = 0 if preferred in ("auto", "cuda") else int(preferred.split(":", 1)[1])
            if index < 0 or (preferred not in ("auto", "cuda")
                             and index >= torch.cuda.device_count()):
                raise ValueError
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid IRIDIUM_DEVICE={prefer!r}; available CUDA/HIP devices: "
                f"0..{torch.cuda.device_count() - 1}"
            ) from exc
        is_rocm = torch.version.hip is not None
        props = torch.cuda.get_device_properties(index)
        name = torch.cuda.get_device_name(index)
        memory = props.total_memory / 1e9
        device_name = "cuda" if index == 0 else f"cuda:{index}"
        if is_rocm:
            arch = getattr(props, "gcnArchName", "") or ""
            known_bf16 = any(tag in arch for tag in
                             ("gfx90a", "gfx94", "gfx110", "gfx112", "gfx115"))
            bf16 = known_bf16 or bool(torch.cuda.is_bf16_supported())
            selected = _select_precision(precision, bf16)
            return DeviceInfo(device_name, "rocm", name, bf16, memory,
                              f"HIP {torch.version.hip}, arch {arch or 'unknown'}", selected)
        major = props.major
        bf16 = major >= 8
        selected = _select_precision(precision, bf16)
        return DeviceInfo(device_name, "cuda", name, bf16, memory,
                          f"CUDA {torch.version.cuda}, sm_{major}{props.minor}", selected)
    if preferred not in ("auto", "mps"):
        raise RuntimeError(f"Unsupported IRIDIUM_DEVICE={prefer!r}; use auto, cpu, cuda[:N], rocm, mps, or xla")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # MPS has no bf16 autocast and no complex FFT; the spectral blocks fall
        # back to CPU there, which is slower than just staying on CPU.
        return DeviceInfo("mps", "mps", "Apple GPU", False, 0.0,
                          "spectral blocks are unsupported on MPS", "fp32")
    return _cpu()


def _cpu() -> DeviceInfo:
    import os
    return DeviceInfo("cpu", "cpu", "CPU", False, 0.0,
                      f"{os.cpu_count()} threads available", "fp32")


def _select_precision(requested: Optional[str], bf16: bool) -> str:
    requested = (requested or "auto").strip().lower()
    aliases = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}
    requested = aliases.get(requested, requested)
    if requested == "auto":
        return "bf16" if bf16 else "fp32"
    if requested not in ("fp32", "bf16", "fp16"):
        raise RuntimeError("IRIDIUM_INFERENCE_DTYPE must be auto, fp32, bf16, or fp16")
    if requested == "bf16" and not bf16:
        raise RuntimeError("bf16 was requested but this PyTorch device reports no bf16 support")
    return requested


def inference_autocast(info: DeviceInfo) -> ContextManager:
    """Return the correct inference autocast context, or a no-op for fp32.

    Keep weights in fp32 when loading a checkpoint.  Autocast supplies the
    reduced precision kernels without permanently converting norms, routers,
    or a user's checkpoint to half precision.
    """
    if info.precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=torch.device(info.device).type, dtype=info.dtype)


def generator_for(device: str | torch.device, seed: int) -> Optional[torch.Generator]:
    """A generator on the *same* device as the tensors it will seed.

    ``torch.randn(..., device="cuda", generator=torch.Generator())`` raises,
    and the flow-matching head samples on the target's device. This is the
    single most common way a working CPU training script dies the moment it
    touches a GPU.

    On XLA (TPU) this returns ``None``: torch_xla draws from its own runtime
    RNG, and a CPU generator cannot seed an XLA allocation. Every sampling call
    in this codebase accepts ``generator=None``; the runtime is seeded with
    ``torch.manual_seed`` / ``xm.set_rng_state`` instead, so a TPU run is
    reproducible per seed but its noise stream is not the same one a CUDA run
    with the same seed would draw.
    """
    dev = torch.device(device)
    if dev.type == "cuda":
        return torch.Generator(device=dev).manual_seed(seed)
    if dev.type == "xla":
        return None
    return torch.Generator().manual_seed(seed)


def _xla(precision: Optional[str]) -> DeviceInfo:
    """A TPU through torch_xla. Only ever explicit: ``auto`` never picks XLA,
    because an XLA device silently recompiles on every new shape and a run
    that did not ask for that should not get it."""
    try:
        import torch_xla.core.xla_model as xm
    except ImportError as exc:
        raise RuntimeError("IRIDIUM_DEVICE=xla needs torch_xla (pip install torch_xla); "
                           "it is preinstalled on Colab/Kaggle TPU runtimes") from exc
    device = str(xm.xla_device())
    chosen = "bf16" if precision in (None, "auto", "bf16", "bfloat16") else precision
    if chosen not in ("bf16", "fp32"):
        raise RuntimeError("XLA supports bf16 or fp32")
    return DeviceInfo(device, "xla", "TPU (torch_xla)", True, 0.0, "", precision=chosen)


def is_xla(device) -> bool:
    return torch.device(device).type == "xla"


def device_of(module: torch.nn.Module) -> torch.device:
    """Where this module's parameters actually live.

    Every path that builds a batch needs this. ``TensorBatch`` defaults to CPU,
    so an evaluation or generation helper that forgets it works perfectly on a
    CPU machine and dies on the first GPU with *"Expected all tensors to be on
    the same device"* — pointing at an embedding lookup, several frames deep,
    nowhere near the line that actually made the CPU tensor. A CPU-only test
    suite cannot catch this, which is exactly why it reaches users.

    Falls back to CPU for a module with no parameters at all.
    """
    for param in module.parameters():
        return param.device
    for buf in module.buffers():
        return buf.device
    return torch.device("cpu")


def verify(info: Optional[DeviceInfo] = None) -> dict:
    """Run the operations this model needs and report which actually work."""
    info = info or detect()
    out: dict[str, object] = {"backend": info.backend, "device": info.device,
                              "name": info.name, "bf16": info.bf16}
    dev = torch.device(info.device)
    try:
        a = torch.randn(64, 64, device=dev)
        out["matmul"] = bool(torch.isfinite(a @ a).all())
    except Exception as exc:
        out["matmul"] = f"failed: {exc}"
    try:
        x = torch.randn(1, 2, 16, 16, device=dev)
        spec = torch.fft.rfftn(x, dim=[-2, -1])
        back = torch.fft.irfftn(spec, s=(16, 16), dim=[-2, -1])
        out["fft"] = bool(torch.allclose(x, back, atol=1e-4))
    except Exception as exc:
        out["fft"] = f"failed: {exc}"
    try:
        g = generator_for(info.device, 0)
        torch.randn(8, device=dev, generator=g)
        out["device_generator"] = True
    except Exception as exc:
        out["device_generator"] = f"failed: {exc}"
    if info.bf16:
        try:
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                out["bf16_autocast"] = bool(
                    torch.isfinite(torch.randn(32, 32, device=dev) @
                                   torch.randn(32, 32, device=dev)).all())
        except Exception as exc:
            out["bf16_autocast"] = f"failed: {exc}"
    return out


if __name__ == "__main__":                              # pragma: no cover
    import json
    info = detect()
    print(info.describe())
    print(json.dumps(verify(info), indent=2, default=str))
