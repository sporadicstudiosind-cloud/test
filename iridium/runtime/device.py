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

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    backend: str            # "rocm" | "cuda" | "cpu" | "mps"
    name: str
    bf16: bool
    total_memory_gb: float
    detail: str

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.bf16 else torch.float32

    def describe(self) -> str:
        mem = f", {self.total_memory_gb:.1f} GB" if self.total_memory_gb else ""
        return (f"{self.backend}: {self.name}{mem}, "
                f"autocast dtype {'bfloat16' if self.bf16 else 'float32'}"
                + (f" ({self.detail})" if self.detail else ""))


def detect(prefer: Optional[str] = None) -> DeviceInfo:
    if prefer == "cpu":
        return _cpu()
    if torch.cuda.is_available():
        is_rocm = torch.version.hip is not None
        props = torch.cuda.get_device_properties(0)
        name = torch.cuda.get_device_name(0)
        memory = props.total_memory / 1e9
        if is_rocm:
            arch = getattr(props, "gcnArchName", "") or ""
            # CDNA (gfx90a/gfx942) and RDNA3 (gfx11xx) carry bf16; older RDNA
            # does not, and fp16 is not a safe substitute for a routed model.
            bf16 = any(t in arch for t in ("gfx90a", "gfx94", "gfx110", "gfx112", "gfx115"))
            if not bf16:
                bf16 = bool(torch.cuda.is_bf16_supported())
            return DeviceInfo("cuda", "rocm", name, bf16, memory,
                              f"HIP {torch.version.hip}, arch {arch or 'unknown'}")
        major = props.major
        return DeviceInfo("cuda", "cuda", name, major >= 8, memory,
                          f"CUDA {torch.version.cuda}, sm_{major}{props.minor}")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # MPS has no bf16 autocast and no complex FFT; the spectral blocks fall
        # back to CPU there, which is slower than just staying on CPU.
        return DeviceInfo("mps", "mps", "Apple GPU", False, 0.0,
                          "spectral blocks are unsupported on MPS")
    return _cpu()


def _cpu() -> DeviceInfo:
    import os
    return DeviceInfo("cpu", "cpu", "CPU", False, 0.0,
                      f"{os.cpu_count()} threads available")


def generator_for(device: str | torch.device, seed: int) -> torch.Generator:
    """A generator on the *same* device as the tensors it will seed.

    ``torch.randn(..., device="cuda", generator=torch.Generator())`` raises,
    and the flow-matching head samples on the target's device. This is the
    single most common way a working CPU training script dies the moment it
    touches a GPU.
    """
    dev = torch.device(device)
    if dev.type == "cuda":
        return torch.Generator(device=dev).manual_seed(seed)
    return torch.Generator().manual_seed(seed)


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
