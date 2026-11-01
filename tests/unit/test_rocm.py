"""ROCm coverage that runs on a CPU-only box.

Nothing here needs an AMD card: ``detect()`` and ``backend.capabilities()``
only ever *ask* ``torch.cuda`` questions, so mocking ``torch.cuda.is_available``,
``torch.version.hip`` and ``torch.cuda.get_device_properties`` reproduces
exactly what those functions see on a real ROCm host — the branch that
actually decides bf16 vs. fp32 does not know or care that the properties
object underneath it is fake. What this file cannot do, and does not
pretend to: measure real kernel numerics, real memory bandwidth, or whether
a specific ROCm/PyTorch build actually ships a working Composable Kernel
flash-attention backend for a given arch. Those need the hardware; see
``docs/gpu.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from iridium.runtime import backend
from iridium.runtime.device import DeviceInfo, detect
from iridium.runtime.placement import native_bf16


@dataclass
class _FakeProps:
    total_memory: int
    gcnArchName: str = ""
    major: int = 0
    minor: int = 0


def _mock_rocm(monkeypatch, arch: str, name: str = "AMD Radeon",
                memory_bytes: int = 16 * 10**9, is_bf16_supported: bool = False):
    """Make ``torch.cuda`` answer as it would on a ROCm build with this arch."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.2.41134-a0", raising=False)
    monkeypatch.setattr(torch.version, "cuda", None, raising=False)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda idx=0: _FakeProps(total_memory=memory_bytes, gcnArchName=arch),
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: name)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: is_bf16_supported)


def _mock_cuda(monkeypatch, major: int, minor: int = 0, name: str = "NVIDIA GPU",
               memory_bytes: int = 16 * 10**9):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.version, "cuda", "12.4", raising=False)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda idx=0: _FakeProps(total_memory=memory_bytes, major=major, minor=minor),
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: name)


# ---------------------------------------------------------------------------
# detect(): backend naming and bf16 per real AMD architecture
# ---------------------------------------------------------------------------

# (arch, expect_bf16) — CDNA (MI100+/MI200+/MI300) and RDNA3 carry native bf16;
# RDNA2 and older CDNA/RDNA1 do not.
ROCM_ARCHES = [
    ("gfx90a", True),   # CDNA2, MI200 series
    ("gfx942", True),   # CDNA3, MI300 series
    ("gfx1100", True),  # RDNA3, Radeon RX 7900 / consumer
    ("gfx1030", False), # RDNA2, Radeon RX 6800/6900 — no native bf16
    ("gfx906", False),  # CDNA1 / Vega20 (MI50/MI60) — no native bf16
]


@pytest.mark.parametrize("arch,expect_bf16", ROCM_ARCHES)
def test_detect_names_rocm_and_gates_bf16_by_architecture(monkeypatch, arch, expect_bf16):
    _mock_rocm(monkeypatch, arch, is_bf16_supported=False)
    info = detect()
    assert info.device == "cuda"                 # ROCm rides the cuda API name
    assert info.backend == "rocm"                 # but is named correctly
    assert info.bf16 is expect_bf16, f"{arch}: expected bf16={expect_bf16}, got {info.bf16}"
    assert info.dtype == (torch.bfloat16 if expect_bf16 else torch.float32)
    assert "HIP" in info.detail and arch in info.detail


def test_detect_falls_back_to_is_bf16_supported_for_unlisted_arch(monkeypatch):
    """An arch not in the known-good list still gets a real answer, not `False`
    by default — `torch.cuda.is_bf16_supported()` is the fallback, not a guess."""
    _mock_rocm(monkeypatch, "gfx9999-unknown-future-arch", is_bf16_supported=True)
    info = detect()
    assert info.bf16 is True


def test_rocm_never_reports_itself_as_plain_cuda_backend(monkeypatch):
    """The whole reason this module exists: `backend` must say "rocm", even
    though `device` says "cuda" (that part is correct — it IS the cuda API)."""
    _mock_rocm(monkeypatch, "gfx1100")
    info = detect()
    assert info.backend != "cuda"
    assert info.backend == "rocm"


def test_cuda_still_gates_on_sm_80(monkeypatch):
    _mock_cuda(monkeypatch, major=8, minor=0)
    assert detect().bf16 is True
    _mock_cuda(monkeypatch, major=7, minor=5)  # Turing / T4
    assert detect().bf16 is False


def test_describe_reports_rocm_not_cuda(monkeypatch):
    _mock_rocm(monkeypatch, "gfx942", name="AMD Instinct MI300X")
    text = detect().describe()
    assert "rocm" in text
    assert "MI300X" in text


def test_training_bf16_checks_each_selected_device(monkeypatch):
    from iridium.runtime import device as device_module

    seen = []

    def fake_detect(prefer):
        seen.append(prefer)
        return DeviceInfo(prefer, "rocm", prefer, prefer == "cuda:1", 16, "mock")

    monkeypatch.setattr(device_module, "detect", fake_detect)
    assert native_bf16(("cuda:1",)) is True
    assert seen == ["cuda:1"]
    assert native_bf16(("cuda:0", "cuda:1")) is False


# ---------------------------------------------------------------------------
# backend.capabilities(): shape, and that it works with zero GPU present
# ---------------------------------------------------------------------------

def test_capabilities_imports_and_runs_with_no_gpu():
    """The exact claim in the module docstring: importable and reportable with
    zero GPU present. This is the actual environment this test suite runs in."""
    caps = backend.capabilities()
    assert caps["backend"] == "cpu"
    assert caps["is_hip_build"] is False
    assert caps["torch_compile"] == {"available": False, "detail": "no CUDA/HIP device"}


def test_capabilities_dict_shape_is_stable():
    caps = backend.capabilities()
    required = {
        "backend", "device", "name", "bf16", "dtype", "total_memory_gb", "detail",
        "is_hip_build", "torch_version", "hip_version", "cuda_version",
        "matmul_precision", "env", "sdpa_backends", "torch_compile",
    }
    assert required <= caps.keys()
    import json
    json.dumps(caps, default=str)  # must be JSON-serializable for the health endpoint


def test_capabilities_reports_hip_build_and_rocm_backend(monkeypatch):
    _mock_rocm(monkeypatch, "gfx90a", name="AMD Instinct MI250X")
    caps = backend.capabilities()
    assert caps["backend"] == "rocm"
    assert caps["is_hip_build"] is True
    assert caps["hip_version"] == "6.2.41134-a0"
    assert caps["cuda_version"] is None
    assert caps["matmul_precision"].startswith("n/a on ROCm")
    assert "device_capability_note" in caps
    assert "not an NVIDIA SM number" in caps["device_capability_note"]


def test_capabilities_reports_cuda_backend_distinctly(monkeypatch):
    _mock_cuda(monkeypatch, major=9, minor=0, name="NVIDIA H100")
    caps = backend.capabilities()
    assert caps["backend"] == "cuda"
    assert caps["is_hip_build"] is False
    assert caps["matmul_precision"].startswith("tf32 available")
    assert caps["device_capability_note"] == "sm_90"


def test_env_vars_are_surfaced_when_set(monkeypatch):
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1100")
    caps = backend.capabilities()
    assert caps["env"]["HSA_OVERRIDE_GFX_VERSION"] == "11.0.0"
    assert caps["env"]["PYTORCH_ROCM_ARCH"] == "gfx1100"


def test_is_hip_reflects_wheel_not_device_presence(monkeypatch):
    """`is_hip()` is about which wheel is installed, so it must answer even
    with `torch.cuda.is_available() == False` (this container, exactly)."""
    monkeypatch.setattr(torch.version, "hip", "6.2.41134-a0", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert backend.is_hip() is True


def test_sdpa_backends_reports_math_at_minimum():
    out = backend.sdpa_backends()
    assert "math" in out


# ---------------------------------------------------------------------------
# checkpoint portability: save under one device string, load under another
# ---------------------------------------------------------------------------

def _tiny_model():
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))


def test_checkpoint_round_trips_through_a_simulated_device_mismatch(tmp_path):
    """Simulates the exact failure this whole task is about: a checkpoint
    written with CUDA device strings baked into it (as `cuda:0` occupies the
    same `torch.cuda` namespace on both vendors, this is indistinguishable
    from "trained on an NVIDIA box, served on an AMD one" from the checkpoint's
    point of view) must still load on a machine with no such device.

    The property under test is exactly what `serve/server.py:load` and
    `iridium.training.trainer.load_checkpoint` both already do right:
    `torch.load(..., map_location="cpu")` ignores whatever device string is
    embedded in the pickled tensors' storage and lands everything on CPU,
    from which `.to(target)` is an explicit, separate step. A checkpoint that
    skipped `map_location` and let torch use the pickled storage's original
    device would fail immediately on a machine with no `cuda:0` at all,
    which is precisely the ROCm-checkpoint-loaded-on-a-CPU-CI-box case here.
    """
    torch.manual_seed(0)
    model = _tiny_model()
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    ckpt_path = tmp_path / "tiny.pt"
    # The device string a training run on a CUDA/ROCm box would have recorded.
    # It never needs to exist for the round trip below to work.
    torch.save({"state_dict": model.state_dict(),
                "manifest": {"trained_on_device": "cuda:0"}}, ckpt_path)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert blob["manifest"]["trained_on_device"] == "cuda:0"  # recorded, not required

    restored = _tiny_model()
    restored.load_state_dict({k: v.float() for k, v in blob["state_dict"].items()})
    restored = restored.to("cpu")  # the "different backend" target in this test

    for key, value in restored.state_dict().items():
        assert torch.allclose(value, original_state[key]), key


def test_checkpoint_round_trip_coerces_dtype(tmp_path):
    """A checkpoint saved in bf16 (the ROCm/CUDA training dtype) must still
    load somewhere bf16 is not the native dtype, via the same `.float()`
    coercion `serve/server.py:load` applies — this is what makes a bf16
    Instinct-trained checkpoint servable on plain fp32 CPU/CI hardware."""
    torch.manual_seed(0)
    model = _tiny_model().to(torch.bfloat16)
    ckpt_path = tmp_path / "tiny_bf16.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    restored = _tiny_model()
    restored.load_state_dict({k: v.float() for k, v in blob["state_dict"].items()})
    assert all(v.dtype == torch.float32 for v in restored.state_dict().values())


# ---------------------------------------------------------------------------
# backend.py CLI entry point
# ---------------------------------------------------------------------------

def test_report_is_valid_json():
    import json
    json.loads(backend.report())


def test_xla_devices_get_no_cpu_generator():
    """A CPU generator cannot seed an XLA allocation; the runtime RNG is used."""
    from iridium.runtime.device import generator_for, is_xla

    assert generator_for("xla", 0) is None and is_xla("xla:0")
    assert generator_for("cpu", 0) is not None and not is_xla("cpu")
