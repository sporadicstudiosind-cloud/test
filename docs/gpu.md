# Running and training on a GPU — NVIDIA, AMD (ROCm), and Colab

The model is ordinary PyTorch, so there is no vendor-specific code in it. What
follows is what actually differs in practice, and the two things that break a
working CPU script the moment it touches a GPU.

## Check the device before trusting a run

```bash
python -m iridium.runtime.device
```

It names the backend (`rocm`, `cuda`, `cpu`, `mps`), reports whether bf16 is
really available, and then *exercises* matmul, the FFT the spectral blocks
need, and a device-side generator — rather than assuming any of them.

For everything *around* the model's own numerics — is this a HIP or CUDA
build, what SDPA kernels are enabled, does `torch.compile` actually work here,
which environment variables are set — there is a second, wider report:

```bash
python -m iridium.runtime.backend
```

It also comes back from the running server at `GET /api/health` under the
`"backend"` key, so a deployed instance can be checked without shelling in.

## AMD (ROCm) — what is verified here and what is not

**Be precise about what "supported" means in this document.** Everything below
was checked by reading the code and by running it *without a GPU present* —
this development environment has no AMD hardware, full stop. "Should work" and
"tested" are marked separately below; treat every "should work" as a real but
unverified claim, not a guarantee, and report back what actually happens on
real silicon.

PyTorch's ROCm build exposes AMD hardware through the same `torch.cuda` API via
HIP, so `.to("cuda")` lands on a Radeon or Instinct card. Nothing in this
repository's model or training code is CUDA-specific — attention goes through
`torch.nn.functional.scaled_dot_product_attention`, never a CUDA-only
`flash_attn` import; there is no `bitsandbytes` dependency on the inference
path (training's 8-bit/paged optimizers do reach for it, and that package has
no official ROCm build — see the table below).

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
python -m iridium.runtime.device        # expect backend: rocm
python -m iridium.runtime.backend       # full capability report
PYTHONPATH=. python -m iridium.training.phase1_pretrain --rung nano100m --steps 3000
```

Or containerised, which is usually less painful and is what `Dockerfile.rocm`
pins a coherent ROCm-base/torch-wheel pair for:

```bash
docker build -f Dockerfile.rocm -t iridium-rocm .
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --security-opt seccomp=unconfined \
  -p 8080:8080 iridium-rocm
```

Consumer Radeon cards outside AMD's official support list (RDNA2 `gfx1030`
family, RDNA1 `gfx1010` family) often need `HSA_OVERRIDE_GFX_VERSION` to claim
a supported arch string — see the comment block at the bottom of
`Dockerfile.rocm` for the exact values and why this is a community workaround,
not something AMD supports: it can produce silently wrong kernels on a card
whose ISA doesn't actually match the one it's told to impersonate.

| | status | detail |
|---|---|---|
| **bf16 gating** | should work | present on CDNA (MI100+, `gfx90a`/`gfx942`) and RDNA3 (`gfx1100`+), absent on older RDNA/CDNA1 (`gfx906` and earlier). `device.detect()` matches the arch string per `tests/unit/test_rocm.py`, which exercises `gfx90a`, `gfx942`, `gfx1100`, `gfx1030`, `gfx906` against mocked device properties — verified as *logic*, not against real hardware. |
| **FFT** | should work | the spectral blocks go through rocFFT via `torch.fft`. `verify()` in `device.py` exercises it rather than assuming, but only ever against whatever backend is actually present when you run it — on this box, that's CPU. |
| **`torch.compile` / inductor** | unverified | inductor's codegen path works on ROCm in principle, but goes through Triton, and ROCm's Triton support lags CUDA's and is version-pair-sensitive. `iridium/runtime/backend.py`'s `compile_available()` compiles and runs a trivial function and reports the real result rather than asserting it works — run it on your card before relying on `torch.compile` in a training or serving path. |
| **`bitsandbytes` (8-bit/paged optimizers)** | **known gap, training only** | `iridium/runtime/memory.py`'s `paged_adamw`/`adamw_8bit` optimizer kinds `import bitsandbytes`, which has no official ROCm build. On a ROCm box, requesting either raises `ImportError` inside the existing "install bitsandbytes or choose eager_adamw" `RuntimeError` — that's the correct failure (explicit, not silently falling back to a different memory budget), but the *fix* is to pick `eager_adamw`/`adamw`/`adafactor` on ROCm, since none of the alternatives get you 8-bit/paged state today. This module is outside this document's edit scope; flagged here so it doesn't come as a surprise mid-training. |
| **`iridium/runtime/placement.py:native_bf16`** | logic covered by CPU mocks; hardware unverified | queries `device.detect()` for each selected GPU, including `cuda:1`, so training and serving use the same vendor-aware bf16 predicate. This removes the earlier NVIDIA-SM test from the ROCm path. Run the device check on each real card before relying on mixed precision. |
| **`HSA_OVERRIDE_GFX_VERSION`** | unverified, unofficial for consumer cards | see above; unset by default in `Dockerfile.rocm` because a wrong value produces silently wrong kernels rather than an error. |
| **`ffmpeg`** | fixed here | `iridium/codecs/media.py` shells out to `ffmpeg`/`ffprobe` for every audio/video span; `Dockerfile.rocm` previously did not install it, so any media-bearing sample would raise `RuntimeError: ffmpeg is required...` on first use. Now installed via `apt-get`. |
| **Device nodes** | operational fact, not code | the container needs `/dev/kfd` and `/dev/dri` plus the `video` group, or `torch.cuda.is_available()` is simply `False` with no explanation. |
| **Multi-GPU (RCCL)** | unverified, likely fine | ROCm's collective library (RCCL) is a drop-in for NCCL at the API level and PyTorch's distributed backend dispatches to it under the same `"nccl"` backend name — there is no separate `dist.init_process_group("rccl")` to ask for. Nothing in this codebase currently sets up multi-node `torch.distributed`; `placement.py`'s multi-GPU path is single-process, multi-device, not multi-node, so RCCL/NCCL differences may not even be reachable yet. Not exercised here either way. |
| **Pinned memory** | not used | nothing in this codebase calls `pin_memory=True` on a `DataLoader` or `.pin_memory()` on a tensor, so there's no CUDA-only pinned-allocator behavior to differ on ROCm. Checked by grep, not by assumption. |

## Why fp16 is not offered as a fallback

On hardware without bf16 this reports **fp32**, not fp16. That is deliberate.
This is a *routed* model: the gating softmax picks which superstack a token
visits, and attention logits feed another softmax. Those are exactly where
fp16's 5-bit exponent overflows — and when it does, the router collapses onto
one stack and the run looks like a bad hyperparameter choice rather than a
numerics failure. Half the cost of a wrong answer here is the week spent
looking in the wrong place. Use bf16 hardware, or fp32.

## NVIDIA

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m iridium.runtime.device        # expect backend: cuda
```

bf16 needs compute capability 8.0+ (Ampere and newer). A free Colab **T4 is
Turing (7.5) and has no bf16** — see above for why that means fp32 there.

## Google Colab

[`notebooks/train_iridium_colab.ipynb`](../notebooks/train_iridium_colab.ipynb)
clones the repo, verifies the parameter accounting and the cache-parity gate
before spending GPU time, trains, grades against independent computation, and
serves the probe UI through a forwarded port. Runtime → Change runtime type →
GPU, then Run all.

## The one bug that kills every GPU run

`torch.Generator()` is a CPU generator, and

```python
torch.randn(shape, device="cuda", generator=cpu_generator)   # RuntimeError
```

The flow-matching head samples its noise on the target tensor's device, so the
trainer's generator has to live on the same device. `runtime/device.py`
provides `generator_for(device, seed)` and the trainer uses it. This is the
single most common way a script that works perfectly on CPU dies instantly on
a GPU, and it is vendor-independent — CUDA and ROCm both.

## Memory, per rung

Full fp32 Adam is 16 bytes per parameter: weights, gradients, an fp32 master
copy, and two moments.

| rung | params | fp32 Adam | fits on |
|---|---|---|---|
| `nano` | 34 M | 0.5 GB | anything |
| `nano100m` | 104 M | 1.7 GB | anything |
| `micro` | 1.0 B | 16 GB | 24 GB card, or bf16 + 8-bit optimizer on 16 GB |
| `test1b` | 1.0 B | 16 GB | same |
| `small` | 25.8 B | 413 GB | a cluster |

`iridium/training/continual.py` supplies the LoRA path when the optimizer state
will not fit: adapters are ~0.9% of the parameters and their Adam state is
correspondingly ~1% of the full figure.
