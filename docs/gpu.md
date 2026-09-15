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

## AMD (ROCm)

PyTorch's ROCm build exposes AMD hardware through the same `torch.cuda` API via
HIP, so `.to("cuda")` lands on a Radeon or Instinct card. Nothing in this
repository needs changing.

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
python -m iridium.runtime.device        # expect backend: rocm
PYTHONPATH=. python -m iridium.training.phase1_pretrain --rung nano100m --steps 3000
```

Or containerised, which is usually less painful:

```bash
docker build -f Dockerfile.rocm -t iridium-rocm .
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --security-opt seccomp=unconfined \
  -p 8080:8080 iridium-rocm
```

Things that actually bite on ROCm:

| | |
|---|---|
| **bf16** | present on CDNA (MI100+, `gfx90a`/`gfx942`) and RDNA3 (`gfx1100`+), absent on older RDNA. `device.detect()` reports it per architecture. |
| **`HSA_OVERRIDE_GFX_VERSION`** | consumer cards sometimes need this to claim a supported arch — e.g. `11.0.0` on RDNA3, `10.3.0` on RDNA2. Unset by default here because setting it wrongly produces silently wrong kernels rather than an error. |
| **FFT** | the spectral blocks go through rocFFT. Supported, and `verify()` tests it rather than assuming. |
| **Device nodes** | the container needs `/dev/kfd` and `/dev/dri` plus the `video` group, or `torch.cuda.is_available()` is simply `False` with no explanation. |

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
