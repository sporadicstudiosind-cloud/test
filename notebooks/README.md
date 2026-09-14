# Training Iridium-1 in the cloud

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sporadicstudiosind-cloud/test/blob/claude/gallant-faraday-lhycva/notebooks/train_iridium_colab.ipynb)

`train_iridium_colab.ipynb` runs end to end: clone, verify, train, grade, save,
serve. **Runtime → Change runtime type → GPU**, then **Run all**.

What it does, in order, and why that order:

1. **Checks the GPU** and reports whether bf16 is real on it.
2. **Verifies the architecture before spending GPU time** — the parameter
   formulae against the instantiated modules (expected difference: exactly
   zero) and the cache-parity gate (cached decoding must compute what teacher
   forcing trained). If either fails, nothing downstream means anything, so it
   is cheaper to find out in twenty seconds than after an hour of training.
3. **Trains** the rung you pick, checkpointing every fifth of the run.
4. **Grades** by free-running generation against an independent computation —
   the analytic Manning law, the spectral solver, the scene environment's own
   goal predicate — each reported beside the baseline a model earns by ignoring
   its input entirely, and separately on an extrapolation split drawn from
   outside the training band.
5. **Saves** an fp16 checkpoint and offers it for download.
6. **Serves** the probe UI on a forwarded port, where a physics question gets a
   number back beside the analytic value, so it can be checked and not believed.

## Which rung

| rung | params | GPU needed | wall time |
|---|---|---|---|
| `nano` | 34 M | anything | ~5 min |
| `nano100m` | 104 M | anything | ~12 min |
| `micro` / `test1b` | 1.0 B | 24 GB preferred | ~40 min |

## Before you pick a free T4

A free Colab T4 is Turing and **has no bf16**. For a routed model that matters
more than usual: the gating softmax that selects a superstack and the attention
logits both live exactly where fp16's exponent overflows, and when they do, the
router collapses onto one stack and the failure reads as a bad hyperparameter
rather than a numerics bug. The notebook therefore runs **fp32 on a T4** and
bf16 only on Ampere or newer. If you want the 1 B rung trained properly, an
A100/L4 on Colab Pro, or a 4090 on RunPod or Vast at roughly $0.30/hour for a
few hours, is the honest path.

## Other services that work

Same notebook, or `PYTHONPATH=. python -m iridium.training.phase1_pretrain`:

- **Kaggle Notebooks** — 30 GPU-hours/week free, P100 16 GB or 2×T4. The best
  free quota of the lot.
- **Lightning AI Studios** — free monthly GPU hours, persistent environment.
- **Modal** — free monthly credits, and a good structural fit since this repo
  is already an importable package.
- **RunPod / Vast.ai** — cheapest route to an Ampere-or-newer card.
- **AMD hardware** — see [`../docs/gpu.md`](../docs/gpu.md); ROCm needs no code
  changes, only the right wheel and the device nodes.

Hugging Face Jobs and ZeroGPU both require a Pro subscription for GPU, so they
are not a free option.
