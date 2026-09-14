# Training Iridium-1 in a high-RAM Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sporadicstudiosind-cloud/test/blob/main/notebooks/train_iridium_colab.ipynb)

`train_iridium_colab.ipynb` is an end-to-end Colab for a **CPU-only 48 GiB
system-RAM runtime**. Select **Runtime → Change runtime type → High-RAM** and
Run all. A GPU accelerates it when Colab gives you one, but it is neither
required nor used as a memory prerequisite.

The default `micro`/`test1b` rung is about 1.0 B parameters. Full fp32 Adam
requires about 16 GB for parameters, gradients, and optimizer state, before
activations and checkpoint-writing overhead. The notebook uses a batch size of
one to keep that work inside a 48 GiB host-RAM session. This is deliberately a
slow CPU configuration; increase `STEPS` only after its first checkpoint fits.
Choose `nano100m` for a faster end-to-end smoke test.

What it does, in order:

1. **Reports system RAM** and warns if the runtime is not high-RAM; it also
   reports any GPU present, without requiring one.
2. **Verifies before training**: parameter accounting and cached-decoding
   parity must pass before the notebook creates the training workload.
3. **Trains in fp32 on CPU** (or uses CUDA if available), checkpointing every
   fifth of the selected run.
4. **Grades by free-running generation** against independent computations and
   separate interpolation/extrapolation splits.
5. **Saves an fp16 checkpoint** for download, while keeping CPU training fp32.
6. **Serves the probe UI** through a forwarded Colab port.

## Memory and runtime expectations

| rung | parameters | full fp32 Adam | high-RAM CPU suitability |
|---|---:|---:|---|
| `nano` | 34 M | ~0.5 GB | easy |
| `nano100m` | 104 M | ~1.7 GB | good for a quick run |
| `micro` / `test1b` | 1.0 B | ~16 GB | fits 48 GiB at batch 1; slow |

The 16 GB figure is an optimizer-state estimate, not a promise of peak RAM:
activations, temporary tensors, Python, the corpus, and saving a checkpoint
need extra room. That is why the high-RAM preset uses `BATCH = 1` and only 25
steps initially. It does not depend on bf16, T4 memory, or any other GPU HBM.

## Optional GPU acceleration

A GPU is strictly optional. If one is available, the notebook selects CUDA and
uses bf16 only on Ampere-or-newer hardware; otherwise it remains in safe fp32.
A T4 has no bf16, so it also stays fp32. See [`../docs/gpu.md`](../docs/gpu.md)
for NVIDIA and ROCm details.

## TPU v5e-1 builder

[![Open TPU Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sporadicstudiosind-cloud/test/blob/main/notebooks/train_iridium_tpu_colab.ipynb)

`train_iridium_tpu_colab.ipynb` is a separate TPU notebook. It offers `34m`,
`100m`, `1b`, `8b`, and `25b` configuration presets plus controls for control
core layers, superstack layers, and the number of superstacks. It trains only
the 34M/100M presets on a single TPU and costs the larger builds without
materialising them; a single v5e-1 is not enough for their full optimizer
state.
