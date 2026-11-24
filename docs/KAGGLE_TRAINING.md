> **Superseded (kept for history).** This describes a pre-1.0 revision. For
> current instructions see [the docs index](README.md) and
> [free-tier training](free-tier-training.md).

# Kaggle training revision: AMP and two T4s

No training, inference, or tests were run by the author for this revision.
The changes were reviewed at source level only. Your runtime trace is evidence
of the earlier failure, not validation of the replacement implementation.

## The reported error

The requires-grad-to-scalar warning came from reporting expected depth. That
statistic now uses detach() before conversion, as do trainer routing diagnostics.
It is separate from the fatal backward dtype mismatch.

The halting path previously passed reduced-precision probabilities to cumprod.
AMP can promote that operation to FP32, while backward's zero-handling path uses
masked scatter. The revised path explicitly casts halting probabilities to FP32
before cumulative products and retains FP32 for depth mixtures and KL. Float64
reference inputs remain float64. This addresses an identified dtype boundary;
the exact originating autograd node was not captured in the supplied traceback,
and the fix has not been reproduced on Kaggle by the author.

Precision selection now checks native CUDA architecture support across every
selected GPU. T4/P100 use FP16 and GradScaler, not emulated BF16. Master weights
remain FP32. FP32 remains a selectable troubleshooting mode with a higher memory
cost. Resuming under a different precision is allowed: the optimizer state is
loaded and only the fp16 loss scale starts fresh (`Trainer.resume` prints a note).

PyTorch references: [AMP guidance](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
and [a related reported AMP backward dtype failure](https://github.com/pytorch/pytorch/issues/81876).
The latter is not proof of the precise cause in this run.

## Using both GPUs

Select **GPU T4 x2** in Kaggle, restart the session, and use the current notebook
(see [kaggle-agent-runbook.md](kaggle-agent-runbook.md)). The notebook no longer
has a `GPU_MODE` setting. It runs `iridium train --gpus auto`, which uses every
visible GPU at once; the log prints `training on 2 GPUs at once`.

This is **data parallelism**, not model parallelism:

- One process per GPU (`iridium/training/parallel.py`). Each rank holds a full
  copy of the model, its gradients and its optimizer state on its own card.
- Each rank runs its share of every global batch. Gradients are averaged with one
  all-reduce per optimizer step, so the batch, the schedule and the result match a
  single-GPU run.
- If any rank fails, the launcher stops the rest. Rank 0 logs to the notebook;
  the other ranks write `rank<N>.log` under the run folder.
- Only rank 0 writes checkpoints.

The older single-process placement (`iridium/runtime/placement.py`, which spread
one model's stacks over cards) is still in the library, but the CLI and the Kaggle
notebook do not use it. Each card must fit a whole replica plus its activations.
The measured `iridium-50m` run on two T4s ended at micro-batch 8 after a memory
fallback.

## Small and medium presets

The notebook prints formula-derived parameter counts before allocation. Names
describe relative geometry, not a promise of a particular trained capability.

| Preset | Width | Core layers | Stacks | Layers per stack |
|---|---:|---:|---:|---:|
| micro | 128 | 2 | 4 | 3 |
| mini | 192 | 3 | 4 | 5 |
| consumer_tiny | 256 | 4 | 4 | 6 |
| small | 320 | 6 | 4 | 9 |
| consumer | 384 | 8 | 4 | 12 |
| small_plus | 512 | 8 | 4 | 14 |
| medium | 640 | 10 | 4 | 18 |
| medium_plus | 768 | 12 | 6 | 22 |
| workstation | 768 | 16 | 6 | 28 |

These geometries belong to the retired placement notebook (its `consumer_tiny`
default is in `iridium/config_builder.py`). The current Kaggle notebook trains
`iridium-50m`, whose geometry is in `iridium/presets.py`. In the table above,
specialist depth is greater than core depth at every size.

## Updating a previously failed Kaggle session

1. Save any checkpoints you want to retain, then restart the kernel/session.
2. Import the latest iridium_studio_kaggle.ipynb from the branch. Its clone cell
   fetches the release branch into a checkout of this repository, or updates an
   existing checkout (a checkout with local edits stops the cell).
3. Run the notebook from the top. Confirm the cell lists two GPUs, and that the
   training log says `training on 2 GPUs at once`. On two T4s the trainer uses
   fp16 autocast with gradient scaling.
4. Run data and training. A checkpoint is written at the end of each round
   (`iridium-50m` has 300 rounds, about every 250 steps), not every 50 steps. GPU
   memory is printed at log intervals.

Do not rerun only the failed cell against old imported classes. A failed backward
can leave partial gradients; this revision does not repair a running old kernel.
The original warning was harmless logging, but the fatal exception cancelled the
current training update. Your step-0 line indicates one reported update, not a
completed training phase.
