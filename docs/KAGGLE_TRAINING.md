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
cost. Changing precision cannot resume an old optimizer checkpoint under the
current strict resume contract; start fresh or explicitly import weights.

PyTorch references: [AMP guidance](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
and [a related reported AMP backward dtype failure](https://github.com/pytorch/pytorch/issues/81876).
The latter is not proof of the precise cause in this run.

## Using both GPUs

Select **GPU T4 x2** in Kaggle, restart the session, and use the updated notebook.
The notebook defaults to `GPU_MODE='auto'`, which selects up to two visible GPUs.
`GPU_MODE='single'` selects only GPU 0. The code prints each device's name and
available memory, the stack-to-device map, per-device parameter estimates, and
actual allocated/reserved/peak memory in training records.

This implementation is single-process **model parallelism**:

- The control core, codecs, router and learned input memory live on GPU 0.
- Whole specialist stacks are greedily distributed by parameter count.
- Each stack's weights, gradients and eager AdamW moments live on its GPU.
- Routed states and bridge context cross devices through differentiable copies;
  specialist outputs return to the core's GPU for integration.
- Components are placed directly from CPU. Training does not first load the
  complete model onto GPU 0.
- Gradient scaling and global clipping apply to the complete update. Checkpoints
  include the device map and device-local optimizer state.

The cards are not one contiguous 32 GB allocation. A component must fit on its
assigned card. Per-device activation/optimizer budgets matter. Stack execution
is scheduled sequentially by the bank, so both cards need not be fully busy at
the same instant. Transfers can make small models slower; this implementation
primarily expands available model capacity. It is not DDP, tensor parallelism,
or pipelined microbatch execution. Unequal core/stack widths and non-eager
optimizers are rejected in multi-device training until explicitly supported.
Optional research subprocesses still use their configured single device.

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

Start with the unchanged consumer_tiny default. Increase only after inspecting
both device plans and live memory. If the conservative plan rejects a choice,
reduce MAX_SEQ_LEN or the preset. Micro-batch reduction cannot solve a model-state
allocation that alone exceeds VRAM. Specialist depth remains greater than core
depth at every size.

## Updating a previously failed Kaggle session

1. Save any checkpoints you want to retain, then restart the kernel/session.
2. Import the latest iridium_studio_kaggle.ipynb from the branch. The new default
   checkout is `/kaggle/working/iridium-updated-v3`; it avoids reusing the old path.
3. Run setup and settings again. Confirm two devices and Selected precision: fp16
   when running on two T4s. Ensure both GPUs appear in the stack placement map.
4. Run data and training. Checkpoints default to every 50 completed optimizer
   steps within each phase. GPU memory is printed at log intervals.

Do not rerun only the failed cell against old imported classes. A failed backward
can leave partial gradients; this revision does not repair a running old kernel.
The original warning was harmless logging, but the fatal exception cancelled the
current training update. Your step-0 line indicates one reported update, not a
completed training phase.
