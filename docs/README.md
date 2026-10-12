# Documentation index

## Start here (1.0)

| Document | What it is |
|---|---|
| [`free-tier-training.md`](free-tier-training.md) | Presets, free tiers, hour estimates, the training workflow |
| [`architecture-options.md`](architecture-options.md) | Every architecture option, what it costs, and its evidence |
| [`small-model-coherence.md`](small-model-coherence.md) | Why small models babble, and what the presets do about it |
| [`tools.md`](tools.md) | Tool calling: format, data, schema-constrained decoding |
| [`streaming.md`](streaming.md) | Realtime streaming and lossless self-speculative decoding |
| [`long-context.md`](long-context.md) | Sliding window, MLA, Gated DeltaNet, YaRN: the 1M-token cache |
| [`world-model.md`](world-model.md) | Cameras, splat scenes, persistent world state, rollouts |
| [`geometry.md`](geometry.md) | Meshes, B-rep and STEP |
| [`gpu.md`](gpu.md) | CUDA, ROCm, precision, several runs on one GPU |
| [`multirun.md`](multirun.md) | Running several trainings on one device |
| [`SMALL_CHAT.md`](SMALL_CHAT.md) | The bundled pre-1.0 nano checkpoint: chatting with it, and why it can't chat well |
| [`training-8b.md`](training-8b.md) | The 8B rung: what it would take (costed, not free) |

## Canonical design and evidence

| Document | What it is |
|---|---|
| [`architecture.md`](architecture.md) | The canonical architecture |
| [`build.md`](build.md) | What was built and how to run it |
| [`verification.md`](verification.md) | Claims in the source plan that were checked |
| [`capability-register.md`](capability-register.md) | Status per capability: specified / implemented / trained / evaluated |
| [`decisions.md`](decisions.md) | Every correction made, with the test that holds it |
| [`evidence.md`](evidence.md) | Sources, and what each does *not* establish |
| [`requirements-traceability.md`](requirements-traceability.md) | Request → requirement → test |
| [`first-slice.md`](first-slice.md), [`results.md`](results.md) | Measured results from the early trained slice |
| [`backlog.md`](backlog.md), [`scenarios.md`](scenarios.md) | Milestones and capability targets |

## Superseded (kept for history)

`ARCHITECTURE_V2.md`, `UPDATE_NOTES.md`, `KAGGLE_TRAINING.md`,
`SELF_IMPROVEMENT.md`, `MULTIMODAL_DATA.md` describe earlier
revisions and their notebooks. Where they disagree with the documents above,
the documents above win. [`history/`](history/) holds the original ORRERY
specification.
