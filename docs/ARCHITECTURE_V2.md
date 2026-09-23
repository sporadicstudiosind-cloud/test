> **Superseded (kept for history).** This describes a pre-1.0 revision. For
> current instructions see [the docs index](README.md) and
> [free-tier training](free-tier-training.md).

# Iridium controller and context revision

Implementation status: source changes only. No tests, model imports, inference,
training, media conversion, or dataset downloads were executed for this revision.
Notebook generation and ZIP packaging are file-writing operations only.
This is not a claim of frontier intelligence, successful Kaggle training, or
measured improvement. Train and independently evaluate before promoting weights.

## What the model now does

```
Native text / RGB / PCM / video / geometry / fields / actions
                 |
        modality projection + observed media coordinates
                 |
     learned memory retrieval from earlier completed blocks
                 |
        FULL GENERAL CONTROL CORE <-------------------+
                 |                                    |
     learned stop + dispatch decisions                 |
        /               |                  \           |
     emit        another core cycle    subject routing |
        |               |                  |           |
   native heads         +------------------|-----------+
                                  deeper subject stacks
                                           |
                           returned latent states ------+
```

`controller_mode=True` runs all core layers each cycle. Domain routing reads that
core state. A core-owned dispatch head chooses specialist work or core recurrence.
Returned specialist states enter the next full core pass. There is no separate
static bank integration gate in this mode. The final cycle cannot dispatch work
because no later core pass would remain to integrate it.

The core halt head trains through task losses at each possible stopping cycle.
Inference uses the first stop probability above the threshold, otherwise the
budget's last cycle. This is bounded recurrent computation, not an unbounded
thought loop. Internally it still computes the budgeted core cycles to maintain
dense, separate per-cycle caches. It does not yet save all the compute of an early
stop. Specialist dispatch itself is sparse. A one-cycle run is core-only.

The specialist bank still uses deeper transformer layers, bridge attention to
first-cycle core perception, optional field spectral operators and learned depth
mixtures. These mechanisms belong to one model checkpoint. They are not calls to
external specialist AI services. Subject names alone are not trained expertise.

## Geometry choices

Use `iridium.config_builder.intelligence_preset(name)`:

| Name | Width | Core layers | Stacks | Layers per stack | Memory slots |
|---|---:|---:|---:|---:|---:|
| consumer_tiny | 256 | 4 | 4 | 6 | 32 |
| consumer | 384 | 8 | 4 | 12 | 64 |
| workstation | 768 | 16 | 6 | 28 | 128 |
| research_large | 2048 | 32 | 12 | 64 | 256 |
| frontier_design | 8192 | 64 | 32 | 128 | 512 |

All use a maximum eight-cycle budget; notebooks start with three cycles and the
smallest preset. Core weights are reused between cycles. Larger versions need
hardware and distributed training engineering; their existence in configuration
is not evidence they fit Kaggle. The source retains old numeric presets and
legacy behavior for older checkpoints. New architecture weights require new
training; an old checkpoint is not silently upgraded into a trained new model.

## Learned context compression

The small `ContextMemory` network is registered inside Iridium and optimized with
it. It has a low-rank encoder/decoder, learned token-importance scorer, retrieval
query/key projections, and an input-dependent read gate.

1. Keep a pending block of raw token embeddings.
2. After the block completes, encode and importance-weight its tokens into a
   learned summary. That summary can only affect later blocks.
3. Attend to summaries and gate the retrieved content into later input embeddings.
4. When capacity fills, consolidate the two oldest summaries into one coarser
   summary. Keep more recent summaries separately.
5. Train through downstream prediction losses, plus a small local embedding
   reconstruction objective. Padding is excluded.

Memory state is per stream, not a global module buffer. Inference does not update
network weights. This is not a full Titans implementation or an autonomous model
weight rewrite. It compresses input embeddings, not every layer's historical KV.
The reconstruction auxiliary trains the low-rank bottleneck; it does not prove
that weighted summary pooling preserves every source fact.

`LongContextSession` retains that memory while resetting exact core, bridge,
router and specialist caches at explicit local-window boundaries. Its default
ingestion budget is 1,048,576 tokens. The number is a runtime budget, NOT a tested
million-token recall result or an exact million-token attention window. Repeated
consolidation can lose names, equations, tiny labels and old details. Local
boundaries also remove exact cross-boundary attention.

Standard training learns within the supplied examples. The default short Kaggle
curriculum does not by itself train million-token retrieval or the long-session
boundary distribution. Long-sequence training and retention evaluation remain
necessary before claiming that capability. Source resolution, exact local window,
compressed memory capacity and total ingestion budget are separate quantities.

`SourceArchive` optionally keeps exact original text in a caller-selected SQLite
file. It provides lexical search and exact chunk lookup with source identifiers.
Callers explicitly add and retrieve documents; nothing silently uploads or
archives them. This is ordinary code, not another neural model. Feed retrieved
source text back as context when exact wording matters. Original image/audio/video
files must likewise remain available for exact rereading. Retrieval is not yet an
autonomously trained core action.

## High-resolution perception

`image_tiles` preserves source resolution up to its default 67,108,864-pixel
budget (8K square), rather than globally resizing to 64 pixels. It emits bounded
tiles, pads only patch edges, and records source coordinates and crop bounds.
`video_tiles` does the same for native spatiotemporal patches with explicit frame
sampling and time coordinates. Video rotation metadata is not automatically
applied on that native path. Frame groups can still consume substantial CPU RAM.

`stream_image` and `stream_video` feed those tiles into a bounded long-context
session. More pixels consume more tokens and compute. Within the model, observed
media coordinates get parameter-free Fourier features; the learned patch
projections now include modality-specific residual nonlinear encoders (two
bottleneck blocks in consumer presets, four in larger presets). Those encoders
and the shared core provide trainable perception. This is not a large,
pretrained vision encoder or a claim that arbitrary 8K images are understood.

Manifest image/video parts may set `native_resolution: true` and `tile_size`.
The ordinary manifest trainer still rejects examples exceeding its context
budget; it does not silently shrink them or pretend to train the whole image.
Use bounded crops for small-device training and a separately designed long-media
curriculum for the full streaming regime. Autoregressive output payloads retain
stream-position conditioning; a layout-conditioned output encoder is future work.

## Real data and subject training

The notebooks now perform general warmup, per-subject specialization, and joint
refinement. During a specialization phase, the target bank stack is trainable,
the other model parameters are frozen, and labelled examples explicitly route to
that stack. General/joint phases train the entire model. Optimizers are fresh at
phase transitions; checkpoints identify the phase. `RESUME` resumes the final
joint phase and skips warmup/specialization. Resuming an interrupted warmup or
subject phase is not automated by the notebook.

Real subject recipes use only dataset **train splits** and pin the resolved
dataset revision in provenance:

- Math: [OpenAI GSM8K](https://huggingface.co/datasets/openai/gsm8k), MIT.
- Code: [Google MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp), CC BY 4.0.
- Science: [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc), CC BY-SA 4.0.
- Native speech: [OpenSLR LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr), CC BY 4.0.

Speech preparation keeps complete utterance/transcript pairs that fit the budget.
It creates both audio-to-transcript and transcript-to-audio targets for Iridium's
own PCM heads. It does not call Whisper or an external TTS network. Speaker/chapter
identity is retained for future disjoint evaluations. Small presets assign speech
to their language stack when they lack a separate audio stack.

Starter subsets are intentionally small and cannot teach broad expert skill.
Math questions are not proof research; school science QA is not a physics solver.
Image/video captioning, editing plans, tool observations, failures and corrective
actions still require user-supplied real paired manifests. This revision does not
fabricate those training results. Text chat alone cannot teach those capabilities.

Subject labels supervise routing and the core dispatch head during training.
They are not required at inference. Missing subject data is reported as skipped.
The optional research loop remains off by default, retains independent evaluation,
and rejects a static bank gate mutation in controller mode. It does not automatically
execute proposed arbitrary code or run independent tests during notebook Run All.

## Why these additions fit; what remains experimental

Learned compression and reconstruction objectives have precedent in
[Compressive Transformers](https://arxiv.org/abs/1911.05507). Neural long-term
memory is also explored in [Titans](https://arxiv.org/abs/2501.00663). These papers
motivate investigating memory; they do not establish that this custom variant
improves Iridium. The present implementation uses simpler fixed weights at inference.

Further candidates worth evaluating are retrieval-trained rereading, memory
retention objectives across actual long-session boundaries, better multilingual
tokenization, learned multiscale perceptual encoders, specialist disagreement
training, and outcome-supervised tool planning. They are not silently represented
as implemented or proven. A tokenizer change requires dataset/checkpoint migration;
an extra verifier head needs reliable correctness labels; destructive token pruning
can erase information that only becomes relevant later. Adding all of these at
once would make regressions difficult to attribute.

For the requested audio-section -> transcript -> subtitled video workflow, the
existing `MediaAgent` and deterministic media tools remain the execution layer.
The model still needs native ASR and tool-planning training to choose the right
actions. Output validity and task success belong in independent evaluation.
Flow matching remains an optional native media head; it does not replace the
core's reasoning or the explicit tool executor.

No success claim substitutes for held-out capability measurements. Per the user's
instruction, none of those measurements were run by the assistant.
