# Latest: controller and learned memory revision

Read `ARCHITECTURE_V2.md` first. It supersedes earlier architecture/default-setting
notes below. All four notebooks now default to `consumer_tiny`, three full-core
cycles, nonlinear native modality encoders, and learned compressed memory.
Real subject and paired speech data preparation runs when you execute the data
cell. General warmup, subject-specific training and joint refinement run when you
execute the training cell. This increases total training beyond the joint STEPS
setting. Custom image/video/tool manifests are still needed for those skills.

The bounded long-context path, source archive and high-resolution tiling helpers
are included. Million-token ingestion is a budget, not measured million-token
recall. No tests, inference, training or dataset downloads were run here.

---

> Follow-up revision: see [SELF_IMPROVEMENT.md](SELF_IMPROVEMENT.md) for the new
> optional researcher loop and further architecture changes. Both this update and
> the follow-up remain untested/untrained per the user's request.

# September 17 update: Kaggle, training, multimodal media and agency

## Delivery and validation status

This revision was implemented from the supplied branch ZIP. The GitHub branch
could not be fetched through the web tool, so the ZIP is the authoritative base.
The pasted original plan was treated as design context, not as verified research
or an instruction to run its deployment/training commands.

**Per the user's request, no tests, model inference, training, dataset downloads,
GPU probes, or media processing were run.** Source was read and edited; the
notebook generator only serialized notebook JSON. No new weights or benchmarks
are included. Existing checkpoint/results files are historical and do not validate
this revision. In particular, no claim of higher intelligence or faster measured
throughput is made.

## What happened in the Kaggle screenshot

The visible traceback enters Trainer initialization, optimizer construction and
PyTorch's lazy compiler import, ending with:

`AttributeError: module 'torch' has no attribute '_dynamo'`

The model construction and initial evaluation printed output beforehand. This
particular failure therefore occurred before the training optimizer could be
constructed. It is not the CUDA OOM traceback. The screenshot does not establish
the precise installed torch version, whether packages had been changed in that
kernel, or the exact cause of the inconsistent import state. A circular/partial
compiler import or mismatched notebook environment are plausible, not proven.

The default `eager_adamw` uses ordinary torch tensor operations and deliberately
does not inherit from `torch.optim.Optimizer`, whose initialization reaches
Dynamo in affected versions. It implements decoupled weight decay and
bias-corrected Adam moments. It does not replace torch modules or pretend to fix a
broken PyTorch installation. This new optimizer implementation is untested in
this delivery and may be slower than native fused AdamW. Native `adamw` remains
selectable after restoring a healthy environment.

1. Upload/merge this revision into the branch, or attach the delivered ZIP.
2. Open `notebooks/iridium_studio_kaggle.ipynb` in a **fresh Kaggle session**.
3. Select a GPU and enable Internet for the default GitHub/pip/dataset path.
4. Run cells in order. For an attached ZIP, fill in `PROJECT_ZIP` in cell 1.
5. Start with `100m`, context 1024, micro-batch 2, effective batch 16, one loop,
   `eager_adamw`, and `PRECISION='auto'`.
6. Download `iridium-inference.pt`, full checkpoints under `runs/studio/`, and
   media artifacts from the working directory. Preserve checkpoints before
   ending the session; notebook output persistence depends on Kaggle's save flow.

The notebook uses one GPU. Two 16 GB T4s are **not** treated as one 32 GB device.
The memory estimate includes a conservative quadratic attention fallback and all
stack depths, but is still an estimate. Arbitrary larger presets are not made
trainable by this patch. If even micro-batch 1 OOMs, reduce model/context size.
No package reinstall can guarantee success if the runtime itself remains broken.

## Architecture and efficiency changes

- Keep the shared control core, causal prefix router, sparse stack-local memory,
  global bridge attention, adaptive depth/loop heads and native emission heads.
- Add configurable, parameter-free per-head Q/K RMS normalization before RoPE.
  Enabled in new notebooks and serialized in config; old configs default off.
- Initialize residual output projections at a depth-scaled variance. This affects
  new models; loaded weights override initialization. Its quality impact is unmeasured.
- Keep routing probabilities/Gumbel calculations in fp32 under AMP, and exclude
  padding from dispatch and the load-balancing/z losses.
- Preserve broadcast attention masks instead of expanding and materializing a
  separate mask for every attention head. SDPA remains the default; actual
  kernel selection and speed depend on device, dtype and shape.
- Allocate continuous payload tensors only for modalities present in a batch.
  Evaluate text and continuous output losses only on their supervised positions.
- Bound flow-time Fourier frequencies to exponents 0..8 instead of 0..63, avoiding
  enormous low-precision phases. This changes continuous-head behavior; legacy
  continuous checkpoints require revalidation/fine-tuning.
- Only field grids enter the physical spectral blocks. RGB/video grids retain
  their media layout without being treated as a 2D physical field.
- Preserve generated action operands in both returned results and the next input;
  supervise only operands used by the opcode.

These are mechanisms for stability, resource use and trainability. They are not
evidence of a smarter trained model. The core is still byte-level for text,
uncompressed patch-level for media, and intended as a small research baseline.

## Training changes

- FP32 master weights and automatic CUDA autocast: fp16 plus loss scaling on
  older GPUs; bf16 where supported. CPU uses fp32. Do not call `model.half()` to train.
- Clip after unscaling; reject nonfinite losses/gradients. Repeated fp16 overflow
  is bounded instead of looping forever.
- Recover forward/backward OOM by discarding **the entire partial update**,
  reducing micro-batch, adjusting accumulation and rebuilding the batch stream.
  New samples are drawn on retry; this is not deterministic replay of the old batch.
- Do not retry an optimizer-step OOM, because some parameters may already have
  changed. Restart from a checkpoint. Sparse gradients are not supported by eager AdamW.
- Save optimizer, scaler, step, config, data metadata and RNG state atomically.
  Resume restores them but starts a fresh data shuffle, so it is not bit-exact replay.
- Reject missing optional optimizer dependencies instead of falling back to an
  optimizer with a larger, unbudgeted footprint. Paged moments are conservatively
  budgeted as potentially GPU-resident, not assumed to cost zero VRAM.
- Reject empty corpora, invalid mixtures and batches with no supervised targets.
  Overlong notebook text/chat examples are removed rather than silently cut.
- All four notebooks come from one generator. They preserve the host torch install,
  expose mixed/media-only data modes and omit automatic tests, evaluation runs,
  interactive loops, inference examples, and servers from Run All.

Gradient checkpointing remains disabled. The ZIP documented prior recomputation
problems; this patch does not claim to have corrected or independently verified them.
Sparse dispatch is still Python-driven; distributed training, tensor sharding,
multi-GPU DDP, and high-throughput serving are not introduced here.

## Multimodal and agentic additions

`codecs/media.py` loads RGB images, temporal RGB patches, and reversible 16 kHz
mono PCM patches. It exports PNG/WAV/MP4. These adapters call no other AI model.
`data/multimodal.py` builds paired interleaved supervision for text, audio,
image, video, numeric geometry/fields/quantities, and actions. Assistant outputs
are targets; user and tool observations are context. See `MULTIMODAL_DATA.md`.

`runtime/media_generation.py` sends text and optional observed media through
Iridium and exports its own continuous-head outputs at a caller-declared layout.
It does not claim high-resolution perceptual quality or semantic correctness.

`agency/media_agent.py` provides two entry points:

- `MediaTools.clip_with_subtitles(source, start, end)`: extract mono audio,
  trim video, transcribe using the same Iridium model, write an SRT, and mux
  replacement audio plus an embedded selectable subtitle track into MP4.
- `MediaAgent.run(request, sources)`: use the same model to emit bounded JSON
  tool calls, observe their results, and iterate. Tools include inspection,
  native image/video description, audio extraction, video trim, native ASR,
  subtitles, composition, and clip concatenation. Every artifact must actually
  exist before it is registered/returned. Errors enter the tool trace.

The agent sees registered asset IDs and tool results, not arbitrary filesystem
paths. Source files are registered by the caller. Generated text cannot become
shell/Python/filter code. No external messages, network tools, or destructive
application control are supplied. This is a bounded media agent, not arbitrary
OS/Blender automation or a security sandbox for hostile media decoders.

Transcription requires paired speech/text training. Captions use chunk start/end
times, **not word alignment**; nonspeech detection and transcription accuracy are
unverified. Subtitle tracks are selectable, not burned into the pixels.
Concatenation standardizes video to 640x360/30 fps and audio to 48 kHz stereo;
it does not carry over input subtitle tracks. Apply captions after stitching.

## What you still need to train

A text/synthetic-only run cannot teach the audio/video heads or the media agent
to work reliably. Supply aligned speech transcripts, visual captions, native
media targets, geometry/field targets and successful/error-recovery tool traces
under the exact training schema. The supplied template is a schema example,
not a training corpus. Short runs on a free GPU are not demonstrated to produce
a broadly intelligent, hallucination-free, fully capable assistant.

Existing legacy checkpoint tensor shapes mostly load, but the flow-time feature
change alters continuous predictions and the RGB/PCM representation is a new
data contract. Do not label the old checkpoint as trained ASR or video synthesis.
Full optimizer resume is only available for checkpoints produced by this revision.

## Acceptance checks for the user's later run (not executed here)

1. Fresh Kaggle session: source path and torch provenance print correctly; the
   default optimizer initializes and training reaches a logged update.
2. Verify finite loss/gradients across several updates, then save and resume;
   completed step and optimizer moments should continue rather than restart.
3. Evaluate the same untouched held-out tasks before/after an equal token budget.
   Compare fp32 vs AMP stability, memory and throughput on the actual accelerator.
4. Re-run existing cache parity, causality and inventory checks before treating
   this changed architecture as validated. Historical results do not transfer.
5. Use a tiny licensed paired-media corpus to inspect per-modality learning;
   evaluate native ASR on an unseen recording and media outputs on unseen prompts.
6. Run the requested clip workflow, inspect speech accuracy, subtitle times,
   selected audio range and final MP4 duration. Play it in a subtitle-capable player.
7. Evaluate agent traces for correct tool arguments, artifact selection, recovery
   after bad arguments and clean termination on an exhausted budget.

## Primary references used to check implementation choices

- [PyTorch AMP examples](https://docs.pytorch.org/docs/main/notes/amp_examples.html):
  loss scaling, accumulation, unscaling and clipping order.
- [PyTorch SDPA API](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html):
  boolean masks, kernel selection and grouped-query implementation constraints.
- [PyTorch issue 158120](https://github.com/pytorch/pytorch/issues/158120): a related
  partially initialized Dynamo import failure; **not proof of this Kaggle incident's cause**.
