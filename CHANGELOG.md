# Changelog

## Unreleased

### Fix: first Kaggle training step crashed on GPU
- `place_model` moved a hardcoded list of submodules to the primary device.
  The per-layer embedding table (`ple`, added with chat-36m) was not on it,
  stayed on the CPU, and the first GPU forward pass raised "Expected all
  tensors to be on the same device ... index is on cuda:0, different from
  other tensors on cpu". It now walks the module tree, so a new submodule
  cannot be left behind. `tests/unit/test_place_model_covers_every_module.py`
  checks every preset on the `meta` device, no GPU needed, and fails on the
  old code.
- The notebook clone cell now updates an existing checkout (fetch + reset,
  refusing if it has local edits) instead of silently reusing stale code,
  prints the source revision, and asks for a kernel restart when the code
  changed under modules already imported.
- `python -m iridium chat` with no `--checkpoint` now prefers
  `$IRIDIUM_CHECKPOINT`, then `weights/iridium-inference.pt`, before the
  bundled demo.
- `tests/unit/test_precision.py`: the fp64-downcast guard matched frames by
  a `"/iridium/"` substring, which never matches Windows paths, so it passed
  vacuously there. It now compares against the package directory.

### `chat-36m`: chat-34m rebuilt from what its first run showed
- The trained chat-34m (20k steps, 330M tokens, val 1.19 bpb text / 0.87 chat)
  spent 24.4M of a 30M body on four superstacks the router used evenly
  (usage entropy 1.38 of 1.39: no specialisation), one carrying Fourier blocks
  no text touches, around a 5.6M shared core; and it saw ~11 tokens per body
  parameter.
- New preset `chat-36m`: d384, 10-layer core (15.7M), three plain 4-layer
  stacks routed top-2 (20.1M), no spectral blocks. Body 35.8M; embeddings
  5.9M on top (tied 8k token table + Gemma 3n per-layer embeddings,
  `ple_dim=32`); media codec heads 4.5M, unused by text.
- `CodecConfig.embed_scale`: Gemma/EmbeddingGemma-style `sqrt(d_model)` input
  scaling of the tied token table. Off by default; on in chat-36m.
- Compute-optimal budget: 20 tokens per *body* parameter = 716M tokens,
  batch 64 x 1024, 10,929 steps, 8 rounds. Mixture 70% text / 25% chat /
  5% unknowable, chat weighted so Dolly and OASST repeat at most ~4 times.
- `IridiumConfig.body_params` / `embedding_params`; `--tokens-per-param` and
  the budget audit now count body parameters (Kaplan-style non-embedding N).
  `iridium presets` shows body, embeddings and tokens per body parameter.
- The Kaggle notebook and `docs/kaggle-agent-runbook.md` now train chat-36m.

## 1.0.0 — 2026-09

The first release meant to be *trained by someone else*: every capability is
reachable from a named preset, sized for free compute, and labelled with how
much has actually been established. **Status: verified in theory, untrained.**

### Data pipeline rebuilt
- One tokenizer interface: byte level is just the smallest tokenizer. Rust
  byte-level BPE (Hugging Face `tokenizers`) is the default, pure Python the
  fallback; every checkpoint stores its tokenizer; `check_fits` refuses a
  tokenizer larger than the embedding table.
- `iridium data prepare`: tokenized, memory-mapped disk shards; training RAM
  no longer scales with the token budget. Sources stream one at a time.
- `adamw8`: 8-bit block-wise AdamW in plain PyTorch (CUDA, ROCm, XLA, CPU).
- `--tokens` / `--tokens-per-param` on `train` and `data prepare`.
- Held-out shards and validation **bits per byte** every 5% of training
  (tokenizer-independent, so runs with different vocabularies compare).
- Chat and tool conversations are packed into windows (+31% real tokens per
  chat window at 512; more at longer windows); exact duplicates are dropped.
  No intra-document attention mask yet (GPT-3/Llama 2 style packing).
- `--hparams deepseek`: lr and batch from DeepSeek LLM's compute-optimal fit,
  for a one-flag comparison against the preset's own values.
- Size ladder presets `50m`, `100m`, `500m`, `1b` (8-bit, fits 16 GB), `2b`, `4b`.

### Priorities, as presets
- `chat-34m`, `chat-100m` (talking + reasoning), `tools-100m` (tool use),
  `omni-100m` (omnimodal), `stem-100m` (physics / STEM), `world-100m` (world
  model); `modern-744m` and the `8b` rung are costed but not free-tier trainable.
- `python -m iridium presets` and `python -m iridium train --preset NAME`
  (`--dry-run`, `--resume`, `--init`). Training runs in rounds of fresh
  streamed data with one learning-rate schedule across the whole run.
- Free-tier hour estimates computed from each config's own FLOPs.

### Tool use
- A tool-call format on reserved control tokens, licensed and exactly-graded
  synthetic tool data, and schema-constrained decoding (see `docs/tools.md`).

### Architecture options (all off by default; exact parameter accounting)
- Layer patterns mixing global, sliding-window, MLA and Gated DeltaNet
  attention, in the control core and the superstacks.
- mHC hyper-connection residual streams; DyT / Derf normalisation-free norms;
  parallel and function-preservingly growable blocks; low-rank dynamic weights.
- Per-layer embeddings and hashed n-gram embeddings (cache-carried).
- M-RoPE grid positions for image / video / camera; YaRN context extension.
- A camera modality; 3D Gaussian splat scenes, persistent world state and
  action-driven rollout.
- Realtime streaming via control-core early exit and lossless
  self-speculative decoding.
- The 8B rung and XLA / TPU training.

### Training recipe
- Warmup-stable-decay schedule, Muon, z-loss, optional loss balancing, weight
  EMA, embeddings excluded from weight decay, fused optimizer steps, document
  packing, a subword tokenizer sized so no embedding row is dead, and a
  pre-flight data-budget audit.

### Fixes
- Sliding-window caches kept every key (and a negative slice start); KV-cache
  accounting was per-model rather than per-layer-kind; embeddings were weight
  decayed; context-budget off-by-one; n-gram history lost across cache steps;
  the typed head ran its trunk twice; the shipped checkpoint failed strict
  loading; config round-trips dropped tokenizer fields; M-RoPE failed on the
  meta device; the T4 materialised the full attention score matrix.

### Found by a CPU smoke run of chat-34m (30 steps, 2 rounds)
- Chat sources underfilled and aborted the run; they now top up from each other.
- Chat data and chat sessions were byte-encoded under a subword preset, which
  overflowed the window; the tokenizer now flows through data, chat and CLI.
- Streaming Gutenberg peaked near 12 GB of host RAM (out of default mixes now);
  OpenR1 read unused columns (~4 GB, now ~0.7 GB).
- A 32 x 512 batch did not fit in memory; presets now use micro-batch 8 with
  accumulation, and CPU/ROCm out-of-memory errors trigger the halving retry.
- Result: loss 9.05 -> 7.2 (ln 8192 = 9.01 at init); the checkpoint loads in
  `iridium chat` and decodes through its own tokenizer. A pipeline check, not
  a quality result.

### Data
- Chat: smol-smoltalk, OpenR1-Math solutions, Dolly, OASST by default;
  UltraChat and Tulu 3 opt-in (`--max-data`). Text: Cosmopedia stories and
  textbooks, FineMath. Tools: glaive and Hermes function calling (Apache 2.0)
  plus exactly graded synthetic tasks. Every source records its licence.

### Packaging
- `pyproject.toml` (private: no licence, `Private :: Do Not Upload`),
  `iridium.__version__`, a CPU CI workflow, a docs index and a free-tier
  training guide. Notebooks rebuilt around presets.

## Pre-1.0

The ORRERY specification, its migration to the routed Iridium-1 design, the
trained 0.8 M first slice, the 34 M nano phase-1 checkpoint, the 1.00 B test
rung and inference server, and the studio notebooks. See `git log` and
`docs/decisions.md`.
