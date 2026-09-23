# Changelog

## 1.0.0 — 2026-09

The first release meant to be *trained by someone else*: every capability is
reachable from a named preset, sized for free compute, and labelled with how
much has actually been established. **Status: verified in theory, untrained.**

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

### Packaging
- `pyproject.toml` (private: no licence, `Private :: Do Not Upload`),
  `iridium.__version__`, a CPU CI workflow, a docs index and a free-tier
  training guide. Notebooks rebuilt around presets.

## Pre-1.0

The ORRERY specification, its migration to the routed Iridium-1 design, the
trained 0.8 M first slice, the 34 M nano phase-1 checkpoint, the 1.00 B test
rung and inference server, and the studio notebooks. See `git log` and
`docs/decisions.md`.
