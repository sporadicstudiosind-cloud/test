# Iridium 1.0

**A routed omnimodal model family you can train on free compute.** It is
built to talk and reason first, to use tools second, and then — in that
order — to handle images, audio, video and geometry, physics and STEM, and
camera-posed 3D worlds.

> **Status: verified in theory, not trained.** Every preset below builds, its
> parameter count matches the accounting formula exactly, and the invariants
> in the ~1,100-test suite hold for it (cached decoding equals the full
> forward pass, growth preserves the function, gradients are finite). **No
> preset has been trained at its recipe yet**, so nothing here is a claim
> about how well a trained Iridium talks, calls tools or sees. The only
> bundled weights are a pre-1.0 34 M checkpoint trained on synthetic tasks.
>
> **Private.** All rights reserved. Not licensed for redistribution.

---

## Train one

```bash
pip install -e ".[data]"                     # torch, numpy, pyyaml, jsonschema, datasets
python -m iridium presets                    # what can be trained, and how long it takes
python -m iridium train --preset chat-34m --dry-run
python -m iridium train --preset chat-34m
python -m iridium chat --checkpoint runs/chat-34m/chat-34m-final.pt
```

Or open a notebook in [`notebooks/`](notebooks/) on Colab or Kaggle; it runs
the same commands, with checkpoints per round so a dropped session costs one
round, not the run.

| preset | for | params | tokens | Colab T4 | Kaggle P100 | TPU v5e-1 |
|---|---|---|---|---|---|---|
| `chat-34m` | talking (start here) | 36 M | 328 M | ~4 h | ~4 h | ~0.2 h |
| `chat-100m` | talking + reasoning | 108 M | 655 M | ~21 h | ~18 h | ~0.9 h |
| `tools-100m` | tool calling | 108 M | 655 M | ~21 h | ~18 h | ~0.9 h |
| `omni-100m` | omnimodal | 112 M | 655 M | ~21 h | ~18 h | ~0.9 h |
| `stem-100m` | physics / STEM | 108 M | 492 M | ~16 h | ~14 h | ~0.6 h |
| `world-100m` | world model | 112 M | 328 M | ~11 h | ~9 h | ~0.4 h |
| `modern-744m` | every option at scale | 744 M | 26 B | costed only | | |

Hours are **optimistic arithmetic** (3 × forward FLOPs, 30% of published
peak), not measurements. The token budgets are below compute-optimal for
these sizes on purpose — that is what fits in free quota — so expect a small
model that writes plausible prose, not an assistant.
[`docs/free-tier-training.md`](docs/free-tier-training.md) has the tiers,
quotas, the reasoning behind the numbers, and what Groq / NVIDIA NIM style
hosted APIs can do here (generate and grade data, subject to each model's
licence; not train).

## What the architecture is

```
input spans ─► typed codecs (text, image, audio, video, camera, geometry, fields…)
            ─► control core, stage I          every token, every layer
            ─► macro-router (top-k)           picks superstacks per token
            ─► superstacks                    deep specialist stacks, cross-attending the core
            ─► control core, stage II
            ─► heads                          text, flow-matching media, typed System-1 head
            ↺  ponder loop                    the core decides how many passes a token gets
```

On top of that skeleton, each piece is a config switch with exact parameter
and cache accounting, all **off by default** and documented with its
evidence and cost in [`docs/architecture-options.md`](docs/architecture-options.md):

- **Attention mix per layer** — global, sliding-window, MLA (latent KV) and
  Gated DeltaNet (linear, constant-size state), in the core and the stacks.
  The `modern` rung uses this to fit a 1M-token cache.
- **Residuals and norms** — mHC hyper-connection streams, DyT / Derf,
  parallel blocks, function-preserving block growth, low-rank dynamic weights.
- **Embeddings** — per-layer embeddings, hashed n-gram embeddings.
- **Positions** — M-RoPE grid positions for media and cameras; YaRN extension.
- **Speed** — realtime streaming by exiting early inside the control core,
  with lossless self-speculative decoding; typed System-1 routing for cheap
  tokens.
- **Tools** — a call format on reserved control tokens and schema-constrained
  decoding ([`docs/tools.md`](docs/tools.md)).
- **World model** — a camera modality, differentiable 3D Gaussian splats,
  persistent world state and action-driven rollout
  ([`docs/world-model.md`](docs/world-model.md)).
- **Hardware** — CUDA, ROCm, XLA/TPU; several runs on one GPU.

## Other commands

```bash
python -m pytest                                 # CPU suite
python -m iridium ladder                         # every rung, 34 M to 8 B and beyond
python -m iridium report modern --verify         # accounting checked against the modules
python -m iridium plan base --gpus 1024          # 4-D parallelism and its cost model
python -m iridium evaluate --help                # graded accuracy on held-out splits
python -m iridium chat                           # the bundled pre-1.0 checkpoint
```

## Layout

```
iridium/
  presets.py   the 1.0 presets and free-tier estimates
  config.py    the scaling ladder; executable parameter and memory accounting
  model/       control core, router, superstacks, attention variants, heads
  codecs/      typed spans and the omnimodal embedding bank
  training/    trainer, run_preset, data mixtures, tokenizer, budget audit
  data/        licensed text and tool corpora
  runtime/     device backends, streaming, tools, constrained decoding, serving
  world/       cameras, splats, world state, rollout
  physics/     spectral solvers and verifiers used for exactly-graded data
  agency/ quant/ parallel/ evaluation/ contracts/ generation/ memory/
notebooks/     Colab, Kaggle, Jupyter and TPU notebooks
docs/          see docs/README.md
tests/         unit / integration / scientific
```

## Principles carried over from before 1.0

**Evidence is earned, not labelled.** "Verified in theory", "trained" and
"evaluated" are different claims; this README makes only the first.
[`docs/capability-register.md`](docs/capability-register.md) tracks the rest
per capability.

**Exact arrays beneath learned representations.** Scientific numbers stay in
typed storage and round-trip bit-exactly; a hidden vector is never the only
record of a result.

**Conservation by construction.** A learned correction applied to oriented
shared faces cannot invent mass; the early trained slice held 9.5e-09 relative
drift where its unconstrained control drifted 1.8%
([`docs/first-slice.md`](docs/first-slice.md)). The same slice also collapsed
out of distribution, which is why solvers remain in the loop.

## What this does not claim

- That any preset produces a good model. None has been trained yet.
- That it scales. Rungs past 100 M are costed, not tested by training.
- That the synthetic families transfer to real-world skill.
- That the hour estimates hold on your session. They are lower bounds.

See [`CHANGELOG.md`](CHANGELOG.md) for what 1.0 changed and
[`docs/README.md`](docs/README.md) for everything else.
