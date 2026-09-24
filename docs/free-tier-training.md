# Training Iridium on free compute

Everything in the 1.0 preset list is sized so that a person with no budget can
train it. This page says what each free tier gives you, how long each preset
should take there, and what the hosted "free AI" services can and cannot do.

> **Read this first.** Every preset is *verified in theory*: it builds, its
> parameter count matches the accounting formula, and the test-suite invariants
> hold for it. **None has been trained at its recipe yet.** The hour figures
> below are arithmetic, not measurements.

## The tiers

| Tier | Device | Precision used | Quota (as published, subject to change) |
|---|---|---|---|
| Colab free | T4, 16 GB (≈15 usable) | fp32 | sessions of a few hours, availability varies |
| Colab free TPU | TPU v5e-1, 16 GB HBM | bf16 | availability varies (replaced v2-8 in 2025) |
| Kaggle | P100 16 GB, or 2×T4 | fp32 | ~30 GPU-hours/week (floating), 12 h sessions |
| Kaggle TPU | TPU v5e-8 | bf16 | 20 h/week, 9 h sessions; often oversubscribed |

The T4 and P100 have no bf16, and this codebase trains in fp32 on them rather
than fp16 because routed softmaxes overflow fp16 (see [`gpu.md`](gpu.md)). That
is why their peaks are the fp32 ones (8.1 and 9.3 TFLOP/s): roughly 20x slower
than a v5e chip in bf16. **If a TPU is available, use it.**

## Presets and estimated hours

`python -m iridium presets` prints this table from the live configs:

| preset | priority | params | tokens | T4 h | P100 h | v5e-1 h |
|---|---|---|---|---|---|---|
| chat-34m | 1 talking | 36.0 M | 328 M | 4.1 | 3.6 | 0.2 |
| chat-100m | 1 talking | 108.3 M | 655 M | 21.1 | 18.4 | 0.9 |
| tools-100m | 2 tools | 108.3 M | 655 M | 21.1 | 18.4 | 0.9 |
| omni-100m | 3 omnimodal | 111.6 M | 655 M | 21.1 | 18.4 | 0.9 |
| stem-100m | 4 STEM | 108.3 M | 492 M | 15.8 | 13.8 | 0.6 |
| world-100m | 5 world | 111.6 M | 328 M | 10.5 | 9.2 | 0.4 |
| modern-744m | costed only | 743.8 M | 26.2 B | 7,056 | 6,145 | 290 |

**How they are computed:** training FLOPs = 3 × forward FLOPs per token
(forward + backward) × tokens, divided by published peak × an assumed 30%
utilisation. Treat them as a lower bound. Small models on free GPUs routinely
reach well under 30% (Python overhead, data streaming, small matmuls), and the
TPU figure assumes XLA compiles one static shape and never recompiles — any
shape churn can erase the TPU's advantage entirely.

**How much data is that?** The Chinchilla compute-optimal ratio is about 20
tokens per parameter (Hoffmann et al., 2022). chat-34m's 328 M tokens is ~9
tokens/parameter, chat-100m's 655 M is ~6. Both are *under* compute-optimal,
deliberately: that is what fits in free quota. Small open models that talk
well were trained at 1,000+ tokens/parameter. Expect a fluent-ish babbler at
these budgets, not an assistant. `iridium train --preset X --dry-run` prints
the full budget audit.

## The workflow

```bash
python -m iridium presets                                   # sizes, budgets, estimates
python -m iridium data prepare --preset chat-34m            # tokenize to disk, once
python -m iridium train --preset chat-34m --data data       # train from the shards
python -m iridium train --preset chat-34m --data data --resume runs/chat-34m/chat-34m-round2.pt
python -m iridium train --preset 1b --tokens-per-param 20 --data data   # any budget
```

**Prepare once, train from disk.** `data prepare` streams the preset's text,
chat and tool sources one source at a time, tokenizes them with the Rust
tokenizer, and writes flat token files (`.bin` ids, `.mask` targets, `.idx`
item offsets). Training memory-maps them, so RAM use does not grow with the
token budget, and the network is used once. Run `prepare` on a free *CPU*
session (Kaggle CPU notebooks do not spend GPU quota) and attach the output to
the GPU/TPU run. Synthetic families are generated on demand from a seeded
index and are never stored.

**Budgets.** `--tokens 2e9` or `--tokens-per-param 20` (Chinchilla) resizes
any preset's run; `data prepare --plan` prints how many items that needs.

**Optimizers.** `eager_adamw` (fp32, 16 bytes/param of training state),
`adamw8` (8-bit block-wise moments, ~10 bytes/param, runs on every backend:
what lets `1b` fit a 16 GB device), and `muon` (~12 bytes/param).

Without `--data`, training falls back to the older in-memory rounds path,
which streams and tokenizes inside the training process.

Recommended order, following the 1.0 priorities: **chat-34m** (a single
session; the smoke test that the pipeline works end to end) → **chat-100m**
(several weeks of Kaggle quota) → **tools-100m** initialised from chat-100m →
omni / stem / world.

## Groq, NVIDIA NIM and other hosted "free" APIs

These are **inference** services: they run someone else's model on their
hardware. They cannot train or fine-tune Iridium. What they can do here:

- **Generate synthetic data** — chats, tool-use traces, worked STEM solutions.
- **Grade outputs** — as a judge for the evaluation harness, where no exact
  checker exists.

Two caveats that matter more than the rate limits:

1. **Licences.** Whether a hosted model's outputs may be used to train another
   model is decided by that model's licence and the provider's terms, and they
   differ. Some prohibit it; some permit it with conditions (attribution,
   naming). Check before generating a training set, and record the source
   model in the data manifest.
2. **Quality ceiling and contamination.** Distilled data inherits the
   teacher's errors and style. The families this repo generates itself
   (fluid, channel, quantity, tool tasks) are *exactly graded* — prefer them
   wherever a checker exists, and use hosted models for what only language can
   supply.

Free-tier limits on these services change often; check the provider's current
page rather than trusting a number written here.
