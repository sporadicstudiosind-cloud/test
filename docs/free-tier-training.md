# Training Iridium on free compute

Everything in the 1.0 preset list is sized so that a person with no budget can
train it. This page says what each free tier gives you, how long each preset
should take there, and what the hosted "free AI" services can and cannot do.

> **Read this first.** Every preset is *verified in theory*: it builds, its
> parameter count matches the accounting formula, and the test-suite invariants
> hold for it. Only `chat-34m` has a recorded full run (validation 1.19 bits per
> byte on text, 0.87 on chat), and nothing has been evaluated. The hour figures
> below are arithmetic, not measurements.

## The tiers

| Tier | Device | Precision used | Quota (as published, subject to change) |
|---|---|---|---|
| Colab free | T4, 16 GB (≈15 usable) | fp16 autocast | sessions of a few hours, availability varies |
| Colab free TPU | TPU v5e-1, 16 GB HBM | bf16 | availability varies (replaced v2-8 in 2025) |
| Kaggle | 2×T4 (both train at once), or P100 16 GB | fp16 autocast | ~30 GPU-hours/week (floating), 12 h sessions |
| Kaggle TPU | TPU v5e-8 | bf16 | 20 h/week, 9 h sessions; often oversubscribed |

The T4 and P100 have no bf16. The trainer runs them under fp16 autocast with
gradient scaling (see [`gpu.md`](gpu.md)); the hour estimates still divide by
the fp32 peaks (8.1 and 9.3 TFLOP/s), so a real GPU run can beat them. The TPU
path is far faster on paper, but only `chat-34m` on a GPU has a recorded run;
the XLA path has not been timed.

**Two T4s.** On Kaggle choose *GPU T4 x2*. `iridium train` starts one process
per GPU and averages their gradients each step, so the batch and schedule are
those of a one-GPU run at about half the time per step. `--gpus 1` turns it
off. This is verified with two CPU processes, not yet timed on GPUs.

## Presets and estimated hours

`python -m iridium presets` prints this table from the live configs:

```
preset        pri       params    body  embed   tokens tok/body    T4 h  P100 h  v5e h  status
----------------------------------------------------------------------------------------------
chat-34m        1   36,038,426   30.1M   2.1M     328M       11     4.1     3.6    0.2  trained once (val 1.19 bpb text, 0.87 chat); not evaluated
chat-36m        1   46,180,869   35.8M   5.9M     716M       20    14.3    12.5    0.6  verified in theory
50m             1   52,719,370   45.1M   3.2M     328M        7     7.0     6.1    0.3  verified in theory
iridium-50m     1   60,273,479   50.4M   7.9M    5000M       99   102.5    89.3    4.2  verified in theory
100m            1  108,304,153   94.5M   8.4M     655M        7    21.1    18.4    0.9  verified in theory
chat-100m       1  108,304,153   94.5M   8.4M     655M        7    21.1    18.4    0.9  verified in theory
500m            1  497,686,932  449.5M  33.6M    1966M        4   458.6   399.4   18.9  verified in theory
1b              1  997,060,000  933.5M  42.0M    3932M        4  1871.7  1630.2   77.0  verified in theory
tools-100m      2  108,304,153   94.5M   8.4M     655M        7    21.1    18.4    0.9  verified in theory
omni-100m       3  111,586,585   94.5M   8.4M     655M        7    21.1    18.4    0.9  verified in theory
stem-100m       4  108,304,153   94.5M   8.4M     492M        5    15.8    13.8    0.6  verified in theory
world-100m      5  111,597,342   94.5M   8.4M     328M        3    10.5     9.2    0.4  verified in theory; no camera-posed training data yet
modern-744m     -  743,752,454  651.4M  67.1M   26214M       40  7055.7  6145.3  290.1  verified in theory
2b              - 2,070,812,835 1972.6M  58.7M    7864M        4  7952.2  6926.1  327.0  verified in theory
4b              - 4,018,897,967 3880.9M  75.5M   15729M        4  30690.4 26730.3 1261.9  verified in theory
(hours are arithmetic at 30% of published fp32 peak, not measurements; chat-34m's recorded run took about half its figure)
```

**How they are computed:** training FLOPs = 3 × forward FLOPs per token
(forward + backward) × tokens, divided by published fp32 peak × an assumed 30%
utilisation. They can miss either way: chat-34m's recorded run (44k tokens/s
under fp16 autocast) took about half its figure, while Python overhead, data
loading and small matmuls push the other way. `iridium-50m`'s row is for one
card at fp32; its own estimate is about 15 hours on two T4s
([`iridium-50m.md`](iridium-50m.md)), also unmeasured. The
TPU figure assumes XLA compiles one static shape and never recompiles — any
shape churn can erase the TPU's advantage entirely.

**How much data is that?** The Chinchilla compute-optimal ratio is about 20
tokens per parameter (Hoffmann et al., 2022). chat-34m's 328 M tokens is ~11
per body parameter, chat-100m's 655 M is ~7: *under* compute-optimal,
deliberately, because that is what fits in free quota. chat-36m sits at 20 and
iridium-50m at about 100. Small open models that talk
well were trained at 1,000+ tokens/parameter. Expect a fluent-ish babbler at
these budgets, not an assistant. `iridium train --preset X --dry-run` prints
the full budget audit.

## The workflow

```bash
python -m iridium presets                                   # sizes, budgets, estimates
python -m iridium data prepare --preset chat-34m            # tokenize to disk, once
python -m iridium train --preset chat-34m --data data       # train from the shards
python -m iridium train --preset chat-34m --data data --resume auto   # newest complete round
python -m iridium train --preset iridium-50m --data data --resume auto --time-budget-hours 11 --keep-rounds 1
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

**Scoreboard.** With `--data`, training evaluates the held-out shards every
5% of steps and logs `val_*_loss` and `val_*_bpb` (bits per byte: loss
converted by the held-out text's own bytes-per-token, so vocabularies of any
size are comparable).

**Learning rate and batch.** `describe` prints DeepSeek LLM's fitted optimum
for the preset's compute (`lr = 0.3118 C^-0.125`, `B = 0.2920 C^0.3271`
tokens). The presets sit at 6-10x smaller batches and 2-3x lower learning
rates -- consistent with each other, but not the fit. `--hparams deepseek`
runs the fit instead (same tokens, larger batch by accumulation, no extra
memory); comparing the two on bits per byte is the cheapest useful sweep.

**Optimizers.** `eager_adamw` (fp32, 16 bytes/param of training state),
`adamw8` (8-bit block-wise moments, ~10 bytes/param, runs on every backend:
what lets `1b` fit a 16 GB device), and `muon` (~12 bytes/param).

Without `--data`, training falls back to the older in-memory rounds path,
which streams and tokenizes inside the training process.

Recommended order: **chat-34m** (a single session; the smoke test that the
pipeline works end to end, and the one preset with a recorded run) →
**iridium-50m** (about two Kaggle sessions on two T4s, by estimate). The
`-100m` presets follow the 1.0 priorities after that: chat-100m (several weeks
of Kaggle quota) → tools-100m initialised from it → omni / stem / world.

**Sessions that end.** Every round writes a checkpoint and then `latest.json`,
so `--resume auto` never picks up a half-written file. `--time-budget-hours`
stops cleanly before a host's limit; `--keep-rounds 1` keeps disk use flat.

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
