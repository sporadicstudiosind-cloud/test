# Iridium 1.0

**A routed omnimodal model family you can train on free compute.** It is
built to talk and reason first, to use tools second, and then — in that
order — to handle images, audio, video and geometry, physics and STEM, and
camera-posed 3D worlds.

> **Status: one small run recorded, nothing evaluated.** Every preset below
> builds, its parameter count matches the accounting formula exactly, and the
> test suite's invariants hold for it (cached decoding equals the full forward
> pass, growth preserves the function, gradients are finite). `chat-34m` has
> one recorded full run (validation 1.19 bits per byte on text, 0.87 on chat).
> `iridium-50m`, the current target, is built and tested on CPU and **not yet
> trained**. Nothing here is a claim about how well a trained Iridium talks,
> calls tools or sees. The bundled weights are a pre-1.0 34 M checkpoint
> trained on synthetic tasks.
>
> All rights reserved. Not licensed for redistribution.

---

## Train `iridium-50m` (the current target)

Four specialist stacks (maths, chat, code, science) around a small shared
core, with file, code and web tools and basic image and audio input. Details,
limits and evidence: [`docs/iridium-50m.md`](docs/iridium-50m.md).

**On Kaggle:** open `notebooks/iridium_studio_kaggle.ipynb`, attach a
`GITHUB_TOKEN` secret, and run it three or more times: once on CPU to write
the data, then on **GPU T4 x2** until the log says the run finished. It
resumes by itself. [`docs/kaggle-agent-runbook.md`](docs/kaggle-agent-runbook.md)
has the steps.

**From a shell:**

```bash
pip install -e ".[data]"                                   # torch, numpy, datasets, tokenizers
python -m iridium train --preset iridium-50m --dry-run     # size, budget, time estimate
python -m iridium data prepare --preset iridium-50m        # tokenize once, to disk (CPU is fine)
python -m iridium train --preset iridium-50m --data data --resume auto   # uses every GPU
python -m iridium battery --checkpoint runs/iridium-50m/iridium-50m-final.pt
python -m iridium specialists --checkpoint runs/iridium-50m/iridium-50m-final.pt --data data
python -m iridium agent --checkpoint runs/iridium-50m/iridium-50m-final.pt --workspace ./sandbox
```

A pipeline check before spending quota: add `--tokens 2e6` to `data prepare`
and `--tokens 2e6 --steps 20` to `train`; it runs in minutes on CPU.

What to watch in the log: the `val_*_bpb` lines should fall (lower is better),
and `tok/s` tells you how long the run will really take. The ~15 hour estimate
for two T4s has not been measured.

## The other presets

`python -m iridium presets` prints this from the live configs:

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

`chat-34m` is the cheapest end-to-end check. `chat-36m` is its rebalanced,
compute-optimal successor. The `-100m` presets follow the 1.0 priority order:
talking, tools, omnimodal, STEM, world model. `500m` and up are costed, not
free-tier trainable. Budgets other than `iridium-50m`'s are below
compute-optimal on purpose, because that is what fits in free quota; expect
plausible prose from those, not an assistant.
[`docs/free-tier-training.md`](docs/free-tier-training.md) has the tiers,
quotas and the reasoning behind the numbers.

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

Every token goes through the core, then through one or two superstacks, then
through the core again; no token skips the stacks. In `iridium-50m` the stacks
are subject specialists and the router is supervised by each training window's
subject.

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
- **Hardware** — CUDA, ROCm, XLA/TPU; one training run across several GPUs
  (one process per GPU, gradients averaged); several runs on one GPU.
- **Thinking effort** — `iridium chat --think N` sets the ponder loops; the
  model may move up to ten either way (`--think-spread`).

## Other commands

```bash
python -m pytest                                 # CPU suite
python -m iridium ladder                         # every rung, 34 M to 8 B and beyond
python -m iridium report modern --verify         # accounting checked against the modules
python -m iridium plan base --gpus 1024          # 4-D parallelism and its cost model
python -m iridium evaluate --help                # graded accuracy on held-out splits
python -m iridium chat                           # the bundled pre-1.0 checkpoint
python -m iridium chat --checkpoint MODEL.pt --think 3   # more ponder loops
python -m iridium export --help                  # a smaller inference-only checkpoint
```

## Layout

```
iridium/
  presets.py   the 1.0 presets and free-tier estimates
  config.py    the scaling ladder; executable parameter and memory accounting
  model/       control core, router, superstacks, attention variants, heads
  codecs/      typed spans and the omnimodal embedding bank
  training/    trainer, run_preset, multi-GPU (parallel.py), data mixtures, budget audit
  data/        licensed text, chat, tool and agent corpora; subjects (domains.py)
  evaluation/  test battery, routing and specialist checks
  alignment/   written principles and an output screen (no training data yet)
  runtime/     device backends, streaming, tools, constrained decoding, serving
  world/       cameras, splats, world state, rollout
  physics/     spectral solvers and verifiers used for exactly-graded data
  agency/ quant/ parallel/ contracts/ generation/ memory/ research/
notebooks/     Colab, Kaggle, Jupyter and TPU notebooks
docs/          see docs/README.md
tests/         unit / integration / scientific
```

## Principles carried over from before 1.0

**Evidence is earned, not labelled.** "Verified in theory", "trained" and
"evaluated" are different claims; this README makes the first, and the second
for one small preset.
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

- That any preset produces a good model. One has been trained; none has been evaluated.
- That the four stacks of `iridium-50m` will specialise at full size. A toy
  experiment says the scheme works; `iridium specialists` is the real check.
- That it scales. Rungs past 100 M are costed, not tested by training.
- That the synthetic families transfer to real-world skill.
- That the hour estimates hold on your session. They are arithmetic.

See [`CHANGELOG.md`](CHANGELOG.md) for what 1.0 changed and
[`docs/README.md`](docs/README.md) for everything else.
