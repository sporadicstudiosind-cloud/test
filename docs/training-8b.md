# The 8B rung: what it is and why it is not a free-tier run

`iridium-1-8b` is an **8,073,035,416-parameter** configuration of the routed
architecture: a 24-layer control core, ten 28-layer bridge-attending
superstacks, top-2 routing, three ponder loops, one spectral stack, and a
65,520-token subword vocabulary. It lives in `iridium/config.py`, so every
figure below is computed from the same tensor geometry as the runnable rungs.

```bash
python -m iridium report 8b      # costs it without allocating a single tensor
```

## Why no free tier can train it

Full BF16 Adam keeps BF16 weights and gradients plus FP32 master weights and
two FP32 moment buffers: **16 bytes per parameter**, or **120.3 GiB** for this
model before activations, temporaries, data workers or checkpoints. The
largest free device (a 16 GB T4, P100 or TPU v5e-1) holds about an eighth of
that. Batch size changes activation memory only; it does not touch the 120 GiB
of model and optimizer state. Swap is not a training implementation.

## What has to exist before an honest 8B run

1. A distributed execution path for the superstack dispatcher, with sharded
   optimizer state and checkpoints. `iridium/parallel/plan.py` costs a 4-D
   parallel layout; it does not execute one.
2. A licensed, versioned multimodal corpus at the scale an 8B model needs
   (hundreds of billions of tokens), with held-out splits.
3. A reproducible recipe: global batch, sequence-length curriculum, optimizer,
   precision, activation checkpointing, seed, mixture, restart policy.
4. Multi-accelerator hardware with enough aggregate memory after sharding, and
   an end-to-end test of a sharded forward, backward, step and restart.
5. Held-out evaluations showing a trained checkpoint beats stated baselines.

Until then the rung is a costed design point, and nothing may claim an 8B
checkpoint exists. The free-tier path is the presets (`python -m iridium
presets`), which top out around 110M parameters.

## TPU support

The trainer runs on XLA devices: the flow-matching noise uses torch_xla's own
RNG (`generator_for` returns `None` on XLA), optimizer steps go through
`xm.optimizer_step` so each step ends a lazy graph, and checkpoints are written
with `xm.save`. These branches are **not exercised by the test suite** — there
is no TPU in CI — so treat the first TPU run as the test.
