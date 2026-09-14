# Iridium-1 8B training boundary

`iridium-1-8b` is an **8,073,027,736-parameter configuration** of the routed
architecture. It has a 24-layer control core, ten 28-layer bridge-attending
superstacks, top-2 causal routing, three ponder loops, and one spectral stack.
It is a real configuration in `iridium/config.py`, so its parameter and memory
figures are calculated from the same tensor geometry as the smaller runnable
rungs.

## Why this is not a 48 GiB Colab training run

Full BF16 Adam uses, at a minimum, BF16 weights and gradients plus FP32 master
weights and two FP32 moment buffers: **16 bytes per parameter**. For 8.073B
parameters that is **120.3 GiB before activations, temporary tensors, data
workers, or checkpoints**. A CPU-only 48 GiB process cannot hold that state.
Changing batch size only reduces activation memory; it does not remove the
120.3 GiB model/optimizer state. CPU swap would make the process impractical
and is not a training implementation.

The repository also does not yet implement the distributed dispatcher,
parameter sharding, optimizer sharding, checkpoint partitioning, or dataset
pipeline required to train this architecture across workers. The existing
`build_corpus` function produces synthetic, verifiable task families for
mechanism tests; it is not a corpus capable of making an 8B general model
work. Therefore this project must not claim that an 8B checkpoint has been
trained or evaluated.

## What is implemented

```bash
python -m iridium report 8b
```

This costs the full architecture without materialising its tensors. The Colab
notebook reports the requirement and deliberately aborts if `RUNG = '8b'` is
selected, rather than beginning an allocation that will fail or silently use
swap.

## What has to exist before an honest 8B run

1. A distributed execution implementation for the superstack dispatcher and
   sharded optimizer/checkpoint state — the planner alone is not execution.
2. A versioned, licensed multimodal training corpus with tokenization and
   codecs that match every declared input/output modality, plus provenance and
   held-out evaluation splits.
3. A reproducible training recipe: global batch, sequence-length curriculum,
   optimizer, precision, activation checkpointing, seed, data mixture, and
   checkpoint/restart policy.
4. Multi-worker hardware with enough aggregate accelerator memory after
   sharding, and an end-to-end test that exercises a real sharded forward,
   backward, optimizer step, and restart.
5. Held-out evaluations that demonstrate a trained checkpoint exceeds specified
   baselines. Until then, the only trained and evaluated results remain the
   documented small synthetic slice.
