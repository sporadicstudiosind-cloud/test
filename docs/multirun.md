# Running several trainings at once (`iridium.training.multirun`)

## The claim, stated plainly

Co-locating K training runs on one GPU speeds things up **only when a single
run leaves the device idle** — a small model, a small batch, a data loader
that is the actual bottleneck, or a hyperparameter sweep at a scale where one
GEMM does not fill the SMs. If a single run already saturates the device,
co-locating K runs gives each of them roughly `1/K` of the compute — worse
than `1/K`, once you add context-switch and memory-pressure overhead — so
**total throughput goes down, not up.**

Nothing in this module is allowed to assert a speedup. `plan()` estimates
whether K runs fit in memory and says explicitly when it has no basis for a
utilisation claim. `benchmark()` times 1 run against K and reports whatever
it measures, labelled with the device it ran on. Every number this module
produces on this machine (no GPU) is a **CPU** number, offered because that
is what CI has, not a stand-in for a GPU number.

## Three strategies

| Strategy | Where the win comes from | What it costs | Right regime |
|---|---|---|---|
| `"processes"` | K independent OS processes, each fully isolated | A CUDA context per process (~300-500MB, before a single weight loads) + K independent reads of the corpus | Different code paths per run, or a sweep large enough that one run's segfault must never touch the others |
| `"streams"` | K models in one process, one per `torch.cuda.Stream`, round-robin, one shared `BatchLoader` | One CUDA context, one data pipeline, but still K full sets of weights/gradients/optimizer state in device memory | Many small variants of the same code path (LR/seed sweep) where "processes"'s per-run overhead would dwarf the actual compute |
| `"vmap"` | K parameter sets stacked into one big batched op via `torch.func.functional_call` + `torch.vmap` | Only works when every replica shares identical tensor shapes | A hyperparameter sweep over the *same* architecture — this is the only strategy that turns K small GEMMs into one large, efficient GEMM, so it is the only one that can multiply throughput rather than merely avoid wasting it |

### Why `vmap` cannot run the routed Iridium-1 model

`vmap` requires every element along the batched axis to execute the *same*
sequence of operations. Iridium-1 breaks that twice:

* the router's **top-k dispatch** sends each token to a different, learned
  subset of stacks — a data-dependent gather/scatter whose shape and target
  depend on a tensor's runtime *value*, not just its shape;
* the **ponder loop** runs a different number of iterations per example,
  decided by a learned halting signal — a data-dependent trip count.

Either one is exactly the control flow `vmap` cannot lift: it would raise on
the first data-dependent Python `if`/index, or — worse, if that branch
happens to be expressed as a tensor mask instead of a Python conditional —
silently apply *one* replica's routing or halting decision to every replica
sharing that batch position. A wrong answer with no error is worse than a
refusal, so `check_vmap_compatible()` detects `Iridium1` by type and refuses
with this reason rather than attempting a trace. `vmap` in this module is
implemented against a plain reference module (any architecture without
data-dependent control flow) for sweeps where K variants share every tensor
shape and differ only in initialisation, learning rate, or data order.

If the routed model itself needs to be swept, use `"processes"` or
`"streams"` — both handle arbitrary forward passes because neither requires
a single shared trace.

### Stream-strategy synchronisation, and why each point is there

Getting this wrong produces silent nondeterminism — a run occasionally reads
stale data with no error, because CUDA's async execution model does not fail
loudly on a missing dependency. `run_streams` synchronises at exactly two
places, both documented in the function's docstring:

1. **Before a run consumes its batch**, if it was produced by the shared
   loader on the default stream: `run_stream.wait_stream(default_stream)` —
   a stream-to-stream event wait, not a device-wide synchronize, so it costs
   nothing for the other runs.
2. **Once, after the whole loop**, `torch.cuda.synchronize()` — because a
   caller reading final weights (a checkpoint, an equivalence check) needs
   the device to have actually finished, and that is not implied by the
   per-step `float(loss)` calls, which only sync the loss tensor.

No synchronisation happens *between* runs mid-loop, on purpose: two runs
touch disjoint parameters and disjoint optimizer state, so letting the
device schedule their kernels concurrently is the entire benefit of using
separate streams. Adding a wait there would silently serialise the streams
and reduce this strategy to `"processes"`'s throughput at `"streams"`'s
memory cost.

On CPU — this repository's only available target — every stream is `None`
and every wait/synchronize above is a no-op, so round-robin execution is
plain sequential Python. `tests/unit/test_multirun.py` uses exactly this
fact: it trains one model interleaved with a sibling and checks its final
weights are bit-exact against training that same model alone on the same
sub-sequence of batches. On a GPU the same code path additionally has real
overlap to get right, which is why the synchronisation points above are
unconditional rather than only exercised when `torch.cuda.is_available()`.

## `plan()`: memory arithmetic, and an honest gap

`plan()` reuses `iridium.runtime.memory.plan_training`'s per-run footprint
estimate (weights + gradients + optimizer state + estimated activations) and
divides the available budget by it — that answers "how many replicas fit."
It does **not** answer "will running them together be faster," which needs a
compute measurement `plan()` has no way to make from a parameter count alone.
Pass `measured_saturation_tokens` from a prior `benchmark()` run on the same
device to let it warn concretely ("your tokens/step already meets the point
this device stopped scaling"); without it, `plan()` says plainly that it has
no utilisation basis, rather than guessing a threshold that would be
precise-looking and wrong for most GPUs.

## `benchmark()`: measure, don't assert

`benchmark(make_step, k, steps, device)` times one run for `steps`
iterations, then times `k` independent runs interleaved round-robin for
`steps` iterations each, and reports the speedup actually observed. It runs
unmodified on CPU (this container's only target), where it typically shows
speedup at or below 1.0 — CPU has no independent per-stream execution units
the way a GPU has SMs, so interleaving CPU-bound steps in one process is at
best indistinguishable from running them sequentially, and at worst slower
for the added bookkeeping. That is the expected, correct result on this
hardware, not a bug in the benchmark.

## Interface summary

```python
from iridium.training.multirun import (
    plan, benchmark,                                   # planning + measurement
    ProcessRun, run_processes, ensure_shared_cache,     # strategy 1
    StreamRun, run_streams,                             # strategy 2
    check_vmap_compatible, stack_replicas,
    build_vmap_ensemble, VmapEnsemble,                  # strategy 3
)
```

See the module's own docstring and each function's docstring for the full
contract; this file exists to state the tradeoffs and the one thing that
must never be asserted without measurement.
