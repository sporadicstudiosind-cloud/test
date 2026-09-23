"""Running several trainings on one GPU: three strategies, one honest claim.

The request behind this module is almost always "make training faster by
running more of it at once." That is true exactly when a single run leaves
the device idle — a small model, a small batch, a CPU-bound data loader, or a
hyperparameter sweep at a scale where one GEMM does not fill the SMs/CUs. It
is false, and actively counterproductive, once a single run already saturates
the device: co-locating K runs then means each gets roughly 1/K of the
compute (worse than 1/K, once you add context-switch and memory-pressure
overhead), so total throughput goes *down*, not up. Nothing in this module is
allowed to assert a speedup; `benchmark()` measures one, on whatever device it
is given, and everything else only plans against that measurement or admits it
has none.

Three strategies, because "run several trainings at once" is not one problem:

* ``"processes"`` — K independent OS processes, each with its own CUDA
  context. The obvious way to get isolation (a crash in one cannot corrupt
  another's optimizer state) but the most wasteful: a CUDA context is
  ~300-500MB of device memory *before either process has loaded a single
  weight*, and K processes independently re-read the corpus. Right regime:
  genuinely different code paths per run (different model classes, different
  frameworks), or a sweep large enough that a segfault in one run must not be
  allowed to take the others down with it.

* ``"streams"`` — K models in one process, each on its own ``torch.cuda.Stream``,
  stepped in round-robin, sharing one ``BatchLoader``. One CUDA context, one
  data pipeline. Right regime: many small variants of the same code path
  (a learning-rate or seed sweep) where the K-copies-of-everything cost of
  "processes" would dwarf the actual compute. The obvious wrong alternative
  here is to skip streams and just call each model's `step()` in turn on the
  default stream — that is legal and deterministic, but it serialises the
  actual kernels an underutilising model would otherwise be able to overlap,
  which throws away the one thing this strategy is for.

* ``"vmap"`` — K parameter sets of the *same* architecture stacked along a
  leading axis and run as one batched model via
  ``torch.func.functional_call`` + ``torch.vmap``. The only one of the three
  that turns K small, underutilising GEMMs into one large, efficient GEMM
  instead of K small GEMMs sharing a device — so it is the only strategy that
  can multiply throughput rather than merely avoid wasting it. It requires
  the K variants to share every tensor shape and differ only in
  initialisation, learning rate, or data order, and it cannot vmap dynamic,
  data-dependent control flow (Iridium-1's router top-k dispatch and ponder
  loop both branch on tensor values, a different path per token or example —
  see ``check_vmap_compatible`` for exactly why that breaks vmap's single-trace
  requirement). This module implements it for a plain reference module and
  refuses, with a specific reason, anything it cannot handle — a wrong
  "success" here is a silently-averaged-together sweep, which is worse than
  a refusal.

``plan()`` estimates how many concurrent runs fit in a memory budget and flags
when co-location is unlikely to help; ``benchmark()`` actually times 1 run
against K to measure whether it did. Every number either strategy reports is
labelled with the device it was measured on: a CPU number here is a CPU
number, offered because CI has no GPU, not a stand-in for one.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import torch
import torch.nn as nn

from ..runtime.memory import plan_training


# ==========================================================================
# plan(): memory arithmetic + the "will this even help" check
# ==========================================================================

#: A CUDA context's fixed device-memory tax, independent of anything the
#: process has allocated for tensors. Not measured in this container (no
#: GPU); this is the commonly-cited range for recent CUDA/driver versions and
#: is documented as such — treat it as a planning constant, not data.
CUDA_CONTEXT_GB = 0.4


@dataclass(frozen=True)
class ConcurrencyPlan:
    per_run_gb: float
    context_overhead_gb: float
    available_gb: float
    max_by_memory: int
    recommended: int
    tokens_per_step: int
    saturating: Optional[bool]  # None: no measurement to judge by
    advice: list[str]

    def render(self) -> str:
        lines = [
            f"  per-run footprint     {self.per_run_gb:7.3f} GB",
            f"  CUDA context tax/run  {self.context_overhead_gb:7.3f} GB "
            f"(processes strategy only; streams/vmap pay this once)",
            f"  available             {self.available_gb:7.3f} GB",
            f"  fits by memory        {self.max_by_memory} concurrent run(s)",
            f"  recommended           {self.recommended} concurrent run(s)",
        ]
        lines += [f"  ! {a}" for a in self.advice]
        return "\n".join(lines)


def plan(
    n_params: int,
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_heads: int,
    available_bytes: float,
    *,
    strategy: str = "streams",
    optimizer_kind: str = "adamw",
    d_model: int = 512,
    d_ff: int = 1536,
    n_loops: int = 1,
    headroom: float = 0.25,
    measured_saturation_tokens: Optional[int] = None,
    k_requested: Optional[int] = None,
) -> ConcurrencyPlan:
    """How many concurrent runs of this model fit, and whether it is wise.

    The memory arithmetic reuses ``runtime.memory.plan_training`` for a single
    run's footprint (weights + gradients + optimizer state + estimated
    activations) and divides the available budget by it. That answers "how
    many fit"; it does not answer "will running them together be faster than
    running them one after another", which is a compute question this
    function cannot see from a parameter count alone.

    ``measured_saturation_tokens``, when given, should be the batch_size *
    seq_len at which a *previous* call to :func:`benchmark` on this exact
    device stopped showing speedup from a larger batch (i.e. the point the
    device saturated). Passing it lets this call warn concretely: "your
    tokens/step already meets or exceeds the point at which this device
    stopped scaling, so co-location will make each run ~K times slower for a
    total throughput that is flat or worse." Without it, the honest answer is
    that this function has no basis for a utilisation claim at all, and it
    says so instead of guessing — the wrong obvious alternative is a made-up
    threshold (e.g. "assume anything over batch 32 saturates a GPU"), which
    would be precise, sound authoritative, and be wrong for most GPUs.
    """
    if strategy not in ("processes", "streams", "vmap"):
        raise ValueError(f"unknown strategy {strategy!r}")
    if available_bytes <= 0:
        raise ValueError("available_bytes must be positive")
    mem = plan_training(
        n_params, batch_size, seq_len, n_layers, n_heads, available_bytes,
        optimizer_kind=optimizer_kind, d_model=d_model, d_ff=d_ff, n_loops=n_loops,
        headroom=headroom,
    )
    per_run_gb = mem.fixed_gb + mem.attention_gb
    available_gb = mem.available_gb
    context_tax = CUDA_CONTEXT_GB if strategy == "processes" else 0.0

    advice: list[str] = []
    if per_run_gb <= 0:
        raise ValueError("estimated per-run footprint is not positive")

    if strategy == "processes":
        # Each extra process pays the context tax *before* the first weight
        # is loaded, on top of its own copy of weights/gradients/optimizer
        # state (co-located processes do not share parameters).
        max_by_memory = int(available_gb // (per_run_gb + context_tax))
    else:
        # streams/vmap: one process, one context; only the per-run tensor
        # footprint multiplies.
        max_by_memory = int(available_gb // per_run_gb)
    max_by_memory = max(max_by_memory, 0)

    tokens_per_step = batch_size * seq_len
    saturating: Optional[bool] = None
    if measured_saturation_tokens is not None:
        saturating = tokens_per_step >= measured_saturation_tokens
        if saturating:
            advice.append(
                f"tokens/step ({tokens_per_step}) already meets or exceeds the "
                f"measured saturation point ({measured_saturation_tokens}) on "
                "this device: a single run is believed to already occupy the "
                "device, so co-locating more will slow each one down for flat "
                "or worse total throughput. This came from a prior benchmark() "
                "call, not this function."
            )
    else:
        advice.append(
            "no measured saturation point was supplied: this plan is memory "
            "arithmetic only and makes no claim about whether co-location "
            "will actually help. Call benchmark() on the target device before "
            "trusting a speedup."
        )

    if max_by_memory <= 1:
        advice.append(
            "a single run's estimated footprint already consumes the "
            "available budget; there is no room to co-locate at all."
        )
    recommended = max_by_memory
    if saturating:
        recommended = min(recommended, 1)
    if k_requested is not None and k_requested > max_by_memory:
        advice.append(
            f"requested {k_requested} concurrent runs but only "
            f"{max_by_memory} fit in the memory budget."
        )

    return ConcurrencyPlan(
        per_run_gb=per_run_gb,
        context_overhead_gb=context_tax,
        available_gb=available_gb,
        max_by_memory=max_by_memory,
        recommended=max(recommended, 0),
        tokens_per_step=tokens_per_step,
        saturating=saturating,
        advice=advice,
    )


# ==========================================================================
# benchmark(): measure, do not assert
# ==========================================================================


@dataclass(frozen=True)
class BenchmarkResult:
    device: str
    device_kind: str  # "CPU" or "GPU" — always stated, never implied
    k: int
    steps: int
    baseline_seconds: float
    colocated_seconds: float
    speedup: float
    aggregate_steps_per_second: float
    baseline_steps_per_second: float
    verdict: str

    def render(self) -> str:
        return (
            f"[{self.device_kind}:{self.device}] k={self.k} steps={self.steps}  "
            f"1x={self.baseline_seconds:.4f}s  "
            f"{self.k}x-colocated={self.colocated_seconds:.4f}s  "
            f"speedup={self.speedup:.2f}x  -- {self.verdict}"
        )


def _device_kind(device: str) -> str:
    return "GPU" if str(device).startswith("cuda") else "CPU"


def benchmark(
    make_step: Callable[[], Callable[[], None]],
    k: int,
    *,
    steps: int = 20,
    warmup: int = 2,
    device: str = "cpu",
) -> BenchmarkResult:
    """Time 1 run against K co-located runs; report the speedup actually seen.

    ``make_step()`` is called ``k + 1`` times and must each time return a
    fresh, independent zero-argument callable that performs one training
    step (forward, backward, optimizer step) for one run — independent so
    that co-locating them has no shared mutable state to cheat the timing
    with. Co-location here means round-robin interleaving within *this*
    process, which is exactly the execution shape of ``strategy="streams"``
    and a fair proxy for ``strategy="processes"``'s wall-clock behaviour
    without paying multiprocessing's fixed costs inside a benchmark loop.

    This runs unmodified on CPU, where it will typically show ``speedup`` at
    or below 1.0 — CPU has no independent compute units per "stream" the way
    a GPU has SMs, so interleaving K CPU-bound steps in one process is, at
    best, indistinguishable from running them one after another, and at worst
    slower for the added Python-level bookkeeping. That is not a bug in the
    benchmark; it is the whole point being demonstrated on hardware this
    container actually has. On a GPU, the same function measures the real
    thing: an underutilising single run should show speedup > 1, a
    saturating one should show speedup <= 1.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if steps < 1 or warmup < 0:
        raise ValueError("steps must be positive and warmup non-negative")
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()

    def _sync():
        if cuda:
            torch.cuda.synchronize(device)

    baseline_step = make_step()
    for _ in range(warmup):
        baseline_step()
    _sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        baseline_step()
    _sync()
    baseline_seconds = time.perf_counter() - t0
    del baseline_step

    colocated_steps = [make_step() for _ in range(k)]
    for fn in colocated_steps:
        for _ in range(warmup):
            fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        for fn in colocated_steps:
            fn()
    _sync()
    colocated_seconds = time.perf_counter() - t0

    speedup = (k * baseline_seconds) / colocated_seconds if colocated_seconds > 0 else float("inf")
    aggregate_sps = (k * steps) / colocated_seconds if colocated_seconds > 0 else float("inf")
    baseline_sps = steps / baseline_seconds if baseline_seconds > 0 else float("inf")

    if speedup >= 1.15:
        verdict = "co-location helped: the single run was underutilising the device"
    elif speedup <= 0.9:
        verdict = "co-location hurt: the single run likely already saturated the device"
    else:
        verdict = "no meaningful difference measured either way"

    return BenchmarkResult(
        device=str(device),
        device_kind=_device_kind(device),
        k=k,
        steps=steps,
        baseline_seconds=baseline_seconds,
        colocated_seconds=colocated_seconds,
        speedup=speedup,
        aggregate_steps_per_second=aggregate_sps,
        baseline_steps_per_second=baseline_sps,
        verdict=verdict,
    )


# ==========================================================================
# strategy 1: co-located processes
# ==========================================================================


@dataclass
class ProcessRun:
    """One process's work order. ``fn`` must be a module-level (picklable)
    callable — a lambda or closure will fail to pickle under the default
    ``"spawn"`` context, which is deliberate: ``"spawn"`` (not ``"fork"``) is
    required once CUDA is involved, because a forked child inherits a copy of
    the parent's already-initialised CUDA context and corrupts it the moment
    it touches the device. Tests in this repo run on CPU and may pass
    ``mp_context="fork"`` to ``run_processes`` for speed; real GPU use must
    not.
    """

    label: str
    fn: Callable[..., dict]
    kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessOutcome:
    label: str
    ok: bool
    elapsed: Optional[float]
    result: Any
    error: Optional[str]
    log_path: str


@dataclass(frozen=True)
class ProcessGroupResult:
    outcomes: list[ProcessOutcome]
    aggregate_throughput: Optional[float]  # sum of per-run steps/sec, successes only

    def render(self) -> str:
        lines = []
        for o in self.outcomes:
            status = "ok" if o.ok else f"FAILED: {o.error}"
            lines.append(f"  [{o.label}] {status}  log={o.log_path}")
        lines.append(f"  aggregate throughput: {self.aggregate_throughput}")
        return "\n".join(lines)


def _process_worker(fn, kwargs, log_path, result_path, device, memory_fraction) -> None:
    # Runs inside the child. Anything raised here is caught and written to
    # the result file rather than propagated — multiprocessing already
    # reports a non-zero exitcode on an uncaught exception, but we want the
    # *reason*, not just the fact, without relying on capturing the child's
    # stderr through the OS pipe (which a segfault would skip entirely).
    log_file = open(log_path, "w")
    import sys
    sys.stdout = log_file
    sys.stderr = log_file
    try:
        if memory_fraction is not None and str(device).startswith("cuda") and torch.cuda.is_available():
            # Caps this process's slice of the device so one greedy run
            # cannot starve its co-located siblings; it does not reserve
            # memory, it only refuses to let this process exceed the slice.
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device=device)
        t0 = time.perf_counter_ns()
        result = fn(**kwargs)
        # Wall-clock time.time() can return the same tick for a short run on
        # Windows, making a successful worker disappear from throughput.
        elapsed = max(time.perf_counter_ns() - t0, 1) / 1e9
        payload = {"ok": True, "result": result, "elapsed": elapsed, "error": None}
    except BaseException as exc:  # noqa: BLE001 - report every failure mode, then exit clean
        payload = {
            "ok": False, "result": None, "elapsed": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        log_file.flush()
        log_file.close()
    Path(result_path).write_text(json.dumps(payload, default=str))


def ensure_shared_cache(cache_dir: Path, build: Callable[[], Any], *, poll: float = 0.05,
                        timeout: float = 300.0) -> None:
    """Build a shared corpus cache exactly once across co-located processes.

    Without this, each of K co-located processes independently streams and
    tokenises the corpus — K times the I/O and CPU work for data that is
    identical across runs. The lock is a plain ``O_CREAT | O_EXCL`` file:
    the first process to create it proceeds to call ``build()`` and then
    writes a ``.done`` marker; every later process sees the lock already
    exists, and polls for the marker instead of re-building. This is
    intentionally not a full-featured lock (no reentrancy, no stale-lock
    recovery beyond the timeout) — one file, one clear failure mode, easy to
    reason about across process boundaries where a Python-level ``Lock``
    object does not exist to share in the first place.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / ".building.lock"
    done_path = cache_dir / ".done"
    if done_path.exists():
        return
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        deadline = time.time() + timeout
        while not done_path.exists():
            if time.time() > deadline:
                raise TimeoutError(f"shared cache at {cache_dir} was never marked done "
                                    f"(builder may have crashed while holding the lock)")
            time.sleep(poll)
        return
    try:
        build()
        done_path.write_text("ok")
    finally:
        # Leave the lock in place after success (a stray "did we build?"
        # check is common); remove only on failure, so a crashed builder is
        # retried by the next caller rather than left permanently locked out.
        if not done_path.exists():
            lock_path.unlink(missing_ok=True)


def run_processes(
    runs: Sequence[ProcessRun],
    *,
    device: str = "cpu",
    log_dir: Path,
    stagger: float = 1.0,
    memory_fraction: Optional[float] = None,
    mp_context: str = "spawn",
    join_timeout: Optional[float] = None,
) -> ProcessGroupResult:
    """Launch K training processes co-located on one device.

    Isolation is the entire point: each run is a real OS process, so one
    run's segfault, one run's ``CUDA error: an illegal memory access``, or one
    run's Python exception cannot corrupt or hang the others — they are not
    sharing an interpreter, a CUDA context, or an optimizer's memory. This
    function never calls ``terminate()`` on a sibling because one run failed;
    it waits out every process and reports each outcome independently.

    ``memory_fraction`` defaults to ``1/len(runs)`` when omitted and a CUDA
    device is given, via ``torch.cuda.set_per_process_memory_fraction`` in
    each child — so a run with a memory leak or an unexpectedly large batch
    hits its own OOM instead of starving its siblings. It caps, it does not
    reserve: idle memory is still available to whichever process asks for it
    first, so this is a fairness backstop, not a guarantee of headroom.

    ``stagger`` sleeps between successive ``Process.start()`` calls (not
    inside the children) so K processes do not all hit
    ``cudaMalloc``/context-creation at the same instant, which is when
    driver-level contention and OOM races are most likely.
    """
    if not runs:
        raise ValueError("no runs given")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ctx = multiprocessing.get_context(mp_context)
    frac = memory_fraction if memory_fraction is not None else 1.0 / len(runs)

    processes = []
    result_paths = []
    log_paths = []
    for i, run in enumerate(runs):
        log_path = log_dir / f"{run.label}.log"
        result_path = log_dir / f"{run.label}.result.json"
        result_path.unlink(missing_ok=True)
        p = ctx.Process(
            target=_process_worker,
            args=(run.fn, run.kwargs, str(log_path), str(result_path), device, frac),
            name=run.label,
        )
        p.start()
        processes.append(p)
        result_paths.append(result_path)
        log_paths.append(log_path)
        if i < len(runs) - 1 and stagger > 0:
            time.sleep(stagger)

    for p in processes:
        p.join(join_timeout)

    outcomes = []
    throughputs = []
    for run, p, result_path, log_path in zip(runs, processes, result_paths, log_paths):
        if result_path.exists():
            payload = json.loads(result_path.read_text())
            ok, elapsed, result, error = payload["ok"], payload["elapsed"], payload["result"], payload["error"]
        else:
            # Died before writing a result: killed, segfaulted, or still
            # running past join_timeout. exitcode is None while alive.
            ok, elapsed, result = False, None, None
            error = f"no result file; exitcode={p.exitcode!r} (process likely crashed or timed out)"
        outcomes.append(ProcessOutcome(run.label, ok, elapsed, result, error, str(log_path)))
        if ok and elapsed and isinstance(result, dict) and "steps" in result:
            throughputs.append(result["steps"] / elapsed)

    aggregate = sum(throughputs) if throughputs else None
    return ProcessGroupResult(outcomes=outcomes, aggregate_throughput=aggregate)


# ==========================================================================
# strategy 2: interleaved in-process runs on separate CUDA streams
# ==========================================================================


@dataclass
class StreamRun:
    label: str
    model: nn.Module
    optimizer: Any
    generator: torch.Generator
    #: step_fn(model, optimizer, batch, generator) -> loss (python float)
    step_fn: Callable[[nn.Module, Any, Any, torch.Generator], float]
    losses: list[float] = field(default_factory=list)


@contextmanager
def _stream_scope(stream):
    if stream is None:
        yield
    else:
        with torch.cuda.stream(stream):
            yield


def _make_streams(k: int, device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return [torch.cuda.Stream(device=device) for _ in range(k)]
    # CPU has no independent execution streams; None is a real "no stream"
    # marker, not a stand-in, and every consumer below treats it as such.
    return [None] * k


def run_streams(
    runs: Sequence[StreamRun],
    shared_batches: Iterable[Any],
    steps: int,
    *,
    device: str = "cpu",
) -> list[StreamRun]:
    """Step K models in round-robin, each on its own stream, from one loader.

    Synchronisation, precisely, and why each point exists:

    1. **Within a run, none is needed between steps.** Every op issued inside
       ``with torch.cuda.stream(run_stream):`` for run *i* is enqueued after
       every previous op on that same stream — CUDA streams are FIFO. The
       next step's forward pass for run *i* cannot start reading run *i*'s
       parameters before the previous step's optimizer update finished
       writing them, with no explicit wait, because they are the same
       stream.

    2. **Across runs, none is needed either, and that is the entire benefit
       of streams over one queue.** Run *i* and run *j* touch disjoint
       parameter tensors and disjoint optimizer state. The device is free to
       schedule their kernels concurrently, which is exactly how an
       underutilising single model gets a second one running in the gaps
       between its own kernel launches. Adding a wait here (e.g.
       "synchronize before switching runs") would silently serialise the
       streams and quietly turn this into strategy 1's throughput at
       strategy 2's memory cost.

    3. **One explicit wait, where a shared resource crosses streams:** the
       batch for run *i* is produced by the *one shared* ``BatchLoader``,
       whose host->device copy (when the loader builds tensors on-device) runs
       on the default stream, not run *i*'s stream. Before run *i* consumes
       that batch, we insert ``run_stream.wait_stream(default_stream)`` — a
       stream-to-stream event wait, not a device-wide synchronize, so it
       only blocks run *i* until *its own* batch is ready, and costs nothing
       for the others. Skipping this is the classic "silent nondeterminism"
       failure: the run occasionally reads a batch tensor before its copy
       has landed, and gets stale or zeroed data with no error raised,
       because CUDA's async model does not fail loudly on a missing
       dependency — it just runs a kernel on whatever is already in memory.

    4. **One explicit ``torch.cuda.synchronize()``, after the loop, not
       inside it.** Reading a CUDA tensor into Python (e.g. ``float(loss)``,
       which every ``step_fn`` here does to append to ``losses``) already
       forces an implicit sync on that tensor in PyTorch, so no extra
       barrier is needed per step. What is *not* implicitly synced is
       everything else still in flight when the loop ends — the final
       optimizer step's writes, the last batch's stream-wait. A caller who
       reads ``run.model.state_dict()`` for a checkpoint, or compares
       weights against a reference run, immediately after this function
       returns needs the device to have actually finished, hence the single
       synchronize here rather than trusting an implicit one that only
       covers the *loss* scalar.

    On CPU (this repo's test target) every stream above is ``None`` and every
    wait/synchronize is a no-op: round-robin on CPU is plain sequential
    Python, which is exactly why the determinism test can assert bit-exact
    equality against sequential single-run training — there is no
    overlap to get wrong.
    """
    if not runs:
        raise ValueError("no runs given")
    if steps < 1:
        raise ValueError("steps must be positive")
    streams = _make_streams(len(runs), device)
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    batch_iter = iter(shared_batches)

    for _ in range(steps):
        for run, stream in zip(runs, streams):
            batch = next(batch_iter)
            with _stream_scope(stream):
                if cuda and stream is not None:
                    stream.wait_stream(torch.cuda.current_stream())
                loss = run.step_fn(run.model, run.optimizer, batch, run.generator)
            run.losses.append(float(loss))

    if cuda:
        torch.cuda.synchronize(device)
    return list(runs)


# ==========================================================================
# strategy 3: vectorised replicas via functional_call + vmap
# ==========================================================================


def check_vmap_compatible(model: nn.Module) -> None:
    """Refuse architectures whose forward pass branches on runtime tensor values.

    ``vmap`` lifts one traced computation to run over a batched leading axis;
    it requires every element of that axis to execute the *same* sequence of
    ops. Iridium-1 breaks this twice over: the router's top-k dispatch picks
    a different subset of stacks per token (a data-dependent gather/scatter
    whose shape and target depend on a tensor's runtime value, not just its
    shape), and the ponder loop runs a different number of iterations per
    example depending on a learned halting signal (a data-dependent trip
    count). Both are exactly the control flow vmap cannot lift: it would
    either raise on the first data-dependent Python ``if``/index, or, if that
    branch happens to be expressed as a tensor op instead (e.g. a boolean
    mask), silently apply *one* replica's routing/halting decision to every
    replica sharing that batch position — a wrong answer with no error,
    which is worse than refusing outright. We therefore check the concrete
    model type here rather than trying to trace-and-hope.
    """
    try:
        from ..model.iridium1 import Iridium1
    except Exception:  # pragma: no cover - iridium1 import path is owned elsewhere
        Iridium1 = None  # type: ignore[assignment]
    if Iridium1 is not None and isinstance(model, Iridium1):
        raise NotImplementedError(
            "Iridium1 cannot be vmapped: the router's top-k dispatch and the "
            "ponder loop are data-dependent control flow (a different path "
            "per token/example), which vmap requires to be identical across "
            "the batched dimension. Use strategy='processes' or 'streams' "
            "for a routed model; vmap here is only for a plain reference "
            "module in a hyperparameter sweep where every replica shares "
            "exact tensor shapes and only initialisation/LR/data order "
            "differ."
        )


def _param_shapes(model: nn.Module) -> dict[str, torch.Size]:
    return {k: v.shape for k, v in model.state_dict().items()}


def stack_replicas(models: Sequence[nn.Module]) -> dict[str, torch.Tensor]:
    """Stack K same-shaped models' parameters/buffers along a new leading axis.

    Raises rather than truncating/broadcasting on any mismatch: a shape
    mismatch here almost always means the caller built K variants of
    *different* architectures (e.g. swept a width, not just a seed/LR), which
    is precisely the case vmap cannot handle — there is no way to batch two
    different matrix shapes into one GEMM.
    """
    if len(models) < 2:
        raise ValueError("need at least two replicas to stack")
    for m in models:
        check_vmap_compatible(m)
    reference_shapes = _param_shapes(models[0])
    for i, m in enumerate(models[1:], start=1):
        shapes = _param_shapes(m)
        if set(shapes) != set(reference_shapes):
            raise ValueError(
                f"replica {i} has different named parameters/buffers than "
                f"replica 0 ({run_label(m, i)}); vmap requires identical "
                "architectures, differing only in initialisation/LR/data order"
            )
        mismatched = {k: (reference_shapes[k], shapes[k])
                     for k in shapes if shapes[k] != reference_shapes[k]}
        if mismatched:
            raise ValueError(
                f"replica {i} has different tensor shapes than replica 0: "
                f"{mismatched}; vmap requires every replica to share exact "
                "shapes (a width/depth sweep cannot be vmapped, only "
                "seed/LR/data-order sweeps can)"
            )
    keys = list(reference_shapes)
    return {k: torch.stack([dict(m.state_dict())[k] for m in models]) for k in keys}


def run_label(model: nn.Module, i: int) -> str:
    return getattr(model, "label", f"replica{i}")


@dataclass
class VmapEnsemble:
    """K replicas of one architecture, trained as a single batched model.

    Construction validates shape-compatibility once (``stack_replicas``, via
    ``check_vmap_compatible``); ``step`` then does one vmapped forward+grad
    and applies a plain elementwise AdamW update across the stacked leading
    axis — the optimizer step does *not* need vmap, since "K independent
    elementwise updates over a stacked tensor" is already a single batched op
    with no per-replica control flow.
    """

    base: nn.Module
    params: dict[str, torch.Tensor]
    lrs: torch.Tensor          # shape [K]
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    step_count: int = 0
    _m: dict[str, torch.Tensor] = field(default_factory=dict)
    _v: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for k, v in self.params.items():
            self._m[k] = torch.zeros_like(v)
            self._v[k] = torch.zeros_like(v)

    def _forward(self, params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(self.base, params, (x,))

    def step(self, x: torch.Tensor, y: torch.Tensor,
             loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """One vmapped forward/backward/update. ``x``/``y`` carry a leading K
        axis: replica ``i`` trains on ``x[i]``/``y[i]``, its own data order.
        """

        def per_replica_loss(params, xi, yi):
            pred = self._forward(params, xi)
            return loss_fn(pred, yi)

        grad_fn = torch.func.grad(per_replica_loss)
        grads = torch.vmap(grad_fn, in_dims=(0, 0, 0))(self.params, x, y)

        self.step_count += 1
        b1, b2 = self.betas
        bias1 = 1 - b1 ** self.step_count
        bias2 = 1 - b2 ** self.step_count
        # Same AdamW arithmetic as EagerAdamW, applied elementwise across the
        # stacked leading (replica) axis — no vmap needed for the optimizer
        # step itself, since per-replica learning rates are just a per-slice
        # scalar broadcast, not control flow. `lr` reshapes to [K, 1, 1, ...]
        # so it multiplies each replica's own slice and nothing else's.
        for k, p in self.params.items():
            g = grads[k]
            m, v = self._m[k], self._v[k]
            m.lerp_(g, 1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            denom = (v / bias2).sqrt().add_(self.eps)
            lr = self.lrs.reshape((-1,) + (1,) * (p.dim() - 1))
            p.mul_(1 - lr * self.weight_decay)
            p.sub_(lr * (m / bias1) / denom)

        return grads  # for tests: per-replica gradient inspection

    def loss_only(self, x: torch.Tensor, y: torch.Tensor,
                  loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        def per_replica_loss(params, xi, yi):
            return loss_fn(self._forward(params, xi), yi)
        return torch.vmap(per_replica_loss, in_dims=(0, 0, 0))(self.params, x, y)


def build_vmap_ensemble(make_model: Callable[[], nn.Module], seeds: Sequence[int],
                        lrs: Sequence[float], **kwargs) -> VmapEnsemble:
    """Instantiate K replicas (one per seed) and stack them into a VmapEnsemble.

    ``make_model`` must construct the model *and* apply ``torch.manual_seed``
    (or an equivalent generator) internally so each seed produces a genuinely
    different initialisation; this function seeds the global RNG around each
    call for convenience, since the reference module used in tests is small
    enough that this is deterministic and side-effect-free between calls.
    """
    if len(seeds) != len(lrs):
        raise ValueError("need one lr per seed")
    if len(seeds) < 2:
        raise ValueError("vmap strategy needs at least two replicas; use a normal Trainer for one")
    models = []
    for seed in seeds:
        torch.manual_seed(seed)
        models.append(make_model())
    stacked = stack_replicas(models)
    # torch.func.grad differentiates functionally (it does not need
    # .requires_grad on the leaves, and setting it would only invite a stray
    # autograd graph to accumulate on tensors we mutate in place below).
    params = {k: v.clone().detach() for k, v in stacked.items()}
    return VmapEnsemble(base=models[0], params=params,
                        lrs=torch.tensor(list(lrs), dtype=torch.float32), **kwargs)
