"""CPU-only coverage for iridium.training.multirun.

Nothing here needs a GPU, and everything runs in well under a second per
test: the planner is arithmetic, the process test kills a child on purpose,
the streams test checks bit-exact determinism (there is no concurrency to
get wrong on CPU, which is exactly what makes the comparison meaningful),
and the vmap test checks numerical equivalence against a hand-written
sequential reference to a tight tolerance. The benchmark test only checks
that a CPU run reports itself as CPU and produces a real measurement, never
that co-location helped — on CPU it should not, and the point of this suite
is to not lie about that either.
"""

from __future__ import annotations

import multiprocessing
import tempfile
import threading
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from iridium.config_builder import build as build_config
from iridium.model.iridium1 import Iridium1
from iridium.training.multirun import (
    BenchmarkResult,
    ProcessRun,
    StreamRun,
    VmapEnsemble,
    benchmark,
    build_vmap_ensemble,
    check_vmap_compatible,
    ensure_shared_cache,
    plan,
    run_processes,
    run_streams,
    stack_replicas,
)


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(4, 8)
        self.l2 = nn.Linear(8, 1)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


class _Wide(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(4, 16)  # different shape than _Tiny -> must be refused
        self.l2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


def _mse(pred, y):
    return ((pred - y) ** 2).mean()


# ==========================================================================
# plan()
# ==========================================================================


def test_plan_memory_arithmetic_matches_hand_computation():
    from iridium.runtime.memory import plan_training

    kwargs = dict(n_params=1_000_000, batch_size=8, seq_len=128, n_layers=4,
                  n_heads=4, available_bytes=8e9)
    p = plan(**kwargs, strategy="streams", headroom=0.25)
    mem = plan_training(**kwargs, headroom=0.25)
    # per-run footprint is exactly runtime.memory's own estimate: this
    # function must not duplicate that arithmetic, only divide by it.
    assert p.per_run_gb == pytest.approx(mem.fixed_gb + mem.attention_gb)
    assert p.available_gb == pytest.approx(8e9 * 0.75 / 1e9, rel=1e-6)
    assert p.max_by_memory == int(p.available_gb // p.per_run_gb)
    assert p.context_overhead_gb == 0.0  # streams: no per-run CUDA context
    assert p.tokens_per_step == 8 * 128
    assert p.saturating is None
    assert any("no measured saturation" in a for a in p.advice)


def test_plan_processes_pays_context_tax_and_fits_fewer():
    common = dict(n_params=1_000_000, batch_size=8, seq_len=128, n_layers=4,
                 n_heads=4, available_bytes=8e9)
    p_streams = plan(**common, strategy="streams")
    p_processes = plan(**common, strategy="processes")
    assert p_processes.context_overhead_gb > 0
    assert p_processes.max_by_memory <= p_streams.max_by_memory


def test_plan_warns_when_saturating():
    p = plan(
        n_params=1_000_000, batch_size=64, seq_len=512, n_layers=4, n_heads=4,
        available_bytes=8e9, measured_saturation_tokens=64 * 512,
    )
    assert p.saturating is True
    assert p.recommended <= 1
    assert any("already meets or exceeds" in a for a in p.advice)


def test_plan_not_saturating_keeps_recommendation():
    p = plan(
        n_params=1_000_000, batch_size=2, seq_len=32, n_layers=2, n_heads=2,
        available_bytes=8e9, measured_saturation_tokens=64 * 512,
    )
    assert p.saturating is False
    assert p.recommended == p.max_by_memory


def test_plan_rejects_bad_inputs():
    with pytest.raises(ValueError):
        plan(1000, 8, 128, 4, 4, available_bytes=8e9, strategy="bogus")
    with pytest.raises(ValueError):
        plan(1000, 8, 128, 4, 4, available_bytes=0)


# ==========================================================================
# benchmark()
# ==========================================================================


def test_benchmark_runs_on_cpu_and_labels_itself():
    def make_step():
        m = _Tiny()
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        x, y = torch.randn(8, 4), torch.randn(8, 1)

        def step():
            opt.zero_grad()
            _mse(m(x), y).backward()
            opt.step()

        return step

    result = benchmark(make_step, k=2, steps=5, warmup=1, device="cpu")
    assert isinstance(result, BenchmarkResult)
    assert result.device_kind == "CPU"
    assert result.k == 2
    assert result.baseline_seconds > 0
    assert result.colocated_seconds > 0
    assert result.aggregate_steps_per_second > 0
    # No claim about direction: CPU co-location may land anywhere near 1x.
    assert result.speedup > 0


def test_benchmark_rejects_bad_k():
    with pytest.raises(ValueError):
        benchmark(lambda: (lambda: None), k=0)


# ==========================================================================
# strategy: processes
# ==========================================================================


def _ok_worker(n: int) -> dict:
    return {"steps": n}


def _failing_worker() -> None:
    raise RuntimeError("deliberate failure for the supervision test")


def test_run_processes_supervises_a_failing_child_without_killing_others():
    with tempfile.TemporaryDirectory() as d:
        runs = [
            ProcessRun("good_a", _ok_worker, {"n": 8}),
            ProcessRun("bad", _failing_worker, {}),
            ProcessRun("good_b", _ok_worker, {"n": 4}),
        ]
        result = run_processes(
            runs, device="cpu", log_dir=Path(d), stagger=0.02,
            mp_context=("fork" if "fork" in multiprocessing.get_all_start_methods()
                        else "spawn"),
        )
        by_label = {o.label: o for o in result.outcomes}
        assert by_label["good_a"].ok is True
        assert by_label["good_b"].ok is True
        assert by_label["bad"].ok is False
        assert "RuntimeError" in by_label["bad"].error
        assert "deliberate failure" in by_label["bad"].error
        # the surviving runs' throughput must still be aggregated despite the failure
        assert result.aggregate_throughput is not None
        assert result.aggregate_throughput > 0
        # every run gets its own log file, present regardless of outcome
        for o in result.outcomes:
            assert Path(o.log_path).exists()


def test_run_processes_rejects_empty():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            run_processes([], device="cpu", log_dir=Path(d))


def test_ensure_shared_cache_builds_once():
    calls = []

    def build():
        calls.append(1)
        time.sleep(0.05)

    with tempfile.TemporaryDirectory() as d:
        cache_dir = Path(d) / "cache"
        # two concurrent callers (threads stand in for co-located processes:
        # the coordination is a plain-file lock, which behaves identically
        # whether the two callers are threads or separate OS processes)
        results = []
        threads = [threading.Thread(target=lambda: (ensure_shared_cache(cache_dir, build), results.append(1)))
                  for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert len(calls) == 1
        assert (cache_dir / ".done").exists()
        # a later caller, after the marker exists, must not rebuild either
        ensure_shared_cache(cache_dir, build)
        assert len(calls) == 1


# ==========================================================================
# strategy: streams
# ==========================================================================


def _make_step_fn():
    def step_fn(model, opt, batch, generator):
        x, y = batch
        opt.zero_grad()
        loss = _mse(model(x), y)
        loss.backward()
        opt.step()
        return loss.item()

    return step_fn


def test_run_streams_matches_sequential_training_bit_exact():
    torch.manual_seed(0)
    batches = [(torch.randn(4, 4), torch.randn(4, 1)) for _ in range(20)]

    torch.manual_seed(11)
    model_a = _Tiny()
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
    torch.manual_seed(22)
    model_b = _Tiny()
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1)

    run_a = StreamRun("a", model_a, opt_a, torch.Generator(), _make_step_fn())
    run_b = StreamRun("b", model_b, opt_b, torch.Generator(), _make_step_fn())
    run_streams([run_a, run_b], iter(batches), steps=10, device="cpu")

    # Sequential baseline for run "a" alone: round-robin over 2 runs means
    # run "a" is the first `next()` each step, i.e. the even-indexed batches.
    torch.manual_seed(11)
    model_a_seq = _Tiny()
    opt_a_seq = torch.optim.SGD(model_a_seq.parameters(), lr=0.1)
    seq_step = _make_step_fn()
    for i in range(0, 20, 2):
        seq_step(model_a_seq, opt_a_seq, batches[i], None)

    for (_, p1), (_, p2) in zip(model_a.named_parameters(), model_a_seq.named_parameters()):
        assert torch.equal(p1, p2)
    assert run_a.losses  # per-run history was recorded


def test_run_streams_requires_at_least_one_run_and_positive_steps():
    with pytest.raises(ValueError):
        run_streams([], iter([]), steps=1)
    run = StreamRun("a", _Tiny(), torch.optim.SGD(_Tiny().parameters(), lr=0.1),
                    torch.Generator(), _make_step_fn())
    with pytest.raises(ValueError):
        run_streams([run], iter([]), steps=0)


# ==========================================================================
# strategy: vmap
# ==========================================================================


def test_vmap_ensemble_matches_sequential_training_closely():
    seeds = [0, 1, 2]
    lrs = [0.05, 0.05, 0.05]
    betas, eps, wd = (0.9, 0.95), 1e-8, 0.01

    torch.manual_seed(999)
    xs = torch.randn(len(seeds), 6, 4)
    ys = torch.randn(len(seeds), 6, 1)

    ensemble = build_vmap_ensemble(lambda: _Tiny(), seeds, lrs,
                                   betas=betas, eps=eps, weight_decay=wd)
    n_steps = 5
    for _ in range(n_steps):
        ensemble.step(xs, ys, _mse)

    # Hand-written sequential AdamW reference, one replica at a time, using
    # the exact same math as VmapEnsemble.step.
    for i, seed in enumerate(seeds):
        torch.manual_seed(seed)
        m = _Tiny()
        params = {k: v.clone() for k, v in m.state_dict().items()}
        m_moment = {k: torch.zeros_like(v) for k, v in params.items()}
        v_moment = {k: torch.zeros_like(v) for k, v in params.items()}
        b1, b2 = betas
        for step in range(n_steps):
            live = {k: v.clone().requires_grad_(True) for k, v in params.items()}
            pred = torch.func.functional_call(m, live, (xs[i],))
            loss = _mse(pred, ys[i])
            grads = dict(zip(live.keys(), torch.autograd.grad(loss, list(live.values()))))
            bias1 = 1 - b1 ** (step + 1)
            bias2 = 1 - b2 ** (step + 1)
            new_params = {}
            for k, p in params.items():
                g = grads[k]
                m_moment[k].lerp_(g, 1 - b1)
                v_moment[k].mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = (v_moment[k] / bias2).sqrt().add_(eps)
                new_params[k] = p * (1 - lrs[i] * wd) - lrs[i] * (m_moment[k] / bias1) / denom
            params = new_params
        for k in params:
            got = ensemble.params[k][i]
            torch.testing.assert_close(got, params[k], atol=1e-5, rtol=1e-4)


def test_stack_replicas_refuses_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        stack_replicas([_Tiny(), _Wide()])


def test_stack_replicas_requires_at_least_two():
    with pytest.raises(ValueError):
        stack_replicas([_Tiny()])


def test_build_vmap_ensemble_requires_matching_lr_and_seed_counts():
    with pytest.raises(ValueError):
        build_vmap_ensemble(lambda: _Tiny(), seeds=[0, 1], lrs=[0.1])
    with pytest.raises(ValueError):
        build_vmap_ensemble(lambda: _Tiny(), seeds=[0], lrs=[0.1])


def test_check_vmap_compatible_refuses_iridium1():
    torch.manual_seed(0)
    cfg = build_config(d_model=64, core_layers=2, n_superstacks=2, superstack_layers=3,
                       d_head=32, n_kv_heads=1, vocab_size=384, max_seq_len=256,
                       name="iridium-1-vmap-refusal-probe")
    model = Iridium1(cfg)
    with pytest.raises(NotImplementedError, match="router.*top-k|ponder"):
        check_vmap_compatible(model)


def test_check_vmap_compatible_accepts_plain_module():
    check_vmap_compatible(_Tiny())  # must not raise
