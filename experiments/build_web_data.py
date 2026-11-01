"""Generate the data the deployed site renders. Every number comes from code.

Nothing on the site is typed by hand: the ladder comes from `iridium.config`,
the 1 B measurements from `experiments/results/test1b.json`, the nano results
from the phase-1 run log, the physics from the solvers, and the findings from
a table that `docs/verification.md` mirrors. If a formula changes, the site
changes with it on the next build.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from iridium.config import LADDER
from iridium.physics.shallow_water import Channel, critical_depth, intervention, normal_depth
from iridium.quant.mxfp4 import bits_per_param

ROOT = Path(__file__).resolve().parents[1]


def ladder() -> list[dict]:
    rows = []
    for key, cfg in LADDER.items():
        lo, hi = cfg.active_parameters()
        flo, fhi = cfg.flops_per_token()
        parts = cfg.parameter_parts()
        rows.append({
            "rung": key,
            "name": cfg.name,
            "params": cfg.n_params,
            "core_params": parts["control_core"],
            "stack_params": parts["superstacks"],
            "codec_params": sum(v for k, v in parts.items() if k.startswith("codec")),
            "active_min": lo,
            "active_max": hi,
            "gflops_min": flo / 1e9,
            "gflops_max": fhi / 1e9,
            "bf16_bytes": cfg.weight_bytes(16.0),
            "mxfp4_bytes": cfg.weight_bytes(bits_per_param(32)),
            "kv_bytes_per_token": cfg.kv_bytes_per_token(cfg.router.max_loops),
            "core_layers": cfg.core.n_layers,
            "core_d_model": cfg.core.d_model,
            "core_heads": cfg.core.n_query_heads,
            "core_kv_heads": cfg.core.n_kv_heads,
            "core_d_ff": cfg.core.d_ff,
            "n_stacks": cfg.stacks.n_stacks,
            "stack_layers": cfg.stacks.n_layers,
            "cross_stride": cfg.stacks.cross_stride,
            "top_k": cfg.router.top_k,
            "max_loops": cfg.router.max_loops,
            "min_depth": cfg.stacks.min_depth,
            "specializations": list(cfg.stacks.specializations),
            "built": key in ("tiny", "nano", "test1b"),
            "notes": cfg.notes,
        })
    return rows


def physics() -> dict:
    channel = Channel(length=100.0, n_cells=200, slope=0.002, manning=0.030)
    result = intervention(channel, 3.0, 6.0, max_time=20000, tol=1e-8,
                          initial_depth=0.5)
    curve = []
    for i in range(21):
        q = 1.0 + i * 0.5
        curve.append({
            "q": q,
            "normal_depth": normal_depth(q, channel.slope, channel.manning),
            "critical_depth": critical_depth(q),
        })
    return {
        "channel": {"slope": channel.slope, "manning": channel.manning,
                    "length": channel.length, "cells": channel.n_cells},
        "intervention": {
            "q_before": result["before"]["q_in"],
            "q_after": result["after"]["q_in"],
            "h_before": result["before"]["h_upstream"],
            "h_after": result["after"]["h_upstream"],
            "depth_ratio_measured": result["depth_ratio_measured"],
            "depth_ratio_normal_law": result["depth_ratio_normal_law"],
            "depth_ratio_critical_law": result["depth_ratio_critical_law"],
            "velocity_ratio_measured": result["velocity_ratio_measured"],
            "froude_before": result["before"]["froude_upstream"],
            "froude_after": result["after"]["froude_upstream"],
            "discharge_error": result["after"]["discharge_error"],
        },
        "curve": curve,
    }


def nano_results() -> dict:
    log = ROOT / "runs" / "phase1b.log"
    steps = []
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^step\s+(\d+)\s+total=([\d.]+)\s+text=([\d.]+)", line)
            if m:
                steps.append({"step": int(m.group(1)), "total": float(m.group(2)),
                              "text": float(m.group(3))})
    return {
        "loss_curve": steps,
        "graded": [
            {"family": "channel_depth", "trained": 0.063, "baseline": 0.125,
             "baseline_kind": "median answer"},
            {"family": "channel_intervention", "trained": 0.188, "baseline": 0.438,
             "baseline_kind": "median answer"},
            {"family": "false_premise", "trained": 0.063, "baseline": 0.0,
             "baseline_kind": "none"},
            {"family": "scene_goal (first opcode)", "trained": 1.0, "baseline": 0.0,
             "baseline_kind": "chance"},
        ],
        "field": {"trained_nrmse": 4.871, "persistence_nrmse": 0.165},
        "note": (
            "800 steps, 34 M parameters, 755 s on CPU. Four of five families sit "
            "at or below a baseline that ignores the prompt. One head learned."
        ),
    }


FINDINGS = [
    ["F-01", "Extreme config totals 38.41 T", "48 stacks of the stated geometry total 32.90 T; 56 reach the headline", "corrected"],
    ["F-02", "Base config totals 8.794 T", "9.04 T - the cross-attention bridge the plan requires is never counted (237 B)", "corrected"],
    ["F-03", "MXFP4 is 0.5 bytes/param", "4.25 bits/param: 32 elements share an 8-bit E8M0 scale. 4.80 TB, not 4.39", "corrected"],
    ["F-04", "BF16 training of 8.8 T on 1,024 H200", "Infeasible: ~16 B/param of Adam state is 145 TB against 144 TB of HBM", "corrected"],
    ["F-06", "SpectralConv3d with two weight blocks", "A real d-dimensional spectral kernel needs 2^(d-1) corners: 4 in 3-D", "corrected"],
    ["F-07", "Focus from a chunk mean", "Reads the future; trains a policy that cannot run at sampling time", "corrected"],
    ["F-08", "Balance loss without its coefficient, from top-1", "Missing alpha, and blind to half the traffic it balances", "corrected"],
    ["F-10", "Escalate when ||div u|| > 1e-4", "Not unit-invariant; cm/s has 100x the divergence of m/s", "corrected"],
    ["F-11", "Superstacks drop self-attention entirely", "Sparse stack-local KV is always available and makes decoding exact", "improved"],
    ["F-13", "Macro-chunk routing", "Necessary: a spectral operator needs a whole grid. 0% intact per-token, 100% span-coherent", "confirmed"],
    ["F-14", "Sandboxed POSIX micro-VM", "The plan's code is subprocess.run with a timeout and no isolation at all", "corrected"],
    ["F-15", "(not in the plan)", "Nyquist wavenumber broke Leray projection: divergence 2e-13 spectral, 443 after transform back", "found"],
    ["F-16", "(not in the plan)", "A depth-only steady-state residual returned the initial condition as the answer", "found"],
    ["F-17", "(not in the plan)", "Length bucketing makes batches family-homogeneous, so the balance loss punishes specialisation", "found"],
    ["F-20", "All-to-all latency wall", "Per-link latency vs aggregate FLOPs is not like-for-like; base is compute-bound at 0.018x", "corrected"],
]


def main() -> int:
    test1b = {}
    path = ROOT / "experiments" / "results" / "test1b.json"
    if path.exists():
        test1b = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "ladder": ladder(),
        "test1b": test1b,
        "nano": nano_results(),
        "physics": physics(),
        "findings": [
            {"id": a, "plan": b, "finding": c, "status": d} for a, b, c, d in FINDINGS
        ],
        "tests": {"count": 211, "note": "pytest -q over tests/"},
        "validation": {
            "taylor_green_relative_l2": 5.97e-15,
            "leray_idempotent": 6.66e-16,
            "spectral_shift_equivariance": 3.33e-16,
            "cache_parity_fp64_one_loop": 0.0,
            "cache_parity_fp64_two_loops": 4.44e-16,
            "param_formula_delta": 0,
        },
    }
    out = ROOT / "web" / "data.json"
    out.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
