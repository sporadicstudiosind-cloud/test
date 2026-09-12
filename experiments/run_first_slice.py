"""Run the first implementation slice end to end (§20.3).

Trains one small Iridium-1 checkpoint on advection-diffusion episodes, evaluates
it on held-out splits, sweeps generation steps for the native image head, then
builds a metric table from the exact arrays, renders a report from it, and
reconciles every numeral in that report against the table.

Everything printed here is measured. Nothing is asserted in advance.

    python3 experiments/run_first_slice.py --steps 4000
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iridium.contracts.metrics import (  # noqa: E402
    MetricEntry,
    MetricTable,
    reconcile,
    render,
)
from iridium.data.advection import (  # noqa: E402
    EXTRAPOLATION_SPEED_RANGE,
    TRAIN_SPEED_RANGE,
    make_split,
)
from iridium.training.first_slice import (  # noqa: E402
    SLICE_CONFIG,
    TrainConfig,
    evaluate,
    image_quality_vs_steps,
    save_run,
    train,
)

OUT = Path(__file__).resolve().parent / "results"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--train-episodes", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    started = time.time()

    # Family-level splits (§18.2). The interpolation split is drawn from the
    # training band with a different seed; the extrapolation split sits wholly
    # outside it. Random samples from one pool would measure only memorization.
    train_eps = make_split(args.train_episodes, seed=args.seed + 1)
    interp_eps = make_split(args.eval_episodes, seed=args.seed + 202)
    extrap_eps = make_split(
        args.eval_episodes, seed=args.seed + 303,
        speed_range=EXTRAPOLATION_SPEED_RANGE,
    )

    print(f"train {len(train_eps)}  interp {len(interp_eps)}  extrap {len(extrap_eps)}")
    print(f"train speeds  {TRAIN_SPEED_RANGE}")
    print(f"extrap speeds {EXTRAPOLATION_SPEED_RANGE}\n")

    cfg = TrainConfig(steps=args.steps, lr=args.lr, seed=args.seed, log_every=500)
    model, history = train(train_eps, cfg)

    print("\n--- held-out evaluation ---")
    results = {
        "train_fit": evaluate(model, train_eps[: args.eval_episodes]),
        "interpolation": evaluate(model, interp_eps),
        "extrapolation": evaluate(model, extrap_eps),
    }
    for name, block in results.items():
        print(f"\n{name}")
        for key, value in block.items():
            print(f"  {key:28s} {value:.6g}")

    print("\n--- native image head: quality vs generation steps ---")
    steps_sweep = image_quality_vs_steps(model, interp_eps[:32])
    for n_steps, err in steps_sweep.items():
        print(f"  {n_steps:3d} steps   nrmse {err:.4f}")

    # ---- metric table, report, reconciliation -----------------------------
    revision = "slice_rev_0001"
    table = MetricTable(revision)
    interp = results["interpolation"]
    extrap = results["extrapolation"]

    for key, value, unit in [
        ("interp.nrmse.flux_head", interp["nrmse_flux_head"], "1"),
        ("interp.nrmse.direct_head", interp["nrmse_direct_head"], "1"),
        ("interp.nrmse.persistence", interp["nrmse_persistence"], "1"),
        ("interp.nrmse.upwind", interp["nrmse_upwind_reference"], "1"),
        ("extrap.nrmse.flux_head", extrap["nrmse_flux_head"], "1"),
        ("extrap.nrmse.persistence", extrap["nrmse_persistence"], "1"),
        ("interp.mass_drift.flux_head", interp["mass_drift_flux_head"], "1"),
        ("interp.mass_drift.direct_head", interp["mass_drift_direct_head"], "1"),
    ]:
        table.add(MetricEntry(key, value, unit, "experiments/results/first_slice.json",
                              revision))

    template = (
        "# First slice result\n\n"
        "On the held-out interpolation split the conservative flux head reaches a "
        "normalized RMSE of {{metric:interp.nrmse.flux_head|.4f}}, against "
        "{{metric:interp.nrmse.persistence|.4f}} for persistence and "
        "{{metric:interp.nrmse.upwind|.4f}} for the first-order upwind reference. "
        "The unconstrained direct head reaches "
        "{{metric:interp.nrmse.direct_head|.4f}}.\n\n"
        "Mass drift is {{metric:interp.mass_drift.flux_head|.3e}} for the flux head "
        "and {{metric:interp.mass_drift.direct_head|.3e}} for the direct head.\n\n"
        "On the extrapolation split the flux head reaches "
        "{{metric:extrap.nrmse.flux_head|.4f}} against "
        "{{metric:extrap.nrmse.persistence|.4f}} for persistence.\n"
    )
    report = render(table, template)
    findings = reconcile(table, report)

    print("\n--- numeric reconciliation ---")
    print(report)
    if findings:
        print("FINDINGS:")
        for f in findings:
            print(" ", f)
    else:
        print("reconciliation clean: every numeral resolves to the metric table")

    payload = {
        "model": "iridium-1-slice",
        "scenario_revision": revision,
        "config": {
            "transformer": SLICE_CONFIG.name,
            "d_model": SLICE_CONFIG.d_model,
            "blocks": [SLICE_CONFIG.n_prelude, SLICE_CONFIG.n_core, SLICE_CONFIG.n_coda],
            "transformer_params": SLICE_CONFIG.transformer_params,
            "total_params": sum(p.numel() for p in model.parameters()),
            "train": vars(args),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "history": history,
        "results": results,
        "image_steps_sweep": {str(k): v for k, v in steps_sweep.items()},
        "reconciliation_findings": [str(f) for f in findings],
        "wall_clock_s": round(time.time() - started, 1),
    }
    save_run(OUT / "first_slice.json", payload)
    (OUT / "first_slice_report.md").write_text(report, encoding="utf-8")
    torch.save(model.state_dict(), OUT / "first_slice.pt")
    print(f"\nwrote {OUT/'first_slice.json'} in {payload['wall_clock_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
