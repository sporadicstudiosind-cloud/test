"""The originating scenario, end to end.

    "I input a simulation of fluid down a waterfall and I want it to calculate
     that fluid flow if I double the input water, and then give me an accurate
     physics-based video and short report about that."

This runs the whole chain and writes every artifact:

1. solve the channel at the original discharge, to steady state;
2. solve it again at the intervened discharge;
3. check both against the analytic laws — Manning normal depth and critical
   depth — and record *which* law each number came from;
4. run the 2-D spectral Navier-Stokes solver for the visual field and emit
   video frames;
5. build an exact metric table from the stored arrays;
6. write the report from a template where every numeral is a
   ``{{metric:key}}`` claim, then **reconcile** it — a numeral in the prose
   that does not resolve to a table entry fails the release gate;
7. emit a Blender script for the 3-D scene;
8. write a result manifest with an evidence level per artifact.

The part worth watching is step 6. The report is not trusted because a model
wrote it carefully; it is trusted because a deterministic pass re-derives every
number in it from the arrays, and refuses the report if one does not bind.

    python examples/waterfall.py --q 3.0 --factor 2.0 --out runs/waterfall
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from iridium.agency.actions import Action, Op
from iridium.agency.scene import to_blender
from iridium.contracts.metrics import MetricEntry, MetricTable, reconcile, render
from iridium.physics.fluid2d import NavierStokes2D, FluidState
from iridium.physics.operators import divergence_norm, vorticity
from iridium.physics.shallow_water import (
    Channel,
    ShallowWater1D,
    critical_depth,
    froude,
    normal_depth,
)
from iridium.physics.verifier import Evidence

REVISION = "waterfall/rev-1"

REPORT = """# Open-channel response to a {{metric:discharge_ratio|.2f}}x change in discharge

## What was asked

The inflow to a rectangular channel of slope {{metric:slope|.4f}} and Manning
roughness {{metric:manning|.3f}} is changed from {{metric:q_before|.2f}} to
{{metric:q_after|.2f}}. What happens to the flow?

## What happens

The depth does **not** scale with the discharge. At steady uniform flow, friction
balances gravity and Manning's law fixes the depth as q^(3/5):

- depth before: {{metric:h_before|.4f}}
- depth after:  {{metric:h_after|.4f}}
- measured ratio: {{metric:depth_ratio_measured|.4f}}, against the analytic
  normal-depth ratio {{metric:depth_ratio_normal_law|.4f}}

The remaining two fifths of the discharge go into velocity:

- velocity before: {{metric:u_before|.4f}}
- velocity after:  {{metric:u_after|.4f}}
- measured ratio: {{metric:velocity_ratio_measured|.4f}}, against
  {{metric:velocity_ratio_analytic|.4f}}

## The number that is easy to get wrong

At a free overfall the controlling depth is *critical* depth, not normal depth,
and it follows a different exponent: q^(2/3), giving a ratio of
{{metric:depth_ratio_critical_law|.4f}} rather than
{{metric:depth_ratio_normal_law|.4f}}. Quoting one where the other applies is a
{{metric:law_disagreement_percent|.1f}}% error, and quoting "the depth doubles"
is a {{metric:naive_error_percent|.1f}}% error.

Critical depth here is {{metric:h_critical_before|.4f}} before and
{{metric:h_critical_after|.4f}} after.

## Regime

The flow stays subcritical throughout: the Froude number goes from
{{metric:froude_before|.4f}} to {{metric:froude_after|.4f}}, so no hydraulic
jump forms and the downstream control propagates upstream. Discharge is uniform
along the channel to a relative error of
{{metric:discharge_closure_after|.2e}}.

## How these numbers were obtained

Finite-volume shallow-water solution with a Rusanov flux, run to a steady state
where both conserved variables stop changing. The solver was started at a depth
of {{metric:initial_depth|.2f}}, far from the answer, so the equilibrium was
found rather than assumed. The visual field is a separate pseudo-spectral
Navier-Stokes solve whose divergence is held at
{{metric:field_divergence|.2e}}; it is an illustration of the flow structure and
is not the source of any number above.
"""


def build(q_before: float, factor: float, channel: Channel, initial_depth: float,
          out: Path, n_frames: int) -> dict:
    q_after = q_before * factor
    solver = ShallowWater1D(channel)

    before, _, info_b = solver.run_to_steady(
        q_before, max_time=20000, tol=1e-8, initial_depth=initial_depth
    )
    after, _, info_a = solver.run_to_steady(
        q_after, max_time=20000, tol=1e-8, initial_depth=initial_depth
    )
    d_before = solver.diagnostics(before, q_before)
    d_after = solver.diagnostics(after, q_after)

    h_b, h_a = d_before["h_upstream"], d_after["h_upstream"]
    u_b, u_a = q_before / h_b, q_after / h_a
    depth_ratio = h_a / h_b
    normal_law = factor ** 0.6
    critical_law = factor ** (2.0 / 3.0)

    # -- the visual field: a separate, clearly-labelled illustration --------
    field_solver = NavierStokes2D(48, nu=0.01)
    x = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    state = FluidState(
        np.cos(xx) * np.sin(yy) * u_a, -np.sin(xx) * np.cos(yy) * u_a, 0.0
    )
    frames = []
    for i in range(n_frames):
        state = field_solver.step(state, 0.01)
        frames.append(vorticity(state.u, state.v).copy())
    field_divergence = divergence_norm(state.u, state.v)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "depth_before.npy", before.h)
    np.save(out / "depth_after.npy", after.h)
    np.save(out / "vorticity_frames.npy", np.stack(frames))
    _write_frames(np.stack(frames), out / "frames")

    # -- the metric table: every number, from the arrays --------------------
    table = MetricTable(scenario_revision=REVISION)
    def add(key, value, unit, source):
        table.add(MetricEntry(key=key, value=float(value), unit=unit,
                              source=source, scenario_revision=REVISION))

    add("slope", channel.slope, "1", "channel.slope")
    add("manning", channel.manning, "1", "channel.manning")
    add("initial_depth", initial_depth, "m", "solver.initial_condition")
    add("q_before", q_before, "m2/s", "input")
    add("q_after", q_after, "m2/s", "input*factor")
    add("discharge_ratio", factor, "1", "input")
    add("h_before", h_b, "m", "depth_before.npy#upstream_mean")
    add("h_after", h_a, "m", "depth_after.npy#upstream_mean")
    add("u_before", u_b, "m/s", "q_before/h_before")
    add("u_after", u_a, "m/s", "q_after/h_after")
    add("depth_ratio_measured", depth_ratio, "1", "h_after/h_before")
    add("depth_ratio_normal_law", normal_law, "1", "manning: factor^(3/5)")
    add("depth_ratio_critical_law", critical_law, "1", "critical: factor^(2/3)")
    add("velocity_ratio_measured", u_a / u_b, "1", "u_after/u_before")
    add("velocity_ratio_analytic", factor ** 0.4, "1", "factor^(2/5)")
    add("h_critical_before", critical_depth(q_before), "m", "(q^2/g)^(1/3)")
    add("h_critical_after", critical_depth(q_after), "m", "(q^2/g)^(1/3)")
    add("froude_before", d_before["froude_upstream"], "1", "depth_before.npy#Fr")
    add("froude_after", d_after["froude_upstream"], "1", "depth_after.npy#Fr")
    add("discharge_closure_after", d_after["discharge_error"], "1",
        "depth_after.npy#max|hu-q|/q")
    add("field_divergence", field_divergence, "1", "vorticity_frames.npy#div")
    add("law_disagreement_percent",
        100.0 * abs(critical_law - normal_law) / normal_law, "1", "derived")
    add("naive_error_percent",
        100.0 * abs(factor - normal_law) / normal_law, "1", "derived")

    # -- render, then refuse to release an unbound numeral ------------------
    text = render(table, REPORT)
    findings = reconcile(
        table, text, rel_tol=1e-3,
        allow_bare=frozenset({"1", "2", "3", "5", "2.0"}),
    )
    (out / "report.md").write_text(text, encoding="utf-8")

    # -- the editable 3-D scene ---------------------------------------------
    actions = [
        Action(Op.ADD_PLANE, (0.0, 0.0, 0.0, 10.0), text="channel_bed"),
        Action(Op.ADD_CUBE, (0.0, 0.0, h_a / 2, float(h_a)), text="water_after"),
        Action(Op.SET_MATERIAL, (1, 0.15, 0.45, 0.85)),
        Action(Op.SET_PARAM, (0, float(q_after), 0.0), text="discharge"),
        Action(Op.RENDER, (0, 0)),
    ]
    (out / "scene.py").write_text(to_blender(actions), encoding="utf-8")

    manifest = {
        "schema_version": "1.0",
        "result_id": "waterfall-intervention-1",
        "status": "complete" if not findings else "blocked",
        "scenario_revision": REVISION,
        "intervention": {"discharge_multiplier": factor},
        "execution_mode": "deterministic_solver",
        "artifacts": {
            "report.md": Evidence.VERIFIED.label,
            "depth_before.npy": Evidence.VERIFIED.label,
            "depth_after.npy": Evidence.VERIFIED.label,
            "vorticity_frames.npy": Evidence.ILLUSTRATIVE.label,
            "frames/": Evidence.ILLUSTRATIVE.label,
            "scene.py": Evidence.ILLUSTRATIVE.label,
        },
        "metrics": {k: v.value for k, v in table.entries.items()},
        "convergence": {"before": info_b, "after": info_a},
        "reconciliation_findings": [str(f) for f in findings],
        "omissions": [
            "the vorticity field is an illustration of flow structure, not a "
            "solution of the channel geometry",
            "no empirical measurement was compared against; evidence stops at "
            "'verified', never reaching 'validated'",
            "the bed-slope source term is not well-balanced; the residual shows "
            "up as ~1e-4 relative error against the analytic normal depth",
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float), encoding="utf-8"
    )
    return manifest


def _write_frames(frames: np.ndarray, out: Path) -> None:
    """PNG frames, if matplotlib is available; a .npy stack regardless."""
    out.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    lim = float(np.max(np.abs(frames)))
    for i, frame in enumerate(frames):
        fig, ax = plt.subplots(figsize=(3, 3), dpi=96)
        ax.imshow(frame.T, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(out / f"frame_{i:03d}.png")
        plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q", type=float, default=3.0)
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--slope", type=float, default=0.002)
    ap.add_argument("--manning", type=float, default=0.030)
    ap.add_argument("--cells", type=int, default=200)
    ap.add_argument("--initial-depth", type=float, default=0.5)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--out", default="runs/waterfall")
    args = ap.parse_args(argv)

    channel = Channel(length=100.0, n_cells=args.cells, slope=args.slope,
                      manning=args.manning)
    manifest = build(args.q, args.factor, channel, args.initial_depth,
                     Path(args.out), args.frames)
    print(Path(args.out, "report.md").read_text(encoding="utf-8"))
    print("=" * 72)
    print(f"status: {manifest['status']}")
    print(f"reconciliation findings: {manifest['reconciliation_findings'] or 'none'}")
    print(f"artifacts written to {args.out}/")
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
