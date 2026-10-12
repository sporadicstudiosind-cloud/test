# 11 — Evaluation

> The field's existing benchmarks cannot measure this system's central claim. That is the first
> problem to solve, not a footnote.

## 11.1 Why standard benchmarks are insufficient here

Physical-plausibility benchmarks for generated video judge whether output *looks* right. Prompts
describe a scene without specifying the expected physical outcome, so raters assess plausibility;
even pixel-referenced benchmarks may not check whether conserved quantities are preserved; and
aggregate scores collapse heterogeneous failure modes into a single number that prevents per-law
diagnosis.

A model can therefore score well while violating mass balance in every output. Since
[`04`](04-generation.md)'s entire thesis is that conservation is the thing that matters, the primary
evaluation instrument has to be built rather than adopted.

Public benchmarks are still run — they measure competitiveness, and regression on them matters — but
they do not measure the claim.

## 11.2 The conservation suite (primary, and built for this)

Scenarios with **analytically or numerically known conserved quantities**, evaluated on the outputs
rather than on their appearance.

| Metric | Definition | Why |
|---|---|---|
| Mass drift | relative change in total mass over the sequence | the single most common generative failure |
| Momentum drift | relative change in total momentum, per component | second most common |
| Energy budget | dissipation accounted against input work | catches unphysical energy injection |
| Divergence residual | ‖∇·u‖ over the domain for incompressible cases | direct constraint violation |
| Spectral fidelity | energy spectrum vs. reference across wavenumbers | catches the low-pass failure of §[`03`](03-physics.md) |
| Per-law breakdown | violations attributed to individual laws, never aggregated | aggregate scores hide which physics broke |

**The decisive experiment.** Same trunk, same scenario, two output paths: rendered-from-field versus
conditionally-generated-from-field. If the conservation metrics do not separate them, the premise of
[`04`](04-generation.md) is wrong — this is kill criterion 8 in [`10`](10-open-problems.md), and it
should be run early, at small scale, before the expensive stages.

## 11.3 Extrapolation: region hold-outs, never random splits

Random held-out samples measure interpolation and will produce a flattering number that means
nothing. Extrapolation requires removing **contiguous regions of Π-space** and testing there.

- **Error versus Π-distance.** The curve, not a scalar. This curve *is* the calibration of `τ_near`
  and `τ_far` in [`03`](03-physics.md) §3.4, so it must be measured, not assumed.
- **Regime-boundary crossings.** Held-out sets that specifically cross `Fr = 1`,
  laminar→turbulent, and `Ma = 1`. The failure to watch for is smooth interpolation across a
  bifurcation — a model that produces no hydraulic jump because it never saw one is producing a
  confident, plausible, wrong answer.
- **The raw-SI control.** A model trained without nondimensionalization, evaluated identically. If
  the curves coincide, Π-space is not earning its place (kill criterion 2).
- **Escalation calibration.** Over the query distribution: how often does the policy escalate, how
  often *should* it have? False-negative escalations — Tier A answering when it should have solved —
  are the dangerous direction and should be reported separately from false positives.

## 11.4 Verification-behavior evaluation

A capability this document treats as first-class: the model's ability to describe its own epistemic
state. Scored explicitly, because a system that cannot do this is not a scientific instrument.

- **Provenance completeness.** Does every physics report carry tier, Π-distance, residuals, order of
  accuracy, and an explicit statement of what was not modelled ([`04`](04-generation.md) §4.4)?
- **Calibration.** Does stated confidence track measured error?
- **Non-convergence reporting.** When a solve fails its gates, is that surfaced or smoothed over?
  Tested with deliberately ill-posed cases where the correct answer is "this did not converge."
- **Internal consistency.** Do the numbers in the report agree with the field they describe, and with
  each other? This is a direct, cheap arithmetic check, and it is the one that catches the class of
  error catalogued in [`12`](12-blueprint-reconciliation.md) §12.3 — mutually inconsistent quantities
  in a fluent, confident analysis.

## 11.5 Cross-modal transfer — the test of the premise

Kill criterion 9 needs a measurement, and it must be cheap enough to run before the expensive stages.

- **Per-modality versus specialist controls.** Does the monolith trail specialists, and by how much?
  Some deficit is expected and acceptable.
- **Composed tasks versus a pipeline of specialists.** `field → render → report → edit`, end to end,
  against the modular alternative wired together conventionally. This is where the monolith must win,
  because it is the only place it can.
- **Ablation of shared representation.** Same components, per-modality coordinate frames. If
  performance is unchanged, the shared frame is decoration.

## 11.6 Component evaluations

- **Agentic.** Long-horizon hybrid GUI+CLI benchmarks, plus 3D-specific scene-construction tasks.
  Expect roughly 40% and report it plainly. Track *safe failure* separately from success: how often
  does the model stop and report versus persist into a wrong state?
- **Depth allocation.** Correlation between requested budget and actual difficulty; the
  train/serve routing consistency test ([`02`](02-trunk.md) §2.3) as a release blocker; depth
  distribution monitored for halting collapse.
- **Runtime.** Interactive tail latency with and without concurrent deep-ponder streams (kill
  criterion 7). Cross-stream isolation verified adversarially — attempt to extract another stream's
  content and confirm the mask holds.
- **Equivariance ablation.** The control from [`01`](01-representation.md) §1.2, evaluated on the
  same suite, reporting the delta at each compute budget rather than at one.

## 11.7 What "working" would mean

Stated concretely so it can be disagreed with:

1. Conservation metrics on rendered output are **orders of magnitude** better than on
   conditionally-generated output, not marginally better.
2. Tier A error versus Π-distance is measurably flatter than the raw-SI control, and the escalation
   policy's false-negative rate is low enough to trust on unfamiliar queries.
3. Composed cross-modal tasks beat a specialist pipeline, even where individual modalities do not.
4. Every physics output carries provenance, and calibration holds.
5. Agentic reliability is around 40% and the system *fails safely* at the other 60%.

Item 5 is not a success criterion anyone will enjoy reading. It is the honest one.
