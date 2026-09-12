# 03 — Physics: Two Tiers

> The model does not learn what fluids do by watching them. It is differentiated *through*
> conservation, and when it leaves the region where that training holds, it stops guessing and
> solves.

## 3.1 Why not just learn the dynamics

Because it has been tried and measured. Under a 10× Reynolds shift a trained FNO reaches ~46.7%
relative L₂ error and is beaten by retrieval; better representations improve this only to ~35%; and
~12 percentage points of the error come from autoregressive drift alone, independent of
representation quality. No tested method demonstrated learned Reynolds-invariance.

A model that answers "double the inflow" from a learned prior is not calculating. It is producing
the most plausible-looking continuation, which is exactly the behavior that makes generated video
violate mass conservation at ~40%.

So physics enters ORRERY through two mechanisms, neither of which is "predict the next field."

## 3.2 Tier A — the differentiable solver, fused into the graph

A differentiable numerical integrator is a **layer inside the trunk's forward pass**, with gradients
flowing through it during training.

**Solver family, by regime:**

| Regime | Integrator |
|---|---|
| Incompressible flow | semi-Lagrangian advection + pressure projection (Chorin splitting) |
| Free surface | level-set / VOF advection with reinitialization |
| Granular, elastic, plastic | Material Point Method |
| Rigid contact | linear complementarity formulation with a smoothed complementarity relaxation |
| General PDE fallback | method-of-lines spatial discretization + differentiable implicit step |

**The critical inversion:** the trunk does **not** predict the next state. It predicts

1. a **correction** to a coarse solve — the residual between the cheap integrator and the truth, and
2. the **solver's own parameters** — subgrid/closure coefficients, effective viscosity, timestep,
   reinitialization frequency.

Error is therefore bounded by the *consistency of the integrator*, not by the network's memory of
similar-looking flows. A coarse solve that conserves mass to machine precision, plus a learned
correction, cannot silently invent matter — which is the failure mode the whole document is organized
around. This is the direct structural attack on the ~12pp drift bottleneck: drift accumulates in the
correction, which is small and mean-reverting, rather than in the state, which is not.

**Why this is the real source of "native physics understanding."** Because gradients pass through
the pressure projection, the trunk's weights are shaped by the actual structure of incompressibility
— the divergence-free constraint is in the loss landscape, not in the training captions. A model
trained this way has representations organized by conservation law. A model trained on rendered
videos has representations organized by appearance.

**Constraint check (C1):** a numerical integrator is a program. No second neural network is invoked.

### Spectral operator sublayers — useful, and bounded

Fourier-neural-operator sublayers sit inside the trunk as a fast, grid-resolution-independent
estimator for smooth field evolution, and an FFT-based Poisson solve is genuinely `O(N log N)`. They
earn their place in Tier A's coarse pass. Three limits are attached to them permanently, because
each one bites precisely where this system's canonical query lives:

- **Mode truncation is a low-pass filter on the answer.** Spectral layers keep the lowest `K` modes.
  Spray, air entrainment, the turbulent cascade, and the Kolmogorov microscale are high-wavenumber
  phenomena. A spectral operator keeping a handful of modes per axis cannot represent them, and FNOs
  carry a documented spectral bias toward low frequencies even before truncation.
- **Boundaries are where the method is weakest and the physics is most interesting.** FNO is not
  strictly restricted to periodic domains — the pointwise linear term recovers some non-periodic
  behavior — but performance is documented to degrade once periodicity fails, and non-periodic data
  through a spectral transform produces Gibbs oscillations. A waterfall is walls, an inlet, an outlet,
  and a free surface.
- **It is the architecture with the measured extrapolation failure.** The ~46.7% relative L₂ error
  under a 10× Reynolds shift that motivates this entire section was measured on an FNO.

Spectral layers therefore produce a *coarse smooth estimate that the trunk corrects*, and they are
never the final authority on a query outside the validated range. Treating the spectral stack as
"the solver" reintroduces the failure the two-tier design exists to avoid.

**Limits:** Tier A covers what is differentiably implementable and numerically well-behaved. Stiff
systems, shocks, high-Re turbulence, and contact-rich scenes either destroy the gradient or require
timesteps that make in-graph solving impractical. That is what Tier B is for.

## 3.3 Tier B — external verified solve

For regimes Tier A cannot honestly reach, the model **writes and runs real solver code** as an
agentic action: mesh generation, case setup, boundary conditions, execution, post-processing.
Candidate backends are ordinary numerical software — NVIDIA Warp, PhiFlow, OpenFOAM, SU2 — driven
through their own interfaces.

Returned fields re-enter through the same coordinate-stamped encoder of
[`01`](01-representation.md) with `provenance = verified_solve`. There is no separate "tool result"
pathway; a solved field is just a field, distinguishable only by its provenance stamp.

**Constraint check (C1):** OpenFOAM is not a model. This is the "code and APIs" the constraints
explicitly permit, and it is the only route by which the word *accurate* is defensible.

### Verification gates — a solve is not trusted because it finished

| Gate | Check | Fail action |
|---|---|---|
| Continuity residual | ∇·u below solver tolerance across the domain | reject; refine and re-run |
| Momentum residual | per-cell imbalance below tolerance | reject |
| Global conservation | total mass/momentum/energy drift over the run within bounds | reject |
| Grid convergence | Richardson extrapolation over ≥3 refinements; observed order of accuracy consistent with the scheme | flag as unconverged; report explicitly |
| Boundary sanity | no reversed flow at outlets, no unphysical negative densities or volume fractions | reject |
| Wall-clock/step budget | did the solve terminate or was it truncated | report as partial |

A result that fails a gate is never quietly returned as an answer. It is either re-run with a refined
setup or surfaced as a failure with the residual attached. **The model is permitted to say the solve
did not converge.** Systems that cannot say this are the ones that fabricate.

## 3.4 The escalation policy — where the canonical query is actually decided

The whole design turns on one measured quantity: **distance in Π-space from the training manifold.**

```
1. Extract governing dimensionless groups Π = (Re, Fr, We, Ma, ...) from the scenario's
   dimension-stamped latents  [01]
2. Apply the requested intervention as a transformation in Π-space
   ("double the inflow" → a coupled move along Re and Fr, with a free-surface height change)
3. d := distance from Π_new to the training manifold, in the metric below
4. Route:
       d < τ_near      → Tier A alone
       τ_near ≤ d < τ_far → Tier A, then Tier B audit on conserved quantities;
                            report calibrated disagreement
       d ≥ τ_far       → Tier B, full verified solve
5. Regardless of tier: report d, the tier used, and the residuals            [04]
```

**The metric.** Not Euclidean distance in raw Π — a factor of 2 in Reynolds means something different
at Re=10 than at Re=10⁶. Distance is measured in **log-Π space**, weighted by the local density of
training coverage (a normalizing-flow density estimate over the training set's Π-points), with an
additional term for *regime-boundary crossings*: transitions like laminar→turbulent, subcritical→
supercritical (Fr crossing 1), or subsonic→supersonic are discontinuities, and crossing one sets
`d ≥ τ_far` regardless of numerical proximity.

That last clause matters for the waterfall. Doubling the inflow can push the flow across the Froude-1
critical transition, which changes the *character* of the solution — a hydraulic jump appears. No
amount of interpolation in a learned prior produces a hydraulic jump that was not in the training
data. The regime-boundary term is what stops the model from smoothly interpolating past a
bifurcation, which is the single most dangerous failure available to a physics surrogate.

**Calibration of τ.** Both thresholds are fit empirically, not chosen: run Tier A and Tier B on a
held-out sweep of Π-points, measure Tier A's error as a function of `d`, and set `τ_near` where error
exceeds the application tolerance and `τ_far` where it exceeds the useful-audit range. This makes the
policy a measured property of the trained model rather than a hyperparameter someone picked.

## 3.5 What happens with "double the water"

Working the canonical query through the policy:

1. The uploaded sim's inlet is a boundary-condition latent ([`01`](01-representation.md) §1.6).
   Doubling the inflow is an **edit to that latent**, not a text instruction to be interpreted.
2. Π-extraction gives the original `(Re, Fr, We)`. The edit propagates: Re roughly doubles for a
   fixed channel; Fr changes with both velocity and depth, and the depth response is itself part of
   the solution — so the model cannot compute the new Fr analytically. It must bound it.
3. That bound almost always straddles or approaches Fr = 1 for a waterfall geometry. Regime-boundary
   term fires. `d ≥ τ_far`.
4. **Tier B.** The model writes a free-surface case, runs it, checks continuity and momentum
   residuals, runs three mesh refinements for a Richardson estimate.
5. The solved field is stamped `verified_solve` and re-enters the trunk.
6. [`04`](04-generation.md) renders it and writes the report, which states: Tier B, the residuals,
   the observed order of accuracy, and that the query sat outside the surrogate's validated range.

The honest headline: **for this query the model runs a real simulation, and that takes real time.**
Anything that returns a physics-accurate waterfall video instantly is returning something else.

## 3.6 Deliberate non-goals

- **No claim of learned Reynolds-invariance.** The spec assumes it does not exist and designs the
  escalation policy on that assumption.
- **No unbounded autoregressive rollout.** Long horizons are Tier B, or Tier A with periodic Tier B
  re-anchoring. A surrogate rolled out indefinitely is a drift generator.
- **No physics from video.** Video is an *output* of the physics path ([`04`](04-generation.md)),
  never a source of dynamics. Learning fluid behavior from rendered clips is how models end up
  violating mass conservation 40% of the time.
