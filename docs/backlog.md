# Implementation Backlog

Milestones are dependency gates, not a schedule. Each names the evidence that closes it.

| ID | Deliverable | Acceptance evidence | Depends on | Status |
|---|---|---|---|---|
| M0 | Corrected Iridium-1 specification and decision log | Source claims classified; contradictions removed; interfaces agreed | — | **done** |
| M1 | Events, exact arrays, units, frames, manifests | Round-trip, malformed-input, ownership and conversion tests | M0 | **done** |
| M2 | Small dense transformer, fixed recurrence | Forward/backward, parameter inventory, mask and cache parity | M1 | **done** |
| M3 | Native symbolic + image generation | Real training run, held-out outputs, no external model calls | M2 | **partial** |
| M4 | Native audio/video and mixed output streams | Decodable outputs, temporal tests, interruption fixtures | M3 | open |
| M5 | Native field prediction and conservative correction | Manufactured solutions, held-out trajectories, budgets, solver-only comparison | M2, M1 | **partial** |
| M6 | Inverse observations and uncertainty | Held-out views/sensors, ambiguous-case behaviour, calibration | M4, M5 | open |
| M7 | Persistent scheduler and memory | Mixed-load, isolation, recurrence change, restart recovery | M2–M6 | open |
| M8 | Blender transactions and software tasks | Reopenable scenes, verified edits, stale-state and rollback tests | M1 + action training | open |
| M9 | Waterfall and coupled demonstrations | End-to-end packages with linked state, media, reports, evidence | M5–M8 | open |
| M10 | Optional improvement loop | Candidate comparison, sealed evaluation, rejection and rollback | M9 | open |
| M11 | Larger-scale training decision | Scaling evidence, resource profiles, unresolved-risk register | earlier gates | open |

## What "partial" means

**M3** — one native continuous head (the diagnostic image) is trained by conditional flow
matching inside the shared trunk, with a measured quality-versus-steps curve. Text generation
is not trained; the instruction is consumed, not produced. Image editing and conditioning
dropout are not implemented.

**M5** — the conservative face-flux correction is trained and its conservation property is
measured on the trained model. Manufactured-solution convergence and open-system budgets are
tested. Missing: multi-step rollout, non-periodic boundaries, nonlinear systems, differentiable
solver-in-the-loop, and a calibrated risk assessment.

## Next milestone and its exact acceptance evidence

**M5 completion — native field prediction beyond one linear family.**

1. **Rollout.** Autoregressive multi-step prediction to a horizon at least 8× the single-step
   horizon, reporting error growth versus step count against the exact solution. Acceptance:
   the curve is reported with confidence intervals over ≥3 seeds; no threshold is pre-committed
   because none has been calibrated.
2. **Non-periodic boundaries.** Dirichlet and outflow conditions on the same equation, with the
   open-system budget closing to the configured tolerance on every episode. Acceptance:
   `Budget.closes(tolerance=1e-8, floor=declared)` for 100% of a 256-episode held-out split.
3. **Nonlinear system.** Burgers or shallow water, where the learned correction has real work to
   do. Acceptance: flux head beats both persistence and the same-cost upwind reference on a
   held-out split, reported per split, with the conservation property retained at ≤1e-8 relative
   drift.
4. **Solver-only control.** The same tasks answered by the numerical reference alone, so the
   native contribution is separable from the tool contribution. Acceptance: a table with three
   arms — native, solver-only, native+solver — on identical episodes.
5. **Ablation.** Conservative flux head versus direct head at matched parameters and steps, and
   fixed depth 1/2/4 at matched measured compute. Acceptance: both reported, whichever way they
   come out.

Gate 3 is the one that matters. The current slice *does* beat the upwind reference in
distribution (0.056 vs 0.113) but collapses out of it (1.48 vs 0.21) — see
[first-slice.md](first-slice.md). Until a native head holds that advantage on a **nonlinear**
system, and degrades gracefully rather than catastrophically off-distribution, "native
physical competence" remains a narrow result rather than a capability.

## Cross-cutting work items not tied to one milestone

- **Isolation beyond attention** (D09): cache allocation, memory retrieval, logging, object
  storage and action handles each need their own boundary and adversarial test. Attention is
  tested; these are not.
- **Spectral axis typing** (D02): axis kind is in the schema but no test distinguishes optical
  wavelength from audio frequency from mode index.
- **Renderer provenance** (Invariant 3): exact-array survival is tested for the store, not
  through a rendering path.
- **Equivariance ablation** (D03): constrained geometric attention versus plain tensor channels,
  at matched compute, before anything claims equivariance.
- **Train/serve stopping-policy parity** on a real trained halting head: the semantics are
  tested, the trained behaviour is not.
