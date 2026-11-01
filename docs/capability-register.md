# Capability Register

Five statuses, and they are not interchangeable:

- **specified** — written down normatively, no code
- **implemented** — code exists and is covered by an automated test
- **trained** — learned weights were actually fit to data
- **evaluated** — measured on a held-out split against a baseline, result recorded
- **unsupported** — not built; a target, not a feature

A capability is only listed at a level when the evidence for that level exists in this
repository. "Implemented" never means "works well"; it means the property is mechanically
checked.

---

## Contracts and representation

| Capability | Status | Evidence |
|---|---|---|
| Event envelope with schema validation | implemented | `tests/unit/test_events.py` (accept + reject) |
| Exact array store, bit-exact round trip | implemented | `test_exact_array_round_trip_is_bit_exact` |
| SI dimensions, rational exponents, affine units | implemented | `tests/unit/test_units.py` |
| Non-numeric vs dimensionless distinction | implemented | `test_non_numeric_is_not_dimensionless` |
| Per-channel field units | implemented | `test_channel_schema_carries_per_channel_units` |
| Typed frame graph, composition, round trip | implemented | `tests/unit/test_frames.py` |
| Pixel → ray (not to a point) | implemented | `test_pixel_maps_to_a_ray_not_a_point` |
| World velocity vs screen motion separation | implemented | `test_world_velocity_and_screen_motion_have_different_units` |
| Event time vs arrival time, causal admission | implemented | `test_causality_rejects_events_that_had_not_arrived` |
| Instruction vs observed/generated authority | implemented | `test_source_role_gates_instruction_authority` |
| Spectral axis typing (wavelength ≠ audio freq ≠ mode) | specified | schema field only; no test — backlog |
| Mesh topology, boundary types, discretization conversions | specified | — |

## Routed architecture (control core + superstacks)

| Capability | Status | Evidence |
|---|---|---|
| Control core, two stages, indexed ponder loop | implemented | `tests/integration/test_kv_parity.py` |
| Causal macro-router: top-k stacks, focus, halting | implemented | `tests/unit/test_router.py` |
| Superstacks with sparse stack-local KV + core bridge | implemented | `test_kv_parity.py` (bit-exact at one loop) |
| PonderNet depth ladder inside a stack | implemented | `test_torch_stopping_matches_the_numpy_contract` |
| Span-coherent routing keeps field grids whole | implemented | `grid_intact_fraction`: 0% per-token, 100% span-coherent |
| Spectral operator, correct corner blocks, 1/2/3-D | implemented | shift-equivariant to 3e-16; exactly resolution-invariant |
| Parameter/memory accounting to 9 T | implemented | formula == modules, delta 0 (`test_config_inventory.py`) |
| Omnimodal codecs: 8 modalities in and out | implemented | `tests/integration/test_model.py`, patchify round-trip exact |
| Flow-matching continuous heads | implemented | trains; sample quality not evaluated |
| Capacity-based token dropping | **unsupported** | refused as non-causal (`test_capacity_dropping_is_refused_as_non_causal`) |
| Per-token ponder granularity (copy-through KV) | specified | `model/iridium1.py` docstring; chunk-uniform implemented instead |
| Distributed dispatcher across nodes | specified | `parallel/plan.py` costs it; no implementation |
| Any capability at micro/small/base/extreme rungs | **unsupported** | costed, never built |

## Physics, runtime and agency

| Capability | Status | Evidence |
|---|---|---|
| Pseudo-spectral incompressible Navier-Stokes | implemented | Taylor-Green to 6e-15 relative; divergence 1e-14 |
| Leray projection as a hard constraint | implemented | idempotent to 7e-16 (after the Nyquist fix, F-15) |
| Open-channel flow; Manning and critical depth | implemented | converges to analytic normal depth from both sides |
| The doubling intervention, end to end | evaluated | measured 1.515647 vs analytic 1.515717 |
| Dual-system escalation with an evidence ladder | implemented | `physics/verifier.py`; drifting prediction escalates |
| Resource-limited sandbox | implemented | timeout, memory, network, env-scrub all blocked |
| Sandbox with seccomp / filesystem namespace | **unsupported** | `SandboxResult.isolation` records their absence |
| Per-stream isolation, no cross-stream state | implemented | output bit-identical with and without neighbours |
| Focus-weighted scheduling with fairness floor | implemented | budget conserved; per-owner cap holds |
| Typed action space; deterministic scene editor | implemented | every op has a checked bpy translation |
| Real Blender actuation (`bpy` in the loop) | specified | script emitted, never executed here |
| MXFP4 / FP8 quantization, measured | implemented | 4.25 bits/param; SQNR 18.9 / 31.5 dB |
| Report reconciliation against an exact metric table | implemented | malformed markers now fail the gate too |

## Model (dense reference, retained)

| Capability | Status | Evidence |
|---|---|---|
| Dense recurrent core, prelude/core/coda, GQA, SwiGLU | implemented | `tests/integration/test_core_reference.py` |
| No expert router in the deployed graph | implemented | `test_no_expert_router_exists` (module-graph check) |
| `H0` reinjection + recurrence conditioning | implemented | `test_recurrence_changes_the_output` |
| Causal mask, perturbation-tested | implemented | `test_causal_mask_blocks_the_future` |
| Noised output-block mask (bidirectional inside, closed outside) | implemented | `test_output_block_mask_is_bidirectional_inside_and_closed_outside` |
| Block-diagonal workspace isolation (attention only) | implemented | `test_workspace_mask_isolates_concurrent_streams` |
| Recurrence-indexed cache; parity with uncached reference | implemented | `test_cached_incremental_decode_matches_uncached_reference` (FP64) |
| Depth change invalidates cache | implemented | `test_changing_depth_invalidates_the_cache` |
| Stopping distribution, per-stop loss, two serving policies | implemented | `tests/unit/test_halting.py` |
| Trained halting head / adaptive depth | unsupported | semantics tested; no trained stopping behaviour |
| Parameter and cache inventory at three scales | implemented | `tests/unit/test_inventory.py` |
| Isolation beyond attention (cache, memory, logs, storage, handles) | specified | backlog; **not** claimed as done |

## Physics

| Capability | Status | Evidence |
|---|---|---|
| Conservative face-flux update (telescoping) | implemented | `test_learned_face_correction_still_conserves` |
| Demonstration that per-cell corrections do **not** conserve | implemented | `test_learned_cell_correction_does_not_conserve` |
| Open-system budget with normalization floor | implemented | `test_open_domain_stores_mass_without_violating_conservation` |
| Manufactured solution, observed order over 4 refinements | implemented | `test_manufactured_solution_converges_at_first_order` |
| Admissibility / bound checking | implemented | `test_admissibility_reports_worst_violation` |
| Dimensionless group algebra, inlet table | implemented | `tests/scientific/test_inlet_arithmetic.py` |
| **Learned** field prediction, single step, one linear family | trained + evaluated | [`first-slice.md`](first-slice.md) — beats upwind in distribution (0.056 vs 0.113), collapses outside it (1.48 vs 0.21) |
| Learned conservation preserved on the trained model | trained + evaluated | 9.5e-09 relative drift vs 1.8e-02 for the unconstrained control |
| Multi-step rollout, non-periodic BCs, nonlinear systems | unsupported | next milestone |
| Differentiable solver in the graph (constrained mode) | unsupported | — |
| Reference mode: writing and running a real solver case | unsupported | — |
| Verification gates on a real solve; grid convergence study | specified | mechanics tested on fixtures, not on a solver run |
| Joint risk assessment and calibrated escalation | specified | — |
| Multiphysics coupling | specified | schema only |

## Generation

| Capability | Status | Evidence |
|---|---|---|
| Conditional flow matching, native continuous head | trained + evaluated | quality-vs-steps curve in [`first-slice.md`](first-slice.md) |
| Step count chosen by measurement, not promise | implemented | `image_quality_vs_steps` |
| Noise time τ distinct from physical time | implemented | separate embedding; distinct axes in the slice |
| Text / code / action generation | unsupported | instruction is consumed, never produced |
| Image editing, inpainting, conditioning dropout, guidance | unsupported | — |
| Video, speech, sound, long-horizon planning | unsupported | — |
| Scientific rendering with artifact lineage | specified | — |
| Evidence-level gating in the manifest | implemented | schema conditionals in `result_manifest.v1.json` |
| Numeric reconciliation of report claims | implemented | `tests/unit/test_metrics.py`, incl. the real contradiction fixture |
| Dependency staleness on scenario revision | implemented | `check_dependency_staleness` |

## Runtime, agency, improvement

| Capability | Status | Evidence |
|---|---|---|
| Resident service, continuous batching, paged cache | specified | — |
| Compute request → broker arbitration | specified | request schema in arch §8.3; no broker |
| Scheduling classes, interactive floor, preemption | specified | — |
| Full duplex, clocks, jitter, interruption | specified | — |
| Durable memory, restart, idempotent replay | specified | — |
| Blender bridge, transactions, postconditions, recovery | specified | `action.v1.json` only |
| GUI/CLI action execution | unsupported | — |
| Inverse problems from partial observation | unsupported | — |
| Active experiment proposal | unsupported | research |
| Self-improvement loops (all three) | specified | — |
| Scenarios A–G | unsupported | targets, see [`scenarios.md`](scenarios.md) |

---

## Summary

| Status | Count (approx.) |
|---|---|
| implemented (test-backed) | 28 |
| trained | 3 |
| evaluated on held-out data | 3 |
| specified only | 19 |
| unsupported | 15 |

The gap between "specified" and "trained" is the honest shape of this project. The
specification is comprehensive; the implementation covers contracts, the model reference,
conservation mechanics and one narrow trained slice. Claims from the first three columns
should not be read as evidence for the last two.
