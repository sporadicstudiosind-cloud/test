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
| Superstacks with sparse stack-local KV + core bridge | implemented | `test_kv_parity.py` (agrees to FP64 rounding, 1e-13, at one loop) |
| PonderNet depth ladder inside a stack | implemented | `test_torch_stopping_matches_the_numpy_contract` |
| Span-coherent routing keeps field grids whole | implemented | `grid_intact_fraction`: 0% per-token, 100% span-coherent |
| Spectral operator, correct corner blocks, 1/2/3-D | implemented | shift-equivariant to 3e-16; exactly resolution-invariant |
| Parameter/memory accounting to 9 T | implemented | formula == modules, delta 0 (`test_config_inventory.py`) |
| Omnimodal codecs: 8 modalities in and out | implemented (no direct test) | `iridium/codecs/spans.py` (`patchify` / `unpatchify`). No `test_model.py` exists, and no patchify round-trip test was found |
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
| Resource-limited sandbox | implemented (partly tested) | `iridium/runtime/sandbox.py`. Tested: timeout and platform run (`tests/unit/test_iridium_50m.py`). Memory, network and env-scrub limits have no test |
| Sandbox with seccomp / filesystem namespace | **unsupported** | `SandboxResult.isolation` records their absence |
| Per-stream isolation, no cross-stream state | implemented | `tests/integration/test_isolation.py::test_a_streams_output_does_not_depend_on_its_neighbours` (bit-identical with and without neighbours) |
| Focus-weighted scheduling with fairness floor | implemented (no direct test) | `iridium/runtime/scheduler.py`, run through the service in `test_isolation.py`. Budget conservation and the per-owner cap are not tested directly |
| Typed action space; deterministic scene editor | implemented (no direct test) | `iridium/agency/scene.py`: `to_blender` maps each op to a bpy call; no test exercises the mapping |
| Real Blender actuation (`bpy` in the loop) | specified | script emitted, never executed here |
| MXFP4 / FP8 quantization, measured | implemented | 4.25 bits/param; SQNR 18.9 / 31.5 dB (`iridium quant`; bit width in `test_config_inventory.py`) |
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
| Isolation beyond attention (cache, memory, logs, storage, handles) | specified | Partly tested: cache release and no shared tensor storage between streams (`tests/integration/test_isolation.py`). Memory, logs and handles are untested; **not** claimed as done |

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
| Text and chat generation | trained | `chat-34m` run, see the training path below |
| Action (opcode) generation | trained | nano phase-1 run: first-opcode accuracy 0.000 → 1.000 on 16 items ([`results.md`](results.md)) |
| Code generation | unsupported | `iridium-50m` trains on code families (`tests/unit/test_iridium_50m.py`); no code output is measured |
| Conditioning dropout and classifier-free guidance | implemented | `iridium/model/heads.py` (`cfg_dropout`, `guidance_scale`); `tests/unit/test_heads.py::test_cfg_dropout_trains_null_branch_and_guidance_changes_output` |
| Image editing, inpainting | unsupported | — |
| Video, speech, sound, long-horizon planning | unsupported | generation only; `iridium-50m` has image and audio input encoders ([`iridium-50m.md`](iridium-50m.md)) |
| Scientific rendering with artifact lineage | specified | — |
| Evidence-level gating in the manifest | implemented | schema conditionals in `result_manifest.v1.json` |
| Numeric reconciliation of report claims | implemented | `tests/unit/test_metrics.py`, incl. the real contradiction fixture |
| Dependency staleness on scenario revision | implemented | `check_dependency_staleness` |

## Runtime, agency, improvement

| Capability | Status | Evidence |
|---|---|---|
| Resident service: one model, many isolated streams | implemented | `iridium/runtime/service.py`; `tests/integration/test_isolation.py`. Cooperative and single-threaded |
| Continuous batching, paged cache | specified | — |
| Compute request → broker arbitration | specified | request schema in arch §8.3; no broker |
| Scheduling classes, interactive floor, preemption | specified | — |
| Full duplex, clocks, jitter, interruption | specified | — |
| Durable memory, restart, idempotent replay | specified | — |
| Blender bridge, transactions, postconditions, recovery | specified | `action.v1.json`, and `iridium/agency/scene.py` (`to_blender`); no transactions or recovery |
| GUI/CLI action execution | unsupported | — |
| Inverse problems from partial observation | unsupported | — |
| Active experiment proposal | unsupported | research |
| Self-improvement loops (all three) | specified | candidate-research controller in `iridium/research/` (controller, policy, worker); no test covers it |
| Scenarios A–G | unsupported | targets, see [`scenarios.md`](scenarios.md) |

---

## Iridium 1.0 training path

| Capability | Status | Evidence |
|---|---|---|
| `chat-34m` text and chat | trained | one recorded run, 20k steps: validation 1.19 bpb text, 0.87 chat (`CHANGELOG.md`). Not evaluated beyond that loss. The preset was later rebuilt as `chat-36m` |
| `iridium-50m` (four specialists, tools, image and audio input) | implemented | `tests/unit/test_iridium_50m.py`; a full-size step runs on CPU. A live Kaggle run is training it (`CHANGELOG.md`); no finished run or battery result is recorded in this repository |
| One run across several GPUs, gradients averaged | implemented | `test_two_processes_train_in_lockstep` (two CPU processes end on identical weights). Timed on Kaggle 2xT4: ~9.8k tok/s for both GPUs together (`CHANGELOG.md`) |
| Ponder loops trained at sampled depths | implemented | `test_sampled_loops_follow_the_probabilities_and_train` |
| Specialist routing curriculum | implemented | `tests/unit/test_specialist_curriculum.py`; a toy CPU experiment in [`iridium-50m.md`](iridium-50m.md). Untested at full size |
| Check that each stack wins its own subject | implemented | `test_specialist_matrix_overrides_routing_and_restores_it`; `python -m iridium specialists` |
| Thinking effort within ±10 loops of the user's choice | implemented | `tests/unit/test_loop_spread.py`. Loops past the trained three are extrapolation |
| Checkpoint per round, safe resume | implemented | `tests/unit/test_checkpointing.py` |
| Agent tools (files, Python, calculator, web) | implemented | workspace-escape tests; `run_python` is a guard rail, not a security boundary |
| Constitution: written principles and an output screen | implemented | `iridium/alignment/constitution.py`. No training data and no runtime guard; see [`constitution-plan.md`](constitution-plan.md) |
| NorMuon and Polar Express Muon (`muon_pe`, `normuon`) | evaluated (toy scale) | `tests/unit/test_muon.py`, `tests/unit/test_muon_variants.py`. 2.1M-parameter CPU ablation, held-out bytes, 1,200 steps, lr 6e-3: NorMuon 0.727 vs AdamW 0.861 ([`iridium-50m.md`](iridium-50m.md)) |
| Canon layers (`canon`) | evaluated (toy scale) | `iridium/model/layers.py` (`CanonConv`); option sets in `tests/unit/test_core_blocks.py`. Same ablation: 0.792 vs 0.861 with AdamW, but neutral under NorMuon (two-seed mean difference 0.001) |
| Gated attention (`attn_gate` = `head` / `elem`) | implemented | `iridium/config.py`, `iridium/model/core_blocks.py`; option sets in `tests/unit/test_core_blocks.py`. Toy ablation showed no measurable effect; off in every shipped preset |
| LittleBit sub-1-bit layers | implemented | `iridium/quant/littlebit.py`; `tests/unit/test_littlebit.py`. Quality loss is large at this model size (`CHANGELOG.md`) |
| `iridium cloze` (HellaSwag, ARC, OpenBookQA, PIQA as likelihoods) | implemented | `iridium/evaluation/cloze.py`; `tests/unit/test_cloze.py`. No scores are recorded in this repository |
| Teacher-written subject text (`iridium teacher generate` / `shard`) | implemented | `iridium/data/teacher_corpus.py`; `tests/unit/test_teacher_corpus.py` |
| Same-subject sequence packing | implemented | `iridium/training/datasets.py` (`BatchLoader`, packed batches); `tests/unit/test_next_run_features.py` |
| Parallel `data prepare` | implemented | `iridium/training/prepare.py`; `tests/unit/test_prepare_speed.py`. Output is byte-identical to the serial path (`CHANGELOG.md`) |
| Control presets `iridium-50m-control-random` / `-dense` | implemented | `iridium/presets.py`. The experiment they exist for has no recorded result |
| Free-after-first-loop routing (`curriculum_first_loop_only`) | implemented | `iridium/config.py`, `iridium/model/iridium1.py`; `tests/unit/test_specialist_curriculum.py`. Opt-in; untested at scale |
| Model council (several instances debating) | specified | design only |

---

## Summary

| Status | Count (rows in the tables above) |
|---|---|
| implemented (of which 4 are partly tested or lack a direct test) | 69 |
| trained | 3 |
| trained and evaluated | 3 |
| evaluated (toy-scale rows included) | 3 |
| specified only | 18 |
| unsupported | 14 |

The gap between "specified" and "trained" is the honest shape of this project. The
specification is comprehensive; the implementation covers contracts, the model reference,
conservation mechanics and one narrow trained slice. Claims from the first three columns
should not be read as evidence for the last two.
