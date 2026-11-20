# Decision Log

Every normative change from the archived ORRERY specification to Iridium-1, with the
reason and the test or labelled experiment that holds it in place.

Status values: **tested** (an automated test fails if the decision is violated),
**schema-enforced** (a JSON Schema rejects the violating document), **specified** (written
down, not yet mechanically checked), **research** (an open question with a named
experiment).

The archived specification is preserved unaltered in [`history/orrery/`](history/orrery/).
It is historical reference, not a normative source.

---

## Part 1 — Corrections to ORRERY

### D01 · One coordinate frame replaced by a typed frame graph
**Was:** a single `(t, x, y, z, λ)` frame was treated as sufficient for all modalities, so a
video pixel and a fluid cell shared a coordinate.
**Now:** a graph of typed frames with explicit observation operators. A pixel identifies a
**ray**; recovering a point requires depth from geometry or an observation.
**Why:** the single-frame design silently assumes the inverse imaging problem is solved. It
is not, and pretending otherwise makes every downstream reconstruction claim unfalsifiable.
**Status:** tested — `test_frames.py::test_pixel_maps_to_a_ray_not_a_point` shows many world
points projecting to one pixel; `test_missing_path_raises_rather_than_guessing` shows an
absent transform raising instead of being invented.

### D02 · One spectral coordinate split into typed axes
**Was:** a single `λ` axis equated optical wavelength, audio frequency and spectral mode index.
**Now:** axis kind, units, basis and transforms are stored explicitly; these are different
coordinates that happen to be called "frequency" in English.
**Why:** they have different dimensions and different transformation laws. Sharing a slot
invites a model to transfer structure between them that does not exist.
**Status:** specified (frame kinds include `spectral`); per-axis metadata is in the event
schema. Not yet exercised by a dedicated test — listed in the backlog.

### D03 · Cl(3,1) equivariance claim scoped down
**Was:** a spacetime Clifford algebra was credited with simultaneous Euclidean, Galilean and
Lorentz equivariance across an arbitrary mixed transformer.
**Now:** scalar/vector/tensor channels with unit-aware transforms are the default.
Constrained Euclidean geometric attention is an experimental block. A Lorentz-equivariant
block is a domain extension with its own metric and tests. A claimed equivariance must hold
through normalization, attention scores, coordinate features, nonlinearities, pooling, masks
and decoding.
**Why:** those three groups are different, no single algebra grants all of them to arbitrary
layers, and unconstrained dense projections over mixed grades destroy grade structure anyway.
The archived spec noted the projection problem and still kept the umbrella claim.
**Status:** research — the ablation is named in the backlog (M2-E). Nothing currently claims
equivariance, so nothing currently has to prove it.

### D04 · Shear rate restored to a tensor
**Was:** shear rate was assigned to a bivector slot alongside vorticity.
**Now:** symmetric strain-rate and stress stay tensors, separate from the antisymmetric
rotation part.
**Why:** they are different objects with different symmetry. Merging them loses the
decomposition that most constitutive models are written in terms of.
**Status:** specified.

### D05 · Non-numeric and dimensionless separated
**Was:** a zero exponent vector marked both text and dimensionless numbers.
**Now:** `units: null` means units do not apply; `units: {}` means a number with a zero
exponent vector. Affine units carry scale *and* offset.
**Why:** "this is not a number" and "this is a number with no dimension" support different
operations. A Reynolds number can be divided by another Reynolds number; a sentence cannot.
**Status:** tested — `test_units.py::test_non_numeric_is_not_dimensionless`,
`test_parse_dimension_distinguishes_null_from_empty`, and the affine round trips.

### D06 · Dynamic similarity requires more than matching a few groups
**Was:** matching Re, Fr, We made two flows "the same problem", and Π-space was presented as
converting an extrapolation cliff into a slope.
**Now:** normalization is a **conditioning aid**. Similarity additionally requires matching
equations, dimensionless geometry, boundaries, initial conditions, material laws and the
relevant groups.
**Why:** two flows can match every listed number and differ because a wall is rough or a
boundary family differs.
**Status:** tested as semantics — `test_frames.py::test_normalization_does_not_imply_similarity`
documents that identical normalized coordinates do not imply identical problems.

### D07 · Continuous latents are not lossless
**Was:** continuous (unquantized) field latents were treated as preserving physical data.
**Now:** two paths are maintained — compressed latents for reasoning, exact arrays for
computation and export. A nominally continuous representation is still finite precision and
frequently lossy. Codec conservation does not imply local fidelity.
**Why:** "continuous" is not "lossless", and the difference is exactly where a scientific
result would quietly degrade.
**Status:** tested — `test_events.py::test_exact_array_round_trip_is_bit_exact` and
`test_float_payload_survives_exactly`.

### D08 · Halting given a single coherent semantics
**Was:** ACT-style hidden-state averaging and a PonderNet geometric prior were combined and
described as one algorithm.
**Now:** an explicit stopping distribution, task loss trained at every stop, and two named
serving policies (seeded sampling, or a separately calibrated threshold rule) whose behaviour
matches training.
**Why:** averaging states and predicting a stopping time are different estimators. Training
one and serving the other is a silent distribution shift.
**Status:** tested — `test_halting.py` checks the distribution sums to one, that sampling
reproduces the training distribution within 2 points over 20,000 draws, and that the
threshold rule is deterministic and distinct.

### D09 · Attention sinks made workspace-private after contextualization
**Was:** shared mutable attention sinks were treated as globally safe.
**Now:** fixed token embeddings may be shared; contextualized sink states and caches are
workspace-private. A block-diagonal mask is necessary but **not sufficient** — cache
allocation, memory retrieval, logging, object storage and action handles need separate
isolation.
**Why:** a sink that has absorbed one workspace's content is that workspace's content.
**Status:** partially tested — `test_core_reference.py::test_workspace_mask_isolates_concurrent_streams`
covers attention by perturbation. The non-attention components are backlog items and are
listed as pending in the invariant table, not as done.

### D10 · Learned corrections must be conservative by construction
**Was:** "a coarse conservative solve plus a learned correction cannot invent mass."
**Now:** it can, if the correction is per cell. The correction must be a **conservative
face flux** on oriented shared faces, or a constrained correction solve, with boundary
budgets tested.
**Why:** this was the load-bearing claim of the archived physics section and it was false as
stated. Nothing ties one cell's correction to its neighbour's unless the correction lives on
the shared face.
**Status:** tested, both directions — `test_conservation.py::test_learned_face_correction_still_conserves`
holds drift below 1e-10 under a deliberately unphysical face correction, while
`test_learned_cell_correction_does_not_conserve` confirms the per-cell form breaks it. The
trained slice reproduces this with a *learned* correction (see [`first-slice.md`](first-slice.md)).

### D11 · Integrator consistency does not bound learned error
**Was:** "error is bounded by the consistency of the integrator."
**Now:** stability, admissibility, refinement behaviour and measured rollout error are all
required. Consistency alone is insufficient.
**Why:** consistency is a statement about truncation error as the step goes to zero. It says
nothing about a learned term at finite step size.
**Status:** tested for the numerical part —
`test_manufactured_solution_converges_at_first_order` measures an observed order rather than
asserting the nominal one. Learned rollout error is measured in the first slice and is
currently poor; that is reported, not hidden.

### D12 · Π-distance replaced by a joint risk assessment
**Was:** distance in Π-space alone selected the execution mode.
**Now:** a joint assessment over geometry, observations, regime, horizon, resolution and
residuals, calibrated on held-out families with false acceptance measured on an untouched
split.
**Why:** a scenario can be Π-adjacent and still fail on geometry or observation quality.
**Status:** specified; calibration is milestone M6.

### D13 · The uncertainty model counts as part of the model
**Was:** a normalizing-flow density estimator was added to drive escalation without being
counted against the one-model constraint.
**Now:** start with deterministic nearest-neighbour coverage statistics and calibrated error
tables. Any learned confidence head is Iridium-1's own weights, in its manifest.
**Why:** a separately trained density model is a second model. The constraint has to bind
even when it is inconvenient.
**Status:** specified.

### D14 · No universal claim about doubling inflow and Fr = 1
**Was:** doubling a waterfall inflow "almost always straddles or approaches Fr = 1", forcing
escalation.
**Now:** the boundary edit and the solved state determine the outcome. §11.2 gives a worked
counterexample where inlet Froude goes 0.479 → 0.958 and does **not** cross 1.
**Why:** it was an unsupported universal claim, and it happens to be false for the
straightforward fixed-area inlet.
**Status:** tested — `test_inlet_arithmetic.py::test_inlet_froude_does_not_cross_one`.

### D15 · Backflow at an outlet is not necessarily unphysical
**Was:** reversed flow at any outlet was a rejection condition.
**Now:** validate against the selected boundary formulation; some valid flows recirculate.
**Why:** a blanket rejection would reject correct solves of a large class of real problems.
**Status:** specified.

### D16 · Open-system budgets replace constant total mass
**Was:** total mass and momentum were expected to stay constant, including for a waterfall.
**Now:** account for inflow, outflow, forces, sources and storage change, with a declared
normalization floor.
**Why:** a waterfall domain is open. Stored mass rising while inflow exceeds outflow is
correct, and testing for constancy would fail a correct solve.
**Status:** tested — `test_open_domain_stores_mass_without_violating_conservation` and
`test_normalization_floor_prevents_meaningless_relative_error`.

### D17 · Rendering does not establish physical accuracy
**Was:** "the word *accurate* survives because the video is a view of a verified field."
**Now:** rendering preserves selected scene data. It cannot validate a wrong solver, closure,
boundary or inference. A visually faithful rendering of an incorrect field is incorrect
physics.
**Why:** the archived framing moved the accuracy claim into the renderer, where it cannot be
earned. Accuracy comes from verification and validation, and rendering only preserves what it
was given.
**Status:** specified; the evidence-level dependency is schema-enforced in
`result_manifest.v1.json`.

### D18 · Screen motion is not fluid velocity
**Was:** motion vectors were "the native intermediate" and equal to **u**.
**Now:** world velocity (m/s), world displacement (m), rigid transforms, mesh deformation,
camera motion and screen motion (px/frame) are distinct outputs with declared units,
visibility and direction conventions. Screen motion is a projection through a
time-dependent camera.
**Why:** they have different units and different values. Reflective or refractive water has
no single optical flow equal to material motion.
**Status:** tested — `test_frames.py::test_screen_motion_includes_camera_motion` and
`test_world_velocity_and_screen_motion_have_different_units`.

### D19 · Step counts are measured, not promised
**Was:** straight flow-matching paths "converge in 4–10 steps".
**Now:** conditional flow matching with independent pairing; step count chosen from measured
error. Calling it optimal-transport flow matching would claim a coupling that was not solved.
**Why:** neither straight paths nor flow matching guarantee a step count.
**Status:** tested as a measurement — `image_quality_vs_steps` sweeps and reports; the curve
for the trained slice is in [`first-slice.md`](first-slice.md).

### D20 · A shared clock does not synchronize audio and video
**Was:** a common temporal axis gave synchronization "by construction".
**Now:** explicit scheduling, buffering, clock correction and synchronization tests.
**Why:** shared labels do not align lips or impact sounds; jitter, drift and buffering do the
actual work.
**Status:** specified; M4.

### D21 · Physics learning from video is supported, not banned
**Was:** "no physics from video. Video is an output of the physics path, never a source of
dynamics."
**Now:** inverse problems and partially observed system identification with calibrated
uncertainty and explicit identifiability reporting. The model may infer candidates, request a
scale reference or propose an experiment; it may not present one reconstruction as uniquely
measured truth.
**Why:** the prohibition threw away a real capability to avoid a real failure mode. The
failure mode is ambiguity, and ambiguity can be represented rather than banned.
**Status:** specified; M6.

### D22 · Self-improvement claims made about this system only
**Was:** universal claims that no functioning system updates its own weights through
improvement.
**Now:** Iridium-1's own gated training and deployment policy, with three separated loops.
**Why:** a claim about every system everywhere is not a design decision and cannot be
verified.
**Status:** specified.

### D23 · Published benchmark scores are not Iridium-1 targets
**Was:** an external agent benchmark score was carried as Iridium-1's expected ceiling and an
acceptance target.
**Now:** measure task-specific completion and recovery on this system's own fixtures. A
different system's score on a different benchmark is not a prediction about this one.
**Why:** it imported someone else's measurement as this project's forecast.
**Status:** specified; M8.

### D24 · Conservation is not recoverable from arbitrary RGB
**Was:** an evaluation plan that measured mass conservation from generated clips.
**Now:** measure underlying states where known, and report the ambiguity of image-based
physical inference where not.
**Why:** total mass is not identifiable from arbitrary RGB. The proposed measurement could
not have been made.
**Status:** specified; M6.

---

## Part 2 — Errors in this project's own earlier analysis

The archived `12-blueprint-reconciliation.md` critiqued a third-party blueprint. Parts of
that critique were wrong. Recording it here because a decision log that only corrects other
people's documents is not doing its job.

### D25 · The rfft corner-block count was wrong
**Claimed:** standard 3-D FNO implementations write "four (2-D) or eight (3-D) corner blocks",
and the reviewed code was discarding low-wavenumber content along two axes.
**Correct:** a 3-D real FFT halves only the final axis. The retained positive/negative
low-wavenumber construction therefore has **four** blocks — two signs on each of the two
untransformed axes, one on the halved axis. Eight would be right for a full complex `fftn`.
**Effect:** the underlying observation (that filling only `[:mx, :my, :mz]` discards
negative-wavenumber content on two axes) still stands. The stated block count did not.

### D26 · The KV cache total was arithmetically wrong
**Claimed:** 24 MiB per token at 1,048,576 tokens is "~25 TiB".
**Correct:** **24 TiB exactly.** 1,048,576 MiB is 1 TiB, so 24 MiB × 1,048,576 = 24 TiB. The
error came from multiplying 24 by 1.05 — applying a MiB→MB style conversion that had already
been accounted for.
**Effect:** the engineering conclusion (that the design needs grouped-query or latent
attention) is unchanged, and is now enforced by `test_inventory.py`. But this is precisely
the class of error that document was criticizing, produced while criticizing it. It is the
strongest single argument in this repository for the automated numeric reconciliation in
§5.6 rather than careful reading.

### D27 · The Fr = 1 claim was an overclaim
**Claimed:** doubling a waterfall's inflow "almost always straddles or approaches the
Froude-1 critical transition", so the query escalates.
**Correct:** see D14. Not established, and false for the fixed-area inlet worked in §11.2.
**Effect:** the escalation policy no longer rests on it.

### D28 · Scope creep from single studies to universal impossibility
**Claimed:** one cross-regime generalization failure showed learned operators fail at
extrapolation; one negative-transfer result showed dense multi-physics training suffers
gradient conflict as a general property.
**Correct:** both are real findings in their studied settings and neither generalizes to
impossibility. Published PDE foundation-model work also demonstrates useful learned solution
operators within evaluated settings.
**Effect:** both are retained in [`evidence.md`](evidence.md) as evidence for specific failure
modes and as motivation for experiments, not as proofs. The dense constraint is held as an
engineering choice with measured diagnostics, not as a settled question.

---

## Part 3 — Decisions taken independently of both documents

### D29 · Model naming
**Decision:** the model and checkpoint family is **Iridium-1**. The Python package and schema
namespace stay lowercase `iridium` (`iridium.event.v1`).
**Why:** contracts should outlive model generations. Versioning the event schema as
`iridium1.event.v1` would force a new schema for Iridium-2, which is backwards.

### D30 · Face transfer folds `dt/dx` into the learned quantity
**Decision:** the conservative head predicts `T = F·dt/dx` in cell-value units rather than a
dimensional flux, and the update is `u_i ← u_i − (T_{i+1/2} − T_{i-1/2})`.
**Why:** conditioning. Dividing by `dx = 1/64` amplified the head's output by 64× and made
early optimization badly scaled; measured training loss improved roughly fourfold at matched
steps after the change. The constraint is untouched — each face still appears once with each
sign — and the dimensional flux is recoverable as `F = T·dx/dt`.
**Status:** tested — the conservation property is asserted on the trained model, not just the
formula.

### D31 · The direct head is a fair control, not a straw man
**Decision:** the ablation control predicts a per-cell *increment* from `u0`, reading the same
trunk output as the conservative head. Both start from zero-initialized output layers, so an
untrained model predicts persistence.
**Why:** an unconstrained head that predicts absolute values from scratch would lose to the
conservative head for reasons unrelated to conservation. The only difference between the two
arms should be the constraint being tested.
**Status:** implemented; results in [`first-slice.md`](first-slice.md).
