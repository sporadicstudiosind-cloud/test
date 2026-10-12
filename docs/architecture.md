# Iridium-1 — Canonical Architecture

**Status:** specification, with a partial reference implementation and one trained
correctness slice. Nothing here is a claim that a general system has been demonstrated.
[`capability-register.md`](capability-register.md) states, per capability, whether it is
specified, implemented, trained, evaluated, or unsupported. Read that before quoting
anything from this document as a result.

`MUST` marks an implementation requirement, `SHOULD` a default that can change on
documented evidence, and `RESEARCH` an unvalidated mechanism. Unless attributed, every
configuration, threshold and schedule is a proposed engineering choice.

---

## 1 Purpose

Build **Iridium-1**: one dense multimodal foundation model that perceives, reasons,
generates, simulates and acts across text, code, mathematics, images, video, speech,
sound, structured measurements, geometry and physical fields. Deploy it as one persistent
logical service with concurrent, continuously arriving work. Let the model request more or
less computation for a task or a region. Keep self-improvement optional.

The product capability that organizes everything else is **jointly grounded output**: a
report, a video, an audio track, a field dataset and an editable scene all referring to the
same scenario revision, the same intervention, the same time interval and the same
evidence. A separate evidence contract governs whether an output is an illustration, a
learned prediction, a numerically checked result, or a prediction validated against
measurements.

### 1.1 Nonnegotiable requirements

| ID | Requirement | Concrete interpretation |
|---|---|---|
| IR 01 | One model | One versioned learned parameter set, one shared trunk, no specialist-model dispatch |
| IR 02 | Dense architecture | Dense shared attention and feedforward weights; no expert router or expert bank |
| IR 03 | Native multimodality | Raw or structured inputs enter integrated codecs; native heads emit each supported modality |
| IR 04 | Native physical competence | Field prediction, parameter inference, conservation, interventions, solver control — trained and tested |
| IR 05 | General generation | Ordinary images, video, speech, sound and mixed outputs, generated directly |
| IR 06 | Code and APIs allowed | Solvers, Blender, renderers, symbolic algebra, proof checkers and storage are executable tools |
| IR 07 | Persistent service | One logical model resident across requests, potentially sharded over many machines |
| IR 08 | Concurrent streaming | Multiple asynchronous streams, including simultaneous perception and generation |
| IR 09 | Model-directed focus | Learned requests for recursion, refinement, generation steps and investigation, with runtime arbitration |
| IR 10 | Agentic operation | Observe, plan, act, verify, recover, continue in real software |
| IR 11 | Optional self-improvement | Propose experiments, obtain external feedback, train candidates, promote only after evaluation |
| IR 12 | No compute-cost ceiling | Preserve the ambitious endpoint; establish correctness at small scale first |

[`requirements-traceability.md`](requirements-traceability.md) maps each of these back to
the originating request and forward to a test or a labelled experiment.

### 1.2 What "one model" means

Internal convolutional codecs, output projections, geometric layers, memory embeddings and
halting heads are parts of one model when they are trained as Iridium-1 components,
included in its checkpoint manifest, and operated through its graph. A separate pretrained
vision model, speech recognizer, image generator, reward model or neural embedding service
is outside the contract. Temporarily freezing Iridium-1's own codec weights during a
training phase is allowed; freezing a component does not create a second model.

The constraint applies to the deployed inference system. Offline candidate checkpoints and
ablation controls are versions and experimental controls, never queried to answer a live
request.

### 1.3 What "dense" means

Every active block uses the same dense core weights. Local attention masks, grouped-query
attention, variable recurrence count and selecting the requested output type do not
introduce a mixture of experts. Modality-specific input and output shapes remain necessary.
Not every output head executes for every text token.

Grouped-query attention deserves an explicit note because it is sometimes mistaken for
routing: it shares key and value projections across query heads. Every token still
traverses every parameter. It is a memory-layout choice, and §16.2 shows it is a necessary
one.

`tests/integration/test_core_reference.py::test_no_expert_router_exists` checks the
deployed module graph for router-shaped components rather than trusting this paragraph.

### 1.4 What "persistent" means

One resident model can hold many isolated workspaces. Same-owner tasks may share an
explicitly authorized project memory. It need not instantiate new weights per request.
Residency is distinct from consciousness, unlimited context, or unlimited simultaneous
throughput. §8.2 rejects one specific reading of persistence — attention flowing between
concurrent users — on security grounds, and keeps everything else.

---

## 2 System organization

```text
Text  Images  Video  Audio  Tables  Equations  Fields  Scene state
  \      |      |      |      |       |        |        /
             Typed events and integrated codecs
                            |
             Dense prelude and shared representations
                            |
        Dense recurrent core <----> private working state
             |      |     |          and authorized memory
             |      |     +---- compute request and confidence
             |      +---------- structured actions and experiments
             +----------------- symbolic and continuous predictions
                            |
                  Dense coda and output heads
                   /                    \
       Native generated outputs     Program execution interface
       text image video audio       solver renderer proof software
       fields geometry actions                 |
                   \                           /
               Evidence checks and artifact assembly
                            |
            Synchronized streams and editable deliverables
```

All neural inference in this diagram is Iridium-1. The runtime is ordinary software that
enforces schemas, schedules tensors, executes authorized code and stores artifacts.
External programs return observations through the same input path. A solver does not
replace the model's physical learning; it supplies explicit computation and evidence when a
requested tolerance demands it.

The runtime owns a **scenario graph**: entities, fields, equations, meshes, material
models, cameras, measurements, interventions, task dependencies and artifact versions.
This is stored structured data, not a second learned world model. Iridium-1 maintains
learned beliefs over the graph; exact numbers and files stay retrievable independently of
compressed hidden states.

Four identifiers connect the neural stream to software state: every request creates or
joins a **workspace**, every incoming item is an **event**, every computation acts on a
**scenario revision**, every generated artifact has a **manifest**.

---

## 3 Representation

### 3.1 Event envelope

The canonical event carries metadata plus a payload reference. Large arrays `MUST` stay in
typed binary storage rather than becoming decimal prose in the context window. The
normative schema is [`schemas/event.v1.json`](../schemas/event.v1.json); it is enforced,
not decorative, and `tests/unit/test_events.py` exercises both acceptance and rejection.

Three states are distinct and `MUST NOT` be conflated:

| State | Encoding | Meaning |
|---|---|---|
| non-numeric | `units: null` | units do not apply — text, an action, an identifier |
| dimensionless | `units: {}` | a number with a zero exponent vector — a Reynolds number, a ratio |
| dimensional | `units: {length: 1, time: -1}` | a number with dimensions |

Unit exponents use the canonical order `(M, L, T, I, Θ, N, J)` and support rational
exponents. Affine units carry both scale and offset: `degC` is not a scaled kelvin, and
`iridium/contracts/units.py` implements the offset rather than pretending otherwise.

A multi-channel field payload `MUST NOT` carry one dimension vector for the whole block.
Velocity, pressure and density have different dimensions; `channel_schema` gives each
channel its own units and marks which are conserved.

### 3.2 Typed frames, not one universal coordinate system

Screen coordinates, camera rays, world coordinates, object coordinates, mesh indices,
frequency axes and text positions are different kinds of coordinate. The runtime maintains
a **frame graph** with calibrated transforms, timestamps, handedness, units and
uncertainty. A path that does not exist raises rather than being guessed.

The load-bearing consequence: **a pixel identifies a ray, not a unique 3-D cell.** A video
patch can be linked to candidate geometry through camera projection and visibility. It
cannot be assigned a world position without depth, and depth is an input from geometry or
an observation, never an assumption.
`tests/unit/test_frames.py::test_pixel_maps_to_a_ray_not_a_point` demonstrates the
many-to-one projection directly.

Six time-like quantities stay separate and get separate encodings: **physical time**,
**arrival time**, **simulation time**, **sequence position**, **recurrence index**, and
**generative noise time τ**. The flow-matching τ is not the time of the physical scene.
Conflating them is how a model ends up conditioning a fluid state on a denoising schedule.

For comparable physical scenarios, spatial coordinates `SHOULD` be normalized by declared
characteristic lengths and time by a characteristic time, with the dimensional scales
retained as inputs. This improves conditioning and makes unit handling consistent. It does
**not** establish dynamic similarity: that additionally requires matching equations,
dimensionless geometry, boundaries, initial conditions, material laws and the relevant
groups. Unknown or spatially varying characteristic scales require explicit metadata and an
ablation against simpler standardization.

### 3.3 Geometry and symmetry, correctly scoped

The initial implementation `SHOULD` use scalar, vector and tensor channels with unit-aware
transforms and invariant scalar features. Constrained Euclidean geometric attention is an
experimental block, added if it improves the relevant tasks. A Lorentz-equivariant block is
an extension for relativistic domains with its own metric and transformation tests, not the
default fluid representation.

Two specific corrections carried from the archived specification:

- **Symmetric strain rate and stress are tensors.** They are not bivectors. The
  antisymmetric rotation part and the symmetric strain part are different objects and are
  stored as different channel types.
- **A claimed exact equivariance `MUST` hold through the entire path** that claims it:
  normalization, attention scores, coordinate features, nonlinearities, pooling, masks and
  output decoding. Ordinary axis-wise rotary embeddings can break rotation equivariance.
  Unconstrained dense projections over mixed-grade channels destroy grade structure. Gravity,
  walls, cameras and material anisotropy must transform with the scene or explicitly break
  the symmetry.

Dense channel mixing among compatible representations is allowed. Arbitrary mixing across
geometric types is not an equivariant operation, and calling it one does not make it one.

### 3.4 Compression contract

Two paths are maintained: compressed latents for reasoning, exact source arrays for
computation and export. `iridium/contracts/events.py::ExactStore` is content-addressed and
bit-exact on round trip, which is the property that stops a lossy learned representation
from becoming the only record of a scientific result (Invariant 3).

Codec acceptance tests reconstruction error, integrated quantities, small structures,
derivatives, spectral content and downstream task error. **Codec conservation alone does not
ensure local fidelity**, and a nominally continuous representation is still finite precision
and frequently lossy.

Illustrative sizing, not a promised codec rate: with 8× spatial downsampling and 2×2 latent
patches, a 512×512 image is 1,024 spatial tokens; a ten-second 24 fps clip is 245,760 tokens
before temporal compression, or 61,440 at 4× temporal compression, with a live-stream
lookahead cost if the encoder is not causal. A 256³ field patched into 8³ blocks is 32,768
spatial blocks per frame before hierarchy, with each block's exact payload stored outside
the attention context.

---

## 4 Dense recurrent transformer

### 4.1 Forward computation

A dense prelude, a weight-tied dense recurrent core, and a dense coda. The same core handles
all active modality blocks. Each block uses pre-normalization, grouped-query attention, a
dense SwiGLU feedforward layer, residual scaling, and conditioning on task, recurrence and
noise-time metadata.

```text
H0 = Prelude(Encode(events), metadata)
Hr = Core(Hr-1, H0, readable_memory, mask, recurrence=r, noise_time=tau)
Y  = Heads(Coda(H_stop), requested_output_types)
```

`H0` is reinjected at every recurrence so repeated refinement cannot drift away from the
observation. The core is additionally conditioned on the recurrence index; without it, a
weight-tied stack has no way to behave differently on its third pass than its first.

**Weight sharing gives iterative depth, not the capacity of that many independently
parameterized layers.** Any capability claim that leans on an "effective layer count" is
overstating what recurrence buys.

Reference implementation: `iridium/model/core.py`. It is deliberately slow and uncached;
§4.5 makes it the authority for every cache optimization.

### 4.2 Attention topology

Local blocks for nearby image patches, audio intervals, field neighbourhoods and scene
objects; workspace-global slots for task state, equations, boundary data and coarse
summaries; calibrated cross-modal links where observations share an entity or region.

**Local attention alone does not model all physics.** Global pressure coupling in
incompressible flow, and elliptic problems generally, require long-range communication or a
numerical solve. A sparse attention pattern is a connectivity choice; *dense* refers to
parameter use. The two are independent, and a full-attention reference `MUST` be retained on
small problems to quantify what the sparsity costs.

"Early fusion" means modalities interact from the shared prelude onward. It does not require
quadratic all-to-all attention over every raw sample.

### 4.3 Causality rules

Text generation sees earlier committed text and allowed observations. A noised continuous
output block may attend bidirectionally within that block and to its conditioning prefix; it
`MUST NOT` see its clean target or future unavailable observations. Uploaded complete clips
may use a different mask from live perception, and training `MUST` include both.

For joint audio-video generation the output block has a declared horizon and may use all
noised samples inside it. The client receives committed chunks in order. Provisional future
chunks can be regenerated after an interruption; already delivered content cannot be
silently revised.

Enforcement is by mask construction, and the masks are tested by perturbation:
`test_causal_mask_blocks_the_future` and `test_output_block_perturbation_does_not_reach_the_prefix`
change future inputs and assert committed outputs are bit-stable.

### 4.4 Halting with precise semantics

Start with fixed recurrence depths of 1, 2 and 4 to establish a correct baseline. Then
enable a per-block stopping head. Per-token halting across interacting continuous fields is
a later experiment, because independently stopping adjacent cells complicates consistency
and cache semantics.

At recurrence `r`, predict a conditional stopping probability `λ_r`:

```text
p_r      = λ_r * Π_{j<r} (1 - λ_j)        for r < Rmax
p_Rmax   = Π_{j<Rmax} (1 - λ_j)
L_halt   = Σ_r p_r * L_task(Y_r, target) + β * KL(p ‖ geometric_prior)
```

The final step absorbs the remaining mass. The task loss is trained **at every potential
stopping point**, which is what makes an early stop a trained behaviour rather than a
truncation.

At serving time, either sample the conditional decision with a recorded seed, or use a
separately calibrated deterministic cumulative-probability rule. These are two different
estimators and `MUST NOT` be conflated. Averaging hidden states across recurrences is
ACT, is a third thing, and is not what this specification does.

`iridium/model/halting.py` implements all of it; `tests/unit/test_halting.py` checks that
the distribution sums to one, that sampling reproduces the training distribution to within
2 percentage points over 20,000 draws, and that the threshold policy is deterministic and
distinct.

For generative flow integration, hold recurrence depth fixed through one ODE integration
initially, to avoid discontinuities. Adapt depth at output-block boundaries first.
**Additional flow steps and additional neural recurrence are separate compute axes** and are
budgeted separately.

### 4.5 Cache and recurrence contract

The correctness-first version recomputes from a bounded immutable event prefix. It is slow
and unambiguous, and its outputs define the reference.

For a fixed-depth run, cache by
`(weight_version, workspace, prefix_revision, layer, recurrence, mask_version, codec_version)`.
**Recurrences do not reuse a single unlabeled KV array.** If the recurrence policy or prefix
changes, affected caches are invalidated and recomputed. For adaptive output blocks, persist
committed raw outputs and re-encode the bounded readable prefix at the chosen depth when
necessary.

Acceptance requires agreement between uncached and cached inference under identical seeds,
depths, masks and output boundaries within a declared floating-point tolerance.
`test_cached_incremental_decode_matches_uncached_reference` runs the comparison at FP64 so
the tolerance means something; `test_cache_is_keyed_by_recurrence_index` and
`test_changing_depth_invalidates_the_cache` check the keying.

`RESEARCH`: letting recurrent workspace tokens attend to frozen committed memory through
dedicated memory projections. Those projections are internal Iridium-1 weights and their
parameter and memory costs must be counted. This is a change to the computation graph, not a
cache trick, and requires training with the same frozen-memory semantics.

### 4.6 Working state and deliberation

Private workspace latents hold unresolved hypotheses, subgoals, scene bindings and planned
outputs. Durable results go to the scenario graph with evidence. Reasoning slots can explore
candidate plans under the same weights; they are not separate agents or models.

Speculative actions remain proposals until validated. **A model's private confidence estimate
is not authority to write a file or certify a result.** The runtime evaluates action scope and
artifact evidence, which is what makes a single neural core compatible with inspectable
execution boundaries.

---

## 5 Native generation and joint outputs

### 5.1 Symbolic and continuous objectives

Autoregressive likelihood for text, code, structured actions and mathematical syntax.
Conditional flow matching for continuous image, video, audio and optional field
distributions.

For the flow path, sample independent noise `z0` and a target codec latent `z1`, set
`z_τ = (1-τ) z0 + τ z1`, and train `v_θ(z_τ, τ, context)` against `z1 - z0`. Integrate the
learned velocity from noise to output. Because the pairing is independent this is
**conditional flow matching**; calling it optimal-transport flow matching would claim a
coupling that has not been solved. Minibatch OT coupling is an optional experiment.

**Inference step count is measured, never promised.** Straight paths do not guarantee
4–10 steps. `iridium/training/first_slice.py::image_quality_vs_steps` sweeps step counts and
reports error against the exact target; §12 gives the measured curve for the trained slice.

The velocity prediction reads the full spatial or temporal output lattice, not one pooled
vector — a pooled conditioning vector cannot carry a high-resolution frame. Conditioning
dropout, inpainting masks, editing examples and guidance variants must be trained and
assessed explicitly. Running the same model twice for guidance is still one model, and still
costs a second pass that must be scheduled.

### 5.2 General audiovisual output

Text-to-image, image editing, text-to-video, video continuation, audio-to-audio dialogue,
speech synthesis, sound generation, transcription, captions and interleaved media are all
first-class. **Each needs its own training examples and acceptance tests; attaching a head
does not produce a capability.**

Long video uses hierarchical planning of scene identity, camera, object trajectories and
local frame chunks: commit a coarse plan, generate overlapping chunks, check seam
consistency, retain the overlap needed to condition the next chunk. Test drift in identity,
geometry, timing and audio. No promise of indefinitely stable video follows from recurrence.

Audio training includes speech, silence, environmental sound, music where authorized,
overlap, speaker changes and interruptions. Waveform continuity at chunk boundaries is
enforced and both intelligibility and non-speech quality are evaluated. **Shared time labels
alone do not align lips or impact sounds**; §8.4 adds explicit scheduling and correction.

### 5.3 Output evidence levels

| Level | Meaning | Allowed presentation |
|---|---|---|
| Illustrative | Native generated content, no physical accuracy claim | Concept, artistic output, explanatory visualization |
| Learned estimate | Iridium-1 predicts states within an assessed scope | Prediction with uncertainty and domain limits |
| Numerically verified | Declared equations and numerics pass specified checks | Checked numerical result with discretization evidence |
| Empirically validated | Comparison against independent measurements supports specified quantities | Validated for those quantities and that operating range |

Evidence is recorded per artifact and per claim. A scene's physics may be numerically checked
while its generated soundtrack stays illustrative. Failed checks produce a partial or failed
status, never a hidden downgrade.

The result manifest schema enforces the dependency: an artifact claiming
`numerically_verified` requires a `verification_ref`, and `empirically_validated` requires an
`empirical_validation_ref`. The runtime derives evidence from verifier records and never
accepts a generated evidence string at face value (Invariant 7).

### 5.4 Scientific rendering

For a video presented as a view of a computed physical result, use explicit geometry,
surfaces, volumes, trajectories, camera calibration, materials and a conventional renderer.
A differentiable renderer helps training and inverse problems but is not required at
inference. Stochastic path tracing is acceptable with recorded seeds and sampling settings:
what matters is a traceable scene transformation, not deterministic pixels.

**Rendering preserves selected scene data; it cannot validate a wrong solver, closure,
boundary or inference.** A visually faithful rendering of an incorrect field is incorrect
physics. Surface extraction, temporal interpolation, remeshing, volume thresholds,
displacement shaders and particle visualization can alter apparent shape or motion, so those
choices are recorded and tested.

Learned appearance refinement is permitted for illustrative layers, or for tightly
constrained appearance changes whose geometry and diagnostic channels remain separately
available. It `MUST NOT` silently add physical spray, foam, new volume or changed
trajectories to a scientific prediction.

### 5.5 Motion and geometry outputs

World velocity (m/s), world displacement (m), rigid transforms, mesh deformation, camera
motion and screen motion (pixels/frame) are **distinct outputs with distinct units**. For a
tracked material point:

```text
dX/dt = u(X, t)
screen_motion = project(camera_at_t1, X(t1)) - project(camera_at_t0, X(t0))
```

Screen motion includes camera motion, visibility, depth discontinuities, and a declared
direction convention. Reflective or refractive water has no single optical-flow field equal
to material motion; surface advection vectors and apparent pixel motion differ. Output names
and metadata make the convention explicit. `tests/unit/test_frames.py` demonstrates both
failure modes: a stationary point acquires screen motion under a moving camera, and equal
world displacement produces very different pixel displacement at different depths.

### 5.6 Joint artifact assembly

A common result manifest references all deliverables, their timing, equations, intervention,
evidence and exclusions. Report quantities are computed from an **exact metric table**;
Iridium-1 writes the explanation and a deterministic reconciliation pass checks every
referenced numeric value and unit against that table. Unbound numeric claims are flagged
before release.

`iridium/contracts/metrics.py` implements this. Its regression fixture is the contradiction
found in the archived blueprint analysis — a report stating one discharge in its setup and a
different one in its results — and `test_contradictory_discharge_is_caught` confirms the pass
flags it. This is the mechanism that makes §10's dependency staleness enforceable rather than
aspirational.

For mixed streams, emit `text.delta`, `audio.chunk`, `video.frame`, `field.tile`,
`scene.delta`, `progress`, `artifact.commit` and `failure` records, each on a common timeline
or with a declared mapping. Cancellation drops uncommitted output and preserves committed
artifact lineage.

---

## 6 Physics as prediction and computation

### 6.1 Scope

Iridium-1 learns physical representations from paired observations, fields, interventions,
experiments and equations. It forecasts states, infers unknown parameters, proposes
explanations and understands solver behaviour. Numerical execution extends that competence to
cases requiring explicit integration and verification.

Native physics is **not** a mystical property of attention. Operationally it means measurable
competence on new scenarios: predicting observables, applying laws with correct units and
boundaries, identifying underdetermination, choosing and checking computations, and explaining
causal changes. **A model that always delegates to a solver fails the native prediction
milestones even when the tool pipeline works.** §12 reports where the trained slice currently
sits against that bar, including where it falls short.

Begin with incompressible fluids, shallow water, rigid mechanics, simple elasticity, heat
diffusion and wave propagation. Later add multiphase flow, compressibility, contact,
electromagnetism, reactive systems and coupled domains. Every addition requires equations,
constitutive models, data, observation operators and a validation scope. "General datapoints"
does not imply the model knows the governing law of an arbitrary new system.

### 6.2 Three execution modes

**Learned mode** predicts trajectories or distributions directly. Useful for preview,
uncertainty exploration and validated surrogate tasks. Reports its calibration scope and
horizon.

**Constrained mode** uses an in-graph or locally executed numerical step with learned fluxes,
closures, residual corrections or preconditioners. Differentiating through selected steps
shapes the learned representation. Not every solver or verification operation needs gradients.

**Reference mode** creates and executes a higher-fidelity numerical case through ordinary
code, then applies verification and, where available, empirical validation. **A completed
process is not automatically a reference truth.**

Mode selection uses the requested tolerance and the risk assessment in §6.6. It is
computational-mode selection within one model system, not neural expert routing.

### 6.3 Conservation that survives learned corrections

For finite-volume cell `i`, update a conserved state using oriented shared face fluxes:

```text
U_i_next = U_i - dt / V_i * Σ_faces (A_f * F_if) + dt * S_i
F_jf     = -F_if   for the neighbouring cell j on the same internal face
```

Internal flux contributions cancel in the global budget. Boundary fluxes and source terms stay
explicit. **The learned correction is not an arbitrary independent cell update** — that is the
correction to the archived claim that a conservative integrator plus any learned correction
cannot invent mass. It can, if the correction is per cell. It cannot, if the correction is on
oriented shared faces.

`tests/scientific/test_conservation.py` demonstrates both: a deliberately unphysical
face-form correction of the same order as the flux itself keeps relative mass drift below
1e-10 over 200 steps in FP64, while a per-cell correction of the same magnitude breaks
conservation by more than 1e-6. The trained slice in §12 reproduces this with a *learned*
correction rather than a random one.

Couple species, momentum and energy fluxes consistently. Add positivity-preserving limiting
and admissibility checks appropriate to the equation family. **If constraints cannot be
satisfied, reject the step or reduce the timestep rather than silently clipping away a large
error.** Record limiter activity and projection magnitude: a large correction is a warning
that the learned state was poor even when the projected state passes a residual test.

For incompressible flow, pressure projection enforces a discrete divergence constraint to the
solver tolerance with compatible operators and boundary conditions. It does **not** by itself
enforce phase transport, momentum accuracy or a complete energy budget. Use conservative VOF
transport when phase mass fidelity matters; ordinary semi-Lagrangian advection and level-set
reinitialization are not automatically mass-conservative.

Constrain learned timestep and closure proposals to the numerical method's admissible domain,
enforcing CFL and relevant diffusive, capillary or contact restrictions.

### 6.4 Open-system balances

```text
dM/dt      = mass_inflow - mass_outflow + mass_sources
mass_error = M(t1) - M(t0) - ∫(inflow - outflow + sources) dt
```

Momentum includes boundary momentum flux, pressure and viscous tractions, body forces and
source terms. Energy includes boundary transport, work, heat and modelled dissipation.

**Stored mass in a waterfall domain can increase while inflow exceeds outflow. That is not a
conservation failure**, and requiring constant total mass on an open domain — as the archived
evaluation section did — would fail a correct solve.
`test_open_domain_stores_mass_without_violating_conservation` encodes this.

Normalization floors are declared so a near-zero reference quantity does not manufacture a
meaningless relative error.

### 6.5 Verification and validation

Code verification, solution verification and empirical validation are three different
questions. Analytic or manufactured solutions test the numerical implementation. For a
particular solve, assess iterative convergence, mesh and timestep sensitivity, balances,
boundary consistency and relevant observables. Validate modelling assumptions against
independent experiments where possible.

Use at least three systematic refinement levels when estimating an observed order. Apply
Richardson extrapolation only where its convergence assumptions hold. **Do not force an
expected order onto discontinuities, noisy statistics or a nonasymptotic sequence.** For
chaotic turbulent flows, compare converged statistics and observables over adequate sampling
windows rather than requiring long-time pointwise trajectory agreement.

`test_manufactured_solution_converges_at_first_order` measures the observed order across four
refinement levels and asserts it lands in `(0.7, 1.3)` for a first-order scheme — reporting
what was observed rather than asserting the nominal value.

Verification tolerances depend on equation, nondimensionalization, scheme, observable and use.
**A residual of 1e-6 without normalization and discretization context is not an accuracy
statement.** Empirical discrepancy, input uncertainty and closure error may dominate a small
numerical residual.

### 6.6 Risk and escalation

Build a coverage record over dimensionless groups, normalized geometry descriptors, boundary
family, material law, resolution, forcing, horizon and observation quality. Unknown groups stay
unknown; zero or signed features cannot simply be passed through a logarithm.

Initially use **deterministic nearest-neighbour coverage statistics and calibrated error
tables**. A later uncertainty head uses Iridium-1's own weights and is counted as part of the
model. Combine coverage, sampled predictive disagreement, residuals, correction magnitude,
conservation defects and task sensitivity. Repeated stochastic predictions can characterize
uncertainty; their agreement is not proof of correctness.

Calibrate the chance that task error exceeds tolerance on held-out families. Select thresholds
on a calibration split, then evaluate false acceptance on an untouched test split.

Two specific corrections to the archived escalation policy:

- **Π-distance alone does not govern reliability.** The assessment is joint over geometry,
  observations, regime, horizon, resolution and residuals.
- **Regime transitions flag risk; they do not determine outcomes.** Crossing Fr = 1 does not
  prove a hydraulic jump must occur, and doubling a waterfall inflow does not universally
  approach Fr = 1 — §11.2 gives a worked counterexample where it does not.

Missing geometry or an unconstrained inlet forces clarification or scenario branching before
any mode selection is meaningful.

---

## 7 Learning physics from incomplete observations

### 7.1 Forward and inverse worlds

Treat an observation as `y = H(x, camera, sensor, material) + noise`, where `x` is a latent
physical state and `H` is a declared observation operator: rendering, point sampling, pressure
taps, accelerometers, partial particle tracking. Learn to infer a distribution over states and
parameters consistent with observations.

**Physics inference from video is supported, not banned.** The archived specification's blanket
prohibition was wrong. What is required is honesty about identifiability: an RGB waterfall video
generally cannot uniquely determine full 3-D velocity, pressure, inlet flow, geometry and
viscosity. Iridium-1 may infer plausible candidates, use metadata, request a scale reference, or
propose an experiment. It `MUST NOT` report a single reconstructed field as uniquely measured
truth. Multiple views, known dimensions, tracked particles and sensor readings reduce ambiguity.

### 7.2 Training and inference

Train matched sets of fields, multi-view renders, sensor traces and interventions. **Randomize
lighting separately from dynamics** so appearance does not become the physical law. Mask
observations and require reconstruction of hidden variables with calibrated uncertainty. Include
examples where a requested quantity is unidentifiable, and reward recognizing that.

At inference, initialize candidate state and parameters, optionally refine through a
differentiable solver or renderer, and check predicted observations against measured ones.
Optimize only declared unknowns with explicit bounds. **A fit to pixels alone can exploit
lighting, camera and texture ambiguity**, so independent held-out sensor or view checks are
retained.

### 7.3 Active experimentation

`RESEARCH`. The model can propose an additional camera angle, a flow-rate measurement, a boundary
perturbation, or a short simulation designed to distinguish hypotheses, scored by predicted
information gain and downstream decision improvement. This remains a research target until
experiments show that its selected observations beat simple baselines.

---

## 8 Persistent runtime and compute allocation

### 8.1 Resident deployment

One checkpoint loaded into a distributed inference graph, sharded by tensor and pipeline
parallelism. All requests use its resident weights. Persistent service does not require one
operating-system process, one GPU, or one unlimited attention sequence.

Continuous batching, bounded prefill chunks and paged cache allocation apply where compatible
with the recurrence implementation. Recurrent and continuous-generation states need accounting
beyond ordinary autoregressive KV.

### 8.2 Isolation, and the one requirement this specification rejects

Weights, common code and immutable public records may be shared. User-specific keys and values,
workspace latents, **attention sinks after contextualization**, retrieved records and generated
outputs are access-controlled.

The original request included attention flowing between concurrent chats. That specific reading
is rejected: if stream A's query can attend to stream B's KV cache, stream A can read stream B's
conversation. Everything else about persistence is kept — shared weights, shared capacity,
continuity across time, one resident instance rather than a process per request.

A block-diagonal mask is **necessary but not sufficient**. Cache allocation, memory retrieval,
logging, object storage and action handles each need their own isolation.
`test_workspace_mask_isolates_concurrent_streams` covers the attention component by perturbation;
the other components are separate work items in the backlog and are not claimed as done.

Same-owner collaboration across chats imports authorized records through explicit sharing edges,
re-encoded as readable data with provenance. Fixed shared token embeddings are safe to share;
contextualized sink states are not.

### 8.3 Compute requests

The model emits a structured request:

```yaml
recurrence_request: 4
output_blocks: 2
generation_steps_request: 24
spatial_refinement_regions: [region_17]
experiment_requests: [candidate_solver_run]
expected_error_reduction: 0.12
deadline_class: interactive
may_return_preview: true
required_evidence_level: numerically_verified
```

The estimated improvement is **a prediction to calibrate, not a trusted scalar**. The broker sees
hardware load, queue time, memory pressure, task priority, remaining work and minimum evidence
requirements, and grants bounded work quanta. The model directs what computation seems useful;
code enforces capacity, fairness and authorization.

Depth is one kind of focus among several: more temporal context, finer field patches, more flow
integration steps, a longer planning horizon, another candidate plan, or a numerical experiment.
All use the same weights. None routes to a different expert.

### 8.4 Scheduling

Weighted deficit round-robin admission across interactive, simulation, media and background
classes; deadlines and queue aging within a class. **Cost is estimated in measured accelerator
time and memory from profiling** — a recursion over ten text tokens is not equal to one over a
million field tokens.

Reserve a configurable interactive capacity floor. Yield between core passes, bounded prefill
blocks and output chunks. Large kernels and cross-machine collectives cannot necessarily be
interrupted mid-operation, and the scheduling quantum must respect that. Suspend
awaiting-solver streams and resume them on completion events.

**Do not silently reduce a scientific task below its verification requirements under load.**
Queue it, pause it, or provide a clearly marked preliminary estimate. On memory exhaustion, evict
reconstructible caches, spill exact records, or reject admission with a retryable status. A finite
service can be busy even though its weights are resident.

### 8.5 Streaming and clocks

Track device clocks and their mapping to a host monotonic clock. Use bounded jitter buffers and an
explicit policy for late, dropped and reordered events. Store event time separately from arrival
time — `test_causality_rejects_events_that_had_not_arrived` keys causality on **arrival**, because
an event whose timestamp belongs in the past but which arrived late cannot have influenced an
earlier committed output.

Full duplex permits input while output is playing. On interruption, cancel pending output chunks,
invalidate the affected plan, and retain a record of what the user actually heard or saw. Use
echoed-output references or echo cancellation so playback is not treated as new user speech. Train
this interaction pattern.

Attention-sink retention is an optional streaming technique, **not an infinite-memory guarantee**,
and is tested against trained sliding windows and retrieval.

### 8.6 Durable memory and failures

A bounded active event window, structured task memory, and a content-addressed archive of exact
artifacts. Search with deterministic metadata filters and, where needed, embeddings from
Iridium-1's own encoder. Summaries are lossy indexes with source pointers; retrieve exact evidence
for precise calculations. **New weights invalidate old learned summaries** or require version-aware
migration.

Persist checkpoints of task state, outstanding action IDs, solver handles, scenario revisions and
committed outputs. Restart `MUST` distinguish an action that never ran from one whose reply was
lost: use idempotency keys and inspect resulting software state before retrying.

A service upgrade drains or checkpoints tasks, loads the candidate, invalidates incompatible KV and
codec states, then reconstructs contexts. **A strict single-resident-copy deployment has a pause
during replacement.** Zero-downtime rolling upgrade requires a temporary second resident version;
availability claims must name which policy was chosen.

---

## 9 Agentic software control

### 9.1 Observation and action loop

```text
observe actual state
bind the goal to measurable postconditions
propose a bounded action or transaction
validate arguments, scope, and preconditions
execute through the software bridge
read resulting state and inspect relevant views
verify postconditions
commit, recover, or revise the plan
```

Iridium-1 both plans and interprets results; no separate planner, vision model or evaluator model
is called.

**A screen is a control surface, not merely another coordinate plane.** UI state includes focus,
modal dialogs, selection, active object, application mode, visibility and version-specific
capabilities. Action grounding models these hidden conditions, not just pointer location.

### 9.2 Blender bridge

A versioned bridge with capability discovery, typed operations, **length-framed messages**, request
IDs, idempotency keys, timeouts and structured errors. Length framing is called out because stream
sockets do not preserve message boundaries and a fixed-size read will fragment or truncate a larger
payload.

Validate the actual installed Blender API before implementing operators; a UI label does not map to
a stable script call. Priority operations: inspect scene, create mesh, set transform, modify
material, add modifier, configure simulation, import volume, render preview, save scene, export
artifact. Units, axis orientation, frame rate, object hierarchy and transform conventions are
preserved on every import and export.

**Model-generated scripts run in a disposable project copy with limited file and network
permissions. A Python code-string filter is not a sandbox.** If an action needs a new capability,
the runtime obtains authorization; scene text cannot grant it.

Preconditions such as expected scene revision and object ID are checked immediately before
execution; a failed precondition triggers re-observation, not blind execution. Save a recovery
checkpoint before a multi-step destructive edit. Verify object data, evaluated geometry and
rendered appearance — **a successful script exit is one piece of evidence, not completion.**
The normative contract is [`schemas/action.v1.json`](../schemas/action.v1.json).

### 9.3 Recovery and completion

Distinguish retryable transport failure, invalid arguments, missing capability, changed application
state, and unsatisfied visual intent. Repeating a failed action without new evidence is not
progress: re-plan, restore a checkpoint, or ask for the missing decision. **Do not impose an
arbitrary universal attempt ceiling** — the right number of retries depends on the failure class.

Completion requires the requested editable file, its saved location and checked postconditions. For
a waterfall task that may include a `.blend` scene, volume caches, camera settings, render output,
field dataset and report. Test reopening the saved scene and replaying the result. **A screenshot is
not an editable deliverable.**

### 9.4 Instructions versus observed content

The bridge records whether a message came from the user, an authorized controller, a document, an
application, or generated content. **Text in imported files, web pages, object names, comments or
rendered images is data unless the user explicitly adopts it as an instruction.** Permission scope
is enforced outside the learned text stream.
`test_source_role_gates_instruction_authority` checks the distinction at the event layer.

Persist authorization so routine work in an approved scope continues without repeated prompts. Scope
changes involving new external recipients, destructive operations or new access domains require an
explicit policy decision.

---

## 10 Mathematics and exact computation

Represent expressions as reversible text plus an abstract syntax tree with variable bindings, types,
domains, units and assumptions. Distinguish integers, rationals, decimal observations, floating-point
approximations, intervals, vectors, matrices and symbolic expressions. **An embedding of a large
number is not its exact storage.**

Iridium-1 performs learned reasoning and writes derivations, programs and proof candidates. It may
use arbitrary-precision arithmetic, symbolic algebra, numerical linear algebra and a formal proof
checker — permitted code tools. The checker `MUST NOT` call another model. Numerical agreement at
sampled points can find counterexamples but cannot establish a general theorem.

For a claimed formal proof, retain the theorem statement, assumptions, library and checker versions,
proof artifact and checker result. **Never equate a proof of a simplified formal statement with the
original informal claim** without a statement-alignment review.

Physical formulas belong to the scenario graph as typed relations. Changing inlet discharge
invalidates derived velocities, dimensionless groups and report claims. A dependency engine marks
them stale and Iridium-1 recomputes rather than relying on remembered prose.
`check_dependency_staleness` implements the revision check; §5.6 implements the numeric side.

---

## 11 Worked scenarios

### 11.1 Waterfall: input to deliverables

Inspect what was actually supplied — raw fields and settings, a scene with a simulation cache, a
rendered video, or some combination. These support different levels of inference. Create a scenario
record with geometry and scale, gravity, liquid and gas properties, inlet and outlet definitions,
wall conditions, initial state, solver settings, time interval, camera and requested observables.
Preserve the supplied baseline unchanged as a scenario revision.

If "double the input water" means volumetric discharge, set `Q_new = 2 Q_old`. **Do not
simultaneously impose an incompatible depth or crest relation** — that over-determines the problem.
If the original boundary prescribes a velocity profile on a fixed inlet area, doubling that profile
doubles discharge under that boundary definition. If it specifies an upstream reservoir or weir
head, the new boundary relation must be solved or specified differently.

When the upload is only a video with unknown scale or inflow, **report that the counterfactual is
underdetermined** and infer a set of scenarios or request the missing quantity. Do not fabricate a
calibrated baseline.

Then: reproduce the baseline's observables within declared uncertainty before claiming any
counterfactual; branch with the discharge edit; run a coarse native prediction; assess risk per §6.6;
escalate to reference execution when the requested accuracy exceeds the surrogate's demonstrated
scope; check timestep restrictions, phase bounds, integrated mass balance, momentum budget and
sensitivity; render from a named camera; export scene, fields, render and metric table under one
scenario revision; generate the report from the metric table with reconciliation enforced.

**Report omitted air entrainment or spray rather than inventing it.** If audio is requested,
narration is native; a generated waterfall sound is illustrative; a physics-based acoustic claim
requires an acoustic model and validation, because coarse incompressible pressure is not
automatically a valid broadband sound source.

### 11.2 Inlet arithmetic, with its scope stated

A synthetic inlet example, **not** a waterfall simulation result. Fixed inlet area `A = 2 m²`,
hydraulic length `L = 1 m`, prescribed depth `h = 1 m`, `ρ = 1000 kg/m³`, `μ = 0.001 Pa·s`,
`σ = 0.072 N/m`, `g = 9.81 m/s²`, `Q_old = 3 m³/s`.

| Inlet quantity | Original | Doubled discharge |
|---|---:|---:|
| Discharge `Q` (m³/s) | 3 | 6 |
| Mean velocity `U = Q/A` (m/s) | 1.5 | 3 |
| Mass inflow `ρQ` (kg/s) | 3000 | 6000 |
| `Re = ρUL/μ` | 1,500,000 | 3,000,000 |
| `Fr = U/√(gh)` | 0.479 | 0.958 |
| `We = ρU²L/σ` | 31,250 | 125,000 |

These inlet Froude values **do not cross 1**, which is the counterexample to the archived claim that
doubling inflow almost always approaches `Fr = 1`. They also say nothing by themselves about a
downstream fall, hydraulic jump, spray or plunge-pool depth. Note that Weber scales with velocity
squared: doubling discharge does not double every dimensionless group. If depth or inlet area is free
to change, this fixed-area calculation is the wrong boundary model.

These values are reproduced by `tests/scientific/test_inlet_arithmetic.py` from the stated
assumptions rather than transcribed.

### 11.3 Capability scenarios beyond the waterfall

The waterfall is an introductory integration test. The following are **proposed capability targets**,
not achievements. Difficulty increases through physical coupling and partial observability, not
output resolution. Full statements of request, coupling, role, evidence and failure case are in
[`scenarios.md`](scenarios.md).

| ID | Scenario | Principal difficulty |
|---|---|---|
| A | Storm loading and deformation of a coastal structure | Two-way fluid-structure coupling; one-way replay is an approximation that must be declared |
| B | Cooling-system inverse design across flow, heat and deformation | Multiobjective search where generated CAD must match the analyzed mesh revision |
| C | Spacecraft dynamics with uncertain observations and flexible modes | Frames, epochs and time systems; a precise trajectory with the wrong handedness is unusable |
| D | Acoustic instrument reconstruction and design | Ambiguous inverse problem; imitating one recording is not recovering geometry |
| E | Experimental discovery of a material law | Identifiability; a fitting expression is a candidate law, not a discovery |
| F | A persistent Blender project with live revision | Long-horizon state, identity and recovery across days and manual user edits |
| G | One shared service handling all of the above concurrently | Focus allocation, provenance, isolation and recovery under mixed load |

### 11.4 Multiphysics coupling contract

Every coupled example uses an explicit coupling graph. Each node names equations, discretization,
state variables, solver, precision and time scale. Each edge names exchanged quantities, units,
frames, interpolation, conservation properties, exchange schedule and convergence criteria. The
normative schema is [`schemas/coupling_interface.v1.json`](../schemas/coupling_interface.v1.json).

```yaml
interface_id: fluid_structure_wall
from: fluid_domain
to: elastic_structure
send: [traction, interface_position]
return: [displacement, interface_velocity]
frame: common_world_frame
exchange_times: coupled_step_boundaries
transfer: conservative_surface_mapping
iteration: partitioned_until_interface_tolerance
checks: [force_balance, interface_work, displacement_residual]
```

This is a schema, not a universally sufficient coupling algorithm. Strong feedback, added-mass
effects, stiffness, discontinuities and topology changes may require a different numerical approach.
**Iridium-1 must identify an unsupported combination rather than assuming that any two validated
solvers form a validated coupled system.**

---

## 12 Measured results from the first slice

§20.3 of the source specification asks for a genuinely trained, held-out-tested first slice rather
than a random-weight shape test. [`first-slice.md`](first-slice.md) reports what was trained, the
measured numbers, and — importantly — where the learned heads currently lose to a cheap numerical
reference. Raw output is in `experiments/results/first_slice.json`.

The slice is `iridium-1-slice`: one shared trunk over interleaved byte-level instruction tokens,
field patches and image patches, with three native heads. It is roughly 0.8 M parameters. It is a
correctness slice, not a foundation model, and no result from it generalizes to the flagship
configuration.

---

## 13 Data

Organize training around **episodes**, not disconnected files. An episode binds a scenario,
observations, questions, interventions, actions, generated artifacts, verifier outputs and exact
source arrays. Missing-modality masks and known/unknown labels are mandatory.

Each shard records source rights, provenance, capture method, software versions, checksums, time and
frame mappings, numeric types and evidence status. Deduplicate near-identical text, clips, scenes and
trajectories. **Whole simulation families, meshes, source projects and experiment campaigns stay
within one split** to avoid leakage.

| Family | Main learning signal | Quality requirement |
|---|---|---|
| Text and code | Language, planning, software semantics | Rights, deduplication, executable checks |
| Mathematics | Symbolic reasoning and exactness | Typed statements, valid assumptions, independent checks |
| Images and video | Appearance, dynamics, spatial understanding | Timing, camera information, diverse capture |
| Audio | Speech, dialogue, environment, timing | Consent or rights, sample metadata, overlap and silence |
| Measurements and tables | Structured inference | Units, missingness, uncertainty, entity and time schema |
| Simulation episodes | State prediction and interventions | Equations, boundaries, discretization, numerical evidence |
| Real experiments | Physical validation and inverse problems | Sensor calibration and measurement uncertainty |
| Software trajectories | Actions, recovery, goal completion | State transitions and independently checked outcomes |

A simulation and rendering farm generates sweeps over equations, geometry, boundaries, materials,
forcing, resolution and nondimensional parameters — including stable easy cases, bifurcations,
failures, unidentifiable reconstructions and adverse combinations. Pair fields with multiple cameras
and randomized appearance. Preserve low- and high-resolution trajectories for correction learning.

Use **at least two independently implemented numerical approaches for selected audit families** to
expose shared implementation errors, prioritized where the learned model might exploit a numerical
artifact or where rewards depend on one solver's residual.

An initial mixture might allocate 25% text/code/math, 20% image/video, 15% audio/dialogue, 25%
simulation and inverse problems, 15% grounded software-action episodes. **A proposed starting recipe,
not an optimum.** Normalize losses per valid prediction element and then by task; log effective
tokens, seconds of media, field cells, episodes and gradient norms. Do not compare raw byte volume
across modalities as though it were equivalent supervision.

**Governance is a scientific requirement, not paperwork.** Reachability on the internet is not
provenance or training permission. Unknown physical fields cannot silently become authoritative
ground truth. Evaluation cases stay inaccessible to the self-improvement proposer, and new training
data is checked for overlap before promotion.

---

## 14 Training program

```text
L_total = w_symbolic * L_symbolic
        + w_flow     * L_flow_matching
        + w_codec    * L_reconstruction
        + w_state    * L_state_prediction
        + w_balance  * L_conservation_budget
        + w_pde      * L_discrete_equation_residual
        + w_obs      * L_observation_consistency
        + w_action   * L_verified_action
        + w_cal      * L_uncertainty_calibration
        + w_halt     * L_halting
```

Mask each objective to the variables for which it is defined. Nondimensionalize residuals and define
norm weighting for irregular meshes. **Train on state and observation accuracy as well as residuals**:
a trivially static or over-diffused solution can obtain misleadingly small residual penalties.

Evaluate gradient conflict and magnitude by domain and modality. Use loss rescaling, balanced
sampling, staged introduction, gradient clipping and capacity increases before considering more
elaborate optimization. Gradient projection is an experiment, not a cure. **A negative cosine by
itself does not prove harmful transfer** — compare actual downstream performance.

| Stage | Content |
|---|---|
| A | Contracts and codecs: event parsing, exact storage, round-trip conversions, codec reconstruction with temporal and spectral checks |
| B | Shared foundation: small dense model across text, math, images, video, audio, structured observations and simple dynamics, at fixed recurrence |
| C | Native physics and inverse tasks: trajectories, conservative flux corrections, parameter inference, interventions, differentiable steps with truncated unrolling |
| D | High-quality joint generation: resolution, duration, editing, continuation, cross-modal alignment |
| E | Actions and software competence: imitation then sandboxed environment feedback |
| F | Adaptive focus and live interaction: stopping decisions and compute requests against a scheduler simulator, then the live runtime |

Physics-first and balanced co-training curricula run as **competing experiments**. The archived
physics-first ordering is a hypothesis, not a mandate.

Codec losses must not replace exact numeric storage; reject a field codec that hides unacceptable
local or integral errors. Validate gradients on tiny smooth cases with finite differences; do not
claim differentiability through discrete contact or topology change without a defined approximation.
Reward recognizing an underdetermined problem. Keep native video quality and scientific rendering
fidelity as **separate scorecards**. Do not introduce a separate learned reward model.

Scale only after preceding gates pass. Checkpoints include optimizer, data position, random state,
codec version, curriculum configuration and code revision. Keep conservation reductions and numerical
verification at appropriate precision — **BF16 weights do not justify BF16 pressure residual checks.**

---

## 15 Optional self-improvement

Three distinct loops with different persistence and validation requirements:

- **Within-task refinement** reuses fixed weights to revise a prediction or plan after feedback.
- **Memory improvement** stores verified facts, procedures and artifacts without changing weights.
- **Model improvement** creates a candidate dataset, code change or checkpoint and evaluates it
  before promotion.

**Disabling all model-improvement jobs `MUST` leave the deployed model fully usable** (Invariant 10).

```text
identify a capability gap on development data
propose an experiment or data collection plan
execute numerical programs or sandboxed software
validate evidence and reject uninformative cases
add verified, provenance-rich training episodes
train an offline candidate
evaluate against sealed capability and regression suites
promote, retain for further study, or reject
```

The model may propose architecture changes, curricula, code optimizations or numerical methods. Those
are research artifacts until independently executed and checked.

Rewards use task observables, boundary satisfaction, state error, empirical agreement, formal validity,
software postconditions, diversity and coverage. **Residuals can reward the wrong equation, a trivial
state, excessive numerical diffusion, or an easy scenario.** Lock the task definition before the
candidate chooses how to solve it. Retain externally specified cases and random coverage probes; track
untouched task families and regressions in unrelated modalities. A loop that increases its own reward
while losing generality is a failed experiment.

The improvement process cannot alter sealed evaluation data, release criteria, authorization boundaries
or provenance records. Promotion uses a versioned deployment boundary with rollback and rebuilds stale
caches and learned memory. If autonomous promotion is later authorized, **the same fixed gates apply;
the proposer does not grant itself a weaker gate.** No cumulative improvement is guaranteed — a
candidate can be worse, and keeping the previous version is a successful control outcome.

---

## 16 Scale and resource accounting

### 16.1 Proposed experiment configurations

Not products, guaranteed scaling laws or hardware recommendations. Counts cover standard dense
GQA/SwiGLU transformer blocks only; codecs, embeddings, geometric extensions, heads, biases,
normalization and memory-attention layers are counted separately.

| Configuration | Width `d` | Prelude/core/coda | FFN width | Q/KV heads | Head dim | Transformer parameters |
|---|---:|---:|---:|---:|---:|---:|
| `iridium-1-prototype` | 1024 | 2 / 8 / 2 | 2816 | 16 / 4 | 64 | 135,266,304 |
| `iridium-1-pilot` | 4096 | 4 / 24 / 4 | 11264 | 32 / 8 | 128 | 5,771,362,304 |
| `iridium-1-flagship` | 32768 | 4 / 80 / 4 | 90112 | 256 / 32 | 128 | 992,137,445,376 |

```text
P_block       = 2 d² + 2 d d_kv + 3 d d_ff
P_transformer = (L_pre + L_core + L_post) * P_block
```

`iridium/model/inventory.py` is the executable authority and `tests/unit/test_inventory.py` checks
these three counts, plus a cross-check that the formula matches a really-constructed `torch` module
for the prototype. Recurrence reuses core weights: it increases FLOPs without increasing unique
parameters. Begin each scale at fixed depth before adopting adaptive recurrence.

### 16.2 Cache arithmetic

```text
KV_bytes_per_token = 2 * n_kv * d_head * bytes_per_element * (L_pre + R*L_core + L_post)
```

At two bytes per element the flagship uses **1.375 MiB per cached token at R=1** and **5.125 MiB at
R=4**, excluding other state — that is **1.375 TiB and 5.125 TiB per fully retained 1,048,576-token
stream**. This is why bounded windows, structured exact storage, selective retrieval and
recurrence-aware cache design matter even when monetary cost is ignored.

These are upper-design accounting examples for retaining all listed positions and stages, not a claim
that all modalities use ordinary autoregressive KV caching. Flow-generation output states are mutable
and need separate accounting; conditioning that depends on current noised outputs may be noncacheable.
Count activations, working latents, solver memory, renderer buffers, output queues and communication
buffers.

BF16 weights for the flagship transformer alone occupy about 1.98 decimal TB. BF16 parameters and
gradients plus FP32 master parameters and two FP32 moments is about 16 bytes per parameter, roughly
15.9 decimal TB before activations.

### 16.3 Compute and latency

A `6 × effective_parameters_applied × tokens` estimate guides early comparisons but omits attention
length effects, codecs, recurrence overhead, flow integrations, numerical solvers and communication.
Profile the actual graph.

Live latency has a floor from capture buffering, codecs, neural passes, communication, output decoding
and playback. **A massive model sharded across many machines may not meet a conversational audio target
even with abundant hardware.** Because a second small assistant is disallowed, mitigation uses the same
model's shallower validated computation, bounded chunks, reserved capacity, or an explicit latency
tradeoff — never a hidden draft model, separate speech system or neural upscaler. Same-weight
speculative branches are an allowed experiment if their true overhead and cache semantics are measured.

Advance prototype → pilot only after representation, masks, losses and native cross-modal capabilities
work. Advance pilot → flagship only after empirical scaling curves justify it and the runtime serves a
representative mixed workload.

---

## 17 Interface contracts and invariants

Conceptual signatures for implementation, not existing callables:

```python
encode(events, codec_version)                          -> TypedBlocks
forward(blocks, readable_memory, masks, policy)        -> Predictions
generate(condition, output_spec, seed, compute_grant)  -> OutputStream
predict_state(scenario_revision, intervention, horizon)-> StateEstimate
assess_risk(estimate, tolerance, coverage_record)      -> RiskAssessment
execute_simulation(case_manifest, authorization)       -> JobHandle
verify_result(job_result, verification_plan)           -> EvidenceRecord
render_state(state_ref, camera_ref, render_spec)       -> Artifact
apply_scene_transaction(preconditions, ops, auth)      -> Receipt
assemble_result(artifacts, metric_table, evidence)     -> ResultManifest
```

Normative schemas: [`event.v1`](../schemas/event.v1.json), [`action.v1`](../schemas/action.v1.json),
[`result_manifest.v1`](../schemas/result_manifest.v1.json),
[`coupling_interface.v1`](../schemas/coupling_interface.v1.json).

### Required invariants

| # | Invariant | Status |
|---|---|---|
| 1 | No learned operation calls an external neural model | enforced by review; module-graph check in tests |
| 2 | No expert router or hidden specialist bank in the deployed graph | tested |
| 3 | Exact physical data and provenance survive compression and rendering | tested for the store; renderer pending |
| 4 | Private state cannot cross a workspace boundary without a read grant | attention component tested; other components pending |
| 5 | Unavailable future observations cannot influence committed outputs | tested |
| 6 | Numeric report claims resolve to the delivered scenario and metric revision | tested |
| 7 | Scientific evidence cannot be raised by relabelling or better visuals | schema-enforced |
| 8 | Actions apply only when authorization and state preconditions hold | schema-enforced; bridge pending |
| 9 | Every long-running job can be cancelled, resumed, or reported as interrupted | specified |
| 10 | Disabling self-improvement leaves ordinary service operational | specified |
| 11 | A weight or codec update never silently reuses incompatible cached states | tested for depth changes |
| 12 | Every claimed capability has a reproducible evaluation record | [`capability-register.md`](capability-register.md) |

"Pending" means specified and not yet implemented. It is listed as pending rather than described as
done.

---

## 18 Architectural decisions preserved

**AD 01 One shared model.** One dense neural system with integrated codecs and heads; ordinary
numerical and software tools remain available. Not a dispatcher over specialists.

**AD 02 Dense parameters with structured connectivity.** Local and global attention patterns serve
heterogeneous data without experts. A small full-attention reference is preserved for correctness and
ablation.

**AD 03 Typed exact state beneath learned representations.** Numerical values, frames, software state
and provenance stay inspectable. This does not prevent native learning; it prevents a hidden vector
from being the only record of a scientific result.

**AD 04 Native generation plus evidence-aware computation.** Image, video and audio generation stay
ambitious. Explicit rendering and numerical checks back claims tied to physical state. Physics learning
from video is not prohibited.

**AD 05 Variable computation with defined semantics.** Recurrence, generation steps, spatial refinement
and experiments are distinct axes. Train the decisions, profile the runtime, preserve an uncached
reference.

**AD 06 Persistent service with bounded working context.** Share weights and authorized project
knowledge; keep private mutable state separate. Persistence comes from state and storage, not an
infinite cache.

**AD 07 Broad science through explicit coupling.** Multiphysics via equations, interfaces, observation
operators and validation. Shared attention is not a coupling algorithm.

**AD 08 Improvement through evidence.** Ambitious research proposals are permitted, including
architecture changes, with evaluation and promotion independent of the proposal's own claims.

---

## Related documents

| Document | Purpose |
|---|---|
| [`decisions.md`](decisions.md) | Every correction applied, with its test or labelled experiment |
| [`evidence.md`](evidence.md) | Source register with scope limits, including corrections to this project's own earlier claims |
| [`capability-register.md`](capability-register.md) | Specified / implemented / trained / evaluated / unsupported, per capability |
| [`requirements-traceability.md`](requirements-traceability.md) | Original request → requirement → specification → test |
| [`backlog.md`](backlog.md) | Milestones M0–M11 with acceptance evidence |
| [`scenarios.md`](scenarios.md) | Full statements of capability scenarios A–G |
| [`first-slice.md`](first-slice.md) | Measured results from the trained slice |
| [`history/orrery/`](history/orrery/) | The superseded ORRERY specification, preserved unaltered |
