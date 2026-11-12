# Capability Scenarios

Proposed capability targets, **not achievements**. The waterfall is an introductory
integration test; these define the larger ambition. Difficulty increases through physical
coupling and partial observability, not output resolution. Every one of them is currently
**unsupported** — see [capability-register.md](capability-register.md).

Each uses the coupling contract in [arch §11.4](architecture.md#114-multiphysics-coupling-contract).

---

## A · Storm loading and deformation of a coastal structure

**Request.** Ingest a coastal scene, wave-tank video, pressure sensors, structural drawings,
material tests and an existing simulation. Predict how a different storm spectrum and a
modified support geometry affect wave loading, deformation and fatigue indicators. Produce
synchronized wave video, structural displacement, stress fields, force histories, a comparison
report and an editable scene.

**Coupling.** Fluid applies surface traction to the structure; structural displacement changes
the fluid boundary. Strong interaction requires subiterations or another assessed scheme. The
fluid mesh may move or be rebuilt, and interface transfer must preserve units, orientation,
forces and work to specified tolerances. Fluid and structural timesteps can differ, requiring
explicit exchange times.

**Iridium-1's role.** Infer uncertain wave and material parameters; maintain candidate
hypotheses; propose mesh and interface refinement; predict low-fidelity responses natively;
configure numerical programs; inspect residuals; explain uncertainty in the most relevant
observables. Propose a sensor location that separates uncertain stiffness from uncertain
loading.

**Evidence.** Interface force balance, transferred power, free-surface transport, structural
residuals, time/grid sensitivity, withheld sensor measurements. Fatigue estimates require
declared material and cycle assumptions.

**Failure case.** A one-way pressure replay misses motion-induced changes in flow. If the
scenario needs two-way coupling, substituting one-way loading **without reporting the
approximation** is the failure. This scenario tests whether the shared model understands
feedback across domains.

---

## B · Cooling-system inverse design across flow, heat and deformation

**Request.** Ingest CAD, pump curves, inlet time series, infrared images, acoustic recordings,
material properties and measured temperatures. Reduce peak temperature and pumping demand under
geometric and manufacturing constraints. Return candidate CAD, transient temperature and
velocity fields, deformation, diagnostic videos and a ranked report.

**Coupling.** Flow transports heat, temperature changes material properties, solids conduct,
thermal expansion can alter geometry and contact. Where expansion is negligible, justify the
simpler model. Keep pressure drop, pump operating point, heat input and boundary heat loss in
one energy and operating-condition record.

**Evidence.** Mass and energy budgets, pressure-drop checks, mesh/time studies, held-out
temperatures, robustness to uncertain operating conditions. **The generated CAD must match the
analyzed mesh revision.** Optimizing one nominal case while violating an off-design constraint
is failure.

**Bounded arithmetic check.** 1000 W fully transferred to 0.05 kg/s of coolant at 4000 J/(kg·K)
gives a 5 K bulk rise. A conservation check under stated assumptions, not a prediction of peak
device temperature. Reproduced by `test_inlet_arithmetic.py::test_scenario_b_bulk_temperature_rise`.

---

## C · Spacecraft dynamics with uncertain observations and flexible components

**Request.** Ingest telescope images, star-tracker observations, ephemerides, actuator logs, an
articulated model and a symbolic maneuver objective. Infer state and uncertainty, compare plans,
simulate attitude and appendage motion, return trajectories, camera views, actuator commands as
a reviewable plan, and checked derivations.

**Coupling.** Orbital motion, rigid-body attitude, flexible modes, actuator limits, sensor
timing, occlusion and thermal effects interact. **Coordinate conventions and time systems are
the substance, not bookkeeping.** Use a Newtonian formulation where adequate; relativistic
corrections need explicit scope and tests rather than automatic use of a spacetime algebra
everywhere.

**Evidence.** Frames, units, time conversion, constraint satisfaction, conservation where
applicable, actuator saturation, integration sensitivity, held-out observations. Commands stay a
simulation or review artifact unless real actuation is separately authorized.

**Failure case.** A precise trajectory with the wrong epoch, handedness or frame is entirely
unusable. The benchmark must include these errors and require Iridium-1 to detect them.

---

## D · Acoustic instrument reconstruction and design

**Request.** Ingest multi-view photographs, performance video, isolated audio, microphone
calibration, material samples and optional vibration measurements. Infer plausible geometry and
excitation parameters, change a design feature to target a timbre, produce editable geometry,
vibration fields, synthesized audio, synchronized video and an explanation.

**Coupling.** Structural vibration, excitation, acoustic radiation and sensor position interact.
Contact nonlinearities and damping may dominate. Native audio supplies creative previews;
acoustically attributed predictions require a physical model or a validated surrogate with a
declared scope.

**Evidence.** Held-out microphone signals, impulse responses, modal frequencies, damping
estimates, spectral envelopes, perceptual assessment. **A good imitation of one recording is not
evidence the geometry or material was recovered.** Soundtrack quality and acoustic fidelity are
separate scores.

---

## E · Experimental discovery of a material law

**Request.** Ingest high-speed video, force and displacement traces, microscopy, temperature
records, existing equations and several mechanical tests. Identify candidate constitutive
models, propose the next discriminating experiment, predict an unseen loading protocol, return
symbolic relations, parameter uncertainty, simulation fields and a research report.

**Coupling.** Deformation, rate dependence, temperature, hysteresis and damage interact.
Observed strain depends on camera calibration and tracking assumptions. **Fit a family of
candidate models** rather than forcing a familiar law onto every dataset.

**Evidence.** Untouched loading histories, thermodynamic admissibility, uncertainty calibration,
counterexamples, experiment quality. Do not declare a universal law from a successful
interpolation. Self-improvement may add validated episodes but may not recycle its own untested
guesses as truth.

---

## F · A persistent Blender project with live revision

**Request.** Maintain a complex animated scene over days while the user speaks, sketches,
supplies reference video, edits geometry manually and changes constraints. Build rigs and
procedural geometry, run simulations, generate original media assets natively, preserve artistic
intent, deliver a reopenable project and a final audiovisual sequence.

**Coupling.** Primarily long-horizon state and agency: object identities, rigs, animation curves,
modifiers, simulation caches, render dependencies, asset rights, timing and the user's manual
changes must stay consistent.

**Evidence.** Reopen the saved file, validate object and rig structure, inspect poses and motion,
verify render/audio timing, compare requested changes with actual scene data. **The agent must
preserve unrelated manual edits** and explain unresolved reference ambiguity. A final beauty
render cannot hide a broken rig or a missing simulation cache.

---

## G · One shared service handling all of the above

Run A–F concurrently in isolated or explicitly shared workspaces while a seventh session holds a
live conversation. Pause an external solver, inject a late sensor event, interrupt speech, change
a scene revision, restart a worker. The model must allocate focus, keep provenance intact and
resume without duplicating writes or leaking data.

**Measure** end-to-end completion, interactive tail latency, starvation, memory pressure,
stale-output rejection and recovery time. Scientific jobs may take substantial time; the system
should show honest progress and preserve the required evidence level.

This is the full "one persistent mind with parallel work" engineering test, and it is the
scenario that most directly exercises the original request. It is also the furthest from being
supported.
