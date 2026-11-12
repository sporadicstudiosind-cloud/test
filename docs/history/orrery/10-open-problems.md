# 10 — Open Problems and Kill Criteria

> Each entry names a measurement that would show the approach is wrong. An open problem without a
> falsification is an excuse.

---

### 1 · Negative transfer under the dense constraint

**Problem.** Dense co-training on incompatible physics regimes produces gradient conflict and
plasticity collapse; the published remedy is sparse routing, which C2 forbids. The spec's answer —
that Π-space brings hostile regimes close enough for one parameter path — is a hypothesis, not a
result.

**Kill criterion.** Any of the thresholds in [`02`](02-trunk.md) §2.4: sustained median inter-domain
gradient cosine below −0.1 across the core; gradient magnitude ratio above 100×; effective-rank
decline over 30% while loss is flat; over 40% high-wavenumber energy loss versus single-domain
controls; or any domain more than 15% worse than its single-domain control.

**If it fires.** Escalate per the ladder in §2.4; if all rungs fail, report to the operator that C2
is not survivable for this domain set. This is a real possible outcome.

---

### 2 · Π-space does not close the extrapolation gap

**Problem.** Nondimensionalization gives exact invariance only where the similarity group is known
and complete. Turbulence has no closed similarity solution. Multiphase flow with surface tension,
phase change, and non-Newtonian rheology generates more independent groups than any dataset can span.
Geometry breaks similarity outright — two flows can match on every dimensionless number and differ
because a wall is rough.

**Kill criterion.** Held-out **region** tests (not random held-out samples): if Tier A error versus
Π-distance is statistically indistinguishable from a model trained in raw SI units, the central
representational claim has failed and the escalation policy is doing all the work.

**Consequence if it fires.** The architecture still functions — escalation to Tier B covers it — but
`τ_near` collapses toward zero and nearly every physics query becomes a full solve. That is a much
slower, much less interesting system, and it should be recognized as such rather than described as
"conservative."

---

### 3 · Autoregressive drift

**Problem.** ~12 percentage points of neural-surrogate error come from error accumulation over
rollout, largely independent of representation quality. Predicting corrections to a conservative
integrator ([`03`](03-physics.md)) attacks this structurally, but does not eliminate it.

**Kill criterion.** Correction-mode error growing super-linearly in rollout length on held-out
trajectories, or conservation drift exceeding tolerance before the horizon the application needs.

**Mitigation if it fires.** Shorter Tier A horizons with more frequent Tier B re-anchoring. Cheap to
implement, expensive to run — this is a cost failure, not a capability failure.

---

### 4 · Joint codec instability

**Problem.** C1 forbids frozen codecs. Stage 0 is the largest schedule risk in the project
([`08`](08-training.md) §8.3).

**Kill criterion.** Frozen-codec ablation controls materially outperforming the jointly-trained codec
on reconstruction, conservation, *and* downstream task loss, and the gap not closing with additional
Stage 0 training.

**If it fires.** The finding is about C1, not about the architecture. The operator decides whether
purity is worth the measured cost — which is exactly why the ablation runs.

---

### 5 · Agentic reliability

**Problem.** Best-in-class hybrid computer-use agents pass 41.2% of long-horizon real-world tasks.
Nothing in this spec improves that. The shared coordinate frame makes control *expressible*, not
*reliable*, and a large share of agentic performance apparently lives in the harness rather than the
weights — an awkward finding for an architecture that refuses external components.

**Kill criterion.** No threshold; this is a known-bad capability. The design response is to fail
safely ([`05`](05-agency.md) §5.5) rather than to assume improvement.

**Practical implication.** Expect agentic control, not physics, to be the practical bottleneck in any
product built on this.

---

### 6 · Solver time as the true data ceiling

**Problem.** The dominant data cost is numerical solving, not crawling ([`07`](07-data.md) §7.2).
Field data at web scale does not exist and cannot be acquired, only manufactured.

**Kill criterion.** Π-space coverage growth per unit of solver compute falling below what the
escalation policy needs to keep `τ_near` useful — i.e. the training manifold growing more slowly than
the space of queries users actually ask.

**Second-order risk.** A farm that is internally consistent and collectively wrong. Only real
measurements detect this, and they are scarce.

---

### 7 · Depth scheduling under pipeline parallelism

**Problem.** Variable-depth recursion and pipeline parallelism interact badly: a stream pondering
deeply occupies its stage while shallow streams wait. The mitigations in [`06`](06-runtime.md) §6.3 —
bounded quanta, micro-batch interleaving, reserved capacity — are partial. This is a genuinely
unsolved scheduling problem at scale.

**Kill criterion.** Interactive-stream tail latency degrading more than a stated factor when
deep-ponder streams are concurrent, after the mitigations are in place.

**If it fires.** Depth allocation becomes coarser — per-stream rather than per-token — which weakens
C3 from "the model directs its attention" to "the scheduler assigns a tier."

---

### 8 · The evaluation gap, which may be the worst one

**Problem.** Existing benchmarks judge *plausibility*, not conservation. Benchmarks in this area rely
on prompts describing a scene without specifying the expected physical outcome, so evaluators assess
whether a video looks right rather than whether the physics is right; even pixel-referenced
benchmarks may not check whether conserved quantities are preserved.

**Why it is the worst.** A model could top every public leaderboard while violating mass balance in
every output. The field's measurement instruments cannot currently distinguish this architecture's
central claim from its competitors' central failure — which means the strongest argument for
render-the-field is one that no existing benchmark will register.

**Kill criterion.** Inverted: if conservation-explicit evaluation ([`11`](11-evaluation.md)) shows no
gap between rendered and conditionally-generated output, the entire premise of
[`04`](04-generation.md) is wrong and should be abandoned.

---

### 9 · The premise itself

**Problem.** A monolith fails illegibly, cannot be upgraded per-component, and is likely worse than a
modular system at each individual task. The case is made in full in [`00`](00-premise.md) §"The case
against," and it does not weaken on repetition.

**Kill criterion.** If cross-modal transfer — the primary justification — does not materialize, i.e.
if per-modality performance matches or trails specialist controls *and* the composed tasks
(field→render→report) show no advantage over a pipeline of specialists, then C1 is buying nothing and
costing a great deal.

**Note.** This is the only kill criterion that invalidates the project rather than a component of it,
which is why it should be measured early and cheaply, at small scale, before the expensive stages.

---

## Summary table

| # | Open problem | Fires when |
|---|---|---|
| 1 | Dense negative transfer | Gradient-conflict thresholds ([`02`](02-trunk.md) §2.4) |
| 2 | Π-space insufficiency | No advantage over raw-SI control on region hold-outs |
| 3 | Autoregressive drift | Super-linear error growth in correction mode |
| 4 | Codec instability | Frozen-codec control wins and stays ahead |
| 5 | Agentic reliability | Known-bad; design for safe failure |
| 6 | Solver-time ceiling | Coverage growth below query growth |
| 7 | Depth scheduling | Interactive tail latency regression |
| 8 | Evaluation gap | No measurable gap between rendered and generated |
| 9 | The monolith premise | No cross-modal transfer advantage over specialists |
