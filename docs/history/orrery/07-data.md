# 07 — Data: The Internet Solves the Easy Half

> Unrestricted web access covers text, code, images, video, and audio completely. It covers the part
> that makes this model different almost not at all.

## 7.1 What the web gives you

Constraint C10 grants unrestricted internet access for training data, and for five of the seven
modalities this settles the question. Text, code, mathematics-as-written, images, video, and audio
are exactly the domains where the corpus *is* the web and scale is the whole game. Nothing in this
document needs to argue about how to obtain them.

Assume that half solved. The rest of this section is about the half that is not.

## 7.2 Ground-truth field data barely exists in the wild

The web has enormous quantities of *footage* of fluids and almost no *fields*. A video of a waterfall
records radiance at a camera; it does not record velocity, pressure, density, or the boundary
conditions that produced them. There is no crawl that yields `(u, p, ρ, φ)` on a mesh with known
inlets and verified residuals.

Serious sources exist and are small by web standards: curated multi-physics collections, PDE
benchmark suites, aerodynamic datasets with parameter sweeps, turbulence databases. Together they are
a rounding error next to a text corpus.

**The binding constraint on this project is solver time, not crawl time.** The dominant data cost is
a simulation farm producing verified fields, and that cost is measured in GPU-months of *numerical
solving*, not in scraping bandwidth. Any schedule that budgets for data acquisition as a crawling
problem has mis-costed the project by a wide margin.

This is worth sitting with, because it inverts the usual intuition: for the capability that
distinguishes this system, more internet access buys almost nothing, and more solver throughput buys
almost everything.

## 7.3 The pairing that render-the-field needs does not occur naturally

[`04`](04-generation.md) requires `(solved field ↔ rendered image)` pairs with the field as ground
truth — that is what teaches the model the relationship between a physical state and its appearance,
and what makes the differentiable renderer trainable.

Scraped video comes with no field. **Every such pair must be synthesized**: solve, render, keep both.
Three consequences:

1. The differentiable renderer belongs in the **data pipeline**, not only in the model. It is the
   instrument that manufactures the training signal.
2. Rendering diversity is a data-design problem — materials, lighting, camera, atmospheric
   scattering. A model trained on one renderer's aesthetic will treat that aesthetic as physics.
3. **Real footage is still needed**, as the corrective. Purely synthetic pairs teach the model the
   renderer's biases. Paired real-experiment data — PIV velocimetry with simultaneous video, wave
   tank studies, wind tunnel runs with imaging — is scarce, expensive, and disproportionately
   valuable, and should be pursued deliberately rather than opportunistically.

## 7.4 Π-space coverage is designed, not sampled

Extrapolation performance is a function of coverage in the dimensionless coordinates
([`01`](01-representation.md) §1.4), and the escalation thresholds in [`03`](03-physics.md) are
calibrated against the training manifold's shape. Web data lands wherever humans pointed cameras,
which is a distribution over *photogenic* phenomena, not over `(Re, Fr, We, Ma)`.

Scenario generation therefore targets the dimensionless axes directly:

- **Span, don't sample.** Cover each governing group across orders of magnitude, including regimes
  nobody films — creeping flow, near-critical transitions, high-Weber atomization.
- **Densify at regime boundaries.** The escalation policy treats `Fr = 1`, laminar→turbulent, and
  `Ma = 1` as discontinuities. Coverage should be heaviest exactly where the solution character
  changes, because that is where interpolation is most dangerous.
- **Record the Π-point with every sample.** The manifold density estimate that drives escalation is
  only as good as this bookkeeping.
- **Hold out whole regions, not random samples.** Random held-out points measure interpolation. To
  measure extrapolation you must remove a contiguous region of Π-space and test there — otherwise the
  eval in [`11`](11-evaluation.md) reports a number that means nothing.

## 7.5 Where internet access genuinely changes the plan

Not as a bulk source of physics, but as a source of **ground truth the solver farm cannot
manufacture**.

A simulation farm can produce unlimited self-consistent data and cannot tell you whether its
turbulence closure matches reality. Published experiments can: a dam-break study with measured wave
arrival times, a wind-tunnel campaign with pressure taps, a river gauging record, a plunge-pool
aeration measurement. These are small, heterogeneous, buried in papers and supplementary material —
and they are the only external check on the entire physics stack.

So internet access is repurposed:

- **Validation targets, not training bulk.** Mine real measurements and use them to score the
  solvers, not just the model.
- **Fuel for the self-curriculum.** [`09`](09-self-improvement.md)'s loop becomes materially stronger
  when the model can find a real experiment, reproduce it in Tier B, and compare. That comparison is
  the one signal in the entire system that is not self-generated.
- **Solver configuration knowledge.** How practitioners set up cases — meshing conventions, closure
  choices, known pitfalls — is written down on the web even where the data is not.

## 7.6 Practical requirements

- **Provenance per shard.** Source, license, and collection date recorded for every shard, with
  physics data additionally recording solver, version, mesh, and residuals. A field whose provenance
  is unknown cannot be used as ground truth, only as a prior.
- **Decontamination against every eval in [`11`](11-evaluation.md)**, including the held-out Π-space
  regions. A crawl that happens to include a benchmark's reference solutions silently converts an
  extrapolation test into a memorization test.
- **Web "physics" is mostly vocabulary.** Text about fluid dynamics teaches the model to *discuss*
  fluid dynamics. It does not teach dynamics. Mixture weights should reflect that these are different
  capabilities, and [`11`](11-evaluation.md) should be able to tell them apart — a model that talks
  fluently about hydraulic jumps and computes them wrongly is the specific failure this whole
  document is organized against.
- **Licensing and consent are real constraints**, not paperwork, and "unrestricted access" describes
  reachability rather than permission.
