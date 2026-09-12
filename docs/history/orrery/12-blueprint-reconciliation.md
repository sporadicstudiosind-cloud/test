# 12 — Reconciliation with the Dense Omni-Monolith Blueprint

An alternative architecture document was supplied during design ("Architectural Blueprint: The Dense
Omni-Modal Foundation Model (JARVIS)"). It is substantially more specified than a sketch, and several
of its mechanisms are better than what this spec originally proposed. It also contains one
architectural mistake that reintroduces the exact failure mode this spec exists to prevent, and its
worked example contains internal contradictions that illustrate why verification gates are not
optional.

This document records what was adopted, what was corrected, and what was checked and found wrong.
Everything below was verified against sources or recomputed, not accepted on presentation.

---

## 12.1 Adopted — these are genuine improvements

### Clifford / geometric-algebra multivector latents (adopt, with a caveat)

The blueprint structures internal channels as multivectors in `Cl(3,1)` with physical quantities
mapped to grades: scalars (pressure, density), vectors (velocity), bivectors (vorticity, angular
momentum, EM field tensor), pseudoscalar (helicity). This is a real and well-supported idea — it is
the Geometric Algebra Transformer line of work. GATr represents states in the geometric algebra
`G(3,0,1)` with equivariant attention and grade-sensitive normalization and outperforms non-geometric
baselines on n-body and robotic planning; L-GATr extends it to a spacetime algebra with Lorentz
equivariance and matches or beats domain-specific baselines in high-energy physics;
Clifford-Steerable CNNs show gains specifically on fluid dynamics and relativistic electrodynamics
forecasting.

This is **strictly better than the original spec's plain content vector**, and it composes with the
dimensional exponents of [`01`](01-representation.md) rather than competing:

| Mechanism | Symmetry it buys |
|---|---|
| Clifford multivector fiber | rotational SO(3), Galilean, Lorentz equivariance |
| SI dimension exponents → Π-space | **scale** similarity (Re, Fr, We, Ma) |

Neither subsumes the other. Rotational equivariance does not help you extrapolate to a doubled
inflow; scale similarity does not tell you that vorticity is a bivector. Together they cover
substantially more of the relevant symmetry group than either alone, and that is the strongest
single upgrade the blueprint contributes.

**The caveat, and it is a live debate rather than a settled objection.** Under constraint C9 (ignore
compute), hard-coded equivariance is not automatically free: unconstrained models with data
augmentation can close the data-efficiency gap given enough epochs, and some recent high-profile
systems have dropped equivariance deliberately. The countervailing result — that equivariant models
outperform non-equivariant ones *at every tested compute budget*, with scaling following a power law
— is what tips the decision here. Adopt, and instrument it: an equivariance-ablated control model
runs alongside so the constraint's value is measured rather than assumed.

**One correction to the blueprint's version.** It defines `W_Q, W_K, W_V ∈ R^(D×D)` as unconstrained
linear maps and then claims "exact rotational SO(3), Galilean, and Lorentz equivariance natively in
attention weights." Those two statements are incompatible. An unconstrained dense matrix over mixed-
grade channels does not preserve grade structure and therefore breaks equivariance. Equivariance
requires the linear maps to be restricted to grade projections and geometric products, which is
precisely what the GATr line constructs and what costs expressivity. Take the constrained layers, or
drop the equivariance claim — not both.

### ND-RoPE over continuous spacetime coordinates (adopt, with a required fix)

The original spec said latents are "stamped with a coordinate" without specifying the mechanism. The
blueprint's answer — partition head dimension into four orthogonal 32-dim subspaces, apply
independent planar rotations parameterized by continuous `(t, x, y, z)` — is the right one, and its
reference implementation is correct.

**The fix it needs.** The rotation angle is `ξ·θᵢ` with `ξ` in **metres and seconds**. That makes the
positional encoding depend on the absolute unit scale of the scene: the same physics simulated in a
1 m flume and a 1 km river produce rotations three orders of magnitude apart, and a scene authored in
millimetres is unrecognizable relative to one in metres. `θ_base = 10000` implicitly assumes
coordinates of order 1–10⁴.

The fix falls directly out of this spec's existing machinery: **normalize coordinates by the
scenario's characteristic length and time scales before encoding** — the same `L` and `T` used to
form the dimensionless groups. ND-RoPE then operates on `(t/T, x/L, y/L, z/L)`, which is
scale-invariant by construction. This is a case where the two documents' ideas repair each other:
the blueprint supplies the encoding mechanism, Π-space supplies the normalization it was missing.

### Attention sinks and block-diagonal multi-stream masks (adopt)

Permanent non-evicting sink tokens absorbing surplus softmax mass are the standard, correct fix for
long-horizon streaming stability, and an always-on instance needs them.

The block-diagonal stream mask is the concrete implementation of what [`06`](06-runtime.md) argued
for on security grounds: streams share the kernel batch and the weights, and `−∞` off-diagonal blocks
make cross-session attention structurally impossible. The blueprint and this spec independently
arrived at the same answer, which is mild evidence it is the right one. Adopted as the normative
mechanism.

### ACT / PonderNet formulation for variable depth (adopt)

The original spec specified variable recursion depth but hand-waved the halting rule. The blueprint
supplies the standard formulation: per-step halting probability from a sigmoid unit, probability-
weighted aggregation of intermediate states, and — importantly — a ponder cost expressed as KL to a
geometric prior. That last detail is the PonderNet construction and it is the right choice, because
the naive ACT ponder penalty is notoriously unstable and tends to collapse to always-halt or
always-max.

Retained from this spec and **not** in the blueprint: the causal/non-causal train-serve mismatch
([`02`](02-trunk.md) §2.3) and the release-blocking consistency test. ACT halting is per-token and
causal, which is better than top-k routing on this axis; the test stays anyway.

### Byte-level input, spectral operators as a sublayer, OT flow matching (adopt)

- **Byte-level tokenization** avoids tokenizer-induced arithmetic blindness on scientific notation
  and code. Correct call. Fixed 4-byte patching is crude compared to entropy-based dynamic patching,
  but the direction is right.
- **FNO spectral sublayers** are a good component for Tier A's coarse operator — see §12.3 for the
  limits that must be attached to them.
- **OT conditional flow matching** with straight probability paths converging in 4–10 ODE steps is
  the correct modern choice for the continuous heads. Adopted as the normative formulation in
  [`04`](04-generation.md).

---

## 12.2 The central correction: conditioning is not rendering

This is the one architectural disagreement that matters, and everything else is detail beside it.

The blueprint's §5.2 Step 4 generates the waterfall video by having the CFM head integrate its vector
field **"conditioned directly on the Level-Set field φ and velocity vectors u."**

That is still a learned sampler producing pixels. Conditioning a flow-matching head on a physical
field makes its output *correlated with* that field; it does not make the output *derived from* it.
Nothing in the flow-matching objective penalizes a frame in which water volume changes between
timesteps — the loss measures distance to a target latent, and mass conservation is not a property of
latent distance.

This is exactly the failure the measurements describe. On VideoPhy-2's hard subset the best model
reaches 21.9% joint performance, with conservation of mass and conservation of momentum each violated
at roughly 40%; Physion-Eval found ≥1 expert-identifiable physics flaw in 83.3% of exocentric and
93.5% of egocentric generated clips. Those numbers come from conditional generators. Adding a
level-set to the conditioning vector is a stronger conditioning signal, not a different mechanism.

**The correction stands as written in [`04`](04-generation.md):** the video must be a *differentiable
rasterization of the solved field* — a deterministic view transform, not a sample. If `φ` says the
free surface is at a given height, the renderer puts it there because it is a function of `φ`, not
because it learned to. Generative heads remain available for illustrative output, labelled and
structurally prevented from impersonating a prediction.

**A related claim to retract.** The blueprint states the video's "motion vectors embedded in the
latent video precisely match the Navier-Stokes velocity vectors **u**." Under conditional generation
there is no mechanism enforcing that match — it is a hope. Under rendering it is a tautology, because
the motion vectors *are* **u**. The claim is only true in this spec's architecture, which is the
argument for this spec's architecture.

---

## 12.3 Checked and found wrong

### The FNO backbone cannot be the solver for this problem

The blueprint states the model "does not call external models like ... OpenFOAM for simulation" and
that its spectral layers solve the pressure Poisson equation `∇²p = −ρ∇·(u·∇u)` in `O(N log N)`
inside the stack. The mechanism is real — FFT-based Poisson solves are genuinely `O(N log N)` — but
three things break it for the waterfall:

1. **It is the architecture with the measured extrapolation failure.** A trained FNO reaches ~46.7%
   relative L₂ error under a 10× Reynolds shift and is beaten by retrieval baselines. The blueprint
   proposes solving a Reynolds-doubling query using the method whose Reynolds-shift failure is the
   best-documented result in the field.
2. **Mode truncation removes exactly the physics being claimed.** The spec sets `K_max = 16` modes
   (the reference implementation uses `modes = (8, 8, 8)`). Spray, air entrainment, the turbulent
   cascade, and the Kolmogorov microscale — all of which the walkthrough reports on — are
   high-wavenumber phenomena. Truncating to 8 modes per axis is a low-pass filter applied to the
   answer. FNOs have a documented spectral bias toward low frequencies even without aggressive
   truncation.
3. **Boundaries.** FNO is not strictly limited to periodic domains — the pointwise linear bias term
   recovers some non-periodic behavior — but performance is documented to drop once periodicity
   fails, and non-periodic data through a spectral transform produces Gibbs oscillations. A waterfall
   is walls, an inlet, an outlet, and a free surface. The interesting physics is *at* the boundaries,
   which is where the method is weakest.

**Disposition:** FNO sublayers are kept as a component of Tier A's coarse operator, where a fast
smooth-field estimate is genuinely useful, and where the trunk predicts a *correction* to it. They
are not the answer to the canonical query. Tier B stands.

### A reference-implementation bug in the spectral layer

`out_ft[:, :, :mx, :my, :mz] = ...` fills only the low positive wavenumbers. `torch.fft.rfftn`
halves only the **last** axis; the `x` and `y` axes retain both positive and negative frequencies,
with the low-magnitude negative wavenumbers living at the *end* of those axes. Standard 3-D FNO
implementations write four (2-D) or eight (3-D) corner blocks for this reason. As written, half the
low-frequency content along two of three axes is discarded — the layer is throwing away
low-wavenumber information it intends to keep.

### The 3.4T parameter count is right; the KV cache is the resource problem

Recomputed: attention `4D² = 4.295 B`/layer; SwiGLU `3 × D × D_ffn = 12.885 B`/layer; total
`17.18 B × 192 layers ≈ 3.30 T`, consistent with the stated ~3.41T including embeddings and spectral
weights. The arithmetic holds.

The unexamined number is the KV cache. With 256 heads × 128 dims and **no** grouped-query or latent
attention:

```
per token per layer :  2 (K,V) × 32,768 × 2 bytes  =  128 KiB
per token           :  128 KiB × 192 layers        =   24 MiB
per 1 M-token stream:  24 MiB × 1,048,576          =  ~25 TiB
```

Roughly **75× the per-token KV footprint of a modern GQA model** of comparable serving ambition. On
the stated 12.58 PB fabric that is ~500 concurrent full-length contexts before weights and
activations — a small number for a system whose premise is serving everything, and flatly
incompatible with the "non-evicting persistent circular cache" the blueprint also specifies.

This is not fatal and not hard: grouped-query or multi-head latent attention cuts it by one to two
orders of magnitude. It is called out because a document that specifies interconnect bandwidth to
three significant figures should not leave its dominant memory term uncomputed.

### "6,144 effective layers" overstates what recursion buys

32 ponder passes through 192 tied-weight layers is a fixed-point iteration, closer in expressivity to
a deep equilibrium model than to 6,144 distinct layers. Recursion with shared weights buys *iterative
refinement*, not *depth-equivalent capacity*, and the distinction matters for any capability claim
that leans on the number.

### The worked example contradicts itself — and nothing in the architecture notices

This is the most instructive finding, because the errors are not in the architecture but in an
answer the architecture would have shipped.

| Check | Result |
|---|---|
| Bélanger conjugate depths (2.28 m, 3.24 m) | **Correct.** Recomputed from `h₂/b = ½(√(1+8Fr²) − 1)` at both states. |
| Parameter count ~3.4 T | **Correct.** |
| Discharge `Q` | **Contradictory.** Step 1 establishes `Q = 3.0 m³/s` from `u = 1.5 m/s × A = 2.0 m²`. The report emits `3.66 → 7.32 m³/s`. Back-solving `b = Q/(W·v)` from the reported `b = 0.125 m` gives `Q = 3.67`, so the later figures are self-consistent with each other and inconsistent with the stated input. |
| Crest velocity | **Contradictory.** The reported `v_impact = 14.69 m/s` with `h = 10 m` implies `v_crest ≈ 4.43 m/s`, not the 1.5 m/s given as the inflow. |
| "Deeper into the supercritical turbulent regime" | **Category error.** Reynolds governs laminar→turbulent; Froude governs subcritical→supercritical. Re doubles, but the reported Froude number *decreases* (13.29 → 9.77) — the flow becomes *less* supercritical while becoming more turbulent. Two different transitions, conflated into one sentence. |
| Over-determined boundary condition | **Inconsistent.** The walkthrough imposes `u'_in = 2·u_in` as a Dirichlet condition *and* applies the Poleni weir relation to let head rise as `2^(2/3)`. You cannot fix the inlet velocity and also let the free surface set it; the velocity/depth split is part of the solution, not an input. |
| "+142% spray / intense aeration increase" | **Contradicted by its own formula.** Using the quoted Ervine-Falvey relation `q_air/q_water ≈ 0.013 Fr²`: air entrainment scales as `q_water × Fr²`, i.e. `2 × (95.5/176.6) = 1.08`. Total air entrainment rises about **8%**, not 142%. The narrative and the cited correlation disagree. |

**Why this matters more than any individual error.** These are exactly the mistakes a fluent,
confident system produces: internally plausible, quantitatively specific, mutually inconsistent, and
undetectable by reading. The blueprint has no verification gates, no residual checks, no convergence
study, no provenance, and no mechanism by which the report's numbers are reconciled against the
field they supposedly describe. It would have emitted this analysis with a photorealistic video
attached and no indication anything was wrong.

Two of the errors — the discharge discontinuity and the aeration figure — are caught by a single
arithmetic consistency check against conserved quantities. That check is what
[`03`](03-physics.md) §3.3 and [`04`](04-generation.md) §4.4 exist to mandate.

**A further observation.** The walkthrough's reliable content is analytic hydraulics — Poleni,
Bélanger, Ervine-Falvey — which are dimensionless control-volume relations, not a Navier-Stokes
solve. The blueprint reaches for closed-form similarity relations precisely when it needs a
trustworthy number. That is an unintentional argument for [`01`](01-representation.md)'s thesis: the
dimensionless relations are where the transferable physics lives.

### Autonomous weight hot-swapping should not ship

The blueprint's serving section specifies "self-improved weights derived from RLVR loops update
dynamically via asynchronous weight double-buffering without interrupting continuous streaming."

Nothing in production AI as of 2026 autonomously rewrites its own weights; every functioning
self-improvement loop modifies prompts, code, tools, and data around fixed parameters, and
self-correction without an external verification signal largely does not work. Pushing autonomously
derived weights into a live multi-tenant instance with no evaluation gate and no rollback discipline
combines an unvalidated capability with an unbounded blast radius. See
[`09`](09-self-improvement.md) for the bounded version: propose, solve, verify, mint data, and
retrain **offline** behind an eval gate.

### The anti-MoE argument is weaker than the constraint it defends

Recorded for accuracy, not to relitigate: the dense constraint (C2) is the operator's decision and
this spec holds it. But the blueprint's *justification* — that routing fractures coupled multiphysics
at interfaces, and that routers collapse on high-entropy continuous manifolds — is theory
contradicted by the directly relevant measurement. In multi-physics operator learning, sparse routing
*eliminated* negative transfer that dense co-training suffered, routing bifurcated cleanly by domain,
and the published design routes shared structure through **shared** experts precisely so that
coupling is not fractured. The interface concern is real but is what shared experts address.

Hold C2 because it is the operator's call and because Π-space gives it a fighting chance — not
because MoE has been shown to fail here. It has been shown to help. That is why
[`02`](02-trunk.md) §2.4 carries a tripwire instead of a victory lap.

### Smaller items

- **`L_geometric` forcing grade-2 attention to equal `∇∧u`** over-constrains the internals and is
  undefined for non-field tokens; it needs masking to physics tokens and should be a weak auxiliary,
  not a primary term. Supervising intermediate representations to equal specific physical quantities
  is a known way to lose performance.
- **The flow head as coded** conditions on a single pooled `h_backbone` vector and emits one
  `d_target = 1024` vector. It cannot produce a 4K frame; the prose implies per-token conditioning
  that the implementation does not have.
- **The Blender bridge** `exec()`s model-generated code with no sandbox, in a system that also has
  internet access — the boundaries in [`05`](05-agency.md) §5.6 are not optional. Separately,
  `recv(65536)` without length-prefixed framing will fragment or truncate any payload over one
  buffer, because stream sockets do not preserve message boundaries.

---

## 12.4 Net effect on this specification

**Changed:**
- [`01`](01-representation.md) — Clifford multivector fiber added alongside dimension exponents;
  ND-RoPE adopted as the coordinate encoding, with mandatory normalization by characteristic scales.
- [`02`](02-trunk.md) — halting formalized as PonderNet-style ACT with a geometric prior; attention
  sinks and block-diagonal stream masks adopted; grouped-query/latent attention made a requirement
  rather than an omission.
- [`03`](03-physics.md) — FNO spectral sublayers admitted as a Tier A component, with mode-truncation
  and boundary limits attached.
- [`04`](04-generation.md) — OT-CFM adopted as the normative continuous formulation; the
  conditioning-is-not-rendering distinction sharpened.

**Unchanged, and reinforced:**
- Render-the-field. The blueprint's own worked example is the argument for it.
- Two-tier physics with external verified solves. An FNO backbone is not a substitute.
- Verification gates, provenance, and reported distance-from-manifold. Six of the seven errors above
  are caught by checks the blueprint does not have.
- Bounded, offline, eval-gated self-improvement.
