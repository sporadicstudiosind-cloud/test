# 01 — Representation

> Everything the model has ever seen occupies one sequence, in one coordinate frame, with its
> physical units attached.

## 1.1 The typed latent

There is one element type in ORRERY. Text, a video patch, a millisecond of audio, a cell of a fluid
mesh, a mouse click, and a term in an equation are all the same object with different stamps.

```
latent := {
    content    : Cl(3,1) ⊕ R^d_sym   # multivector fiber ⊕ symbolic channels  (§1.2)
    type       : R^d_type            # text | patch | audio | field | action | symbol | meta
    coord      : R^5                 # (t, x, y, z, λ)  — continuous, normalized (§1.3)
    dim        : Z^7                 # exponents of (M, L, T, I, Θ, N, J)      (§1.4)
    frame_id   : int                 # which coordinate frame coord is expressed in
}
```

`content` carries geometric structure rather than being an unstructured vector; `type` and `dim` are
embedded and added; `coord` enters through rotary encoding rather than addition. Each of the three
structural fields is doing distinct work, and the division is worth stating up front because it is
easy to assume one subsumes another:

| Field | Symmetry / structure it supplies |
|---|---|
| `content` as a Clifford multivector | rotational SO(3), Galilean, Lorentz equivariance |
| `dim` → Π-space | **scale** similarity: Re, Fr, We, Ma |
| `coord` → ND-RoPE | continuous relative position across space and time |

Rotational equivariance does not help a model extrapolate to a doubled inflow. Scale similarity does
not tell it that vorticity is a bivector. They compose.

### Why coordinates instead of per-modality position embeddings

Standard practice gives each modality its own positional scheme: 1-D for text, 2-D for images, 3-D
for video, a separate mesh index for simulation. That choice quietly makes cross-modal reasoning a
*translation* problem — the model must learn a mapping between four incommensurable indexing systems
before it can relate a pixel to a pressure.

ORRERY uses one frame. A fluid cell at `(0.4 s, 1.2 m, 0.0 m, 3.1 m)`, the video pixel that observes
it, and a mouse click on the viewport showing it carry coordinates in the same space, related by a
declared camera transform. Three consequences follow directly:

- Rendering a solved field is a *change of view* within the representation, not a handoff to a
  decoder that must re-learn geometry. This is the precondition for [`04`](04-generation.md).
- GUI control needs no separate spatial machinery — a screen is a 2-D slice, a click is a latent at
  `(t, x, y)`. See [`05`](05-agency.md).
- Multi-scale physics is expressible: `frame_id` lets a nested high-resolution patch declare its own
  origin and scale, with the transform to the parent frame as a learnable, inspectable object.

`λ` is the spectral axis — wavelength for imaging, frequency band for audio, mode number for spectral
solvers. It costs one dimension and makes Fourier-domain and pixel-domain data commensurable.

## 1.2 The Clifford multivector fiber

`content` is not an unstructured vector. Its physical channels are structured as multivectors in the
spacetime Clifford algebra `Cl(3,1)`, a 16-dimensional graded space, with physical quantities mapped
to the grade that actually describes them:

| Grade | Object | Physical content |
|---|---|---|
| 0 | scalar | pressure, density, temperature, turbulent kinetic energy, level-set distance |
| 1 | vector | velocity, acceleration, body force |
| 2 | bivector | **vorticity**, angular momentum, shear rate, EM field tensor |
| 3 | pseudovector | magnetic flux density, helical current |
| 4 | pseudoscalar | helicity, topological invariants |

Vorticity is a bivector, not a vector. Representing it as a 3-vector is a coordinate accident of
three dimensions that breaks under Lorentz transformation and obscures the fact that `ω = ∇∧u` is an
oriented plane element. Encoding the grade structure means the network's linear operations preserve
geometric meaning instead of learning to approximate it.

**Evidence this pays.** Geometric-algebra transformers outperform non-geometric baselines on n-body
dynamics and robotic planning; the Lorentz-equivariant variant matches or beats domain-specific
baselines across high-energy physics tasks; Clifford-steerable convolutions show gains specifically
on fluid dynamics and relativistic electrodynamics forecasting.

**Cost, honestly.** Equivariance is a constraint, and constraints cost expressivity. Under C9 the
usual efficiency argument for equivariance weakens — an unconstrained model with enough augmentation
and enough epochs closes much of the data-efficiency gap, and some prominent systems have dropped
equivariance deliberately. The result that tips the decision is that equivariant models outperform
non-equivariant ones *at each tested compute budget*, with a power-law scaling advantage rather than
a fixed offset. This is adopted as a **measured bet**: an equivariance-ablated control trains
alongside, and [`11`](11-evaluation.md) reports the delta.

**Implementation constraint that must not be violated.** Equivariance requires the linear maps
`W_Q, W_K, W_V` to be restricted to grade projections and geometric products. An unconstrained dense
`R^(D×D)` matrix over mixed-grade channels destroys grade structure, and a design that uses
unconstrained projections cannot also claim exact equivariance. Take the constrained layers or drop
the claim — not both.

Symbolic channels (`R^d_sym`) carry text, code, and action content and are grade-agnostic; the fiber
is the direct sum, so a sentence and a velocity field share a space without the sentence pretending
to have a grade.

## 1.3 Coordinates enter as rotation, normalized by characteristic scale

Coordinates are encoded with **N-dimensional rotary position embedding**: the head dimension is
partitioned into orthogonal subspaces, one per axis, each rotated by an angle proportional to that
token's continuous coordinate. Relative position across space and time then falls out of the query-key
inner product, exactly as 1-D RoPE gives relative sequence position — but continuous, and in four
dimensions.

**The normalization that makes it work.** The rotation angle is proportional to the raw coordinate
value, so encoding in metres and seconds makes the representation depend on the absolute unit scale
of the scene: the same physics in a 1 m flume and a 1 km river produce rotations three orders of
magnitude apart, and a scene authored in millimetres is unrecognizable relative to one in metres.

Coordinates are therefore normalized by the scenario's characteristic length and time scales —
**the same `L` and `T` used to form the dimensionless groups in §1.4** — before encoding:

```
coord_encoded = ND-RoPE( t/T, x/L, y/L, z/L, λ/Λ )
```

This is not a detail. It means the positional encoding and the physics representation are
nondimensionalized by the same quantities, so a scenario and its dynamically similar counterpart at
another scale produce the *same* positional structure as well as the same Π-point. Without it, the
scale-invariance argument of §1.4 is undone by the position encoding.

## 1.4 Dimensional exponents, and why they are the most important field

Every numeric latent carries the exponents of the seven SI base dimensions. A velocity is
`(0, 1, −1, 0, 0, 0, 0)`. A pressure is `(1, −1, −2, 0, 0, 0, 0)`. A dimensionless quantity is the
zero vector. Text and action latents carry the zero vector and a `type` that marks them
non-dimensional.

This is a small change with a large consequence: **Buckingham-π nondimensionalization becomes
expressible inside the network.** Given a set of latents with dimension vectors, the null space of
the dimension matrix *is* the set of valid dimensionless groups. The trunk can form Re, Fr, We, and
Ma as linear operations in exponent space and use them to rescale content.

### The argument this rests on

The Reynolds-generalization study found that representation geometry is the primary variable
governing cross-regime transfer — methods cluster by *how they encode state*, not by whether their
dynamics are learned or retrieved, and no tested method demonstrated learned Reynolds-invariance.

The inference drawn here is not "try harder to learn invariance." It is: **choose a representation
in which the invariance is already true.** In SI units, a 0.1 m/s laboratory flume and a 40 m/s
atmospheric jet are distant points. In Π-space, if their Reynolds and Froude numbers match, they are
*the same problem*, and a model trained on one has genuinely seen the other. Doubling the inflow of a
waterfall stops being an out-of-distribution jump and becomes a bounded translation along two known
axes.

This is also the primary mitigation for the negative-transfer finding of
[`00`](00-premise.md): open-channel and porous-media flows impose incompatible demands when compared
in raw units, but their conflict narrows substantially once each is expressed relative to its own
characteristic scales. Whether it narrows *enough* to save the dense constraint is an empirical
question with a tripwire in [`02`](02-trunk.md).

### Where it stops working — state this in any derived work

Nondimensionalization delivers exact invariance only when the similarity group is known and complete.

- Fully-developed turbulence has no closed similarity solution; Re-matching does not make two
  turbulent flows equivalent at all scales.
- Multiphase flow with surface tension, phase change, and non-Newtonian rheology generates more
  independent groups than can be spanned by any practical dataset.
- Real geometry breaks similarity: two flows can match on every dimensionless number and differ
  because a boundary is rough.

Π-space converts a cliff into a slope. It does not remove the slope. The escalation policy in
[`03`](03-physics.md) exists precisely because this guarantee is partial.

## 1.5 Codecs are layers, not models

Constraint C1 forbids a frozen third-party VAE, audio tokenizer, or vision encoder. Perceptual
codecs are therefore *layers of ORRERY*, in the same parameter set, trained end to end with
everything else.

**Discrete where structure is symbolic, continuous where fidelity is physical:**

| Modality | Encoding | Rationale |
|---|---|---|
| Text, code, actions | discrete tokens | symbolic; exact reproduction matters; tractable likelihood |
| Images, video, audio | continuous latents | fidelity loss from quantization is unacceptable downstream |
| Simulation fields | continuous latents, no quantization | discretizing a velocity field discards the conserved quantities the whole system exists to preserve |
| Mathematics | dual — symbolic tokens *and* continuous numeric latents | expressions need structure; magnitudes need range |

Fusion is early and total: all latents enter the trunk together from layer zero. Late fusion — encode
separately, concatenate at the top — reintroduces the translation boundary that 1.1 exists to
eliminate.

### The cost, stated plainly

Joint codec training is the single largest schedule risk in this project. Frozen codecs are frozen
for a reason: they stabilize everything downstream, and training them jointly with a large trunk
invites posterior collapse, codebook death in the discrete branch, and a reconstruction/likelihood
tug-of-war that can silently degrade both. [`08`](08-training.md) allocates Stage 0 entirely to
this problem and [`10`](10-open-problems.md) carries the kill criterion.

The mitigation is not clever: it is a long Stage 0 with reconstruction-only objectives, a
conservation penalty on field reconstruction (a codec that loses mass is disqualified regardless of
its PSNR), and codec-frozen ablations run in parallel so the cost of C1 is *measured* rather than
assumed.

## 1.6 Field cells are not pixels

A simulation field arrives with structure that must not be flattened away:

- **Mesh topology.** Unstructured meshes carry adjacency. Cell latents get neighbor edges, and the
  trunk's attention is biased by mesh distance rather than index distance.
- **Boundary conditions.** Walls, inlets, outlets, and free surfaces are *first-class latents* with
  their own type, not properties smuggled into cell content. A doubled inflow is literally an edit to
  an inlet latent — which is what makes the canonical query expressible at all.
- **Conserved quantities.** Total mass, momentum, and energy are attached as `meta` latents. They
  give the model a directly readable signal for whether its own output conserves, and they are what
  the loss in [`08`](08-training.md) penalizes.
- **Provenance.** Every field latent records whether it came from an uploaded simulation, a Tier A
  correction, a Tier B verified solve, or a generative prior. The model must be able to tell you
  where a number came from, and it cannot do that if provenance is not carried in the
  representation.

## 1.7 Consequences for the rest of the spec

| This section establishes | Which enables |
|---|---|
| One coordinate frame | Render-the-field ([`04`](04-generation.md)); GUI as geometry ([`05`](05-agency.md)) |
| Dimension exponents | Π-space escalation policy ([`03`](03-physics.md)); dense-constraint survival ([`02`](02-trunk.md)) |
| Boundary conditions as latents | "Double the inflow" is an editable object, not a prompt |
| Provenance in the latent | Reports that state which tier answered ([`04`](04-generation.md)) |
| In-weight codecs | C1 honored; Stage 0 risk ([`08`](08-training.md)) |
