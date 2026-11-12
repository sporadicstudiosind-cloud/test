# 04 — Generation: Render the Field

> The clip is a photograph of a solve, not a dream about one.

## 4.1 Decode heads

All heads read from the same trunk output. None of them is a separate model; they are output
projections of the shared parameter set.

| Head | Mechanism | Outputs |
|---|---|---|
| Symbolic | autoregressive over discrete tokens | text, code, math expressions, action commands |
| Continuous | OT conditional flow matching over continuous latents | image, video, audio latents |
| Field | OT-CFM in field-latent space, conservation-projected | velocity, pressure, density, phase fraction, stress |
| Geometric | direct regression in the shared frame | motion vectors, displacement fields, camera transforms, mesh deltas |

The continuous formulation is **optimal-transport conditional flow matching**: straight probability
paths from a Gaussian base to the target latent, with the trunk predicting the velocity field of the
transport. Straight paths mean generation converges in roughly 4–10 ODE steps rather than the
hundreds of denoising steps a diffusion schedule needs — which matters for an always-on system where
a continuous head may be running on many streams at once. Conditioning is per-latent, not from a
single pooled state; a pooled conditioning vector cannot carry a high-resolution frame.

The symbolic and continuous heads are conventional. The **field** and **geometric** heads are where
this architecture departs from every current omnimodal model, and §4.2 is the reason.

## 4.2 The central rule: physics video is rendered, never generated

**The evidence.** On the hard subset of VideoPhy-2 the best model reaches 21.9% joint performance —
clips scoring ≥4/5 on *both* semantic adherence and physical commonsense. Conservation of momentum
and conservation of mass are each violated at roughly 40%. Physion-Eval, using STEM-trained
annotators on 12,718 generated videos, found at least one identifiable physics flaw in 83.3% of
exocentric and 93.5% of egocentric clips.

These are not prompt-engineering problems. A video head samples from a distribution over *pixel
sequences that look like the training data*. Conservation of mass is not a visual property. Nothing
in that objective penalizes matter appearing.

**Nor is stronger conditioning a fix.** The tempting near-miss is to keep the generative head and
condition it on the solved field — feed the level-set `φ` and velocity `u` into the flow-matching
head and let it paint the frames. This is worth naming explicitly because it looks like it should
work and does not. Conditioning makes the output *correlated with* the field; it does not make the
output *a function of* the field. The flow-matching objective measures distance to a target latent,
and no term in it penalizes a frame where water volume changed since the last one. Every measurement
above was produced by conditional generators. A richer conditioning vector is a stronger hint, not a
different mechanism.

The distinction is between a hint and a constraint. Rendering is a constraint.

**The rule.** For any output claiming physical accuracy, the pipeline is:

```
solved field (Tier A or Tier B)  →  differentiable renderer  →  video
```

not

```
prompt + conditioning  →  video head  →  video
```

The renderer is differentiable rasterization / volumetric integration in the shared coordinate frame
of [`01`](01-representation.md), so it is a *view transform* on data the model already holds, not a
translation into a foreign representation. Because the frame is shared, the camera is just another
latent, and "show me that from underneath" is a coordinate change rather than a re-generation.

**Motion vectors are the native intermediate.** The geometric head emits velocity and displacement
fields directly; the video is derived from them. This inverts the usual pipeline, where motion is
estimated *from* generated pixels and inherits every physical error those pixels contain. Asking
ORRERY for motion vectors is asking for something it already has.

### The generative video head still exists — with a hard boundary

Non-physical video (a stylized illustration, a mood board, a UI mockup animation, a diagram) is
generated conventionally by the continuous head. That is legitimate and useful.

What is prohibited is **laundering**: generative output must never be presented as, or silently
substituted for, a physical prediction. Every rendered frame carries the provenance of the field it
came from ([`01`](01-representation.md) §1.6), and any output whose provenance chain does not
terminate in a solve is labelled as an illustration. This is enforced structurally — the renderer
takes a field as input and the field carries provenance — rather than by asking the model to be
careful.

### Cost of the rule

- Physics video is only as fast as the solve behind it. For Tier B queries that is minutes to hours.
- Photorealism is bounded by the renderer, not by a learned prior. Rendered output will look like a
  scientific visualization unless substantial work goes into materials and lighting. A generative
  model produces prettier waterfalls, and wrong ones.
- Anything the solver does not model — spray, foam microstructure, fine mist — cannot be rendered
  from the field and must be either simulated explicitly or acknowledged as absent. Adding it
  generatively is the laundering the previous paragraph forbids.

This is a real product tradeoff, and the answer is: if the user wants a beautiful waterfall, use the
generative head and label it. If they want to know what happens when the inflow doubles, they get the
render.

## 4.3 Conservation projection on field output

The field head's raw output is projected onto the constraint manifold before it is used:

- **Divergence-free projection** for incompressible velocity — solve a Poisson equation for the
  pressure correction, exactly as the integrator does.
- **Mass rebalancing** for density and phase fractions, against the `meta` conserved-quantity latents.
- **Bound enforcement** — non-negative densities, volume fractions in [0,1], physically admissible
  temperatures.

A learned head that emits a *nearly* divergence-free field is emitting a field with sources and sinks
in it. Projection is cheap and it converts "approximately conserving" into "conserving," which is the
difference between a result and an artifact.

## 4.4 Reports

Text is decoded by the trunk while the solved field is *in context* as field latents — not from a
summary, not from a caption, from the numbers. A report therefore quotes actual quantities: the new
discharge, where the hydraulic jump forms, the change in wetted area.

**Mandatory provenance block.** Every physics report states:

1. which tier produced the answer (A, A-with-audit, or B);
2. the Π-space distance `d` from the training manifold, and whether a regime boundary was crossed;
3. the verification residuals and, for Tier B, the observed order of accuracy from the grid
   convergence study;
4. what the model did *not* model.

A system that can produce a confident video and a confident paragraph but cannot say *"this sat 1.8
Π-units outside my validated range and I solved it numerically; continuity residual 3e-7; the spray
is not modelled"* is not a scientific instrument. This block is the difference.

## 4.5 Audio, images, and the rest

Conventional, and deliberately unremarkable:

- **Audio** decodes continuously with the same temporal axis as video, so lip-sync and
  physical-event/sound alignment are frame-consistency constraints rather than a post-hoc merge.
  Physically-derived audio (the sound of that waterfall at that discharge) is a rendering of the
  pressure field, subject to the same rule as video — and is flagged as an approximation, since
  acoustic rendering from a coarse CFD pressure field is a weak model of real sound.
- **Images** are single-frame video.
- **Mathematics** decodes dually: symbolic tokens for the expression, continuous latents for
  numerical magnitudes, checked against each other. A derivation whose symbolic and numeric branches
  disagree is flagged rather than emitted.

## 4.6 What this section commits to

- The word "accurate," applied to a video, means the pixels are a view of a verified field.
- Motion vectors are primary output, not derived.
- Generative video is permitted, labelled, and structurally prevented from impersonating a
  prediction.
- Field output is projected onto conservation constraints, not merely trained toward them.
- Reports carry provenance, distance-from-manifold, and residuals, or they are not reports.
