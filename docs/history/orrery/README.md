# ORRERY

**An architecture specification for a physics-native omnimodal foundation model.**

An orrery is a machine that models a solar system — not a picture of one, a mechanism whose gears
are constrained to move the way the planets actually move. That distinction is the entire thesis of
this document.

---

## What this is

A buildable specification for a single dense model that ingests text, images, video, audio, tabular
data, mathematics, and live physics simulations, and emits any of those back — including velocity
fields, motion vectors, and GUI actions. One parameter set. No expert routing. No calling out to
other models; code and APIs only. One always-resident instance that allocates its own compute across
concurrent streams.

It is written to be honest about which parts are engineering and which are open research. Section
[`10-open-problems.md`](spec/10-open-problems.md) carries the failures, each with a kill criterion,
and [`00-premise.md`](spec/00-premise.md) closes with the strongest argument *against* building this
at all.

## The thesis in three claims

1. **Represent physics dimensionlessly or don't bother.** Every latent carries a physical coordinate
   and a vector of SI dimension exponents, so the trunk can work in Π-space (Re, Fr, We, Ma). This is
   a direct response to the finding that no neural PDE method has been shown to learn
   Reynolds-invariance, and that *representation geometry* — not learned-vs-retrieved dynamics — is
   the variable that governs cross-regime transfer.
2. **Never generate physics video. Render it.** The model emits velocity and density fields; video is
   a differentiable rasterization of those fields. Video heads sampled from a visual prior violate
   conservation of mass and momentum at roughly a 40% rate each, and 83–94% of generated clips
   contain at least one expert-identifiable physics flaw. There is no prompt that fixes this, only an
   architecture that routes around it.
3. **Focus is recursion depth.** A dense trunk applied a variable number of times per token gives
   "divert more compute to this stream" a mechanical meaning: a budget of recursion steps, allocated
   by the model and enforced by the scheduler.

## Read in this order

| Doc | What it settles |
|---|---|
| [`00-premise.md`](spec/00-premise.md) | Goals, hard constraints, and the case against the whole design |
| [`01-representation.md`](spec/01-representation.md) | Typed latents, the shared coordinate frame, Π-space, in-weight codecs |
| [`02-trunk.md`](spec/02-trunk.md) | Dense weights, recursive depth, what "focus" physically is |
| [`03-physics.md`](spec/03-physics.md) | The two-tier solver and when each tier fires |
| [`04-generation.md`](spec/04-generation.md) | Decode heads, motion vectors, render-the-field |
| [`05-agency.md`](spec/05-agency.md) | Action latents, Blender, and an unflattering reliability number |
| [`06-runtime.md`](spec/06-runtime.md) | The resident instance, scheduling, duplex streaming, KV isolation |
| [`07-data.md`](spec/07-data.md) | Why internet access solves the easy half and none of the hard half |
| [`08-training.md`](spec/08-training.md) | Seven-stage curriculum, losses, stability risks |
| [`09-self-improvement.md`](spec/09-self-improvement.md) | The bounded loop, and why residuals are a good reward |
| [`10-open-problems.md`](spec/10-open-problems.md) | What is unsolved, with kill criteria |
| [`11-evaluation.md`](spec/11-evaluation.md) | How you would know any of this worked |
| [`12-blueprint-reconciliation.md`](spec/12-blueprint-reconciliation.md) | What was adopted, corrected, and found wrong in an alternative blueprint |
| [`references.md`](references.md) | Every quantitative claim, with source |

[`12`](spec/12-blueprint-reconciliation.md) is worth reading even out of order. An alternative
architecture document was supplied during design; several of its mechanisms were better than what
this spec originally had and were adopted (Clifford multivector latents, ND-RoPE, attention sinks,
PonderNet-style halting). Its worked example also contained six mutually inconsistent quantities that
no part of its architecture would have caught — which is the clearest available argument for the
verification gates in [`03`](spec/03-physics.md) and [`04`](spec/04-generation.md).

---

## The canonical query, traced end to end

> *"Here's a simulation of fluid going down a waterfall. Calculate the flow if I double the input
> water, then give me an accurate physics-based video and a short report."*

This query is the spec's test case because it exercises every subsystem and because the obvious
architecture answers it wrongly and confidently.

**1 · Ingest** — [`01`](spec/01-representation.md)
The uploaded simulation arrives as fields on a mesh. Each cell becomes one latent stamped with its
true position `(t, x, y, z)` in metres and seconds, its type (`field-cell`), and a dimension vector
marking it as a velocity (`L¹T⁻¹`), a pressure (`M¹L⁻¹T⁻²`), or a density (`M¹L⁻³`). The prompt text
becomes latents in the same stream with sequence positions. Nothing is converted, captioned, or
handed to a subsystem — the fluid and the sentence are adjacent tokens in one sequence.

**2 · Nondimensionalize** — [`01`](spec/01-representation.md)
The trunk reads the geometry and the dimension vectors and forms the governing dimensionless groups:
Reynolds, Froude, Weber. The scenario is now a *point in Π-space* rather than a specific waterfall.

**3 · Locate the request** — [`03`](spec/03-physics.md)
"Double the input water" is not a doubling of Reynolds number — it is a coupled move along Re and Fr
with a free-surface height change. The model computes the new Π-point and measures its distance from
the training manifold. This is the decision that everything downstream depends on, and it is a
*measured* distance, not a vibe.

**4 · Choose a tier** — [`03`](spec/03-physics.md)
Near the manifold, Tier A answers in one forward pass: the fused differentiable solver runs a coarse
integration and the trunk predicts a correction plus closure parameters. Far from it — which a
doubled inflow on a turbulent free surface usually is — the model escalates to Tier B: it writes a
real solver configuration, runs it as code, and checks continuity and momentum residuals plus grid
convergence before believing the result. Neither tier is a different model. Tier B is a numerical
program, which is exactly the tool use the constraints permit.

**5 · Render, don't dream** — [`04`](spec/04-generation.md)
The solved velocity and density fields are rasterized by a differentiable renderer. The video is a
*view* of the verified field. Motion vectors are not extracted from the video; the video is derived
from them. This is why the word "accurate" survives contact with the output.

**6 · Report** — [`04`](spec/04-generation.md)
Text decodes from the same trunk that holds the solved field in context — so the report describes
*this* solve, with its actual numbers, and states which tier produced it, what the residuals were,
and how far outside the training distribution the query sat. A model that cannot say "I extrapolated
1.8 Π-units past my data and here's the residual" should not be trusted with the question.

**7 · Allocate** — [`06`](spec/06-runtime.md)
Throughout, the stream holds a recursion budget. Cells in the turbulent shear layer recurse deep;
the report's boilerplate exits shallow. The Tier B solve runs asynchronously while the resident
instance keeps serving other streams — the model is never "busy," only differently weighted.

---

## Status

Specification only. No implementation in this repository. Every number in these documents traces to
[`references.md`](references.md), and every claim that is a bet rather than a finding is labelled as
one.
