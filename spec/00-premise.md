# 00 — Premise, Constraints, and the Case Against

## What is being built

One model that natively understands and computes across text, images, video, audio, arbitrary
datapoints, symbolic mathematics, and physical simulation — and that can act, not just answer:
driving software, editing scenes, running solvers, emitting motion.

The ambition is a Jarvis: not an orchestrator that dispatches to specialists, but a single system
whose understanding of a fluid and its understanding of a sentence about that fluid live in the same
representation.

## Hard constraints

These are given, not derived. Where the spec disagrees with one, it says so in the open and proposes
a measurement rather than quietly reinterpreting it.

| # | Constraint | Where honored | Status |
|---|---|---|---|
| C1 | One model. No second model under the hood. | Throughout; codecs are layers ([`01`](01-representation.md)) | Honored, at a stated cost |
| C2 | Not MoE. Dense parameters. | [`02`](02-trunk.md) | Honored, with a falsifiable tripwire |
| C3 | Dynamic compute — the model decides its own focus. | [`02`](02-trunk.md), [`06`](06-runtime.md) | Honored as recursion depth |
| C4 | May call code and APIs; may not call models. | [`03`](03-physics.md), [`05`](05-agency.md) | Honored — solvers are programs |
| C5 | One always-running, always-streaming instance. | [`06`](06-runtime.md) | Honored; one sub-claim rejected |
| C6 | Native physics, not physics-flavored generation. | [`03`](03-physics.md), [`04`](04-generation.md) | The core of the design |
| C7 | Agentic control equivalent to a human operator. | [`05`](05-agency.md) | Weakest subsystem; stated plainly |
| C8 | Recursive self-improvement, optional. | [`09`](09-self-improvement.md) | Scoped to what actually works |
| C9 | Compute cost is not a design constraint. | Throughout | Taken literally |
| C10 | Unrestricted internet access for training data. | [`07`](07-data.md) | Less decisive than it sounds |

### The one sub-claim this spec rejects

C5 includes the idea that the model should be able to *divert attention between concurrent chats* —
one instance whose focus flows across all live requests.

Half of this is already true and trivially adoptable: frontier serving keeps weights resident and
multiplexes requests through continuous batching, so "one instance rather than a new model per
request" is a solved engineering problem, not a research goal.

The other half must not be built. Attention flowing between chats means one user's KV cache is
readable from another user's forward pass. That is not a capability, it is a cross-tenant data leak
with extra steps. What *can* legitimately be global is the compute budget, the scheduler, and a
resident memory the operator owns. See [`06`](06-runtime.md) for the version that keeps the
capability and drops the vulnerability.

## Three findings that shaped everything

The design is not derived from first principles; it is derived from three published failures.

**Learned dynamics do not extrapolate across regimes.** Under a 10× Reynolds-number shift, a trained
Fourier Neural Operator reaches ~46.7% relative L₂ error, and simple retrieval baselines beat it.
The controlling variable is representation geometry — global-spectral (~47%) < global-linear (~42%)
< learned-local (~38%) < local-multiscale (~35%) — and the authors explicitly decline to claim any
method learned Reynolds-invariant physics. Autoregressive drift alone accounts for ~12 percentage
points. *Consequence:* the model must not be asked to imagine unseen regimes. See
[`01`](01-representation.md) and [`03`](03-physics.md).

**Generated video does not conserve anything.** On the hard subset of VideoPhy-2, the best model
(Wan2.1-14B) reaches **21.9%** joint performance — the fraction of clips scoring ≥4/5 on *both*
semantic adherence and physical commonsense. Conservation of momentum and conservation of mass are
each violated at ~40%. Independently, Physion-Eval had STEM-trained annotators inspect 12,718
generated videos and found ≥1 identifiable physics flaw in **83.3%** of exocentric and **93.5%** of
egocentric clips. *Consequence:* the video path must be a renderer, not a generator. See
[`04`](04-generation.md).

**Dense multi-physics co-training fights itself.** Training one dense parameter path on incompatible
regimes — broadband chaotic open-channel flow against stiff, confined porous-media flow — produces
gradient conflict and plasticity collapse, where stiff residuals dominate and effectively low-pass
filter the chaotic features. The published remedy is sparse routing, which C2 forbids; in that work
routing bifurcated perfectly (100% of open-channel tokens to one expert, 100% of porous tokens to
the other). *Consequence:* if the dense constraint is to hold, the regimes must be brought closer
together in representation rather than separated in parameters. See [`01`](01-representation.md) and
the tripwire in [`02`](02-trunk.md).

## The case against this entire architecture

Included because a spec that only argues for itself is marketing. These are the strongest
objections, stated as an opponent would state them.

**A monolith fails illegibly.** When a modular system produces a wrong answer, you can ask *which
component* was wrong, replace it, and re-test. When ORRERY produces a wrong answer, the fluid
representation, the depth controller, the renderer, and the language head are entangled in one set
of weights and one training run. Debugging becomes archaeology. Every mature engineering discipline
has moved toward interface boundaries; this design deliberately removes them.

**"No other models" is aesthetics, not engineering.** C1 is a purity constraint, and it is
expensive. It forbids frozen, well-characterized perceptual codecs — the single most stabilizing
component in modern generative stacks — and forces joint codec training, which is the largest
schedule risk in [`08`](08-training.md). It forbids adopting a better video decoder next year
without retraining the world. A system that called three specialist models would likely be better at
every individual task, ship sooner, and be cheaper to fix.

**The dense constraint has a published counterexample.** C2 is contradicted by direct evidence of
negative transfer in exactly the multi-physics setting this model targets. The spec's answer —
dimensionless representation, so that hostile regimes become neighbors — is a *hypothesis*, not a
result. It may simply fail, and the honest posture is to name in advance the measurement that would
prove it did.

**Π-space is not a general solution.** Nondimensionalization works where the similarity group is
known and complete. Turbulence has no closed similarity solution; multiphase flow with surface
tension, phase change, and non-Newtonian rheology has too many groups to span. The technique buys
range, not invariance.

**Agentic control is nowhere near "exactly like a human."** The best hybrid computer-use agents pass
**41.2%** of long-horizon real-world tasks (114 tasks, 8 domains). A subsystem at 41% cannot be
described as human-equivalent, and no part of this spec makes it better — it merely places it in a
nicer coordinate frame.

### Why build it anyway

Three arguments survive the objections, and they are the only three.

1. **Cross-modal transfer is the whole point.** A shared latent frame is what allows a solved
   velocity field to be rendered, described, and edited without three lossy translations. The
   modular system is better at each task and worse at the composition.
2. **Render-the-field is impossible without it.** The one architectural idea here that genuinely
   defeats a documented failure mode — accurate physics video — requires the field, the renderer,
   and the language model to share a coordinate system. Bolt them together and the physics is a
   caption again.
3. **Latency and duplex.** A resident instance with a shared temporal axis can be interrupted
   mid-generation and can act unprompted. A dispatch graph over specialists cannot, not really.

If those three do not justify the cost to you, build the modular system. It is the better engineering
decision for most goals, and this document will still be useful as a component design.
