# 02 — The Trunk: Dense Weights, Variable Depth

> One parameter set. No experts. Tokens differ in how many times they pass through it.

## 2.1 Shape

A single dense stack of `L` layers, organized as a **recursive core**: a block of `k` layers that a
token may traverse repeatedly, up to `R_max` recursions. Effective depth for a token is `k × r`,
where `r` is chosen per token, per stream, at inference time.

```
prelude   (few layers, always executed)   — codec output → shared latent space
core      (k layers, recurrent 1..R_max)  — the dense body
coda      (few layers, always executed)   — → decode heads
```

Attention is full and early-fused across all modalities. There is no per-modality block, no adapter,
no cross-attention bridge — the ideas from [`01`](01-representation.md) are load-bearing here: the
only reason a fluid cell and a sentence can share a stack is that they already share a frame.

### Why recursion rather than more layers

Under C9 (ignore compute cost) the obvious move is simply a deeper stack. Recursion is chosen
instead for one reason: it makes compute allocation *continuous and controllable at runtime*, which
is the mechanism C3 requires. A 400-layer fixed stack spends the same compute on a turbulent shear
layer and on the word "the." A 40-layer core traversed 1–10 times does not.

Parameter sharing across recursions also means the depth axis is not new capacity to be trained — it
is the same capacity applied longer. This is what keeps C2 intact: there is exactly one set of
weights, and no token has access to parameters another token cannot reach. That is the precise sense
in which this is not MoE.

## 2.2 Focus is recursion depth

This is the spec's answer to "the model should decide how much of its attention to divert to each
task."

**Per-token.** A lightweight controller reads the current hidden state and emits a continue/exit
decision. Tokens in a chaotic region, an ambiguous clause, or a high-residual patch of a field keep
recursing; tokens in a boilerplate preamble exit after one pass.

Formally, at recursion `k` a halting unit emits `p⁽ᵏ⁾ = σ(w_halt·h⁽ᵏ⁾ + b)`; the output is the
probability-weighted mixture of intermediate states, with residual mass absorbed at `R_max`. The
ponder cost is **KL divergence from the halting distribution to a geometric prior**, not a raw sum of
halting steps. This distinction is load-bearing: the naive adaptive-computation-time penalty is
notoriously unstable and tends to collapse to always-halt-immediately or always-run-to-max, while the
geometric-prior formulation gives a well-behaved target distribution with a tunable expected depth.

**Per-stream.** Each live request carries a **depth budget** — total recursion-steps per unit time.
The model emits a requested budget per stream from its own assessment of task difficulty; the
scheduler in [`06`](06-runtime.md) reconciles requests across streams against the hardware and hands
back an allocation. The model proposes, the scheduler disposes; without that split, one stream can
starve every other by declaring itself important.

**Why this is the only honest reading of "focus."** Under strictly uniform depth, "more focus" could
only mean *more tokens* — longer internal deliberation — which is real but is a different mechanism
and cannot be targeted at a *region* of a field. Variable depth is what allows the model to think
harder about the shear layer specifically.

## 2.3 The routing problem nobody advertises

Depth routing has a train/serve mismatch that must be designed around, not discovered in production.

Top-k selection over a batch is **non-causal**: deciding whether token *i* recurses requires knowing
the other tokens competing for the same budget, which at autoregressive inference time do not exist
yet. Training with batch-wise top-k and serving with per-token thresholds produces a silent
distribution shift.

**Design response:**

1. Train the controller as a **causal per-token predictor** with a soft gate on the residual update,
   supervised against the batch-wise top-k decision as a teacher signal. The gate is differentiable;
   the hard decision at serve time is a threshold on it.
2. Enforce the budget with a **global expected-depth penalty** rather than a hard per-batch cap, so
   the total is predictable in aggregate while individual decisions stay local and causal.
3. Ship a **train/serve consistency test** as a first-class eval: for a held-out batch, measure the
   distribution of per-token depths under teacher-forced batch routing versus causal serve routing.
   Divergence above threshold is a release blocker.

## 2.4 The dense constraint, and the tripwire that makes it falsifiable

C2 forbids expert routing. The literature contains a direct counterexample: dense co-training on
incompatible physics regimes produces gradient conflict and plasticity collapse, and sparse routing
resolves it so cleanly that routing bifurcates perfectly by domain.

This spec holds the constraint anyway, on the hypothesis from [`01`](01-representation.md) that the
conflict is substantially an artifact of comparing regimes in raw units, and that Π-space brings them
close enough for one parameter path to serve both.

**That hypothesis may be wrong.** The following instrumentation runs continuously during
multi-physics training, and the thresholds are the conditions under which the dense bet is declared
lost:

| Metric | What it detects | Kill threshold |
|---|---|---|
| Pairwise gradient cosine similarity between physics domains, per layer | Direct gradient conflict | Sustained median < −0.1 across the core for > 10% of training |
| Gradient magnitude ratio between domains | Stiff residuals dominating soft ones | Sustained > 100× |
| Effective rank of core activations | Plasticity collapse / representational narrowing | Monotone decline > 30% from peak while loss is flat |
| High-frequency spectral energy in chaotic-domain reconstructions | The specific documented failure: stiff physics low-pass-filtering chaotic features | > 40% loss of energy above the mesh Nyquist/4 relative to a single-domain control |
| Per-domain held-out loss vs. single-domain control models | The bottom line: is co-training worse than not co-training? | Any domain > 15% worse than its control |

**If the tripwire fires**, the escalation ladder is, in order: (1) gradient surgery — project
conflicting components out, PCGrad/CAGrad style; (2) per-domain loss balancing with uncertainty
weighting; (3) curriculum separation, interleaving domains at longer period; (4) widen the core, since
C9 permits it, on the theory that conflict is partly a capacity problem; and only then (5) report to
the operator that C2 is not survivable for this domain set, with the evidence.

Step 5 is a real possible outcome and this document does not pretend otherwise. What it refuses to do
is discover the failure implicitly, six months in, as "the model is oddly bad at porous media."

## 2.5 Attention over heterogeneous latents

Full attention across a fluid mesh, a video, and a document is not affordable even under C9 — not for
cost reasons but because dense attention over 10⁷ field cells is numerically and practically
unworkable. The trunk uses **structured attention biases derived from the representation**, not
arbitrary sparsity patterns:

- **Mesh-local:** field cells attend within a topological neighborhood, plus a learned set of global
  summary latents per region. Locality comes from the mesh, which is physically correct — information
  in a hyperbolic PDE propagates at finite speed.
- **Coordinate-proximate:** cross-modal attention is biased by distance in the shared `(t,x,y,z,λ)`
  frame, so a video patch preferentially attends to the field cells it actually observes.
- **Global:** text, `meta`, and boundary-condition latents attend and are attended to globally. They
  are few and they matter everywhere.

This is a *representation-derived* sparsity — it follows from physics rather than from a compute
budget — and it does not create separate parameter paths, so C2 is untouched.

Two further mechanisms are normative, both concerning the always-on case:

**Attention sinks.** A small number of permanent, non-evicting anchor tokens sit at the head of the
persistent context and absorb surplus softmax mass. Without them, streaming attention degrades as
context grows because the softmax denominator has nowhere to put probability when no token is
genuinely relevant. For an instance that never restarts, this is not optional.

**Block-diagonal stream masking.** Concurrent streams occupy the same batch and the same weights,
with `−∞` on all off-diagonal stream blocks. Sinks and operator-owned global latents are attended by
everyone; nothing else crosses. This is how [`06`](06-runtime.md)'s security requirement is *enforced*
rather than promised — cross-stream attention is not policy, it is a masked-out region of the matrix.

### The KV-cache term, which dominates and is easy to get wrong

Full multi-head attention at large `d_model` produces a per-token KV footprint that scales with
`2 × d_model × n_layers`. At `d_model ≈ 32k` and ~200 layers with no head sharing, that is on the
order of **24 MiB per token** — roughly 75× a contemporary grouped-query model — which makes a
million-token persistent context cost tens of terabytes per stream and turns a "non-evicting
persistent cache" into an arithmetic impossibility at any useful concurrency.

**Grouped-query or multi-head-latent attention is therefore a requirement, not an optimization.** It
reduces the term by one to two orders of magnitude and is orthogonal to C2 — sharing KV heads is not
expert routing, and every token still traverses every parameter. Any design document that specifies
interconnect bandwidth precisely while leaving this term uncomputed has skipped its dominant memory
cost.

## 2.6 What this section commits to

- One dense parameter set; depth, not identity, varies per token.
- "Focus" has a unit: recursion-steps.
- The dense constraint is held as a *hypothesis with a named falsification*, not as an article of
  faith.
- The known non-causality flaw in depth routing is handled explicitly, with a release-blocking test.
