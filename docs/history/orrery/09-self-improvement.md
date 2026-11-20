# 09 — Self-Improvement, Bounded

> Solver residuals are a reward you cannot argue with. That is rarer and more valuable than it
> sounds, and it is not a licence for the model to rewrite itself.

## 9.1 What is actually true in 2026

Constraint C8 makes recursive self-improvement optional, which is fortunate, because the honest state
of the field is narrower than the term suggests:

- **No production system rewrites its own weights.** Every functioning self-improvement loop modifies
  prompts, code, tools, and data around fixed parameters.
- **Self-correction without an external signal largely does not work.** Absent grounding, models do
  not reliably identify their own reasoning errors, and essentially every serious 2026 system anchors
  its critique in an external check.
- **Real results exist and are bounded.** Large fractions of production code written by models, and
  research loops closing most of a benchmark gap, are genuine. They are also narrow and measured, and
  they do not indicate a runaway.

Any design that assumes otherwise is designing against a capability that does not exist.

## 9.2 Why this system is an unusually good case anyway

A text model improving itself faces a circularity: the thing judging the output is the thing that
produced it. Preference models can be gamed, LLM judges can be flattered, and a model that has
learned to write convincing answers has learned to write convincing wrong answers.

A physics model has an exit from that loop. **Continuity and momentum residuals are not opinions.**
A field either satisfies the discretized conservation laws to tolerance or it does not, and no amount
of fluency changes the number. Grid convergence is similarly non-negotiable: refine three times and
the observed order of accuracy either matches the scheme or it does not.

This is the strongest argument in the entire document for pursuing self-improvement here rather than
elsewhere. The reward is external, cheap to compute, and structurally immune to persuasion.

## 9.3 The loop

```
1  PROPOSE   model generates a scenario, targeting Π-space regions where its own
             Tier A surrogate disagrees most with Tier B  ([03], [07])
2  PREDICT   Tier A answers — fast, cheap, possibly wrong
3  SOLVE     Tier B runs the verified numerical solve, subject to every gate in [03] §3.3
4  SCORE     disagreement between prediction and verified solve, weighted by solve confidence
5  MINT      the verified solve becomes a new training example, stamped with full provenance
6  RETRAIN   offline, on accumulated failures, behind an evaluation gate  (§9.5)
```

Step 1 is the design decision that makes this more than data augmentation. Proposing scenarios the
model already handles well generates volume and no information. **Proposals are scored by expected
surrogate-versus-solver disagreement** — the loop deliberately hunts for its own blind spots, and a
scenario is interesting precisely to the degree the model expects to be wrong about it.

## 9.4 What can go wrong, and what to do about it

**Distribution collapse.** The most likely failure is not runaway improvement but quiet narrowing:
the proposer converges on a comfortable family of scenarios and the model gets better and better at
less and less. Countermeasures: an explicit novelty term over Π-space coverage; a fixed fraction of
proposals drawn from an external scenario bank rather than generated; and monitoring of proposal
diversity as a first-class metric, treated as a stop condition when it falls.

**Solver error laundered into training data.** Tier B is authoritative only within its own validity.
A badly meshed case that passes residual checks but is under-resolved will mint confidently wrong
ground truth, and the model will learn it as fact. Mitigation: grid convergence is mandatory for
minted data, not optional; minted examples carry solver configuration in their provenance so a later
discovered systematic error can be traced and the affected shards removed.

**Reward hacking on the residual.** The residual is hard to argue with but not impossible to game —
a model could learn to propose scenarios that are trivially conservative, or to bias solver setups
toward configurations that converge easily. Countermeasure: proposal difficulty and solver setup are
scored independently, and a held-out set of human-authored scenarios is used to verify that measured
improvement transfers off the self-generated distribution.

**The real-measurement gap.** A solver farm can be internally consistent and collectively wrong; a
turbulence closure that is systematically off produces perfectly self-consistent data forever. This
is why [`07`](07-data.md) §7.5 treats published experiments as the loop's most valuable external
input. Reproducing a real measurement is the only step in this cycle that can detect a shared
delusion.

## 9.5 Hard boundaries

These are not adjustable by the loop, and they are the reason this section exists in a document that
otherwise permits a great deal.

1. **No autonomous weight updates in production.** Retraining is offline, evaluated, and deployed at
   a deliberate boundary with rollback. Hot-swapping self-derived weights into a live multi-tenant
   instance combines an unvalidated capability with an unbounded blast radius, and it is not a
   throughput optimization worth the exposure.
2. **The evaluation gate is not self-authored.** A model that proposes its own eval and passes it has
   demonstrated nothing. Gates come from [`11`](11-evaluation.md), including held-out Π-regions and
   real-measurement benchmarks the loop cannot mint.
3. **No self-modification of the boundaries themselves** — the escalation policy in
   [`03`](03-physics.md), the verification gates, the action limits in [`05`](05-agency.md), and this
   list. A loop permitted to relax its own verification will, because relaxing verification improves
   every metric it can see.
4. **Provenance is permanent.** Self-generated training data is labelled as such forever. When a
   later problem is traced to a systematic solver error, the affected data must be findable.
5. **The loop is stoppable and its outputs are inspectable.** An operator can halt it and read what
   it proposed, solved, and minted. A self-improvement process that cannot be audited is not a
   self-improvement process, it is an unmonitored training run.

## 9.6 Honest expectation

What this buys: a surrogate that steadily improves in the regions where it was weakest, with
measurable coverage growth in Π-space, without human labelling. That is genuinely valuable and it is
the realistic ceiling.

What it does not buy: architectural self-modification, autonomous capability jumps, or any escape
from the extrapolation limits in [`01`](01-representation.md) §1.4. The loop expands the training
manifold. It does not change what happens outside it — and outside it, the answer is still
[`03`](03-physics.md)'s escalation to a real solve.
