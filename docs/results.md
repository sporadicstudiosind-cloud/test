# Measured results — `nano`, phase 1

One run. 34,028,056 parameters, 800 steps, batch 8, 8,000 synthetic items, CPU,
755 seconds. Checkpoint and manifest in `runs/phase1/`, log in `runs/phase1b.log`.

**Headline: the architecture trains, and the model did not learn the tasks.**
Four of five families sit at or below a baseline that ignores the input
entirely. This page reports that, because a build whose only published numbers
are the ones that came out well is not a measurement.

## Training loss

| | step 0 | step 799 |
|---|---|---|
| total | 9.0160 | 1.7236 |
| text (bytes) | 6.0063 | 0.2859 |
| field | 2.5023 | 1.2509 |
| action opcode | 0.0000 | 0.0007 |
| slot type | 2.4436 | 0.0165 |

Router health throughout: stack-usage entropy **1.383** against a maximum of
`log 4 = 1.386` — no collapse, and no specialisation either (phase 2 is what
would produce that). Mean focus 0.80, mean executed depth 3.97 of 8.

## Graded accuracy, free-running generation

Baseline for the numeric families is the corpus median answer — the best score
obtainable by ignoring the prompt. Baseline for `field_rollout` is persistence:
emit the input frame unchanged.

| family | n | trained | baseline | verdict |
|---|---|---|---|---|
| channel_depth | 16 | 0.063 | 0.125 | **below baseline** |
| channel_intervention | 16 | 0.188 | 0.438 | **below baseline** |
| false_premise | 16 | 0.063 | 0.000 | above a zero baseline; not meaningful at n=16 |
| field_rollout (NRMSE) | 16 | 4.871 | 0.165 | **30x worse than persistence** |
| scene_goal (goal satisfied) | 16 | 0.000 | 0.000 | no better |
| scene_goal (first opcode) | 16 | **1.000** | 0.000 | **learned** |

Extrapolation split (discharges outside the training band) is worse across the
board, as expected when the interpolation split has not been learned:
`channel_depth` 0.000, `channel_intervention` 0.063, `false_premise` 0.000.

## What actually happened

**One head learned cleanly.** First-opcode accuracy went 0.000 → 1.000. That is
a 24-way discrete classification conditioned on a text spec and a rendered
image, so the codec path, the router, the superstacks and the action head are
all carrying signal end to end. The scene goal still fails because satisfying it
needs the *operands* to land within 1e-4, and those come from an unweighted
regression term the run barely trained.

**The numeric families did not.** Byte cross-entropy fell to 0.29 nats, which is
real fitting — but a graded answer needs six consecutive bytes exactly right.
At ~0.29 nats/byte the per-byte error rate is high enough that compounding over
six positions leaves almost nothing correct, while the median-answer baseline
scores 12–44% because the tolerance is 2% and the corpus distribution is narrow.
Low teacher-forced loss and zero graded accuracy is the expected shape at this
budget, not a contradiction.

**The field head got worse, not better.** NRMSE 3.72 untrained → 4.87 trained,
against persistence at 0.165. The flow-matching head is being scored on a single
emitted patch against the mean target patch after 8 integration steps, and 800
steps is not enough to learn a velocity field over a 256-dimensional patch
space. Persistence is a strong baseline here and the model is nowhere near it.

## What this does and does not establish

Establishes: the full path — eight modalities in, routed through core and
superstacks, ponder loop, five heads, free-running generation, graded against
independent computation — runs, trains, and moves at least one head from chance
to perfect.

Does not establish: any claim about physical competence, numeric reasoning,
anti-sycophancy or agentic capability. Those are the things the tasks were built
to measure, and the measurement says no.

## What the next run needs

In rough order of expected effect, none of it done here:

1. **More steps.** 800 at 34 M is roughly two orders of magnitude short. The
   loss was still falling.
2. **A digit-level objective for the numeric families.** Byte cross-entropy is
   the wrong loss for an answer graded to 2% relative — a per-position digit
   loss, or emitting the number through a continuous head, aligns the objective
   with the grader.
3. **Phase 2, then phase 3.** Specialisation (`I(family; stack)` is 0.049 nats of
   1.386 — essentially zero) and then RLVR on exactly these graded rewards.
   Phase 3 optimizes the metric this page reports; phase 1 optimizes a proxy.
4. **A field objective matched to the grader** — score the emitted field against
   the whole target, not one patch against a mean.

The numbers above are the pre-phase-2 baseline those runs would have to beat.
