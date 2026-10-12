# First Slice — Measured Results

A genuinely trained, held-out-tested slice (§20.3), not a random-weight shape test. Every
number below was produced by `experiments/run_first_slice.py` and is recorded in
`experiments/results/first_slice.json`. Reproduce with:

```bash
python3 experiments/run_first_slice.py --steps 4000 --train-episodes 1024 --eval-episodes 128
```

## What was trained

`iridium-1-slice`: **817,705 parameters total**, one shared dense recurrent trunk
(`d_model` 128, 1 prelude / 2 core / 1 coda, GQA 4 query heads over 2 KV heads, recurrence
depth 2) reading one interleaved sequence of:

- a **byte-level instruction** ("advect field for 0.250 s at speed 0.412 m/s with diffusivity
  0.00219 m2/s; report the final field"),
- **field patches** of the exact initial state on a 64-cell periodic mesh,
- **image patches** of a noised 16×64 space-time diagnostic.

Three native heads on the same trunk: a conservative face-transfer head, an unconstrained
direct head (the ablation control), and a conditional-flow-matching image head. No external
model is called; the codecs and every head are in the one checkpoint.

Physics: linear advection–diffusion `u_t + c u_x = ν u_xx`, periodic, with the **exact
spectral solution** as the target, so the learned error is the model's own and not a
reference solver's. 4,000 steps, batch 32, AdamW + OneCycle, 948 s on CPU.

Splits are family-level (§18.2): training speeds `c ∈ [0.2, 0.8]`, an interpolation split
drawn from the same band with a different seed, and an **extrapolation** split at
`c ∈ [1.0, 1.4]`, wholly outside training.

## Result 1 — the learned heads beat the numerical reference in distribution

Normalized RMSE against the exact solution, 128 held-out episodes:

| Predictor | Interpolation | Extrapolation |
|---|---:|---:|
| Conservative flux head | **0.0555** | 1.4787 |
| Direct head (control) | **0.0459** | 1.4768 |
| First-order upwind reference | 0.1128 | 0.2098 |
| Persistence baseline | 1.2613 | 1.6482 |

In distribution the native learned prediction is **about 2× better than the same-task upwind
solver** and ~25× better than persistence. This matters because it is the distinction §6.1
insists on: a system that only ever delegates to a solver has not demonstrated native
physical competence. Here, on this narrow family, it has.

I expected the opposite before running it, and said so in an earlier draft. The measurement
disagreed and the measurement wins.

## Result 2 — and they collapse out of distribution

On unseen wave speeds the learned heads degrade to **1.48**, barely better than persistence
at 1.65, while the upwind solver degrades only from 0.11 to 0.21.

This reproduces, at toy scale and on a linear problem, the cross-regime generalization
failure that motivates the whole escalation design in §6.6. The learned operator is excellent
inside its training band and close to useless outside it; the numerical method does not care.
**This is the single strongest argument in this repository for the joint risk assessment and
for reference-mode escalation**, and it was measured here rather than cited.

## Result 3 — conservation holds on the trained model

Maximum relative mass drift across held-out episodes:

| Head | Interpolation | Extrapolation |
|---|---:|---:|
| Conservative flux head | **9.46e-09** | 9.13e-09 |
| Direct head (control) | 1.83e-02 | 1.94e-02 |
| Exact solution (FP64 control) | 2.71e-16 | 4.17e-16 |

The flux head conserves to float32 machine precision — about **six orders of magnitude**
better than the unconstrained control — and it does so *out of distribution too*, where its
accuracy has collapsed. That separation is the point: conservation is a property of the
update form, not of how well the network has learned.

The control is a fair one (D31): same trunk output, also predicts an increment from `u0`,
also zero-initialized. The only difference is whether the correction lives on shared faces.

Note the direction of Result 1 versus Result 3: the *direct* head is slightly more accurate
(0.046 vs 0.056) while violating mass by ~1.8%. Constraining the update costs a little
pointwise accuracy and buys exact conservation. That trade is worth stating plainly rather
than pretending the constrained head wins on every axis.

## Result 4 — generation steps, measured not promised

Native image head, NRMSE against the exact space-time diagnostic:

| ODE steps | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| NRMSE | 0.2063 | 0.1216 | 0.0981 | **0.0901** | 0.0903 | 0.0930 |

Quality improves to 8 steps and then **plateaus and slightly worsens**. There is no universal
4–10 step guarantee to appeal to (D19); on this task 8 happens to be the knee, and the
slight regression past it is a real property of the learned velocity field worth investigating
rather than smoothing over.

## Result 5 — the reconciliation pass caught its own bug

The generated report is rendered from the metric table and every numeral is checked back
against it. On the first run it produced one finding:

```
[unbound] numeral 9.457e-09 is not bound to a metric  (in: '9.457e-09 for')
```

The value was correct; the *checker* was wrong, in two ways. It treated the following English
word "for" as a unit token, and its rounding tolerance worked in decimal places, so rounding
`9.45711e-09` gave zero. Both are fixed and both now have regression tests
(`test_scientific_notation_rounds_by_significant_figures`,
`test_following_english_word_is_not_treated_as_a_unit`). The rerun reconciles clean.

Worth noting which way it failed: it refused to certify a number it could not verify. A
checker that errs toward flagging is the correct failure direction for this component.

## Scope

This is one linear PDE family, one spatial dimension, periodic boundaries, single-step
prediction, 0.8 M parameters, one seed. It establishes that the contracts, the shared trunk,
the conservative head and the native continuous head work together and can be measured. It
establishes nothing about nonlinear systems, multi-step rollout, real geometry, other
modalities, or any larger configuration.

[`backlog.md`](backlog.md) names the next milestone and the five pieces of evidence that
close it. Gate 3 there — beating a cheap solver on a *nonlinear* system — is the one that
would make "native physical competence" more than a narrow result.
