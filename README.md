# Iridium-1

**One dense multimodal foundation model** that perceives, reasons, generates, simulates and
acts across text, code, mathematics, images, video, speech, sound, structured measurements,
geometry and physical fields — served as one persistent instance that allocates its own
computation across concurrent work.

One **64-layer control core** that every token passes through, dispatching to a bank of
deep **superstacks** and deciding for itself how much depth and how many passes a token
gets. No external neural models: solvers, renderers, proof checkers, compilers and software
APIs are permitted programs, and every learned component belongs to one versioned
Iridium-1 checkpoint.

> **Supersedes the dense-only requirement.** The earlier specification forbade any router
> (IR 02). The current design is explicitly routed: a control core plus conditional
> superstacks. `docs/verification.md` F-11 to F-13 record what that buys and what it costs,
> and `test_capacity_dropping_is_refused_as_non_causal` marks the one form of routing that
> is still refused — because ranking tokens against each other inside a chunk is not
> executable at sampling time.

> **Status: a working implementation of the routed architecture at small scale, plus a
> costed ladder up to 9 T parameters.** The control core, macro-router, superstack bank,
> omnimodal codecs, physics solvers, agentic environment, sandbox, persistent runtime,
> quantizer and parallelism planner are built and tested (375 tests). Nothing here
> demonstrates a general system: the trained rung is 34 M parameters on a synthetic corpus.
> [`docs/build.md`](docs/build.md) says what to run,
> [`docs/verification.md`](docs/verification.md) lists the twenty findings against the
> source plan, and [`docs/capability-register.md`](docs/capability-register.md) states, per
> capability, whether it is specified, implemented, trained, evaluated or unsupported —
> read it before quoting anything here as a result.

---

## Start here

| Document | What it is |
|---|---|
| [`docs/build.md`](docs/build.md) | **What was built, how to run it, what it does and does not establish** |
| [`docs/verification.md`](docs/verification.md) | Every claim in the source plan that was checked, and the result |
| [`docs/architecture.md`](docs/architecture.md) | The canonical architecture. Everything normative. |
| [`docs/requirements-traceability.md`](docs/requirements-traceability.md) | The original request → requirement → specification → test |
| [`docs/decisions.md`](docs/decisions.md) | Every correction made, with the test that holds it |
| [`docs/capability-register.md`](docs/capability-register.md) | Honest status per capability |
| [`docs/first-slice.md`](docs/first-slice.md) | Measured results from the trained slice |
| [`docs/backlog.md`](docs/backlog.md) | M0–M11 with acceptance evidence, and the next milestone |
| [`docs/evidence.md`](docs/evidence.md) | Sources, with what each does *not* establish |
| [`docs/scenarios.md`](docs/scenarios.md) | Capability targets A–G beyond the waterfall |
| [`docs/history/orrery/`](docs/history/orrery/) | The superseded ORRERY specification, preserved unaltered |
| [`notebooks/`](notebooks/) | **Train it yourself on free hardware** — Colab, Kaggle or any Jupyter host, at any rung from 50 M to 1 T |

## Run it

```bash
pip install numpy torch pytest jsonschema
python3 -m pytest                                # 375 tests, all passing
python3 -m iridium ladder                        # tiny -> nano -> ... -> 9 T
python3 -m iridium report nano --verify          # accounting, checked against the modules
python3 -m iridium plan base --gpus 1024         # 4-D parallelism and its cost model
python3 -m iridium waterfall --q 3 --factor 2    # the originating question, answered
python3 -m iridium.training.phase1_pretrain --rung nano --steps 1800
python3 -m iridium.config_builder                # the 50 M -> 1 T preset ladder
```

[`docs/build.md`](docs/build.md) has the rest, and
[`notebooks/README.md`](notebooks/README.md) runs the whole thing — design, fit
check, licensed data, train, grade, chat — on free hardware.

## Layout

```
docs/          canonical architecture, decisions, evidence, backlog, history
schemas/       event, action, result manifest, coupling interface (JSON Schema, enforced)
iridium/
  contracts/   units, typed frames, events + exact array store, metric reconciliation
  config.py    the scaling ladder; executable parameter and memory accounting
  model/       control core, macro-router, superstacks, spectral ops, heads
  codecs/      typed spans and the unified omnimodal embedding bank
  physics/     spectral Navier-Stokes, shallow water, operators, dual-system verifier
  agency/      typed action space, deterministic scene editor, Blender emission
  runtime/     sandbox, per-stream registry, focus scheduler, persistent service
  training/    four phases, verifiable tasks, continual-learning flywheel
  quant/       MXFP4 and FP8, measured
  parallel/    4-D partitioning and the communication cost model
  evaluation/  graded accuracy, routing mutual information
tests/         unit / integration / scientific
experiments/   runnable experiments and their recorded results
configs/       prototype, pilot, flagship
```

## Three ideas the design turns on

**Exact arrays beneath lossy learned representations.** Compressed latents are for reasoning;
the authoritative numbers stay in typed binary storage and survive round-trip bit-exactly. A
hidden vector is never the only record of a scientific result.

**Conservation by construction, not by hope.** A learned correction applied per cell can
invent mass. The same correction applied to *oriented shared faces* cannot, because each face
enters two cells with opposite sign and the interior telescopes. Both are demonstrated in
`tests/scientific/test_conservation.py`, and the trained slice reproduces it with a learned
correction at 9.5e-09 relative drift while its unconstrained control drifts 1.8%.

**Evidence is earned, not labelled.** Illustrative, learned estimate, numerically verified and
empirically validated are four different claims. The result manifest schema refuses a
`numerically_verified` artifact without a verification record, and every numeral in a report
must resolve to an exact metric table — a check whose regression fixture is a real
contradiction that human review missed.

## What this deliberately does not claim

- That it scales. One trained slice at 0.8 M parameters says nothing about the ~992 B
  flagship configuration.
- That native physics works in general. The trained flux head does beat a first-order upwind
  solver in distribution (0.056 vs 0.113 NRMSE) — but it **collapses out of distribution**
  (1.48, barely better than persistence, while the solver degrades only to 0.21). Both numbers
  are in [`docs/first-slice.md`](docs/first-slice.md); the second is why escalation exists.
- That attention can flow between concurrent users. That one reading of "one instance" is
  rejected as a cross-tenant read; everything else about persistence is kept.
- That "understands everything" has been decomposed into anything finite. It has no
  acceptance test, so the backlog replaces it with expanding measured competence.

Prior errors — including three arithmetic and scope errors made in this project's own earlier
analysis — are recorded in [`docs/decisions.md`](docs/decisions.md) D25–D28 rather than
quietly fixed.
