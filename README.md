# Iridium-1

**One dense multimodal foundation model** that perceives, reasons, generates, simulates and
acts across text, code, mathematics, images, video, speech, sound, structured measurements,
geometry and physical fields — served as one persistent instance that allocates its own
computation across concurrent work.

No mixture of experts. No external neural models: solvers, renderers, proof checkers,
compilers and software APIs are permitted programs, and every learned component belongs to
one versioned Iridium-1 checkpoint.

> **Status: specification, plus a partial reference implementation and one trained
> correctness slice.** Nothing here demonstrates a general system. What is trained is ~0.8 M
> parameters on one linear PDE family. [`docs/capability-register.md`](docs/capability-register.md)
> states, per capability, whether it is specified, implemented, trained, evaluated or
> unsupported — read it before quoting anything from this repository as a result.

---

## Start here

| Document | What it is |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | The canonical architecture. Everything normative. |
| [`docs/requirements-traceability.md`](docs/requirements-traceability.md) | The original request → requirement → specification → test |
| [`docs/decisions.md`](docs/decisions.md) | Every correction made, with the test that holds it |
| [`docs/capability-register.md`](docs/capability-register.md) | Honest status per capability |
| [`docs/first-slice.md`](docs/first-slice.md) | Measured results from the trained slice |
| [`docs/backlog.md`](docs/backlog.md) | M0–M11 with acceptance evidence, and the next milestone |
| [`docs/evidence.md`](docs/evidence.md) | Sources, with what each does *not* establish |
| [`docs/scenarios.md`](docs/scenarios.md) | Capability targets A–G beyond the waterfall |
| [`docs/history/orrery/`](docs/history/orrery/) | The superseded ORRERY specification, preserved unaltered |

## Run it

```bash
pip install numpy torch pytest jsonschema
python3 -m pytest                                     # 136 tests, all passing
python3 experiments/run_first_slice.py --steps 4000  # trains, evaluates, reconciles
python3 -m iridium.model.inventory                 # parameter and cache accounting
```

## Layout

```
docs/          canonical architecture, decisions, evidence, backlog, history
schemas/       event, action, result manifest, coupling interface (JSON Schema, enforced)
iridium/
  contracts/   units, typed frames, events + exact array store, metric reconciliation
  model/       dense recurrent core, stopping-time semantics, parameter inventory
  physics/     conservative finite-volume updates, open-system budgets
  data/        episode generation
  training/    the first slice
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
