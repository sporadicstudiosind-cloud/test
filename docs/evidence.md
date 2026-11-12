# Evidence Register

Each source with what it supports **here** and what it does not establish. A linked paper
supports an individual technique or finding, never an end-to-end Iridium-1 system.

Numbers derived in this repository are marked **recomputed** and are reproduced by a test
rather than transcribed.

## Architecture and generation

| Source | Supports | Does not establish |
|---|---|---|
| [Transfusion](https://arxiv.org/abs/2408.11039) | Joint discrete text and continuous image generation in one transformer | Full audio, video, physics and software agency |
| [Chameleon](https://arxiv.org/abs/2405.09818) | Early-fusion mixed image/text modeling | Universal numerical or physical understanding |
| [Flow Matching](https://arxiv.org/abs/2210.02747) | Continuous generative vector-field training | Guaranteed few-step high-quality output |
| [Universal Transformers](https://arxiv.org/abs/1807.03819) | Recurrent self-attentive computation | Equivalence to untied deep capacity |
| [PonderNet](https://arxiv.org/abs/2107.05407) | Explicit stopping distribution regularized to a geometric prior | Useful adaptive depth at Iridium-1 scale |
| [GATr](https://arxiv.org/abs/2305.18415) | E(3)-equivariant geometric transformer | Automatic Galilean/Lorentz symmetry of arbitrary layers |
| [L-GATr](https://arxiv.org/abs/2405.14806) | Lorentz equivariance for a specified domain | A universal shared physical symmetry |

## Physics

| Source | Supports | Does not establish |
|---|---|---|
| [Poseidon](https://arxiv.org/abs/2405.19101) | Learned PDE operators with evaluated transfer | Unrestricted multiphysics extrapolation |
| [Reynolds generalization study](https://arxiv.org/abs/2605.30112) | One studied cross-regime generalization failure | Impossibility of all native learned dynamics |
| [Multi-physics negative transfer](https://arxiv.org/abs/2605.15179) | Negative transfer in the studied setting | Impossibility of every dense design |
| [The Well](https://arxiv.org/abs/2412.00568) | Diverse physical simulation data | Complete coverage or real-world validation |
| [NASA grid convergence](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html) | Discretization and refinement assessment | That modelled physics matches reality |
| [NASA validation assessment](https://www.grc.nasa.gov/www/wind/valid/tutorial/valassess.html) | Comparing simulation with observation | A universal numerical tolerance |

## Serving

| Source | Supports | Does not establish |
|---|---|---|
| [PagedAttention](https://arxiv.org/abs/2309.06180) | KV-memory management for serving | Adaptive recurrent streaming support |
| [Attention sinks](https://arxiv.org/abs/2309.17453) | A windowed streaming technique | Infinite memory, or safe *shared mutable* sinks |
| [Moshi](https://arxiv.org/abs/2410.00037) | Joint speech/text full-duplex interaction | Iridium-1's all-modality integration |

## Recomputed in this repository

| Quantity | Value | Reproduced by |
|---|---|---|
| Prototype / pilot / flagship transformer parameters | 135,266,304 / 5,771,362,304 / 992,137,445,376 | `test_inventory.py` |
| Flagship KV per token, R=1 / R=4 | 1.375 MiB / 5.125 MiB | `test_inventory.py` |
| Flagship KV per 1,048,576-token stream | 1.375 TiB / 5.125 TiB | `test_inventory.py` |
| GQA cache reduction vs full MHA at flagship shape | 8× | `test_inventory.py` |
| Inlet table (Re, Fr, We at Q and 2Q) | see [arch §11.2](architecture.md#112-inlet-arithmetic-with-its-scope-stated) | `test_inlet_arithmetic.py` |
| Scenario B bulk temperature rise | 5 K | `test_inlet_arithmetic.py` |
| Upwind advection observed order | ≈1 over four refinements | `test_conservation.py` |

## Claims withdrawn from the archived specification

The archived `references.md` labelled several figures **secondary** and then extended scoped
results into broad conclusions. Those are not carried forward as design constants. Three
specific numeric or arithmetic errors made in this project's own earlier analysis are
recorded as D25–D27 in [decisions.md](decisions.md): a corner-block count, a KV total that
should have read 24 TiB, and an unsupported claim about Froude crossing.

Where an exact figure is not needed for an architectural decision, it is omitted rather than
granted false authority. Where a number is load-bearing, it is recomputed by a test.

## Standing caution

No source here, and no result in this repository, establishes that a single dense model
attains general multimodal competence. The measured slice covers one linear PDE family at
~0.8 M parameters. Everything above that is specification and named experiment.
