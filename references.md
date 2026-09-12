# References

Every quantitative claim in this specification traces to an entry here. Each is marked with how it
was verified: **[fetched]** means the source document was retrieved and the figure read from it;
**[secondary]** means the figure was confirmed from a source citing the paper rather than the paper
itself, and should be re-checked before being relied on in a decision.

---

## Physics extrapolation and neural PDE surrogates

**Striding Across Reynolds Numbers: Representation Geometry in Neural PDE Generalisation** —
arXiv:2605.30112 · **[fetched]**
<https://arxiv.org/html/2605.30112v1>

Used for: FNO reaches **46.68%** relative L₂ error under a 10× Reynolds shift, beaten by retrieval
baselines. Error clusters by representation geometry — global spectral (FNO) ~47%, global linear
(PCA) ~42%, learned local (ConvAE) ~38%, local multi-scale (U-Net) ~35%. Autoregressive drift is the
**primary bottleneck at ~12 percentage points**. The authors explicitly state they "do not claim that
Reynolds-invariant physics has been learned by any tested method." ConvAE-Relay's advantage reverses
at 100× shift as the source-regime database stops indexing the targets.

Cited in: [`00`](spec/00-premise.md), [`01`](spec/01-representation.md), [`03`](spec/03-physics.md),
[`10`](spec/10-open-problems.md), [`12`](spec/12-blueprint-reconciliation.md)

---

**Eradicating Negative Transfer in Multi-Physics Foundation Models via Sparse Mixture-of-Experts
Routing** — arXiv:2605.15179 · **[fetched]**
<https://arxiv.org/html/2605.15179v1>

Used for: dense co-training across disparate PDE regimes induces "gradient conflict, unstable
optimization, and plasticity loss." Mechanism: task gradients point in conflicting directions or
differ drastically in magnitude; stiff residuals dominate smoother dynamics, "effectively low-pass
filtering chaotic features." Open-channel (broadband, chaotic) versus porous-media (stiff, confined)
impose "incompatible spectral and geometric demands on a single dense parameter path." Routing
bifurcates completely — 100% of open-channel tokens to one expert, 100% of porous-media tokens to the
other — with latent MSEs of 2.46×10⁻⁵ and 9.76×10⁻⁶. Notably, the published design routes shared
structure through **shared** experts.

Cited in: [`00`](spec/00-premise.md), [`02`](spec/02-trunk.md), [`10`](spec/10-open-problems.md),
[`12`](spec/12-blueprint-reconciliation.md)

---

**FNO limitations — boundary conditions and spectral bias** · **[secondary]**
<https://www.physicsx.ai/newsroom/how-a-fourier-neural-operator-learns-to-solve-pdes----and-where-it-falls-short>
· <https://neuraloperator.github.io/dev/theory_guide/fno.html>
· *Render unto Numerics: Orthogonal Polynomial Neural Operator for PDEs with Non-periodic Boundary
Conditions*, arXiv:2206.12698

Used for: FNO is not strictly restricted to periodic domains — the pointwise linear term recovers
some non-periodic behavior — but "performance of the FNO drops once the assumption of periodicity is
not satisfied," and applying spectral methods to non-periodic data produces Gibbs oscillations.
Documented spectral bias toward low frequencies; standard practice truncates high modes.

Cited in: [`03`](spec/03-physics.md), [`12`](spec/12-blueprint-reconciliation.md)

---

## Generated video and physical law

**VideoPhy-2: A Challenging Action-Centric Physical Commonsense Evaluation in Video Generation** —
arXiv:2503.06800 (ICLR 2026) · **[fetched]**
<https://arxiv.org/html/2503.06800>

Used for: best model (Wan2.1-14B) reaches **21.9% joint performance** on the hard subset, where joint
performance is "the fraction of videos that both adhere closely to the text prompt (SA≥4) and follow
physical commonsense to a high degree (PC≥4)" on a 5-point scale. Conservation of momentum and
conservation of mass are each violated at ~40%; reflection and buoyancy violate below 20%.

Cited in: [`00`](spec/00-premise.md), [`04`](spec/04-generation.md),
[`12`](spec/12-blueprint-reconciliation.md)

---

**Physion-Eval: Evaluating Physical Realism in Generated Video via Human Reasoning** —
arXiv:2603.19607 · **[secondary]**
<https://arxiv.org/abs/2603.19607>

Used for: STEM-trained expert annotators inspecting generated video found ≥1 identifiable physics
flaw in **83.3%** of third-person (exocentric) and **93.5%** of first-person (egocentric) clips.
12,718 generated videos from five state-of-the-art models; 10,990 expert reasoning traces across 22
fine-grained physical categories.

Note: an earlier search attributed these figures to a different paper. They belong to Physion-Eval.
Corrected before inclusion.

Cited in: [`00`](spec/00-premise.md), [`04`](spec/04-generation.md),
[`12`](spec/12-blueprint-reconciliation.md)

---

**Benchmark limitations** · **[secondary]** — from the survey framing in the Physics-IQ-verified and
VideoPhy-2 literature

Used for: benchmarks "rely on implicit prompts that describe a scene without specifying its expected
physical outcome, allowing evaluators to only judge whether the video looks plausible"; Physics-IQ
"does not assess whether conserved quantities such as energy or momentum are preserved"; aggregate
scoring "collapses heterogeneous failure modes and prevents per-law diagnostics."

Cited in: [`10`](spec/10-open-problems.md), [`11`](spec/11-evaluation.md)

---

## Architecture

**Geometric Algebra Transformer** — arXiv:2305.18415 · **[secondary]**
<https://arxiv.org/pdf/2305.18415>

**Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics** — arXiv:2405.14806
(NeurIPS 2024) · **[secondary]** · <https://arxiv.org/html/2405.14806v1>

Used for: GATr represents states in the geometric algebra G(3,0,1) with grade-sensitive
normalization, equivariant attention, and multivector interactions; is E(3)-equivariant; outperforms
non-geometric baselines on n-body modelling and robotic planning. L-GATr uses a spacetime algebra
with Lorentz equivariance and is on par with or better than domain-specific baselines. Clifford-
Steerable CNNs show gains on fluid dynamics and relativistic electrodynamics forecasting.

Cited in: [`01`](spec/01-representation.md), [`12`](spec/12-blueprint-reconciliation.md)

---

**Does equivariance matter at scale?** — arXiv:2410.23179 · **[secondary]**
<https://arxiv.org/abs/2410.23179>

Used for: equivariance improves data efficiency, and non-equivariant models with augmentation can
close that gap given sufficient epochs — but scaling with compute follows a power law with
**equivariant models outperforming non-equivariant ones at each tested compute budget**. Included
specifically as the counterweight to the "equivariance is unnecessary at scale" position, which is
live and not settled.

Cited in: [`01`](spec/01-representation.md), [`12`](spec/12-blueprint-reconciliation.md)

---

**PonderNet: Learning to Ponder** — arXiv:2107.05407 · **[secondary]**
<https://arxiv.org/abs/2107.05407>

Used for: learns an explicit distribution over stopping times and regularizes it toward a **geometric
prior via a KL term**, making the compute/accuracy trade-off more stable than Adaptive Computation
Time and yielding a usable halting rule at deployment. Fully differentiable with unbiased,
low-variance gradient estimates.

Cited in: [`02`](spec/02-trunk.md), [`08`](spec/08-training.md),
[`12`](spec/12-blueprint-reconciliation.md)

---

**Mixture-of-Depths: Dynamically allocating compute in transformer-based language models** —
arXiv:2404.02258 · **[secondary]** · <https://arxiv.org/abs/2404.02258>

**Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation** —
arXiv:2507.10524 · **[secondary]** · <https://arxiv.org/html/2507.10524v1>

Used for: compute expenditure predictable in total but dynamic and context-sensitive per token;
tokens either receive computation or pass through a residual connection; matches baseline
performance at equivalent FLOPs and is up to 50% faster at sampling. MoR applies recursive blocks a
learned number of times per token. Also the source of the non-causality problem: top-k routing
requires knowing the batch, which autoregressive inference does not.

Cited in: [`02`](spec/02-trunk.md)

---

**Efficient Streaming Language Models with Attention Sinks** — arXiv:2309.17453 (ICLR 2024) ·
**[secondary]** · <https://arxiv.org/abs/2309.17453>

Used for: initial tokens act as attention sinks absorbing surplus softmax mass even when semantically
unimportant; retaining their KV recovers window-attention performance and enables stable modelling up
to 4M tokens without fine-tuning.

Cited in: [`02`](spec/02-trunk.md), [`06`](spec/06-runtime.md)

---

**Toward Native Multimodal Modeling: A Roadmap** — arXiv:2605.25343 · **[fetched]**
<https://arxiv.org/pdf/2605.25343>

Used for: native multimodal modelling — shared tokenization and transformer layers — outperforms
adapter-based approaches; **early fusion over late fusion**; the discrete-versus-continuous
tokenization tradeoff, with hybrids (discrete for understanding, continuous for generation) as the
emerging middle ground; open problems in architectural convergence, modality imbalance, and
evaluation gaps. Also the shift toward streaming decoding, duplex concurrency, and resource-adaptive
serving.

Cited in: [`01`](spec/01-representation.md), [`04`](spec/04-generation.md)

---

## Runtime and serving

**Continuous batching, paged KV, chunked prefill** · **[secondary]**
<https://huggingface.co/blog/continuous_batching> ·
<https://www.spheron.network/blog/llm-serving-optimization-continuous-batching-paged-attention/>

Used for: iteration-level scheduling admits and retires requests each decode step; 4–8× throughput
versus static batching; one batch simultaneously holds prefills and decodes. Priority scheduling:
dual-heap arrangements jointly optimizing scheduling and cache retention prevent priority inversions
between prefill and decode.

Cited in: [`06`](spec/06-runtime.md)

---

**Full-duplex streaming multimodal systems** · **[secondary]** — MiniCPM-o 4.5 (arXiv:2604.27393),
StreamMind, and the survey at arXiv:2606.19453

Used for: Omni-Flow aligns multimodal input and output along a **shared temporal axis**, formulating
interaction as a continuous full-duplex process; two-tier architectures assign latency-critical
interaction to frontend workers while backend workers asynchronously build persistent multimodal
memory.

Cited in: [`06`](spec/06-runtime.md)

---

## Agentic control

**WeaveBench: A Long-Horizon, Real-World Benchmark for Computer-Use Agents with Hybrid Interfaces** —
arXiv:2606.09426 · **[secondary]** · <https://arxiv.org/pdf/2606.09426>

Used for: **114 tasks across 8 real-world work domains** requiring coordinated GUI and CLI work.
Claude Opus 4.7 on its native runtime reaches **41.2%** PassRate (highest); the same model on the
reference runtime reaches 35.1%; GPT-5.5 reaches 33.3%. The model-runtime gap indicates a large share
of agentic performance lives in the harness rather than the weights.

Note: the PDF did not extract; figures confirmed from a secondary index of the paper. Re-verify
before relying on the exact numbers.

Cited in: [`00`](spec/00-premise.md), [`05`](spec/05-agency.md), [`10`](spec/10-open-problems.md)

---

## Self-improvement

**Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops** —
arXiv:2607.07663 · **[secondary]** · <https://arxiv.org/abs/2607.07663>

**AI's recursive self-improvement might not come so quickly after all** — MIT Technology Review,
August 2026 · **[secondary]** ·
<https://www.technologyreview.com/2026/08/18/1142188/ai-recursive-self-improvement/>

Used for: every real self-improvement loop running in 2026 is bounded, and none redesign their own
weights end to end — they rewrite prompts, code, and tools around a fixed model. Absent external
feedback, LLMs largely cannot self-correct reasoning, and nearly every 2026 system grounds critique
in an external signal. Real results exist (a large majority of merged code at one lab being
model-written; a research loop closing most of a benchmark gap) and remain narrow and measured.

Cited in: [`00`](spec/00-premise.md), [`09`](spec/09-self-improvement.md),
[`12`](spec/12-blueprint-reconciliation.md)

---

## Data sources named in the specification

Referenced as candidate corpora rather than as evidence for a claim: The Well, PDEBench, AirfRANS,
JHTDB (Johns Hopkins Turbulence Databases). Solver backends named: NVIDIA Warp, PhiFlow, OpenFOAM,
SU2.

---

## Recomputed, not cited

The following figures in [`12`](spec/12-blueprint-reconciliation.md) were derived directly rather
than taken from a source, and can be re-derived from the stated inputs:

- Dense parameter count at `D = 32,768`, `D_ffn = 131,072`, 192 layers: `4D² + 3·D·D_ffn = 17.18 B`
  per layer, `≈ 3.30 T` total.
- KV cache at 256 heads × 128 dims, no head sharing: 128 KiB per token per layer; 24 MiB per token;
  ~25 TiB per 1M-token stream.
- Bélanger conjugate depths at the two stated Froude numbers: 2.29 m and 3.24 m.
- Implied crest velocity from the stated impact velocity and 10 m drop: ~4.43 m/s.
- Implied discharge from the stated jet thickness, width, and impact velocity: ~3.67 m³/s.
- Air entrainment scaling under the quoted Ervine-Falvey correlation: `2 × (95.5/176.6) ≈ 1.08`,
  i.e. ~8%.
