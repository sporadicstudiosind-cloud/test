# Verification Ledger

Every quantitative claim in the source plan that this build checked, with the
result and the test that holds it. "Confirmed" means the plan was right.
"Corrected" means it was not, and the corrected value is what the code uses.

Sources for the external facts are in [`evidence.md`](evidence.md).

---

## Arithmetic and accounting

| ID | Claim in the plan | Finding | Status | Held by |
|---|---|---|---|---|
| F-01 | Extreme config: 48 stacks x 196 layers x d=18432 totals **38.41 T** | 48 stacks of that geometry total **32.90 T**. 56 stacks reach the stated headline. The ladder uses 56. | **corrected** | `test_config_inventory.py::test_ladder_is_monotonic`, `iridium ladder` |
| F-02 | Base config totals **8.794 T** | **9.04 T**. The plan's own §2.1 requires a cross-attention bridge in every superstack and never counts it: 16 bridges x 462 M x 32 stacks = 237 B, plus entry/exit projections and halting heads. | **corrected** | `test_total_parameter_count_is_exact` (delta 0 vs real modules) |
| F-03 | MXFP4 at **0.5 bytes/param** -> 4.39 TB | MXFP4 is **4.25 bits/param**: 32 elements share one 8-bit E8M0 scale, so the scale costs 0.25 bits/element. Base weights are **4.80 TB**, 6.25% more. | **corrected** | `test_mxfp4_is_four_and_a_quarter_bits_not_four` |
| F-04 | BF16 training of the 8.8 T model on **1,024 H200** | Infeasible. Mixed-precision Adam needs ~16 B/param (BF16 weights + BF16 grads + FP32 master + two FP32 moments) = **145 TB**; 1,024 H200s hold 144 TB total, before any activation. The plan's table lists only the 17.6 TB of weights. Minimum to hold state alone: **1,207 H200** or **887 B200**. | **corrected** | `iridium plan base --training --accelerator h200` returns `infeasible` |
| F-05 | Active params/token: 168.5 B min, 676.4 B max | The two ends assume different `k` (min is top-1, max is top-2). Also conflates two quantities: a layer run three times *touches* one parameter set and *costs* three passes. Split into `active_parameters` and `flops_per_token`. | **corrected** | `test_flops_and_active_parameters_are_different_quantities` |

## Architecture and numerics

| ID | Claim in the plan | Finding | Status | Held by |
|---|---|---|---|---|
| F-06 | `SpectralConv3d` with **two** complex weight tensors | A real `d`-dimensional spectral kernel has `2**(d-1)` independent corner blocks: 1 in 1-D, **2** in 2-D, **4** in 3-D. `rfftn` halves only the last axis. With two blocks in 3-D, every mode with a negative second-axis wavenumber is written as zero — the operator is not merely under-parameterised, it is anisotropic. | **corrected** | `model/fno.py::n_corner_blocks`, shift-equivariance 3e-16 |
| F-07 | Focus metric `F(x)` from `h_norm.mean(dim=1)` | A chunk mean contains future tokens. The depth chosen for token *t* would depend on tokens that do not exist at sampling time, so training and serving compute different functions. Replaced by a cached running prefix statistic. | **corrected** | `test_router.py::test_focus_is_causal`, `::test_prefix_summary_survives_chunk_boundaries` |
| F-08 | Load-balance loss `N * sum m_j P_j`, `m_j` from top-1 while dispatching top-2 | Missing the `alpha` coefficient, and blind to half the traffic it balances. Also: the term penalises *correlation* between dispatch and probability, so with uniform `P` it is constant — worth stating, since it is usually misread as penalising unevenness. | **corrected** | `test_balance_loss_penalises_dispatch_probability_correlation` |
| F-09 | Gating weights = `softmax(top-k(logits + gumbel))` | Softmaxing the noised logits biases the combination weights toward whatever the noise favoured. Selection uses noise; weighting uses clean probabilities. | **corrected** | `model/router.py` |
| F-10 | Escalate when `\|\|div u\|\|_2 > 1e-4` | Dimensional quantity against a bare constant: the same flow in cm/s has 100x the divergence of the same flow in m/s. Replaced by the dimensionless `\|\|div u\|\| L / U`. | **corrected** | `test_fluid2d.py::test_divergence_norm_is_unit_invariant` |
| F-11 | Superstacks drop self-attention entirely and only cross-attend to the core | Works, but discards a stack's ability to relate a token to the other tokens it specialises in. **Sparse stack-local KV** — attend to the subset of earlier tokens also routed here — is always available (those tokens ran these layers, so their K/V exist) and makes decoding exact. | **improved** | `test_kv_parity.py` (bit-exact at 1 loop) |
| F-12 | Top-k routing is causal | True for top-k *over experts*. Top-k *over the sequence* (capacity dropping) is not: membership depends on later tokens, which is why Mixture-of-Depths needs an auxiliary predictor. Capacity dropping is refused rather than silently enabled. | **confirmed with scope** | `test_capacity_dropping_is_refused_as_non_causal` |
| F-13 | "Macro-chunk routing" | Necessary, and for a reason the plan does not give: a spectral operator needs a *complete* grid. Per-token routing shredded field spans across stacks — measured **0%** of grids intact. Span-coherent routing (pooling logits *and* the exploration noise) gives **100%**. | **confirmed, mechanism supplied** | `stats["grid_intact_fraction"]` |
| F-14 | Sandboxed "POSIX micro-VM" | The plan's code is `subprocess.run(["python3", path])` with a timeout: the child inherits environment, filesystem, network and privileges. Replaced with process-group isolation, RLIMIT_CPU/AS/FSIZE/NPROC, a scrubbed environment, and a network namespace where available — plus an explicit record of the guarantees **not** obtained (no seccomp, no filesystem namespace). | **corrected** | `test_sandbox.py` |

## Found while building, not in the plan

| ID | Finding | Held by |
|---|---|---|
| F-15 | **Nyquist wavenumber breaks Leray projection.** On an even grid `fftfreq` assigns `-n/2` to a bin that is its own conjugate partner, so any odd-order multiplier is non-Hermitian there and `real(ifft(...))` discards a large imaginary part. Measured: projection exact in spectral space (2e-13) and divergence **443** after the transform back, with projection no longer idempotent. Taylor-Green hides it completely — a two-mode analytic flow has no Nyquist content, which is why validating only against smooth solutions is insufficient. | `test_leray_projection_is_idempotent` |
| F-16 | **A depth-only steady-state residual returns the initial condition.** A uniform initial depth has zero depth tendency at the first step while its momentum is far from balance, so the solver declared convergence in 1 step and reported the initial guess as the answer. The residual must watch both conserved variables. | `test_solver_finds_normal_depth_from_either_side` (asserts `steps > 100`) |
| F-17 | **Length bucketing defeats specialisation.** Item length correlates with task family, so length buckets are family-homogeneous — and the balance objective is per batch. On a single-family batch it demands the stacks be used equally *for that family*, the opposite of what phase 2 wants. Bucketing is off by default. | `training/datasets.py::BatchLoader` |
| F-18 | **Loop-level cache coherence.** The ponder loop has the superstacks' problem one level up: if tokens take different loop counts, the core's loop-L cache has holes. Three ways out (copy-through, sparse loop history, chunk-uniform loops); chunk-uniform is implemented, the other two are specified. Choosing sparse-loop-history silently is the version that passes every unit test and then samples differently from how it was trained. | `model/iridium1.py` docstring; `test_kv_parity.py` |
| F-19 | **Atomic spans constrain the serving path.** A spectral block needs the whole grid in one chunk, so prefill boundaries must not split a field span. Splitting does not degrade the answer, it changes the function — the operator does not fire at all. | `test_splitting_a_field_span_across_chunks_changes_the_answer` |
| F-20 | **Aggregate vs per-link bandwidth.** Comparing one link's latency against the cluster's aggregate FLOPs makes any routed design look communication-bound by ~200x. On a like-for-like throughput basis the base rung is compute-bound (0.018x). The bridge strategy still matters: broadcast costs **7.34 MB/token**, cached bridge K/V **0.115 MB/token**, a 64x difference. | `iridium plan base --bridge {broadcast,cache_kv}` |

## Confirmed as stated

- Grouped-query attention, SwiGLU, RMSNorm, RoPE geometry.
- Switch-Transformer load balancing is the right family of objective.
- PonderNet stopping distributions; the torch implementation reproduces the
  repository's existing numpy contract to float64 (`test_torch_stopping_matches_the_numpy_contract`).
- Fourier Neural Operators solve parametric PDEs in latent space; the
  implementation is shift-equivariant to 3e-16 and exactly resolution-invariant.
- B200 = 192 GB HBM3e, NVLink 5 at 1.8 TB/s; H200 = 141 GB.
- Mixed-precision Adam is ~16 bytes/parameter.
- MX formats: block of 32, E2M1 elements, E8M0 shared scale (OCP MX v1.0).
