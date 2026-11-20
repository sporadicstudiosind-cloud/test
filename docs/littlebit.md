# Sub-1-bit compression (LittleBit / LittleBit-2) at iridium's scale

`iridium/quant/littlebit.py` implements LittleBit (Lee et al., NeurIPS 2025,
arXiv 2506.13771) with LittleBit-2's Joint-ITQ initialisation (arXiv 2603.00042,
ICML 2026). It is written from the papers' equations, because Samsung's code is
CC BY-NC 4.0. A Sonnet review checked it against both papers, covering:

- the factorisation;
- SmoothSign with k = 100;
- Dual-SVID;
- Joint-ITQ, with a symmetric sqrt(Σ) split and 50 iterations;
- the residual path, fitted to what the binarised primary misses and then trained jointly;
- the bit formula;
- the distillation loss of KL plus 10 × MSE on hidden states.

**It compresses a trained model; it does not make one smarter.** Each eligible
weight matrix becomes `diag(h)·sign(U)·diag(l)·sign(V)^T·diag(g)`. The rank sets
the bit budget, and the sign matrices cost one bit per entry.

## Measured on a 2M-parameter test model

The test model is the tiny Iridium-1 used for the CPU ablations: d = 128, with
2 core layers and 2 stack layers per stack. It was trained for 600 steps on
TinyStories bytes. The table shows held-out loss per token (lower is better)
after compression and 300 steps of distillation from the uncompressed model.
All 78 hidden matrices were compressed. Embeddings, heads, norms and routers
stayed in full precision.

| bits/weight (target) | Joint-ITQ | loss straight after compression | after 300 distillation steps |
|---|---|---|---|
| uncompressed (fp32) | – | 1.034 | – |
| 2.0 | yes | 1.888 | **1.088** (+5%) |
| 1.0 | yes | 3.056 | 1.226 (+19%) |
| 1.0 | no | 4.489 | 1.267 (+23%) |
| 0.55 (real 0.58: 28 layers over budget) | yes | 7.231 | 1.700 (+64%) |

What this shows:

* **Joint-ITQ helps here too.** At 1 bpw it starts at 3.06 instead of 4.49 and
  ends at 1.226 instead of 1.267. Its starting reconstruction error on a
  384 × 1024 test matrix was 46% instead of 62%.
* **At this size the usable budget is about 2 bits per weight, not 0.1–1.** The
  papers' sub-1-bit results are on 7B–70B models, which have far more redundancy
  per matrix.
* **There is a hard floor.** The fp16 scale vectors cost
  `32 · (out + in)` bits per matrix with the residual path. On iridium-50m's
  largest matrices (1024 × 384) that is about 0.115 bpw before a single sign bit
  is stored. `compress()` reports the layers it could not fit
  (`over_budget_layers`).
* **What 2 bpw buys.** The 50M body would shrink from about 100 MB in fp16 to
  about 13 MB. Embeddings at 16k × 384 add about 12 MB in fp16 on top. int4 would
  give about 25 MB with a far simpler runtime. Pick LittleBit only if file size
  matters more than both quality and simplicity.

These are CPU runs at toy scale with a single seed. The direction is reliable,
but the exact percentages will differ at 50M. Small and over-trained models
lose the most to quantisation (Ouyang et al. 2024, arXiv 2411.17691), and
iridium-50m is both. So expect the real penalty to be no smaller than this.

## Using it

```python
from iridium.quant.littlebit import compress, distill
student = copy.deepcopy(model)
report = compress(student, bpw=2.0)                 # in place; returns a size report
distill(student, model, batches, logits_of, steps=2000, lr=5e-4)
```

`logits_of(model, batch)` returns `(logits, hidden, mask)`;
`experiments/littlebit_tiny.py` is the script behind the table above.
`LittleBitLinear.packed()` returns the stored form: signs packed eight per byte
and scales in fp16. A runtime that executes packed binary matmuls is not part of
this repository. The PyTorch layer computes with ±1 floats.
