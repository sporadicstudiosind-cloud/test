# Making a ~50M model smarter and faster to train: literature notes (Nov 2026)

Twenty-nine short literature searches, each scoped to one question about iridium-50m (a
looped, subject-routed ~50M model on free T4s). Several hit search rate limits, so many
numbers below come from abstracts or summaries, not tables. Treat **unverified** as
"plausible, check the paper before relying on it". What was acted on is in
docs/iridium-50m.md ("The next run").

## Already in iridium-50m

QK-norm, tied embeddings, one token per digit, input reinjection on every loop with a
loop-index signal, router z-loss, optional attention softcap, WSD schedule, EMA.

## Acted on (iridium-50m-next)

- **Muon.** ~1.4x vs a tuned AdamW at 130M, 1.1x at 1.2B; tuning the AdamW learning rate
  alone gave up to 2x ([Wen et al., ICLR 2026](https://arxiv.org/abs/2509.02046)).
  Moonshot's RMS matching lets the AdamW rate carry over
  ([arXiv 2502.16982](https://arxiv.org/abs/2502.16982)). On a T4, run Newton-Schulz in
  fp32: Turing has no bf16 tensor cores and `torch.cuda.is_bf16_supported()` still
  returns True there.
- **Packing** away the ~17% padding ([Krell et al.](https://arxiv.org/abs/2107.02027)).
- **Decay-phase data** ([OLMo 2](https://arxiv.org/abs/2501.00656),
  [SmolLM2](https://arxiv.org/abs/2502.02737)). Easy-to-hard curricula mostly fail
  (BabyLM 2025, [arXiv 2504.08165](https://arxiv.org/abs/2504.08165)).

## Worth doing next, roughly in order

1. **Learning-rate sweep** (above). The cheapest large gain the literature supports.
2. **Better data for each stack** (all free on Hugging Face; check each licence):
   - General: `HuggingFaceFW/fineweb-edu` (ODC-BY), `HuggingFaceTB/smollm-corpus`.
   - Maths: `HuggingFaceTB/finemath` 3+ (ODC-BY), `open-web-math/open-web-math`.
     Avoid `GAIR/MathPile` (NC-SA).
   - Code: `HuggingFaceTB/stack-edu` (Stack v2 terms apply), Python first.
   - Science: Cosmopedia's textbook-style splits and Wikipedia over raw papers. Simple
     text is what tiny models learn from ([TinyStories](https://arxiv.org/abs/2305.07759)).
   - Chat: `HuggingFaceTB/smol-smoltalk` and `everyday-conversations` (Apache-2.0).
     UltraChat, OpenHermes and WildChat carry OpenAI-output terms.
   - Tools: `Salesforce/xlam-function-calling-60k` (CC-BY-4.0, gated),
     `glaiveai/glaive-function-calling-v2` (Apache-2.0); `nvidia/When2Call` for when *not*
     to call.
   - Reasoning: short chains beat long ones below ~3B
     ([Li et al. 2025](https://arxiv.org/abs/2502.12143)). TinyGSM reports 63% GSM8K at
     125M with Python-solution data (**unverified**; check Table 1 of
     [arXiv 2312.09241](https://arxiv.org/abs/2312.09241)).
3. **Teacher-written subject text** (sequence-level distillation, one subject per stack).
   No tokenizer match is needed. Free API tiers give roughly 0.2-2M tokens/day per
   model, and several providers' terms bar training competing models, so this is a slow
   top-up, not a corpus.
4. **Mixture search with tiny proxies** ([RegMix](https://arxiv.org/abs/2407.01492):
   1M-parameter proxies picked the best 1B mixture). Transfer to a routed model is untested.
5. **Checkpoint averaging** of the last stable-phase checkpoints
   ([PMA](https://arxiv.org/abs/2505.12082)). Free, given checkpoints every ~250 steps.
6. **Small architecture ablations**, one at a time at 50M: final-logit softcap, value
   residuals ([arXiv 2410.17897](https://arxiv.org/abs/2410.17897)), gated attention
   ([arXiv 2505.06708](https://arxiv.org/abs/2505.06708)), n-gram input embeddings
   ([arXiv 2501.16975](https://arxiv.org/abs/2501.16975)). All were shown at larger scale.

## Not recommended now

- **Logit distillation:** the vocabulary is custom, and top-k caching for 5B tokens is ~320 GB.
  Distillation beats plain training only in some compute regimes
  ([Busbridge et al.](https://proceedings.mlr.press/v267/busbridge25a.html)).
- **Multi-token prediction:** hurts small models without a curriculum
  ([arXiv 2505.22757](https://arxiv.org/abs/2505.22757)).
- **RL at 50M:** no evidence it works this small. At 0.5B, RL on an instruct model gains
  a few points ([arXiv 2506.13404](https://arxiv.org/abs/2506.13404)), and "spurious
  reward" gains come from priors the model already had
  ([arXiv 2506.10947](https://arxiv.org/abs/2506.10947)).
- **Colab TPU:** the routing has data-dependent shapes and PyTorch/XLA is deprecated.

## Expectations and doubts

- **Capacity:** about 2 bits of knowledge per parameter
  ([Allen-Zhu & Li](https://arxiv.org/abs/2404.05405)), so about 12 MB of facts at 50M.
  Looping helps maths more than recall (secondary report).
- **Benchmarks:** SmolLM2-135M scores HellaSwag ~42 and GSM8K 1.4. For iridium-50m, expect
  HellaSwag ~28-35, MMLU at chance and GSM8K near 0. Use cloze-form evaluation
  ([OLMES](https://arxiv.org/abs/2406.08446)) and per-domain bits per byte.
- **Routing:** no study isolates subject- or MoE-routing gains below 100M parameters.
  Branch-Train-Merge shows domain experts must be forced to specialise
  ([arXiv 2208.03306](https://arxiv.org/abs/2208.03306)). A dense model at equal compute
  is the comparison that would settle whether the stacks earn their parameters.
- **Tool use:** reported reliability starts around 1B parameters. At 50M, expect
  narrow, fixed-schema tool use only.

## Second pass (20 Nov 2026): LittleBit-2 and the 2025-26 crop

Thirty-one more searches, the decisive claims re-checked against the papers'
full text and official code. What was built is opt-in in the code unless
"in iridium-50m-next" says otherwise; measured CPU results are in
docs/iridium-50m.md ("Tiny-model ablations").

**LittleBit-2** ([arXiv 2603.00042](https://arxiv.org/abs/2603.00042), ICML 2026)
is a *compression* method: a trained model's matrices become rank-r products of
binary matrices with real scale vectors, reaching 0.1-1 bits per weight, and the
Joint-ITQ rotation is a better starting point for that (Llama-3 8B at 1 bpw:
perplexity 11.5 vs 16.3 for LittleBit). It does not make a model smarter, and
every result is at 1.3B-70B; at 1.3B the papers already note the residual path
can hurt at very low bits. Two facts specific to iridium-50m: the fp16 scale
vectors alone cost about 0.115 bpw on its largest (1024 x 384) matrices, so the
0.1 bpw setting cannot exist here; and small, over-trained models lose the most
to quantisation ([Ouyang et al.](https://arxiv.org/abs/2411.17691)). Implemented
in `iridium/quant/littlebit.py` from the equations (the official code is
CC BY-NC); see docs/littlebit.md for what it measured.

| Idea | Evidence (verified) | Status |
|---|---|---|
| Gated attention, sigmoid gate per head after SDPA | NeurIPS 2025 best paper ([2505.06708](https://arxiv.org/abs/2505.06708)); 1.7B dense: PPL 7.50 → 7.40, trains at ~2x LR where the baseline diverges | `attn_gate` (opt-in; no effect in the 300-step ablation) |
| Canon layers A+C, residual causal width-4 conv | [2512.17351](https://arxiv.org/abs/2512.17351) + official code (Kaiming init, no bias, no activation); 2-4x synthetic reasoning depth; ~2.8% lower loss on a 4M TinyStories model (third party) | `canon`, **in iridium-50m-next** (ablation: 1.315 → 1.081) |
| Polar Express Newton-Schulz coefficients | [2505.16932](https://arxiv.org/abs/2505.16932): GPT-2 small val loss 3.588 vs 3.639 (Jordan's) at 1B tokens; checked numerically here | optimizer `muon_pe` |
| NorMuon per-neuron normalisation | [2510.05491](https://arxiv.org/abs/2510.05491): 21.7% vs 10.4% steps saved over Adam at 1.1B (Muon); ~6% fewer iterations than Muon at 124M | optimizer `normuon`, **in iridium-50m-next** |
| Linear decay to zero | [Bergsma et al., ICLR 2025](https://arxiv.org/abs/2502.15938) | `decay_shape="linear"`, `min_lr_ratio=0` |
| Ponder prior on the batch-mean exit distribution | one 160M paper ([2610.09013](https://arxiv.org/abs/2610.09013)), with a beta ramp to 0.05; Ouro's fixed 0.1 collapsed | `ponder_global` |
| Do the stacks earn their parameters? | learned routing does not make domain experts ([2601.03425](https://arxiv.org/abs/2601.03425)); nothing measured below 100M | presets `iridium-50m-control-random`, `-control-dense` |
| Self-improvement on exact-checkable tasks | [Lee et al., ICML 2025](https://arxiv.org/abs/2502.01612): a 14M model went from 16- to ~100-digit addition on its own filtered outputs | recipe noted, not built |

Not pursued, with the reason: hybrid linear attention (gains appear at 400M+ and long
context; no confirmed Turing kernels), byte-level models (behind BPE below ~1B),
diffusion LMs (win only at hundreds of epochs; this run repeats data twice),
ternary training from scratch (ParetoQ: −2.0 points at 125M; a final QAT phase
is the better route to a small file), mHC/Engram (shown at 3B-27B; the repo already
has hyper-connections), RL at 50M (no evidence), test-time verifiers (need a
trained verifier; later).

Data and deployment notes: rephrasing data with a local 1-3B model costs roughly
175 T4-hours per 500M tokens (estimate); use Apache-2.0 generators (Qwen3-1.7B,
SmolLM2), since Llama's licence requires a derived model's name to start with
"Llama" and Gemma's terms reach models trained on its outputs. For phones, the
routing's data-dependent shapes are what block ExecuTorch/ONNX export; a
fixed-capacity dispatch would unlock both.
