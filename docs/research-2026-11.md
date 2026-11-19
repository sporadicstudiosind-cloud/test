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
