# Architecture options

Every option below is a config field whose default rebuilds the original model
exactly. They are independent and composable; `tests/unit/test_core_blocks.py`
runs each one, and all of them together, through the real model and checks:

* the parameter count equals the torch-free formula in `config.py`, exactly;
* cached token-by-token decoding equals the uncached forward to 1e-10 in
  float64, including through the two-loop ponder path;
* zeroing a block's output projections makes it an exact identity, so depth
  growth is function-preserving;
* every core-layer parameter receives a finite gradient.

**None of this is trained.** Each option changes what the model *can*
represent or how cheaply it runs. Whether it makes a trained model better is
measured only by training it, and nothing here has been. See
[`small-model-coherence.md`](small-model-coherence.md) for why that is the
binding constraint.

The `modern` rung (`get_config("modern")`, 743,752,454 parameters) assembles the
options with published evidence at scale and leaves out the ones without. Its
computed cache at 1M tokens is 10.7 GB (bf16, three ponder loops, worst-case
routing), against 75 GB for `test1b`; see `CoreConfig.cache_bytes`.

## Control core — `CoreConfig`, and superstacks — `SuperstackConfig`

Both accept the same per-layer fields (the superstack ones act on each stack's
self-attention; the bridge cross-attention onto the core always stays GQA).
`hyper_streams` is core-only.

| field | values | what it does | evidence / caveat |
| --- | --- | --- | --- |
| `layer_pattern` | tuple of `global`, `local`, `mla`, `deltanet`, repeated | per-layer attention kind | 3:1 `deltanet`:full is Qwen3-Next / Kimi Linear; 5 `local` : 1 `global` is Gemma 3 |
| `local_window` | int | sliding-window width for `local` layers | the cache keeps `window - 1` keys (an earlier version kept all of them) |
| `mla_kv_rank`, `mla_q_rank`, `mla_rope_dim` | ints | multi-head latent attention (DeepSeek-V2/V3) | cache is flat in head count. **At the shipped head counts narrow GQA is already cheaper**; MLA pays when you raise heads. MLA-only cores lift the `d_model == heads × d_head` constraint |
| `deltanet_conv` | int | Gated DeltaNet short-conv width | fixed-size decode state; cannot replace bridge cross-attention or sparse masks |
| `norm_kind` | `rms`, `dyt`, `derf` | normalisation-free elementwise norms | DyT's α is reported sensitive for LLMs; Derf's outer affine is an extrapolation |
| `block_kind` | `sequential`, `parallel` | PaLM parallel attention+FFN | PaLM measured a small loss at 8B, neutral at 62B; the matmul fusion that makes it faster is not implemented |
| `ffn_dynamic_rank` | int | token-conditioned low-rank FFN corrections | +2/4/8 % params at rank 8/16/32 (d=768); no evidence at scale |
| `hyper_streams`, `hyper_dynamic`, `hyper_sinkhorn_iters` | ints / bool | mHC residual streams (DeepSeek-V4 uses 4) | measured: unconstrained mixing amplified 1,069,385× over 48 layers; mHC stayed at 1.02×. Streams live inside each run of core layers |

## Model and codecs

| field | where | what it does |
| --- | --- | --- |
| `text_vocab_size` | `IridiumConfig` | subword vocabulary (0 = byte level); `codecs.vocab_size` must be ≥ it + 16 |
| `ple_dim` | `IridiumConfig` | Gemma 3n per-layer embeddings; zero-init, exactly function-preserving |
| `qk_norm` | `IridiumConfig` | RMSNorm on Q and K before RoPE |
| `mrope_sections` | `IridiumConfig` | M-RoPE (Qwen2-VL): media patches positioned by their grid (t, y, x); text exactly 1-D. Before this a 2×4 and a 4×2 image were indistinguishable |
| `rope_yarn_factor`, `rope_original_max_position` | `IridiumConfig` | YaRN context extension for every rotary table |
| `ngram_table_size`, `ngram_orders` | `CodecConfig` | hashed n-gram input embeddings (Over-Tokenized Transformer); history carried in the cache |
| `continuous_conditioning` | `CodecConfig` | flow head conditioning: `add` or `adaln` (DiT/SD3) |
| `flow_timestep_sampling` | `CodecConfig` | `uniform` or `logit_normal` (SD3) |
| `camera_features` | `CodecConfig` | camera modality (6 = Plücker ray per patch); needs `n_modalities >= 10`; input-only |

## Runtime

| module | what it does |
| --- | --- |
| `runtime.system1` | typed one-pass answers with calibrated confidence; escalates to generation when unsure (Jev-style routing) |
| `runtime.streaming` | lossless self-speculative decoding (exact) and CALM-style latency-budget exits (approximate, refuses options it cannot honour) |
| `runtime.context_budget` | memory cost of 128k–1M tokens per rung and layer schedule |
| `runtime.checkpoint_compat` | loads older checkpoints, filling only keys it can justify |
| `model.blocks.grow_depth` | function-preserving depth growth for progressive training |
| `Iridium1.enable_gradient_checkpointing` | exact (`use_reentrant=False`); off by default |
| `world.*` | cameras, Gaussian-splat scenes and `.ply` I/O, persistent world memory, action rollout |

## Training — `TrainConfig`, `phase1_pretrain --recipe omni`

| field | what it does | status |
| --- | --- | --- |
| `schedule="wsd"`, `decay_ratio` | warmup–stable–decay with a `1 − √` cooldown (Hägele et al. 2024); extendable runs | standard |
| `optimizer="muon"` | Muon (Moonlight RMS-matched) on hidden matrices, AdamW on embeddings/head/norms/biases | toy-scale only here |
| `loss_balance="ema"` | per-modality task losses divided by a running estimate of their own scale; raw values still logged | heuristic, unvalidated, off by default |
| `ema_decay` | exponential moving average of weights, saved in checkpoints, for evaluating generators | standard |
| `decay_groups` | weight decay on hidden matrices only; embeddings excluded (OLMo 2) | default |
| `OMNI_MIXTURE` | half natural language (prose + chat), half synthetic families | needs network |

`--recipe omni` combines the omni mixture, the rung's own tokenizer (also for
the evaluation corpora), WSD and EMA 0.999, and prints the data-budget audit
before training.

## Not integrated

* M-RoPE applies to the control core; superstacks and the bridge use the
  scalar order (their packing and position masks need it).
* Generated media tokens get 1-D rotary positions, not the grid positions
  training used; a layout-aware decoder is needed for that.
* Streaming supports only `controller_mode=False`, one loop, batch size 1.
* Nothing is trained.
