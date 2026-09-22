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

The `modern` rung (`get_config("modern")`, 677 M parameters) assembles the
options with published evidence at scale and leaves out the ones without.

## Control core — `CoreConfig`

| field | values | what it does | evidence / caveat |
| --- | --- | --- | --- |
| `layer_pattern` | tuple of `global`, `local`, `mla`, `deltanet`, repeated | per-layer attention kind | 3:1 `deltanet`:full is Qwen3-Next / Kimi Linear; 5 `local` : 1 `global` is Gemma 3 |
| `local_window` | int | sliding-window width for `local` layers | bounds KV growth on those layers |
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

## Not integrated

* Options apply to the **control core** only. Superstacks keep GQA and
  SwiGLU; at long context their KV cache dominates (`long-context.md`).
* `codecs/spatial.py`'s multi-axis RoPE and `world.tokens.anchor_coordinates`
  are built and tested but not yet fed into attention.
* RoPE supports linear and NTK scaling, not full YaRN.
* Streaming supports only `controller_mode=False`, one loop, batch size 1.
