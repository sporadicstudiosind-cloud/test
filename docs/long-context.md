# Getting to a 1M-token context window

This is a requirements document, not a status report. Nothing in this
architecture "has" a million-token window today, and shipping
`iridium/model/deltanet.py` does not change that by itself — it makes one
necessary piece of it *possible*. The other pieces are listed below as
requirements because they are not optional, and one of them (training) is a
data/compute decision this document cannot make for you.

## What 1M tokens requires, in order

**(a) Attention whose cost and memory do not grow with T on most layers.**
Full causal attention keeps every token's K/V forever: at 1M tokens that is
gigabytes per layer (see the table below), and its own compute is
quadratic. There is no way to run 1M tokens through an architecture where
every layer does this, on any hardware this repository's rungs target.

**(b) Positional encoding that extends past what the model was trained at.**
`iridium/model/rope.py` already implements the two static, verified pieces of
this — linear position interpolation (Chen et al. 2023) and NTK-aware base
adjustment — and is explicit that it is *not* full YaRN (no per-frequency
ramp, no attention-logit temperature correction). See "RoPE extension"
below for what that gap means in practice.

**(c) A KV cache that fits.** Even the layers that keep exact attention need
their cache priced honestly at 1M tokens, including the mechanisms this
architecture already has beyond a single core cache: superstack stack-local
KV and the bridge's read of core KV (`iridium/model/context_memory.py`,
`iridium/runtime/long_context.py`).

**(d) Training at long lengths, on long documents.** This is the one this
document cannot discharge with code. A model whose architecture *can* run
1M tokens but was never trained past, say, 8k tokens of contiguous document
will not use tokens 500,000 through 1,000,000 for anything — RoPE
extrapolation and hybrid linear-attention layers both need to see long-range
dependencies during training to learn to use them, per every cited paper
below. This needs: a progressive length-extension curriculum (train short,
extend the context window in stages — this is what RoPE `scaling=` and a
periodic-reset `LongContextSession` are *for*), and a corpus with actual
long-range structure (books, long code repositories, long conversation
logs) rather than short documents concatenated to fill a context window,
which teaches a model to ignore anything past the document boundary it was
actually trained on. Neither exists yet in this repository's training
pipeline as far as this document's author checked; both are prerequisites,
not follow-ups.

## Current practice (verified, not assumed)

| Model | Mechanism | Ratio | Source |
|---|---|---|---|
| Qwen3-Next | Gated DeltaNet (linear) : Gated Attention (full) | 3:1 (every 4th layer full) | [arXiv 2412.06464](https://arxiv.org/abs/2412.06464) for Gated DeltaNet itself; ratio per public technical writeups of Qwen3-Next's `full_attention_interval` |
| Kimi Linear | Kimi Delta Attention (KDA, a finer-gated Gated DeltaNet) : full attention (MLA) | 3:1 | [arXiv 2510.26692](https://arxiv.org/abs/2510.26692) |
| Gemma 3 | local sliding-window (window 1024) : global full attention | 5:1 | [arXiv 2503.19786](https://arxiv.org/pdf/2503.19786) |
| DeepSeek-V4 | **Not** MLA + sparse alone as a first guess would have it: a hybrid of Compressed Sparse Attention (4x KV pooling + a lightning indexer selecting top-k compressed blocks per query) and Heavily Compressed Attention (more aggressive KV compression, dense) | claims 1M tokens at ~10% of DeepSeek-V3.2's KV cache and ~27% of its FLOPs at that length | [arXiv 2606.19348](https://arxiv.org/abs/2606.19348) |

Gated DeltaNet (Yang, Kautz, Hatamizadeh; ICLR 2025) is the paper
`deltanet.py` implements. Its own headline result is not "linear attention
matches full attention" in the abstract sense — it is that gating (fast
memory erasure) and the delta rule (targeted, not additive, memory writes)
are *complementary*, and that hybridizing Gated DeltaNet with sliding-window
attention or Mamba2 layers gets both training efficiency and task quality
better than either alone. That is the same shape of finding as Qwen3-Next's
and Kimi Linear's 3:1 hybrids, and is why this document recommends a hybrid
schedule rather than "replace every layer."

## Which layers in *this* architecture can become DeltaNet, and which can't

This is architecture-specific, not a paraphrase of the papers above, and
comes from reading `iridium/model/layers.py`, `iridium/model/control_core.py`
and `iridium/model/superstack.py` directly.

**Can (self-attention with a plain causal-plus-padding `keep`):**

- The control core's `TransformerBlock` layers. `TransformerControlCore.
  stream_keep` builds `causal_keep(...) & valid[:, None, None, :]` — exactly
  the convention `GatedDeltaNet` accepts (see its module docstring and
  `_validity_from_keep`). This is the layer stack that actually needs to
  span the full 1M-token stream, so it is also the one whose cost matters
  most.
- A superstack's own self-attention over its stack-local, already-routed
  tokens (`superstack.py`'s `self_keep = causal_keep(m, ...) & valid`). The
  router has already decided *which* stream tokens land in this stack and in
  what order; once gathered, the recurrence only needs that order, not the
  original stream positions (this is also why `positions` is ignored — see
  `deltanet.py`'s module docstring). Whether it is *worth* doing here depends
  on how large `m` gets in practice — see "not addressed" below.

**Cannot:**

- `BridgeCrossAttention`. A superstack token attends onto the *control
  core's* KV, a different sequence than the one that would produce the
  query-side recurrence. There is no self-recurrence to run; a fixed-size
  state built by walking the superstack's own tokens has nothing to do with
  the core's separately-produced states. Keep this an attention layer, full
  stop — extending the core's context length is what makes the KV it reads
  large, which is a reason to make the *core* cheap, not the bridge.
- Any layer whose `keep` varies which keys are visible *per query row*
  beyond plain causality — a per-row sparse selection, a capacity-drop mask,
  or (per `deltanet.py`'s docstring) anything not reducible to "one
  per-position validity vector, same for every query row past the causal
  cutoff." `GatedDeltaNet._validity_from_keep` raises rather than silently
  computing something wrong if the shape it's handed doesn't fit that
  convention — it does not attempt to detect or reject a mask that fits the
  shape but violates the convention some other way, so this is a contract on
  the caller, not a runtime guarantee.
- Sliding-window layers, *as an exact replacement*. `GatedDeltaNet`'s
  `alpha_t` is a soft, learned decay, not `sliding_window_keep`'s hard
  cutoff at exactly `window` tokens. A hybrid schedule that wants a real
  bound on a "local" layer's influence needs either to keep it as
  `GroupedQueryAttention(window=...)`, or to periodically reset the
  DeltaNet state the way `LongContextSession` already resets its raw KV at
  window boundaries — an explicit engineering decision, not something the
  gate does for free.

**Not addressed by this document or by `deltanet.py`:** how large a
superstack's own `m` (routed sequence length) gets at 1M input tokens. That
depends on the router's `top_k` and its balance loss, not on anything in
this file, and is worth checking before deciding stack layers need this
treatment at all — if routing already keeps `m` small, the stack-local
cache may already be cheap enough that converting it buys little.

## RoPE extension for the remaining global layers

`rope.py`'s `scaling=` already gives two of YaRN's three ingredients:

- `{"type": "linear", "factor": F}` — position interpolation, divides
  positions by `F` so the model never sees an index past what it trained on.
- `{"type": "ntk", "factor": F}` — raises `theta` so short-wavelength
  (high-frequency) rotary dimensions barely stretch while long-wavelength
  ones absorb most of the extension.

What it does **not** do (per its own docstring, restated here because it
matters for a 1M claim specifically): no per-frequency-band ramp between the
NTK and linear treatments, and no `1/sqrt(t)`-style attention-logit
temperature correction. Full YaRN's ablations are the reason these two
pieces exist — going to 1M tokens (a much larger extension factor than YaRN
was demonstrated at) is exactly the regime where the missing ramp and
temperature correction are more likely to matter, not less. Recommendation:
start with `ntk` scaling on the remaining full-attention layers, sized to
the rung's `max_seq_len` → target-length ratio, and treat "does this still
need the ramp and temperature term" as an empirical question to check with
real long-document eval, not one to assume away.

## The 1M-token memory table

Computed with `iridium/runtime/context_budget.py` (no model instantiated;
see that module's docstring for why that is the right way to answer this
question). Two schedules per rung:

- **baseline**: today's architecture, every core layer full causal
  attention (`window=None`), superstack KV priced exactly as
  `IridiumConfig.kv_bytes_per_token` already does.
- **hybrid**: `hybrid_schedule(n_layers, ("deltanet","deltanet","deltanet","global"))`
  applied to the *core only* (Qwen3-Next / Kimi Linear's 3:1 ratio).
  Superstack KV is left unchanged in this table — see the caveat below.

| rung | tokens | dtype | baseline total | hybrid total (core only) | baseline core cache | hybrid core cache |
|---|---:|---|---:|---:|---:|---:|
| nano | 131,072 | bf16 | 0.87 GB | 0.67 GB | 0.27 GB | 0.067 GB |
| nano | 1,048,576 | bf16 | 6.51 GB | 4.90 GB | 2.15 GB | 0.537 GB |
| nano | 1,048,576 | fp8 | 3.29 GB | 2.48 GB | 1.07 GB | 0.269 GB |
| nano100m | 131,072 | bf16 | 1.42 GB | 1.08 GB | 0.40 GB | 0.068 GB |
| nano100m | 1,048,576 | bf16 | 9.87 GB | 7.19 GB | 3.22 GB | 0.537 GB |
| nano100m | 1,048,576 | fp8 | 5.04 GB | 3.70 GB | 1.61 GB | 0.269 GB |
| test1b | 131,072 | bf16 | 11.13 GB | 9.52 GB | 2.15 GB | 0.538 GB |
| test1b | 1,048,576 | bf16 | 75.02 GB | 62.13 GB | 17.18 GB | 4.30 GB |
| test1b | 1,048,576 | fp8 | 38.51 GB | 32.07 GB | 8.59 GB | 2.15 GB |

The core cache shrinks by roughly the fraction of layers converted (3/4 of
core layers become O(1)-state, so the core term drops to ~1/4 plus a fixed
DeltaNet state that is, at these widths, single-digit megabytes — see
`GatedDeltaNet.state_bytes`). **The total barely moves, and that is the
honest finding, not a hedge**: at every rung above, superstack stack-local
KV (`top_k * SuperstackConfig.kv_bytes_per_token(...) * tokens`) is larger
than the entire baseline core cache and is *unchanged* by this table's
hybrid schedule, because converting only the core does not touch it. At
`test1b`, 1M tokens, bf16: baseline core 17.18 GB, hybrid core 4.30 GB, but
superstack KV is 55.84 GB either way — the schedule change saves ~13 GB out
of a ~75 GB baseline, not the order-of-magnitude the core numbers alone
would suggest.

Illustrative (not wired, not in `context_budget.budget`'s current
`include_stacks=True` path): if superstack self-attention *also* became
DeltaNet — which the "can" list above says is architecturally sound —
`deltanet_state_bytes` on the stack's own shapes gives a **fixed**,
token-independent state of 0.34 MB (nano), 0.90 MB (nano100m) and 5.83 MB
(test1b) per stream, regardless of whether the stream is 1,024 or
1,048,576 tokens. That is the number that would actually make a 1M-token
window cheap end to end, rather than merely cheaper in the core. Whether
it's worth doing depends on the open "how large does `m` get" question
above — this document flags it rather than deciding it, and
`context_budget.py` deliberately does not bake it in as a default, since
nothing in this codebase currently instantiates a DeltaNet-based superstack
layer to check the claim against.

## Recommended schedule for this architecture

1. Core: `hybrid_schedule(core.n_layers, ("deltanet","deltanet","deltanet","global"))`
   — 3 `GatedDeltaNet` layers to 1 `GroupedQueryAttention` layer, matching
   Qwen3-Next/Kimi Linear's verified ratio, not a number invented for this
   repo. Put the DeltaNet layers first in each group of 4 (paper's own
   hybrids put the full-attention "refresh" layer periodically, not at the
   very start where nothing has accumulated into an RNN state yet).
2. The remaining core global layers: RoPE with `scaling={"type": "ntk",
   "factor": target_len / max_seq_len}`, per "RoPE extension" above, and
   flagged for a linear-vs-NTK, and eventually full-YaRN-ramp, ablation
   before trusting it past a few hundred thousand tokens.
3. Superstacks: leave as-is for a first cut (this is the lower-risk,
   lower-effort path — it does not touch the router, the bridge, or
   `context_memory.py`), but track the "how large does `m` get" question
   above; if it turns out routed sequence lengths approach the stream
   length, the same 3:1 DeltaNet treatment applies to stack self-attention
   for the reasons under "can" above, and the illustrative numbers show what
   it would buy.
4. `BridgeCrossAttention` and the router: unchanged. Neither is a
   self-recurrence a DeltaNet-style state can replace (see "cannot" above).
5. Training curriculum: progressive length extension (start at the current
   `max_seq_len`, extend in stages toward the 1M target, adjusting RoPE
   `scaling` per stage) on data with genuine long-range structure. This is
   listed as a requirement, not scheduled or estimated here, because no
   training run against it exists yet to report a result from.

## Wiring notes for whoever owns `iridium1.py` / `control_core.py` / `config.py`

- `GatedDeltaNet(d_model, n_heads, d_head, conv_size=4, chunk_size=64,
  eps=1e-6)` — same constructor shape as `GroupedQueryAttention` minus
  `n_kv_heads` (no GQA grouping; one gate gate per head) and `rope` (unused,
  see its docstring). `forward(x, positions, keep, cache, cache_key)` is
  signature-identical, so a per-layer `layer_types: list[str]` in
  `CoreConfig` (mirroring `hybrid_schedule`'s output) is enough to pick
  `GatedDeltaNet` vs. `TransformerBlock` per layer index at model-build time.
- `keep` must be built the same way `TransformerControlCore.stream_keep`
  already builds it (causal & key-padding, nothing sparser) for any layer
  index assigned `"deltanet"`. This is already true for the core; it would
  need checking again if `"deltanet"` is ever assigned to a superstack layer
  index, against `superstack.py`'s `self_keep` construction specifically
  (not `bridge_keep`, which must stay on a `BridgeCrossAttention` layer).
- Cache dict entries for a DeltaNet layer are `(state, conv_q, conv_k,
  conv_v)`, not `(k, v)` — any code that inspects cache entries by shape
  (rather than treating them as an opaque per-layer blob, which is how
  `GroupedQueryAttention` and `TransformerControlCore._run` already treat
  them) needs to branch on layer type before doing so.
- `GatedDeltaNet.param_count(d_model, n_heads, d_head, conv_size)` and
  `GatedDeltaNet.state_bytes(n_heads, d_head, conv_size, bytes_per_element)`
  exist for `IridiumConfig`'s own parameter/memory accounting the way
  `CoreConfig.params_per_layer` and `.kv_bytes_per_token` do today; wiring a
  per-layer-type schedule into `CoreConfig` should call these instead of
  assuming every core layer costs the same as `GroupedQueryAttention`.
