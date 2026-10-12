# Realtime streaming: cutting the control core back mid-decode

Implementation: `iridium/runtime/streaming.py` (`StreamingSession`),
`iridium/model/exits.py` (`ExitHead`, `early_exit_loss`, `layer_dropout_schedule`,
`curriculum_weights`). Tests: `tests/unit/test_streaming.py`,
`tests/unit/test_exits.py`.

Two ways to make the control core cheaper per token, and they trade off
different things, so they are two modes rather than one knob:

| | `mode="lossless"` | `mode="budget"` |
|---|---|---|
| Paper | LayerSkip (Elhoushi et al., 2024, [arXiv:2404.16710](https://arxiv.org/abs/2404.16710), ACL 2024) | CALM (Schuster et al., 2022, [arXiv:2207.07061](https://arxiv.org/abs/2207.07061), NeurIPS 2022) |
| Mechanism | draft with a shallow prefix of the core, verify with one full-depth pass, accept/correct | exit the core at the first layer whose confidence (or deadline) says "stop" |
| Output vs. full-depth greedy | **identical**, always | an approximation; never identical |
| What is skipped | router, superstacks, remaining core layers -- for the *draft* only; every emitted token still gets a full-depth pass | the remaining stage-one core layers, for real, for the emitted token itself |
| Failure mode of an untrained exit | slower than plain decoding (low acceptance -> wasted verify passes), never wrong | worse text, not slower |

## Why lossless mode is exact

Self-speculative decoding does not change what full-depth greedy decoding
would output; it changes how many full-depth forward calls it takes to
compute that output. Concretely: a shallow forward through the core's first
`min_layers` layers plus an `ExitHead` proposes `k` draft tokens. Those `k`
tokens (plus the one token the full-depth model had already decided but not
yet run through itself) are then given to the *real* model in a single
forward call against the *real* KV cache. Because a transformer's hidden
state at position `i` is a pure function of the tokens at positions `<= i`
and the cache holding correct full-depth K/V before that (exactly the
property `tests/integration/test_kv_parity.py` exists to pin down), that one
verify call computes precisely the hidden state -- and therefore precisely
the `argmax` -- that full-depth greedy decoding would have computed at every
one of those positions, had it decoded them one at a time. Comparing the
draft to that `argmax` is not an approval heuristic, it *is* the full-depth
greedy decision; the module only ever emits that decision, whether the draft
happened to match it or not. This is why the exactness test
(`test_lossless_matches_full_depth_greedy`) checks *token-id equality*, not
closeness, against `iridium.runtime.generate.generate(..., temperature=0)`,
over several prompts and seeds, and passes.

**What acceptance rate changes:** how many full-depth forward calls it takes
to produce a given number of tokens. It cannot change which tokens come out.

**What acceptance rate does *not* change:** correctness. Even an
untrained, randomly initialized `ExitHead` (the default when
`StreamingSession` is not given one) produces exactly the same output ids as
plain decoding -- it just does so slower, because it drafts nothing useful.
The tests confirm this too (`test_lossless_produces_at_least_one_rejection_across_seeds`,
`test_lossless_acceptance_and_forward_count_are_reported`).

### The cache correctness problem, and how it is solved here

A verify call writes real cache entries for its whole chunk. When the
accepted prefix is shorter than the chunk, those extra entries describe
tokens that were never actually emitted, and every later token would attend
to or read that phantom state if it were left in place.

The natural first idea -- slice every cache tensor back by the rejected
count -- works for the control core's own `(k, v)` entries and for the
bridge state, which are plain `[b, ..., seq, ...]` tensors. It does **not**
work for two other cache families this architecture has:

* **superstack caches** are keyed on a *packed*, member-only sequence
  (`iridium/model/router.py:pack_routed` gives a slot only to the tokens
  actually routed to that stack in a given call), so "drop the last N
  positions" does not even parse without first re-deriving, per stack, how
  many of the rejected tokens landed there;
* **the router's prefix-mean statistic** (`MacroRouter.prefix_summary`) is a
  running sum and a count, not a per-position list, for exactly the reason
  its own docstring gives (a token decoded one at a time must see the same
  summary a token under teacher forcing would).

`StreamingSession` instead snapshots the whole cache dict (a shallow copy is
exact here, because every cache write in this model is a key
*reassignment*, never an in-place mutation of a tensor already in the
cache) before the verify call, and on a rejection restores that snapshot and
replays only the accepted prefix as one ordinary chunked forward -- the same
code path the KV-parity test already certifies, so it is correct for every
cache family at once without any family-specific logic. The cost is
redoing the accepted prefix's stage-one/router/bank/stage-two work a second
time on a rejection, which a from-scratch per-key truncation could in
principle avoid; that was judged not worth the risk of getting the
superstack/router cases subtly wrong for a first implementation.
`test_cache_after_speculative_decoding_matches_plain_decoding` compares the
resulting cache, not just the emitted ids, against plain token-at-a-time
decoding's cache from scratch, because a wrong cache does not show up in the
very next token -- only in one further downstream, which an ids-only test
would miss.

## Why budget mode is lossy, and what the approximation costs

CALM's premise is that not every token needs the model's full capacity.
`StreamingSession(mode="budget")` exits the control core's stage one at the
first layer `e >= min_layers` whose `ExitHead` confidence clears
`confidence`, or when a wall-clock deadline derived from
`target_tokens_per_s` would otherwise be missed for this token, whichever
comes first (see `_budget_round`). This is a real, permanent shortcut, not
a draft that gets corrected: whatever the exit state was, that is what the
router, every superstack, and stage two of the core actually see and act on
for this token. There is no verification step and no way to recover the
full-depth answer after the fact.

**The specific approximation, and its cost.** Every layer this token
skipped still needs *some* key/value pair in the cache, because later
tokens attend back to this position at every layer, skipped or not. This
follows CALM's "state propagation": the exit hidden state is fed through
each skipped layer's own K/V projections (never through that layer's
residual or FFN path) to produce a stand-in K/V. This is an approximation,
not a computation of what that layer would actually have produced: the real
K/V for layer `e+1` depends on layer `e+1`'s own attention and FFN having
run on the *previous* tokens' true layer-`e+1` states, not on being handed
layer-`e`'s output directly. The cost is compounding and directional: every
token that attends back to a shallow-exited position is reading a strictly
worse approximation of that position the deeper the layer being
approximated, and because K/V for a skipped layer is derived from the exit
state rather than being independently computed, an early exit at position
`t` also slightly degrades the state-copy K/V *available to* every later
early-exited token that attends to `t`. There is no bound proven here on how
this compounds over a long generation; a real deployment would need to
measure downstream quality (not just latency) as a function of how
aggressively `confidence`/`min_layers`/`target_tokens_per_s` are set, which
is out of scope for this change.

## Training the exits: untrained gives ~0% acceptance, and that is expected

Both modes read the same `ExitHead`: an exit-specific `RMSNorm` (the only
new large-ish tensor) followed by the model's own tied text head (reused,
not copied -- see `ExitHead`'s docstring for why it is passed to `forward`
rather than stored as a submodule) plus a one-unit confidence head. Nothing
about `ExitHead`'s *architecture* makes early layers predictive of the final
output; that has to be trained in, which is exactly what LayerSkip's
early-exit loss and layer-dropout curriculum are for
(`iridium/model/exits.py:early_exit_loss`, `layer_dropout_schedule`,
`curriculum_weights`). None of this is wired into any training loop by
default -- it is inert until a training script calls `early_exit_loss`
explicitly, matching this repo's convention for optional mechanisms
(compare `Iridium1.enable_gradient_checkpointing`).

Measured here on `iridium.config.get_config("tiny")` with default
(randomly initialized, **never trained**) `Iridium1` weights and a default
(randomly initialized, **never trained**) `ExitHead`, `draft_len=4`, prompt
`"the quick brown fox jumps over the lazy dog"`, 24 generated tokens, CPU,
single run (`OMP_NUM_THREADS=1`, no averaging -- treat as one data point,
not a benchmark):

```
[lossless] drafted=42 accepted=0 verify_rounds=12 full_depth_forwards=35
plain generate():            24 tokens, ~137 ms wall
StreamingSession(lossless):  24 tokens, ~238 ms wall  (35 full-depth forwards vs. 24 for plain)
```

Acceptance was 0/42: with random weights, the shallow draft's greedy choice
has no reason to agree with the full-depth model's, so every round pays for
a verify pass over the whole draft *and* a replay of the accepted-prefix
work *and* still needs the corrected-token step -- strictly more full-depth
compute than plain decoding, not less. This is the correct, expected
behavior of an untrained exit under this design (see the module docstring:
"acceptance rate changes how much compute is saved; it cannot change the
output"), and it is *not* a labelled speedup -- there isn't one here.
Whether draft-and-verify pays off in wall-clock time depends entirely on
how well `early_exit_loss` has trained the exit head to predict the
full-depth model's greedy choice; that training has not been run as part of
this change (no training script invokes it), so no acceptance number better
than "measured at ~0% on an untrained model" can honestly be reported yet.
The paper's own reported speedups (up to 2.16x on summarization, 1.82x on
coding, 2.0x on TOPv2, on their models and hardware, not this one) are a
property of a model actually trained with the early-exit loss and
layer-dropout schedule; quote them as the paper's numbers, not this repo's.

Budget mode, same config family, CPU, single run, `min_layers=1`,
`confidence=0.9`:

```
tiny  (core.split=2): confidence-driven exit never fires below full depth
                       at this depth (mean exit_layer = split = 2 of 2) --
                       there are only two stage-one layers to skip, so there
                       is little room to observe the mechanism at this scale.
nano  (core.split=4): target_tokens_per_s=None:  mean exit_layer = 4 of 4 (no deadline pressure)
                       target_tokens_per_s=1e9:   mean exit_layer = 1 of 4 (deadline forces min_layers)
```

This shows the deadline half of the mechanism working (an impossibly tight
deadline reliably forces the earliest allowed exit) on an untrained
confidence head, which is exactly the part that does not depend on
training. It does not, and cannot, show the confidence half working
usefully, since an untrained confidence head's output is close to
`sigmoid` of a random projection and has no relationship to whether the
exit's own logits are actually trustworthy; `early_exit_loss`'s confidence
term (`ExitHead`'s `confidence` output is trained against "did this exit's
own argmax match the label") exists specifically to make that number mean
something, and, as with lossless mode's acceptance, this has not been
trained as part of this change.

## API

```python
from iridium.runtime.streaming import StreamingSession, StreamEvent

session = StreamingSession(
    model,                      # an Iridium1 instance
    target_tokens_per_s=None,   # mode="budget" only: soft per-token deadline
    confidence=0.9,             # mode="budget" only: exit-confidence threshold
    min_layers=None,            # draft depth / earliest allowed exit; default max(1, split // 2)
    mode="lossless",            # "lossless" | "budget"
    draft_len=4,                # mode="lossless" only: tokens drafted per round
    exit_head=None,             # optional pre-trained ExitHead; else a fresh untrained one
    max_new_tokens=64,
    stop_ids=(2,),
    text_offset=16,
    seed=0,
)

for event in session.stream(sample):     # sample: iridium.codecs.spans.Sample
    ...  # event: StreamEvent(token, text, modality, exit_layer, latency_ms, mode, accepted, position)

session.interrupt()          # stop after the token currently in flight
session.push_input(sample2)  # splice new input in between decode rounds, cache stays valid
session.stats                # {"drafted", "accepted", "verify_rounds", "full_depth_forwards"}
```

`StreamingSession` currently requires `model.cfg.controller_mode is False`,
a single ponder loop, batch size 1, and `model.cfg.memory_slots == 0` --
exactly what `get_config("tiny")` and `get_config("nano")` already are. See
the final report for what extending this to `controller_mode` or a
multi-loop ponder core would need from `iridium1.py`.

## Continuous perception (`iridium.runtime.live`)

Computer-use agents poll: screenshot, full re-encode, reply, act, repeat.
`LiveSession` instead keeps **one persistent context** and feeds it
**deltas**:

- `FrameDelta` patchifies each frame and sends only patches whose mean
  absolute change exceeds a threshold, each with its own `(t, y, x)` M-RoPE
  position. A static screen costs 0 tokens; a blinking cursor 1; a scroll what
  scrolled (a mostly static 10-frame test stream: >80% fewer tokens).
- `observe(frame)` appends the delta straight into the KV cache (no
  re-prefill); `hear(text)` adds instructions; `think(n)` decodes text, tool
  calls or actions from the same cache; `step(frame, n)` is one full-duplex
  tick.
- `max_cache_tokens` evicts the oldest frame tokens (never text). Exact for
  attention (positions are baked into cached keys); Gated DeltaNet layers keep
  a constant-size recurrent summary of what was evicted.

Tested: incremental observation equals one full forward over the same tokens
(1e-8, fp64); eviction bounds the cache while decoding continues. Untrained:
using a delta stream well needs training on screen recordings or video.

## Adaptive thinking (`iridium.runtime.thinking`)

The control core's ponder loop has a learned halting head. With
`halt_threshold`, `Iridium1.forward` **stops looping as soon as every token in
the step is confident**, so easy tokens take one loop and hard ones take more.
Skipped loops still need the position in their caches; the halted loop's
entries are copied in (CALM-style state propagation, per loop), so decoding
stays aligned.

`ThinkingBudget` sets the loop cap and threshold per token from:
- **effort** -- `instant` (1 loop), `fast`, `balanced`, `deep`, `max` (always all);
- **speed** -- optional `latency_ms`; running over lowers the threshold (think
  less), running under restores it;
- **difficulty** -- next-token entropy: an unsure step raises the bar for the next.

`generate(..., thinking=ThinkingBudget("balanced", latency_ms=50))`; the
loops each token used are in `result.loops`. The halting head must be trained
(ponder KL + subject losses) for the depth to track real difficulty; the
mechanics are tested with forced-halt and never-halt heads.
