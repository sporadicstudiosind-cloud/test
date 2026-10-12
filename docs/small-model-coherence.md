# Why the small rungs babble

A model at the `nano100m` or `test1b` rung produces incoherent text. The
natural conclusion is that something is wrong with the architecture — a routed,
multimodal, ponder-looping design has a lot of surface area to be wrong on, and
each mechanism is a plausible suspect.

Almost none of it is architectural. This document records what was measured,
in the order the evidence should be read, because the order matters: three of
these findings make the others unmeasurable until they are fixed.

Run the numbers yourself before trusting this page:

```
python -m iridium.training.budget --rung test1b
```

## 1. The default mixture contains no natural language

`training/datasets.py:DEFAULT_MIXTURE` is five synthetic structured families —
`channel_depth`, `channel_intervention`, `field_rollout`, `scene_goal`,
`false_premise`. There is no `text_lm` weight and no `chat` weight, and
`build_corpus` filters out families with zero weight before generating
anything. The command in the `phase1_pretrain` docstring therefore trains a
language model on no language.

This is not a subtle failure, but it is an easy one to ship, because the two
halves live in different modules: the mixture is defined in one file and the
complaint ("it does not write properly") arrives from another. Nothing computed
the relationship between them.

The synthetic families are not wrong — they are what teaches the physics,
routing and action machinery, and they are exactly checkable in a way prose
never is. They are simply not a substitute for text when the question is about
text.

**Fix:** put `text_lm` and `chat` weight in the mixture. `budget.audit` now
refuses to be quiet about a mixture without them.

## 2. The corpus is smaller than the model by several orders of magnitude

The default is 12,000 items of 256 bytes: **3.07 MB**, or about 3.07M
byte-level tokens.

| rung | parameters | tokens/parameter | short of 20 tok/param by |
| --- | --- | --- | --- |
| `nano` | 34.0 M | 0.090 | 222x |
| `nano100m` | 104.1 M | 0.030 | 678x |
| `test1b` | 1.00 B | 0.003 | 6,522x |

The 20 tokens/parameter figure is Chinchilla's compute-optimal ratio (Hoffmann
et al., 2022). It is a *floor*, and it answers a question nobody is asking
here: "given fixed compute, how should I split it between model size and data".
When the size is fixed for deployment reasons — which is the whole premise of a
small rung — the right move is to train far past compute-optimal. The small
open models that are actually coherent sit somewhere between 1,000 and 12,000
tokens per parameter. Against that comparison `test1b`'s corpus is short by a
factor of roughly 326,000.

A 1B-parameter model given 3 MB of text has vastly more capacity than the data
can constrain. It will memorise. That is not a bug in the optimizer.

**Fix:** there is no code fix. This is a data and compute decision, and it is
the decision that dominates every other item on this page.

## 3. Byte-level text costs about four times its worth

`CodecConfig.vocab_size` was 384 and text was encoded as raw bytes.

Byte-level has real advantages, and the original docstring named them honestly:
no vocabulary to train, nothing to go stale, no silent mismatch between a
tokenizer trained on one corpus and a model trained on another, and losslessness
on arbitrary input. All true.

The cost is that a byte model spends roughly one token per character where a
subword model spends one per four. At a fixed `max_seq_len` and a fixed step
budget that is a ~4x difference in how much text the model sees, and a ~4x
difference in how far its context actually reaches in words. The capacity that
buys orthography is capacity not spent on meaning — and at these rungs capacity
is the binding constraint.

**Fix:** `iridium/data/tokenizer.py` (a dependency-free byte-level BPE),
`IridiumConfig.text_vocab_size`, and `training/tokenizer_bridge.py` so exactly
one function decides which vocabulary a config implies. `text_vocab_size=0`
keeps the byte path.

Note the trap this creates and the guard against it: a train/serve tokenizer
mismatch does not raise. The ids stay in range, the forward pass runs, and the
model emits fluent nonsense. `tokenizer_manifest` records the vocabulary in the
run manifest, and reports `fell_back` when a requested vocabulary was not the
one actually used.

## 4. Training windows claimed to be document starts

`text_items` cut random 256-byte windows out of documents and wrapped each in
BOS/EOS. A window beginning mid-word, wrapped in a begin-of-sequence marker,
teaches the model that documents begin mid-word — and every fourth token of
training was such a claim.

**Fix:** document packing. Documents are concatenated with EOS separators and
contiguous windows are sliced from the stream; BOS marks only real starts. The
old behaviour is still available as `pack=False` for comparison.

## 5. Weight decay reached tensors it is wrong for

Decay is a prior that a *matrix* should be small. Applied uniformly it also hit:

* **RMSNorm gains**, which multiply the residual stream — so decay here is a
  slow global attenuation of the network, not regularisation;
* **embedding rows**, which are updated only when their token appears but
  decayed every step regardless, so the rarest tokens are pulled toward zero
  most often relative to their useful updates;
* **halting biases**, which encode a calibrated stopping prior. Decaying a
  `-2.0` halting bias toward zero decays the ponder loop toward "always stop",
  which presents as a collapsed loop and reads as an architecture fault.

**Fix:** `runtime.memory.decay_groups` splits by tensor rank, the rule GPT-3,
Chinchilla and Llama all converged on. Rank rather than a name match, because a
name match silently stops working when a module is renamed and the failure is
invisible.

## 6. The output softmax had no z-loss

Cross-entropy is invariant to adding a constant to every logit, so nothing in
the objective prevents the whole logit vector from drifting. In bf16 — 8 bits
of mantissa — a vector centred on +30 has about 0.25 of absolute resolution,
which quantises the *differences* between logits into noise. The differences
are the only part that carries information.

At vocab 384 this is survivable. At the 32k a subword vocabulary implies it is
close to mandatory, and that is exactly the moment nobody thinks to add it.

**Fix:** `text_z` loss term, weight 1e-4 (PaLM's value), reported separately so
a rising z-loss is visible before it becomes a plateau.

## 7. Warmup was too short for a routed model

Fifty steps. The macro-router's gate decides which superstack sees which token;
at a high learning rate in the first hundred steps it will commit to a partition
before any stack has learned anything that would justify one. The commitment is
self-reinforcing — unvisited stacks get no gradient, so they never become worth
visiting — and the resulting loss curve looks mediocre rather than broken.

**Fix:** `TrainConfig.warmup_ratio`, defaulting to 2% of the run, with the
explicit `warmup` still winning when it is larger.

## What was *not* wrong

Worth recording, because these were checked:

* Incremental decoding matches teacher forcing. The KV-parity gate holds.
* Attention already routed through `scaled_dot_product_attention` rather than a
  hand-rolled score matrix, so the fused-kernel portability story was already
  right.
* Checkpoints already load with `map_location="cpu"`, so there is no device
  string baked into saved state.

## Reading order

Items 1 and 2 dominate. Until the mixture contains prose and the corpus is
three to four orders of magnitude larger, items 3–7 are unmeasurable: every one
of them is a change whose effect shows up as a better loss on data that is not
currently there. Fixing 3–7 first and re-running the same recipe will produce
the same babble, and the reasonable conclusion from that experiment — that the
fixes did nothing — would be wrong.
