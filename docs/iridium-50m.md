# Iridium-50m

Status: **built and tested on CPU, not trained.** Nothing below says how well it will
work. `python -m iridium battery` is how to find out once it has been trained.

## Shape

| Part | Parameters | Notes |
|---|---:|---|
| Control core | 9.4M | 6 layers, down from chat-36m's 10; every token uses it |
| Superstacks | 40.9M | 4 stacks of 6 layers: maths, chat, code, science; each token uses 1 or 2 |
| Body total | 50.4M | What token budgets are sized from |
| Embeddings | 7.9M | 16k token table and per-layer embeddings; memory, not compute |
| Image and audio input | 2.0M | Encoders only; it cannot generate media |
| Total | 60.3M | |

The core is smaller than chat-36m's, on request. That is a trade: chat-36m made the
core larger because every token shares it. This run tests whether specialist stacks
are worth it, so compare it with chat-36m on the battery rather than assuming.

**Specialisation** comes from each training window's subject label supervising the
router (`subject_routing` loss). `iridium.evaluation.routing` measures whether routing
actually depends on the subject.

**Four specialists.** Stack 1 is maths, 2 is chat (conversation, stories, how-tos,
saying "I don't know"), 3 is code (including tool use), 4 is science. Every text family
belongs to one of them; only the image and audio practice data has no subject.

**How they are made specialists** (training only). A window always visits its
subject's stack. Half the windows visit that stack alone, so it has to do its subject
without help; the other half add the router's best other stack as a partner. No stack
is ever deliberately trained on another stack's subject.

**How to check it.** `python -m iridium specialists --checkpoint MODEL.pt --data data`
measures, for each subject's held-out text, the loss with each stack alone and with the
control core alone. A subject passes when its own stack beats every other stack and the
core. The Kaggle notebook runs this at the end and saves `specialists.json`.

**Evidence so far** is a small CPU experiment only (four synthetic subjects, 1,000
steps, two seeds). Every stack won its own subject in every run, with and without this
scheme. The scheme widened the gap to the best rival (about 0.7 loss per token before,
0.8 with half the windows alone, 1.2 with all of them alone). It did so by making the
other stacks worse at the subject; the specialist's own loss did not change (1.20).
Whether any of this holds at 50M on real text is untested.

**Thinking loops** are trained: 70% of steps run one loop, 20% two and 10% three
(`loop_probs`), and the core is told which pass it is on. That costs about 1.4x the
compute of one-loop training. Loops beyond three at inference are still extrapolation.

## Training budget

5B tokens, about 100 per body parameter (chat-36m had 20). 2.5B distinct tokens are
stored, so most data is seen about twice.

**Speed.** Choose Kaggle's **GPU T4 x2**. Training runs one process per GPU and
averages their gradients every step (`iridium/training/parallel.py`), so the batch and
schedule are the same as on one GPU and each step takes about half as long. Batches are
built on a background thread, and each forward pass carries 16 sequences (the trainer
halves that by itself if a card runs out of memory).

**Measured, first session (before the fixes below):** about 7.8 s per step, roughly
8k tokens/s, with the trainer forced down to 4 sequences per pass. At that speed the
run is about 165 hours. Memory was not leaking (the logged allocated memory stayed at
1.25 GB from step 0 to 762); one step simply needed too much. Two fixes followed, neither
changing what the model computes: masked attention on CUDA no longer passes the GQA flag,
which on several torch versions excluded the memory-efficient kernel and fell back to
storing every head's full score matrix; and the text loss is computed in chunks that
are recomputed in the backward pass, instead of holding 64 MB of logits per sequence
per loop. The log now says once which attention kernel CUDA can use.

The original estimate, also unmeasured: chat-36m measured 68k tokens/s on
one GPU; if this model matches it, two T4s give roughly 120-130k tokens/s, and the
trained loops cost 1.4x, so the run is about 15 hours, or two Kaggle sessions. The
`tok/s` figure on each log line in the first session is the number that tests this.

| Share | Family | Source |
|---:|---|---|
| 16% | science_text | Wikipedia, FineWeb-Edu, Cosmopedia textbooks |
| 6% | science_qa | Natural Questions (short answers), OpenStax-seeded Cosmopedia, ARC |
| 9% | math_text | FineMath, OpenWebMath |
| 5% | math_qa | OpenR1 solutions, GSM8K |
| 4% | arithmetic_drill | Generated, exact |
| 12% | writing_text | Cosmopedia stories and how-tos |
| 7% | writing_qa | Story requests answered with stories |
| 5% | code_qa | Self-OSS-Instruct, Tested-22k, python-codes-25k, MBPP |
| 6% | code_exec | Generated programs; tests run them |
| 15% | agent_episode | Generated using the real file tools |
| 11% | chat_general | SmolTalk, OpenAssistant, Dolly |
| 2% | unknowable | Generated; teaches saying it does not know |
| 2% | image and audio foundation | Generated shapes and tones |

Openly licensed code data is scarce: about 40M distinct tokens, so `code_qa` repeats
roughly six times at this budget, past the ~4 where repeats stop helping. `nampdn-ai/tiny-codes`
would fix that, but it is gated and cannot be downloaded unattended.

The sources' licences are recorded in `iridium/data/chat_corpus.py` and `text_corpus.py`
and copied into every checkpoint. Several require attribution or share-alike, and
SmolTalk's Llama outputs add naming conditions, so check them before any public release.

## Tools

```bash
python -m iridium agent --checkpoint MODEL.pt --workspace ./sandbox
python -m iridium agent --checkpoint MODEL.pt --workspace ./sandbox --without web python
```

| Group | Tools |
|---|---|
| `files` | `list_dir`, `read_file`, `write_file`, `edit_file`, `search_files` |
| `python` | `run_python` |
| `calc` | `calculator` |
| `web` | `web_search`, `web_fetch` |
| `fun` | `roll_dice`, `remember`, `recall` |

All groups are on by default and run without asking. In Python, use
`iridium.runtime.agent_tools.build_registry(workspace, groups=...)`, or register your
own `Tool`.

What the limits really are:

- **File tools** cannot leave the workspace, including through `..`, absolute paths
  or symlinks. Tests cover each of these.
- **`run_python` is a guard rail, not a security boundary.** An audit hook blocks file
  access outside the workspace, subprocesses and sockets, which stops accidents, but
  Python code can defeat it deliberately. On Windows there are no CPU or memory limits
  either, only the timeout. Use a container or VM for anything you do not trust.
- **`web_fetch`** only reaches public addresses, including after redirects.
- **`web_search`** reads DuckDuckGo's HTML page, which may rate-limit or change. Pass your
  own `search_backend` for anything dependable.
- **Prompt injection is not solved.** Pages are marked as data and training includes
  ignoring injected orders, but a small model will not do that reliably. With `web` and
  `python` both on, a hostile page could steer what happens inside the workspace.

## What to expect

Realistic at this size, if training goes well: short tool loops like those it practised,
small Python functions, simple stories and explanations, and sums within the drilled
ranges. Not realistic: planning across many steps, large codebases, or reliable facts.
Check anything important.

Image and audio are a foundation. The generated shapes and tones give the encoders real
signal, but this will not describe photographs or transcribe speech.

## Run it

Kaggle: `notebooks/iridium_studio_kaggle.ipynb`, or `docs/kaggle-agent-runbook.md` for an agent.
Anywhere else:

```bash
python -m iridium data prepare --preset iridium-50m
python -m iridium train --preset iridium-50m --data data --resume auto   # uses every GPU; --gpus 1 for one
python -m iridium battery --checkpoint runs/iridium-50m/iridium-50m-final.pt
```

## The next run: `iridium-50m-next`

The live run keeps training as it is. The next one starts **fresh**: it adds Canon
layers and switches optimizer, so iridium-50m's checkpoints do not load into it. It
reads the same prepared shards. In the Kaggle notebook, set `PRESET = 'iridium-50m-next'`.

| Change | Why | Evidence and caveat |
|---|---|---|
| NorMuon on hidden matrices (Muon + Polar Express coefficients + per-neuron normalisation), weight decay 0.1 | fewer tokens to the same loss | ~1.4x vs a *tuned* AdamW at 130M, 1.1x at 1.2B (Wen et al., ICLR 2026); NorMuon ~6% fewer steps than Muon at 124M (arXiv 2510.05491). In the 2M ablation below, Muon beat AdamW by ~0.15 loss at matched learning rates and NorMuon beat Muon. |
| Peak learning rate 3e-3 (was 1.5e-3) | both optimizers were under-driven | In the ablation, AdamW improved from 1.5e-3 to 1.2e-2 and Muon to at least 2.4e-2. Toy-scale optima overshoot; 3e-3 is a hedge. Sweep it. |
| Canon layers A and C in the core | cheap local mixing (Physics of LMs 4.1) | Ablation: 1.081 vs 1.315 over two seeds, for ~12k parameters at 50M; byte-level text flatters it. |
| Same-subject sequence packing | ~17% of tokens were padding | Packing is standard (Krell et al., arXiv 2107.02027). No document mask: items in a row can attend to earlier items, as in GPT-style pretraining. Rows keep one subject so subject forcing still holds. |
| Decay-phase mixture (`_IRIDIUM50_ANNEAL`) | finish on the most targeted data | OLMo 2's Dolmino mid-training (arXiv 2501.00656), Llama 3, SmolLM2 (arXiv 2502.02737). Magnitude at 50M unknown; keep watching the general-text bpb for forgetting. |

**Learning rate.** Nothing has swept it at 50M. Wen et al. found an untuned learning
rate alone can cost up to 2x, more than the optimizer choice, and the ablation below
agrees: at 3e-3 Muon *lost* to AdamW, at 1.2e-2 it won by 0.15. A sweep that fits one
Kaggle session: three short runs with the full schedule compressed into 600 steps, then
compare the last `val_*_bpb`:

```
for LR in 1.5e-3 3e-3 6e-3; do
  python -m iridium train --preset iridium-50m-next --steps 600 --rounds 3 \
      --lr $LR --out sweep/lr$LR --data /path/to/data
done
```

Short-run optima drift toward higher rates than the full run wants, so if two are
close, take the lower.

**Teacher-written subject text** (per-stack distillation). Generate, then shard
into the prepared data, then add the family to a mixture:

```
export TEACHER_API_KEY=...        # from your provider; never commit it
python -m iridium teacher generate --subject math --n 2000 --out teacher/math.jsonl \
    --base-url https://api.groq.com/openai/v1 --model <model-id> --sleep 2
python -m iridium teacher shard --in teacher/*.jsonl --data data/iridium-50m --preset iridium-50m
```

Then add e.g. `"math_teacher": 0.03` to the mixture (and the anneal mixture). Check the
provider's and the model's terms on training with outputs first; `teacher_manifest.json`
records which models and hosts wrote each family. Chat samples are plain text with
`User:`/`Assistant:` labels, not the chat control tokens.

**Not done here** (see docs/research-2026-11.md): logit distillation (the 16k
vocabulary matches no public teacher, and caching top-k logits for 5B tokens is ~320 GB),
multi-token prediction (reported to hurt small models), nGPT and differential attention.


## Tiny-model ablations (CPU, 20 Nov 2026)

These runs used a 2.1M-parameter Iridium-1 with the same architecture as
iridium-50m: d = 128, 2 core layers and 4 subject stacks of 2 layers, routed
top-2. It trained for 300 steps on TinyStories, byte-level, at batch 16 × 256,
with the WSD schedule. The score is held-out text loss per byte, lower is
better. Two seeds differ by about 0.02–0.06, so smaller gaps are noise.
Reproduce with `experiments/tiny_ablation.py`.

| Run | Loss (mean of seeds) | Seeds |
|---|---|---|
| **NorMuon, lr 1.2e-2, Canon** (iridium-50m-next's recipe, at the toy's own best rate) | **0.935** | 2 |
| AdamW, lr 1.2e-2, Canon | 0.977 | 1 |
| Muon, lr 2.4e-2 | 0.989 | 1 |
| NorMuon, lr 1.2e-2 | 0.990 | 2 |
| Muon, lr 1.2e-2 | 1.028 | 1 |
| AdamW, lr 3e-3, Canon | 1.081 | 2 |
| Muon, lr 6e-3 | 1.135 | 1 |
| AdamW, lr 1.2e-2 | 1.181 | 2 |
| AdamW, lr 6e-3 | 1.226 | 1 |
| AdamW, lr 3e-3, loops 1/2/3 at 0.7/0.2/0.1 | 1.301 | 2 |
| same, ponder prior on the batch mean | 1.309 | 1 |
| **AdamW, lr 3e-3 (the original recipe)** | **1.315** | 2 |
| AdamW, lr 3e-3, gated attention (head) | 1.335 | 2 |
| NorMuon, lr 3e-3 | 1.339 | 2 |
| AdamW, lr 3e-3, linear decay to zero | 1.356 | 2 |
| Polar Express Muon, lr 3e-3 | 1.371 | 2 |
| Muon, lr 3e-3 | 1.403 | 2 |
| Muon, lr 3e-3, weight decay 0.01 | 1.438 | 1 |
| AdamW, lr 1.5e-3 | 1.471 | 1 |

What this supports:

* **The learning rate was the biggest lever.** Both optimizers improved from
  1.5e-3 up to 1.2e-2. At the AdamW-matched 3e-3, Muon looked worse than AdamW.
  At matched higher rates, Muon won by about 0.15. NorMuon beat plain Muon and
  Polar Express Muon at every rate tried.
* **Canon layers help with either optimizer.** They lowered loss by 0.23 with
  AdamW at 3e-3, 0.20 at 1.2e-2, and 0.06 on top of NorMuon. They add about
  2k parameters here and about 12k at 50M. Byte-level text flatters them,
  because neighbouring bytes form words. Expect less on 16k-token text.
* **No measurable effect at this length:**
  - gated attention, whose paper's gain is stability at high learning rates;
  - looping;
  - the batch-mean ponder prior.
  
  All three stay opt-in.
* **Linear decay to zero was worse here.** Its 60-step decay is far shorter
  than the long runs where it was shown to win. It stays opt-in.

What it does not show: anything at 50M, at 16k tokens, or over 76k steps.
Toy-scale optimal learning rates overshoot what long runs want, so
iridium-50m-next uses 3e-3, not 1.2e-2. Run the sweep above before
committing a session to it.
