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
