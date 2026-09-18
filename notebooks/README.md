# Iridium-1 Studio

Four notebooks. One of them is the old single-rung trainer; the other three are
the same studio built for three different free services, generated from one
source (`build_notebooks.py`) so they cannot drift apart.

| notebook | service | open |
|---|---|---|
| `iridium_studio.ipynb` | Google Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sporadicstudiosind-cloud/test/blob/claude/gallant-faraday-lhycva/notebooks/iridium_studio.ipynb) |
| `iridium_studio_kaggle.ipynb` | Kaggle Notebooks | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/sporadicstudiosind-cloud/test/blob/claude/gallant-faraday-lhycva/notebooks/iridium_studio_kaggle.ipynb) |
| `iridium_studio_jupyter.ipynb` | any Jupyter host — Lightning AI, SageMaker Studio Lab, Paperspace, RunPod, or your own box | open it |
| `train_iridium_colab.ipynb` | Colab, the earlier fixed-rung trainer | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sporadicstudiosind-cloud/test/blob/claude/gallant-faraday-lhycva/notebooks/train_iridium_colab.ipynb) |

Run all. Ten sections: detect the hardware, design a model, check it fits, stream
licensed data, verify the architecture, train, grade, inspect the routing, save,
chat.

## The ladder

Any of these is one dropdown away, and every count below is the formula's, not a
nameplate — `tests/unit/test_presets.py` recomputes the transformer body from
retyped arithmetic and demands the two agree to the parameter.

| preset | parameters | core | superstacks | d_model | active/token |
|---|---:|---:|---|---:|---:|
| `50m` | 49,713,034 | 6L | 2 × 9L | 384 | 31.1 M |
| `100m` | 99,841,306 | 6L | 4 × 8L | 512 | 55.9 M |
| `500m` | 495,705,494 | 10L | 2 × 15L | 1024 | 301.0 M |
| `1b` | 1,007,202,723 | 12L | 3 × 14L | 1280 | 728.4 M |
| `8b` | 7,894,763,402 | 24L | 4 × 36L | 2048 | 4.46 G |
| `16b` | 15,889,000,377 | 28L | 5 × 38L | 2560 | 7.49 G |
| `24b` | 23,894,349,882 | 32L | 4 × 48L | 3072 | 13.5 G |
| `100b` | 100,151,262,631 | 48L | 5 × 60L | 5120 | 47.8 G |
| `200b` | 199,732,231,081 | 56L | 5 × 86L | 6144 | 92.9 G |
| `1t` | 998,290,891,190 | 64L | 8 × 103L | 10240 | 301.8 G |

**Describing a geometry and being able to train it are different things.** Every
rung above is a real configuration whose cost the notebook computes exactly.
Section 3 tells you, from the memory your device actually reports, whether it
trains there, and names a cheaper optimizer strategy when the answer is no. On a
free 16 GB card, with a quarter of it reserved for activations: `50m`, `100m`
and `500m` train under plain AdamW fp32; `1b` needs 8-bit moments or Adafactor;
`8b` and above need sharding across devices, whatever the strategy.

Set `PRESET = 'custom'` and every knob is yours: `D_MODEL`, `CORE_LAYERS`,
`N_SUPERSTACKS`, `SUPERSTACK_LAYERS`, `D_HEAD`, `N_KV_HEADS`, `TOP_K`,
`MAX_LOOPS`, `MIN_DEPTH`, `CROSS_STRIDE`, `SPECTRAL_STACKS`, `VOCAB_SIZE`,
`MAX_SEQ_LEN`.

Two constraints the builder enforces rather than trusting you to remember:

- **`TOP_K` is clamped to `N_SUPERSTACKS - 1`.** Routing to every bank is not
  routing — the gate becomes decorative, the balance loss is satisfied by
  construction, and what you have is a dense ensemble wearing a router.
- **`VOCAB_SIZE` defaults to 384.** The text codec is byte-level: it emits raw
  UTF-8 bytes shifted past the control ids, so 256 values plus control room is
  the entire reachable vocabulary. A 32,000-row embedding is not extra capacity,
  it is rows that never receive a gradient and a softmax over classes the data
  cannot produce. This ladder previously carried tokenizer-sized vocabularies;
  at the `1t` rung that was 1.3 B dead parameters, and they now buy depth.

## The data

Real text, streamed, with its licence recorded and carried into the run
manifest — because a model that cannot say what it was trained on cannot honour
a share-alike obligation.

| source | licence | what it obliges |
|---|---|---|
| Project Gutenberg (`manu/project_gutenberg`, `en`) | public domain (US) | nothing on the works; the header PG prepends carries its own terms and is stripped |
| Wikipedia (`wikimedia/wikipedia`, `20231101.en`) | CC BY-SA 4.0 | attribution **and share-alike on derivatives** |
| FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, `sample-10BT`) | ODC-By 1.0 | attribution to the dataset |

Recording provenance is the auditable part. It does not make the corpora clean:
public-domain books are old and carry the assumptions of their period, an
encyclopedia has documented systemic gaps in coverage and authorship, and a web
crawl filtered for "educational" is still a web crawl.

Three details that were wrong in the obvious implementation and are worth
knowing if you change the mixture:

- A shuffle buffer is counted in **rows** and paid for in **bytes**. A Gutenberg
  row is a whole book; a thousand-row buffer downloads a gigabyte before
  yielding a single document. Each source sets its own.
- Taking every consecutive window of a document does **not** respect the mixture
  weights. One book is four thousand windows, so "40% Gutenberg" delivers four
  thousand items from one author before touching anything else. Windows are
  capped per document and spread across it.
- Three live HTTP streams interleaved is what stalls behind a proxy or a
  container's connection limits, with no error to read. Sources are drained one
  at a time and shuffled afterwards.
- **Seeding the splits differently does not separate them.** A streaming shuffle
  permutes within a buffer while traversing the file in order, so every split
  still sees the same early documents — measured at 10% of an 80-document test
  set drawn from training documents. Split membership is instead a hash of the
  document's own bytes, which makes it a property of the document: the same book
  lands in the same split under any seed, on any machine, a month later.
  Otherwise bits per byte measures memorisation and reports it as
  generalisation.

`realised_mixture()` reports what the items actually contain, so you can check
the mixture rather than trust it. Set `USE_REAL_TEXT = False` for synthetic
families only — no network, and every answer exactly checkable.

## The grading

Free-running generation scored against **independent computation** — Manning's
law in closed form, the spectral solver, the scene environment's own goal
predicate — each reported beside the baseline a model earns by ignoring its
input entirely, and again on an extrapolation split drawn from outside the
training band. A number without its baseline says nothing: `false_premise` is
roughly balanced between true and false claims, so a model that answers "true"
every time scores about 0.52, and an accuracy of 0.5 is worth exactly nothing.

Section 8 asks the question the balance loss cannot: **did the bank specialise,
or merely balance?** It reports `I(family; stack)` in nats against the maximum
`ln(n_stacks)`. Balanced dispatch with near-zero mutual information means the
router is spreading load without learning what the stacks are for — which is a
real finding, not a failure to hide.

## Before you pick a free T4

A free Colab T4 is Turing and **has no bf16**. For a routed model that matters
more than usual: the gating softmax that selects a superstack and the attention
logits both sit exactly where fp16's exponent overflows, and when they do the
router collapses onto one stack and the failure reads as a bad hyperparameter
rather than a numerics bug. The notebooks run **fp32 on a T4** and bf16 only on
Ampere or newer.

## TPUs

`torch_xla` compiles static shapes. This router dispatches a **variable number
of tokens** to each superstack every step, which forces a recompile per shape or
padding to a fixed capacity. So on a TPU VM the notebook defaults to the **host
CPU** — on a Colab v5e-1 that is 48 GB of RAM, genuinely useful for the larger
presets, just slow. `FORCE_XLA = True` drives the accelerator anyway. This is a
real consequence of dynamic routing on XLA, not a missing feature.

## AMD

ROCm needs no code changes — HIP is reached through the `torch.cuda` API. You
need the right wheel and the device nodes; see [`../docs/gpu.md`](../docs/gpu.md).

## Service notes

- **Colab** — port forwarding via `eval_js`, so the chat UI opens in the
  notebook. Free tier is a T4 with a session limit.
- **Kaggle** — 30 GPU-hours a week, the best free quota here; P100 16 GB or
  2× T4. Turn **Settings → Internet → On** or nothing streams. No public port
  forwarding, so chat goes through the `ask()` function rather than the UI.
- **Lightning AI / SageMaker Studio Lab / Paperspace / RunPod** — the Jupyter
  notebook. Studio Lab and Lightning forward ports from their sidebar.
- **Hugging Face Jobs and ZeroGPU** both require a Pro subscription for GPU, so
  they are not a free option.

## Regenerating

```bash
python notebooks/build_notebooks.py
```

Edit `build_notebooks.py`, never the `.ipynb` files — all three are generated
from it, and a hand-edit to one is a divergence waiting to be overwritten.
