# Iridium 1.0 Studio notebooks

Five notebooks. Three are the same studio workflow generated from one source
(`build_notebooks.py`) for three different free services, so they cannot drift
apart; one is a short redirect for the old fixed-preset Colab notebook; one is
a separate TPU builder. Edit `build_notebooks.py`, never a `.ipynb` file
directly -- `tests/unit/test_notebooks.py` fails the build the moment a
generated notebook no longer matches what the script would produce.

| notebook | service | default preset | open |
|---|---|---|---|
| `iridium_studio.ipynb` | Google Colab (free T4) | `chat-34m` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sporadicstudiosind-cloud/test/blob/claude/gallant-faraday-lhycva/notebooks/iridium_studio.ipynb) |
| `iridium_studio_kaggle.ipynb` | Kaggle Notebooks (P100 or 2xT4) | `chat-100m` | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/sporadicstudiosind-cloud/test/blob/claude/gallant-faraday-lhycva/notebooks/iridium_studio_kaggle.ipynb) |
| `iridium_studio_jupyter.ipynb` | any Jupyter host -- Lightning AI, SageMaker Studio Lab, Paperspace, RunPod, your own box | `chat-100m` | open it |
| `train_iridium_colab.ipynb` | Colab | -- | redirects to `iridium_studio.ipynb` (kept so the old badge/link still works) |
| `train_iridium_tpu_colab.ipynb` | Colab/Kaggle free TPU v5e-1 | `chat-100m`, with geometry overrides | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sporadicstudiosind-cloud/test/blob/claude/gallant-faraday-lhycva/notebooks/train_iridium_tpu_colab.ipynb) |

## What every studio notebook actually does

Detect hardware -> pick a preset -> `dry_run` (build the model on the meta
device, verify its parameter count against the formula, print the
data-budget audit and free-tier time estimate) -> train with `train_preset`
in rounds of fresh data, saving a checkpoint every round -> chat with the
result via `ChatSession` (or `python -m iridium chat --checkpoint`) ->
optionally stage a second preset (`tools-100m`) initialised from the chat
checkpoint.

**The model is untrained until you run the training cell.** Nothing in `Run
All` up to that point produces anything but random weights and a cost report.

**What a free session buys you** is not a guess: section 3 of every studio
notebook prints `iridium.presets.preset_table()` and the `dry_run` cost/audit
report before training starts, both computed by `iridium.presets.estimate_hours`
and `iridium.training.budget.audit` from the config's own FLOP formula --
arithmetic, not a measurement, and optimistic (30% of published peak assumed;
see the warning in `iridium/presets.py`).

**Network and `datasets` are required** for the text/chat/tokenizer path:
`iridium.training.datasets.build_corpus` streams real text over the network,
and a preset's subword tokenizer is trained from that same stream the first
time it runs (`iridium.training.tokenizer_bridge`). No network, or no
`datasets` package, and `train_preset` refuses to fall back to a silently
mismatched byte-level vocabulary -- it raises, rather than training something
that looks fine and serves nonsense.

## The private-repo clone

The repository is private until Iridium 1.0 ships. Section 1 of every
notebook reads a GitHub token from the host's own secret store and never
prints or writes it to disk:

- **Colab**: the key icon in the left sidebar -> Secrets -> add a secret named
  `GITHUB_TOKEN` (a fine-grained personal access token scoped to read this
  repo) -> toggle notebook access on for it.
- **Kaggle**: Add-ons -> Secrets -> add `GITHUB_TOKEN` the same way -> attach
  it to the notebook.
- **Plain Jupyter**: set the `GITHUB_TOKEN` environment variable before
  starting Jupyter, or just launch the notebook from inside an already-cloned
  checkout (the clone cell detects that and skips cloning).

Without a token the cell prints the instructions above and falls back to an
unauthenticated clone, which only succeeds once the repository is public. The
notebooks clone `sporadicstudiosind-cloud/test` at branch
`claude/gallant-faraday-lhycva` -- the release branch until it merges. Update
that constant in `build_notebooks.py` (`RELEASE_BRANCH`) once it does, and
regenerate.

## The presets

Every preset is a complete recipe -- model config, data mixture, tokenizer
size, schedule, optimizer, step and batch budget -- ordered by Iridium 1.0's
own priorities: talking and reasoning first, then tool use, then
omnimodality, then physics/STEM, then world model.

```
preset        pri       params   tokens    T4 h  P100 h  v5e h  status
----------------------------------------------------------------------
chat-34m        1   36,038,426     328M     4.1     3.6    0.2  verified in theory
chat-100m       1  108,304,153     655M    21.1    18.4    0.9  verified in theory
tools-100m      2  108,304,153     655M    21.1    18.4    0.9  verified in theory
omni-100m       3  111,586,585     655M    21.1    18.4    0.9  verified in theory
stem-100m       4  108,304,153     492M    15.8    13.8    0.6  verified in theory
world-100m      5  111,597,342     328M    10.5     9.2    0.4  verified in theory; no camera-posed training data yet
modern-744m     -  743,752,454   26214M  7055.7  6145.3  290.1  verified in theory
(hours assume 30% of published peak; optimistic)
```

Regenerate this table yourself with `python -m iridium presets` -- it is
computed from `iridium/presets.py`, not transcribed, so it will drift from
this file before it drifts from the code. **`verified in theory`** means the
configuration builds, its parameter count matches the formula, and the
invariant test suite holds for it. It does not mean the preset has been
trained or that training it produces a good model -- nothing at these recipes
has been trained yet.

`modern-744m` and `8b` are costed, not free-tier trainable (see the `T4 h` /
`P100 h` / `v5e h` columns above); they exist so the cost is visible, not as
something to run in these notebooks.

## Resuming after a session ends

`train_preset` saves a checkpoint every round: `runs/<preset>/<preset>-round0.pt`,
`<preset>-round1.pt`, ..., and a final `<preset>-final.pt`. A free session can end without
warning, so:

- **Colab**: mount Drive in section 4 (the notebook does this itself if you
  let it) and checkpoints land under `/content/drive/MyDrive/iridium-runs`,
  which survives a runtime reset.
- **Kaggle**: checkpoints land under `/kaggle/working`, which persists only if
  you *Save Version* (commit the notebook) before the session ends -- an
  interactive-only session that is never committed loses them.
- **Jupyter**: checkpoints land under `runs/` on whatever storage that host
  gives you; point `OUT_DIR` at your own persistent volume if the default
  location does not survive a restart.

To continue: set `RESUME_FROM` in the training cell to the last `roundN.pt`
you have and rerun the notebook (cloning and installing are idempotent).
`train_preset` restores the optimizer, step count and learning-rate schedule
-- the schedule spans the whole run -- but starts a new data shuffle, so a
resumed run is not bit-for-bit replay of the interrupted round.

## Chatting with a checkpoint

Inside the notebook, `ChatSession` from `iridium.runtime.chat` holds a
conversation across turns. Outside it, the same checkpoint works from a
terminal:

```bash
python -m iridium chat --checkpoint runs/chat-34m/chat-34m-final.pt --device auto
```

## Tool use (`tools-100m`)

The optional second stage in each studio notebook trains `tools-100m`,
initialised (`--init`) from the chat checkpoint rather than from scratch. It
depends on `iridium.runtime.tools`, which is being built alongside these
notebooks by a separate track of work. The cell checks
`importlib.util.find_spec('iridium.runtime.tools')` first and prints a plain
message instead of failing partway through a training run if that module has
not landed in your checkout yet.

## The TPU notebook

`train_iridium_tpu_colab.ipynb` targets the free Colab/Kaggle **TPU v5e-1**
runtime instead of a GPU. It starts from a preset, then exposes the raw
geometry -- `CORE_LAYERS`, `SUPERSTACK_LAYERS`, `SUPERSTACK_COUNT` -- as
overrides, because the TPU host's extra memory is exactly the room to try a
deeper or wider variant of a preset before it has earned a name. It calls the
same `train_preset`/`Trainer` the other notebooks call with `device='xla'`
and does not reimplement any XLA training mechanics itself -- that support
(`xm.optimizer_step`, `xm.save`, a `None` RNG generator on XLA) lives in the
library (`iridium/runtime/device.py`, `iridium/training/trainer.py`), not in
this notebook.

This router dispatches a *variable* number of tokens to each superstack per
step, which forces `torch_xla` to recompile per shape (or pad to a fixed
capacity) -- a real consequence of dynamic routing on a compiled accelerator,
not a bug in this notebook. Training is gated behind an explicit
`CONFIRM_TRAIN = True` after the dry-run cell, because a free TPU session is
quota you cannot get back.

This notebook was ported from an open PR (`codex/create-colab-with-48gb-ram`)
whose base had gone stale against this branch; its content was rebuilt here
against the current `presets`/`run_preset` API rather than merged directly.

## Regenerating

```bash
python notebooks/build_notebooks.py
```

`tests/unit/test_notebooks.py` checks (statically, no network or GPU needed):
every notebook is valid nbformat JSON; every code cell compiles as Python
after stripping magics/shell lines; every `iridium` import resolves against
the current package; every preset name and CLI subcommand referenced actually
exists; and that `build_notebooks.py` reproduces every generated `.ipynb`
byte-for-byte, so a hand-edit or a stale copy is caught immediately.
`tests/unit/test_notebook_names.py` additionally checks that no cell in a
studio notebook uses a name (like `DEVICE`) before an earlier, always-run cell
defines it.
