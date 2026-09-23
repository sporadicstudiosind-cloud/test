"""Generate every notebook in this directory; this script only writes JSON,
it never trains or tests anything itself.

One source, several targets. The three studio notebooks (Colab, Kaggle, any
Jupyter host) share the same flow -- clone, detect hardware, pick a preset,
dry-run, train in rounds, chat, optionally stage tool-use -- and differ only
in how each host authenticates, persists checkpoints and reports hardware.
Writing that flow once here is the point: a fix or a wording change applies to
all three the next time this script runs, instead of drifting across four
hand-edited files.

``train_iridium_colab.ipynb`` is a short redirect: the old fixed single-preset
Colab notebook is gone as a separate workflow, folded into the Colab studio
notebook, and this file just says so and links to it (the badge keeps working
because the file still exists at the same path).

``train_iridium_tpu_colab.ipynb`` is a separate flow, not a studio variant: it
targets the free Colab/Kaggle TPU v5e-1, exposes the raw geometry knobs
(core layers, superstack layers, superstack count) instead of a fixed preset
ladder, and gates training behind an explicit confirmation because a TPU
session is quota you cannot get back.

Regenerate with::

    python notebooks/build_notebooks.py

Edit this file, never the ``.ipynb`` files directly --
``tests/unit/test_notebooks.py`` fails the build if a generated notebook no
longer matches what this script would produce.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent

REPO_OWNER = "sporadicstudiosind-cloud"
REPO_NAME = "test"
RELEASE_BRANCH = "claude/gallant-faraday-lhycva"

GENERATED_STUDIO = {
    "colab": "iridium_studio.ipynb",
    "kaggle": "iridium_studio_kaggle.ipynb",
    "jupyter": "iridium_studio_jupyter.ipynb",
}
GENERATED_OTHER = {
    "colab_redirect": "train_iridium_colab.ipynb",
    "tpu": "train_iridium_tpu_colab.ipynb",
}
TARGETS = {**GENERATED_STUDIO, **GENERATED_OTHER}

COLAB_BADGE = (
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/{owner}/{repo}/blob/{branch}/notebooks/{name})"
).format(owner=REPO_OWNER, repo=REPO_NAME, branch=RELEASE_BRANCH, name="{name}")

KAGGLE_BADGE = (
    "[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]"
    "(https://kaggle.com/kernels/welcome?src=https://github.com/{owner}/{repo}"
    "/blob/{branch}/notebooks/{{name}})"
).format(owner=REPO_OWNER, repo=REPO_NAME, branch=RELEASE_BRANCH)


def cell(kind: str, text: str) -> dict:
    out = {"cell_type": kind, "metadata": {}, "source": textwrap.dedent(text).strip("\n").splitlines(True)}
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


class Builder:
    """Accumulates cells; ``md``/``code`` mirror the old script's helpers."""

    def __init__(self) -> None:
        self.cells: list[dict] = []

    def md(self, text: str) -> None:
        self.cells.append(cell("markdown", text))

    def code(self, text: str) -> None:
        self.cells.append(cell("code", text))


# --------------------------------------------------------------------------
# Shared building blocks
# --------------------------------------------------------------------------

def honesty_banner(b: Builder, title: str, default_preset: str) -> None:
    b.md(f"""
    # {title}

    **The model is untrained until you run the training cell below.** Loading
    this notebook, or running the setup cells, produces random weights and
    nothing else. A checkpoint means something only after a training run has
    actually completed.

    **What a free session buys you** is printed by the dry-run cell (section
    3) before anything trains: it prints `iridium.presets.preset_table()` and
    `iridium.training.run_preset.dry_run()`'s cost/audit report, both computed
    from `estimate_hours` and `iridium.training.budget.audit` -- not measured,
    arithmetic, and optimistic (see the warning in `iridium/presets.py`). Read
    those numbers before starting a long run; they tell you honestly whether
    the chosen preset's step budget fits in the quota you have.

    **Network is required.** The text/chat data path
    (`iridium.training.datasets.build_corpus`) streams real text from the
    Hugging Face `datasets` library, and the preset's subword tokenizer is
    trained from that same stream the first time it runs. No network (or no
    `datasets` package) means `train_preset` refuses to fall back to a
    silently mismatched byte-level vocabulary and raises instead -- see
    `iridium/training/run_preset.py`.

    Default preset here: `{default_preset}`. Change `PRESET` in section 2 to
    train a different one; `python -m iridium presets` (or the table this
    notebook prints) lists every option and what each needs.
    """)


def private_clone_cell(target: str) -> str:
    """The token-read + clone cell, one per host style.

    The token is read from the host's own secret store, passed to git once through
    an environment-scoped auth header, and never printed, put in a URL, or
    written to disk. A public checkout (no
    token, no existing local repo) gets a plain instruction instead of an
    opaque git failure. Each branch is written out in full (rather than
    spliced together from indented fragments) so the generated source has one
    consistent indentation level throughout.
    """
    if target == "colab":
        read_token_block = (
            "token = None\n"
            "try:\n"
            "    from google.colab import userdata\n"
            "    token = userdata.get('GITHUB_TOKEN')\n"
            "except Exception:\n"
            "    token = None\n"
        )
        missing_token_hint = (
            "In Colab: the key icon in the left sidebar, Secrets, add "
            "GITHUB_TOKEN with a fine-grained PAT that can read this "
            "repository, then toggle notebook access on."
        )
    elif target == "kaggle":
        read_token_block = (
            "token = None\n"
            "try:\n"
            "    from kaggle_secrets import UserSecretsClient\n"
            "    token = UserSecretsClient().get_secret('GITHUB_TOKEN')\n"
            "except Exception:\n"
            "    token = None\n"
        )
        missing_token_hint = (
            "In Kaggle: Add-ons, Secrets, add GITHUB_TOKEN with a "
            "fine-grained PAT that can read this repository, then attach it "
            "to this notebook."
        )
    else:
        read_token_block = "token = os.environ.get('GITHUB_TOKEN')\n"
        missing_token_hint = (
            "Set the GITHUB_TOKEN environment variable to a fine-grained PAT "
            "that can read this repository before starting Jupyter, or clone "
            "the repository yourself and launch this notebook from inside it."
        )

    header = f'''
    # This repository is private until Iridium 1.0 ships. This cell never
    # hard-codes or prints a token: it reads one from the host's own secret
    # store (or an env var on a plain Jupyter host) and hands it to git only
    # through an environment-scoped header, so it never lands in .git/config.
    import os, sys, subprocess
    from pathlib import Path
    '''
    footer = f'''
    REPO_OWNER = {REPO_OWNER!r}
    REPO_NAME = {REPO_NAME!r}
    RELEASE_BRANCH = {RELEASE_BRANCH!r}
    ROOT = Path.cwd()

    if (ROOT / 'iridium' / 'presets.py').is_file():
        print('Already inside a checkout of the repository:', ROOT)
    else:
        ROOT = Path.cwd() / REPO_NAME
        if ROOT.is_dir():
            print('Reusing existing checkout at', ROOT)
        elif token:
            # The token travels in an environment-scoped git config header:
            # never in the URL (git would store it in .git/config) and never in
            # the argv (a failed subprocess prints its argv).
            import base64
            basic = base64.b64encode(f'x-access-token:{{token}}'.encode()).decode()
            env = dict(os.environ, GIT_CONFIG_COUNT='1',
                       GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
                       GIT_CONFIG_VALUE_0=f'AUTHORIZATION: basic {{basic}}',
                       GIT_TERMINAL_PROMPT='0')
            done = subprocess.run(['git', 'clone', '--depth', '1', '--branch', RELEASE_BRANCH,
                                   f'https://github.com/{{REPO_OWNER}}/{{REPO_NAME}}.git',
                                   str(ROOT)], env=env, capture_output=True, text=True)
            del env, basic
            if done.returncode != 0:
                raise RuntimeError('git clone failed (exit %d). Check the token can read '
                                   'the repository and the branch exists.' % done.returncode)
            print('Cloned', REPO_OWNER + '/' + REPO_NAME, '@', RELEASE_BRANCH, 'to', ROOT)
        else:
            print('No GITHUB_TOKEN found. {missing_token_hint}')
            print('Trying an unauthenticated clone (works only if the repo is public)...')
            subprocess.run(['git', 'clone', '--depth', '1', '--branch', RELEASE_BRANCH,
                           f'https://github.com/{{REPO_OWNER}}/{{REPO_NAME}}.git', str(ROOT)],
                          check=True)
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    print('working directory:', Path.cwd())
    '''
    return textwrap.dedent(header).strip("\n") + "\n" + read_token_block + textwrap.dedent(footer).rstrip("\n")


def install_cell(target: str) -> str:
    # The ``data`` extra brings ``datasets``, which every preset needs to
    # stream text and train its tokenizer. torch is already on every host.
    return '''
    !pip install -q -e ".[data]"
    '''.rstrip()


def hardware_and_preset_cell(target: str, default_preset: str) -> str:
    return f'''
    import torch
    from iridium.runtime.device import detect
    from iridium.presets import PRESETS, FREE_TIERS, get_preset, preset_table

    info = detect()
    DEVICE = info.device
    print(info.describe())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))

    # {"Colab free tier: a T4 by default." if target == "colab" else
       "Kaggle: P100 or 2xT4 depending on what you selected in Settings." if target == "kaggle" else
       "Any Jupyter host: pick the preset that matches what you actually have."}
    PRESET = {default_preset!r}
    preset = get_preset(PRESET)
    print()
    print(preset_table())
    '''


def dry_run_cell() -> str:
    return '''
    from iridium.training.run_preset import dry_run

    dry_run_result = dry_run(preset)
    if not dry_run_result['match']:
        raise RuntimeError(
            f"parameter count mismatch: built {dry_run_result['parameters_built']:,}, "
            f"formula says {dry_run_result['parameters_formula']:,}. Do not train "
            "until this reconciles -- report it rather than proceeding."
        )
    '''


def persistence_cell(target: str) -> str:
    if target == "colab":
        return '''
        # Round checkpoints survive a disconnected Colab session only if they
        # land on Drive. Mounting is optional; without it OUT_DIR is local to
        # the VM and is deleted when the runtime recycles.
        from pathlib import Path
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            OUT_DIR = Path('/content/drive/MyDrive/iridium-runs')
        except Exception:
            print('Drive not mounted; checkpoints will not survive a runtime restart.')
            OUT_DIR = Path('runs')
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        print('checkpoints ->', OUT_DIR)
        '''
    if target == "kaggle":
        return '''
        # /kaggle/working is preserved as this notebook's Output when you
        # commit the notebook (Save Version); it is not preserved for an
        # interactive-only session that is never committed.
        from pathlib import Path
        OUT_DIR = Path('/kaggle/working/iridium-runs')
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        print('checkpoints ->', OUT_DIR, '(persists once you Save Version)')
        '''
    return '''
    # Plain Jupyter: checkpoints land under the working directory. Point this
    # at whatever storage on this host actually survives a restart.
    from pathlib import Path
    OUT_DIR = Path('runs')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('checkpoints ->', OUT_DIR.resolve())
    '''


def train_cell() -> str:
    return '''
    from iridium.training.run_preset import train_preset

    STEPS = None        # None: use the preset's own step budget
    ROUNDS = None        # None: use the preset's own round count
    RESUME_FROM = ''      # e.g. str(OUT_DIR / preset.name / f'{preset.name}-round2.pt') after a restart

    checkpoint = train_preset(
        preset, steps=STEPS, rounds=ROUNDS, device=DEVICE, out=str(OUT_DIR),
        resume=(RESUME_FROM or None), seed=0,
    )
    print('final checkpoint:', checkpoint)
    '''


def resume_markdown(target: str) -> str:
    where = {
        "colab": "under `OUT_DIR` on Drive, so it is there after a disconnect",
        "kaggle": "under `/kaggle/working`, so it is there only if you Saved a "
                  "Version before the session ended",
        "jupyter": "under `OUT_DIR` on this host's own disk",
    }[target]
    return f"""
    ## 5. Resuming after a session ends

    Every round from section 4 is a checkpoint: `train_preset` saves
    `OUT_DIR/<preset>/<preset>-round0.pt`, `-round1.pt`, ... and a final `<preset>-final.pt`. A
    free session can end mid-run without warning, and each round's file is
    {where}.

    To continue: set `RESUME_FROM` in the training cell above to the last
    `roundN.pt` you have, re-run this notebook from the top (cloning and
    installing are idempotent), and re-run the training cell. `train_preset`
    restores the optimizer, step count and learning-rate schedule -- the
    schedule spans the whole run, so a resumed run is not a fresh one with the
    clock reset. It does start a new data shuffle; resuming is not bit-for-bit
    replay of the interrupted round.
    """


def chat_cell() -> str:
    return '''
    from iridium.training.trainer import load_checkpoint
    from iridium.runtime.chat import ChatSession

    model, manifest = load_checkpoint(str(checkpoint), device=DEVICE)
    chat = ChatSession(model)
    print(chat.send('Hello! What are you, and what can you actually do right now?'))
    '''


def chat_cli_markdown() -> str:
    return """
    The same checkpoint works from a terminal, outside this notebook:

    ```bash
    python -m iridium chat --checkpoint {checkpoint} --device auto
    ```

    (`python -m iridium chat` also defaults to the bundled small demo
    checkpoint if you omit `--checkpoint`; that one is a fixed-recipe
    fine-tune, not the model this notebook trained.)
    """.replace("{checkpoint}", "$OUT_DIR/<preset>/<preset>-final.pt")


def tools_stage_cell() -> str:
    return '''
    import importlib.util

    TOOLS_PRESET = 'tools-100m'

    if importlib.util.find_spec('iridium.runtime.tools') is None:
        print(
            'iridium.runtime.tools is not importable in this checkout yet -- '
            'tool use is still landing. Re-clone once it has merged and rerun '
            'this cell; skipping the tools stage for now.'
        )
        tools_checkpoint = None
    else:
        tools_preset = get_preset(TOOLS_PRESET)
        tools_checkpoint = train_preset(
            tools_preset, device=DEVICE, out=str(OUT_DIR), init=str(checkpoint), seed=0,
        )
        print('tools checkpoint:', tools_checkpoint)
    '''


def build_studio(target: str, title: str, default_preset: str) -> list[dict]:
    b = Builder()
    honesty_banner(b, title, default_preset)

    b.md("## 1. Get the source (private repo)\n\nNever prints or stores the token; see `notebooks/README.md` for how to add one.")
    b.code(private_clone_cell(target))

    b.md("## 2. Install and detect hardware")
    b.code(install_cell(target))
    b.code(hardware_and_preset_cell(target, default_preset))

    b.md("""
    ## 3. Dry run -- build, cost, audit, before anything trains

    No network, no training: builds the model on the meta device, checks the
    instantiated parameter count against `IridiumConfig`'s own formula, and
    prints the data-budget audit (`iridium.training.budget.audit`) plus the
    free-tier time estimates for every tier in `FREE_TIERS`. Read this before
    committing a free session to a long run.
    """)
    b.code(dry_run_cell())

    b.md("## 4. Train in rounds\n\nCheckpoint persistence for this host:")
    b.code(persistence_cell(target))
    b.md("""
    Training streams real text (and, depending on the preset, chat/tool/media
    data) over the network in rounds of fresh data -- see the module docstring
    in `iridium/training/run_preset.py` for why rounds exist. Each round saves
    a checkpoint; the loop below trains the whole preset unless you lower
    `STEPS`/`ROUNDS` for a shorter first try.
    """)
    b.code(train_cell())
    b.md(resume_markdown(target))

    b.md("## 6. Chat with the result")
    b.code(chat_cell())
    b.md(chat_cli_markdown())

    b.md("""
    ## 7. Optional: stage 2, add tool use (`tools-100m`)

    Initialises from the chat checkpoint you just trained (`--init`) rather
    than from scratch. This stage depends on `iridium.runtime.tools`, which is
    being built alongside this notebook; the cell below checks for it and
    tells you plainly if it is not there yet rather than failing deep inside a
    training run.
    """)
    b.code(tools_stage_cell())
    return b.cells


def build_colab_redirect() -> list[dict]:
    b = Builder()
    b.md(f"""
    # Iridium 1.0 -- start here: `iridium_studio.ipynb`

    {COLAB_BADGE.format(name='iridium_studio.ipynb')}

    This notebook (the old fixed single-preset Colab trainer) has been folded
    into **`iridium_studio.ipynb`**, in this same `notebooks/` directory. That
    notebook does everything this one used to -- detect the T4, pick a preset,
    dry-run it, train in rounds, chat with the result -- and adds the private-
    repo clone step, resume-after-disconnect instructions, and the optional
    tool-use second stage.

    Open `iridium_studio.ipynb` instead of this file; nothing below trains
    anything.
    """)
    b.code('''
    print(
        "This notebook is a redirect. Open iridium_studio.ipynb in this same "
        "notebooks/ directory instead -- it replaces this one."
    )
    ''')
    return b.cells


def build_tpu() -> list[dict]:
    """The ported TPU v5e-1 builder: preset + geometry overrides + a gate.

    Rebuilt from the description in the task brief -- the source PR's branch
    (``codex/create-colab-with-48gb-ram``) is not fetchable from here (only a
    local ``git show`` against an already-fetched ref is allowed, and that ref
    is not present in this checkout's remotes). Nothing below re-implements
    XLA training mechanics: it calls the same ``train_preset``/``Trainer`` the
    other notebooks call, with ``device='xla'``, and lets the library's own
    XLA support (``xm.optimizer_step``, ``xm.save``, a ``None`` generator on
    XLA -- see ``iridium/runtime/device.py:generator_for``) do the actual work.
    """
    b = Builder()
    b.md(f"""
    # Iridium 1.0 -- TPU v5e-1 builder (Colab/Kaggle)

    {COLAB_BADGE.format(name='train_iridium_tpu_colab.ipynb')}

    **The model is untrained until you run the training cell, and that cell is
    gated behind an explicit confirmation below** -- a free TPU session is
    quota you cannot get back, and this notebook would rather make you say so
    than start on autopilot.

    This targets the free **TPU v5e-1** runtime (`torch_xla`), not a GPU. It
    starts from a preset like the other studio notebooks, then exposes the raw
    geometry -- core layers, superstack layers, superstack count -- as
    overrides, because a TPU's extra memory (48 GB host RAM on a Colab v5e-1)
    is exactly the room to try a deeper or wider variant of a preset before it
    has a name.

    Needs network and the `datasets` package for the same reason as the other
    notebooks: the text/chat/tokenizer path streams real data and trains a
    subword tokenizer from it.
    """)

    b.md("## 1. Get the source (private repo)")
    b.code(private_clone_cell("colab"))

    b.md("## 2. Install (adds torch_xla for the TPU runtime)")
    b.code('''
    !pip install -q -e . 2>/dev/null || pip install -q torch numpy datasets pyyaml psutil
    !pip install -q torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
    ''')

    b.md("## 3. Detect the TPU and pick a starting preset")
    b.code('''
    import torch
    from iridium.presets import PRESETS, FREE_TIERS, get_preset, preset_table
    from iridium.runtime.device import detect

    try:
        import torch_xla.core.xla_model as xm
        DEVICE = str(xm.xla_device())
        print('XLA device:', DEVICE)
    except Exception as exc:
        print('torch_xla is not usable here (', exc, '); falling back to CPU/GPU detection.')
        DEVICE = detect().device

    PRESET = 'chat-100m'  # a TPU v5e-1's 16 GB and 48 GB host RAM affords more than chat-34m
    preset = get_preset(PRESET)
    print()
    print(preset_table())
    ''')

    b.md("""
    ## 4. Geometry overrides

    Leave any override at `None` to keep the preset's own value. These change
    `preset.config.core`/`preset.config.stacks` directly; the dry-run cell
    below re-derives the parameter count from the *overridden* shapes, so a
    mismatch here is caught before training, not after.
    """)
    b.code('''
    from dataclasses import replace

    CORE_LAYERS = None         # int, overrides preset.config.core.n_layers
    SUPERSTACK_LAYERS = None    # int, overrides preset.config.stacks.n_layers
    SUPERSTACK_COUNT = None     # int, overrides preset.config.stacks.n_stacks

    cfg = preset.config
    new_core = cfg.core if CORE_LAYERS is None else replace(cfg.core, n_layers=CORE_LAYERS)
    new_stacks = cfg.stacks
    if SUPERSTACK_LAYERS is not None:
        new_stacks = replace(new_stacks, n_layers=SUPERSTACK_LAYERS)
    if SUPERSTACK_COUNT is not None:
        new_stacks = replace(new_stacks, n_stacks=SUPERSTACK_COUNT)
    cfg = replace(cfg, core=new_core, stacks=new_stacks)
    preset = replace(preset, config=cfg)
    print(preset.config.report().render())
    ''')

    b.md("## 5. Dry run -- build, cost, audit")
    b.code(dry_run_cell())

    b.md("""
    ## 6. Confirm and train

    `torch_xla` compiles static shapes; this architecture's router dispatches
    a *variable* number of tokens per superstack per step, which forces a
    recompile per shape (or padding to a fixed capacity) on a real TPU core.
    That is a real consequence of dynamic routing on XLA, not a missing
    feature here -- expect the first several steps to be slow while XLA traces
    shapes, and expect a shape change (a new batch composition) to trigger
    another trace.

    Set `CONFIRM_TRAIN = True` once the dry run above looks right. This is the
    gate: nothing above this cell spends TPU quota.
    """)
    b.code('''
    from pathlib import Path
    from iridium.training.run_preset import train_preset

    CONFIRM_TRAIN = False
    OUT_DIR = Path('/content/drive/MyDrive/iridium-runs') if Path('/content/drive').is_dir() else Path('runs')
    RESUME_FROM = ''

    if not CONFIRM_TRAIN:
        print('CONFIRM_TRAIN is False; not training. Set it to True and rerun this cell.')
        checkpoint = None
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = train_preset(
            preset, device=DEVICE, out=str(OUT_DIR), resume=(RESUME_FROM or None), seed=0,
        )
        print('final checkpoint:', checkpoint)
    ''')

    b.md("## 7. Chat with the result")
    b.code('''
    from iridium.training.trainer import load_checkpoint
    from iridium.runtime.chat import ChatSession

    if checkpoint is None:
        print('No checkpoint yet -- confirm and train in section 6 first.')
    else:
        model, manifest = load_checkpoint(str(checkpoint), device='cpu')
        chat = ChatSession(model)
        print(chat.send('Hello! What are you, and what can you actually do right now?'))
    ''')
    return b.cells


def main() -> None:
    for target, name in GENERATED_STUDIO.items():
        title_prefix = {"colab": "Google Colab", "kaggle": "Kaggle", "jupyter": "Jupyter"}[target]
        default_preset = {"colab": "chat-34m", "kaggle": "chat-100m", "jupyter": "chat-100m"}[target]
        cells = build_studio(target, f"Iridium 1.0 Studio -- {title_prefix}", default_preset)
        write(name, cells)

    write(GENERATED_OTHER["colab_redirect"], build_colab_redirect())
    write(GENERATED_OTHER["tpu"], build_tpu())


def write(name: str, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU" if "tpu" not in name else "TPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    (HERE / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Wrote", name)


if __name__ == "__main__":
    main()
