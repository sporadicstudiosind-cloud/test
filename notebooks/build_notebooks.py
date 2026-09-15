"""Generate the Colab, Kaggle and generic-Jupyter notebooks from one source.

Three copies of a notebook drift. The shared cells live here once; each target
differs only in the parts that genuinely differ — how it installs, where it
detects its accelerator, how it saves, and how it exposes a port.

    python notebooks/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

BRANCH = "claude/gallant-faraday-lhycva"
REPO = "https://github.com/sporadicstudiosind-cloud/test.git"
HERE = Path(__file__).resolve().parent


def md(*l):
    return {"cell_type": "markdown", "metadata": {}, "source": [x + "\n" for x in l]}


def code(*l):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [x + "\n" for x in l]}


# --------------------------------------------------------------------------
# shared body
# --------------------------------------------------------------------------

def intro(target: str):
    free_note = {
        "colab": "Runtime → Change runtime type → GPU (or TPU), then work down the cells.",
        "kaggle": ("Settings → Accelerator → GPU T4 x2 (or P100). Kaggle gives 30 GPU-hours "
                   "a week, the most generous free quota of the three, and the session "
                   "survives longer than a Colab one."),
        "jupyter": ("Runs on Lightning AI Studios, SageMaker Studio Lab, Paperspace, a "
                    "RunPod/Vast pod, or your own machine. Nothing here is host-specific."),
    }[target]
    return [
        md("# Iridium-1 Studio" + ("" if target == "colab" else f" — {target.title()}"),
           "",
           "Build, train, grade and talk to a routed control-core model at any size from",
           "**50 M to 1 T parameters**, on **CUDA, ROCm or TPU**, trained on **licensed,",
           "attributed** text.",
           "",
           "One control core that every token passes through, dispatching to deep domain",
           "superstacks, choosing its own depth and how many passes to spend. Omnimodal in",
           "and out: text, image, video, audio, physical fields, geometry, actions and",
           "typed quantities.",
           "",
           free_note,
           "",
           "---",
           "",
           "### Read this before choosing a size",
           "",
           "Every preset is a *real* geometry whose parameter count is computed from tensor",
           "shapes, not asserted. That is not the same as being trainable on the machine you",
           "have. The fit cell computes what your device can actually hold and says so",
           "plainly — including when the answer is no.",
           "",
           "Rough guide on a free 16 GB accelerator:",
           "",
           "| preset | trains free? | why |",
           "|---|---|---|",
           "| 50 M – 500 M | yes, comfortably | AdamW fp32 fits with room for activations |",
           "| 1 B | with `adamw_8bit` or `adafactor` | full fp32 Adam is 16 GB of state alone |",
           "| 8 B | LoRA only | 127 GB of optimizer state otherwise |",
           "| 16 B and up | no | needs sharding across many devices |",
           "",
           "You can build and cost every size regardless. A 1 T config is a costed design,",
           "not a model you are about to train."),
    ]


def setup(target: str):
    if target == "colab":
        install = [
            f"!git clone --depth 1 --branch {BRANCH} {REPO} iridium 2>/dev/null || (cd iridium && git pull -q)",
            "%cd iridium",
            "!pip -q install pyyaml datasets psutil",
        ]
    elif target == "kaggle":
        install = [
            "# Kaggle needs internet enabled: Settings -> Internet -> On",
            f"!git clone --depth 1 --branch {BRANCH} {REPO} /kaggle/working/iridium 2>/dev/null || true",
            "%cd /kaggle/working/iridium",
            "!pip -q install pyyaml datasets psutil",
        ]
    else:
        install = [
            f"!git clone --depth 1 --branch {BRANCH} {REPO} iridium 2>/dev/null || (cd iridium && git pull -q)",
            "%cd iridium",
            "!pip -q install pyyaml datasets psutil",
            "# If torch is absent, install the build that matches your hardware:",
            "#   CUDA : pip install torch --index-url https://download.pytorch.org/whl/cu124",
            "#   ROCm : pip install torch --index-url https://download.pytorch.org/whl/rocm6.2",
        ]
    return [
        md("## 1 · Hardware and code"),
        code(*install),
        code(
            "import sys, os, json; sys.path.insert(0, '.')",
            "import torch",
            "from iridium.runtime.device import detect, verify",
            "info = detect()",
            "print('torch', torch.__version__)",
            "print(info.describe())",
            "print(json.dumps(verify(info), indent=2, default=str))",
            "",
            "def host_ram_bytes():",
            "    \"\"\"Host RAM, without assuming psutil is installed.",
            "",
            "    Colab and Kaggle ship psutil; a bare RunPod or Studio Lab image often",
            "    does not, and a memory *check* that crashes for want of a memory library",
            "    is a poor joke. os.sysconf works on any Linux, and the cgroup limit,",
            "    where present, is the number that actually applies to a container --",
            "    which is the one that decides whether your run gets OOM-killed.\"\"\"",
            "    try:",
            "        import psutil",
            "        total = psutil.virtual_memory().total",
            "    except ImportError:",
            "        try:",
            "            total = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')",
            "        except (ValueError, OSError):",
            "            return 8e9   # a deliberately pessimistic floor",
            "    for path in ('/sys/fs/cgroup/memory.max',",
            "                 '/sys/fs/cgroup/memory/memory.limit_in_bytes'):",
            "        try:",
            "            with open(path) as fh:",
            "                limit = int(fh.read().strip())",
            "            if 0 < limit < total:",
            "                total = limit",
            "        except (OSError, ValueError):",
            "            pass",
            "    return total",
            "",
            "print(f'host RAM: {host_ram_bytes()/1e9:.1f} GB')",
        ),
    ]


def tpu_cell(target: str):
    if target != "colab":
        return [code(
            "DEVICE = info.device",
            "print('DEVICE =', DEVICE)",
        )]
    return [
        md("### TPU (optional)",
           "",
           "PyTorch reaches TPUs through `torch_xla`, and it is worth knowing what that",
           "means for *this* architecture before spending time on it: XLA compiles static",
           "shapes, and the router dispatches a **variable number of tokens** to each",
           "superstack every step. That forces a recompile per shape, or padding to a fixed",
           "capacity. So TPU work runs on the **host CPU** of the TPU VM by default — on a",
           "Colab v5e-1 that is 48 GB of RAM, genuinely useful for the larger presets, just",
           "slow. Set `FORCE_XLA = True` to drive the accelerator anyway.",
           "",
           "This is a real consequence of dynamic routing on XLA, not a missing feature."),
        code(
            "FORCE_XLA = False  #@param {type:'boolean'}",
            "",
            "if os.environ.get('COLAB_TPU_ADDR') or os.environ.get('TPU_WORKER_ID'):",
            "    if FORCE_XLA:",
            "        !pip -q install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html",
            "        import torch_xla.core.xla_model as xm",
            "        DEVICE = xm.xla_device()",
            "        print('using XLA — expect a recompile on every new routing shape')",
            "    else:",
            "        DEVICE = 'cpu'",
            "        print(f'TPU VM; using host CPU with {host_ram_bytes()/1e9:.0f} GB RAM')",
            "else:",
            "    DEVICE = info.device",
            "print('DEVICE =', DEVICE)",
        ),
    ]


def param(target: str, line: str, form: str) -> str:
    """Colab and Kaggle render `#@param` forms; plain Jupyter ignores them."""
    return f"{line}  #@param {form}" if target in ("colab", "kaggle") else line


def design(target: str):
    p = lambda line, form: param(target, line, form)  # noqa: E731
    return [
        md("## 2 · Design the model",
           "",
           "Pick a preset, or set `PRESET = 'custom'` and drive every knob yourself. These",
           "are the actual architectural degrees of freedom:",
           "",
           "- **`CORE_LAYERS`** — depth of the control stack every token passes through",
           "- **`N_SUPERSTACKS`** — how many domain banks the router can choose between",
           "- **`SUPERSTACK_LAYERS`** — depth of each bank. The design wants this *deeper*",
           "  than the core; that is what makes it a superstack rather than an expert",
           "- **`TOP_K`** — how many banks each token visits",
           "- **`MAX_LOOPS`** — ponder budget: how many times a token may go round again",
           "- **`MIN_DEPTH`** — floor on the focus ladder, so a token cannot skip everything",
           "- **`CROSS_STRIDE`** — how often a stack layer cross-attends to the core's history",
           "- **`SPECTRAL_STACKS`** — which banks carry Fourier operator blocks. These cost",
           "  `C² × modes²` weights each, so they are opt-in per stack rather than global",
           "- **`VOCAB_SIZE`** — leave it at 384 unless you know why you are changing it.",
           "  The text codec is byte-level: it emits raw UTF-8 bytes shifted past the",
           "  control ids, so 256 values plus control room is the entire reachable",
           "  vocabulary. A 32,000-row embedding here is not extra capacity, it is rows",
           "  that never receive a gradient and a softmax over classes the data cannot",
           "  produce. At the 1 T geometry that mistake costs 1.3 B dead parameters.",
           "",
           "`TOP_K` is clamped to `N_SUPERSTACKS - 1`. Routing to every bank is not",
           "routing: the gate becomes decorative, the balance loss is satisfied by",
           "construction, and you have a dense ensemble wearing a router."),
        code(
            p("PRESET = '100m'", "['50m','100m','500m','1b','8b','16b','24b','100b','200b','1t','custom']"),
            "",
            "# --- custom geometry (used when PRESET = 'custom') ---",
            p("D_MODEL           = 512", "{type:'integer'}"),
            p("CORE_LAYERS       = 6", "{type:'slider', min:2, max:128, step:1}"),
            p("N_SUPERSTACKS     = 4", "{type:'slider', min:1, max:64, step:1}"),
            p("SUPERSTACK_LAYERS = 8", "{type:'slider', min:1, max:256, step:1}"),
            p("D_HEAD            = 64", "[32, 64, 128] {type:'raw'}"),
            p("N_KV_HEADS        = 2", "{type:'slider', min:1, max:32, step:1}"),
            p("TOP_K             = 2", "{type:'slider', min:1, max:8, step:1}"),   # clamped to N_SUPERSTACKS - 1
            p("MAX_LOOPS         = 3", "{type:'slider', min:1, max:8, step:1}"),
            p("MIN_DEPTH         = 2", "{type:'slider', min:1, max:64, step:1}"),
            p("CROSS_STRIDE      = 4", "{type:'slider', min:1, max:32, step:1}"),
            p("VOCAB_SIZE        = 384", "[288, 320, 384, 512, 1024] {type:'raw'}"),
            p("MAX_SEQ_LEN       = 2048", "[512, 1024, 2048, 4096, 8192, 16384] {type:'raw'}"),
            p("SPECTRAL_STACKS   = '0'", "{type:'string'}"),
            "",
            "from iridium.config_builder import build, preset, ladder_table, fits",
            "print(ladder_table()); print()",
            "",
            "if PRESET == 'custom':",
            "    cfg = build(d_model=D_MODEL, core_layers=CORE_LAYERS,",
            "                n_superstacks=N_SUPERSTACKS, superstack_layers=SUPERSTACK_LAYERS,",
            "                d_head=D_HEAD, n_kv_heads=N_KV_HEADS, top_k=TOP_K,",
            "                max_loops=MAX_LOOPS, min_depth=MIN_DEPTH,",
            "                cross_stride=CROSS_STRIDE, vocab_size=VOCAB_SIZE,",
            "                max_seq_len=MAX_SEQ_LEN,",
            "                spectral_stacks=tuple(int(x) for x in SPECTRAL_STACKS.split(',') if x.strip()),",
            "                name='iridium-1-custom')",
            "else:",
            "    cfg = preset(PRESET)",
            "",
            "print(cfg.report().render())",
            "lo, hi = cfg.flops_per_token()",
            "print(f'\\n  forward FLOPs/token  {lo/1e9:,.2f} - {hi/1e9:,.2f} GFLOP')",
            "print(f'  core                 {cfg.core.n_layers} layers, d={cfg.core.d_model}')",
            "print(f'  superstacks          {cfg.stacks.n_stacks} x {cfg.stacks.n_layers} layers')",
            "print(f'  routing              top-{cfg.router.top_k}, up to {cfg.router.max_loops} loops')",
        ),
        md("## 3 · Will it actually train here?",
           "",
           "The cell that stops you wasting an afternoon. It compares the optimizer state",
           "your geometry needs against the memory the device reports, and names a cheaper",
           "strategy when the answer is no.",
           "",
           "| strategy | bytes/param | what it gives up |",
           "|---|---|---|",
           "| `adamw_fp32` | 16 | nothing — the reference |",
           "| `adamw_8bit` | 6 | quantised moments |",
           "| `adafactor` | 6 | the first moment entirely |",
           "| `sgd` | 4 | all momentum |",
           "| `lora` | 2.2 | the base weights stay frozen |"),
        code(
            p("STRATEGY = 'adamw_fp32'", "['adamw_fp32','adamw_bf16','adamw_8bit','adafactor','sgd','lora']"),
            "",
            "if str(DEVICE).startswith('cuda'):",
            "    avail = torch.cuda.get_device_properties(0).total_memory",
            "    where = torch.cuda.get_device_name(0)",
            "else:",
            "    avail = host_ram_bytes(); where = 'host RAM'",
            "print(f'device: {where}  ({avail/1e9:.1f} GB)')",
            "print(fits(cfg, avail, STRATEGY).render()); print()",
            "for s in ['adamw_fp32','adamw_8bit','adafactor','sgd','lora']:",
            "    r = fits(cfg, avail, s)",
            "    print(f\"   {s:<12}{r.bytes_needed/1e9:9.1f} GB  {'fits' if r.fits else 'no'}\")",
        ),
    ]


def data(target: str):
    p = lambda line, form: param(target, line, form)  # noqa: E731
    return [
        md("## 4 · Data — licensed, attributed, streamed",
           "",
           "Every source names its licence and what that licence obliges you to do, and the",
           "obligation travels into the run manifest. A CC BY-SA corpus requires attribution",
           "and share-alike on derivatives; a model that cannot say what it was trained on",
           "cannot honour that.",
           "",
           "Text streams rather than downloads — the notebook reads the few hundred megabytes",
           "it will consume rather than the terabytes it will not.",
           "",
           "**`CHAT_WEIGHT` is what makes it answer you.** Byte prediction over books and",
           "encyclopedia articles produces a model that *continues prose*: prompted with",
           "\"what is a weir?\" it writes the next paragraph of an article about weirs rather",
           "than replying to you. Turn-taking has to be in the training data, so real",
           "conversations are — Dolly 15k (CC BY-SA 3.0) and OASST1 (Apache 2.0), both",
           "human-written, supervised on the assistant's turns only.",
           "",
           "The synthetic families alongside it are **exactly checkable**, and that is what",
           "makes the grading later mean something. Real text can only be scored by",
           "likelihood; `channel_depth` can be scored against Manning's law."),
        code(
            p("USE_REAL_TEXT = True", "{type:'boolean'}"),
            p("CHAT_WEIGHT   = 0.45", "{type:'slider', min:0, max:0.9, step:0.05}"),
            p("TEXT_WEIGHT   = 0.25", "{type:'slider', min:0, max:0.9, step:0.05}"),
            p("N_TRAIN_ITEMS = 40000", "{type:'integer'}"),
            "",
            "from iridium.data.text_corpus import DEFAULT_MIX, licence_notice, probe_availability",
            "from iridium.data.chat_corpus import DEFAULT_CHAT_MIX, chat_licence_notice",
            "from iridium.training.datasets import build_corpus, describe",
            "",
            "mixture = {'channel_depth': 0.30, 'channel_intervention': 0.20,",
            "           'false_premise': 0.10, 'field_rollout': 0.10, 'scene_goal': 0.10}",
            "if USE_REAL_TEXT:",
            "    real = CHAT_WEIGHT + TEXT_WEIGHT",
            "    scale = max(0.0, 1.0 - real) / sum(mixture.values())",
            "    mixture = {k: v * scale for k, v in mixture.items()}",
            "    mixture['text_lm'] = TEXT_WEIGHT",
            "    mixture['chat'] = CHAT_WEIGHT",
            "    print(licence_notice(DEFAULT_MIX)); print()",
            "    print(chat_licence_notice()); print()",
            "    print('reachability:', probe_availability(DEFAULT_MIX)); print()",
            "",
            "train = build_corpus(N_TRAIN_ITEMS, seed=0, split='train', mixture=mixture)",
            "test  = build_corpus(800, seed=1000, split='test', mixture=mixture)",
            "extra = build_corpus(400, seed=2000, split='extrapolation', mixture=mixture)",
            "print(describe(train))",
        ),
        md("## 5 · Verify the architecture before spending time on it",
           "",
           "Two properties, twenty seconds. The parameter formulae that cost a 1 T",
           "configuration are the same ones describing this model — at the rungs that",
           "instantiate, the difference is exactly zero. And the parity gate proves cached",
           "decoding computes what teacher forcing trained; three mechanisms in this design",
           "could break that silently, and two of them did during development.",
           "",
           "If either fails, nothing downstream means anything."),
        code("!python -m pytest tests/unit/test_config_inventory.py tests/integration/test_kv_parity.py -q"),
    ]


def train_and_grade(target: str):
    p = lambda line, form: param(target, line, form)  # noqa: E731
    return [
        md("## 6 · Train"),
        code(
            p("STEPS      = 3000", "{type:'integer'}"),
            p("BATCH_SIZE = 32", "{type:'integer'}"),
            p("LR         = 5e-4", "{type:'number'}"),
            p("N_LOOPS    = 1", "{type:'slider', min:1, max:3, step:1}"),
            "",
            "import time",
            "from pathlib import Path",
            "from iridium.model.iridium1 import Iridium1",
            "from iridium.training.trainer import TrainConfig, Trainer",
            "from iridium.training.losses import LossWeights",
            "from iridium.evaluation.harness import evaluate",
            "",
            "model = Iridium1(cfg).to(DEVICE)",
            "print(f'{sum(q.numel() for q in model.parameters()):,} parameters on {DEVICE}')",
            "before = evaluate(model, test, max_per_family=12)",
            "print('untrained:', json.dumps(before)[:500])",
            "",
            "tcfg = TrainConfig(steps=STEPS, batch_size=BATCH_SIZE, lr=LR, n_loops=N_LOOPS,",
            "                   seed=0, label=f'studio-{PRESET}',",
            "                   log_every=max(STEPS//60, 1), checkpoint_every=max(STEPS//6, 1))",
            "trainer = Trainer(model, train, tcfg, LossWeights(),",
            "                  out_dir=Path('runs/studio'), device=str(DEVICE))",
            "t0 = time.time(); trainer.train()",
            "print(f'trained in {(time.time()-t0)/60:.1f} min')",
        ),
        md("## 7 · Grade — against independent computation, not against itself",
           "",
           "Free-running generation, checked against the analytic Manning law, the spectral",
           "solver, or the environment's own goal predicate. Every score sits beside the",
           "baseline a model earns by **ignoring its input entirely**, because a number with",
           "no baseline cannot be read.",
           "",
           "`extrapolation` draws from a parameter band the training split never contains, so",
           "a score there cannot come from having seen a neighbour."),
        code(
            "model.eval()",
            "results = {'interpolation': evaluate(model, test, max_per_family=40),",
            "           'extrapolation': evaluate(model, extra, max_per_family=40),",
            "           'untrained': before}",
            "print(f\"{'family':<26}{'trained':>9}{'baseline':>10}   verdict\")",
            "for fam, r in results['interpolation'].items():",
            "    acc, base = r.get('accuracy', 0), r.get('baseline', 0)",
            "    v = 'LEARNED' if acc > base + 0.1 else ('at baseline' if acc >= base else 'BELOW baseline')",
            "    note = ''",
            "    if 'median_relative_error' in r:",
            "        note = f\"   median err {r['median_relative_error']*100:.2f}%\"",
            "    if 'bits_per_byte' in r:",
            "        # Not an accuracy: the column holds the fraction of a uniform byte",
            "        # model's entropy removed, and bits/byte is the number to read.",
            "        note = f\"   {r['bits_per_byte']:.3f} bits/byte (uniform = 8.0)\"",
            "    print(f'{fam:<26}{acc:>9.3f}{base:>10.3f}   {v}{note}')",
        ),
        md("## 8 · Did the bank specialise, or just balance?",
           "",
           "These pull in opposite directions and a router can look healthy on one while",
           "failing the other. Maximal entropy with zero mutual information means every stack",
           "gets an equal share of every task — balanced, and a very expensive way to be one",
           "stack. `I(family; stack)` is what separates them."),
        code(
            "from iridium.evaluation.routing import analyse",
            "from iridium.training.datasets import BatchLoader",
            "print(analyse(model, BatchLoader(test, cfg.codecs, 16, 0, device=str(DEVICE))).render())",
        ),
    ]


def save_and_chat(target: str):
    save_extra = {
        "colab": ["# from google.colab import drive; drive.mount('/content/drive')",
                  "# !cp iridium-fp16.pt /content/drive/MyDrive/"],
        "kaggle": ["# /kaggle/working persists as a notebook output — the file is downloadable",
                   "!ls -la /kaggle/working/iridium/iridium-fp16.pt"],
        "jupyter": ["# The checkpoint is on local disk; copy it wherever this host persists."],
    }[target]
    port_cell = {
        "colab": ["from google.colab.output import eval_js",
                  "print('Chat UI:', eval_js('google.colab.kernel.proxyPort(8080)'))"],
        "kaggle": ["# Kaggle does not forward ports. Use ask() above, or expose it with a",
                   "# tunnel you trust if you want the browser UI.",
                   "print('use ask(...) above; Kaggle has no public port forwarding')"],
        "jupyter": ["# Lightning AI and Studio Lab forward ports from the sidebar;",
                    "# elsewhere use ssh -L 8080:localhost:8080.",
                    "print('serving on :8080 — forward it from your host UI')"],
    }[target]
    return [
        md("## 9 · Save",
           "",
           "fp16, with the licence notice in the manifest so the corpus obligations travel",
           "with the weights."),
        code(
            "from iridium.data.text_corpus import licence_notice, DEFAULT_MIX",
            "path = trainer.save('final', extra={",
            "    'evaluation': results,",
            "    'data_licences': licence_notice(DEFAULT_MIX) if USE_REAL_TEXT else 'synthetic only'})",
            "print('saved', path)",
            "blob = torch.load(path, map_location='cpu', weights_only=False)",
            "half = {k: (v.half() if v.is_floating_point() else v) for k, v in blob['state_dict'].items()}",
            "torch.save({'state_dict': half, 'manifest': blob['manifest']}, 'iridium-fp16.pt')",
            "print('fp16:', os.path.getsize('iridium-fp16.pt')/1e6, 'MB')",
            *save_extra,
        ),
        md("## 10 · Talk to it",
           "",
           "A real conversation: type anything, get a reply, history carried across turns.",
           "`chat.send(...)` conditions on the whole exchange, stops when the model ends its",
           "turn, and samples with a nucleus filter and a repetition penalty — greedy",
           "decoding at this size loops inside two sentences.",
           "",
           "**What to expect.** A 50 M–1 B model trained for an hour or two on a free GPU",
           "will take turns, stay on topic for a sentence or so, and be wrong about facts.",
           "It is not going to be ChatGPT, which was trained on a budget several orders of",
           "magnitude larger. Longer training and a bigger preset both help, and neither",
           "closes that gap. The physics questions below are the part with a checkable",
           "answer, and they are where this model is actually worth anything."),
        code(
            "from iridium.runtime.chat import ChatSession",
            "",
            "chat = ChatSession(model, temperature=0.8, top_p=0.92,",
            "                   repetition_penalty=1.15, max_new_tokens=160)",
            "",
            "for message in ['Hello! Who are you?',",
            "                'What is a weir?',",
            "                'Can you explain it more simply?']:",
            "    print(f'you     : {message}')",
            "    print(f'iridium : {chat.send(message)}\\n')",
        ),
        md("### Your turn",
           "",
           "Run this cell and type. Blank line or `quit` ends it; `reset` clears the history."),
        code(
            "while True:",
            "    try:",
            "        message = input('you     : ').strip()",
            "    except (EOFError, KeyboardInterrupt):",
            "        break",
            "    if not message or message.lower() in ('quit', 'exit'):",
            "        break",
            "    if message.lower() == 'reset':",
            "        chat.reset(); print('(history cleared)'); continue",
            "    print(f'iridium : {chat.send(message)}\\n')",
        ),
        md("### The part with a checkable answer",
           "",
           "Ask for a number and it answers with one — beside the analytic value, so you can",
           "check it rather than believe it. Numbers travel as **typed quantities**, never as",
           "decimal prose: the same mapping learned from digit-bytes puts 18.6% of answers",
           "inside a 2% tolerance, and learned from typed values, 100%."),
        code(
            "import threading, subprocess, urllib.request, time",
            "os.environ.update({'IRIDIUM_CHECKPOINT': str(path), 'PYTHONPATH': '.',",
            "                   'PORT': '8080', 'IRIDIUM_ENABLE_1B': '0'})",
            "threading.Thread(target=lambda: subprocess.run(['python','serve/server.py']),",
            "                 daemon=True).start()",
            "time.sleep(30)",
            "",
            "def ask(q, loops=1):",
            "    body = json.dumps({'prompt': q, 'loops': loops}).encode()",
            "    req = urllib.request.Request('http://127.0.0.1:8080/api/ask', body,",
            "                                 {'Content-Type': 'application/json'})",
            "    return json.load(urllib.request.urlopen(req, timeout=180))",
            "",
            "for q in ['normal depth | S=0.0020 n=0.030 q=3.0',",
            "          'depth ratio | S=0.0020 n=0.030 q=3.0 x2.0',",
            "          'doubling the discharge doubles the flow depth']:",
            "    r = ask(q); a = r['answer']; t = r['telemetry']",
            "    if 'predicted' in a:",
            "        print(f\"{q}\\n   model {a['predicted']:.4f}   analytic {a['analytic']:.4f}\"",
            "              f\"   err {a['relative_error']*100:.2f}%   within 2%: {a['within_2pct']}\")",
            "    else:",
            "        print(f\"{q}\\n   verdict {a['verdict']}\")",
            "    print(f\"   {t['stacks_used']}/{len(t['routing'])} stacks,\"",
            "          f\" focus {t['mean_focus']:.3f}, {t['expected_loops']:.2f} ponder loops\")",
        ),
        code(*port_cell),
        md("---",
           "",
           "### Reading the results honestly",
           "",
           "- `channel_depth` and `channel_intervention` are **affine in log space** once",
           "  inputs arrive as typed quantities. Above 0.85 on interpolation is the expected",
           "  outcome; well below means something is wrong, not that the task is hard. A",
           "  reference 104 M run reached **0.850 at 1.16% median error**.",
           "- `extrapolation` will be lower. The gap between the columns is the honest",
           "  measure of what was learned versus fitted.",
           "- `false_premise` draws from only ten distinct claims, so a high score is",
           "  **memorisation, not judgement**. Do not read it as calibration.",
           "- `field_rollout` must beat the persistence baseline (emit the input frame",
           "  unchanged) to mean anything at all.",
           "- `text_lm` is bits per byte against a uniform baseline of 8.0. It is a",
           "  likelihood, not an accuracy, and a good number says nothing about whether the",
           "  model is *right* about anything.",
           "- Stack entropy near `log(n_stacks)` means no collapse. `I(family; stack)` near",
           "  zero means no specialisation — balanced but undifferentiated, which is an",
           "  expensive way to be one stack."),
    ]


TARGETS = {
    "colab": ("iridium_studio.ipynb", {"accelerator": "GPU", "colab": {"provenance": [], "toc_visible": True}}),
    "kaggle": ("iridium_studio_kaggle.ipynb", {"accelerator": "GPU"}),
    "jupyter": ("iridium_studio_jupyter.ipynb", {}),
}


def main() -> int:
    for target, (filename, meta) in TARGETS.items():
        cells = (intro(target) + setup(target) + tpu_cell(target) + design(target)
                 + data(target) + train_and_grade(target) + save_and_chat(target))
        nb = {"cells": cells,
              "metadata": {**meta,
                           "kernelspec": {"display_name": "Python 3", "name": "python3"},
                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 0}
        path = HERE / filename
        path.write_text(json.dumps(nb, indent=1))
        print(f"wrote {path.name}: {len(cells)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
