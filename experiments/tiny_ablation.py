"""CPU ablation of Iridium-1 options on a 2M-parameter model (docs/iridium-50m.md).

Data: TinyStories-valid.txt (huggingface.co/datasets/roneneldan/TinyStories), path in
$TINYSTORIES. Usage: python experiments/tiny_ablation.py NAME steps=300 seed=0
[opt=normuon] [lr=..] [wd=..] [shape=linear minlr=0] [loops=0.7,0.2,0.1]
[core.attn_gate='"head"'] [core.canon=true] [router.ponder_global=true]
Threads from $THREADS (default 2); results appended to $ABLATION_OUT.
"""
import sys, json, math, time, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, torch
from dataclasses import replace
torch.set_num_threads(int(os.environ.get("THREADS", 2)))
from iridium.presets import get_preset
from iridium.codecs.spans import Sample, Span, text_span
from iridium.training.tasks import BOS, EOS, Item
from iridium.training.datasets import Corpus, BatchLoader
from iridium.training.trainer import TrainConfig, Trainer
from iridium.model.iridium1 import Iridium1
from iridium.config import TEXT_ID_OFFSET

name = sys.argv[1]
kv = dict(a.split("=", 1) for a in sys.argv[2:])
steps = int(kv.pop("steps", 300)); window = int(kv.pop("window", 256)); seed = int(kv.pop("seed", 0))
shape = kv.pop("shape", "sqrt"); minlr = float(kv.pop("minlr", 0.1)); opt = kv.pop("opt", "eager_adamw"); lr = float(kv.pop("lr", 3e-3)); wd = float(kv.pop("wd", 0.01))
loops = kv.pop("loops", "1")
d = int(kv.pop("d", 128))
CFG = get_preset("iridium-50m").config
core = replace(CFG.core, d_model=d, n_layers=2, n_query_heads=d // 32, n_kv_heads=2, d_head=32, d_ff=d * 8 // 3 // 8 * 8)
stacks = replace(CFG.stacks, d_model=d, core_d_model=d, n_layers=2, n_query_heads=d // 32, n_kv_heads=2,
                 d_head=32, d_ff=core.d_ff, min_depth=1)
cfg = replace(CFG, core=core, stacks=stacks, ple_dim=4, perception_layers=1, perception_rank=8,
              max_seq_len=window, text_vocab_size=0, codecs=replace(CFG.codecs, vocab_size=320))
# feature flags: key=val applied to cfg / cfg.core / stacks via "core.x" etc.
for k, v in list(kv.items()):
    val = json.loads(v)
    if k.startswith("core."):
        cfg = replace(cfg, core=replace(cfg.core, **{k[5:]: val}))
    elif k.startswith("stacks."):
        cfg = replace(cfg, stacks=replace(cfg.stacks, **{k[7:]: val}))
    elif k.startswith("router."):
        cfg = replace(cfg, router=replace(cfg.router, **{k[7:]: val}))
    else:
        cfg = replace(cfg, **{k: val})
text = open(os.environ.get("TINYSTORIES", "TinyStories-valid.txt"), encoding="utf-8").read()
stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]
def item(s):
    b = s.encode("utf-8")[: window - 2].decode("utf-8", "ignore")
    return Item(sample=Sample([Span("control", np.array([BOS], np.int64), supervised=False),
                               text_span(b, supervised=True, offset=TEXT_ID_OFFSET),
                               Span("control", np.array([EOS], np.int64), supervised=True)]), family="text")
train = Corpus([item(s) for s in stories[:20000]], "train")
val = Corpus([item(s) for s in stories[-400:]], "test")
torch.manual_seed(seed)
model = Iridium1(cfg)
tcfg = TrainConfig(steps=steps, batch_size=16, log_every=10**9, max_length=window, optimizer=opt, lr=lr,
                   weight_decay=wd, seed=seed, warmup_ratio=0.05, decay_shape=shape, min_lr_ratio=minlr,
                   loop_probs=tuple(json.loads(f"[{loops}]")) if "," in loops else ())
import io, contextlib
t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    tr = Trainer(model, train, tcfg)
    tr.train()
model.eval()
loader = BatchLoader(val, cfg.codecs, 25, seed=0, max_length=window)
tot = n = 0
with torch.no_grad():
    for batch, items in loader.batches():
        losses, _ = model.losses(batch)
        tot += float(losses["text"]) * batch.n_tokens; n += batch.n_tokens
res = {"name": name, "val_text": round(tot / n, 4), "params": sum(p.numel() for p in model.parameters()),
       "secs": round(time.time() - t0), "steps": steps, "opt": opt, "lr": lr}
print(json.dumps(res))
with open(os.environ.get("ABLATION_OUT", "ablation_results.jsonl"), "a") as f:
    f.write(json.dumps({**res, "args": sys.argv[2:]}) + "\n")
