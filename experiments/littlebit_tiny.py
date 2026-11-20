"""LittleBit on a 2M-parameter Iridium-1: train, compress, distill, compare (docs/littlebit.md).

Data: TinyStories-valid.txt from huggingface.co/datasets/roneneldan/TinyStories,
path in $TINYSTORIES. Usage: python experiments/littlebit_tiny.py [train_steps]
"""
import sys, os, json, copy, time, io, contextlib
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, torch
torch.set_num_threads(2)
from dataclasses import replace
from iridium.presets import get_preset
from iridium.codecs.spans import Sample, Span, text_span
from iridium.codecs.spans import MODALITY_INDEX
from iridium.training.tasks import BOS, EOS, Item
from iridium.training.datasets import Corpus, BatchLoader
from iridium.training.trainer import TrainConfig, Trainer
from iridium.model.iridium1 import Iridium1
from iridium.config import TEXT_ID_OFFSET
from iridium.quant.littlebit import compress, distill

W = 256; d = 128
CFG = get_preset("iridium-50m").config
core = replace(CFG.core, d_model=d, n_layers=2, n_query_heads=4, n_kv_heads=2, d_head=32, d_ff=336)
stacks = replace(CFG.stacks, d_model=d, core_d_model=d, n_layers=2, n_query_heads=4, n_kv_heads=2, d_head=32, d_ff=336, min_depth=1)
cfg = replace(CFG, core=core, stacks=stacks, ple_dim=4, perception_layers=1, perception_rank=8,
              max_seq_len=W, text_vocab_size=0, codecs=replace(CFG.codecs, vocab_size=320))
text = open(os.environ.get("TINYSTORIES", "TinyStories-valid.txt"), encoding="utf-8").read()
stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]
def item(s):
    b = s.encode("utf-8")[: W - 2].decode("utf-8", "ignore")
    return Item(sample=Sample([Span("control", np.array([BOS], np.int64), supervised=False),
                               text_span(b, supervised=True, offset=TEXT_ID_OFFSET),
                               Span("control", np.array([EOS], np.int64), supervised=True)]), family="text")
train = Corpus([item(s) for s in stories[:20000]], "train")
val = Corpus([item(s) for s in stories[-400:]], "test")

def val_loss(m):
    m.eval(); tot = n = 0
    with torch.no_grad():
        for batch, _ in BatchLoader(val, cfg.codecs, 25, seed=0, max_length=W).batches():
            losses, _ = m.losses(batch); tot += float(losses["text"]) * batch.n_tokens; n += batch.n_tokens
    return round(tot / n, 4)

def logits_of(m, batch):
    out = m(batch)
    h = out.hidden[:, :-1]
    mask = batch.supervised[:, 1:] & batch.valid[:, 1:] & (batch.modality[:, 1:] == MODALITY_INDEX["text"])
    return m.codecs.text_head(h), h, mask

def stream():
    loader = BatchLoader(train, cfg.codecs, 16, seed=7, max_length=W)
    while True:
        for b, _ in loader.batches():
            yield b

torch.manual_seed(0)
teacher = Iridium1(cfg)
with contextlib.redirect_stdout(io.StringIO()):
    Trainer(teacher, train, TrainConfig(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 600, batch_size=16,
            log_every=10**9, max_length=W, lr=3e-3, warmup_ratio=0.05)).train()
base = val_loss(teacher)
print(json.dumps({"fp32_teacher": base}), flush=True)
for bpw, itq, residual in [(1.0, True, True), (1.0, False, True), (2.0, True, True), (0.55, True, True)]:
    student = copy.deepcopy(teacher)
    rep = compress(student, bpw=bpw, residual=residual, itq=itq)
    before = val_loss(student)
    t0 = time.time()
    distill(student, teacher, stream(), logits_of, steps=300, lr=5e-4)
    after = val_loss(student)
    print(json.dumps({"bpw": bpw, "itq": itq, "residual": residual, "layers": rep["layers"],
                      "over_budget": rep["over_budget_layers"], "real_bpw": round(rep["bits_per_weight"], 3),
                      "val_before_qat": before, "val_after_qat": after, "teacher": base,
                      "secs": round(time.time() - t0)}), flush=True)
