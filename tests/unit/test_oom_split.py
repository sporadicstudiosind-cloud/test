"""An OOM first splits each micro-batch into more passes; it only halves the micro-batch as a last resort."""
import torch

from iridium.config import get_config
from iridium.model.iridium1 import Iridium1
from iridium.training import trainer as tr
from iridium.training.datasets import build_corpus
from iridium.training.trainer import TrainConfig, Trainer


def _trainer(tmp_path, steps=6):
    torch.manual_seed(0)
    model = Iridium1(get_config("nano"))
    cfg = TrainConfig(steps=steps, batch_size=4, log_every=1)
    return Trainer(model, build_corpus(32, seed=0), cfg, out_dir=tmp_path), model


def test_oom_splits_passes_then_decays_back(tmp_path, monkeypatch):
    t, model = _trainer(tmp_path)
    real = model.losses
    seen = []

    def losses(batch, **kw):
        rows = batch.modality.shape[0]
        seen.append(rows)
        if rows > 2 and len(seen) < 4:          # the first full-size pass "runs out of memory"
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB")
        return real(batch, **kw)

    monkeypatch.setattr(model, "losses", losses)
    monkeypatch.setattr(tr, "OOM_SPLIT_DECAY_STEPS", 2)
    t.train()
    assert t.loader.batch_size == 4, "micro-batch must not be halved when splitting suffices"
    assert 2 in seen and t.completed_steps == 6
    assert t._oom_split == 1, "the split decays back after quiet steps"
    assert 4 in seen[4:], "full-size passes resume after the decay"


def test_oom_halves_micro_batch_only_when_splitting_cannot_help(tmp_path, monkeypatch):
    t, model = _trainer(tmp_path, steps=2)
    real = model.losses
    state = {"n": 0}

    def losses(batch, **kw):
        state["n"] += 1
        if state["n"] <= 3:                    # OOM even at 2 and 1 rows per pass
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB")
        return real(batch, **kw)

    monkeypatch.setattr(model, "losses", losses)
    t.train()
    assert t.loader.batch_size == 2 and t.completed_steps == 2
