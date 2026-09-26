"""The training-recipe options: schedule, optimizer, loss balancing, weight EMA."""
from __future__ import annotations

import math

import pytest
import torch

from iridium.config import get_config
from iridium.model.iridium1 import Iridium1
from iridium.training.datasets import OMNI_MIXTURE, build_corpus
from iridium.training.losses import LossBalancer
from iridium.training.trainer import TrainConfig, Trainer, learning_rate, wsd_lr


def test_wsd_warms_up_holds_then_cools_down_to_the_floor():
    cfg = TrainConfig(steps=100, lr=1.0, warmup=10, warmup_ratio=0.0, schedule="wsd",
                      decay_ratio=0.2, min_lr_ratio=0.1)
    rates = [learning_rate(s, cfg) for s in range(100)]
    assert rates[0] == pytest.approx(0.1) and rates[9] == pytest.approx(1.0)
    assert all(r == 1.0 for r in rates[10:80])                  # stable phase
    assert rates[80] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(rates[80:], rates[81:]))  # monotone cooldown
    assert rates[-1] == pytest.approx(0.1 + 0.9 * (1 - math.sqrt(19 / 20)))


def test_unknown_schedule_is_refused():
    with pytest.raises(ValueError):
        learning_rate(0, TrainConfig(schedule="linear"))


def test_balancer_equalises_task_terms_and_leaves_regularisers_alone():
    bal = LossBalancer("ema", momentum=0.5)
    losses = {"text": torch.tensor(6.0), "image": torch.tensor(0.02),
              "router_balance": torch.tensor(3.0)}
    out = bal(losses)
    assert float(out["text"]) == pytest.approx(1.0)
    assert float(out["image"]) == pytest.approx(1.0)
    assert float(out["router_balance"]) == 3.0
    assert LossBalancer("none")(losses)["text"] is losses["text"]


def test_balancer_ignores_absent_modalities():
    bal = LossBalancer("ema")
    out = bal({"video": torch.tensor(0.0)})
    assert float(out["video"]) == 0.0 and "video" not in bal.scale


def _trainer(tmp_path, **overrides):
    torch.manual_seed(0)
    # nano, not tiny: the synthetic families emit 8x8 patches, which is nano's
    # patch size and not tiny's.
    model = Iridium1(get_config("nano"))
    corpus = build_corpus(16, seed=0)
    cfg = TrainConfig(steps=3, batch_size=2, log_every=1, **overrides)
    return Trainer(model, corpus, cfg, out_dir=tmp_path)


@pytest.mark.parametrize("overrides", [
    dict(optimizer="muon"),
    dict(schedule="wsd", loss_balance="ema"),
    dict(ema_decay=0.9),
])
def test_recipe_options_train_end_to_end(tmp_path, overrides):
    trainer = _trainer(tmp_path, **overrides)
    history = trainer.train()
    assert all(math.isfinite(h["total"]) for h in history if "total" in h)
    if overrides.get("loss_balance") == "ema":
        assert any(k.startswith("raw.") for k in history[-1])


def test_ema_weights_lag_the_live_weights_and_survive_save(tmp_path):
    trainer = _trainer(tmp_path, ema_decay=0.9)
    before = {k: v.clone() for k, v in trainer.ema_state_dict().items()}
    trainer.train()
    live, ema = trainer.model.state_dict(), trainer.ema_state_dict()
    moved = [k for k in live if live[k].is_floating_point() and not torch.equal(live[k], before[k])]
    assert moved
    k = moved[0]
    # The average sits strictly between where it started and where training went.
    assert not torch.equal(ema[k], live[k]) and not torch.equal(ema[k], before[k])
    blob = torch.load(trainer.save("t"), map_location="cpu", weights_only=False)
    assert blob["ema_state_dict"] is not None and blob["loss_balancer"]["mode"] == "none"


def test_muon_routes_embeddings_to_adamw(tmp_path):
    trainer = _trainer(tmp_path, optimizer="muon")
    emb = trainer.model.codecs.text_embedding.weight
    for group in trainer.optimizer.param_groups:
        if any(p is emb for p in group["params"]):
            assert group["use_muon"] is False
            break
    else:
        pytest.fail("text embedding not in any optimizer group")


def test_omni_mixture_is_half_natural_language():
    total = sum(OMNI_MIXTURE.values())
    assert (OMNI_MIXTURE["text_lm"] + OMNI_MIXTURE["chat"]) / total == pytest.approx(0.5)
