"""Loading older checkpoints: fill what is provably inert, refuse everything else."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from iridium.config import IridiumConfig, get_config
from iridium.model.iridium1 import Iridium1
from iridium.runtime.checkpoint_compat import CheckpointIncompatible, load_compatible

SHIPPED = Path(__file__).resolve().parents[2] / "serve" / "weights" / "nano-phase1-fp16.pt"


@pytest.fixture(scope="module")
def pair():
    torch.manual_seed(0)
    return Iridium1(get_config("tiny")), Iridium1(get_config("tiny"))


def test_identical_state_loads_exactly(pair):
    src, dst = pair
    report = load_compatible(dst, src.state_dict())
    assert report.exact
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v)


def test_missing_guidance_vectors_are_filled_and_named(pair):
    src, dst = pair
    state = {k: v for k, v in src.state_dict().items() if not k.endswith("null_cond")}
    report = load_compatible(dst, state)
    assert report.filled and all(k.endswith("null_cond") for k in report.filled)


def test_an_entirely_absent_modality_is_reported_as_untrained(pair):
    src, dst = pair
    state = {k: v for k, v in src.state_dict().items() if ".quantity." not in k}
    report = load_compatible(dst, state)
    assert report.untrained_modalities == ["quantity"]


def test_a_partially_missing_modality_is_a_real_mismatch(pair):
    src, dst = pair
    state = dict(src.state_dict())
    del state["codecs.encoders.quantity.weight"]          # bias still present
    with pytest.raises(CheckpointIncompatible, match="not safe to fill"):
        load_compatible(dst, state)


def test_a_missing_core_weight_is_refused(pair):
    src, dst = pair
    state = dict(src.state_dict())
    del state["core.layers.0.attn.wq.weight"]
    with pytest.raises(CheckpointIncompatible, match="core.layers.0.attn.wq.weight"):
        load_compatible(dst, state)


def test_shape_mismatch_and_unexpected_keys_are_refused(pair):
    src, dst = pair
    state = dict(src.state_dict())
    state["core.out_norm.weight"] = torch.ones(3)
    with pytest.raises(CheckpointIncompatible, match="shape mismatch"):
        load_compatible(dst, state)
    state = dict(src.state_dict())
    state["core.surprise"] = torch.zeros(1)
    with pytest.raises(CheckpointIncompatible, match="unexpected"):
        load_compatible(dst, state)


@pytest.mark.skipif(not SHIPPED.exists(), reason="shipped weights not present")
def test_the_shipped_checkpoint_loads():
    """It has not loaded strictly since the quantity modality was added."""
    blob = torch.load(SHIPPED, map_location="cpu", weights_only=False)
    model = Iridium1(IridiumConfig.from_dict(blob["manifest"]["model_config"]))
    report = load_compatible(model, {k: v.float() for k, v in blob["state_dict"].items()})
    assert report.untrained_modalities == ["quantity"]
    assert all("quantity" in k or k.endswith("null_cond") for k in report.filled)
