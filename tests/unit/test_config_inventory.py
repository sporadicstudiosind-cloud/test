"""The parameter formulae must match the real modules exactly, not approximately.

``config.py`` costs configurations nobody can instantiate. That is only useful
if the same formulae are exact where they *can* be checked, so these tests
build the real torch modules at the small rungs and demand a delta of zero.
"""

import pytest
import torch

from iridium.codecs.bank import CodecBank
from iridium.config import LADDER, ConfigError, CoreConfig, SuperstackConfig, get_config
from iridium.model.iridium1 import Iridium1
from iridium.model.rope import RotaryEmbedding
from iridium.model.superstack import SuperstackBank

SMALL = ("tiny", "nano")


@pytest.mark.parametrize("rung", SMALL)
def test_total_parameter_count_is_exact(rung):
    cfg = get_config(rung)
    model = Iridium1(cfg)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == cfg.n_params, (
        f"{rung}: formula says {cfg.n_params:,}, module has {actual:,}"
    )


@pytest.mark.parametrize("rung", SMALL)
def test_component_counts_are_exact(rung):
    cfg = get_config(rung)
    model = Iridium1(cfg)
    inventory = model.parameter_inventory()
    assert inventory["core"] == cfg.core.params
    assert inventory["superstacks"] == cfg.stacks.params
    assert inventory["router"] == cfg.router.params(cfg.core.d_model, cfg.stacks.n_stacks)
    assert inventory["codecs"] == sum(cfg.codecs.params(cfg.core.d_model).values())


@pytest.mark.parametrize("rung", SMALL)
def test_superstack_bank_matches_formula(rung):
    cfg = get_config(rung)
    bank = SuperstackBank(cfg, RotaryEmbedding(cfg.core.d_head))
    assert sum(p.numel() for p in bank.parameters()) == cfg.stacks.params


@pytest.mark.parametrize("rung", SMALL)
def test_codec_bank_matches_formula(rung):
    cfg = get_config(rung)
    bank = CodecBank(cfg.codecs, cfg.core.d_model)
    expected = sum(cfg.codecs.params(cfg.core.d_model).values())
    assert sum(p.numel() for p in bank.parameters()) == expected


def test_ladder_is_monotonic():
    order = ["tiny", "nano", "micro", "8b", "small", "base", "extreme"]
    sizes = [LADDER[k].n_params for k in order]
    assert sizes == sorted(sizes), "the ladder must grow monotonically"


def test_active_parameters_never_exceed_total():
    for name, cfg in LADDER.items():
        lo, hi = cfg.active_parameters()
        assert 0 < lo <= hi <= cfg.n_params, name


def test_flops_and_active_parameters_are_different_quantities():
    """A layer run three times touches one set of weights and costs three passes.

    Conflating them is how a specification quotes an 'active parameter' figure
    that no memory system ever has to hold.
    """
    cfg = get_config("nano")
    _, active_hi = cfg.active_parameters()
    _, flops_hi = cfg.flops_per_token()
    assert flops_hi > 2 * active_hi, "the ponder loop must show up in FLOPs"


def test_head_dimension_must_factor_the_model_width():
    with pytest.raises(ConfigError):
        CoreConfig(d_model=100, n_layers=4, n_query_heads=3, n_kv_heads=1,
                   d_head=32, d_ff=256)


def test_query_heads_must_be_a_multiple_of_kv_heads():
    with pytest.raises(ConfigError):
        CoreConfig(d_model=128, n_layers=4, n_query_heads=4, n_kv_heads=3,
                   d_head=32, d_ff=256)


def test_spectral_stacks_must_be_named_explicitly():
    """C^2 * modes^2 weights must never be handed to every stack by accident."""
    with pytest.raises(ConfigError, match="spectral_stacks"):
        SuperstackConfig(
            n_stacks=4, n_layers=4, d_model=64, n_query_heads=4, n_kv_heads=2,
            d_head=16, d_ff=176, spectral_stride=2, spectral_stacks=(),
        )


def test_mxfp4_is_four_and_a_quarter_bits_not_four():
    """The shared scale is not free; the plan's 0.5 bytes/param omits it."""
    from iridium.quant.mxfp4 import bits_per_param

    cfg = get_config("base")
    assert bits_per_param(32) == 4.25
    plan_claim = cfg.weight_bytes(4.0)
    actual = cfg.weight_bytes(4.25)
    assert actual > plan_claim
    assert actual / plan_claim == pytest.approx(4.25 / 4.0)


def test_roundtrip_through_dict():
    cfg = get_config("nano")
    from iridium.config import IridiumConfig

    again = IridiumConfig.from_dict(cfg.to_dict())
    assert again.n_params == cfg.n_params
    assert again.stacks.spectral_stacks == cfg.stacks.spectral_stacks


def test_8b_rung_is_full_architecture_and_costed_for_distributed_training():
    cfg = get_config("8b")
    assert 8.0e9 < cfg.n_params < 8.2e9
    assert cfg.stacks.n_stacks == 10
    assert cfg.router.top_k == 2
    # BF16 weights + grads + FP32 master and two moments. This excludes
    # activations, so a 48 GiB Colab must never be presented as sufficient.
    assert cfg.training_state_bytes() / 2**30 > 120
