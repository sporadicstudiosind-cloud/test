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
    order = ["tiny", "nano", "micro", "small", "base", "extreme"]
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


def test_text_id_offset_matches_the_definition_in_tasks():
    """config.TEXT_ID_OFFSET is a copy; this is what stops it drifting.

    ``iridium.config`` may not import torch — the multi-trillion rungs have to
    be costable on a laptop — and ``training.tasks`` does, so the constant is
    duplicated rather than imported. A duplicated constant is fine exactly as
    long as something fails when the two disagree.
    """
    from iridium.config import TEXT_ID_OFFSET
    from iridium.training.tasks import TEXT_OFFSET

    assert TEXT_ID_OFFSET == TEXT_OFFSET


def test_vocab_headroom_is_checked_before_a_run_not_during_one():
    """An id past the embedding table is not an error anyone sees in time."""
    import dataclasses

    import pytest

    from iridium.config import ConfigError, get_config

    cfg = get_config("nano")
    with pytest.raises(ConfigError, match="reserved control ids"):
        dataclasses.replace(cfg, text_vocab_size=8192)

    sized = dataclasses.replace(
        cfg,
        text_vocab_size=8192,
        codecs=dataclasses.replace(cfg.codecs, vocab_size=8192 + 16),
    )
    assert sized.text_vocab_size == 8192


def test_flow_head_options_are_validated_where_the_cost_is_computed():
    import dataclasses

    import pytest

    from iridium.config import ConfigError, get_config

    codecs = get_config("nano").codecs
    for field, bad in (("continuous_conditioning", "film"),
                       ("flow_timestep_sampling", "cosine"),
                       ("continuous_head", "diffusion")):
        with pytest.raises(ConfigError):
            dataclasses.replace(codecs, **{field: bad})


def test_adaln_conditioning_is_costed_against_the_real_module():
    """The formula must track the option, not just the default."""
    import dataclasses

    import torch

    from iridium.codecs.bank import CodecBank
    from iridium.config import get_config

    cfg = get_config("nano")
    for conditioning in ("add", "adaln"):
        codecs = dataclasses.replace(cfg.codecs, continuous_conditioning=conditioning)
        bank = CodecBank(codecs, cfg.core.d_model)
        expected = sum(codecs.params(cfg.core.d_model).values())
        actual = sum(p.numel() for p in bank.parameters())
        assert actual == expected, f"{conditioning}: formula {expected}, module {actual}"


def test_modern_rung_is_costed_exactly_without_allocating_it():
    """0.7 B parameters is too large to allocate in a unit test; the meta device
    builds every module and shape without any storage, which is all the
    accounting check needs."""
    import torch

    from iridium.model.iridium1 import Iridium1

    cfg = get_config("modern")
    with torch.device("meta"):
        model = Iridium1(cfg)
    assert sum(p.numel() for p in model.parameters()) == cfg.n_params
    assert cfg.text_vocab_size and cfg.codecs.vocab_size >= cfg.text_vocab_size + 16
    assert set(cfg.core.layer_kinds()) == {"deltanet", "mla"}
