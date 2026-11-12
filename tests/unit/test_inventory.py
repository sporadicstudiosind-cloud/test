"""Parameter and cache accounting.

§19.1 says an executable inventory is the final authority. This suite is that
authority: it recomputes the document's stated counts, and for the prototype it
checks them against a really-constructed ``torch`` module.
"""

import pytest

from iridium.model.inventory import (
    CONFIGS,
    FLAGSHIP,
    MIB,
    PILOT,
    PROTOTYPE,
    TIB,
    TransformerConfig,
)


@pytest.mark.parametrize(
    "config,expected",
    [
        (PROTOTYPE, 135_266_304),
        (PILOT, 5_771_362_304),
        (FLAGSHIP, 992_137_445_376),
    ],
)
def test_transformer_parameter_counts_match_specification(config, expected):
    assert config.transformer_params == expected


def test_flagship_is_about_one_trillion_transformer_parameters():
    assert 0.99e12 < FLAGSHIP.transformer_params < 1.0e12


def test_head_geometry_is_consistent():
    for config in CONFIGS.values():
        assert config.d_model == config.n_query_heads * config.d_head
        assert config.n_query_heads % config.n_kv_heads == 0


def test_inconsistent_config_rejected():
    with pytest.raises(ValueError, match="d_model"):
        TransformerConfig("bad", 100, 1, 1, 1, 256, 8, 2, 64)


@pytest.mark.parametrize(
    "recurrence,expected_mib", [(1, 1.375), (4, 5.125)]
)
def test_flagship_kv_per_token(recurrence, expected_mib):
    got = FLAGSHIP.kv_bytes_per_token(recurrence) / MIB
    assert got == pytest.approx(expected_mib, abs=1e-9)


@pytest.mark.parametrize(
    "recurrence,expected_tib", [(1, 1.375), (4, 5.125)]
)
def test_flagship_kv_per_million_token_stream(recurrence, expected_tib):
    got = FLAGSHIP.kv_bytes_for_stream(1_048_576, recurrence) / TIB
    assert got == pytest.approx(expected_tib, abs=1e-9)


def test_recurrence_indexed_cache_grows_with_depth():
    """§6.5: recurrences do not reuse one unlabeled KV array."""
    r1 = FLAGSHIP.kv_bytes_per_token(1)
    r4 = FLAGSHIP.kv_bytes_per_token(4)
    assert r4 > r1
    core_share = FLAGSHIP.n_core
    assert (r4 - r1) == 2 * FLAGSHIP.d_kv * 2 * (3 * core_share)


def test_grouped_query_attention_reduces_cache_substantially():
    """GQA is the mitigation; it is not expert routing and adds no router."""
    mha = TransformerConfig(
        "flagship_mha_control", 32768, 4, 80, 4, 90112, 256, 256, 128
    )
    ratio = mha.kv_bytes_per_token(1) / FLAGSHIP.kv_bytes_per_token(1)
    assert ratio == pytest.approx(8.0)          # 256 kv heads vs 32


def test_weight_and_optimizer_state_accounting():
    assert FLAGSHIP.weight_bytes() / 1e12 == pytest.approx(1.984, abs=0.01)
    assert FLAGSHIP.training_state_bytes() / 1e12 == pytest.approx(15.874, abs=0.02)


def test_prototype_matches_a_real_torch_module():
    """Cross-check the formula against a module that actually exists."""
    torch = pytest.importorskip("torch")
    from iridium.model.core import IridiumCore

    model = IridiumCore(PROTOTYPE)
    inventory = model.parameter_inventory()

    assert inventory["transformer_matmul_only"] == PROTOTYPE.transformer_params

    # Norms, embeddings, gates and the halting head are real parameters and are
    # counted separately, exactly as §19.1 requires.
    extra = inventory["total"] - inventory["transformer_matmul_only"]
    assert extra > 0
    assert extra / inventory["total"] < 0.01


def test_report_renders():
    text = __import__(
        "iridium.model.inventory", fromlist=["report"]
    ).report(PROTOTYPE)
    assert "iridium-1-prototype" in text
    assert "transformer params" in text
