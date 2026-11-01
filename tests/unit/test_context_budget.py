"""context_budget: pure-Python arithmetic, checked against its own formulas.

No model runs here -- the whole point of this module (see its docstring) is
answering "does 1M tokens fit" *before* anything at that scale is
instantiated. So these tests check the arithmetic against hand-derived
numbers and against the invariants the module claims (linear vs. bounded vs.
constant growth in ``tokens``), and pin the "no torch" promise as a static
source check rather than a runtime one, since by the time this test file
runs in the same process as ``test_deltanet.py``, torch is already imported
regardless of what ``context_budget`` itself does.
"""
from __future__ import annotations

import pathlib

import pytest

import iridium.runtime.context_budget as cb
from iridium.config import get_config


def test_module_source_never_imports_torch():
    src = pathlib.Path(cb.__file__).read_text()
    assert "import torch" not in src
    assert "from torch" not in src


def test_hybrid_schedule_tiles_and_truncates():
    assert cb.hybrid_schedule(4, ("deltanet", "deltanet", "deltanet", "global")) == [
        "deltanet", "deltanet", "deltanet", "global",
    ]
    assert cb.hybrid_schedule(10, ("deltanet", "deltanet", "deltanet", "global")) == [
        "deltanet", "deltanet", "deltanet", "global",
        "deltanet", "deltanet", "deltanet", "global",
        "deltanet", "deltanet",
    ]
    assert cb.hybrid_schedule(3, ("local",) * 5 + ("global",)) == ["local", "local", "local"]


def test_hybrid_schedule_rejects_bad_input():
    with pytest.raises(ValueError):
        cb.hybrid_schedule(0, ("global",))
    with pytest.raises(ValueError):
        cb.hybrid_schedule(4, ())
    with pytest.raises(ValueError):
        cb.hybrid_schedule(4, ("not-a-real-type",))


def test_full_attention_cache_grows_linearly_with_tokens():
    a = cb.full_attention_cache_bytes(1000, d_kv=64, n_layers=8, bytes_per_element="bf16")
    b = cb.full_attention_cache_bytes(2000, d_kv=64, n_layers=8, bytes_per_element="bf16")
    assert b == 2 * a == 2 * 64 * 2 * 8 * 2000


def test_sliding_window_cache_saturates_at_window():
    below = cb.sliding_window_cache_bytes(500, window=1024, d_kv=32, n_layers=4)
    at = cb.sliding_window_cache_bytes(1024, window=1024, d_kv=32, n_layers=4)
    above = cb.sliding_window_cache_bytes(1_048_576, window=1024, d_kv=32, n_layers=4)
    assert below < at == above
    # window - 1: what GroupedQueryAttention's cache actually keeps.
    assert at == 2 * 32 * 2 * 4 * 1023


def test_sliding_window_formula_agrees_with_the_config_accounting():
    """Two implementations of one quantity; this keeps them from drifting.

    The config's copy was fixed to window - 1 first and this one was missed;
    the model's real cache is the arbiter (tests/unit/test_core_blocks.py).
    """
    import dataclasses

    from iridium.config import get_config

    core = dataclasses.replace(get_config("tiny").core, layer_pattern=("local",),
                               local_window=64)
    for tokens in (10, 63, 64, 5000):
        assert core.cache_bytes(tokens) == cb.sliding_window_cache_bytes(
            tokens, window=64, d_kv=core.d_kv, n_layers=core.n_layers)


def test_deltanet_state_is_independent_of_tokens():
    small = cb.deltanet_state_bytes(n_heads=8, d_head=64, n_layers=6)
    # No `tokens` argument exists to grow -- assert the formula itself, and
    # that two very different implied context lengths give one call each.
    assert small == (8 * 64 * 64 + 3 * 8 * 64 * 3) * 2 * 6


def test_mla_requires_latent_dim_or_formula():
    with pytest.raises(ValueError):
        cb.mla_latent_cache_bytes(1_000_000, n_layers=4)
    via_dim = cb.mla_latent_cache_bytes(1000, n_layers=2, latent_dim=16)
    assert via_dim == 16 * 2 * 2 * 1000
    via_formula = cb.mla_latent_cache_bytes(1000, n_layers=2, formula=lambda t, n, b: t * n * b)
    assert via_formula == 1000 * 2 * 2  # bf16 -> 2 bytes


@pytest.mark.parametrize("tokens", [1024, 131_072, 1_048_576])
def test_deltanet_beats_full_attention_at_scale(tokens):
    full = cb.layer_cache_bytes("global", tokens, d_kv=64)
    delta = cb.layer_cache_bytes("deltanet", tokens, n_heads=8, d_head=8)
    if tokens > 8:
        assert delta < full


def test_schedule_cache_bytes_sums_layers():
    schedule = ["global", "deltanet", "deltanet"]
    total = cb.schedule_cache_bytes(schedule, 1000, d_kv=32, n_heads=4, d_head=8)
    expected = (
        cb.full_attention_cache_bytes(1000, 32)
        + 2 * cb.deltanet_state_bytes(4, 8)
    )
    assert total == expected


def test_flops_ordering_matches_cache_ordering():
    tokens, d_model = 100_000, 512
    full = cb.full_attention_flops(tokens, d_model)
    window = cb.sliding_window_flops(tokens, 1024, d_model)
    delta = cb.deltanet_flops(tokens, n_heads=8, d_head=64)
    assert delta < window < full


def test_schedule_flops_rejects_unknown_layer_type():
    with pytest.raises(ValueError):
        cb.schedule_flops(["not-a-type"], 100, d_model=64)


# -- budget()/fits() against a real ladder rung ------------------------------

@pytest.mark.parametrize("rung", ["nano", "nano100m", "test1b"])
def test_budget_hybrid_shrinks_core_cache_at_1m_tokens(rung):
    cfg = get_config(rung)
    n = cfg.core.n_layers
    baseline = ["global"] * n
    hybrid = cb.hybrid_schedule(n, ("deltanet", "deltanet", "deltanet", "global"))
    tokens = 1_048_576
    base = cb.budget(cfg, baseline, tokens, bytes_per_element="bf16")
    hyb = cb.budget(cfg, hybrid, tokens, bytes_per_element="bf16")
    # The core cache shrinks by roughly the fraction of layers that became
    # DeltaNet; the stack term (untouched by core_schedule) is identical.
    assert hyb.core_cache_bytes < base.core_cache_bytes
    assert hyb.stack_cache_bytes == base.stack_cache_bytes
    assert hyb.total_bytes < base.total_bytes


def test_budget_rejects_schedule_of_wrong_length():
    cfg = get_config("nano")
    with pytest.raises(ValueError):
        cb.budget(cfg, ["global"] * (cfg.core.n_layers - 1), 1000)


def test_fits_is_consistent_with_budget():
    cfg = get_config("nano")
    schedule = ["global"] * cfg.core.n_layers
    tokens = 1_048_576
    report = cb.budget(cfg, schedule, tokens, bytes_per_element="bf16")
    small_gb = report.total_bytes / 1e9 / 2
    large_gb = report.total_bytes / 1e9 * 2
    assert not cb.fits(cfg, schedule, tokens, small_gb, bytes_per_element="bf16")
    assert cb.fits(cfg, schedule, tokens, large_gb, bytes_per_element="bf16")


def test_layer_cache_bytes_validates_required_kwargs():
    with pytest.raises(ValueError):
        cb.layer_cache_bytes("global", 1000)  # missing d_kv
    with pytest.raises(ValueError):
        cb.layer_cache_bytes("local", 1000, d_kv=32)  # missing window
    with pytest.raises(ValueError):
        cb.layer_cache_bytes("deltanet", 1000)  # missing n_heads/d_head
    with pytest.raises(ValueError):
        cb.layer_cache_bytes("bogus", 1000)
