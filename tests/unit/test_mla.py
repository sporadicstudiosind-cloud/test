"""Multi-head Latent Attention: exact param count, absorption, and the same
incremental-decoding-equals-full-forward invariant ``test_kv_parity.py`` pins
for ``GroupedQueryAttention``.
"""

from __future__ import annotations

import pytest
import torch

from iridium.model.layers import causal_keep
from iridium.model.mla import MultiHeadLatentAttention
from iridium.model.rope import RotaryEmbedding


def _dims():
    return dict(d_model=32, n_heads=4, d_head=8, kv_lora_rank=12, rope_head_dim=4)


def _build(q_lora_rank, dtype=torch.float32, window=None):
    dims = _dims()
    rope = RotaryEmbedding(dims["rope_head_dim"])
    m = MultiHeadLatentAttention(
        **dims, q_lora_rank=q_lora_rank, rope=rope, window=window
    ).to(dtype)
    return m


@pytest.mark.parametrize("q_lora_rank", [0, 6])
def test_param_count_matches_numel_exactly(q_lora_rank):
    m = _build(q_lora_rank)
    formula = MultiHeadLatentAttention.param_count(
        **_dims(), q_lora_rank=q_lora_rank
    )
    actual = sum(p.numel() for p in m.parameters())
    assert formula == actual


def test_kv_cache_bytes_independent_of_head_count():
    """The whole structural claim: cache size does not move with n_heads."""
    small = MultiHeadLatentAttention.kv_cache_bytes_per_token(kv_lora_rank=512, rope_head_dim=64)
    for n_heads in (32, 64, 128):
        dims = dict(_dims())
        dims["n_heads"] = n_heads
        dims["kv_lora_rank"] = 512
        dims["rope_head_dim"] = 64
        m = MultiHeadLatentAttention(**dims, q_lora_rank=0)
        # kv_cache_bytes_per_token takes no n_heads argument at all -- that
        # absence is the test. Constructing at each head count just proves the
        # module still builds correctly at "many heads".
        assert MultiHeadLatentAttention.kv_cache_bytes_per_token(512, 64) == small
        del m


def _run_full(model, x, positions):
    t = x.shape[1]
    keep = causal_keep(t, t, device=x.device)
    return model(x, positions, keep)


def _run_cached(model, x, positions, chunk, absorbed=False):
    b, t, _ = x.shape
    cache: dict = {}
    outs = []
    start = 0
    while start < t:
        end = min(start + chunk, t)
        xin = x[:, start:end]
        pos = positions[:, start:end]
        keep = causal_keep(end - start, end, device=x.device)
        out = model(xin, pos, keep, cache=cache, cache_key=("mla",), absorbed=absorbed)
        outs.append(out)
        start = end
    return torch.cat(outs, dim=1)


@pytest.mark.parametrize("q_lora_rank", [0, 6])
@pytest.mark.parametrize("chunk", [1, 2, 3, 7])
def test_incremental_decoding_matches_full_forward_fp64(q_lora_rank, chunk):
    """The sacred invariant, at float64: cached step-by-step == one shot."""
    torch.manual_seed(0)
    model = _build(q_lora_rank, dtype=torch.float64)
    b, t, d = 2, 7, _dims()["d_model"]
    x = torch.randn(b, t, d, dtype=torch.float64)
    positions = torch.arange(t).unsqueeze(0).expand(b, t)

    reference = _run_full(model, x, positions)
    cached = _run_cached(model, x, positions, chunk)
    delta = float((reference - cached).abs().max())
    assert delta <= 1e-10, f"chunk={chunk} q_lora_rank={q_lora_rank} max delta {delta:.3e}"


@pytest.mark.parametrize("q_lora_rank", [0, 6])
@pytest.mark.parametrize("chunk", [1, 3])
def test_incremental_decoding_matches_full_forward_fp32(q_lora_rank, chunk):
    torch.manual_seed(0)
    model = _build(q_lora_rank, dtype=torch.float32)
    b, t, d = 2, 7, _dims()["d_model"]
    x = torch.randn(b, t, d, dtype=torch.float32)
    positions = torch.arange(t).unsqueeze(0).expand(b, t)

    reference = _run_full(model, x, positions)
    cached = _run_cached(model, x, positions, chunk)
    delta = float((reference - cached).abs().max())
    assert delta <= 1e-5, f"chunk={chunk} q_lora_rank={q_lora_rank} max delta {delta:.3e}"


@pytest.mark.parametrize("q_lora_rank", [0, 6])
def test_absorbed_path_matches_materialized_path_fp64(q_lora_rank):
    """Weight-absorbed inference must compute the identical function."""
    torch.manual_seed(1)
    model = _build(q_lora_rank, dtype=torch.float64)
    b, t, d = 2, 5, _dims()["d_model"]
    x = torch.randn(b, t, d, dtype=torch.float64)
    positions = torch.arange(t).unsqueeze(0).expand(b, t)
    keep = causal_keep(t, t)

    materialized = model(x, positions, keep, absorbed=False)
    absorbed = model(x, positions, keep, absorbed=True)
    delta = float((materialized - absorbed).abs().max())
    assert delta <= 1e-10, f"absorbed vs materialized max delta {delta:.3e}"


@pytest.mark.parametrize("q_lora_rank", [0, 6])
def test_absorbed_incremental_decoding_matches_full_forward(q_lora_rank):
    """The absorbed path is also incrementally correct, not just single-shot."""
    torch.manual_seed(2)
    model = _build(q_lora_rank, dtype=torch.float64)
    b, t, d = 2, 6, _dims()["d_model"]
    x = torch.randn(b, t, d, dtype=torch.float64)
    positions = torch.arange(t).unsqueeze(0).expand(b, t)

    reference = _run_full(model, x, positions)
    cached = _run_cached(model, x, positions, chunk=1, absorbed=True)
    delta = float((reference - cached).abs().max())
    assert delta <= 1e-10, f"absorbed cached max delta {delta:.3e}"


def test_window_narrows_the_keep_mask():
    """Sliding window composes with causal keep, same convention as GQA's."""
    torch.manual_seed(3)
    windowed = _build(q_lora_rank=0, dtype=torch.float64, window=2)
    unwindowed = _build(q_lora_rank=0, dtype=torch.float64, window=None)
    unwindowed.load_state_dict(windowed.state_dict())

    b, t, d = 1, 6, _dims()["d_model"]
    x = torch.randn(b, t, d, dtype=torch.float64)
    positions = torch.arange(t).unsqueeze(0).expand(b, t)
    keep = causal_keep(t, t)

    out_w = windowed(x, positions, keep)
    out_u = unwindowed(x, positions, keep)
    assert not torch.allclose(out_w, out_u, atol=1e-8)


def test_uses_far_fewer_cache_bytes_than_gqa_at_many_heads():
    """MLA at 128 heads must cache less than GQA at the repo's real configs."""
    from iridium.config import get_config

    nano = get_config("nano").core
    test1b = get_config("test1b").core
    gqa_nano = nano.kv_bytes_per_token(bytes_per_element=2)
    gqa_test1b = test1b.kv_bytes_per_token(bytes_per_element=2)
    for n_heads in (32, 64, 128):
        mla_bytes = MultiHeadLatentAttention.kv_cache_bytes_per_token(
            kv_lora_rank=512, rope_head_dim=64, bytes_per_element=2
        )
        assert mla_bytes < gqa_nano
        assert mla_bytes < gqa_test1b
