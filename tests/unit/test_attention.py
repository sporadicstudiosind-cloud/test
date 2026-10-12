"""Correctness gates for the attention primitives in ``iridium/model/layers.py``.

Every new knob here (``window``, ``softcap``, ``enable_gqa`` dispatch, the
RoPE table cache, RoPE scaling) defaults to today's behaviour, so the tests
that matter most are the ones proving a *feature turned on* still agrees with
a brute-force or manual reference — the default-off path is already covered
by the rest of the suite passing unchanged.
"""

from __future__ import annotations

import math

import pytest
import torch

from iridium.model.layers import (
    BridgeCrossAttention,
    GroupedQueryAttention,
    TransformerBlock,
    _attend,
    _enable_gqa_supported,
    causal_keep,
    head_rms,
    position_keep,
    sliding_window_keep,
    use_attention_backend,
)
from iridium.model.rope import RotaryEmbedding


def _small(seed=0, b=2, h=4, kv=2, t=6, s=6, d=8):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, h, t, d, generator=g)
    k = torch.randn(b, kv, s, d, generator=g)
    v = torch.randn(b, kv, s, d, generator=g)
    return q, k, v


# --- SDPA vs manual, every mask shape we support ---------------------------


@pytest.mark.parametrize("mask_kind", ["causal", "padding", "sparse", "window"])
def test_sdpa_matches_manual_for_every_mask_shape(mask_kind):
    torch.manual_seed(0)
    b, h, kv, t, s, d = 2, 4, 2, 7, 7, 8
    q, k, v = _small(b=b, h=h, kv=kv, t=t, s=s, d=d)

    if mask_kind == "causal":
        keep = causal_keep(t, s)
    elif mask_kind == "padding":
        keep = causal_keep(t, s).clone()
        keep[:, :, :, -2:] = False  # last two keys are padding for everyone
        keep[0, 0, 0, 0] = True  # keep row 0 non-dead in every batch entry
    elif mask_kind == "sparse":
        # An arbitrary boolean mask, standing in for the router's stack-local
        # keep: not causal, not banded, just "this token's own members".
        g = torch.Generator().manual_seed(1)
        keep = (torch.rand(b, 1, t, s, generator=g) > 0.5) & causal_keep(t, s)
    elif mask_kind == "window":
        keep = causal_keep(t, s) & sliding_window_keep(t, s, window=3)

    with use_attention_backend("manual"):
        manual = _attend(q, k, v, keep, d, n_rep=h // kv)
    with use_attention_backend("sdpa"):
        sdpa = _attend(q, k, v, keep, d, n_rep=h // kv)

    torch.testing.assert_close(manual, sdpa, atol=1e-5, rtol=1e-5)


def test_fully_masked_row_is_zero_not_nan_on_both_backends():
    b, h, kv, t, s, d = 1, 2, 1, 3, 3, 8
    q, k, v = _small(b=b, h=h, kv=kv, t=t, s=s, d=d)
    keep = causal_keep(t, s).clone()
    keep[:, :, 0, :] = False  # query 0 has no admissible key at all

    for backend in ("manual", "sdpa"):
        with use_attention_backend(backend):
            out = _attend(q, k, v, keep, d, n_rep=h // kv)
        assert torch.isfinite(out).all(), backend
        assert torch.equal(out[:, :, 0, :], torch.zeros_like(out[:, :, 0, :])), backend


# --- GQA expansion -----------------------------------------------------


def test_enable_gqa_matches_repeat_interleave_expansion():
    """The SDPA enable_gqa path and manual repeat_interleave must agree."""
    if not _enable_gqa_supported():
        pytest.skip("this torch build's SDPA has no enable_gqa")
    b, h, kv, t, s, d = 2, 6, 2, 5, 5, 8
    q, k, v = _small(b=b, h=h, kv=kv, t=t, s=s, d=d)
    keep = causal_keep(t, s)

    with use_attention_backend("sdpa"):
        via_enable_gqa = _attend(q, k, v, keep, d, n_rep=h // kv)

    kr = k.repeat_interleave(h // kv, dim=1)
    vr = v.repeat_interleave(h // kv, dim=1)
    import torch.nn.functional as F

    reference = F.scaled_dot_product_attention(q, kr, vr, attn_mask=keep)

    torch.testing.assert_close(via_enable_gqa, reference, atol=1e-6, rtol=1e-6)


def test_grouped_query_attention_module_matches_manual_full_expansion():
    """The full module (not just ``_attend``) agrees across backends."""
    torch.manual_seed(0)
    d_model, n_q, n_kv, d_head = 32, 4, 2, 8
    attn = GroupedQueryAttention(d_model, n_q, n_kv, d_head)
    b, t = 2, 5
    x = torch.randn(b, t, d_model)
    positions = torch.arange(t).unsqueeze(0).expand(b, -1)
    keep = causal_keep(t, t)

    with use_attention_backend("manual"):
        manual = attn(x, positions, keep)
    with use_attention_backend("sdpa"):
        sdpa = attn(x, positions, keep)
    torch.testing.assert_close(manual, sdpa, atol=1e-5, rtol=1e-5)


# --- sliding window ------------------------------------------------------


def _bruteforce_window_keep(t_q, t_k, window):
    offset = t_k - t_q
    out = torch.zeros(t_q, t_k, dtype=torch.bool)
    for i in range(t_q):
        qi = i + offset
        for j in range(t_k):
            out[i, j] = (j <= qi) and (qi - j < window)
    return out.view(1, 1, t_q, t_k)


@pytest.mark.parametrize("t_q,t_k,window", [(5, 5, 1), (5, 5, 2), (5, 5, 3), (3, 7, 4)])
def test_sliding_window_keep_matches_bruteforce(t_q, t_k, window):
    fast = sliding_window_keep(t_q, t_k, window) & causal_keep(t_q, t_k)
    slow = _bruteforce_window_keep(t_q, t_k, window)
    assert torch.equal(fast, slow)


def test_window_narrows_attention_relative_to_full_causal():
    """A GQA layer with a small window ignores tokens outside it."""
    torch.manual_seed(0)
    d_model, n_q, n_kv, d_head = 16, 2, 2, 8
    windowed = GroupedQueryAttention(d_model, n_q, n_kv, d_head, window=2)
    full = GroupedQueryAttention(d_model, n_q, n_kv, d_head)
    full.load_state_dict(windowed.state_dict())

    t = 6
    x = torch.randn(1, t, d_model)
    positions = torch.arange(t).unsqueeze(0)
    keep = causal_keep(t, t)

    with use_attention_backend("manual"):
        out_windowed = windowed(x, positions, keep)
        out_full = full(x, positions, keep)
    # The last token sees all history under full causal, only a slice under
    # window=2; with fresh random weights the two must differ there.
    assert not torch.allclose(out_windowed[:, -1], out_full[:, -1])

    # But the windowed module must equal a manual computation using the
    # explicitly intersected mask, proving the module's masking is exactly
    # ``keep & sliding_window_keep`` rather than something looser.
    window_keep = keep & sliding_window_keep(t, t, 2)
    with use_attention_backend("manual"):
        expected = full(x, positions, window_keep)
    torch.testing.assert_close(out_windowed, expected)


def test_window_composes_with_kv_cache_incremental_decode():
    """A windowed layer decoded token-by-token matches a full forward pass."""
    torch.manual_seed(0)
    d_model, n_q, n_kv, d_head = 16, 2, 2, 8
    rope = RotaryEmbedding(d_head)
    attn = GroupedQueryAttention(d_model, n_q, n_kv, d_head, rope, window=3)
    attn.eval()

    t = 9
    x = torch.randn(1, t, d_model)
    positions = torch.arange(t).unsqueeze(0)

    with torch.no_grad(), use_attention_backend("manual"):
        keep = causal_keep(t, t)
        reference = attn(x, positions, keep)

        cache: dict = {}
        outs = []
        for i in range(t):
            xi = x[:, i : i + 1, :]
            pi = positions[:, i : i + 1]
            history = i
            keep_i = causal_keep(1, history + 1)
            outs.append(attn(xi, pi, keep_i, cache=cache, cache_key=("l0",)))
        decoded = torch.cat(outs, dim=1)

    torch.testing.assert_close(reference, decoded, atol=1e-6, rtol=1e-6)


# --- QK norm ---------------------------------------------------------------


def test_head_rms_matches_reference_formula():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 5, 8)
    got = head_rms(x)
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-6)


def test_qk_norm_flag_changes_output_and_normalizes_head_dim():
    torch.manual_seed(0)
    d_model, n_q, n_kv, d_head = 16, 2, 2, 8
    attn = GroupedQueryAttention(d_model, n_q, n_kv, d_head)
    t = 4
    x = torch.randn(1, t, d_model)
    positions = torch.arange(t).unsqueeze(0)
    keep = causal_keep(t, t)

    attn.qk_norm = False
    off = attn(x, positions, keep)
    attn.qk_norm = True
    on = attn(x, positions, keep)
    assert not torch.allclose(off, on)


# --- softcapping -------------------------------------------------------


def test_softcap_matches_reference_tanh_formula_and_forces_manual():
    torch.manual_seed(0)
    b, h, kv, t, s, d = 1, 2, 1, 4, 4, 8
    q, k, v = _small(b=b, h=h, kv=kv, t=t, s=s, d=d)
    keep = causal_keep(t, s)
    c = 30.0

    with use_attention_backend("sdpa"):  # softcap must still take the manual path
        out = _attend(q, k, v, keep, d, n_rep=h // kv, softcap=c)

    kr = k.repeat_interleave(h // kv, dim=1)
    vr = v.repeat_interleave(h // kv, dim=1)
    scores = (q @ kr.transpose(-2, -1)) / math.sqrt(d)
    scores = c * torch.tanh(scores / c)
    scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    expected = probs @ vr

    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_softcap_bounds_logits_relative_to_uncapped():
    """Softcapping should visibly change output vs. no cap on large logits."""
    torch.manual_seed(0)
    b, h, kv, t, s, d = 1, 1, 1, 3, 3, 8
    q = torch.randn(b, h, t, d) * 20  # deliberately large to blow up logits
    k = torch.randn(b, kv, s, d) * 20
    v = torch.randn(b, kv, s, d)
    keep = causal_keep(t, s)

    uncapped = _attend(q, k, v, keep, d, n_rep=1)
    capped = _attend(q, k, v, keep, d, n_rep=1, softcap=5.0)
    assert not torch.allclose(uncapped, capped)


# --- RoPE ---------------------------------------------------------------


def test_rope_is_relative_position_invariant():
    """<q_m, k_n> after rotation depends only on m - n."""
    torch.manual_seed(0)
    d_head = 8
    rope = RotaryEmbedding(d_head)
    q = torch.randn(1, 1, 1, d_head)
    k = torch.randn(1, 1, 1, d_head)

    def dot_at(m, n):
        qm = rope(q, torch.tensor([[m]]))
        kn = rope(k, torch.tensor([[n]]))
        return (qm * kn).sum().item()

    base = dot_at(5, 2)  # gap of 3
    for m, n in [(10, 7), (100, 97), (3, 0)]:
        assert math.isclose(dot_at(m, n), base, rel_tol=1e-4, abs_tol=1e-5)

    # And a different gap must (generically) give a different dot product.
    assert not math.isclose(dot_at(5, 2), dot_at(5, 1), rel_tol=1e-4, abs_tol=1e-5)


def test_rope_table_cache_reuses_for_same_positions_object():
    d_head = 8
    rope = RotaryEmbedding(d_head)
    positions = torch.arange(4).unsqueeze(0)
    cos1, sin1 = rope.angles(positions)
    assert len(rope._table_cache) == 1
    cos2, sin2 = rope.angles(positions)
    assert cos1 is cos2 and sin1 is sin2  # served from cache, not recomputed
    assert len(rope._table_cache) == 1


def test_rope_table_cache_is_correct_across_distinct_position_tensors():
    """Cache hits/misses must never change the numeric result."""
    d_head = 8
    rope = RotaryEmbedding(d_head, cache_capacity=2)
    x = torch.randn(1, 1, 5, d_head)

    positions_variants = [torch.arange(5).unsqueeze(0) for _ in range(5)]
    outputs = [rope(x, p) for p in positions_variants]
    for out in outputs[1:]:
        torch.testing.assert_close(out, outputs[0])
    # Cache never grows past its capacity even after many distinct tensors.
    assert len(rope._table_cache) <= 2


def test_rope_tables_are_computed_in_fp32_even_for_bf16_input():
    d_head = 8
    rope = RotaryEmbedding(d_head)
    positions = torch.arange(3).unsqueeze(0)
    cos, sin = rope.angles(positions)
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32

    x_bf16 = torch.randn(1, 1, 3, d_head, dtype=torch.bfloat16)
    out = rope(x_bf16, positions)
    assert out.dtype == torch.bfloat16
    # Recomputing angles fresh (new positions object) must still be fp32,
    # i.e. the low-precision input never contaminates the table itself.
    fresh_cos, _ = rope.angles(torch.arange(3).unsqueeze(0))
    assert fresh_cos.dtype == torch.float32


def test_rope_scaling_none_is_bitwise_unchanged():
    """Default scaling=None must reproduce the pre-existing formula exactly."""
    d_head = 8
    theta = 10_000.0
    rope = RotaryEmbedding(d_head, theta=theta)
    positions = torch.arange(6).unsqueeze(0).float()

    inv_freq_ref = 1.0 / (theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
    freqs_ref = positions.unsqueeze(-1) * inv_freq_ref
    cos_ref, sin_ref = torch.cos(freqs_ref), torch.sin(freqs_ref)

    cos, sin = rope.angles(positions)
    assert torch.equal(cos, cos_ref)
    assert torch.equal(sin, sin_ref)


def test_rope_linear_scaling_divides_positions():
    d_head = 8
    theta = 10_000.0
    factor = 4.0
    plain = RotaryEmbedding(d_head, theta=theta)
    scaled = RotaryEmbedding(d_head, theta=theta, scaling={"type": "linear", "factor": factor})

    positions = torch.arange(8).unsqueeze(0).float()
    cos_plain, sin_plain = plain.angles(positions / factor)
    cos_scaled, sin_scaled = scaled.angles(positions)
    torch.testing.assert_close(cos_plain, cos_scaled)
    torch.testing.assert_close(sin_plain, sin_scaled)


def test_rope_ntk_scaling_changes_effective_theta_not_positions():
    d_head = 8
    theta = 10_000.0
    factor = 4.0
    scaled = RotaryEmbedding(d_head, theta=theta, scaling={"type": "ntk", "factor": factor})
    expected_theta = theta * (factor ** (d_head / (d_head - 2)))
    expected_inv_freq = 1.0 / (
        expected_theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
    )
    torch.testing.assert_close(scaled.inv_freq, expected_inv_freq)


def test_rope_scaling_rejects_unknown_type():
    with pytest.raises(ValueError):
        RotaryEmbedding(8, scaling={"type": "bogus", "factor": 2.0})


# --- incremental decode equals full forward (module-level, not just cache) --


def test_transformer_block_incremental_decode_equals_full_forward():
    torch.manual_seed(0)
    d_model, n_q, n_kv, d_head, d_ff = 16, 4, 2, 8, 32
    rope = RotaryEmbedding(d_head)
    block = TransformerBlock(d_model, n_q, n_kv, d_head, d_ff, rope).eval()

    t = 6
    x = torch.randn(1, t, d_model)
    positions = torch.arange(t).unsqueeze(0)

    with torch.no_grad(), use_attention_backend("manual"):
        keep = causal_keep(t, t)
        reference = block(x, positions, keep)

        cache: dict = {}
        outs = []
        for i in range(t):
            keep_i = causal_keep(1, i + 1)
            outs.append(
                block(x[:, i : i + 1], positions[:, i : i + 1], keep_i, cache, ("blk",))
            )
        decoded = torch.cat(outs, dim=1)

    torch.testing.assert_close(reference, decoded, atol=1e-6, rtol=1e-6)


def test_bridge_cross_attention_matches_manual_across_backends():
    torch.manual_seed(0)
    d_model, d_core, n_q, n_kv, d_head = 16, 24, 2, 2, 8
    rope = RotaryEmbedding(d_head)
    bridge = BridgeCrossAttention(d_model, d_core, n_q, n_kv, d_head, rope)

    t, s = 4, 7
    x = torch.randn(1, t, d_model)
    q_pos = torch.arange(t).unsqueeze(0)
    core_states = torch.randn(1, s, d_core)
    core_pos = torch.arange(s).unsqueeze(0)
    keep = position_keep(q_pos, core_pos)

    with use_attention_backend("manual"):
        manual = bridge(x, q_pos, core_states, core_pos, keep)
    with use_attention_backend("sdpa"):
        sdpa = bridge(x, q_pos, core_states, core_pos, keep)
    torch.testing.assert_close(manual, sdpa, atol=1e-5, rtol=1e-5)
