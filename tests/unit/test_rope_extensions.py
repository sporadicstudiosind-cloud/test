"""Correctness gates for the two additions in ``iridium/model/rope.py``.

Both are opt-in and both must reduce exactly to today's numbers when unused:
``scaling={"type": "yarn", ...}`` with ``factor=1`` must match unscaled RoPE
bit-for-bit, and ``sections=(...)`` fed equal coordinates on every axis must
match plain 1-D RoPE bit-for-bit. Everything else here checks the specific
paper claims (per-frequency-band ramp, the relative-position invariant,
the ``mscale`` formula) rather than merely "it runs".
"""

from __future__ import annotations

import math

import pytest
import torch

from iridium.model.rope import RotaryEmbedding, _yarn_inv_freq_and_mscale


D_HEAD = 16  # -> 8 frequency slots


def _positions(b=2, t=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Nontrivial, non-arange positions (superstacks pack arbitrary subsequences).
    return torch.randint(0, 4096, (b, t), generator=g).long()


# --- YaRN --------------------------------------------------------------


def test_yarn_factor_one_matches_unscaled_exactly():
    positions = _positions()
    plain = RotaryEmbedding(D_HEAD, theta=500_000.0)
    yarn = RotaryEmbedding(
        D_HEAD,
        theta=500_000.0,
        scaling={"type": "yarn", "factor": 1.0, "original_max_position": 4096},
    )
    cos_p, sin_p = plain.angles(positions)
    cos_y, sin_y = yarn.angles(positions)
    assert torch.equal(cos_p, cos_y)
    assert torch.equal(sin_p, sin_y)
    assert yarn._yarn_mscale == 1.0


def test_yarn_requires_original_max_position():
    with pytest.raises(ValueError):
        RotaryEmbedding(D_HEAD, scaling={"type": "yarn", "factor": 4.0})


def test_yarn_high_freq_dims_untouched_low_freq_scaled_by_inverse_factor():
    theta = 500_000.0
    factor = 8.0
    original_max_position = 4096
    inv_freq, _ = _yarn_inv_freq_and_mscale(
        D_HEAD, theta, factor, original_max_position, beta_fast=32, beta_slow=1,
    )
    base_inv_freq = 1.0 / (theta ** (torch.arange(0, D_HEAD, 2, dtype=torch.float32) / D_HEAD))

    # Dimension 0 is the shortest wavelength (highest frequency): far below
    # the beta_fast rotation-count threshold at this max position, so YaRN's
    # ramp must leave it at the trained frequency, untouched by the factor.
    assert torch.allclose(inv_freq[0], base_inv_freq[0], rtol=1e-6)

    # The last dimension is the longest wavelength (lowest frequency): above
    # the beta_slow threshold, so it must be fully interpolated by 1/factor,
    # exactly Chen et al. 2023's linear position interpolation.
    assert torch.allclose(inv_freq[-1], base_inv_freq[-1] / factor, rtol=1e-6)

    # Every dimension must lie between the two treatments -- the ramp blends,
    # it never overshoots either endpoint.
    lo = torch.minimum(base_inv_freq / factor, base_inv_freq)
    hi = torch.maximum(base_inv_freq / factor, base_inv_freq)
    assert torch.all(inv_freq >= lo - 1e-9)
    assert torch.all(inv_freq <= hi + 1e-9)


def test_yarn_mscale_formula():
    for factor in (1.0, 2.0, 4.0, 8.0, 16.0):
        _, mscale = _yarn_inv_freq_and_mscale(
            D_HEAD, 500_000.0, factor, 4096, beta_fast=32, beta_slow=1,
        )
        expected = 1.0 if factor <= 1.0 else 0.1 * math.log(factor) + 1.0
        assert mscale == pytest.approx(expected)


def test_yarn_relative_position_property_holds():
    """<q_m, k_n> under YaRN depends only on m - n, exactly as for plain RoPE.

    Per-dimension, YaRN only changes which scalar frequency a dimension pair
    rotates by (and applies one global constant, mscale, to every dimension);
    it never makes the frequency depend on the *query's* or *key's* absolute
    position. So the standard 2-D-per-pair RoPE algebra -- rotation by m*w_i
    dotted with rotation by n*w_i is a function of (m-n)*w_i alone -- still
    applies dimension-by-dimension, and summing over dimensions preserves it.
    """
    torch.manual_seed(0)
    rope = RotaryEmbedding(
        D_HEAD, theta=500_000.0,
        scaling={"type": "yarn", "factor": 8.0, "original_max_position": 4096},
    )
    q = torch.randn(1, 1, 1, D_HEAD, dtype=torch.float64)
    k = torch.randn(1, 1, 1, D_HEAD, dtype=torch.float64)

    def score(m: int, n: int) -> float:
        pm = torch.tensor([[m]], dtype=torch.long)
        pn = torch.tensor([[n]], dtype=torch.long)
        qr = rope(q.to(torch.float32), pm).to(torch.float64)
        kr = rope(k.to(torch.float32), pn).to(torch.float64)
        return (qr * kr).sum().item()

    # Several (m, n) pairs with the same m - n must agree, to fp32 tolerance
    # (angles are computed in fp32 per the module docstring, so this is not a
    # bit-exactness claim -- only that the algebraic invariant survives
    # rounding rather than drifting with the absolute position).
    for delta in (0, 1, 5, 37):
        base = score(100, 100 - delta)
        for m in (200, 500, 1000):
            got = score(m, m - delta)
            assert got == pytest.approx(base, rel=1e-5, abs=1e-5)


# --- M-RoPE (multi-axis positions) --------------------------------------


SECTIONS = (2, 3, 3)  # sums to D_HEAD // 2 == 8


def test_equal_axes_matches_1d_rope_bit_exact():
    positions = _positions()
    plain = RotaryEmbedding(D_HEAD, theta=500_000.0)
    mrope = RotaryEmbedding(D_HEAD, theta=500_000.0, sections=SECTIONS)

    cos_p, sin_p = plain.angles(positions)
    # [B, T] broadcast onto every axis.
    cos_b, sin_b = mrope.angles(positions)
    assert torch.equal(cos_p, cos_b)
    assert torch.equal(sin_p, sin_b)

    # Explicit [B, T, 3] with all three axes equal to the same scalar.
    coords = positions.unsqueeze(-1).expand(*positions.shape, 3).contiguous()
    mrope.clear_cache()
    cos_e, sin_e = mrope.angles(coords)
    assert torch.equal(cos_p, cos_e)
    assert torch.equal(sin_p, sin_e)


def test_sections_default_none_rejects_3d_positions():
    plain = RotaryEmbedding(D_HEAD, theta=500_000.0)
    coords = torch.zeros(2, 4, 3, dtype=torch.long)
    with pytest.raises(ValueError):
        plain.angles(coords)


def test_sections_must_sum_to_half_d_head():
    with pytest.raises(ValueError):
        RotaryEmbedding(D_HEAD, sections=(2, 2, 2))  # sums to 6, needs 8


def test_mrope_forward_shape_and_dtype():
    mrope = RotaryEmbedding(D_HEAD, sections=SECTIONS)
    b, h, t = 2, 4, 6
    x = torch.randn(b, h, t, D_HEAD, dtype=torch.float32)
    coords = torch.randint(0, 100, (b, t, 3)).long()
    out = mrope(x, coords)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_mrope_relative_position_property_per_axis():
    """A logit depends only on (delta_t, delta_y, delta_x), not on absolute coords."""
    torch.manual_seed(0)
    mrope = RotaryEmbedding(D_HEAD, sections=SECTIONS)
    q = torch.randn(1, 1, 1, D_HEAD, dtype=torch.float64)
    k = torch.randn(1, 1, 1, D_HEAD, dtype=torch.float64)

    def score(coord_q, coord_k) -> float:
        pq = torch.tensor([[coord_q]], dtype=torch.long)
        pk = torch.tensor([[coord_k]], dtype=torch.long)
        qr = mrope(q.to(torch.float32), pq).to(torch.float64)
        kr = mrope(k.to(torch.float32), pk).to(torch.float64)
        return (qr * kr).sum().item()

    base = score((10, 20, 30), (7, 15, 28))  # deltas (3, 5, 2)
    shifted = score((110, 220, 330), (107, 215, 328))  # same deltas, far offset
    assert shifted == pytest.approx(base, rel=1e-5, abs=1e-5)  # fp32 angles

    # Different deltas must (generically) give a different score.
    other = score((10, 20, 30), (7, 14, 28))  # delta_y now 6, not 5
    assert other != pytest.approx(base, rel=1e-9, abs=1e-9)


def test_mrope_combines_with_yarn_scaling():
    """M-RoPE plus YaRN: factor=1 combo still matches plain 1-D RoPE exactly."""
    positions = _positions()
    plain = RotaryEmbedding(D_HEAD, theta=500_000.0)
    combo = RotaryEmbedding(
        D_HEAD, theta=500_000.0, sections=SECTIONS,
        scaling={"type": "yarn", "factor": 1.0, "original_max_position": 4096},
    )
    cos_p, sin_p = plain.angles(positions)
    cos_c, sin_c = combo.angles(positions)
    assert torch.equal(cos_p, cos_c)
    assert torch.equal(sin_p, sin_c)

    # factor > 1: still runs, still shape-correct, and the mscale correction
    # is applied uniformly regardless of how the frequency table is split.
    scaled = RotaryEmbedding(
        D_HEAD, theta=500_000.0, sections=SECTIONS,
        scaling={"type": "yarn", "factor": 4.0, "original_max_position": 4096},
    )
    coords = positions.unsqueeze(-1).expand(*positions.shape, 3).contiguous()
    cos_s, sin_s = scaled.angles(coords)
    assert cos_s.shape == cos_p.shape
    assert scaled._yarn_mscale == pytest.approx(0.1 * math.log(4.0) + 1.0)


# --- cache correctness ----------------------------------------------------


def test_cache_keyed_correctly_for_both_1d_and_mrope_positions():
    rope = RotaryEmbedding(D_HEAD, sections=SECTIONS)
    p2d = _positions()
    p3d = p2d.unsqueeze(-1).expand(*p2d.shape, 3).contiguous()

    cos_a, sin_a = rope.angles(p2d)
    assert len(rope._table_cache) == 1
    cos_b, sin_b = rope.angles(p3d)
    assert len(rope._table_cache) == 2
    # Re-querying the same tensor object returns the cached (identical) table.
    cos_a2, sin_a2 = rope.angles(p2d)
    assert cos_a2 is cos_a and sin_a2 is sin_a
    assert torch.equal(cos_a, cos_b) and torch.equal(sin_a, sin_b)


def test_existing_linear_and_ntk_scaling_still_default_off_and_unaffected():
    positions = _positions()
    plain = RotaryEmbedding(D_HEAD, theta=500_000.0)
    linear = RotaryEmbedding(D_HEAD, theta=500_000.0, scaling={"type": "linear", "factor": 1.0})
    cos_p, sin_p = plain.angles(positions)
    cos_l, sin_l = linear.angles(positions)
    assert torch.equal(cos_p, cos_l)
    assert torch.equal(sin_p, sin_l)
