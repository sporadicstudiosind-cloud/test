"""M-RoPE (axial rotary encoding) correctness.

The property under test is the entire point of using RoPE at all: the
attention logit between two rotated vectors is a function of their
*relative* coordinate, never their absolute one. Everything else here
(reduction to 1-D RoPE, native-resolution invariance, token accounting) is
in service of that property meaning something for a mixed text+media
sequence.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from iridium.codecs.spatial import (
    AxialRotaryEmbedding,
    compressed_token_count,
    grid_coordinates,
    interpolate_coordinates,
    pixel_shuffle_merge,
    text_coordinates,
)
from iridium.model.rope import RotaryEmbedding


def _dot(rope, q, k, q_coord, k_coord):
    rq = rope(q, q_coord)
    rk = rope(k, k_coord)
    return (rq * rk).sum(-1)


def test_relative_position_property_single_axis():
    """Same Δt, different absolute t: identical attention logit."""
    torch.manual_seed(0)
    rope = AxialRotaryEmbedding(d_head=16, axes=("t",))
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    for base_q, base_k in [(0.0, 3.0), (100.0, 103.0), (-50.0, -47.0)]:
        q_coord = torch.tensor([[[base_q]]]).unsqueeze(-1)
        k_coord = torch.tensor([[[base_k]]]).unsqueeze(-1)
        logit = _dot(rope, q, k, q_coord, k_coord)
        if base_q == 0.0:
            reference = logit
        else:
            assert torch.allclose(logit, reference, atol=1e-4), (base_q, base_k)


def test_relative_position_property_3d():
    """Same (Δt, Δy, Δx) at different absolute (t, y, x): identical logit.

    This is the hard version of the property: three independent axes, each
    rotating a disjoint chunk of the head dimension, all held to the same
    relative offset while the absolute anchor point is moved around,
    including through negative coordinates and non-integer pixel offsets.
    """
    torch.manual_seed(1)
    d_head = 24
    rope = AxialRotaryEmbedding(d_head=d_head, axes=("t", "y", "x"))
    q = torch.randn(2, 3, 5, d_head)
    k = torch.randn(2, 3, 5, d_head)
    delta = torch.tensor([2.0, 37.5, -12.0])

    anchors = [
        torch.zeros(3),
        torch.tensor([10.0, 512.0, 300.0]),
        torch.tensor([-40.0, -8.0, 1024.5]),
    ]
    logits = []
    for anchor in anchors:
        q_coord = anchor.view(1, 1, 3).expand(2, 5, 3)
        k_coord = (anchor + delta).view(1, 1, 3).expand(2, 5, 3)
        logits.append(_dot(rope, q, k, q_coord, k_coord))
    for other in logits[1:]:
        assert torch.allclose(logits[0], other, atol=1e-3)

    # Sanity: the property is not vacuous — a different delta gives a
    # different logit almost everywhere.
    other_delta = torch.tensor([2.0, 37.5, 250.0])
    q_coord = anchors[0].view(1, 1, 3).expand(2, 5, 3)
    k_coord = (anchors[0] + other_delta).view(1, 1, 3).expand(2, 5, 3)
    changed = _dot(rope, q, k, q_coord, k_coord)
    assert not torch.allclose(logits[0], changed, atol=1e-2)


def test_reduces_to_1d_rope_for_text():
    """Feeding the same scalar position on every axis reproduces model/rope.py exactly."""
    torch.manual_seed(2)
    d_head = 32
    base = RotaryEmbedding(d_head)
    axial = AxialRotaryEmbedding(d_head, axes=("t", "y", "x"))

    x = torch.randn(2, 4, 7, d_head)
    positions = torch.arange(7).unsqueeze(0).expand(2, 7)

    out_base = base(x, positions)
    out_axial = axial(x, text_coordinates(positions, n_axes=3))
    assert torch.allclose(out_base, out_axial, atol=1e-5)


def test_single_axis_reduction_is_literal():
    """axes=('t',) IS model/rope.py's RotaryEmbedding, not merely close to it."""
    torch.manual_seed(3)
    d_head = 16
    base = RotaryEmbedding(d_head, theta=10_000.0)
    axial = AxialRotaryEmbedding(d_head, axes=("t",), theta=10_000.0)
    assert torch.allclose(base.inv_freq, axial.inv_freq)

    x = torch.randn(1, 2, 5, d_head)
    positions = torch.arange(5).unsqueeze(0)
    out_base = base(x, positions)
    out_axial = axial(x, text_coordinates(positions, n_axes=1))
    assert torch.allclose(out_base, out_axial, atol=1e-6)


def test_native_resolution_coordinates_are_resolution_independent():
    """A patch at the same physical pixel location gets the same coordinate
    regardless of the total image size — the property a normalized
    grid-index scheme (index / grid_size) would violate."""
    patch = 8
    small_grid = (8, 8)    # 64x64 image at patch 8
    large_grid = (16, 16)  # 128x128 image at patch 8
    small = grid_coordinates(small_grid, patch)
    large = grid_coordinates(large_grid, patch)

    # Patch (row=4, col=4) exists in both grids at pixel location (36, 36).
    row, col = 4, 4
    small_idx = row * small_grid[1] + col
    large_idx = row * large_grid[1] + col
    assert np.allclose(small[small_idx], large[large_idx])
    assert np.allclose(small[small_idx][1:], [36.0, 36.0])


def test_native_resolution_invariance_in_attention():
    """The same physical patch pair yields the same rotated dot product
    whether it sits inside a 512x512 tile or a 1024x768 tile."""
    torch.manual_seed(4)
    patch = 8
    rope = AxialRotaryEmbedding(d_head=16, axes=("t", "y", "x"))
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    small_grid = (64, 64)   # 512x512 / patch 8
    large_grid = (128, 96)  # 1024x768 / patch 8
    coords_small = grid_coordinates(small_grid, patch)
    coords_large = grid_coordinates(large_grid, patch)

    i_row, i_col = 4, 4
    j_row, j_col = 5, 6
    i_small, j_small = i_row * small_grid[1] + i_col, j_row * small_grid[1] + j_col
    i_large, j_large = i_row * large_grid[1] + i_col, j_row * large_grid[1] + j_col
    assert np.allclose(coords_small[i_small][1:], coords_large[i_large][1:])
    assert np.allclose(coords_small[j_small][1:], coords_large[j_large][1:])

    q_coord = torch.from_numpy(coords_small[i_small]).view(1, 1, 1, 3)
    k_coord = torch.from_numpy(coords_small[j_small]).view(1, 1, 1, 3)
    logit_small = _dot(rope, q, k, q_coord, k_coord)

    q_coord = torch.from_numpy(coords_large[i_large]).view(1, 1, 1, 3)
    k_coord = torch.from_numpy(coords_large[j_large]).view(1, 1, 1, 3)
    logit_large = _dot(rope, q, k, q_coord, k_coord)
    assert torch.allclose(logit_small, logit_large, atol=1e-4)


def test_interpolate_coordinates_is_noop_within_training_extent():
    coords = torch.tensor([0.0, 100.0, 500.0])
    out = interpolate_coordinates(coords, train_extent=1024.0, native_extent=800.0)
    assert torch.equal(out, coords)


def test_interpolate_coordinates_rescales_beyond_training_extent():
    coords = torch.tensor([0.0, 1000.0, 2000.0])
    out = interpolate_coordinates(coords, train_extent=1024.0, native_extent=2048.0)
    assert torch.allclose(out, coords * 0.5)


@pytest.mark.parametrize(
    "height,width,patch,compression,expected",
    [
        (224, 224, 8, 1, 28 * 28),
        (512, 512, 8, 1, 64 * 64),
        (512, 512, 8, 2, 32 * 32),
        (512, 512, 8, 4, 16 * 16),
        (1024, 768, 8, 1, 128 * 96),
        (1024, 768, 8, 2, 64 * 48),
        (1024, 768, 8, 4, 32 * 24),
    ],
)
def test_token_count_accounting(height, width, patch, compression, expected):
    assert compressed_token_count(height, width, patch, compression) == expected


def test_token_count_rejects_indivisible_compression():
    with pytest.raises(ValueError):
        compressed_token_count(100, 100, 10, compression=3)


def test_pixel_shuffle_merge_shape_and_grid():
    grid = (8, 8)
    x = torch.randn(2, 64, 6)
    merged, new_grid = pixel_shuffle_merge(x, grid, factor=2)
    assert new_grid == (4, 4)
    assert merged.shape == (2, 16, 24)


def test_pixel_shuffle_merge_preserves_all_values():
    """Merging must be a lossless reshuffle: every input scalar appears
    exactly once in the merged tensor (concatenation, not pooling)."""
    grid = (4, 4)
    x = torch.arange(16 * 3, dtype=torch.float32).reshape(1, 16, 3)
    merged, new_grid = pixel_shuffle_merge(x, grid, factor=2)
    assert new_grid == (2, 2)
    assert merged.shape == (1, 4, 12)
    assert torch.equal(torch.sort(merged.flatten())[0], torch.sort(x.flatten())[0])


def test_pixel_shuffle_merge_factor_one_is_identity():
    x = torch.randn(3, 9, 5)
    merged, grid = pixel_shuffle_merge(x, (3, 3), factor=1)
    assert grid == (3, 3)
    assert torch.equal(merged, x)


def test_pixel_shuffle_merge_video_leaves_time_axis_alone():
    grid = (2, 4, 4)  # t, h, w
    x = torch.randn(1, 32, 3)
    merged, new_grid = pixel_shuffle_merge(x, grid, factor=2)
    assert new_grid == (2, 2, 2)
    assert merged.shape == (1, 8, 12)
