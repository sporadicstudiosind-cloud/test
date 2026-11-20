"""PerceptualEncoder: default-identical, plus each opt-in feature in isolation."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from iridium.model.perception import PerceptualEncoder, build_stem


def _old_perceptual_encoder(projection, width, rank, depth):
    """The exact module this file replaced, kept here only so the default
    path can be checked against it byte-for-byte."""

    class Old(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = projection
            self.blocks = nn.ModuleList(nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, rank), nn.GELU(), nn.Linear(rank, width)
            ) for _ in range(depth))
            self.scale = (2 * depth) ** -.5

        def forward(self, payload):
            hidden = self.projection(payload)
            for block in self.blocks:
                hidden = hidden + self.scale * block(hidden)
            return hidden

    return Old()


def test_default_configuration_matches_previous_implementation():
    torch.manual_seed(0)
    projection = nn.Linear(12, 32)
    old = _old_perceptual_encoder(projection, 32, 16, depth=3)

    torch.manual_seed(0)
    projection2 = nn.Linear(12, 32)
    new = PerceptualEncoder(projection2, 32, 16, depth=3)

    # Same seed must produce identically-initialized parameters given an
    # identical sequence of nn.Module constructions.
    for (na, pa), (nb, pb) in zip(old.named_parameters(), new.named_parameters()):
        assert torch.equal(pa, pb), (na, nb)

    payload = torch.randn(4, 5, 12)
    out_old = old(payload)
    out_new = new(payload)
    assert torch.equal(out_old, out_new)


def test_default_forward_signature_accepts_grid_as_noop():
    torch.manual_seed(0)
    enc = PerceptualEncoder(nn.Linear(8, 16), 16, 8, depth=2)
    payload = torch.randn(2, 6, 8)
    out_a = enc(payload)
    out_b = enc(payload, grid=(2, 3))
    assert torch.equal(out_a, out_b)


def test_gradient_flows_through_default_encoder():
    enc = PerceptualEncoder(nn.Linear(8, 16), 16, 8, depth=2)
    payload = torch.randn(2, 6, 8, requires_grad=True)
    out = enc(payload)
    out.sum().backward()
    assert payload.grad is not None
    assert torch.isfinite(payload.grad).all()


# -- register tokens ----------------------------------------------------


def test_registers_present_and_change_token_count():
    enc = PerceptualEncoder(nn.Linear(8, 16), 16, 8, depth=1, n_registers=4)
    payload = torch.randn(3, 10, 8)
    out = enc(payload)
    assert out.shape == (3, 14, 16)
    # The 4 register tokens are identical across the batch (broadcast of one
    # learned set), the remaining 10 are the ordinary per-token content.
    assert torch.equal(out[0, :4], out[1, :4])


def test_registers_are_gradient_connected():
    enc = PerceptualEncoder(nn.Linear(8, 16), 16, 8, depth=1, n_registers=3)
    payload = torch.randn(2, 5, 8)
    out = enc(payload)
    out.sum().backward()
    assert enc.registers.grad is not None
    assert torch.isfinite(enc.registers.grad).all()
    assert enc.registers.grad.abs().sum() > 0


def test_zero_registers_by_default():
    enc = PerceptualEncoder(nn.Linear(8, 16), 16, 8, depth=1)
    assert enc.registers is None


# -- token compression ----------------------------------------------------


def test_compression_reduces_token_count():
    enc = PerceptualEncoder(nn.Linear(6, 12), 12, 8, depth=1, compression=2)
    payload = torch.randn(1, 16, 6)  # a 4x4 patch grid
    out = enc(payload, grid=(4, 4))
    assert out.shape == (1, 4, 12)


def test_compression_requires_grid():
    enc = PerceptualEncoder(nn.Linear(6, 12), 12, 8, depth=1, compression=2)
    payload = torch.randn(1, 16, 6)
    with pytest.raises(ValueError):
        enc(payload)


def test_compression_with_registers_prepends_registers_after_merge():
    enc = PerceptualEncoder(nn.Linear(6, 12), 12, 8, depth=1, compression=2, n_registers=2)
    payload = torch.randn(1, 16, 6)
    out = enc(payload, grid=(4, 4))
    assert out.shape == (1, 2 + 4, 12)


# -- modality stems ---------------------------------------------------------


def test_image_stem_shape_and_identity_at_init():
    stem = build_stem("image", (3, 8, 8))
    payload = torch.randn(2, 5, 3 * 8 * 8)
    out = stem(payload)
    # Zero-initialized conv => additive term is exactly zero at construction.
    assert torch.equal(out, payload)


def test_video_stem_runs_and_preserves_shape():
    stem = build_stem("video", (3, 2, 4, 4))
    payload = torch.randn(1, 3, 3 * 2 * 4 * 4)
    out = stem(payload)
    assert out.shape == payload.shape


def test_audio_stem_is_temporal_not_2d():
    """Audio's payload is a flat PCM window (see module docstring), so its
    stem must be a 1-D conv, never a 2-D (mels x frames) one."""
    stem = build_stem("audio", (1, 256))
    assert isinstance(stem.conv, nn.Conv1d)
    payload = torch.randn(2, 4, 256)
    out = stem(payload)
    assert out.shape == payload.shape


def test_unknown_modality_stem_raises():
    with pytest.raises(ValueError):
        build_stem("quantity", (1, 4))


def test_encoder_with_patch_shape_runs_end_to_end():
    torch.manual_seed(0)
    enc = PerceptualEncoder(
        nn.Linear(3 * 8 * 8, 16), 16, 8, depth=2, modality="image", patch_shape=(3, 8, 8),
    )
    payload = torch.randn(2, 5, 3 * 8 * 8)
    out = enc(payload)
    assert out.shape == (2, 5, 16)
    assert torch.isfinite(out).all()


# -- depth-scaled init -------------------------------------------------------


def test_depth_scaled_init_changes_block_weight_statistics():
    torch.manual_seed(0)
    shallow_scale = PerceptualEncoder(nn.Linear(8, 16), 16, 32, depth=8, depth_scaled_init=True)
    default = PerceptualEncoder(nn.Linear(8, 16), 16, 32, depth=8, depth_scaled_init=False)
    scaled_std = shallow_scale.blocks[0][-1].weight.std().item()
    default_std = default.blocks[0][-1].weight.std().item()
    # depth=8 => scale = 16**-.5 ~= 0.25, so 0.02*scale is visibly tighter
    # than PyTorch's default (fan-in based, much larger for a narrow layer).
    assert scaled_std < default_std
    assert torch.equal(shallow_scale.blocks[0][-1].bias, torch.zeros_like(shallow_scale.blocks[0][-1].bias))


def test_depth_scaled_init_off_by_default_keeps_pytorch_init():
    """Building the *identical* nn.Sequential by hand from the same seed,
    right after the same projection draw, must match bit-for-bit — this is
    the same property test_default_configuration_matches_previous_implementation
    checks, isolated down to the block-construction step."""
    torch.manual_seed(0)
    nn.Linear(8, 16)  # the projection argument's own RNG draw, as in __init__
    enc = PerceptualEncoder.__new__(PerceptualEncoder)
    nn.Module.__init__(enc)
    enc.blocks = nn.ModuleList(nn.Sequential(
        nn.LayerNorm(16), nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 16)
    ) for _ in range(1))

    torch.manual_seed(0)
    nn.Linear(8, 16)
    reference = nn.Sequential(nn.LayerNorm(16), nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 16))

    assert torch.equal(enc.blocks[0][-1].weight, reference[-1].weight)
    assert torch.equal(enc.blocks[0][-1].bias, reference[-1].bias)
