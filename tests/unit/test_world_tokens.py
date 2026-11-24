"""camera_span shape/Plücker values, anchor_coordinates round-trip, scene_span canonical order."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from iridium.world.camera import Camera, look_at
from iridium.world.tokens import anchor_coordinates, camera_span, scene_span


def _cam(eye=(0.0, 0.0, -5.0), target=(0.0, 0.0, 0.0), size=(32, 16)):
    w, h = size
    return Camera.from_fov(60.0, w, h, look_at(eye, target, dtype=torch.float64))


# -- camera_span --------------------------------------------------------------


def test_camera_span_shape_matches_patch_grid():
    cam = _cam(size=(32, 16))
    span = camera_span(cam, patch=8)
    assert span.grid == (2, 4)          # (h // patch, w // patch)
    assert len(span) == 8
    assert span.payload.shape == (8, 6)


def test_camera_span_payload_matches_camera_plucker_directly():
    cam = _cam(size=(32, 16))
    span = camera_span(cam, patch=8)
    expected = cam.plucker(8).reshape(-1, 6).to(torch.float32).numpy()
    assert np.allclose(span.payload, expected)


def test_camera_span_is_unsupervised_and_observed_by_default():
    span = camera_span(_cam(), patch=8)
    assert span.supervised is False
    assert span.observed is True
    assert span.meta["kind"] == "camera_plucker"


def test_camera_span_modality_is_overridable_for_future_dedicated_modality():
    span = camera_span(_cam(), patch=8, modality="field")
    assert span.modality == "field"


# -- anchor_coordinates ---------------------------------------------------------


def test_anchor_coordinates_round_trips_through_project():
    cam = _cam(size=(32, 16))
    depth = torch.full((cam.height, cam.width), 4.0, dtype=torch.float64)
    coords = anchor_coordinates(cam, depth, patch=8)
    assert coords.shape == (8, 3)

    # Reprojecting must land back on the same patch centres camera.rays uses.
    uv, z = cam.project(coords)
    assert torch.allclose(z[:, 0], torch.full((8,), 4.0, dtype=torch.float64), atol=1e-9)
    h, w = cam.height // 8, cam.width // 8
    vv, uu = torch.meshgrid((torch.arange(h, dtype=torch.float64) + 0.5) * 8,
                            (torch.arange(w, dtype=torch.float64) + 0.5) * 8, indexing="ij")
    expected_uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)
    assert torch.allclose(uv, expected_uv, atol=1e-9)


def test_anchor_coordinates_rejects_mismatched_depth_shape():
    cam = _cam(size=(32, 16))
    with pytest.raises(ValueError):
        anchor_coordinates(cam, torch.zeros(4, 4), patch=8)


# -- scene_span -----------------------------------------------------------------


@dataclass
class FakeScene:
    means: torch.Tensor
    log_scales: torch.Tensor
    quats: torch.Tensor
    opacity_logits: torch.Tensor
    sh0: torch.Tensor


def _random_scene(n=40, seed=0) -> FakeScene:
    g = torch.Generator().manual_seed(seed)
    return FakeScene(
        means=torch.randn(n, 3, generator=g),
        log_scales=torch.randn(n, 3, generator=g),
        quats=torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        opacity_logits=torch.randn(n, generator=g),
        sh0=torch.rand(n, 3, generator=g),
    )


def test_scene_span_width_is_14_floats_per_splat():
    span = scene_span(_random_scene(10))
    assert span.payload.shape == (10, 14)
    assert span.modality == "geometry"


def test_scene_span_is_invariant_to_input_permutation():
    scene = _random_scene(30)
    perm = torch.randperm(30, generator=torch.Generator().manual_seed(1))
    permuted = FakeScene(scene.means[perm], scene.log_scales[perm], scene.quats[perm],
                         scene.opacity_logits[perm], scene.sh0[perm])

    a = scene_span(scene)
    b = scene_span(permuted)
    assert np.allclose(a.payload, b.payload)


def test_scene_span_truncation_keeps_highest_opacity_and_stays_permutation_invariant():
    scene = _random_scene(50)
    perm = torch.randperm(50, generator=torch.Generator().manual_seed(2))
    permuted = FakeScene(scene.means[perm], scene.log_scales[perm], scene.quats[perm],
                         scene.opacity_logits[perm], scene.sh0[perm])

    a = scene_span(scene, max_splats=10)
    b = scene_span(permuted, max_splats=10)
    assert a.payload.shape == (10, 14)
    assert np.allclose(a.payload, b.payload)
    assert a.meta["truncated"] is True
    assert a.meta["n_total"] == 50 and a.meta["n_kept"] == 10

    # the 10 kept splats really are the 10 highest-opacity ones
    kept_means = set(map(tuple, np.round(a.payload[:, :3], 4)))
    top_idx = torch.argsort(scene.opacity_logits, descending=True)[:10]
    expected_means = set(map(tuple, np.round(scene.means[top_idx].numpy(), 4)))
    assert kept_means == expected_means


def test_scene_span_no_truncation_when_under_budget():
    span = scene_span(_random_scene(5), max_splats=100)
    assert span.meta["truncated"] is False
    assert span.payload.shape == (5, 14)
