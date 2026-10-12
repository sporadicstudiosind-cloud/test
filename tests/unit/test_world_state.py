"""WorldState: frustum-overlap retrieval, coverage-driven eviction, fusion, save/load.

Everything here uses a tiny fake splat scene/renderer rather than importing
``iridium.world.splats`` (owned by a concurrent agent and not guaranteed to
exist yet) -- ``world_state.py`` takes the scene as an opaque object and the
renderer/point-constructor as injected callables specifically so this file
never needs that import.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch

from iridium.world.camera import Camera, look_at
from iridium.world.world_state import Keyframe, WorldState


def _cam(eye=(0.0, 0.0, -5.0), target=(0.0, 0.0, 0.0), size=(16, 12)):
    w, h = size
    return Camera.from_fov(70.0, w, h, look_at(eye, target, dtype=torch.float32))


def _flat_depth(camera: Camera, value: float = 5.0) -> torch.Tensor:
    return torch.full((camera.height, camera.width), value, dtype=torch.float32)


def _flat_rgb(camera: Camera, value: float = 0.4) -> torch.Tensor:
    return torch.full((camera.height, camera.width, 3), value, dtype=torch.float32)


# -- a minimal duck-typed stand-in for splats.GaussianScene ------------------


@dataclass
class FakeScene:
    means: torch.Tensor
    log_scales: torch.Tensor
    quats: torch.Tensor
    opacity_logits: torch.Tensor
    sh0: torch.Tensor
    sh_rest: torch.Tensor | None = None

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def concat(self, other: "FakeScene") -> "FakeScene":
        return FakeScene(
            torch.cat([self.means, other.means]),
            torch.cat([self.log_scales, other.log_scales]),
            torch.cat([self.quats, other.quats]),
            torch.cat([self.opacity_logits, other.opacity_logits]),
            torch.cat([self.sh0, other.sh0]),
        )

    def select(self, mask: torch.Tensor) -> "FakeScene":
        return FakeScene(self.means[mask], self.log_scales[mask], self.quats[mask],
                         self.opacity_logits[mask], self.sh0[mask])

    @staticmethod
    def from_points(points: torch.Tensor, rgb: torch.Tensor, scale: float) -> "FakeScene":
        n = points.shape[0]
        return FakeScene(
            means=points.to(torch.float32),
            log_scales=torch.full((n, 3), float(np.log(scale)), dtype=torch.float32),
            quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n, dtype=torch.float32),
            opacity_logits=torch.zeros(n, dtype=torch.float32),
            sh0=rgb.to(torch.float32),
        )


@dataclass
class FakeRenderOutput:
    rgb: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    visible: torch.Tensor


def _fake_renderer_from_coverage(covered: bool):
    """A renderer whose alpha is uniformly ``1.0`` if ``covered`` else ``0.0``."""
    def render(scene, camera, background=(0.0, 0.0, 0.0)):
        h, w = camera.height, camera.width
        value = 1.0 if covered and len(scene) > 0 else 0.0
        return FakeRenderOutput(
            rgb=torch.zeros(h, w, 3), alpha=torch.full((h, w), value),
            depth=torch.zeros(h, w), visible=torch.ones(len(scene), dtype=torch.bool),
        )
    return render


# -- retrieval: overlap, not centre distance ---------------------------------


def test_retrieve_prefers_co_oriented_camera_over_co_located_opposite_facing():
    query = _cam(eye=(0.0, 0.0, -5.0), target=(0.0, 0.0, 0.0))
    co_oriented = _cam(eye=(0.2, 0.0, -6.0), target=(0.0, 0.0, 0.0))
    co_located_opposite = replace(
        query, world_to_camera=look_at((0.0, 0.0, -5.0), (0.0, 0.0, -10.0), dtype=torch.float32))

    state = WorldState(max_keyframes=8)
    state.add_keyframe(Keyframe(co_located_opposite, _flat_rgb(co_located_opposite),
                                _flat_depth(co_located_opposite), step=0))
    state.add_keyframe(Keyframe(co_oriented, _flat_rgb(co_oriented),
                                _flat_depth(co_oriented), step=1))

    [best] = state.retrieve(query, k=1)
    assert torch.allclose(best.camera.world_to_camera, co_oriented.world_to_camera)


def test_retrieve_returns_fewer_than_k_when_bank_is_smaller():
    query = _cam()
    state = WorldState()
    state.add_keyframe(Keyframe(query, _flat_rgb(query), _flat_depth(query), step=0))
    assert len(state.retrieve(query, k=5)) == 1


def test_retrieve_empty_bank_returns_empty_list():
    assert WorldState().retrieve(_cam(), k=4) == []


# -- coverage -----------------------------------------------------------------


def test_coverage_zero_with_no_scene_or_renderer():
    state = WorldState()
    assert state.coverage(_cam()) == 0.0
    state.scene = FakeScene.from_points(torch.zeros(1, 3), torch.zeros(1, 3), 0.1)
    assert state.coverage(_cam()) == 0.0  # no renderer injected


def test_coverage_reads_renderer_alpha():
    state = WorldState(scene=FakeScene.from_points(torch.zeros(3, 3), torch.zeros(3, 3), 0.1))
    assert state.coverage(_cam(), renderer=_fake_renderer_from_coverage(True)) == pytest.approx(1.0)
    assert state.coverage(_cam(), renderer=_fake_renderer_from_coverage(False)) == pytest.approx(0.0)


# -- eviction ------------------------------------------------------------------


def test_eviction_keeps_the_unique_new_view_over_redundant_duplicates():
    """Many near-duplicate keyframes of one view plus one keyframe of a distinct
    view: FIFO would age out the distinct one first; coverage-based eviction
    must not."""
    state = WorldState(max_keyframes=3)
    duplicate_cam = _cam(eye=(0.0, 0.0, -5.0), target=(0.0, 0.0, 0.0))
    distinct_cam = _cam(eye=(0.0, 0.0, -5.0), target=(0.0, 0.0, -10.0))  # facing away

    # Oldest: the one unique view.
    state.add_keyframe(Keyframe(distinct_cam, _flat_rgb(distinct_cam),
                                _flat_depth(distinct_cam), step=0))
    # Then several near-duplicates of the common view, overflowing capacity.
    for i in range(1, 5):
        dup = _cam(eye=(0.0 + 0.01 * i, 0.0, -5.0), target=(0.0, 0.0, 0.0))
        state.add_keyframe(Keyframe(dup, _flat_rgb(dup), _flat_depth(dup), step=i))

    assert len(state.keyframes) == 3
    kept_steps = {kf.step for kf in state.keyframes}
    assert 0 in kept_steps, "the only keyframe of the distinct view must survive eviction"


def test_eviction_never_leaves_the_bank_over_capacity():
    state = WorldState(max_keyframes=2)
    for i in range(6):
        cam = _cam(eye=(0.1 * i, 0.0, -5.0), target=(0.0, 0.0, 0.0))
        state.add_keyframe(Keyframe(cam, _flat_rgb(cam), _flat_depth(cam), step=i))
    assert len(state.keyframes) == 2


# -- fusion ---------------------------------------------------------------------


def test_fuse_creates_scene_from_depth_when_none_exists():
    state = WorldState()
    cam = _cam()
    kf = Keyframe(cam, _flat_rgb(cam), _flat_depth(cam), step=0)
    state.fuse(kf, from_points_fn=FakeScene.from_points)
    assert state.scene is not None and len(state.scene) > 0
    assert len(state.keyframes) == 1


def test_fuse_skips_geometry_without_depth_but_still_remembers_keyframe():
    state = WorldState()
    cam = _cam()
    kf = Keyframe(cam, _flat_rgb(cam), depth=None, step=0)
    state.fuse(kf, from_points_fn=FakeScene.from_points)
    assert state.scene is None
    assert len(state.keyframes) == 1


def test_fuse_does_not_add_points_where_coverage_is_already_high():
    cam = _cam()
    existing = FakeScene.from_points(torch.zeros(5, 3), torch.zeros(5, 3), 0.1)
    state = WorldState(scene=existing)
    kf = Keyframe(cam, _flat_rgb(cam), _flat_depth(cam), step=0)
    state.fuse(kf, renderer=_fake_renderer_from_coverage(True),
              from_points_fn=FakeScene.from_points)
    assert len(state.scene) == 5  # nothing new added, everything already covered


# -- save / load -----------------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    cam = _cam()
    state = WorldState(max_keyframes=8)
    state.fuse(Keyframe(cam, _flat_rgb(cam, 0.3), _flat_depth(cam, 4.0), step=0, source="observed"),
              from_points_fn=FakeScene.from_points)
    state.add_keyframe(Keyframe(_cam(eye=(1.0, 0.0, -5.0)), _flat_rgb(cam, 0.7), None, step=1,
                                source="generated"))

    state.save(tmp_path)

    # load() needs to construct a real GaussianScene; monkeypatch the lazy
    # import target so this test does not depend on splats.py existing yet.
    import iridium.world.world_state as world_state_module
    import sys
    import types
    fake_module = types.ModuleType("iridium.world.splats")
    fake_module.GaussianScene = FakeScene
    sys.modules["iridium.world.splats"] = fake_module
    try:
        loaded = WorldState.load(tmp_path)
    finally:
        del sys.modules["iridium.world.splats"]

    assert len(loaded.keyframes) == 2
    steps = sorted(kf.step for kf in loaded.keyframes)
    assert steps == [0, 1]
    depths = {kf.step: kf.depth for kf in loaded.keyframes}
    assert depths[0] is not None
    assert depths[1] is None
    assert loaded.scene is not None
    assert torch.allclose(loaded.scene.means, state.scene.means)
