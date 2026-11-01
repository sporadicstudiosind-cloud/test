"""ActionMap deltas, the render/retrieve/generate/fuse loop, and revisit consistency.

Uses a fake renderer + fake splat scene throughout (never imports
``iridium.world.splats``), and ``RenderOnlyFrameModel`` as the frame model
under test, per the ground rules against depending on the concurrent
``splats.py`` module.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from iridium.world.camera import Camera, look_at, orbit
from iridium.world.rollout import (ActionMap, RenderOnlyFrameModel, WorldRollout,
                                   revisit_consistency)
from iridium.world.world_state import Keyframe, WorldState


def _cam(eye=(0.0, 0.0, -5.0), target=(0.0, 0.0, 0.0), size=(16, 12)):
    w, h = size
    return Camera.from_fov(70.0, w, h, look_at(eye, target, dtype=torch.float32))


# -- ActionMap ----------------------------------------------------------------


def test_action_map_wasd_keys_produce_expected_deltas():
    m = ActionMap(move_speed=0.5, turn_speed_degrees=15.0)
    assert m.resolve("w") == pytest.approx(
        dict(forward=0.5, right=0.0, down=0.0, yaw_degrees=0.0, pitch_degrees=0.0))
    assert m.resolve("a") == pytest.approx(
        dict(forward=0.0, right=-0.5, down=0.0, yaw_degrees=0.0, pitch_degrees=0.0))
    assert m.resolve("yaw_right") == pytest.approx(
        dict(forward=0.0, right=0.0, down=0.0, yaw_degrees=15.0, pitch_degrees=0.0))


def test_action_map_string_with_magnitude_scales_linearly():
    m = ActionMap(move_speed=1.0)
    out = m.resolve("forward 0.5")
    assert out["forward"] == pytest.approx(0.5)
    out2 = m.resolve("forward 2")
    assert out2["forward"] == pytest.approx(2.0)


def test_action_map_dict_is_literal_move_kwargs():
    m = ActionMap(move_speed=100.0)  # must NOT scale a dict action
    out = m.resolve({"forward": 0.3, "yaw_degrees": 5.0})
    assert out == {"forward": 0.3, "yaw_degrees": 5.0}


def test_action_map_dict_rejects_unknown_keys():
    with pytest.raises(ValueError):
        ActionMap().resolve({"sideways": 1.0})


def test_action_map_continuous_vector_scales_translation_and_rotation_separately():
    m = ActionMap(move_speed=2.0, turn_speed_degrees=10.0)
    out = m.resolve([1.0, 0.0, 0.0, 1.0, 0.0])
    assert out["forward"] == pytest.approx(2.0)
    assert out["yaw_degrees"] == pytest.approx(10.0)


def test_action_map_continuous_vector_drops_roll():
    m = ActionMap()
    out5 = m.resolve([0.1, 0.0, 0.0, 0.0, 0.0])
    out6 = m.resolve([0.1, 0.0, 0.0, 0.0, 0.0, 999.0])  # roll must be ignored
    assert out5 == out6


def test_action_map_unknown_key_raises():
    with pytest.raises(KeyError):
        ActionMap().resolve("teleport")


# -- fake renderer + scene for rollout tests ---------------------------------


@dataclass
class FakeScene:
    means: torch.Tensor
    log_scales: torch.Tensor
    quats: torch.Tensor
    opacity_logits: torch.Tensor
    sh0: torch.Tensor

    def __len__(self):
        return int(self.means.shape[0])

    def concat(self, other):
        return FakeScene(*(torch.cat([a, b]) for a, b in zip(
            (self.means, self.log_scales, self.quats, self.opacity_logits, self.sh0),
            (other.means, other.log_scales, other.quats, other.opacity_logits, other.sh0))))

    @staticmethod
    def from_points(points, rgb, scale):
        n = points.shape[0]
        return FakeScene(points.to(torch.float32), torch.zeros(n, 3),
                         torch.tensor([[1.0, 0, 0, 0]] * n), torch.zeros(n), rgb.to(torch.float32))


@dataclass
class FakeRenderOutput:
    rgb: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    visible: torch.Tensor


def _static_scene_renderer(fixed_depth: float = 5.0):
    """A deterministic function of camera pose: same pose -> bit-identical rgb, always,
    different pose -> (almost certainly) different rgb.

    Stands in for a real splat renderer for the purpose of these tests: what
    is being tested is the rollout/world-state/revisit plumbing, not
    rasterization, but the output has to depend on pose or a revisit check
    would pass trivially for any pair of frames rather than specifically for
    the ones whose cameras coincide.
    """
    def render(scene, camera, background=(0.0, 0.0, 0.0)):
        h, w = camera.height, camera.width
        if len(scene) == 0:
            return FakeRenderOutput(torch.zeros(h, w, 3), torch.zeros(h, w),
                                    torch.zeros(h, w), torch.zeros(0, dtype=torch.bool))
        pos, fwd = camera.position, camera.forward
        value = 0.5 + 0.1 * torch.sin(pos.sum() * 11.0) + 0.05 * torch.cos(fwd.sum() * 7.0)
        return FakeRenderOutput(
            rgb=torch.full((h, w, 3), float(value)),
            alpha=torch.ones(h, w),
            depth=torch.full((h, w), fixed_depth),
            visible=torch.ones(len(scene), dtype=torch.bool),
        )
    return render


def _seed_state_with_a_wall() -> WorldState:
    state = WorldState(max_keyframes=16)
    cam = _cam()
    state.fuse(Keyframe(cam, torch.full((cam.height, cam.width, 3), 0.6),
                        torch.full((cam.height, cam.width), 5.0), step=-1),
              from_points_fn=FakeScene.from_points)
    return state


# -- rollout loop ---------------------------------------------------------------


def test_ten_step_wasd_loop_runs_and_advances_camera():
    state = _seed_state_with_a_wall()
    start = _cam()
    rollout = WorldRollout(
        world_state=state, frame_model=RenderOnlyFrameModel(), camera=start,
        renderer=_static_scene_renderer(), action_map=ActionMap(move_speed=0.2),
    )
    actions = ["w", "w", "a", "d", "s", "yaw_left", "yaw_right", "w", "pitch_up", "pitch_down"]
    frames = rollout.play(actions)
    assert len(frames) == 10
    assert rollout.step_count == 10
    for frame in frames:
        assert frame.rgb.shape == (start.height, start.width, 3)
    # camera actually moved from its start pose
    assert not torch.allclose(frames[-1].camera.world_to_camera, start.world_to_camera)


def test_render_only_frame_model_falls_back_to_grey_with_no_scene():
    state = WorldState()
    rollout = WorldRollout(world_state=state, frame_model=RenderOnlyFrameModel(),
                           camera=_cam(), renderer=_static_scene_renderer())
    frame = rollout.step("w")
    assert torch.allclose(frame.rgb, torch.full_like(frame.rgb, 0.5))


def test_fly_visits_explicit_camera_path():
    state = _seed_state_with_a_wall()
    rollout = WorldRollout(world_state=state, frame_model=RenderOnlyFrameModel(),
                           camera=_cam(), renderer=_static_scene_renderer())
    path = orbit(_cam(), target=(0.0, 0.0, 0.0), radius=5.0, n=6)
    frames = rollout.fly(path)
    assert len(frames) == 6
    assert frames[-1].camera is path[-1]


# -- revisit consistency ---------------------------------------------------------


def test_revisit_consistency_is_near_zero_for_render_only_model_on_static_scene():
    state = _seed_state_with_a_wall()
    rollout = WorldRollout(world_state=state, frame_model=RenderOnlyFrameModel(),
                           camera=_cam(), renderer=_static_scene_renderer(),
                           action_map=ActionMap(move_speed=0.3, turn_speed_degrees=30.0))
    # One forward step (heading unchanged), then a full 360-degree turn in
    # twelve 30-degree steps: no further translation happens during the
    # turn, so the last of those frames sits at the same position and
    # heading as the first -- an exact revisit.
    actions = ["w"] + ["yaw_right"] * 12
    frames = rollout.play(actions)
    pairs = revisit_consistency(frames, position_threshold=0.5, direction_threshold=0.99)
    assert len(pairs) > 0, "test setup should produce at least one near-revisit"
    assert max(p.error for p in pairs) < 1e-6


def test_revisit_consistency_is_positive_when_frame_model_injects_noise():
    state = _seed_state_with_a_wall()

    class NoisyFrameModel:
        def __call__(self, context):
            base = context.anchor_rgb if context.anchor_rgb is not None else torch.zeros(
                context.camera.height, context.camera.width, 3)
            noise = torch.rand_like(base) * 0.5
            return base + noise, context.anchor_depth

    rollout = WorldRollout(world_state=state, frame_model=NoisyFrameModel(),
                           camera=_cam(), renderer=_static_scene_renderer(),
                           action_map=ActionMap(move_speed=0.3, turn_speed_degrees=30.0))
    actions = ["w"] + ["yaw_right"] * 12
    frames = rollout.play(actions)
    pairs = revisit_consistency(frames, position_threshold=0.5, direction_threshold=0.99)
    assert len(pairs) > 0
    assert max(p.error for p in pairs) > 0.0
