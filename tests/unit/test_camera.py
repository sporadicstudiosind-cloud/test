"""Camera geometry: every test here pins down a convention.

Camera bugs do not crash. A flipped axis renders a mirrored scene, a
half-pixel offset shifts every ray, a non-orthonormal rotation shears the
world -- and a model trained on the result learns the bug. So each property
below is one that a plausible wrong implementation fails.
"""
from __future__ import annotations

import math

import pytest
import torch

from iridium.world.camera import (Camera, interpolate, look_at, move, orbit,
                                  rigid_inverse, trajectory)


def _cam(eye=(0.0, 0.0, -5.0), target=(0.0, 0.0, 0.0), size=(64, 48)):
    w, h = size
    return Camera.from_fov(60.0, w, h, look_at(eye, target, dtype=torch.float64))


def test_look_at_puts_the_target_on_the_principal_point():
    cam = _cam(eye=(1.0, -2.0, -4.0), target=(0.3, 0.1, 0.2))
    uv, depth = cam.project(torch.tensor([[0.3, 0.1, 0.2]], dtype=torch.float64))
    assert torch.allclose(uv[0], torch.tensor([cam.cx, cam.cy], dtype=torch.float64), atol=1e-9)
    assert depth[0, 0] > 0


def test_opencv_convention_y_down_image_rows_grow_downward():
    """A point above the target (world -y, since up=-y) must land in a smaller row."""
    cam = _cam()
    above = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64)
    uv, _ = cam.project(above)
    assert uv[0, 1] < cam.cy
    right = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    uv_r, _ = cam.project(right)
    assert uv_r[0, 0] > cam.cx


def test_rotation_is_orthonormal_and_inverse_is_rigid():
    cam = _cam(eye=(2.0, -1.0, -3.0), target=(0.5, 0.5, 0.5))
    r = cam.world_to_camera[:3, :3]
    assert torch.allclose(r @ r.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
    assert torch.isclose(torch.det(r), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(rigid_inverse(cam.world_to_camera) @ cam.world_to_camera,
                          torch.eye(4, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(cam.position, torch.tensor([2.0, -1.0, -3.0], dtype=torch.float64))


def test_project_unproject_round_trip():
    cam = _cam(eye=(1.0, -0.5, -6.0))
    g = torch.Generator().manual_seed(0)
    pts = torch.randn(200, 3, generator=g, dtype=torch.float64)
    uv, depth = cam.project(pts)
    back = cam.unproject(uv, depth)
    assert torch.allclose(back, pts, atol=1e-9)


def test_rays_pass_through_pixel_centres_not_corners():
    """Project a point along the ray of pixel (u, v): it must land at (u+0.5, v+0.5)."""
    cam = _cam()
    origins, dirs = cam.rays()
    for v, u in [(0, 0), (10, 20), (cam.height - 1, cam.width - 1)]:
        point = origins[v, u] + 3.0 * dirs[v, u]
        uv, _ = cam.project(point[None])
        assert torch.allclose(uv[0], torch.tensor([u + 0.5, v + 0.5], dtype=torch.float64), atol=1e-9)


def test_patch_rays_pass_through_patch_centres():
    cam = _cam(size=(64, 48))
    origins, dirs = cam.rays(stride=8)
    assert dirs.shape == (6, 8, 3)
    uv, _ = cam.project((origins[2, 3] + dirs[2, 3])[None])
    assert torch.allclose(uv[0], torch.tensor([3 * 8 + 4.0, 2 * 8 + 4.0], dtype=torch.float64), atol=1e-9)


def test_plucker_is_invariant_to_origin_along_the_ray():
    cam = _cam(eye=(1.0, 2.0, -3.0))
    origins, dirs = cam.rays(stride=4)
    plucker = cam.plucker(stride=4)
    shifted = torch.linalg.cross(origins + 7.3 * dirs, dirs, dim=-1)
    assert torch.allclose(plucker[..., 3:], shifted, atol=1e-9)
    assert torch.allclose(plucker[..., :3].norm(dim=-1),
                          torch.ones(plucker.shape[:2], dtype=torch.float64), atol=1e-12)
    # Moment is orthogonal to direction by construction.
    assert torch.allclose((plucker[..., :3] * plucker[..., 3:]).sum(-1),
                          torch.zeros(plucker.shape[:2], dtype=torch.float64), atol=1e-9)


def test_resize_keeps_the_same_rays():
    cam = _cam(size=(64, 48))
    half = cam.resized(32, 24)
    _, d_full = cam.rays(stride=2)
    _, d_half = half.rays(stride=1)
    assert torch.allclose(d_full, d_half, atol=1e-9)


def test_sees_rejects_points_behind_the_camera():
    cam = _cam()
    front = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    behind = torch.tensor([[0.0, 0.0, -10.0]], dtype=torch.float64)
    assert cam.sees(front).all() and not cam.sees(behind).any()


def test_orbit_cameras_all_look_at_the_target_from_the_same_radius():
    cams = orbit(_cam(), target=(0.0, 0.0, 0.0), radius=4.0, n=8)
    for c in cams:
        assert math.isclose(float(c.position.norm()), 4.0, rel_tol=1e-9)
        uv, _ = c.project(torch.zeros(1, 3, dtype=torch.float64))
        assert torch.allclose(uv[0], torch.tensor([c.cx, c.cy], dtype=torch.float64), atol=1e-8)


def test_interpolate_hits_endpoints_and_stays_rigid():
    a = _cam(eye=(0.0, 0.0, -5.0))
    b = _cam(eye=(5.0, -1.0, 0.0))
    assert torch.allclose(interpolate(a, b, 0.0).world_to_camera, a.world_to_camera, atol=1e-9)
    assert torch.allclose(interpolate(a, b, 1.0).world_to_camera, b.world_to_camera, atol=1e-9)
    mid = interpolate(a, b, 0.5)
    r = mid.world_to_camera[:3, :3]
    assert torch.allclose(r @ r.T, torch.eye(3, dtype=torch.float64), atol=1e-9)
    # The centre follows a straight line between centres, not an arc.
    assert torch.allclose(mid.position, 0.5 * (a.position + b.position), atol=1e-9)


def test_move_forward_travels_along_the_view_direction():
    cam = _cam()
    moved = move(cam, forward=2.0)
    assert torch.allclose(moved.position - cam.position, 2.0 * cam.forward, atol=1e-9)
    turned = move(cam, yaw_degrees=90.0)
    assert torch.allclose(turned.position, cam.position, atol=1e-9)
    assert abs(float(turned.forward @ cam.forward)) < 1e-9


def test_trajectory_densifies_between_keyframes():
    a, b, c = _cam(eye=(0, 0, -5)), _cam(eye=(5, 0, 0)), _cam(eye=(0, 0, 5))
    path = trajectory([a, b, c], steps_between=4)
    assert len(path) == 9
    assert torch.allclose(path[4].world_to_camera, b.world_to_camera, atol=1e-9)


def test_invalid_cameras_are_refused():
    with pytest.raises(ValueError):
        Camera.from_fov(0.0, 10, 10)
    with pytest.raises(ValueError):
        look_at((0, 0, 0), (0, 0, 0))
    with pytest.raises(ValueError):
        look_at((0, 0, 0), (0, 1, 0), up=(0, 1, 0))
