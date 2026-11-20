"""Typed frames. Gate: §18.3 "Frames and units".

Covers the correction that replaces ORRERY's single universal coordinate frame:
a pixel identifies a ray, not a unique 3-D cell.
"""

import numpy as np
import pytest

from iridium.contracts.frames import (
    Camera,
    Frame,
    FrameError,
    FrameGraph,
    Transform,
    normalize_coordinates,
    screen_motion,
)


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@pytest.fixture
def graph() -> FrameGraph:
    g = FrameGraph()
    g.add_frame(Frame("world", "world", "m", "right"))
    g.add_frame(Frame("rig", "object", "m", "right"))
    g.add_frame(Frame("cam", "camera", "m", "right"))
    g.add_transform(Transform("world", "rig", rot_z(0.3), np.array([1.0, -2.0, 0.5])))
    g.add_transform(Transform("rig", "cam", rot_z(-0.7), np.array([0.0, 0.0, 3.0])))
    return g


def test_transform_round_trip_is_exact(graph):
    t = graph.lookup("world", "cam")
    points = np.random.default_rng(0).normal(size=(32, 3))
    there = t.apply(points)
    back = t.inverse().apply(there)
    assert np.allclose(back, points, atol=1e-12)


def test_composition_matches_sequential_application(graph):
    a = graph.lookup("world", "rig")
    b = graph.lookup("rig", "cam")
    direct = graph.lookup("world", "cam")
    points = np.random.default_rng(1).normal(size=(16, 3))
    assert np.allclose(direct.apply(points), b.apply(a.apply(points)), atol=1e-12)


def test_identity_for_same_frame(graph):
    t = graph.lookup("cam", "cam")
    points = np.random.default_rng(2).normal(size=(4, 3))
    assert np.allclose(t.apply(points), points, atol=1e-15)


def test_missing_path_raises_rather_than_guessing():
    g = FrameGraph()
    g.add_frame(Frame("world", "world"))
    g.add_frame(Frame("screen", "image"))
    with pytest.raises(FrameError, match="no transform path"):
        g.lookup("world", "screen")


def test_frame_kinds_are_restricted():
    with pytest.raises(FrameError):
        Frame("weird", "hyperspace")


def test_rotation_must_be_orthonormal_and_proper():
    with pytest.raises(FrameError, match="orthonormal"):
        Transform("a", "b", np.full((3, 3), 0.5), np.zeros(3))
    reflection = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(FrameError, match="determinant"):
        Transform("a", "b", reflection, np.zeros(3))


def make_camera(name="cam0", tz=5.0) -> Camera:
    w2c = Transform("world", "cam", np.eye(3), np.array([0.0, 0.0, tz]))
    return Camera(name, fx=500.0, fy=500.0, cx=320.0, cy=240.0,
                  width=640, height=480, world_to_camera=w2c)


def test_pixel_maps_to_a_ray_not_a_point():
    """The central correction to the archived single-frame design."""
    cam = make_camera()
    ray = cam.pixel_to_ray(400.0, 300.0)

    # Many distinct world points project to the same pixel.
    p_near = ray.point_at(2.0)
    p_far = ray.point_at(9.0)
    assert not np.allclose(p_near, p_far)
    assert np.allclose(cam.project(p_near)[0], cam.project(p_far)[0], atol=1e-8)

    # Depth is an input; the ray alone does not determine a position.
    assert np.linalg.norm(ray.direction) == pytest.approx(1.0)


def test_projection_round_trip_through_known_depth():
    cam = make_camera()
    world = np.array([[0.7, -0.4, -1.5]])
    pixel = cam.project(world)[0]
    ray = cam.pixel_to_ray(*pixel)
    depth = float(cam.world_to_camera.apply(world)[0, 2])
    recovered = ray.point_at(depth / ray.direction[2] * 1.0)
    # Recovering the point needs the depth; with it, the round trip closes.
    assert np.allclose(cam.project(recovered)[0], pixel, atol=1e-6)


def test_points_behind_camera_are_not_projected():
    cam = make_camera()
    behind = np.array([[0.0, 0.0, -10.0]])  # z_cam = -5 < 0
    assert np.all(np.isnan(cam.project(behind)))


def test_screen_motion_includes_camera_motion():
    """§7.5: screen motion is not fluid velocity."""
    cam0 = make_camera(tz=5.0)
    cam1 = make_camera(tz=5.0)
    stationary = np.array([0.2, 0.1, 0.0])

    # A stationary point with a stationary camera has zero screen motion.
    assert np.allclose(screen_motion(cam0, cam1, stationary, stationary), 0.0)

    # The same stationary point moves on screen when the camera moves.
    cam_moved = make_camera(tz=6.0)
    motion = screen_motion(cam0, cam_moved, stationary, stationary)
    assert np.linalg.norm(motion) > 1.0

    # So a nonzero screen motion does not imply the material moved.


def test_world_velocity_and_screen_motion_have_different_units():
    """A material point moving 1 m/s does not move a fixed number of pixels."""
    cam = make_camera()
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 0.0, 0.0])          # 1 m displacement
    near = screen_motion(cam, cam, p0, p1)

    far_cam = make_camera(tz=50.0)
    far = screen_motion(far_cam, far_cam, p0, p1)

    # Identical world displacement, very different pixel displacement.
    assert np.linalg.norm(near) > 5 * np.linalg.norm(far)


def test_normalization_by_characteristic_length():
    coords = np.array([[2.0, 4.0, 6.0]])
    assert np.allclose(normalize_coordinates(coords, 2.0), [[1.0, 2.0, 3.0]])
    with pytest.raises(FrameError):
        normalize_coordinates(coords, 0.0)


def test_normalization_does_not_imply_similarity():
    """Two scenes can normalize identically and still be different problems.

    §5.3 and the §3 correction: matching a few normalized numbers is not
    dynamic similarity. This test documents the semantics rather than asserting
    a physical claim -- normalization is a conditioning aid only.
    """
    a = normalize_coordinates(np.array([[1.0, 0.0, 0.0]]), 1.0)
    b = normalize_coordinates(np.array([[1000.0, 0.0, 0.0]]), 1000.0)
    assert np.allclose(a, b)
    # Identical normalized coordinates; the scenarios still differ in their
    # dimensional scales, which are retained as separate inputs.
    assert 1.0 != 1000.0
