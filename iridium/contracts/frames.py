"""Typed coordinate frames and the frame graph.

Architecture §5.3 replaces ORRERY's single universal coordinate frame. Screen
coordinates, camera rays, world coordinates, object coordinates, mesh indices,
frequency axes and text positions are *different kinds of coordinate* and are
not interconvertible by assumption.

The central correction: a pixel identifies a **ray**, not a unique 3-D point.
``Camera.pixel_to_ray`` returns a ray; recovering a point requires depth, which
must come from geometry or a depth observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


class FrameError(ValueError):
    """Raised for an undefined or inconsistent frame operation."""


# Kinds of coordinate that MUST NOT be silently mixed.
FRAME_KINDS = (
    "world",       # metric 3-D scene coordinates
    "object",      # rigid-body local coordinates
    "camera",      # camera-centred metric coordinates
    "image",       # pixel coordinates on a sensor or screen
    "mesh_index",  # discrete topological indices
    "spectral",    # frequency / wavenumber / mode index
    "text",        # token sequence position
)


@dataclass(frozen=True)
class Frame:
    name: str
    kind: str
    units: str | None = None
    handedness: str | None = None  # "right" | "left" | None for non-metric

    def __post_init__(self) -> None:
        if self.kind not in FRAME_KINDS:
            raise FrameError(f"unknown frame kind: {self.kind}")


@dataclass(frozen=True)
class Transform:
    """A rigid transform mapping points in ``parent`` to points in ``child``.

    ``x_child = rotation @ x_parent + translation``
    """

    parent: str
    child: str
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        R = np.asarray(self.rotation, dtype=np.float64)
        if R.shape != (3, 3):
            raise FrameError("rotation must be 3x3")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-9):
            raise FrameError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-9):
            raise FrameError("rotation must have determinant +1 (no reflection)")
        if np.asarray(self.translation, dtype=np.float64).shape != (3,):
            raise FrameError("translation must have shape (3,)")

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return pts @ np.asarray(self.rotation).T + np.asarray(self.translation)

    def inverse(self) -> "Transform":
        R = np.asarray(self.rotation, dtype=np.float64)
        t = np.asarray(self.translation, dtype=np.float64)
        return Transform(self.child, self.parent, R.T, -R.T @ t)

    def compose(self, other: "Transform") -> "Transform":
        """``self ∘ other``: other maps A→B, self maps B→C, result maps A→C."""
        if other.child != self.parent:
            raise FrameError(
                f"cannot compose {other.parent}->{other.child} with "
                f"{self.parent}->{self.child}"
            )
        R1, t1 = np.asarray(other.rotation), np.asarray(other.translation)
        R2, t2 = np.asarray(self.rotation), np.asarray(self.translation)
        return Transform(other.parent, self.child, R2 @ R1, R2 @ t1 + t2)


class FrameGraph:
    """A graph of frames connected by rigid transforms.

    Lookups traverse the graph; there is no assumption that all frames share a
    common origin, and a path that does not exist raises rather than guessing.
    """

    def __init__(self) -> None:
        self._frames: dict[str, Frame] = {}
        self._edges: dict[tuple[str, str], Transform] = {}

    def add_frame(self, frame: Frame) -> None:
        if frame.name in self._frames:
            raise FrameError(f"frame already defined: {frame.name}")
        self._frames[frame.name] = frame

    def frame(self, name: str) -> Frame:
        if name not in self._frames:
            raise FrameError(f"unknown frame: {name}")
        return self._frames[name]

    def add_transform(self, transform: Transform) -> None:
        for n in (transform.parent, transform.child):
            if n not in self._frames:
                raise FrameError(f"unknown frame: {n}")
        self._edges[(transform.parent, transform.child)] = transform
        inv = transform.inverse()
        self._edges[(inv.parent, inv.child)] = inv

    def lookup(self, source: str, target: str) -> Transform:
        """Breadth-first search for a composed transform ``source -> target``."""
        if source == target:
            return Transform(source, target, np.eye(3), np.zeros(3))
        seen = {source}
        queue: list[tuple[str, Transform | None]] = [(source, None)]
        while queue:
            node, acc = queue.pop(0)
            for (a, b), edge in self._edges.items():
                if a != node or b in seen:
                    continue
                composed = edge if acc is None else edge.compose(acc)
                if b == target:
                    return composed
                seen.add(b)
                queue.append((b, composed))
        raise FrameError(f"no transform path from {source} to {target}")


@dataclass(frozen=True)
class Ray:
    """A ray in a named metric frame. Origin in metres, unit direction."""

    frame: str
    origin: np.ndarray
    direction: np.ndarray

    def point_at(self, depth: float) -> np.ndarray:
        """The point at a given depth. Depth is an *input*, never inferred here."""
        return np.asarray(self.origin) + depth * np.asarray(self.direction)


@dataclass(frozen=True)
class Camera:
    """A pinhole camera with an explicit pose and an explicit image frame."""

    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    world_to_camera: Transform

    def pixel_to_ray(self, u: float, v: float) -> Ray:
        """Map a pixel to a ray in world coordinates.

        This is the operation ORRERY's single-frame design elided. A pixel does
        not identify a 3-D cell; it identifies the set of points along this ray.
        """
        d_cam = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        d_cam /= np.linalg.norm(d_cam)
        cam_to_world = self.world_to_camera.inverse()
        R = np.asarray(cam_to_world.rotation)
        origin = np.asarray(cam_to_world.translation)
        return Ray(self.world_to_camera.parent, origin, R @ d_cam)

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """Project world points to pixels. Points behind the camera give NaN."""
        pts = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
        cam = self.world_to_camera.apply(pts)
        z = cam[:, 2]
        out = np.full((cam.shape[0], 2), np.nan)
        valid = z > 0
        out[valid, 0] = self.fx * cam[valid, 0] / z[valid] + self.cx
        out[valid, 1] = self.fy * cam[valid, 1] / z[valid] + self.cy
        return out


def screen_motion(
    camera_t0: Camera,
    camera_t1: Camera,
    point_t0: np.ndarray,
    point_t1: np.ndarray,
) -> np.ndarray:
    """Screen motion in pixels for a tracked material point.

    Architecture §7.5. This is *not* the fluid velocity: it includes camera
    motion and the perspective divide, and it is undefined where the point is
    not visible. Returning it in the same array as world velocity, or labelling
    it "motion vectors" without a frame, is the error being corrected here.
    """
    p0 = camera_t0.project(np.atleast_2d(point_t0))
    p1 = camera_t1.project(np.atleast_2d(point_t1))
    return p1 - p0


def normalize_coordinates(
    coords: np.ndarray, characteristic_length: float
) -> np.ndarray:
    """Divide spatial coordinates by a declared characteristic length.

    Architecture §5.3 keeps this as a *conditioning* aid. It improves numerical
    conditioning and makes unit handling consistent. It does not by itself
    establish dynamic similarity between two scenarios; that requires matching
    equations, geometry, boundaries, initial conditions and material laws.
    """
    if characteristic_length <= 0:
        raise FrameError("characteristic length must be positive")
    return np.asarray(coords, dtype=np.float64) / characteristic_length
