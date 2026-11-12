"""Pinhole cameras, the rays they cast, and how they move.

Convention, stated once because every bug in this area is a convention bug:

* **OpenCV / COLMAP camera frame.** +x right, +y *down*, +z *forward* (into the
  scene). This is what 3D Gaussian Splatting, COLMAP, and the standard 3DGS
  ``.ply`` format assume, and matching it is what lets a scene produced here
  open in the tools Marble-style splat worlds are exported to. OpenGL's
  convention (+y up, -z forward) is the obvious alternative and the one most
  graphics tutorials use; mixing the two produces scenes that are mirrored or
  upside down, which a loss curve never reveals.
* **``world_to_camera``** is the stored extrinsic, a rigid 4x4 ``[R | t]``: a
  world point ``p`` lands at ``R p + t`` in camera coordinates. The inverse is
  computed rigidly (``R^T``, ``-R^T t``), never with a general matrix inverse,
  which would let numerical drift make ``R`` non-orthonormal.
* **Pixel centres are at half-integers.** Pixel ``(u, v)`` covers
  ``[u, u+1) x [v, v+1)`` and its ray passes through ``(u + 0.5, v + 0.5)``.
  Casting through the integer corner shifts every ray by half a pixel, which is
  invisible at 512 px and a quarter-patch error at the 8 px patch size the codecs
  use.

Why the model sees **Plücker rays** rather than the camera matrix itself: a
camera is 12 numbers that mean nothing locally, and a network asked to condition
a patch on them has to learn projective geometry from scratch. The Plücker
embedding ``(d, o x d)`` of the ray through each patch hands every patch its own
line in space directly -- direction plus moment, six numbers, invariant to where
along the line the origin is taken. It is the conditioning CameraCtrl, CAT3D and
most camera-controlled video models converged on, and it is exactly the
"camera geometry as a native input type" World Labs describes for Atlas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence

import torch

__all__ = [
    "Camera",
    "look_at",
    "rigid_inverse",
    "orbit",
    "interpolate",
    "move",
    "trajectory",
]


def rigid_inverse(matrix: torch.Tensor) -> torch.Tensor:
    """Invert a rigid ``[R | t]`` 4x4 exactly, without a general inverse."""
    if matrix.shape[-2:] != (4, 4):
        raise ValueError("expected a 4x4 rigid transform")
    rotation = matrix[..., :3, :3]
    translation = matrix[..., :3, 3:]
    out = torch.zeros_like(matrix)
    out[..., :3, :3] = rotation.transpose(-1, -2)
    out[..., :3, 3:] = -rotation.transpose(-1, -2) @ translation
    out[..., 3, 3] = 1.0
    return out


def look_at(eye: Sequence[float] | torch.Tensor,
            target: Sequence[float] | torch.Tensor,
            up: Sequence[float] | torch.Tensor = (0.0, -1.0, 0.0),
            dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``world_to_camera`` for a camera at ``eye`` looking at ``target``.

    ``up`` defaults to world ``-y``, because in the OpenCV convention image
    rows grow *downward*: a camera whose "up" is +y renders upside down. Pass
    ``(0, 0, 1)``-style ups only if the world you are building is z-up, and be
    consistent about it for the whole scene.
    """
    eye_t = torch.as_tensor(eye, dtype=dtype)
    forward = torch.as_tensor(target, dtype=dtype) - eye_t
    norm = forward.norm()
    if norm < 1e-12:
        raise ValueError("eye and target coincide")
    forward = forward / norm
    up_t = torch.as_tensor(up, dtype=dtype)
    # Camera +x is right, +y is down, +z is forward. With `up` pointing toward
    # the top of the image, down = -up, and right = down x forward.
    right = torch.linalg.cross(-up_t, forward)
    if right.norm() < 1e-9:
        raise ValueError("up is parallel to the viewing direction")
    right = right / right.norm()
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack([right, down, forward])      # rows: camera axes in world
    out = torch.eye(4, dtype=dtype)
    out[:3, :3] = rotation
    out[:3, 3] = -rotation @ eye_t
    return out


@dataclass(frozen=True)
class Camera:
    """A pinhole camera: intrinsics in pixels, extrinsics as ``world_to_camera``."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    world_to_camera: torch.Tensor

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("image size must be positive")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")
        if tuple(self.world_to_camera.shape) != (4, 4):
            raise ValueError("world_to_camera must be 4x4")

    # -- construction ------------------------------------------------------

    @classmethod
    def from_fov(cls, fov_y_degrees: float, width: int, height: int,
                 world_to_camera: Optional[torch.Tensor] = None) -> "Camera":
        """Square pixels, principal point at the image centre."""
        if not 0.0 < fov_y_degrees < 180.0:
            raise ValueError("vertical field of view must lie in (0, 180)")
        f = 0.5 * height / math.tan(math.radians(fov_y_degrees) / 2.0)
        return cls(f, f, width / 2.0, height / 2.0, width, height,
                   torch.eye(4) if world_to_camera is None else world_to_camera)

    # -- derived -----------------------------------------------------------

    @property
    def camera_to_world(self) -> torch.Tensor:
        return rigid_inverse(self.world_to_camera)

    @property
    def position(self) -> torch.Tensor:
        """Camera centre in world coordinates."""
        return self.camera_to_world[:3, 3]

    @property
    def forward(self) -> torch.Tensor:
        """Unit viewing direction (+z of the camera) in world coordinates."""
        return self.world_to_camera[2, :3]

    @property
    def intrinsics(self) -> torch.Tensor:
        dtype = self.world_to_camera.dtype
        return torch.tensor([[self.fx, 0.0, self.cx],
                             [0.0, self.fy, self.cy],
                             [0.0, 0.0, 1.0]], dtype=dtype)

    def resized(self, width: int, height: int) -> "Camera":
        """Same camera, different resolution: intrinsics scale, pose does not."""
        sx, sy = width / self.width, height / self.height
        return replace(self, fx=self.fx * sx, fy=self.fy * sy,
                       cx=self.cx * sx, cy=self.cy * sy, width=width, height=height)

    # -- geometry ----------------------------------------------------------

    def to_camera(self, points: torch.Tensor) -> torch.Tensor:
        """World points ``[..., 3]`` to camera coordinates ``[..., 3]``."""
        w2c = self.world_to_camera.to(points)
        return points @ w2c[:3, :3].T + w2c[:3, 3]

    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """World points to pixel coordinates and depth.

        Returns ``(uv [..., 2], depth [..., 1])``. Points behind the camera
        get a non-positive depth and meaningless ``uv``; the caller masks them,
        because clamping here would silently turn behind-camera geometry into
        geometry at the image border.
        """
        cam = self.to_camera(points)
        z = cam[..., 2:3]
        safe = torch.where(z.abs() < 1e-9, torch.full_like(z, 1e-9), z)
        u = self.fx * cam[..., 0:1] / safe + self.cx
        v = self.fy * cam[..., 1:2] / safe + self.cy
        return torch.cat([u, v], dim=-1), z

    def unproject(self, uv: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Pixel coordinates and z-depth back to world points."""
        x = (uv[..., 0:1] - self.cx) / self.fx * depth
        y = (uv[..., 1:2] - self.cy) / self.fy * depth
        cam = torch.cat([x, y, depth], dim=-1)
        c2w = self.camera_to_world.to(cam)
        return cam @ c2w[:3, :3].T + c2w[:3, 3]

    def rays(self, stride: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """World-space ray origins and unit directions, one per pixel or patch.

        With ``stride > 1`` each ray passes through the *centre* of a
        ``stride x stride`` patch, which is the ray a patch token should be
        conditioned on. Sampling the patch's top-left pixel instead biases
        every patch ray by half a patch toward the image corner.
        """
        if stride < 1:
            raise ValueError("stride must be positive")
        dtype = self.world_to_camera.dtype
        h, w = self.height // stride, self.width // stride
        v = (torch.arange(h, dtype=dtype) + 0.5) * stride
        u = (torch.arange(w, dtype=dtype) + 0.5) * stride
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        local = torch.stack([(uu - self.cx) / self.fx,
                             (vv - self.cy) / self.fy,
                             torch.ones_like(uu)], dim=-1)
        c2w = self.camera_to_world
        directions = local @ c2w[:3, :3].T
        directions = directions / directions.norm(dim=-1, keepdim=True)
        origins = c2w[:3, 3].expand_as(directions)
        return origins, directions

    def plucker(self, stride: int = 1) -> torch.Tensor:
        """Plücker ray embedding ``(d, o x d)`` per pixel/patch: ``[H', W', 6]``.

        Invariant to the choice of origin along the ray -- replacing ``o`` by
        ``o + s d`` leaves ``o x d`` unchanged -- which is the property that
        makes it a representation of the *line*, not of a point.
        """
        origins, directions = self.rays(stride)
        moment = torch.linalg.cross(origins, directions, dim=-1)
        return torch.cat([directions, moment], dim=-1)

    def sees(self, points: torch.Tensor, margin: float = 0.0,
             near: float = 1e-3) -> torch.Tensor:
        """Boolean mask of points inside the view frustum (with pixel margin)."""
        uv, depth = self.project(points)
        z = depth[..., 0]
        return ((z > near)
                & (uv[..., 0] >= -margin) & (uv[..., 0] < self.width + margin)
                & (uv[..., 1] >= -margin) & (uv[..., 1] < self.height + margin))


# --------------------------------------------------------------------------
# trajectories and motion
# --------------------------------------------------------------------------


def orbit(template: Camera, target: Sequence[float], radius: float,
          n: int, elevation_degrees: float = 15.0,
          up: Sequence[float] = (0.0, -1.0, 0.0)) -> list[Camera]:
    """``n`` cameras evenly spaced on a circle around ``target``, all looking at it.

    The circle lies in the plane perpendicular to ``up``; elevation lifts it
    toward ``up``. A turntable, the standard trajectory for checking that a
    scene is consistent from every side.
    """
    if n < 1 or radius <= 0:
        raise ValueError("need a positive radius and at least one camera")
    dtype = template.world_to_camera.dtype
    up_t = torch.as_tensor(up, dtype=dtype)
    up_t = up_t / up_t.norm()
    seed = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    if abs(float(seed @ up_t)) > 0.9:
        seed = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
    a = torch.linalg.cross(up_t, seed)
    a = a / a.norm()
    b = torch.linalg.cross(up_t, a)
    centre = torch.as_tensor(target, dtype=dtype)
    elev = math.radians(elevation_degrees)
    cams = []
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        offset = radius * (math.cos(elev) * (math.cos(theta) * a + math.sin(theta) * b)
                           + math.sin(elev) * up_t)
        cams.append(replace(template, world_to_camera=look_at(centre + offset, centre,
                                                              up_t, dtype=dtype)))
    return cams


def _rotation_to_quaternion(r: torch.Tensor) -> torch.Tensor:
    """Rotation matrix to unit quaternion ``(w, x, y, z)``, numerically robust."""
    m = r.to(torch.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2
        q = torch.stack([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                         (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = torch.stack([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                         (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = torch.stack([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                         0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = torch.stack([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                         (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return (q / q.norm()).to(r.dtype)


def _quaternion_to_rotation(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = (q / q.norm()).unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


def interpolate(a: Camera, b: Camera, t: float) -> Camera:
    """Pose between ``a`` and ``b``: slerp on rotation, lerp on camera centre.

    Interpolating the rotation matrices entrywise is the tempting shortcut and
    it is wrong: the midpoint of two rotations is not a rotation, and the
    camera shears. Interpolating ``t`` (the extrinsic translation) rather than
    the camera *centre* is the other trap -- ``t`` depends on the rotation, so
    lerping it swings the camera along an arc it never actually travels.
    Intrinsics are taken from ``a``.
    """
    qa = _rotation_to_quaternion(a.world_to_camera[:3, :3])
    qb = _rotation_to_quaternion(b.world_to_camera[:3, :3])
    dot = float(qa @ qb)
    if dot < 0.0:                    # take the short way round
        qb, dot = -qb, -dot
    if dot > 0.9995:
        q = qa + t * (qb - qa)
    else:
        omega = math.acos(max(-1.0, min(1.0, dot)))
        q = (math.sin((1 - t) * omega) * qa + math.sin(t * omega) * qb) / math.sin(omega)
    rotation = _quaternion_to_rotation(q)
    centre = (1 - t) * a.position + t * b.position
    w2c = torch.eye(4, dtype=a.world_to_camera.dtype)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = -rotation @ centre
    return replace(a, world_to_camera=w2c)


def move(camera: Camera, forward: float = 0.0, right: float = 0.0, down: float = 0.0,
         yaw_degrees: float = 0.0, pitch_degrees: float = 0.0) -> Camera:
    """First-person motion in the camera's own frame -- the primitive actions map onto.

    Translation is applied along the camera's current axes (so "forward" means
    where it is looking), then yaw turns about the camera's *down* axis and
    pitch about its *right* axis. Rotating first and translating second would
    make "forward" mean the new heading, which is the other defensible
    convention; this one matches how a walking-camera game control feels,
    where a key press moves you along your current view.
    """
    c2w = camera.camera_to_world.clone()
    axes = c2w[:3, :3]                       # columns: right, down, forward in world
    c2w[:3, 3] = c2w[:3, 3] + axes @ torch.tensor([right, down, forward], dtype=c2w.dtype)
    yaw, pitch = math.radians(yaw_degrees), math.radians(pitch_degrees)
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    # Rotations expressed in the camera's local frame, applied on the right.
    r_yaw = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=c2w.dtype)
    r_pitch = torch.tensor([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=c2w.dtype)
    c2w[:3, :3] = axes @ r_yaw @ r_pitch
    return replace(camera, world_to_camera=rigid_inverse(c2w))


def trajectory(cameras: Iterable[Camera], steps_between: int) -> list[Camera]:
    """Densify a sparse list of keyframe cameras by pose interpolation."""
    keys = list(cameras)
    if len(keys) < 2 or steps_between < 1:
        return keys
    out: list[Camera] = []
    for a, b in zip(keys[:-1], keys[1:]):
        for i in range(steps_between):
            out.append(interpolate(a, b, i / steps_between))
    out.append(keys[-1])
    return out
