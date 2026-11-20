"""3D Gaussian scenes: the persistent memory this package's hybrid world model writes into.

A :class:`GaussianScene` is exactly what World Labs' Marble exports and what
INRIA's reference implementation, gsplat and every downstream viewer (SuperSplat,
PlayCanvas, the Unreal/Unity importers) read as a 3DGS ``.ply``: a cloud of
anisotropic 3D Gaussians, each with a position, an oriented scale, an opacity
and a view-dependent color given by spherical-harmonic coefficients. Storing
the scene this way rather than as, say, a NeRF MLP or a voxel grid is what lets
:func:`render` be *differentiable in closed form* -- every pixel is an
analytic function of the splat parameters, so a photometric loss against a
frame the model generated can update the scene directly (:func:`fit`), which
is the mechanism the rest of this package uses to fold new observations back
into persistent memory.

**The renderer here is a reference implementation, not the real thing.** The
official 3DGS rasterizer (and gsplat) tiles the screen and sorts/blends on the
GPU in a custom CUDA kernel processing millions of splats in real time; this
module is pure, unfused PyTorch that materializes an explicit
``[chunk, H, W]`` density tensor per chunk of splats. It is orders of
magnitude slower and is meant for producing a training signal and for tests
on scenes of hundreds to low thousands of splats, not for interactive
viewing. Anyone wanting real-time display should export via :func:`save_ply`
and open the result in gsplat/SuperSplat/Marble, which is exactly the point
of matching their file format.

Convention notes that matter for correctness (see :mod:`iridium.world.camera`
for the camera side): means live in world space in the same OpenCV/COLMAP
frame as :class:`~iridium.world.camera.Camera` (+x right, +y down, +z
forward), so ``camera.world_to_camera`` is applied to them directly with no
axis flip. Rotations are unit quaternions ``(w, x, y, z)`` -- scalar first,
matching the convention already used by ``camera._rotation_to_quaternion`` --
normalised on every use rather than kept normalised in storage, because a
gradient step on an unconstrained quaternion is simpler than one that has to
stay on the unit sphere, and renormalising once per forward pass is cheap
compared to the rest of the rasteriser.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from iridium.world.camera import Camera

__all__ = [
    "GaussianScene",
    "RenderOutput",
    "render",
    "save_ply",
    "load_ply",
    "fit",
]

# Spherical-harmonic normalisation constants (real SH basis, as used by the
# original 3DGS codebase's `sh_utils.py` and every compatible tool). C0 is
# Y_0^0 = 1/(2*sqrt(pi)); C1 is the shared coefficient of the three degree-1
# terms. Baking these in is what makes `sh0`/`sh_rest` mean the same physical
# quantity here as in a Marble/gsplat export, rather than an arbitrary
# rescaling of it.
_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199

# The 3DGS rasterizer inflates every projected 2D covariance by adding this to
# both diagonal entries (graphdeco-inria/gaussian-splatting, forward.cu,
# `computeCov2D`: `cov[0][0] += 0.3f; cov[1][1] += 0.3f;`). Without it, a
# splat whose projected footprint is smaller than a pixel evaluates its
# Gaussian density at only the pixel centres around it, which can all land in
# the tails -- the splat becomes invisible even though it should tint the
# pixel it falls inside. The obvious "fix", supersampling, is what this
# approximates cheaply: convolving the screen-space Gaussian with a ~1px
# Gaussian low-pass kernel and folding the two covariances together.
_LOW_PASS_PX = 0.3


def _inverse_sigmoid(p: float) -> float:
    return math.log(p / (1.0 - p))


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Unit-normalised ``(w, x, y, z)`` quaternions ``[..., 4]`` to rotation matrices ``[..., 3, 3]``."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)


@dataclass
class GaussianScene:
    """A cloud of ``N`` anisotropic 3D Gaussians -- position, shape, opacity, color.

    Every field is stored in the *unconstrained* space its optimiser sees,
    not the physical space it means: ``log_scales`` not scales, logits not
    probabilities, DC spherical-harmonic coefficients not RGB. That is a
    deliberate mismatch with a name like ``colors`` holding sigmoid-ed RGB
    directly, which is the friendlier-looking but wrong choice: it would
    need a clamp to stay in ``[0, 1]``, clamped values have zero gradient
    outside the clamp, and it would not match what a ``.ply`` file (or
    :func:`load_ply` fed one from Marble) actually stores. Keeping the
    unconstrained form here means :func:`fit` can run unconstrained Adam on
    every field directly and :func:`save_ply` never has to invert an
    activation to write a lossless file.

    Fields:
        means: ``[N, 3]`` world-space centres.
        log_scales: ``[N, 3]`` log of the per-axis standard deviation, so the
            physical scale is ``exp(log_scales)`` and is always positive by
            construction rather than by clamping.
        quats: ``[N, 4]`` rotation as ``(w, x, y, z)``, not necessarily
            normalised -- normalise on use (see module docstring).
        opacity_logits: ``[N]`` pre-sigmoid opacity.
        sh0: ``[N, 3]`` degree-0 SH DC term; ``color = 0.5 + _SH_C0 * sh0``
            reproduces exactly what 3DGS calls the "base color" and is
            piecewise-invertible, so import/export round-trips losslessly.
        sh_rest: ``[N, K, 3]`` higher-degree SH coefficients, ``K`` a
            multiple of 3 covering complete degrees (``K=3`` is degree 1,
            ``K=8`` would add degree 2, etc). May be empty (``K=0``): most
            scenes built by :meth:`random`/:meth:`from_points` have no
            view-dependent color and there is nothing to gain from padding
            it. :meth:`colors_rgb` and :func:`render` only evaluate up to
            degree 1 even if more is stored, since that is the only band this
            module implements the basis functions for.
    """

    means: torch.Tensor
    log_scales: torch.Tensor
    quats: torch.Tensor
    opacity_logits: torch.Tensor
    sh0: torch.Tensor
    sh_rest: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 0, 3))

    def __post_init__(self) -> None:
        n = self.means.shape[0]
        if self.sh_rest.numel() == 0 and self.sh_rest.shape[0] != n:
            # Allow the empty-tensor default to adopt N=0's shape trivially,
            # but a genuinely mismatched N is almost certainly a bug at the
            # call site, not intentional broadcasting.
            self.sh_rest = self.sh_rest.new_zeros((n, 0, 3))
        for name in ("log_scales", "quats", "opacity_logits", "sh0", "sh_rest"):
            got = getattr(self, name).shape[0]
            if got != n:
                raise ValueError(f"field {name!r} has {got} rows, means has {n}")

    # -- basic container protocol ------------------------------------------

    def __len__(self) -> int:
        return self.means.shape[0]

    def to(self, device=None, dtype=None) -> "GaussianScene":
        """Move/cast every field. ``dtype`` only applies to floating fields; ints (none here) would be skipped."""
        return GaussianScene(
            means=self.means.to(device=device, dtype=dtype),
            log_scales=self.log_scales.to(device=device, dtype=dtype),
            quats=self.quats.to(device=device, dtype=dtype),
            opacity_logits=self.opacity_logits.to(device=device, dtype=dtype),
            sh0=self.sh0.to(device=device, dtype=dtype),
            sh_rest=self.sh_rest.to(device=device, dtype=dtype),
        )

    def concat(self, other: "GaussianScene") -> "GaussianScene":
        """Merge two scenes into one cloud (e.g. fusing a newly generated frame's splats into memory).

        Requires matching ``sh_rest`` degree ``K``: silently zero-padding the
        shorter one would let two scenes with genuinely different SH bands
        merge into a cloud where the same column means different things for
        different splats, which is a much worse bug than the loud error here.
        """
        if self.sh_rest.shape[1] != other.sh_rest.shape[1]:
            raise ValueError(
                f"cannot concat scenes with different SH degree: "
                f"{self.sh_rest.shape[1]} vs {other.sh_rest.shape[1]} rest bands"
            )
        return GaussianScene(
            means=torch.cat([self.means, other.means], dim=0),
            log_scales=torch.cat([self.log_scales, other.log_scales], dim=0),
            quats=torch.cat([self.quats, other.quats], dim=0),
            opacity_logits=torch.cat([self.opacity_logits, other.opacity_logits], dim=0),
            sh0=torch.cat([self.sh0, other.sh0], dim=0),
            sh_rest=torch.cat([self.sh_rest, other.sh_rest], dim=0),
        )

    def select(self, mask: torch.Tensor) -> "GaussianScene":
        """Index every field by the same boolean or integer ``mask``."""
        return GaussianScene(
            means=self.means[mask],
            log_scales=self.log_scales[mask],
            quats=self.quats[mask],
            opacity_logits=self.opacity_logits[mask],
            sh0=self.sh0[mask],
            sh_rest=self.sh_rest[mask],
        )

    def requires_grad_(self) -> "GaussianScene":
        """Turn on gradients for every optimisable field in place; returns ``self`` for chaining."""
        for t in (self.means, self.log_scales, self.quats, self.opacity_logits, self.sh0, self.sh_rest):
            t.requires_grad_(True)
        return self

    # -- derived geometry ----------------------------------------------------

    def covariances(self) -> torch.Tensor:
        """World-space 3x3 covariances ``R S S^T R^T``, ``[N, 3, 3]``.

        Built as ``M = R * scales`` (scaling ``R``'s columns) then ``M M^T``
        rather than forming ``diag(scale^2)`` explicitly and matmul-ing three
        matrices: same result, one fewer ``[N,3,3]`` matmul, and it stays
        exactly symmetric to floating-point round-off since ``M M^T`` is
        symmetric by construction while ``R diag Rᵀ`` computed the naive way
        can drift by a few ULPs.
        """
        r = _quat_to_rotmat(self.quats)
        scales = self.log_scales.exp()
        m = r * scales.unsqueeze(-2)
        return m @ m.transpose(-1, -2)

    def colors_rgb(self, view_dirs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluate per-splat RGB, degree 0 (+ degree 1 if present and ``view_dirs`` given), clamped to ``[0, 1]``.

        ``view_dirs`` is the unit direction from each splat toward the
        viewer, ``[N, 3]``, in world space -- pass ``None`` when
        ``sh_rest`` is empty, or when a view-independent approximation is
        acceptable (e.g. for a quick preview).
        """
        color = 0.5 + _SH_C0 * self.sh0
        k = self.sh_rest.shape[1]
        if k >= 3 and view_dirs is not None:
            x, y, z = view_dirs.unbind(-1)
            basis = torch.stack([-y, z, -x], dim=-1)  # (m=-1, 0, +1)
            color = color + _SH_C1 * torch.einsum("nb,nbc->nc", basis, self.sh_rest[:, :3, :])
        return color.clamp(0.0, 1.0)

    # -- construction ----------------------------------------------------------

    @classmethod
    def random(cls, n: int, extent: float = 1.0,
               generator: Optional[torch.Generator] = None) -> "GaussianScene":
        """``n`` splats scattered uniformly in a cube of half-width ``extent``, random shape/color.

        Meant for smoke tests and gradient checks, not for a plausible scene:
        scales are drawn small (a few percent of ``extent``) so that splats
        do not blanket the whole cube, and quaternions come from a normal
        draw then get normalised on use, which is the standard trick for a
        uniform-on-the-sphere-ish rotation without rejection sampling.
        """
        g = generator
        means = (torch.rand(n, 3, generator=g) * 2 - 1) * extent
        log_scales = torch.log(torch.rand(n, 3, generator=g) * 0.04 * extent + 0.005 * extent)
        quats = torch.randn(n, 4, generator=g)
        opacity_logits = torch.randn(n, generator=g) * 0.5 + _inverse_sigmoid(0.6)
        sh0 = torch.randn(n, 3, generator=g) * 0.5
        return cls(means, log_scales, quats, opacity_logits, sh0)

    @classmethod
    def from_points(cls, points: torch.Tensor, rgb: torch.Tensor,
                     scale: float | torch.Tensor = 0.01) -> "GaussianScene":
        """Splats seeded directly from points + color, e.g. an unprojected depth map.

        This is the on-ramp from a rendered/generated frame back into scene
        memory: ``camera.unproject(uv, depth)`` gives ``points``, the frame
        itself gives ``rgb`` per pixel, and the result here is a scene ready
        to :meth:`concat` onto persistent memory or refine with :func:`fit`.
        Isotropic scale and identity rotation are deliberately naive (no
        estimate of surface orientation from neighbouring points) -- getting
        that right needs a local point-cloud normal estimate, which belongs
        in the caller (e.g. :mod:`world_state`), not silently guessed here.
        """
        n = points.shape[0]
        if rgb.shape != (n, 3):
            raise ValueError(f"rgb must be [{n}, 3], got {tuple(rgb.shape)}")
        scale_t = torch.as_tensor(scale, dtype=points.dtype).expand(n, 3) if not torch.is_tensor(scale) or scale.dim() == 0 \
            else scale
        log_scales = torch.log(scale_t.clamp_min(1e-8) * torch.ones(n, 3, dtype=points.dtype))
        quats = torch.zeros(n, 4, dtype=points.dtype)
        quats[:, 0] = 1.0
        opacity_logits = torch.full((n,), _inverse_sigmoid(0.95), dtype=points.dtype)
        sh0 = (rgb - 0.5) / _SH_C0
        return cls(points, log_scales, quats, opacity_logits, sh0)


@dataclass
class RenderOutput:
    """What :func:`render` produces for one camera.

    ``depth`` is the *alpha-normalised* expected depth (accumulated
    front-to-back weight, not raw accumulated ``weight * depth``): dividing
    by the accumulated alpha is what makes a single, non-opaque splat report
    its own depth at its centre instead of ``opacity * depth`` -- the
    un-normalised version is the more common mistake and it makes the
    reasonable-looking test "depth equals the splat's depth" fail whenever
    opacity isn't exactly 1.
    """

    rgb: torch.Tensor      # [H, W, 3]
    alpha: torch.Tensor    # [H, W]
    depth: torch.Tensor    # [H, W]
    visible: torch.Tensor  # [N] bool


def render(scene: GaussianScene, camera: Camera,
           background: Sequence[float] = (0.0, 0.0, 0.0),
           near: float = 0.01, max_splats: Optional[int] = None) -> RenderOutput:
    """Differentiable EWA-splatting rasteriser (Kerbl et al., *3D Gaussian Splatting for
    Real-Time Radiance Field Rendering*, SIGGRAPH 2023): project each 3D Gaussian to a 2D
    one via the perspective Jacobian, then alpha-composite front to back.

    Per visible splat: camera-space covariance ``Sigma_cam = W Sigma W^T``
    (``W`` the extrinsic's rotation, since translation doesn't affect
    covariance); the projection Jacobian ``J`` (2x3, `d(u,v)/d(x,y,z)`)
    linearises the perspective divide at the splat's own depth; ``Sigma' = J
    Sigma_cam J^T`` is the projected 2x2 screen-space covariance, inflated by
    :data:`_LOW_PASS_PX` on the diagonal (see module docstring). At each
    pixel centre ``(u+0.5, v+0.5)``, ``alpha = opacity * exp(-0.5 d^T
    Sigma'^-1 d)`` clamped to ``0.99`` (never fully opaque -- keeps the
    compositing well-conditioned and matches the reference implementation);
    splats are processed in front-to-back depth order and composited with
    running transmittance ``T``.

    Complexity and memory: with ``N`` visible splats and an ``H x W`` image,
    the exact algorithm is ``O(N * H * W)`` -- every splat is evaluated at
    every pixel, since this is a dense reference renderer with no
    screen-space tiling/culling per splat (that bookkeeping is most of what
    makes the CUDA kernel fast and most of what would make this one hard to
    keep readable and differentiable). Memory is bounded, not ``O(N*H*W)``:
    splats are processed in chunks of ``max_splats`` (default 256) in their
    front-to-back order, carrying only the running per-pixel transmittance
    ``[H, W]`` and accumulators between chunks, so peak memory is
    ``O(max_splats * H * W)`` regardless of ``N``. Compositing is exact
    across chunk boundaries because front-to-back alpha blending is
    associative in the transmittance it carries forward.

    Culling is depth (``> near``) plus a fixed pixel margin around the frame;
    a splat whose *centre* projects more than that margin outside the image
    is dropped even if its tails would reach into frame. This is a
    correctness/performance approximation appropriate for scenes seen from a
    stable camera; a splat sitting exactly at the frustum edge can dim near
    that edge as a result.
    """
    if len(background) != 3:
        raise ValueError("background must have 3 channels")
    n = len(scene)
    device = scene.means.device
    dtype = scene.means.dtype
    height, width = camera.height, camera.width
    bg = torch.as_tensor(background, dtype=dtype, device=device)

    visible = torch.zeros(n, dtype=torch.bool, device=device)
    if n == 0:
        rgb = bg.expand(height, width, 3).clone()
        return RenderOutput(rgb, torch.zeros(height, width, dtype=dtype, device=device),
                             torch.zeros(height, width, dtype=dtype, device=device), visible)

    cam_xyz = camera.to_camera(scene.means)          # [N,3]
    depth = cam_xyz[:, 2]
    uv = torch.stack([
        camera.fx * cam_xyz[:, 0] / depth.clamp_min(1e-9) + camera.cx,
        camera.fy * cam_xyz[:, 1] / depth.clamp_min(1e-9) + camera.cy,
    ], dim=-1)
    margin = 0.25 * max(width, height)
    frustum = ((uv[:, 0] > -margin) & (uv[:, 0] < width + margin)
               & (uv[:, 1] > -margin) & (uv[:, 1] < height + margin))
    mask = (depth > near) & frustum
    visible = mask.clone()
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        rgb = bg.expand(height, width, 3).clone()
        return RenderOutput(rgb, torch.zeros(height, width, dtype=dtype, device=device),
                             torch.zeros(height, width, dtype=dtype, device=device), visible)

    depth_v = depth[idx]
    order = torch.argsort(depth_v)                   # front-to-back
    idx = idx[order]
    depth_v = depth_v[order]
    uv_v = uv[idx]
    cam_xyz_v = cam_xyz[idx]

    w_rot = camera.world_to_camera[:3, :3].to(dtype=dtype, device=device)
    sigma_world = scene.covariances()[idx]            # [M,3,3]
    sigma_cam = w_rot @ sigma_world @ w_rot.T

    tx, ty, tz = cam_xyz_v[:, 0], cam_xyz_v[:, 1], cam_xyz_v[:, 2]
    limx = 1.3 * (width / (2 * camera.fx)) * tz
    limy = 1.3 * (height / (2 * camera.fy)) * tz
    tx = tx.clamp(-limx, limx)
    ty = ty.clamp(-limy, limy)
    zero = torch.zeros_like(tz)
    j = torch.stack([
        torch.stack([camera.fx / tz, zero, -camera.fx * tx / (tz * tz)], dim=-1),
        torch.stack([zero, camera.fy / tz, -camera.fy * ty / (tz * tz)], dim=-1),
    ], dim=-2)                                        # [M,2,3]
    sigma2d = j @ sigma_cam @ j.transpose(-1, -2)      # [M,2,2]
    sigma2d = sigma2d + torch.eye(2, dtype=dtype, device=device) * _LOW_PASS_PX

    a, b = sigma2d[:, 0, 0], sigma2d[:, 0, 1]
    c, d = sigma2d[:, 1, 0], sigma2d[:, 1, 1]
    det = (a * d - b * c).clamp_min(1e-12)
    inv_a, inv_b, inv_c, inv_d = d / det, -b / det, -c / det, a / det

    opacity = torch.sigmoid(scene.opacity_logits[idx])
    view_dirs = camera.position.to(dtype=dtype, device=device) - scene.means[idx]
    view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    colors = scene.colors_rgb(view_dirs)[idx] if scene.sh_rest.shape[1] >= 3 else scene.colors_rgb()[idx]

    py = (torch.arange(height, dtype=dtype, device=device) + 0.5)
    px = (torch.arange(width, dtype=dtype, device=device) + 0.5)
    grid_v, grid_u = torch.meshgrid(py, px, indexing="ij")   # [H,W] each

    chunk = max_splats or 256
    m = idx.numel()
    t_carry = torch.ones(height, width, dtype=dtype, device=device)
    rgb_acc = torch.zeros(height, width, 3, dtype=dtype, device=device)
    depth_acc = torch.zeros(height, width, dtype=dtype, device=device)
    alpha_acc = torch.zeros(height, width, dtype=dtype, device=device)
    min_alpha = 1.0 / 255.0

    for start in range(0, m, chunk):
        end = min(start + chunk, m)
        du = grid_u.unsqueeze(0) - uv_v[start:end, 0].view(-1, 1, 1)   # [C,H,W]
        dv = grid_v.unsqueeze(0) - uv_v[start:end, 1].view(-1, 1, 1)
        ia = inv_a[start:end].view(-1, 1, 1)
        ib = inv_b[start:end].view(-1, 1, 1)
        ic = inv_c[start:end].view(-1, 1, 1)
        idd = inv_d[start:end].view(-1, 1, 1)
        maha = ia * du * du + (ib + ic) * du * dv + idd * dv * dv
        gauss = torch.exp(-0.5 * maha)
        chunk_alpha = (opacity[start:end].view(-1, 1, 1) * gauss).clamp(max=0.99)
        chunk_alpha = torch.where(chunk_alpha < min_alpha, torch.zeros_like(chunk_alpha), chunk_alpha)

        one_minus = 1.0 - chunk_alpha
        cum = torch.cumprod(one_minus, dim=0)
        t_before_local = torch.cat([torch.ones_like(cum[:1]), cum[:-1]], dim=0)
        weight = t_carry.unsqueeze(0) * t_before_local * chunk_alpha        # [C,H,W]

        rgb_acc = rgb_acc + torch.einsum("chw,cj->hwj", weight, colors[start:end])
        depth_acc = depth_acc + (weight * depth_v[start:end].view(-1, 1, 1)).sum(dim=0)
        alpha_acc = alpha_acc + weight.sum(dim=0)
        t_carry = t_carry * cum[-1]

    rgb_out = rgb_acc + t_carry.unsqueeze(-1) * bg
    depth_out = depth_acc / alpha_acc.clamp_min(1e-9)
    return RenderOutput(rgb_out, 1.0 - t_carry, depth_out, visible)


# --------------------------------------------------------------------------
# .ply I/O -- standard 3D Gaussian Splatting format
# --------------------------------------------------------------------------
#
# Property list and order verified against INRIA's reference implementation
# (graphdeco-inria/gaussian-splatting, `scene/gaussian_model.py`,
# `construct_list_of_attributes`/`save_ply`), which gsplat, SuperSplat and the
# PlayCanvas/Unreal/Unity importers all consume: 'x y z nx ny nz f_dc_0..2
# f_rest_0..(3K-1) opacity scale_0..2 rot_0..3', binary_little_endian, one
# float32 per property per vertex. `f_rest` is channel-major (all K
# coefficients for R, then all K for G, then all K for B) because INRIA's
# code transposes `features_rest` to `[N, 3, K]` before flattening it into
# properties -- storing it band-major (interleaved per channel) is the
# mistake naive re-implementations make and it silently scrambles every
# imported scene's view-dependent color.
_PLY_HEADER_FIELDS_FIXED = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
_PLY_HEADER_FIELDS_TAIL = ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def save_ply(scene: GaussianScene, path: str) -> None:
    """Write ``scene`` as a binary-little-endian 3DGS ``.ply``.

    Writes `opacity` and `scale_*` in the *unconstrained* space
    (`opacity_logits`, `log_scales`) exactly as this module stores them --
    that is what the format actually specifies (every compatible tool
    applies its own sigmoid/exp on load) and it is the only way the round
    trip through :func:`load_ply` is exact to the float32 bit pattern. A
    "friendlier" writer that stores sigmoid(opacity) and exp(scale) directly
    produces a file that *looks* right in a viewer but is not a valid 3DGS
    ply and corrupts on reload by every real importer, including this one.
    """
    n = len(scene)
    k = scene.sh_rest.shape[1]
    rest_names = [f"f_rest_{i}" for i in range(3 * k)]
    names = _PLY_HEADER_FIELDS_FIXED + rest_names + _PLY_HEADER_FIELDS_TAIL

    means = scene.means.detach().cpu().numpy().astype("<f4")
    normals = np.zeros((n, 3), dtype="<f4")
    sh0 = scene.sh0.detach().cpu().numpy().astype("<f4")
    # channel-major: [N, 3, K] -> [N, 3K]
    rest = scene.sh_rest.detach().cpu().numpy().astype("<f4")
    rest_cm = np.transpose(rest, (0, 2, 1)).reshape(n, 3 * k) if k > 0 else np.zeros((n, 0), dtype="<f4")
    opacity = scene.opacity_logits.detach().cpu().numpy().astype("<f4").reshape(n, 1)
    scales = scene.log_scales.detach().cpu().numpy().astype("<f4")
    quats = scene.quats.detach().cpu().numpy().astype("<f4")

    data = np.concatenate([means, normals, sh0, rest_cm, opacity, scales, quats], axis=1).astype("<f4")
    assert data.shape[1] == len(names)

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
    ] + [f"property float {name}" for name in names] + ["end_header"]
    header = ("\n".join(header_lines) + "\n").encode("ascii")

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(data.tobytes(order="C"))


def load_ply(path: str) -> GaussianScene:
    """Read a binary-little-endian 3DGS ``.ply`` written by this module, INRIA's code, or gsplat.

    Parses the ASCII header for the property list and count rather than
    assuming the fixed layout :func:`save_ply` writes, so files with a
    different (but still ``float``-only, binary-little-endian) property set
    or SH degree still load; an ASCII-format or double-precision ``.ply`` --
    both valid PLY, neither what 3DGS tools emit -- raises rather than
    silently misreading the header as binary.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    header_end = raw.find(b"end_header\n")
    if header_end < 0:
        raise ValueError("not a valid ply file: no end_header")
    header_text = raw[:header_end].decode("ascii")
    body = raw[header_end + len(b"end_header\n"):]

    lines = header_text.splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError("not a valid ply file: missing magic")
    if "format binary_little_endian" not in header_text:
        raise ValueError("only binary_little_endian ply is supported")

    n = None
    prop_names: list[str] = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element" and parts[1] == "vertex":
            n = int(parts[2])
        elif parts[0] == "property":
            if parts[1] != "float":
                raise ValueError(f"only float properties are supported, got {parts[1]!r}")
            prop_names.append(parts[2])
    if n is None:
        raise ValueError("no 'element vertex' found")

    stride = len(prop_names)
    arr = np.frombuffer(body, dtype="<f4", count=n * stride).reshape(n, stride)
    cols = {name: arr[:, i] for i, name in enumerate(prop_names)}

    rest_indices = sorted((i for i, name in enumerate(prop_names) if name.startswith("f_rest_")),
                          key=lambda i: int(prop_names[i].split("_")[-1]))
    k = len(rest_indices) // 3

    def col(name):
        return torch.from_numpy(np.ascontiguousarray(cols[name])).to(torch.float32)

    means = torch.stack([col("x"), col("y"), col("z")], dim=-1)
    sh0 = torch.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], dim=-1)
    if k > 0:
        rest_flat = torch.stack([col(prop_names[i]) for i in rest_indices], dim=-1)  # [N, 3K], channel-major
        sh_rest = rest_flat.view(n, 3, k).transpose(1, 2).contiguous()               # -> [N, K, 3]
    else:
        sh_rest = torch.zeros(n, 0, 3)
    opacity_logits = col("opacity")
    log_scales = torch.stack([col("scale_0"), col("scale_1"), col("scale_2")], dim=-1)
    quats = torch.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], dim=-1)

    return GaussianScene(means, log_scales, quats, opacity_logits, sh0, sh_rest)


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


def fit(scene: GaussianScene, cameras: Sequence[Camera], images: Sequence[torch.Tensor],
        steps: int = 100, lr: float = 0.01) -> tuple[GaussianScene, list[float]]:
    """Photometric optimisation: adjust ``scene`` so it renders ``images`` from ``cameras``.

    Pure L1, no D-SSIM/perceptual term -- the point here is to demonstrate
    (and let tests check) that :func:`render` is actually differentiable
    end-to-end and that gradient descent on splat parameters converges, not
    to match 3DGS's published reconstruction quality, which also needs
    periodic densification/pruning of the splat set that this loop does not
    do. This is also, mechanically, how the rest of the package is meant to
    fuse a newly generated frame back into persistent scene memory: render
    the current scene, compare against the frame the model produced, and
    step the scene toward it instead of discarding the frame.

    Detaches and re-wraps every field as a fresh leaf tensor so ``scene``
    itself is left untouched; the returned scene is a new object sharing no
    storage with the input. Returns ``(fitted_scene, losses)`` where
    ``losses[i]`` is the mean-over-views L1 after optimiser step ``i``.
    """
    if len(cameras) != len(images):
        raise ValueError("cameras and images must have the same length")
    if len(cameras) == 0:
        raise ValueError("need at least one (camera, image) pair")

    params = {
        "means": scene.means.detach().clone().requires_grad_(True),
        "log_scales": scene.log_scales.detach().clone().requires_grad_(True),
        "quats": scene.quats.detach().clone().requires_grad_(True),
        "opacity_logits": scene.opacity_logits.detach().clone().requires_grad_(True),
        "sh0": scene.sh0.detach().clone().requires_grad_(True),
    }
    sh_rest = scene.sh_rest.detach().clone()
    if sh_rest.numel() > 0:
        sh_rest.requires_grad_(True)
        params["sh_rest"] = sh_rest

    optimizer = torch.optim.Adam(list(params.values()), lr=lr)
    losses: list[float] = []

    for _ in range(steps):
        working = GaussianScene(
            means=params["means"], log_scales=params["log_scales"], quats=params["quats"],
            opacity_logits=params["opacity_logits"], sh0=params["sh0"],
            sh_rest=params.get("sh_rest", sh_rest),
        )
        optimizer.zero_grad()
        total = 0.0
        loss = torch.zeros((), dtype=scene.means.dtype)
        for cam, img in zip(cameras, images):
            out = render(working, cam)
            loss = loss + (out.rgb - img).abs().mean()
        loss = loss / len(cameras)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    fitted = GaussianScene(
        means=params["means"].detach(), log_scales=params["log_scales"].detach(),
        quats=params["quats"].detach(), opacity_logits=params["opacity_logits"].detach(),
        sh0=params["sh0"].detach(), sh_rest=params.get("sh_rest", sh_rest).detach(),
    )
    return fitted, losses
