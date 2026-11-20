"""Persistent scene memory: keyframes, frustum-overlap retrieval, coverage, fusion.

This is the piece that makes revisit consistency a property of the *data
structure* rather than a hope about the model. Genie-3-style autoregressive
generation has no persistent scene at all -- "what is behind me" is whatever
the frame history happens to encode, and a model that has drifted has nothing
to check itself against. Storing a splat scene plus a bank of keyframes and
retrieving from it by *view overlap* gives the frame model something to
anchor to that does not depend on it having predicted well a hundred steps
ago.

The retrieval key is the one design decision worth arguing about, so the
argument is here rather than left implicit in the code: **overlap by
frustum/geometry, not by camera-centre distance.** Two cameras can sit at the
same point in space and see nothing in common (a character spinning in place
looks at four different walls from one spot), and two cameras far apart can
see the same wall dead-on. Centre distance gets the first case backwards and
the second case is exactly the "look away and come back" scenario a world
model has to get right. Overlap is estimated by unprojecting a coarse sample
of one camera's pixels (using its stored depth where available, otherwise a
depth guess) into world points and measuring what fraction of them the other
camera's frustum contains (:meth:`Camera.sees`) -- a cheap proxy for
intersection-over-union of the two view volumes that needs nothing more than
what a `Keyframe` already stores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence

import numpy as np
import torch

from .camera import Camera

Source = Literal["observed", "generated"]

__all__ = ["Keyframe", "WorldState"]


@dataclass
class Keyframe:
    """One remembered view: the camera it was seen from, and what was seen.

    ``depth`` is optional because a keyframe fresh out of an RGB-only frame
    model has none yet -- it can still anchor future overlap tests coarsely
    (see ``WorldState.retrieve``) and can still be looked at, it just cannot
    be unprojected into the splat scene until depth exists. ``source``
    distinguishes ground truth from the model's own output because the two
    should never be silently treated as equally trustworthy fusion targets
    (a generated frame with a hallucinated object should not become permanent
    "known geometry" as readily as an observed one -- callers can use this
    field to bias or skip fusion for generated content).
    """

    camera: Camera
    rgb: torch.Tensor                     # [H, W, 3]
    depth: Optional[torch.Tensor]         # [H, W] or None
    step: int
    source: Source = "observed"


def _sample_grid_uv(height: int, width: int, grid: int) -> torch.Tensor:
    """``grid`` x ``grid`` pixel-centre coordinates spanning the image, ``[grid*grid, 2]``."""
    v = (torch.arange(grid, dtype=torch.float32) + 0.5) * (height / grid)
    u = (torch.arange(grid, dtype=torch.float32) + 0.5) * (width / grid)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)


def _overlap_score(query: Camera, keyframe: Keyframe, grid: int = 8,
                    default_depth: float = 5.0) -> float:
    """Fraction of a coarse sample of ``keyframe``'s content visible from ``query``.

    With a depth map, the sample is unprojected from the keyframe's own
    camera -- exact, in the sense that it queries points the keyframe
    actually observed. Without one (a keyframe freshly written by an
    RGB-only frame model), there is nothing to unproject *from* the
    keyframe, so the sample is cast the other way: a coarse grid of query
    rays is pushed out to ``default_depth`` and checked against the
    keyframe's frustum. This is a weaker signal (it tests "would the
    keyframe have seen roughly this volume of space", not "did it see this
    surface") but it degrades gracefully rather than refusing to retrieve
    anything for depth-less memory.
    """
    if keyframe.depth is not None:
        uv = _sample_grid_uv(keyframe.camera.height, keyframe.camera.width, grid)
        vi = uv[:, 1].long().clamp(0, keyframe.camera.height - 1)
        ui = uv[:, 0].long().clamp(0, keyframe.camera.width - 1)
        depth = keyframe.depth[vi, ui].reshape(-1, 1).to(torch.float32)
        valid = depth[:, 0] > 1e-6
        if not bool(valid.any()):
            return 0.0
        points = keyframe.camera.unproject(uv[valid], depth[valid])
        seen = query.sees(points)
        return float(seen.float().mean())
    uv = _sample_grid_uv(query.height, query.width, grid)
    depth = torch.full((uv.shape[0], 1), default_depth, dtype=torch.float32)
    points = query.unproject(uv, depth)
    seen = keyframe.camera.sees(points)
    return float(seen.float().mean())


@dataclass
class WorldState:
    """The persistent world: an optional splat scene plus a bounded keyframe bank.

    ``scene`` is left as ``Any`` rather than typed against ``splats.GaussianScene``
    so that this module never imports ``splats`` at module scope -- the
    concurrent module under active development should not be able to break
    an import of ``world_state`` just by having a syntax error mid-edit, and
    every place this module needs a splat-specific operation takes it as an
    injected callable instead (dependency injection over a hard import, the
    same reason ``frame_model`` is injected in ``rollout.py``).
    """

    scene: object = None
    max_keyframes: int = 32
    keyframes: list[Keyframe] = field(default_factory=list)

    # -- retrieval -----------------------------------------------------

    def retrieve(self, camera: Camera, k: int = 4, grid: int = 8,
                 default_depth: float = 5.0) -> list[Keyframe]:
        """The ``k`` stored keyframes whose view overlaps ``camera`` most.

        See the module docstring for why this is overlap and not centre
        distance. Ties (e.g. an empty bank, or ``k`` >= bank size) resolve by
        returning everything available rather than padding, so callers must
        handle ``len(result) < k``.
        """
        if not self.keyframes:
            return []
        scored = [(_overlap_score(camera, kf, grid, default_depth), i)
                  for i, kf in enumerate(self.keyframes)]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [self.keyframes[i] for _, i in scored[:max(k, 0)]]

    def coverage(self, camera: Camera, renderer: Optional[Callable] = None,
                 background: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> float:
        """Fraction of ``camera``'s view already explained by stored geometry.

        This is the signal that separates "render the known wall" from "the
        model has to imagine this" -- it is deliberately *not* keyframe
        overlap: a keyframe can overlap a view heavily and still leave most
        of its pixels unfused (fusion only adds points into low-coverage
        regions, see ``fuse``), so coverage has to come from the scene
        itself, not from what has merely been looked at. Requires an
        injected ``renderer`` (``splats.render``-shaped) because this module
        does not import ``splats``; with no renderer or no scene yet, nothing
        is known, so this returns 0.0 rather than raising -- an empty world
        legitimately has zero coverage everywhere.
        """
        if renderer is None or self.scene is None or len(self.scene) == 0:
            return 0.0
        out = renderer(self.scene, camera, background)
        return float(out.alpha.mean())

    # -- capacity --------------------------------------------------------

    def add_keyframe(self, keyframe: Keyframe) -> None:
        """Append, then evict down to ``max_keyframes`` if over budget."""
        self.keyframes.append(keyframe)
        while len(self.keyframes) > self.max_keyframes:
            self._evict_one()

    def _evict_one(self) -> None:
        """Drop the keyframe most redundant with the rest of the bank.

        Not FIFO: FIFO evicts by age, which has nothing to do with what is
        spatially unique. A rollout that lingers in one room produces many
        near-duplicate keyframes of it and then walks into a new room once --
        FIFO would age out that one new-room keyframe (the only view of
        unexplored geometry) long before it ages out the twentieth
        near-identical view of the room the camera is sitting in. Instead,
        for each keyframe, average its overlap against every other keyframe;
        the one with the highest average overlap is the one whose content the
        rest of the bank already explains, so removing it loses the least
        unique coverage. O(n^2) in bank size, which is fine at
        ``max_keyframes`` in the tens-to-low-hundreds this is sized for.
        """
        n = len(self.keyframes)
        if n <= 1:
            if self.keyframes:
                self.keyframes.pop(0)
            return
        redundancy = [0.0] * n
        for i in range(n):
            total = 0.0
            for j in range(n):
                if i == j:
                    continue
                total += _overlap_score(self.keyframes[i].camera, self.keyframes[j])
            redundancy[i] = total / (n - 1)
        worst = max(range(n), key=lambda i: redundancy[i])
        self.keyframes.pop(worst)

    # -- fusion ------------------------------------------------------------

    def fuse(self, keyframe: Keyframe, renderer: Optional[Callable] = None,
             from_points_fn: Optional[Callable] = None,
             fit_fn: Optional[Callable] = None,
             coverage_threshold: float = 0.35, grid: int = 48,
             point_scale: float = 0.02, fit_steps: int = 0,
             fit_lr: float = 1e-2) -> None:
        """Fold a frame's content into the splat scene, then remember it.

        The keyframe is always appended to the bank (a depth-less generated
        frame is still useful for retrieval and revisit comparison even
        though it cannot be turned into geometry). Geometry fusion itself
        only runs when ``keyframe.depth`` is present, and only *adds* splats
        in pixels where current coverage is below ``coverage_threshold`` --
        the minimal-viable version of "anchor to what is known, extend where
        it is not": re-splatting already-covered pixels on every step would
        both waste splats and let a frame model's per-step noise slowly drift
        the scene it is supposed to be anchored to.

        ``from_points_fn`` defaults to a lazy import of
        ``splats.GaussianScene.from_points`` -- never imported at module
        scope, per the module docstring -- so a test can inject a fake with
        no dependency on that module existing yet. ``fit_fn`` (shaped like
        ``splats.fit``) is optional least-effort refinement; with
        ``fit_steps == 0`` (the default) it is never called, since a single
        unprojected-points scene is already usable and fitting on every fuse
        call would be the dominant cost of the whole rollout loop.
        """
        if keyframe.depth is None:
            self.add_keyframe(keyframe)
            return

        camera = keyframe.camera
        height, width = camera.height, camera.width
        uv = _sample_grid_uv(height, width, grid)
        vi = uv[:, 1].long().clamp(0, height - 1)
        ui = uv[:, 0].long().clamp(0, width - 1)
        depth = keyframe.depth[vi, ui].reshape(-1, 1).to(torch.float32)
        valid = depth[:, 0] > 1e-6

        if renderer is not None and self.scene is not None and len(self.scene) > 0:
            out = renderer(self.scene, camera, (0.0, 0.0, 0.0))
            alpha = out.alpha[vi, ui]
            low_coverage = alpha < coverage_threshold
            valid = valid & low_coverage

        if bool(valid.any()):
            points = camera.unproject(uv[valid], depth[valid])
            rgb = keyframe.rgb[vi[valid], ui[valid]].to(torch.float32)
            if from_points_fn is None:
                from .splats import GaussianScene
                from_points_fn = GaussianScene.from_points
            new_scene = from_points_fn(points, rgb, point_scale)
            if self.scene is None or len(self.scene) == 0:
                self.scene = new_scene
            else:
                self.scene = self.scene.concat(new_scene)
            if fit_steps > 0 and fit_fn is not None:
                cameras = [kf.camera for kf in self.keyframes] + [camera]
                images = [kf.rgb for kf in self.keyframes] + [keyframe.rgb]
                self.scene, _ = fit_fn(self.scene, cameras, images, fit_steps, fit_lr)

        self.add_keyframe(keyframe)

    # -- serialisation -------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Keyframes as one ``.npz`` each; the scene as raw tensor arrays.

        The scene is saved by reading its dataclass fields directly (``means``,
        ``log_scales``, ...) rather than calling into ``splats`` -- every
        field a ``GaussianScene`` has is already a plain tensor, so no import
        of that module is needed to serialise or restore one, only to
        reconstruct the dataclass on load (done lazily, see ``load``).
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        for i, kf in enumerate(self.keyframes):
            cam = kf.camera
            np.savez(
                path / f"keyframe_{i:05d}.npz",
                fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy,
                width=cam.width, height=cam.height,
                world_to_camera=cam.world_to_camera.detach().cpu().numpy(),
                rgb=kf.rgb.detach().cpu().numpy(),
                depth=(kf.depth.detach().cpu().numpy() if kf.depth is not None
                      else np.zeros(0, dtype=np.float32)),
                has_depth=kf.depth is not None,
                step=kf.step, source=kf.source,
            )
        if self.scene is not None and len(self.scene) > 0:
            np.savez(
                path / "scene.npz",
                means=self.scene.means.detach().cpu().numpy(),
                log_scales=self.scene.log_scales.detach().cpu().numpy(),
                quats=self.scene.quats.detach().cpu().numpy(),
                opacity_logits=self.scene.opacity_logits.detach().cpu().numpy(),
                sh0=self.scene.sh0.detach().cpu().numpy(),
                sh_rest=(self.scene.sh_rest.detach().cpu().numpy()
                        if getattr(self.scene, "sh_rest", None) is not None
                        else np.zeros(0, dtype=np.float32)),
            )

    @classmethod
    def load(cls, directory: str | Path, max_keyframes: int = 32) -> "WorldState":
        path = Path(directory)
        keyframes: list[Keyframe] = []
        for kf_path in sorted(path.glob("keyframe_*.npz")):
            data = np.load(kf_path, allow_pickle=False)
            w2c = torch.as_tensor(data["world_to_camera"], dtype=torch.float32)
            camera = Camera(float(data["fx"]), float(data["fy"]), float(data["cx"]),
                            float(data["cy"]), int(data["width"]), int(data["height"]), w2c)
            depth = (torch.as_tensor(data["depth"], dtype=torch.float32)
                    if bool(data["has_depth"]) else None)
            keyframes.append(Keyframe(
                camera=camera,
                rgb=torch.as_tensor(data["rgb"], dtype=torch.float32),
                depth=depth,
                step=int(data["step"]),
                source=str(data["source"]),
            ))
        scene = None
        scene_path = path / "scene.npz"
        if scene_path.exists():
            from .splats import GaussianScene
            data = np.load(scene_path, allow_pickle=False)
            sh_rest = data["sh_rest"]
            scene = GaussianScene(
                means=torch.as_tensor(data["means"], dtype=torch.float32),
                log_scales=torch.as_tensor(data["log_scales"], dtype=torch.float32),
                quats=torch.as_tensor(data["quats"], dtype=torch.float32),
                opacity_logits=torch.as_tensor(data["opacity_logits"], dtype=torch.float32),
                sh0=torch.as_tensor(data["sh0"], dtype=torch.float32),
                sh_rest=(torch.as_tensor(sh_rest, dtype=torch.float32)
                        if sh_rest.size else None),
            )
        return cls(scene=scene, max_keyframes=max_keyframes, keyframes=keyframes)
