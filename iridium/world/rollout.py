"""The interactive loop: action -> camera move -> render/retrieve/generate/fuse.

Each step is the hybrid from ``iridium/world/__init__.py`` spelled out as
code: move the camera by the action, render the persistent scene from the
new pose (the anchor -- concrete evidence for what is already known), pull
the ``k`` most-overlapping keyframes from memory (context for what nearby
views looked like), hand all of it plus the camera's own Plücker rays to a
frame model, then fuse whatever comes back into the scene so the *next*
revisit sees it too. Nothing here decides what the frame model looks like --
that is Iridium-1 itself, wired in later -- this module only fixes the
contract it must satisfy and ships one trivial, correct implementation of
that contract (:class:`RenderOnlyFrameModel`) to run the loop against before
a trained model exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Union

import torch

from .camera import Camera, move
from .world_state import Keyframe, WorldState

__all__ = [
    "ActionMap", "FrameContext", "Frame", "RenderOnlyFrameModel",
    "WorldRollout", "RevisitPair", "revisit_consistency",
]

MoveKwargs = dict[str, float]

# -- actions -----------------------------------------------------------------

#: unit (forward, right, down, yaw, pitch) direction per named key. Translation
#: entries are unit steps scaled by `move_speed`; rotation entries by
#: `turn_speed_degrees`. WASD is the FPS-standard layout; the others are
#: named aliases for the same six degrees of freedom `Camera.move` exposes.
_DEFAULT_KEYMAP: dict[str, tuple[float, float, float, float, float]] = {
    "forward": (1.0, 0.0, 0.0, 0.0, 0.0), "w": (1.0, 0.0, 0.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0, 0.0, 0.0), "backward": (-1.0, 0.0, 0.0, 0.0, 0.0),
    "s": (-1.0, 0.0, 0.0, 0.0, 0.0),
    "left": (0.0, -1.0, 0.0, 0.0, 0.0), "a": (0.0, -1.0, 0.0, 0.0, 0.0),
    "right": (0.0, 1.0, 0.0, 0.0, 0.0), "d": (0.0, 1.0, 0.0, 0.0, 0.0),
    "up": (0.0, 0.0, -1.0, 0.0, 0.0), "e": (0.0, 0.0, -1.0, 0.0, 0.0),
    "down": (0.0, 0.0, 1.0, 0.0, 0.0), "q": (0.0, 0.0, 1.0, 0.0, 0.0),
    "yaw_left": (0.0, 0.0, 0.0, -1.0, 0.0),
    "yaw_right": (0.0, 0.0, 0.0, 1.0, 0.0),
    "pitch_up": (0.0, 0.0, 0.0, 0.0, -1.0),
    "pitch_down": (0.0, 0.0, 0.0, 0.0, 1.0),
}

_STR_ACTION_RE = re.compile(r"^\s*([A-Za-z_]+)\s*([-+]?[0-9]*\.?[0-9]+)?\s*$")


@dataclass
class ActionMap:
    """Turns a user action -- key name, ``"forward 0.5"`` string, dict, or a
    6-DoF continuous vector -- into the kwargs :func:`camera.move` takes.

    Three input shapes, each covering a different actual caller:

    * **string**, e.g. ``"w"`` or ``"forward 0.5"``: a discrete key, looked up
      in ``keymap``, optionally scaled by a trailing magnitude (default 1.0).
      This is keyboard-style input -- WASD, arrow-key yaw/pitch.
    * **dict**: taken as literal ``move()`` kwargs, unscaled. This is the
      escape hatch for a caller that has already computed exact deltas (e.g.
      replaying a logged trajectory) and does not want ``move_speed`` applied
      a second time.
    * **sequence of 5 or 6 floats** (list, tuple, tensor, ndarray): a
      continuous action vector ``[forward, right, down, yaw, pitch]`` or
      ``[forward, right, down, yaw, pitch, roll]``. The first three entries
      scale by ``move_speed``, the next two by ``turn_speed_degrees``. A
      sixth "roll" entry is accepted (continuous 6-DoF policies commonly
      emit one) and dropped: ``Camera.move`` has deliberately no roll
      parameter (see its docstring -- camera "up" is a scene-wide
      convention, not a per-frame degree of freedom), so a roll component
      here would either silently do nothing useful or require reinventing
      what "up" means mid-rollout, which is out of scope for an action map.
    """

    move_speed: float = 0.5
    turn_speed_degrees: float = 15.0
    keymap: dict[str, tuple[float, float, float, float, float]] = field(
        default_factory=lambda: dict(_DEFAULT_KEYMAP))

    def resolve(self, action: Union[str, dict, Sequence[float]]) -> MoveKwargs:
        if isinstance(action, str):
            return self._resolve_string(action)
        if isinstance(action, dict):
            return self._resolve_dict(action)
        return self._resolve_vector(action)

    def _resolve_string(self, action: str) -> MoveKwargs:
        match = _STR_ACTION_RE.match(action)
        if not match:
            raise ValueError(f"unrecognised action string {action!r}")
        key, magnitude_text = match.group(1).lower(), match.group(2)
        if key not in self.keymap:
            raise KeyError(f"no action mapped for key {key!r}")
        magnitude = float(magnitude_text) if magnitude_text is not None else 1.0
        forward, right, down, yaw, pitch = self.keymap[key]
        return dict(
            forward=forward * magnitude * self.move_speed,
            right=right * magnitude * self.move_speed,
            down=down * magnitude * self.move_speed,
            yaw_degrees=yaw * magnitude * self.turn_speed_degrees,
            pitch_degrees=pitch * magnitude * self.turn_speed_degrees,
        )

    @staticmethod
    def _resolve_dict(action: dict) -> MoveKwargs:
        allowed = {"forward", "right", "down", "yaw_degrees", "pitch_degrees"}
        unknown = set(action) - allowed
        if unknown:
            raise ValueError(f"unknown move() kwargs in action dict: {sorted(unknown)}")
        return {k: float(v) for k, v in action.items()}

    def _resolve_vector(self, action: Sequence[float]) -> MoveKwargs:
        values = [float(v) for v in action]
        if len(values) not in (5, 6):
            raise ValueError(
                f"continuous action vector must have 5 or 6 entries, got {len(values)}")
        forward, right, down, yaw, pitch = values[:5]
        return dict(
            forward=forward * self.move_speed,
            right=right * self.move_speed,
            down=down * self.move_speed,
            yaw_degrees=yaw * self.turn_speed_degrees,
            pitch_degrees=pitch * self.turn_speed_degrees,
        )


# -- frame model contract -----------------------------------------------------


@dataclass
class FrameContext:
    """Everything a frame model is conditioned on for one step.

    ``anchor_rgb``/``anchor_alpha``/``anchor_depth`` are the stored scene
    rendered from ``camera`` -- ``None`` when there is no renderer or no
    scene yet, in which case the frame model is generating into unexplored
    space with no geometric anchor at all. ``memory`` is
    ``WorldState.retrieve``'s output: keyframes chosen by view overlap with
    ``camera``, each carrying its own camera so the model can relate them
    geometrically (e.g. via their own Plücker rays) rather than just seeing
    unlabelled reference images. ``previous_frame`` is the immediately
    preceding step's output, for temporal continuity independent of the
    persistent-memory path.
    """

    camera: Camera
    plucker: torch.Tensor                       # [H', W', 6]
    anchor_rgb: Optional[torch.Tensor]           # [H, W, 3] or None
    anchor_alpha: Optional[torch.Tensor]         # [H, W] or None
    anchor_depth: Optional[torch.Tensor]         # [H, W] or None
    memory: list[Keyframe]
    previous_frame: Optional["Frame"]
    action: Any


@dataclass
class Frame:
    """One step's output, plus the context it was produced from (for logging/metrics)."""

    camera: Camera
    rgb: torch.Tensor                            # [H, W, 3]
    depth: Optional[torch.Tensor]                # [H, W] or None
    step: int
    context: FrameContext


class RenderOnlyFrameModel:
    """The identity frame model: returns the anchor render, generating nothing.

    This is not a strawman -- it is the correct baseline for two different
    purposes. As a *lower bound*, it is what the system produces with zero
    learned generative capability: pure re-rendering of stored geometry, no
    hallucinated content, so its quality on any benchmark is what "the splat
    memory alone" buys before a frame model contributes anything on top. As a
    *test fixture*, it makes the rest of the rollout loop's correctness
    checkable without a trained model: revisiting the same pose on a static
    scene must reproduce the same anchor render exactly (see
    ``revisit_consistency``), which is a property of the memory and retrieval
    code, not of a network's generalisation, and this is the model that lets
    the test isolate it.

    Falls back to a mid-grey frame the same shape as the requested camera
    when there is no anchor yet (nothing rendered because no scene exists),
    since a frame model's contract requires *some* rgb every step, not
    ``None``.
    """

    def __call__(self, context: FrameContext) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if context.anchor_rgb is not None:
            return context.anchor_rgb, context.anchor_depth
        cam = context.camera
        return torch.full((cam.height, cam.width, 3), 0.5, dtype=torch.float32), None


FrameModel = Callable[[FrameContext],
                      Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]]]
Renderer = Callable[[object, Camera, tuple[float, float, float]], Any]


@dataclass
class WorldRollout:
    """The render -> condition -> generate -> fuse loop, one step per action.

    ``frame_model`` is the sole integration point for the learned model: it
    is called with a :class:`FrameContext` and must return either an
    ``rgb [H, W, 3]`` tensor, or an ``(rgb, depth)`` tuple where ``depth`` is
    an ``[H, W]`` tensor or ``None``. ``rgb`` must match ``context.camera``'s
    resolution. Returning depth is optional but is what lets a step's output
    be fused back into the splat scene (see ``WorldState.fuse``); an
    RGB-only model still works, it just never grows the persistent geometry.
    """

    world_state: WorldState
    frame_model: FrameModel
    camera: Camera
    renderer: Optional[Renderer] = None
    action_map: ActionMap = field(default_factory=ActionMap)
    memory_k: int = 4
    plucker_stride: int = 8
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    step_count: int = 0
    frames: list[Frame] = field(default_factory=list)
    _last_frame: Optional[Frame] = None

    def _render_anchor(self, camera: Camera):
        if (self.renderer is None or self.world_state.scene is None
                or len(self.world_state.scene) == 0):
            return None, None, None
        out = self.renderer(self.world_state.scene, camera, self.background)
        return out.rgb, out.alpha, out.depth

    def _advance(self, camera: Camera, action: Any) -> Frame:
        anchor_rgb, anchor_alpha, anchor_depth = self._render_anchor(camera)
        memory = self.world_state.retrieve(camera, self.memory_k)
        context = FrameContext(
            camera=camera,
            plucker=camera.plucker(self.plucker_stride),
            anchor_rgb=anchor_rgb, anchor_alpha=anchor_alpha, anchor_depth=anchor_depth,
            memory=memory, previous_frame=self._last_frame, action=action,
        )
        result = self.frame_model(context)
        rgb, depth = result if isinstance(result, tuple) else (result, anchor_depth)

        self.step_count += 1
        keyframe = Keyframe(camera=camera, rgb=rgb, depth=depth,
                            step=self.step_count, source="generated")
        self.world_state.fuse(keyframe, renderer=self.renderer)

        frame = Frame(camera=camera, rgb=rgb, depth=depth,
                      step=self.step_count, context=context)
        self.camera = camera
        self._last_frame = frame
        self.frames.append(frame)
        return frame

    def step(self, action: Union[str, dict, Sequence[float]]) -> Frame:
        """One WASD/continuous-action step: move the current camera and advance."""
        delta = self.action_map.resolve(action)
        new_camera = move(self.camera, **delta)
        return self._advance(new_camera, action)

    def play(self, actions: Sequence[Union[str, dict, Sequence[float]]]) -> list[Frame]:
        """Run a sequence of actions from the current camera, in order."""
        return [self.step(a) for a in actions]

    def fly(self, cameras: Sequence[Camera]) -> list[Frame]:
        """Camera-path mode: visit each given camera directly, bypassing ``action_map``.

        For pre-authored trajectories (``camera.orbit``, ``camera.trajectory``)
        where the caller already has explicit poses and there is no
        first-person action to resolve them from.
        """
        return [self._advance(cam, action=None) for cam in cameras]


# -- the Genie-3 "look away and back" test, made measurable ------------------


@dataclass
class RevisitPair:
    """One detected revisit: two steps whose cameras nearly coincide, and how much the frame changed."""

    i: int
    j: int
    error: float


def revisit_consistency(frames: Sequence[Frame], position_threshold: float = 0.2,
                        direction_threshold: float = 0.98) -> list[RevisitPair]:
    """Photometric error between pairs of frames whose cameras nearly coincide.

    "Nearly coincide" is deliberately stricter than the frustum-overlap test
    ``WorldState.retrieve`` uses: retrieval wants *any* useful reference view,
    but a revisit-consistency check is specifically Genie 3's "look away and
    look back" scenario -- the camera returns to close to the same position
    *and* the same heading, not merely to a view that shares some content
    with an old one. Two cameras qualify as a revisit when their positions
    are within ``position_threshold`` (world units) and the cosine similarity
    of their forward vectors exceeds ``direction_threshold``. For each
    qualifying pair the error is the mean absolute pixel difference between
    the two frames' ``rgb`` (both are renders/generations of the same scene
    from ~the same pose, so no reprojection is needed -- they should already
    line up pixel-for-pixel if the world is consistent).

    On a static scene driven purely by :class:`RenderOnlyFrameModel`, this is
    ≈0 by construction: the anchor render is a deterministic function of
    camera pose and the (unchanged) scene, so revisiting a pose reproduces
    the same image. Any positive value under that combination indicates a
    bug in retrieval, fusion, or the renderer contract, not model error.
    """
    pairs: list[RevisitPair] = []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            a, b = frames[i].camera, frames[j].camera
            if a.width != b.width or a.height != b.height:
                continue
            dist = float((a.position - b.position).norm())
            cos = float((a.forward @ b.forward)
                       / (a.forward.norm() * b.forward.norm() + 1e-12))
            if dist <= position_threshold and cos >= direction_threshold:
                diff = (frames[i].rgb.to(torch.float32) - frames[j].rgb.to(torch.float32))
                pairs.append(RevisitPair(i=i, j=j, error=float(diff.abs().mean())))
    return pairs
