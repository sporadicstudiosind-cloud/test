"""A deterministic 3-D scene editor with a real renderer.

This is the environment the agentic phase trains and is evaluated in. It is not
a mock in the sense of "returns canned responses": it maintains a scene graph,
applies transforms, renders an image the model actually consumes through the
image codec, and evaluates goal predicates exactly.

Why not drive Blender directly? Because a training loop that requires Blender
cannot run in a test, and an environment that cannot run in a test cannot hold
a regression. ``BlenderScript`` below emits the *same* action sequence as real
``bpy`` calls, so a policy trained here transfers as a program rather than as a
set of pixel reflexes — and ``tests/unit/test_agency.py`` checks that every op
in the vocabulary has a Blender translation, so the two cannot drift apart.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

import numpy as np

from .actions import IRREVERSIBLE, N_SCALARS, Action, Op


@dataclass
class SceneObject:
    kind: str                       # "cube" | "sphere" | "plane"
    position: np.ndarray            # (3,)
    scale: np.ndarray               # (3,)
    rotation: float = 0.0           # radians about z
    colour: tuple[float, float, float] = (0.7, 0.7, 0.75)
    name: str = ""

    def copy(self) -> "SceneObject":
        return SceneObject(
            self.kind, self.position.copy(), self.scale.copy(),
            self.rotation, self.colour, self.name,
        )


@dataclass
class SceneState:
    objects: list[SceneObject] = field(default_factory=list)
    selected: Optional[int] = None
    cursor: tuple[float, float] = (0.5, 0.5)
    params: dict[str, float] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)

    def copy(self) -> "SceneState":
        return SceneState(
            [o.copy() for o in self.objects], self.selected, self.cursor,
            dict(self.params), list(self.log),
        )


class SceneEditor:
    """Apply actions, render the result, and evaluate goals exactly."""

    def __init__(self, resolution: int = 32, history: int = 32) -> None:
        self.resolution = resolution
        self.history_limit = history
        self.state = SceneState()
        self._history: list[SceneState] = []
        self.irreversible_log: list[str] = []

    # -- lifecycle --------------------------------------------------------

    def reset(self, state: Optional[SceneState] = None) -> SceneState:
        self.state = state.copy() if state else SceneState()
        self._history.clear()
        self.irreversible_log.clear()
        return self.state

    def _push(self) -> None:
        self._history.append(self.state.copy())
        if len(self._history) > self.history_limit:
            self._history.pop(0)

    # -- stepping ---------------------------------------------------------

    def step(self, action: Action) -> tuple[SceneState, dict[str, Any]]:
        info: dict[str, Any] = {"op": action.op.name, "applied": True}
        if action.op is not Op.UNDO:
            self._push()
        op, s = action.op, action.operands

        if op is Op.NOOP:
            pass
        elif op is Op.MOVE_CURSOR:
            self.state.cursor = (float(s[0]), float(s[1]))
        elif op in (Op.CLICK, Op.DOUBLE_CLICK):
            self.state.cursor = (float(s[0]), float(s[1]))
            self.state.selected = self._pick(*self.state.cursor)
            info["selected"] = self.state.selected
        elif op is Op.DRAG:
            self.state.cursor = (float(s[2]), float(s[3]))
            if self.state.selected is not None:
                obj = self.state.objects[self.state.selected]
                obj.position[0] += float(s[2]) - float(s[0])
                obj.position[1] += float(s[3]) - float(s[1])
        elif op is Op.SCROLL:
            self.state.params["zoom"] = self.state.params.get("zoom", 1.0) * (
                1.0 + 0.1 * float(s[1])
            )
        elif op is Op.TYPE_TEXT:
            self.state.log.append(action.text)
        elif op is Op.KEY:
            self.state.log.append(f"key:{int(s[0])}")
        elif op is Op.SELECT_OBJECT:
            idx = int(s[0])
            if 0 <= idx < len(self.state.objects):
                self.state.selected = idx
            else:
                info["applied"] = False
                info["error"] = f"no object {idx}"
        elif op is Op.DESELECT_ALL:
            self.state.selected = None
        elif op in (Op.ADD_CUBE, Op.ADD_SPHERE, Op.ADD_PLANE):
            kind = {Op.ADD_CUBE: "cube", Op.ADD_SPHERE: "sphere", Op.ADD_PLANE: "plane"}[op]
            obj = SceneObject(
                kind=kind,
                position=np.array([s[0], s[1], s[2]], dtype=float),
                scale=np.full(3, max(float(s[3]), 1e-3)),
                name=action.text or f"{kind}_{len(self.state.objects)}",
            )
            self.state.objects.append(obj)
            self.state.selected = len(self.state.objects) - 1
        elif op is Op.DELETE:
            idx = int(s[0])
            if 0 <= idx < len(self.state.objects):
                self.state.objects.pop(idx)
                self.state.selected = None
            else:
                info["applied"] = False
        elif op in (Op.TRANSLATE, Op.ROTATE, Op.SCALE):
            idx = int(s[0])
            if not (0 <= idx < len(self.state.objects)):
                info["applied"] = False
                info["error"] = f"no object {idx}"
            else:
                obj = self.state.objects[idx]
                if op is Op.TRANSLATE:
                    obj.position = obj.position + np.array([s[1], s[2], s[3]])
                elif op is Op.ROTATE:
                    obj.rotation += float(s[1])
                else:
                    obj.scale = obj.scale * np.array(
                        [max(s[1], 1e-6), max(s[2], 1e-6), max(s[3], 1e-6)]
                    )
        elif op is Op.SET_PARAM:
            self.state.params[action.text or f"p{int(s[0])}"] = float(s[1])
        elif op is Op.SET_MATERIAL:
            idx = int(s[0])
            if 0 <= idx < len(self.state.objects):
                self.state.objects[idx].colour = (
                    float(np.clip(s[1], 0, 1)),
                    float(np.clip(s[2], 0, 1)),
                    float(np.clip(s[3], 0, 1)),
                )
            else:
                info["applied"] = False
        elif op is Op.RENDER:
            info["image"] = self.render()
        elif op is Op.RUN_SCRIPT:
            self.irreversible_log.append(action.text)
            info["script"] = action.text
        elif op is Op.ASSERT:
            info["assertion"] = self._assertion(action)
        elif op is Op.UNDO:
            if self._history:
                self.state = self._history.pop()
            else:
                info["applied"] = False
                info["error"] = "nothing to undo"
        elif op is Op.SAVE:
            self.irreversible_log.append(f"save:{action.text}")
        else:                                           # pragma: no cover
            raise ValueError(f"unhandled op {op}")

        if op in IRREVERSIBLE:
            info["irreversible"] = True
        return self.state, info

    def run(self, actions: Sequence[Action]) -> list[dict[str, Any]]:
        return [self.step(a)[1] for a in actions]

    # -- perception -------------------------------------------------------

    def _pick(self, x: float, y: float) -> Optional[int]:
        """Nearest object under the cursor in screen space, or None."""
        best, best_d = None, 0.12
        for i, obj in enumerate(self.state.objects):
            sx, sy = self._project(obj.position)
            d = float(np.hypot(sx - x, sy - y))
            if d < best_d:
                best, best_d = i, d
        return best

    @staticmethod
    def _project(p: np.ndarray) -> tuple[float, float]:
        """Orthographic top-down projection into the unit square."""
        return float(0.5 + 0.1 * p[0]), float(0.5 + 0.1 * p[1])

    def render(self) -> np.ndarray:
        """Painter's-algorithm raster, ``[3, R, R]`` in ``[0, 1]``.

        Deterministic, dependency-free, and differentiable-free: this is an
        observation, not a training target for a renderer.
        """
        r = self.resolution
        img = np.zeros((3, r, r), dtype=np.float32)
        img[0] += 0.06
        img[1] += 0.07
        img[2] += 0.09
        ys, xs = np.mgrid[0:r, 0:r]
        u = (xs + 0.5) / r
        v = (ys + 0.5) / r
        for i, obj in enumerate(sorted(self.state.objects, key=lambda o: o.position[2])):
            cx, cy = self._project(obj.position)
            half = 0.05 * float(np.mean(obj.scale))
            if obj.kind == "sphere":
                mask = ((u - cx) ** 2 + (v - cy) ** 2) <= half ** 2
            elif obj.kind == "plane":
                mask = (np.abs(u - cx) <= half * 2.5) & (np.abs(v - cy) <= half * 0.35)
            else:
                ca, sa = np.cos(-obj.rotation), np.sin(-obj.rotation)
                du, dv = u - cx, v - cy
                ru, rv = ca * du - sa * dv, sa * du + ca * dv
                mask = (np.abs(ru) <= half) & (np.abs(rv) <= half)
            for c in range(3):
                img[c] = np.where(mask, obj.colour[c], img[c])
        if self.state.selected is not None and 0 <= self.state.selected < len(self.state.objects):
            cx, cy = self._project(self.state.objects[self.state.selected].position)
            ring = np.abs(np.hypot(u - cx, v - cy) - 0.085) < 0.008
            img[1] = np.where(ring, 1.0, img[1])
        cx, cy = self.state.cursor
        cross = (np.abs(u - cx) < 0.5 / r) | (np.abs(v - cy) < 0.5 / r)
        near = np.hypot(u - cx, v - cy) < 0.05
        img[0] = np.where(cross & near, 1.0, img[0])
        return np.clip(img, 0.0, 1.0)

    # -- goals ------------------------------------------------------------

    def _assertion(self, action: Action) -> bool:
        idx, axis, target = int(action.operands[0]), int(action.operands[1]), action.operands[2]
        if not (0 <= idx < len(self.state.objects)):
            return False
        return bool(abs(self.state.objects[idx].position[axis % 3] - target) < 1e-6)


@dataclass(frozen=True)
class Goal:
    """A checkable target state. ``satisfied`` is exact, not a similarity."""

    description: str
    n_objects: Optional[int] = None
    kinds: Optional[tuple[str, ...]] = None
    positions: Optional[tuple[tuple[float, float, float], ...]] = None
    tolerance: float = 1e-4

    def satisfied(self, state: SceneState) -> bool:
        if self.n_objects is not None and len(state.objects) != self.n_objects:
            return False
        if self.kinds is not None:
            if tuple(o.kind for o in state.objects) != self.kinds:
                return False
        if self.positions is not None:
            if len(state.objects) != len(self.positions):
                return False
            for obj, want in zip(state.objects, self.positions):
                if float(np.max(np.abs(obj.position - np.array(want)))) > self.tolerance:
                    return False
        return True

    def progress(self, state: SceneState) -> float:
        """A dense score in [0, 1]. Only ``satisfied`` gates a reward."""
        score, total = 0.0, 0.0
        if self.n_objects is not None:
            total += 1.0
            score += 1.0 if len(state.objects) == self.n_objects else 0.0
        if self.kinds is not None:
            total += 1.0
            got = tuple(o.kind for o in state.objects)
            match = sum(1 for a, b in zip(got, self.kinds) if a == b)
            score += match / max(len(self.kinds), 1)
        if self.positions is not None:
            total += 1.0
            if state.objects:
                errs = [
                    float(np.max(np.abs(o.position - np.array(w))))
                    for o, w in zip(state.objects, self.positions)
                ]
                score += float(np.mean([1.0 / (1.0 + e) for e in errs]))
        return score / total if total else 0.0


BLENDER_TRANSLATION: dict[Op, str] = {
    Op.NOOP: "pass",
    Op.MOVE_CURSOR: "bpy.context.scene.cursor.location = ({0}, {1}, 0)",
    Op.CLICK: "bpy.ops.view3d.select(location=({0}, {1}))",
    Op.DOUBLE_CLICK: "bpy.ops.view3d.select(location=({0}, {1}), deselect_all=False)",
    Op.DRAG: "bpy.ops.transform.translate(value=({2}-{0}, {3}-{1}, 0))",
    Op.SCROLL: "bpy.ops.view3d.zoom(delta={1})",
    Op.TYPE_TEXT: "bpy.ops.font.text_insert(text={text!r})",
    Op.KEY: "bpy.ops.screen.keyframe_jump(next={0})",
    Op.SELECT_OBJECT: "bpy.context.view_layer.objects.active = bpy.data.objects[int({0})]",
    Op.DESELECT_ALL: "bpy.ops.object.select_all(action='DESELECT')",
    Op.ADD_CUBE: "bpy.ops.mesh.primitive_cube_add(location=({0}, {1}, {2}), size={3})",
    Op.ADD_SPHERE: "bpy.ops.mesh.primitive_uv_sphere_add(location=({0}, {1}, {2}), radius={3})",
    Op.ADD_PLANE: "bpy.ops.mesh.primitive_plane_add(location=({0}, {1}, {2}), size={3})",
    Op.DELETE: "bpy.data.objects.remove(bpy.data.objects[int({0})], do_unlink=True)",
    Op.TRANSLATE: "bpy.data.objects[int({0})].location += Vector(({1}, {2}, {3}))",
    Op.ROTATE: "bpy.data.objects[int({0})].rotation_euler[2] += {1}",
    Op.SCALE: "bpy.data.objects[int({0})].scale = ({1}, {2}, {3})",
    Op.SET_PARAM: "bpy.context.scene[{text!r}] = {1}",
    Op.SET_MATERIAL: "set_material(bpy.data.objects[int({0})], ({1}, {2}, {3}, 1))",
    Op.RENDER: "bpy.ops.render.render(write_still=True)",
    Op.RUN_SCRIPT: "exec({text!r})",
    Op.ASSERT: "assert abs(bpy.data.objects[int({0})].location[int({1})] - {2}) < 1e-6",
    Op.UNDO: "bpy.ops.ed.undo()",
    Op.SAVE: "bpy.ops.wm.save_as_mainfile(filepath={text!r})",
}


def to_blender(actions: Sequence[Action]) -> str:
    """Emit a real ``bpy`` script for an action sequence.

    The environment above is a stand-in for Blender's *state*, not for its API.
    Emitting the script is how a policy learned in the stand-in reaches the
    real application: as a program that can be reviewed before it runs, rather
    than as synthetic mouse events nobody can audit.
    """
    lines = ["import bpy", "from mathutils import Vector", ""]
    for a in actions:
        template = BLENDER_TRANSLATION[a.op]
        ops = list(a.operands) + [0.0] * (N_SCALARS - len(a.operands))
        lines.append(template.format(*ops, text=a.text))
    return "\n".join(lines)
