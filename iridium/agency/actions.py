"""The action vocabulary, and what an action is allowed to mean.

An action is ``<OP, operands...>``: a discrete opcode plus up to six continuous
operands. The opcode determines which operands are *used*; the rest are
unsupervised, because training a model to reproduce padding teaches it that
padding is meaningful.

Three properties are enforced rather than assumed:

* an action is **typed** — an opcode that needs a target names which operands
  are coordinates, and coordinates carry a frame (``iridium/contracts/frames``);
* an action is **reversible or declared irreversible** — ``UNDO`` exists and
  the environment maintains a stack; ops that cannot be undone say so;
* an action is **checkable** — every op has a postcondition the environment can
  evaluate, which is what makes behaviour cloning verifiable rather than
  merely imitative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional, Sequence

import numpy as np


class Op(IntEnum):
    NOOP = 0
    MOVE_CURSOR = 1
    CLICK = 2
    DOUBLE_CLICK = 3
    DRAG = 4
    SCROLL = 5
    TYPE_TEXT = 6
    KEY = 7
    SELECT_OBJECT = 8
    DESELECT_ALL = 9
    ADD_CUBE = 10
    ADD_SPHERE = 11
    ADD_PLANE = 12
    DELETE = 13
    TRANSLATE = 14
    ROTATE = 15
    SCALE = 16
    SET_PARAM = 17
    SET_MATERIAL = 18
    RENDER = 19
    RUN_SCRIPT = 20
    ASSERT = 21
    UNDO = 22
    SAVE = 23


N_OPS = len(Op)
N_SCALARS = 6

#: Which operand slots each opcode actually uses.
OPERAND_ARITY: dict[Op, int] = {
    Op.NOOP: 0, Op.MOVE_CURSOR: 2, Op.CLICK: 3, Op.DOUBLE_CLICK: 2,
    Op.DRAG: 4, Op.SCROLL: 2, Op.TYPE_TEXT: 1, Op.KEY: 1,
    Op.SELECT_OBJECT: 1, Op.DESELECT_ALL: 0,
    Op.ADD_CUBE: 4, Op.ADD_SPHERE: 4, Op.ADD_PLANE: 4, Op.DELETE: 1,
    Op.TRANSLATE: 4, Op.ROTATE: 4, Op.SCALE: 4,
    Op.SET_PARAM: 3, Op.SET_MATERIAL: 4, Op.RENDER: 2,
    Op.RUN_SCRIPT: 1, Op.ASSERT: 3, Op.UNDO: 0, Op.SAVE: 1,
}

IRREVERSIBLE: frozenset[Op] = frozenset({Op.RUN_SCRIPT, Op.SAVE, Op.RENDER})


@dataclass
class Action:
    op: Op
    operands: tuple[float, ...] = ()
    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.op = Op(self.op)
        need = OPERAND_ARITY[self.op]
        vals = tuple(float(v) for v in self.operands)
        if len(vals) < need:
            raise ValueError(
                f"{self.op.name} needs {need} operands, got {len(vals)}"
            )
        self.operands = vals[:N_SCALARS]

    @property
    def reversible(self) -> bool:
        return self.op not in IRREVERSIBLE

    def to_vector(self) -> np.ndarray:
        """``[op, s0..s5]`` — the row layout an action span carries."""
        row = np.zeros(1 + N_SCALARS, dtype=np.float32)
        row[0] = int(self.op)
        row[1 : 1 + len(self.operands)] = self.operands
        return row

    @classmethod
    def from_vector(cls, row: Sequence[float], text: str = "") -> "Action":
        op = Op(int(round(float(row[0]))))
        need = OPERAND_ARITY[op]
        return cls(op, tuple(float(v) for v in row[1 : 1 + need]), text)

    def operand_mask(self) -> np.ndarray:
        mask = np.zeros(N_SCALARS, dtype=np.float32)
        mask[: OPERAND_ARITY[self.op]] = 1.0
        return mask

    def describe(self) -> str:
        used = ", ".join(f"{v:g}" for v in self.operands[: OPERAND_ARITY[self.op]])
        text = f' "{self.text}"' if self.text else ""
        return f"{self.op.name}({used}){text}"


def actions_to_span_payload(actions: Sequence[Action]) -> np.ndarray:
    return np.stack([a.to_vector() for a in actions]) if actions else np.zeros((0, 1 + N_SCALARS), np.float32)


def operand_masks(actions: Sequence[Action]) -> np.ndarray:
    return np.stack([a.operand_mask() for a in actions]) if actions else np.zeros((0, N_SCALARS), np.float32)
