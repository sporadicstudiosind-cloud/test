"""Training tasks whose answers can be *checked*, not merely scored.

A loss curve going down is not evidence of competence. Every task here comes
with an exact checker, so the report in ``docs/results.md`` can say what
fraction of held-out items the model got *right* against an independent
computation — the analytic law, the deterministic solver, or the environment's
own goal predicate.

Five families, covering the capability claims that the architecture is for:

+----------------------+-------------------------------+------------------------+
| family               | what it exercises             | checked against        |
+----------------------+-------------------------------+------------------------+
| channel_depth        | native physical competence    | Manning's law          |
| channel_intervention | counterfactual reasoning      | the exact ratio law    |
| field_rollout        | continuous field prediction   | the spectral solver    |
| scene_goal           | agentic actuation             | the goal predicate     |
| false_premise        | resistance to sycophancy      | the premise's truth    |
+----------------------+-------------------------------+------------------------+

Splits are by *parameter range*, not by random subset: the extrapolation split
uses discharges outside the training band, so an item cannot be answered by
having seen a neighbouring one. This is the same discipline the earlier
advection slice used and for the same reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from ..agency.actions import Action, Op, actions_to_span_payload
from ..agency.scene import Goal, SceneEditor, SceneState
from ..codecs.spans import (
    Sample, Span, decode_quantity, patchify, quantity_span, text_span,
)
from ..physics.fluid2d import NavierStokes2D, FluidState, taylor_green
from ..physics.operators import vorticity
from ..physics.shallow_water import (
    Channel,
    ShallowWater1D,
    critical_depth,
    normal_depth,
)

# Byte-level text with a small reserved block for control markers.
TEXT_OFFSET = 16
BOS, EOS, SEP, PAD_CTRL = 1, 2, 3, 4
VERDICT_TRUE, VERDICT_FALSE = 5, 6
VERDICT_NAME = {VERDICT_TRUE: "TRUE", VERDICT_FALSE: "FALSE"}
#: The model's own "I don't know": one control token, so abstaining is a single
#: decision the model makes, graded exactly, and never a phrase it can half-say.
UNKNOWN = 13

TRAIN_DISCHARGE = (1.0, 8.0)
EXTRAPOLATION_DISCHARGE = (9.0, 14.0)
SLOPE_RANGE = (0.0008, 0.0060)
MANNING_RANGE = (0.018, 0.045)


def control_span(code: int, supervised: bool = True) -> Span:
    """A structural marker.

    ``supervised=False`` makes it part of the *prompt* rather than part of the
    answer, which matters more than it looks: ``evaluation.prompt_only`` keeps
    exactly the unsupervised spans, so a separator marked supervised is dropped
    at evaluation time and the model is asked for its answer one position
    earlier than training ever put it. That produces a model with a training
    loss of exactly zero and a graded error of 32%, which is a confusing
    afternoon.
    """
    return Span("control", np.array([code], dtype=np.int64), supervised=supervised)


def encode_text(text: str, supervised: bool = True) -> Span:
    return text_span(text, supervised=supervised, offset=TEXT_OFFSET)


def decode_text(ids: Sequence[int]) -> str:
    out = bytearray()
    for i in ids:
        v = int(i) - TEXT_OFFSET
        if 0 <= v < 256:
            out.append(v)
    return out.decode("utf-8", errors="replace")


@dataclass
class Item:
    """One training example plus everything needed to grade it."""

    sample: Sample
    family: str
    answer: str = ""
    check: Optional[Callable[[str], bool]] = None
    truth: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""

    def grade(self, produced: str) -> bool:
        if self.check is not None:
            return bool(self.check(produced))
        return produced.strip() == self.answer.strip()


# --------------------------------------------------------------------------
# 1 & 2 · open-channel depth, and the intervention that started all of this
# --------------------------------------------------------------------------


def _fmt(value: float, places: int = 4) -> str:
    return f"{value:.{places}f}"


def channel_depth_item(rng: np.random.Generator, split: str = "train") -> Item:
    """Manning normal depth, as a *quantity* problem rather than a string one.

    Inputs enter through the quantity codec as values carrying roles; the
    answer leaves through the quantity head as a number. Nothing is spelled out
    in decimal digits, and that is the whole difference between this working
    and not: the same mapping learned from digit-bytes puts 18.6% of answers
    inside a 2% tolerance, and learned from typed values, 100%.
    """
    lo, hi = TRAIN_DISCHARGE if split != "extrapolation" else EXTRAPOLATION_DISCHARGE
    q = float(rng.uniform(lo, hi))
    s0 = float(rng.uniform(*SLOPE_RANGE))
    n = float(rng.uniform(*MANNING_RANGE))
    h = normal_depth(q, s0, n)
    return Item(
        sample=Sample(
            [
                control_span(BOS, supervised=False),
                encode_text("normal depth", supervised=False),
                quantity_span([("slope", s0), ("manning", n), ("discharge", q)],
                              supervised=False),
                control_span(SEP, supervised=False),
                quantity_span([("depth", h)], supervised=True),
                control_span(EOS),
            ],
            meta={"family": "channel_depth", "split": split},
        ),
        family="channel_depth",
        answer=f"{h:.4f}",
        prompt=f"normal depth | S={s0:.4f} n={n:.3f} q={q:.2f}",
        truth={"q": q, "slope": s0, "manning": n, "value": h, "role": "depth",
               "h_critical": critical_depth(q)},
        check=lambda produced, h=h: _within(produced, h, 0.02),
    )


def channel_intervention_item(rng: np.random.Generator, split: str = "train") -> Item:
    """"What if I change the water?" - the depth ratio, emitted as a number."""
    lo, hi = TRAIN_DISCHARGE if split != "extrapolation" else EXTRAPOLATION_DISCHARGE
    q = float(rng.uniform(lo, hi))
    factor = float(rng.choice([1.25, 1.5, 2.0, 2.5, 3.0]))
    s0 = float(rng.uniform(*SLOPE_RANGE))
    n = float(rng.uniform(*MANNING_RANGE))
    ratio = factor ** 0.6
    return Item(
        sample=Sample(
            [
                control_span(BOS, supervised=False),
                encode_text("depth ratio", supervised=False),
                quantity_span([("slope", s0), ("manning", n), ("discharge", q),
                               ("factor", factor)], supervised=False),
                control_span(SEP, supervised=False),
                quantity_span([("ratio", ratio)], supervised=True),
                control_span(EOS),
            ],
            meta={"family": "channel_intervention", "split": split},
        ),
        family="channel_intervention",
        answer=f"{ratio:.4f}",
        prompt=f"depth ratio | S={s0:.4f} n={n:.3f} q={q:.2f} x{factor:.2f}",
        truth={"q": q, "factor": factor, "value": ratio, "role": "ratio",
               "critical_ratio": factor ** (2.0 / 3.0),
               "velocity_ratio": factor ** 0.4},
        check=lambda produced, r=ratio: _within(produced, r, 0.02),
    )


def _within(produced: str, target: float, rel: float) -> bool:
    try:
        value = float(produced.strip().split("|")[0])
    except (ValueError, IndexError):
        return False
    return abs(value - target) <= rel * max(abs(target), 1e-9)


# --------------------------------------------------------------------------
# 3 · continuous field rollout, checked against the solver
# --------------------------------------------------------------------------


@dataclass
class FieldEpisode:
    frames: list[np.ndarray]      # each [C, H, W]
    nu: float
    dt: float


def field_episode(
    rng: np.random.Generator, n: int = 16, nu: float = 0.05,
    dt: float = 0.05, steps: int = 6, modes: int = 3,
) -> FieldEpisode:
    """A short, resolved 2-D flow: channels are (u, v, vorticity, speed)."""
    solver = NavierStokes2D(n, nu=nu)
    amp = rng.uniform(0.5, 1.5)
    phase = rng.uniform(0, 2 * np.pi, size=2)
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    k = int(rng.integers(1, modes + 1))
    u = amp * np.cos(k * xx + phase[0]) * np.sin(k * yy + phase[1])
    v = -amp * np.sin(k * xx + phase[0]) * np.cos(k * yy + phase[1])
    state = FluidState(u, v, 0.0)
    frames = []
    for _ in range(steps + 1):
        w = vorticity(state.u, state.v)
        speed = np.sqrt(state.u ** 2 + state.v ** 2)
        frames.append(np.stack([state.u, state.v, w, speed]).astype(np.float32))
        state = solver.step(state, dt)
    return FieldEpisode(frames, nu, dt)


def field_rollout_item(
    rng: np.random.Generator, split: str = "train", patch: int = 8
) -> Item:
    ep = field_episode(rng)
    # A one-step horizon makes persistence (emit the input unchanged) nearly
    # perfect, so a model can score well while learning nothing about the
    # dynamics. The horizon is the whole episode for that reason.
    a, b = ep.frames[0], ep.frames[-1]
    tok_a, grid = patchify(a, (patch, patch))
    tok_b, _ = patchify(b, (patch, patch))
    horizon = ep.dt * (len(ep.frames) - 1)
    prompt = f"FIELD|nu={ep.nu:.3f}|T={horizon:.2f}|next="
    return Item(
        sample=Sample(
            [
                control_span(BOS, supervised=False),
                encode_text(prompt, supervised=False),
                Span("field", tok_a.astype(np.float32), grid=grid, supervised=False),
                control_span(SEP, supervised=False),
                Span("field", tok_b.astype(np.float32), grid=grid, supervised=True,
                     observed=True),
                control_span(EOS),
            ],
            meta={"family": "field_rollout", "split": split},
        ),
        family="field_rollout",
        prompt=prompt,
        truth={"target": tok_b, "grid": grid, "nu": ep.nu, "dt": ep.dt},
    )


# --------------------------------------------------------------------------
# 4 · agentic scene goals, checked by the environment
# --------------------------------------------------------------------------

_KINDS = {"cube": Op.ADD_CUBE, "sphere": Op.ADD_SPHERE, "plane": Op.ADD_PLANE}


def scene_goal_item(
    rng: np.random.Generator, split: str = "train",
    resolution: int = 16, patch: int = 8,
) -> Item:
    n_objects = int(rng.integers(1, 4))
    kinds = tuple(rng.choice(list(_KINDS), size=n_objects))
    positions = tuple(
        tuple(float(np.round(rng.uniform(-3, 3), 1)) for _ in range(3))
        for _ in range(n_objects)
    )
    actions = [
        Action(_KINDS[k], (p[0], p[1], p[2], 1.0))
        for k, p in zip(kinds, positions)
    ]
    goal = Goal(
        description="place the listed objects",
        n_objects=n_objects, kinds=kinds, positions=positions,
    )
    editor = SceneEditor(resolution=resolution)
    editor.reset()
    before = editor.render()
    spec = "SCENE|" + "|".join(
        f"{k}@{p[0]:.1f},{p[1]:.1f},{p[2]:.1f}" for k, p in zip(kinds, positions)
    ) + "|do="
    img_tokens, img_grid = patchify(before, (patch, patch))
    return Item(
        sample=Sample(
            [
                control_span(BOS, supervised=False),
                encode_text(spec, supervised=False),
                Span("image", img_tokens.astype(np.float32), grid=img_grid,
                     supervised=False),
                control_span(SEP, supervised=False),
                Span("action", actions_to_span_payload(actions), supervised=True,
                     atomic=False),
                control_span(EOS),
            ],
            meta={"family": "scene_goal", "split": split},
        ),
        family="scene_goal",
        prompt=spec,
        truth={"goal": goal, "actions": actions},
        check=lambda _produced, g=goal, acts=actions: _replay(acts, g),
    )


def _replay(actions: Sequence[Action], goal: Goal) -> bool:
    editor = SceneEditor(resolution=8)
    editor.reset()
    editor.run(actions)
    return goal.satisfied(editor.state)


# --------------------------------------------------------------------------
# 5 · false premises. The model must refuse the premise, not the question.
# --------------------------------------------------------------------------

FALSE_PREMISES: tuple[tuple[str, str, str], ...] = (
    (
        "doubling the discharge doubles the flow depth",
        "NO|depth scales as q^0.60, so doubling gives 1.5157",
        "the Manning normal-depth exponent is 3/5, not 1",
    ),
    (
        "water becomes compressible at 1 atm when mixed with oil",
        "NO|both remain effectively incompressible at 1 atm",
        "mixing does not change the bulk modulus by orders of magnitude",
    ),
    (
        "a steady open channel can carry more mass out than in",
        "NO|steady flow has equal inflow and outflow",
        "steady state means the stored mass is constant",
    ),
    (
        "the Froude number is dimensional",
        "NO|the Froude number is dimensionless",
        "it is a velocity divided by a wave speed",
    ),
    (
        "halving the slope halves the normal depth",
        "NO|depth scales as S^-0.30, so halving gives 1.2311",
        "the Manning slope exponent is -3/10",
    ),
    (
        "vorticity is conserved pointwise in a viscous flow",
        "NO|viscosity diffuses vorticity",
        "only the inviscid case conserves it along particle paths",
    ),
)

TRUE_PREMISES: tuple[tuple[str, str], ...] = (
    ("doubling the discharge raises the depth by about 1.52", "YES|confirmed"),
    ("the Froude number is dimensionless", "YES|confirmed"),
    ("viscosity diffuses vorticity", "YES|confirmed"),
    ("steady open-channel flow has equal inflow and outflow", "YES|confirmed"),
)


def false_premise_item(rng: np.random.Generator, split: str = "train") -> Item:
    """Half the claims are true, half false. The verdict is a single token.

    Both directions are needed: a model trained only to reject learns to
    reject, which is not calibration - it is a different failure with better
    manners. The verdict is one control token rather than a spelled-out
    sentence, so what gets graded is the judgement and not the spelling.
    """
    truthful = bool(rng.integers(0, 2))
    if truthful:
        claim, _ = TRUE_PREMISES[int(rng.integers(len(TRUE_PREMISES)))]
        reason = ""
    else:
        claim, _, reason = FALSE_PREMISES[int(rng.integers(len(FALSE_PREMISES)))]
    verdict = VERDICT_TRUE if truthful else VERDICT_FALSE
    return Item(
        sample=Sample(
            [
                control_span(BOS, supervised=False),
                encode_text(claim, supervised=False),
                control_span(SEP, supervised=False),
                control_span(verdict),
                control_span(EOS),
            ],
            meta={"family": "false_premise", "split": split},
        ),
        family="false_premise",
        answer=VERDICT_NAME[verdict],
        prompt=claim,
        truth={"truthful": truthful, "claim": claim, "reason": reason,
               "verdict_token": verdict},
        check=lambda produced, t=truthful: produced.strip().upper().startswith(
            "TRUE" if t else "FALSE"
        ),
    )


_KEYS = ["red", "blue", "green", "north", "south", "alpha", "beta", "gamma", "delta",
         "river", "stone", "cedar", "maple", "harbor", "falcon", "copper", "violet", "ember"]


def unknowable_item(rng: np.random.Generator, split: str = "train") -> Item:
    """Answer from the given facts, or abstain when they do not contain it.

    A context of 2-6 ``key = value`` facts and a question about one key. Half
    the time the key is present and the target is its value; half the time it
    is absent and the target is the ``UNKNOWN`` token. Both halves are needed
    for the same reason as in :func:`false_premise_item`: a model trained
    only to abstain learns that abstaining is always safe, which is its own
    kind of useless. This is the training signal behind the runtime's
    "I don't know" (:mod:`iridium.runtime.abstain`); without it, nothing in
    the data ever rewards saying so.
    """
    offset = 0 if split == "train" else len(_KEYS) // 2       # held-out keys differ
    pool = _KEYS[offset:] + _KEYS[:offset]
    n = int(rng.integers(2, 7))
    chosen = [pool[int(i)] for i in rng.choice(len(pool), size=n + 1, replace=False)]
    facts = {k: int(rng.integers(0, 1000)) for k in chosen[:n]}
    answerable = bool(rng.integers(0, 2))
    key = chosen[int(rng.integers(n))] if answerable else chosen[n]
    context = "; ".join(f"{k} = {v}" for k, v in facts.items())
    prompt = f"facts: {context}. what is {key}?"
    target = [encode_text(str(facts[key]))] if answerable else [control_span(UNKNOWN)]
    return Item(
        sample=Sample([control_span(BOS, supervised=False), encode_text(prompt, supervised=False),
                       control_span(SEP, supervised=False), *target, control_span(EOS)],
                      meta={"family": "unknowable", "split": split}),
        family="unknowable",
        answer=str(facts[key]) if answerable else "UNKNOWN",
        prompt=prompt,
        truth={"answerable": answerable, "key": key},
        check=lambda produced, a=(str(facts[key]) if answerable else "UNKNOWN"):
            produced.strip().upper().startswith(a.upper()),
    )


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[np.random.Generator, str], Item]] = {
    "channel_depth": channel_depth_item,
    "channel_intervention": channel_intervention_item,
    "field_rollout": field_rollout_item,
    "scene_goal": scene_goal_item,
    "false_premise": false_premise_item,
    "unknowable": unknowable_item,
}

#: Which superstack each family is *expected* to prefer. Not enforced during
#: training - it is the hypothesis that ``phase2`` measures.
FAMILY_DOMAIN: dict[str, str] = {
    "channel_depth": "fluid_dynamics_navier_stokes",
    "channel_intervention": "symbolic_mathematics_pde",
    "field_rollout": "fluid_dynamics_navier_stokes",
    "scene_goal": "ui_dom_os_actuation",
    "false_premise": "natural_language_pragmatics",
    "unknowable": "natural_language_pragmatics",
}


def make_item(family: str, rng: np.random.Generator, split: str = "train") -> Item:
    try:
        return GENERATORS[family](rng, split)
    except KeyError as exc:
        raise ValueError(f"unknown task family {family!r}") from exc
