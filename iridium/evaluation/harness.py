"""Grading: what fraction of held-out items the model actually gets right.

Loss is not accuracy and accuracy is not competence, but graded accuracy on
items whose answers come from an independent computation is the strongest of
the three and the only one worth quoting. Every number this module returns is
produced by free-running generation and checked against the analytic law, the
deterministic solver or the environment's goal predicate.

Baselines are reported alongside every score, because a number with no baseline
cannot be interpreted. For the numeric families the baseline is the corpus
median answer — the best a model can do by ignoring the input entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

from ..codecs.bank import TensorBatch, continuous_dims
from ..codecs.spans import MODALITY_INDEX, Sample, collate
from ..runtime.generate import generate
from ..training.tasks import Item


@dataclass
class FamilyResult:
    family: str
    n: int
    correct: int
    baseline_correct: int = 0
    extra: dict[str, float] = field(default_factory=dict)
    examples: list[dict[str, str]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def baseline(self) -> float:
        return self.baseline_correct / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family, "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "baseline": round(self.baseline, 4),
            **{k: round(v, 6) for k, v in self.extra.items()},
        }


def prompt_only(item: Item) -> Sample:
    """Drop every supervised span: what the model is actually given."""
    spans = [s for s in item.sample.spans if not s.supervised]
    return Sample(spans or item.sample.spans[:1], meta=item.sample.meta)


@torch.no_grad()
def grade_text_family(
    model, items: Sequence[Item], max_new_tokens: int = 16,
    n_loops: int = 1, keep_examples: int = 3,
) -> FamilyResult:
    if not items:
        return FamilyResult(family="empty", n=0, correct=0)
    family = items[0].family
    result = FamilyResult(family=family, n=len(items), correct=0)

    median_answer = _median_answer(items)
    for i, item in enumerate(items):
        out = generate(model, prompt_only(item), max_new_tokens=max_new_tokens,
                       n_loops=n_loops)
        ok = item.grade(out.text)
        result.correct += int(ok)
        if median_answer is not None:
            result.baseline_correct += int(item.grade(median_answer))
        if i < keep_examples:
            result.examples.append(
                {"prompt": item.prompt, "target": item.answer,
                 "produced": out.text, "correct": str(ok)}
            )
    return result


def _median_answer(items: Sequence[Item]) -> Optional[str]:
    values = []
    for item in items:
        try:
            values.append(float(item.answer.split("|")[0]))
        except (ValueError, IndexError):
            return None
    if not values:
        return None
    return f"{float(np.median(values)):.4f}"


@torch.no_grad()
def grade_field_family(
    model, items: Sequence[Item], n_loops: int = 1, flow_steps: int = 16
) -> FamilyResult:
    """NRMSE of the emitted next frame against the solver's frame.

    The persistence baseline — emit the input frame unchanged — is the number
    that matters. A field model that cannot beat persistence has learned
    nothing about the dynamics, only about the field's marginal statistics.
    """
    if not items:
        return FamilyResult(family="field_rollout", n=0, correct=0)
    dims = continuous_dims(model.cfg.codecs)
    errors, persistence = [], []
    for item in items:
        sample = prompt_only(item)
        batch = TensorBatch(collate([sample], dims))
        from ..runtime.decode import atomic_chunks, slice_batch

        cache: dict = {}
        hidden = None
        for lo, hi in atomic_chunks(batch, int(batch.modality.shape[1])):
            out = model(slice_batch(batch, lo, hi), n_loops=n_loops, cache=cache)
            hidden = out.hidden
        target = np.asarray(item.truth["target"], dtype=np.float64)
        n_patch = target.shape[0]
        pred = model.codecs.decode_continuous(
            hidden[:, -1:], "field", steps=flow_steps
        )[0, 0].detach().cpu().numpy().astype(np.float64)
        # One emitted patch is compared against the mean target patch: the
        # rollout head emits patch-by-patch, and grading the first emission
        # keeps the comparison honest about what was actually produced.
        tgt = target.mean(axis=0)
        scale = float(np.sqrt(np.mean(tgt ** 2))) + 1e-12
        errors.append(float(np.sqrt(np.mean((pred - tgt) ** 2))) / scale)
        source = np.asarray(
            [s.payload for s in item.sample.spans if s.modality == "field"][0],
            dtype=np.float64,
        ).mean(axis=0)
        persistence.append(float(np.sqrt(np.mean((source - tgt) ** 2))) / scale)
    result = FamilyResult(family="field_rollout", n=len(items), correct=0)
    result.extra = {
        "nrmse": float(np.mean(errors)),
        "nrmse_persistence_baseline": float(np.mean(persistence)),
        "beats_persistence": float(np.mean(np.array(errors) < np.array(persistence))),
    }
    result.correct = int(np.sum(np.array(errors) < np.array(persistence)))
    return result


def evaluate(
    model, corpus, max_per_family: int = 40, n_loops: int = 1,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    model.eval()
    results: dict[str, Any] = {}
    for family, items in corpus.by_family().items():
        subset = items[:max_per_family]
        if family == "field_rollout":
            results[family] = grade_field_family(model, subset, n_loops).as_dict()
        elif family == "scene_goal":
            results[family] = _grade_scene(model, subset, n_loops).as_dict()
        else:
            results[family] = grade_text_family(
                model, subset, max_new_tokens, n_loops
            ).as_dict()
    return results


@torch.no_grad()
def _grade_scene(model, items, n_loops: int = 1) -> FamilyResult:
    """Opcode accuracy on the first emitted action, plus goal satisfaction."""
    from ..agency.actions import Action, Op
    from ..agency.scene import SceneEditor

    result = FamilyResult(family="scene_goal", n=len(items), correct=0)
    dims = continuous_dims(model.cfg.codecs)
    op_hits = 0
    for item in items:
        batch = TensorBatch(collate([prompt_only(item)], dims))
        from ..runtime.decode import atomic_chunks, slice_batch

        cache: dict = {}
        hidden = None
        for lo, hi in atomic_chunks(batch, int(batch.modality.shape[1])):
            hidden = model(slice_batch(batch, lo, hi), n_loops=n_loops, cache=cache).hidden
        op_logits, scalars = model.codecs.action_head(hidden[:, -1:])
        op = int(op_logits[0, 0].argmax())
        want = item.truth["actions"][0]
        op_hits += int(op == int(want.op))
        try:
            produced = Action(Op(op), tuple(float(v) for v in scalars[0, 0][:4]))
        except (ValueError, KeyError):
            continue
        editor = SceneEditor(resolution=8)
        editor.reset()
        editor.run([produced])
        result.correct += int(item.truth["goal"].satisfied(editor.state))
    result.extra = {"first_opcode_accuracy": op_hits / max(len(items), 1)}
    return result
