"""Multiple-choice benchmarks scored as cloze likelihoods: the signal small models give.

Below a few hundred million parameters, models score at chance when a benchmark
asks them to *name* the right letter ("A", "B", ...), but they still rank the
answers' likelihoods well above chance. OLMES (Gu et al. 2024, arXiv 2406.08446)
therefore evaluates small models in the cloze form: score
``log p(answer | question)`` for each choice and pick the highest. DataDecide
(arXiv 2504.11393) finds such likelihood metrics are what make small-scale runs
predict large-scale rankings.

Each choice is scored the way the model was trained to predict: each ponder
loop's cross-entropy weighted by its stopping probability (as in
``Iridium1.losses``). Three accuracies are reported:

* ``acc``: raw summed log-likelihood, which favours short answers.
* ``acc_norm``: log-likelihood per UTF-8 byte of the answer, OLMES's character
  normalisation. This is the headline number for HellaSwag, ARC and OpenBookQA.
* ``acc_uncond``: ``log p(answer | question) - log p(answer | "Answer:")``,
  the PMI-style score OLMES uses for ARC-Challenge and OpenBookQA, since it
  cancels how likely an answer is regardless of the question.

Tasks use the public Hugging Face copies (parquet, no loading scripts):
``hellaswag`` (Rowan/hellaswag, validation), ``arc_easy`` and ``arc_challenge``
(allenai/ai2_arc, test), ``openbookqa`` (allenai/openbookqa main, test) and
``piqa`` (baber/piqa, a parquet mirror of ybisk/piqa, validation). Chance is
25% except PIQA's 50%. For calibration, SmolLM2-135M reports HellaSwag ~42 and
ARC ~44 (average of Easy and Challenge), from 2T tokens; expect a 50M model on
5B tokens to sit well below that.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..codecs.bank import TensorBatch, continuous_dims
from ..codecs.spans import MODALITY_INDEX, Sample, Span, collate, text_span
from ..config import TEXT_ID_OFFSET
from ..runtime.device import device_of

BOS = 1


@dataclass
class ClozeItem:
    context: str
    choices: list[str]
    label: int
    #: Context for the unconditional score (OLMES uses "Answer:").
    uncond_context: str = "Answer:"


@dataclass
class TaskResult:
    task: str
    n: int
    chance: float
    acc: float
    acc_norm: float
    acc_uncond: float
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"task": self.task, "n": self.n, "chance": self.chance, "acc": round(self.acc, 4),
                "acc_norm": round(self.acc_norm, 4), "acc_uncond": round(self.acc_uncond, 4),
                **self.extra}


# -- loading --------------------------------------------------------------------

def _hellaswag_text(text: str) -> str:
    # The preprocessing lm-evaluation-harness applies to WikiHow-derived items.
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _choices(record) -> tuple[list[str], list[str]]:
    c = record["choices"]
    return list(c["text"]), list(c["label"])


def load_task(name: str, limit: Optional[int] = None) -> list[ClozeItem]:
    """The task's items, in dataset order (first ``limit``)."""
    from datasets import load_dataset

    items: list[ClozeItem] = []
    if name == "hellaswag":
        for r in load_dataset("Rowan/hellaswag", split="validation"):
            ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
            items.append(ClozeItem(_hellaswag_text(r["activity_label"] + ": " + ctx),
                                   [" " + _hellaswag_text(e) for e in r["endings"]],
                                   int(r["label"]), uncond_context=""))
            if limit and len(items) >= limit:
                break
    elif name in ("arc_easy", "arc_challenge"):
        cfg = "ARC-Easy" if name == "arc_easy" else "ARC-Challenge"
        for r in load_dataset("allenai/ai2_arc", cfg, split="test"):
            texts, labels = _choices(r)
            if r["answerKey"] not in labels:
                continue
            items.append(ClozeItem(f"Question: {r['question']}\nAnswer:", [" " + t for t in texts],
                                   labels.index(r["answerKey"])))
            if limit and len(items) >= limit:
                break
    elif name == "openbookqa":
        for r in load_dataset("allenai/openbookqa", "main", split="test"):
            texts, labels = _choices(r)
            items.append(ClozeItem(f"Question: {r['question_stem']}\nAnswer:", [" " + t for t in texts],
                                   labels.index(r["answerKey"])))
            if limit and len(items) >= limit:
                break
    elif name == "piqa":
        for r in load_dataset("baber/piqa", split="validation"):
            items.append(ClozeItem(f"Question: {r['goal']}\nAnswer:", [" " + r["sol1"], " " + r["sol2"]],
                                   int(r["label"])))
            if limit and len(items) >= limit:
                break
    else:
        raise ValueError(f"unknown task {name!r}; choose from {', '.join(TASKS)}")
    return items


TASKS = ("hellaswag", "arc_easy", "arc_challenge", "openbookqa", "piqa")
CHANCE = {"piqa": 0.5}


# -- scoring --------------------------------------------------------------------

def _sample(context: str, choice: str, tokenizer, max_len: int) -> Sample:
    answer = text_span(choice, supervised=True, offset=TEXT_ID_OFFSET, tokenizer=tokenizer)
    spans = [Span("control", np.array([BOS], np.int64), supervised=False)]
    if context:
        ctx = text_span(context, supervised=False, offset=TEXT_ID_OFFSET, tokenizer=tokenizer)
        room = max_len - 1 - len(answer)
        if room <= 0:
            raise ValueError("an answer longer than the model's context cannot be scored")
        if len(ctx) > room:
            # Drop the oldest context, never the answer being scored.
            ctx = Span("text", ctx.payload[-room:], supervised=False)
        spans.append(ctx)
    spans.append(answer)
    return Sample(spans)


@torch.no_grad()
def choice_logprobs(model, contexts: Sequence[str], choices: Sequence[str], tokenizer=None,
                    n_loops: int = 1) -> list[float]:
    """``log p(choice | context)`` for each pair, scored as training scores it."""
    dims = continuous_dims(model.cfg.codecs)
    max_len = model.cfg.max_seq_len
    samples = [_sample(c, x, tokenizer, max_len) for c, x in zip(contexts, choices)]
    batch = TensorBatch(collate(samples, dims, max_length=max_len), device=device_of(model))
    out = model(batch, n_loops=n_loops)
    target = batch.discrete[:, 1:]
    mask = batch.supervised[:, 1:] & batch.valid[:, 1:] & (batch.modality[:, 1:] == MODALITY_INDEX["text"])
    nll = torch.zeros_like(mask, dtype=torch.float32)
    for index, hidden in enumerate(out.per_loop_hidden):
        logits = model.codecs.text_head(hidden[:, :-1]).float()
        loop_nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   target.reshape(-1).clamp(0, logits.shape[-1] - 1),
                                   reduction="none").view_as(mask)
        nll = nll + out.loop_stopping[:, :-1, index].float() * loop_nll
    return (-(nll * mask).sum(dim=1)).tolist()


def evaluate_items(model, items: Iterable[ClozeItem], tokenizer=None, n_loops: int = 1,
                   task: str = "custom") -> TaskResult:
    model.eval()
    hit = hit_norm = hit_uncond = n = skipped = 0
    for item in items:
        k = len(item.choices)
        try:
            cond = choice_logprobs(model, [item.context] * k, item.choices, tokenizer, n_loops)
            uncond = choice_logprobs(model, [item.uncond_context] * k, item.choices, tokenizer, n_loops)
        except ValueError:
            skipped += 1          # an answer longer than the context window
            continue
        nbytes = [max(len(c.encode("utf-8")), 1) for c in item.choices]
        hit += int(np.argmax(cond)) == item.label
        hit_norm += int(np.argmax([lp / b for lp, b in zip(cond, nbytes)])) == item.label
        hit_uncond += int(np.argmax([c - u for c, u in zip(cond, uncond)])) == item.label
        n += 1
    scored = max(n, 1)
    return TaskResult(task, n, CHANCE.get(task, 0.25), hit / scored, hit_norm / scored,
                      hit_uncond / scored, extra={"skipped_too_long": skipped})


def run_cloze(model, tokenizer=None, tasks: Sequence[str] = TASKS, limit: Optional[int] = 500,
              n_loops: int = 1) -> list[TaskResult]:
    results = []
    for name in tasks:
        result = evaluate_items(model, load_task(name, limit), tokenizer, n_loops, task=name)
        skipped = result.extra.get("skipped_too_long", 0)
        print(f"{name:<14} n={result.n:<5} acc={result.acc:.3f}  acc_norm={result.acc_norm:.3f}  "
              f"acc_uncond={result.acc_uncond:.3f}  (chance {result.chance:.2f})"
              + (f"  [{skipped} skipped: too long]" if skipped else ""), flush=True)
        results.append(result)
    return results
