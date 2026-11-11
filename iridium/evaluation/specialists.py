"""Is each superstack the best place for its own subject?

The design claim is that a token is sent to a stack because that stack does the
job better than the control core could alone and better than any other stack.
This measures exactly that. For each subject's held-out text, the loss is taken
with routing overridden four ways:

* ``routed`` -- the model as it normally runs;
* ``core``   -- no stack at all, the control core alone;
* one stack alone, for every stack.

A subject **passes** when its own stack alone beats every other stack alone and
beats the core alone. ``margin`` is how much worse the best rival is, in loss
per token; a specialist that wins by 0.01 is a specialist in name only.

The core is never trained to work alone, so it losing is expected and proves
little on its own. The comparison between stacks is the informative one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

__all__ = ["specialist_matrix", "held_out_batches", "describe"]


@torch.no_grad()
def specialist_matrix(model, batches: dict[str, list], autocast=None) -> dict:
    """``{subject: {route: loss}}`` plus a verdict per subject."""
    import contextlib

    names = list(model.cfg.stacks.specializations)
    was_training = model.training
    model.eval()
    report: dict = {"stacks": names, "subjects": {}}
    try:
        for subject, group in batches.items():
            if subject not in names or not group:
                continue
            row = {}
            for route in ["routed", "core", *names]:
                model.route_override = (None if route == "routed" else
                                        "core" if route == "core" else names.index(route))
                total = 0.0
                for batch in group:
                    with (autocast() if autocast else contextlib.nullcontext()):
                        total += float(model.losses(batch)[0]["text"])
                row[route] = round(total / len(group), 4)
            rivals = {r: v for r, v in row.items() if r not in ("routed", subject)}
            best_rival = min(rivals, key=rivals.get)
            row["best_rival"] = best_rival
            row["margin"] = round(rivals[best_rival] - row[subject], 4)
            row["passes"] = row["margin"] > 0
            report["subjects"][subject] = row
    finally:
        model.route_override = None
        model.train(was_training)
    report["all_pass"] = bool(report["subjects"]) and all(
        r["passes"] for r in report["subjects"].values())
    return report


def held_out_batches(shard_dir, cfg, device="cpu", batch_size: int = 8, window: int = 1024,
                     max_batches: int = 4, tokenizer=None) -> dict[str, list]:
    """Held-out batches grouped by subject, from prepared shards plus the
    generated families (which need no shard)."""
    from ..data.domains import GENERATED_SUBJECT, subject_of
    from ..data.shards import MixedCorpus, Shard
    from ..training.datasets import BatchLoader

    out: dict[str, list] = {}

    def add(family: str, corpus) -> None:
        subject = subject_of(family)
        if not subject:
            return
        loader = BatchLoader(corpus, cfg.codecs, batch_size, seed=0, device=device,
                             max_length=window)
        for i, (batch, _) in enumerate(loader.batches()):
            if i >= max_batches:
                break
            out.setdefault(subject, []).append(batch)

    shard_dir = Path(shard_dir) if shard_dir else None
    if shard_dir is not None and (shard_dir / "prepared.json").exists():
        held = json.loads((shard_dir / "prepared.json").read_text()).get("held_out") or {}
        for family in held:
            n = len(Shard(shard_dir / f"{family}-test"))
            add(family, MixedCorpus.build({family: n}, shard_dir, split="test"))
    for family in GENERATED_SUBJECT:
        if subject_of(family):
            add(family, MixedCorpus.build({family: batch_size * max_batches}, None, seed=7,
                                          split="test", tokenizer=tokenizer))
    return out


def describe(report: dict) -> str:
    names = report["stacks"]
    routes = ["routed", "core", *names]
    lines = ["loss per token (lower is better); * marks the subject's own stack",
             f"{'subject':10}" + "".join(f"{r:>10}" for r in routes) + "   verdict"]
    for subject, row in report["subjects"].items():
        cells = "".join(f"{row[r]:>9.3f}{'*' if r == subject else ' '}" for r in routes)
        verdict = (f"pass, beats {row['best_rival']} by {row['margin']:.3f}" if row["passes"]
                   else f"FAIL, {row['best_rival']} is better by {-row['margin']:.3f}")
        lines.append(f"{subject:10}{cells}   {verdict}")
    lines.append("every stack is the best at its own subject" if report["all_pass"]
                 else "not every stack is the best at its own subject")
    return "\n".join(lines)
