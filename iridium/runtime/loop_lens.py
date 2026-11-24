"""Read the ponder loop: what the model would say after each loop, and why.

Iridium thinks silently: each ponder loop reruns the control core and the
routed superstacks on the same token. This module makes that legible without
touching the model. It runs one full forward (every loop, no early halting,
no cache) and, per position and per loop, reports:

* the top next-token guesses, read with the model's own final norm and tied
  text head (a "logit lens": ``per_loop_hidden`` is already finalized, so no
  new parameters are involved);
* the probability that the halting head stops at that loop;
* which superstacks the router sent the token to, and with what weight;
* the loop at which the top guess settled for good.

Read-only by construction: no parameter, buffer or checkpoint format changes,
so every existing checkpoint works and training is unaffected.

Limits worth keeping in mind. A lens shows what a state *predicts*, not why;
a loop can do real work (sharpening confidence, preparing later tokens) that
never changes the top guess. Early-loop readings can also be miscalibrated,
because a tuned lens (a small learned per-loop translator) is not applied
here. Never train on these readouts: a model rewarded for what the lens shows
learns to perform for the lens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from ..codecs.bank import TensorBatch, continuous_dims
from ..codecs.spans import Sample, collate
from .device import device_of

__all__ = ["LoopReading", "TokenReading", "read_loops", "format_readings"]


@dataclass
class LoopReading:
    loop: int                                   # 1-based
    top: list[tuple[str, float]]                # (token text, probability)
    stop_p: float                               # P(halting stops at this loop)
    stacks: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class TokenReading:
    position: int
    token: str                                  # the input token at this position
    actual_next: Optional[str]                  # the token that follows, if any
    loops: list[LoopReading]
    settled_at: int                             # first loop whose top guess never changes after


def _stack_names(model) -> list[str]:
    stacks = model.cfg.stacks
    names = list(getattr(stacks, "specializations", None) or [])
    return names if len(names) == stacks.n_stacks else [f"stack{i}" for i in range(stacks.n_stacks)]


def _decoder(tokenizer, text_offset: int):
    def decode(token_id: int) -> str:
        if token_id < text_offset:
            return f"<ctl{token_id}>"
        raw = token_id - text_offset
        if tokenizer is not None:
            return tokenizer.decode([raw])
        return bytes([raw]).decode("utf-8", "replace") if raw < 256 else f"<{raw}>"
    return decode


@torch.no_grad()
def read_loops(model, sample: Sample, n_loops: Optional[int] = None, top_k: int = 3,
               tokenizer=None, text_offset: int = 16) -> list[TokenReading]:
    """Per-position, per-loop readout of one full forward over ``sample``."""
    if n_loops is None:
        n_loops = model.cfg.router.max_loops
    if not 1 <= n_loops <= model.cfg.router.loop_ceiling:
        raise ValueError(f"n_loops must be in [1, {model.cfg.router.loop_ceiling}]")
    was_training = model.training
    model.eval()
    try:
        batch = TensorBatch(collate([sample], continuous_dims(model.cfg.codecs)),
                            device=device_of(model))
        out = model(batch, n_loops=n_loops)
    finally:
        model.train(was_training)

    decode = _decoder(tokenizer, text_offset)
    names = _stack_names(model)
    ids = batch.discrete[0].tolist()
    valid = batch.valid[0].tolist()
    probs = [torch.softmax(model.codecs.text_head(h[0]).float(), -1) for h in out.per_loop_hidden]
    tops = [p.topk(min(top_k, p.shape[-1]), dim=-1) for p in probs]
    stop = out.loop_stopping[0].float()                      # [T, loops]
    routes = [(d.stack_index[0], d.stack_weight[0].float(),
               None if d.valid is None else d.valid[0]) for d in out.decisions]

    readings: list[TokenReading] = []
    for t, ok in enumerate(valid):
        if not ok:
            continue
        loops = []
        for L, top in enumerate(tops):
            stacks: list[tuple[str, float]] = []
            if L < len(routes):
                index, weight, used = routes[L]
                if used is None or bool(used[t]):
                    stacks = [(names[int(i)], float(w)) for i, w in zip(index[t], weight[t])
                              if float(w) > 0]
            loops.append(LoopReading(
                loop=L + 1,
                top=[(decode(int(i)), float(v)) for v, i in zip(top.values[t], top.indices[t])],
                stop_p=float(stop[t, L]) if L < stop.shape[-1] else 0.0,
                stacks=stacks,
            ))
        picks = [int(top.indices[t, 0]) for top in tops]
        settled = len(picks)
        while settled > 1 and picks[settled - 2] == picks[-1]:
            settled -= 1
        token = decode(int(ids[t]))
        nxt = decode(int(ids[t + 1])) if t + 1 < len(ids) and valid[t + 1] else None
        readings.append(TokenReading(t, token, nxt, loops, settled))
    return readings


def format_readings(readings: list[TokenReading]) -> str:
    """A compact human-readable table, one block per position."""
    lines = []
    for r in readings:
        head = f"[{r.position}] {r.token!r}"
        if r.actual_next is not None:
            head += f" -> next {r.actual_next!r}"
        lines.append(f"{head}   (settled at loop {r.settled_at})")
        for lp in r.loops:
            guesses = ", ".join(f"{tok!r} {p:.2f}" for tok, p in lp.top)
            route = " + ".join(f"{n} {w:.2f}" for n, w in lp.stacks) or "core only"
            lines.append(f"    loop {lp.loop}: {guesses}   stop {lp.stop_p:.2f}   via {route}")
    return "\n".join(lines)
