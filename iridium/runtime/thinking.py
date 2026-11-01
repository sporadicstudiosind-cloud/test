"""How hard to think: per-token ponder depth from difficulty and a speed target.

Iridium's control core loops (the ponder loop), and a learned halting head
says, per token, whether another pass is worth it. ``Iridium1.forward`` with
``halt_threshold`` stops a step as soon as every token is confident, so depth
already varies with difficulty. This module decides the threshold and the loop
cap for each step from three signals:

* **Effort** -- the caller's standing request: ``instant`` (one loop, no
  pondering), ``fast``, ``balanced``, ``deep``, ``max`` (every loop, always).
* **Speed requirement** -- an optional per-token latency target. A simple
  feedback controller lowers the threshold (think less) while steps run over
  the target and raises it back toward the effort level while they run under.
* **Loops** -- optionally, a user-chosen base depth. The model then picks its
  own depth within ``loops +- spread`` (default +-10) from its uncertainty:
  confident steps think less than the base, uncertain ones more, clamped to
  ``[1, loop_ceiling]`` (the trained ``max_loops`` plus inference headroom).
* **Difficulty** -- the model's own uncertainty about what comes next. High
  entropy in the previous step's next-token distribution raises the threshold
  for the next step (demand more confidence before halting); a near-certain
  prediction lowers it.

The halting head is only as good as its training (the ponder KL and the
subject losses in ``Iridium1.losses``); untrained, it halts arbitrarily. The
mechanics -- depth that varies per token, bounded by effort and latency -- do
not depend on that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

__all__ = ["ThinkingBudget", "EFFORT_LEVELS"]

#: effort -> (fraction of max loops allowed, base halting threshold).
#: ``None`` threshold means never halt early (run the whole cap).
EFFORT_LEVELS = {
    "instant": (0.0, None),
    "fast": (1.0, 0.55),
    "balanced": (1.0, 0.8),
    "deep": (1.0, 0.95),
    "max": (1.0, None),
}


@dataclass
class ThinkingBudget:
    effort: str = "balanced"
    #: User-chosen base loop count; ``None`` derives depth from ``effort``.
    loops: Optional[int] = None
    #: How far the model may move from ``loops`` on its own, each way.
    spread: int = 10
    latency_ms: Optional[float] = None
    #: How strongly next-token entropy moves the threshold (0 disables).
    difficulty_gain: float = 0.15
    #: Feedback step per token while over or under the latency target.
    latency_step: float = 0.05
    floor: float = 0.5
    ceiling: float = 0.99
    _latency_shift: float = 0.0
    _difficulty_shift: float = 0.0
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {sorted(EFFORT_LEVELS)}")
        if self.loops is not None and self.loops < 1:
            raise ValueError("loops must be >= 1")
        if self.spread < 0:
            raise ValueError("spread must be >= 0")

    def cap(self, max_loops: int, ceiling: Optional[int] = None) -> int:
        """Most loops any step of this budget can ask for (prefill depth)."""
        if self.loops is None:
            return self.plan(max_loops)[0]
        return max(1, min(ceiling or max_loops, self.loops + self.spread))

    def plan(self, max_loops: int, ceiling: Optional[int] = None) -> tuple[int, Optional[float]]:
        """``(n_loops, halt_threshold)`` for the next step."""
        frac, base = EFFORT_LEVELS[self.effort]
        if self.loops is not None:
            # difficulty in [-1, 1]: entropy above/below half-uniform, minus
            # any latency pressure. The model's own call moves depth +-spread.
            signal = self._difficulty_shift / self.difficulty_gain if self.difficulty_gain else 0.0
            signal = max(-1.0, min(1.0, signal + 2 * self._latency_shift))
            n_loops = self.loops + round(self.spread * signal)
            n_loops = max(1, min(ceiling or max_loops, n_loops))
        else:
            n_loops = max(1, round(1 + frac * (max_loops - 1)))
        if base is None or n_loops == 1:
            return n_loops, None
        thr = base + self._difficulty_shift + self._latency_shift
        return n_loops, min(self.ceiling, max(self.floor, thr))

    def observe(self, logits=None, elapsed_ms: Optional[float] = None,
                loops_used: Optional[int] = None) -> None:
        """Update from the step just taken."""
        if logits is not None and self.difficulty_gain:
            import torch
            p = torch.softmax(logits.float(), dim=-1)
            entropy = float(-(p * p.clamp_min(1e-12).log()).sum())
            normalised = entropy / math.log(p.numel())            # 0 certain .. 1 uniform
            self._difficulty_shift = self.difficulty_gain * (normalised - 0.5) * 2
        if self.latency_ms is not None and elapsed_ms is not None:
            if elapsed_ms > self.latency_ms:
                self._latency_shift = max(self._latency_shift - self.latency_step, -0.5)
            else:
                self._latency_shift = min(self._latency_shift + self.latency_step / 2, 0.0)
        self.history.append({"loops": loops_used, "ms": elapsed_ms,
                             "shift": round(self._difficulty_shift + self._latency_shift, 3)})
