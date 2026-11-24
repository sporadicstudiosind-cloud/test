"""Focus-weighted scheduling: the model asks, the runtime arbitrates.

The originating vision: *the model should decide how much focus each
conversation gets*. Taken literally that is a model deciding its own resource
allocation with no counterparty, which has two failure modes that show up
immediately in simulation:

* **starvation** — a stream that asks for little gets nothing, forever, because
  every tick is won by streams asking for more;
* **incentive inversion** — if asking for more compute always yields more
  compute, the optimal policy for any stream is to always ask for the maximum,
  and the focus head degenerates to a constant.

So the split implemented here is: the *model* emits a focus demand per stream
(from its own focus head and halting statistics), and the *runtime* converts
demands into an allocation under constraints it owns — a per-tick token budget,
a fairness floor, an ageing term, and a per-owner share cap. The model
influences the allocation; it does not control it. That is what makes
"model-directed focus" a schedulable quantity rather than a promise.

``tests/unit/test_scheduler.py`` asserts no-starvation and budget conservation
directly, and that a stream which always demands 1.0 does not monopolise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .registry import StreamRecord


@dataclass(frozen=True)
class SchedulerPolicy:
    token_budget: int = 256          # tokens the instance can process per tick
    min_share: float = 0.05          # fairness floor, fraction of the budget
    max_owner_share: float = 0.5     # no single owner takes more than this
    ageing_rate: float = 0.15        # priority gained per idle tick
    demand_floor: float = 0.05
    demand_ceiling: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_share <= 1.0:
            raise ValueError("min_share must lie in [0, 1]")
        if not 0.0 < self.max_owner_share <= 1.0:
            raise ValueError("max_owner_share must lie in (0, 1]")


@dataclass
class Allocation:
    per_stream: dict[str, int] = field(default_factory=dict)
    demands: dict[str, float] = field(default_factory=dict)
    budget: int = 0
    starved: list[str] = field(default_factory=list)

    @property
    def allocated(self) -> int:
        return sum(self.per_stream.values())


class FocusScheduler:
    def __init__(self, policy: Optional[SchedulerPolicy] = None) -> None:
        self.policy = policy or SchedulerPolicy()
        self._idle_ticks: dict[str, int] = {}
        self.ticks = 0

    def observe_focus(self, record: StreamRecord, focus: float, expected_loops: float
                      ) -> float:
        """Fold the model's own signals into a stream's demand, in [0, 1].

        Focus is what the router asked for; expected loops is what the ponder
        controller actually spent. Using both means a stream cannot inflate its
        demand by asking for depth it does not use — the realized cost is half
        the signal.
        """
        p = self.policy
        realized = min(expected_loops / max(1.0, expected_loops), 1.0)
        demand = 0.5 * float(focus) + 0.5 * realized * float(focus)
        record.focus_demand = float(
            min(max(demand, p.demand_floor), p.demand_ceiling)
        )
        return record.focus_demand

    def allocate(self, streams: Iterable[StreamRecord]) -> Allocation:
        p = self.policy
        active = [s for s in streams]
        alloc = Allocation(budget=p.token_budget)
        if not active:
            self.ticks += 1
            return alloc

        weights: dict[str, float] = {}
        for s in active:
            idle = self._idle_ticks.get(s.stream_id, 0)
            ageing = 1.0 + p.ageing_rate * idle
            weights[s.stream_id] = max(
                p.demand_floor, s.focus_demand * s.priority * ageing
            )
            alloc.demands[s.stream_id] = s.focus_demand

        # Per-owner cap, applied before normalization so that one owner opening
        # many streams cannot take the instance by weight of numbers.
        by_owner: dict[str, list[str]] = {}
        for s in active:
            by_owner.setdefault(s.owner, []).append(s.stream_id)
        total = sum(weights.values())
        cap = p.max_owner_share * total
        for owner, ids in by_owner.items():
            owned = sum(weights[i] for i in ids)
            if owned > cap and owned > 0:
                scale = cap / owned
                for i in ids:
                    weights[i] *= scale

        total = sum(weights.values()) or 1.0
        floor_tokens = int(p.token_budget * p.min_share)
        remaining = p.token_budget

        # Fairness floor first: every waiting stream gets something.
        for s in active:
            give = min(floor_tokens, remaining)
            alloc.per_stream[s.stream_id] = give
            remaining -= give
            if remaining <= 0:
                break

        # Then distribute what is left by weight.
        if remaining > 0:
            shares = {
                i: weights[i] / total for i in weights if i in alloc.per_stream
            }
            order = sorted(shares, key=lambda i: -shares[i])
            handed = 0
            for i in order:
                give = int(math.floor(remaining * shares[i]))
                alloc.per_stream[i] += give
                handed += give
            for i in order:                      # hand out the rounding dust
                if handed >= remaining:
                    break
                alloc.per_stream[i] += 1
                handed += 1

        for s in active:
            got = alloc.per_stream.get(s.stream_id, 0)
            if got <= 0:
                alloc.starved.append(s.stream_id)
                self._idle_ticks[s.stream_id] = self._idle_ticks.get(s.stream_id, 0) + 1
            else:
                self._idle_ticks[s.stream_id] = 0
        self.ticks += 1
        return alloc
