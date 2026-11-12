"""The persistent instance: one resident model, many isolated streams.

This is the runtime the originating vision asked for — a single always-running
model that many conversations arrive at concurrently, where the model itself
influences how much computation each one receives, and where nothing leaks
between them.

    admit -> enqueue work -> scheduler allocates a token budget per stream
          -> the model runs each stream against its own cache
          -> emissions stream back, evidence-labelled

Concurrency here is cooperative and single-threaded on purpose. The thing worth
demonstrating at this scale is the *allocation policy* and the *isolation*, and
both are easier to hold to a bit-exact standard without a thread pool in the
way. The distributed execution model lives in ``iridium/parallel/``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch

from ..codecs.bank import TensorBatch, continuous_dims
from ..codecs.spans import Sample, collate
from .decode import atomic_chunks, slice_batch
from .registry import StreamRecord, StreamRegistry
from .scheduler import Allocation, FocusScheduler, SchedulerPolicy


@dataclass
class Emission:
    stream_id: str
    tokens: int
    hidden: torch.Tensor
    focus: float
    expected_loops: float
    evidence: str = "predicted"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TickReport:
    tick: int
    allocation: Allocation
    emissions: list[Emission]
    wall_seconds: float
    total_tokens: int

    def summary(self) -> str:
        parts = ", ".join(
            f"{e.stream_id}:{e.tokens}t/f{e.focus:.2f}" for e in self.emissions
        )
        return (
            f"tick {self.tick:3d} | {self.total_tokens:4d} tokens | "
            f"{self.wall_seconds * 1e3:6.1f} ms | {parts}"
        )


class IridiumService:
    def __init__(
        self,
        model,
        policy: Optional[SchedulerPolicy] = None,
        registry: Optional[StreamRegistry] = None,
        n_loops: int = 1,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.registry = registry or StreamRegistry()
        self.scheduler = FocusScheduler(policy)
        self.n_loops = n_loops
        self.device = device
        self.dims = continuous_dims(model.cfg.codecs)
        self._pending: dict[str, TensorBatch] = {}
        self._cursor: dict[str, int] = {}
        self.tick_count = 0

    # -- admission --------------------------------------------------------

    def admit(self, stream_id: str, owner: str, sample: Sample, **meta: Any) -> StreamRecord:
        record = self.registry.open(stream_id, owner, **meta)
        batch = TensorBatch(collate([sample], self.dims), device=self.device)
        self._pending[stream_id] = batch
        self._cursor[stream_id] = 0
        return record

    def append(self, stream_id: str, sample: Sample, owner: Optional[str] = None) -> None:
        """Add more input to a live stream, continuing its positions."""
        record = self.registry.get(stream_id, owner)
        batch = TensorBatch(collate([sample], self.dims), device=self.device)
        batch.positions = batch.positions + record.n_tokens
        self._pending[stream_id] = batch
        self._cursor[stream_id] = 0

    def close(self, stream_id: str) -> None:
        self.registry.close(stream_id)
        self._pending.pop(stream_id, None)
        self._cursor.pop(stream_id, None)

    def waiting(self) -> list[StreamRecord]:
        return [
            r for r in self.registry
            if self._remaining(r.stream_id) > 0
        ]

    def _remaining(self, stream_id: str) -> int:
        batch = self._pending.get(stream_id)
        if batch is None:
            return 0
        return int(batch.modality.shape[1]) - self._cursor.get(stream_id, 0)

    # -- execution --------------------------------------------------------

    @torch.no_grad()
    def tick(self) -> TickReport:
        started = time.monotonic()
        pending = self.waiting()
        allocation = self.scheduler.allocate(pending)
        emissions: list[Emission] = []

        for record in pending:
            budget = allocation.per_stream.get(record.stream_id, 0)
            if budget <= 0:
                continue
            emission = self._advance(record, budget)
            if emission is not None:
                emissions.append(emission)
                self.scheduler.observe_focus(
                    record, emission.focus, emission.expected_loops
                )

        self.tick_count += 1
        return TickReport(
            tick=self.tick_count,
            allocation=allocation,
            emissions=emissions,
            wall_seconds=time.monotonic() - started,
            total_tokens=sum(e.tokens for e in emissions),
        )

    def _advance(self, record: StreamRecord, budget: int) -> Optional[Emission]:
        batch = self._pending[record.stream_id]
        start = self._cursor[record.stream_id]
        total = int(batch.modality.shape[1])
        if start >= total:
            return None

        # Chunk boundaries must not split an atomic span, or the spectral
        # blocks silently do not fire. The budget is a target, not a mandate.
        bounds = atomic_chunks(batch, budget)
        end = start
        for lo, hi in bounds:
            if lo >= start:
                end = hi
                break
        end = min(max(end, start + 1), total)

        chunk = slice_batch(batch, start, end)
        out = self.model(chunk, n_loops=self.n_loops, cache=record.cache)
        n = end - start
        self._cursor[record.stream_id] = end
        record.n_tokens += n
        record.touch()

        focus = float(out.decisions[0].focus.mean())
        loops = float(out.expected_loops.mean())
        record.emitted.append(out.hidden[:, -1].detach())
        return Emission(
            stream_id=record.stream_id,
            tokens=n,
            hidden=out.hidden,
            focus=focus,
            expected_loops=loops,
            meta={
                "stacks": out.stats["stack_stats"][0]["per_stack_tokens"],
                "cache_bytes": record.cache_bytes(),
            },
        )

    def run(self, max_ticks: int = 100) -> list[TickReport]:
        reports = []
        for _ in range(max_ticks):
            if not self.waiting():
                break
            reports.append(self.tick())
        return reports

    # -- introspection ----------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "streams": len(self.registry),
            "ticks": self.tick_count,
            "cache_bytes": self.registry.total_cache_bytes(),
            "isolation": self.registry.aliasing_report()["isolated"],
            "waiting": [r.stream_id for r in self.waiting()],
        }
