"""Per-stream cache registry, with isolation that is structural.

Architecture §8 is explicit that a block-diagonal attention mask is *necessary
but not sufficient* for isolating concurrent work. This is the rest of it: the
cache, the bridge states, the router's running statistics and the emitted
tokens all live in a per-stream object that no other stream holds a reference
to. Nothing is shared and then masked; there is nothing to mask.

The originating vision asked for one always-resident model that serves every
request from a single instance and learns across all of them. Those are two
different things and only one of them is safe:

* **shared weights, isolated activations** — one resident parameter set serving
  many streams, with no state crossing between them. Implemented here.
* **shared activations** — attention flowing between concurrent users' streams.
  Rejected: it is a cross-tenant data leak by construction, and no amount of
  post-hoc filtering makes an attention head forget what it attended to.

Cross-stream learning happens offline, from privacy-filtered traces, in
``iridium/training/continual.py`` — not through the live residual stream.
``tests/integration/test_isolation.py`` asserts that a stream's outputs are
bit-identical whether or not other streams exist.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import torch


@dataclass
class StreamRecord:
    """Everything one stream owns. Nothing here is shared."""

    stream_id: str
    owner: str
    cache: dict = field(default_factory=dict)
    n_tokens: int = 0
    created_at: float = field(default_factory=time.monotonic)
    last_active: float = field(default_factory=time.monotonic)
    focus_demand: float = 0.5
    priority: float = 1.0
    emitted: list = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_active = time.monotonic()

    def cache_bytes(self) -> int:
        total = 0
        for value in self.cache.values():
            if isinstance(value, tuple):
                total += sum(
                    v.numel() * v.element_size()
                    for v in value if isinstance(v, torch.Tensor)
                )
            elif isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
        return total

    def tensor_ids(self) -> set[int]:
        """Identity of every tensor this stream holds, for the aliasing audit."""
        ids: set[int] = set()
        for value in self.cache.values():
            items = value if isinstance(value, tuple) else (value,)
            for v in items:
                if isinstance(v, torch.Tensor):
                    ids.add(v.data_ptr())
        return ids


class StreamRegistry:
    """Owns every live stream, and the only place a cache can be reached from."""

    def __init__(self, max_streams: int = 256, max_cache_bytes: int = 1 << 30) -> None:
        self.max_streams = max_streams
        self.max_cache_bytes = max_cache_bytes
        self._streams: dict[str, StreamRecord] = {}

    def __len__(self) -> int:
        return len(self._streams)

    def __contains__(self, stream_id: str) -> bool:
        return stream_id in self._streams

    def __iter__(self) -> Iterator[StreamRecord]:
        return iter(list(self._streams.values()))

    def open(self, stream_id: str, owner: str, **meta: Any) -> StreamRecord:
        if stream_id in self._streams:
            raise KeyError(f"stream {stream_id!r} is already open")
        if len(self._streams) >= self.max_streams:
            self._evict_oldest_idle()
        record = StreamRecord(stream_id=stream_id, owner=owner, meta=dict(meta))
        self._streams[stream_id] = record
        return record

    def get(self, stream_id: str, owner: Optional[str] = None) -> StreamRecord:
        record = self._streams.get(stream_id)
        if record is None:
            raise KeyError(f"no such stream: {stream_id!r}")
        if owner is not None and record.owner != owner:
            # Not "return empty": a wrong-owner read is a bug or an attack, and
            # either way the caller must not proceed as if it were a cache miss.
            raise PermissionError(
                f"stream {stream_id!r} belongs to another owner"
            )
        record.touch()
        return record

    def close(self, stream_id: str) -> None:
        record = self._streams.pop(stream_id, None)
        if record is not None:
            record.cache.clear()
            record.emitted.clear()

    def _evict_oldest_idle(self) -> None:
        if not self._streams:
            return
        victim = min(self._streams.values(), key=lambda r: r.last_active)
        self.close(victim.stream_id)

    # -- audits -----------------------------------------------------------

    def total_cache_bytes(self) -> int:
        return sum(r.cache_bytes() for r in self._streams.values())

    def aliasing_report(self) -> dict[str, Any]:
        """Any tensor storage reachable from two different streams.

        This is the check that would catch a cache pool handing the same buffer
        to two owners. An empty ``collisions`` list is the property; a non-empty
        one is a cross-tenant leak, not a performance note.
        """
        seen: dict[int, str] = {}
        collisions: list[tuple[str, str, int]] = []
        for record in self._streams.values():
            for ptr in record.tensor_ids():
                if ptr in seen and seen[ptr] != record.stream_id:
                    collisions.append((seen[ptr], record.stream_id, ptr))
                else:
                    seen[ptr] = record.stream_id
        return {
            "streams": len(self._streams),
            "distinct_storages": len(seen),
            "collisions": collisions,
            "isolated": not collisions,
        }

    def fingerprint(self, stream_id: str) -> str:
        """Stable digest of a stream's cache, for reproducibility checks."""
        record = self.get(stream_id)
        digest = hashlib.sha256()
        for key in sorted(record.cache, key=repr):
            value = record.cache[key]
            items = value if isinstance(value, tuple) else (value,)
            digest.update(repr(key).encode())
            for v in items:
                if isinstance(v, torch.Tensor):
                    digest.update(v.detach().cpu().numpy().tobytes())
                else:
                    digest.update(repr(v).encode())
        return digest.hexdigest()
