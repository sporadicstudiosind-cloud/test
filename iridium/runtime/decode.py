"""Incremental decoding, and the parity contract it has to satisfy.

The rule this module exists to enforce: **cached incremental decoding must
compute the same function as an uncached forward pass over the whole
sequence.** Not approximately — to floating-point reproduction in float64.

That is a strong constraint on this architecture specifically, because three
separate mechanisms could break it and each one fails silently:

* the superstacks keep their own sparse KV, so a token's stack-local history
  must be the same set whether it was gathered by a mask or accumulated by a
  cache;
* the bridge reads core stage-I states for the whole stream, so the cache has
  to carry them forward;
* the ponder loop indexes the cache, so a layer run twice must not alias.

``tests/integration/test_kv_parity.py`` runs the comparison. If it fails, the
serving path is computing something other than what was trained, and no
downstream metric means anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from ..codecs.bank import TensorBatch
from ..codecs.spans import Batch


def empty_cache() -> dict:
    return {}


def slice_batch(batch: TensorBatch, start: int, end: int) -> TensorBatch:
    """A view of ``[start, end)`` that keeps absolute positions intact."""
    view = object.__new__(TensorBatch)
    view.modality = batch.modality[:, start:end]
    view.discrete = batch.discrete[:, start:end]
    view.scalars = batch.scalars[:, start:end]
    view.positions = batch.positions[:, start:end]
    view.valid = batch.valid[:, start:end]
    view.supervised = batch.supervised[:, start:end]
    view.span_id = batch.span_id[:, start:end]
    view.continuous = {k: v[:, start:end] for k, v in batch.continuous.items()}
    # A grid is only usable by a chunk that contains all of it.
    view.grids = [
        (b, s - start, shape)
        for (b, s, shape) in batch.grids
        if s >= start and s + int(np.prod(shape)) <= end
    ]
    coords = getattr(batch, 'media_coordinates', None)
    view.media_coordinates = coords[:, start:end] if coords is not None else None
    cv = getattr(batch, 'coordinate_valid', None)
    view.coordinate_valid = cv[:, start:end] if cv is not None else None
    rp = getattr(batch, 'rope_positions', None)
    view.rope_positions = rp[:, start:end] if rp is not None else None
    view.meta = batch.meta
    return view


@torch.no_grad()
def run_chunked(
    model,
    batch: TensorBatch,
    chunk: int = 1,
    n_loops: int = 1,
    cache: Optional[dict] = None,
) -> torch.Tensor:
    """Feed the sequence in chunks through the cache; return all hidden states.

    ``chunk=1`` is token-at-a-time decoding. ``chunk=T`` is a single prefill.
    Both must agree with an uncached pass.
    """
    cache = empty_cache() if cache is None else cache
    total = batch.modality.shape[1]
    outputs = []
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        out = model(slice_batch(batch, start, end), n_loops=n_loops, cache=cache)
        outputs.append(out.hidden)
    return torch.cat(outputs, dim=1)


@dataclass
class StreamState:
    """One isolated stream's decoding state.

    The cache is *per stream*. Nothing in this object is shared with another
    stream, and ``iridium/runtime/registry.py`` is responsible for making that
    structural rather than a convention.
    """

    stream_id: str
    cache: dict = field(default_factory=empty_cache)
    n_tokens: int = 0
    emitted: list = field(default_factory=list)

    def reset(self) -> None:
        self.cache.clear()
        self.n_tokens = 0
        self.emitted.clear()


def atomic_chunks(batch: TensorBatch, chunk: int) -> list[tuple[int, int]]:
    """Chunk boundaries that never split an atomic span.

    A spectral block mixes every patch of a field grid, so it can only run when
    the whole grid is present in the chunk being processed. Splitting a field
    across two prefill chunks does not produce a slightly worse answer, it
    produces a *different function*: the operator silently does not fire. This
    is a genuine constraint on the serving path and is stated here rather than
    discovered later as a quality regression.

    Span-coherent routing (``Span.atomic``) keeps a field together across
    *stacks*; this keeps it together across *chunks*. Both are needed.
    """
    total = batch.modality.shape[1]
    forbidden: set[int] = set()
    for b, start, shape in batch.grids:
        n = int(np.prod(shape))
        forbidden.update(range(start + 1, start + n))
    span = batch.span_id
    for i in range(span.shape[0]):
        ids = span[i]
        for sid in torch.unique(ids[ids >= 0]).tolist():
            where = torch.nonzero(ids == sid).flatten()
            forbidden.update(range(int(where[0]) + 1, int(where[-1]) + 1))

    bounds: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(start + chunk, total)
        while end < total and end in forbidden:
            end += 1
        bounds.append((start, end))
        start = end
    return bounds


@torch.no_grad()
def run_atomic_chunked(
    model, batch: TensorBatch, chunk: int = 1, n_loops: int = 1,
    cache: Optional[dict] = None,
) -> torch.Tensor:
    cache = empty_cache() if cache is None else cache
    outputs = []
    for start, end in atomic_chunks(batch, chunk):
        out = model(slice_batch(batch, start, end), n_loops=n_loops, cache=cache)
        outputs.append(out.hidden)
    return torch.cat(outputs, dim=1)
