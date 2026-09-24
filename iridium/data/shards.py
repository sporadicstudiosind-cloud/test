"""Tokenized training data on disk, read by memory map.

The pre-1.0 corpus was a Python list of ``Item`` objects, rebuilt every round
by streaming and tokenizing inside the training process. At free-tier budgets
(hundreds of millions of tokens) that is gigabytes of Python objects, repeated
network streaming, and a training loop that waits on tokenization.

A shard is three flat files, written once by ``python -m iridium data prepare``
and never loaded whole:

* ``<name>.bin``  -- model ids (control ids + offset token ids), ``uint16``
  while the model vocabulary fits in 65,536 rows, else ``uint32``
* ``<name>.mask`` -- one byte per token, 1 where the token is a training
  target (all of prose; only the assistant's side of a conversation)
* ``<name>.idx``  -- ``int64`` start offsets of each item, plus the end

plus ``<name>.json`` describing it (family, token count, tokenizer, sources).

Training memory-maps the files; the OS pages in only the windows a batch
touches, so a 10 GB shard costs a few megabytes of RAM. An item is rebuilt as
an ordinary :class:`~iridium.codecs.spans.Sample` only when its batch is drawn,
so the model, collation and loss code are unchanged.

What goes in a shard: text, chat and tool conversations -- anything that is a
sequence of control and text ids. The synthetic families (fields, scenes,
actions) carry continuous payloads and are cheap to generate, so they are
generated on demand from a seeded index instead (see :class:`MixedCorpus`).
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from ..codecs.spans import Sample, Span
from ..config import TEXT_ID_OFFSET

__all__ = ["ShardWriter", "Shard", "MixedCorpus", "write_items"]

_SHARDABLE = ("control", "text")


class ShardWriter:
    """Append items to a shard without holding them in memory."""

    def __init__(self, prefix: str | Path, family: str, vocab_rows: int,
                 meta: Optional[dict] = None) -> None:
        self.prefix = Path(prefix)
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        self.dtype = np.uint16 if vocab_rows <= 1 << 16 else np.uint32
        self.family = family
        self.meta = dict(meta or {})
        self._bin = open(self.prefix.with_suffix(".bin"), "wb")
        self._mask = open(self.prefix.with_suffix(".mask"), "wb")
        self._offsets = [0]
        self._vocab_rows = vocab_rows

    def add(self, sample: Sample) -> None:
        ids, mask = [], []
        for span in sample.spans:
            if span.modality not in _SHARDABLE:
                raise ValueError(f"a {span.modality!r} span cannot be sharded; only "
                                 "control/text sequences are (synthetic families are "
                                 "generated on demand instead)")
            ids.append(np.asarray(span.payload, dtype=np.int64))
            mask.append(np.full(len(span.payload), bool(span.supervised), dtype=np.uint8))
        flat = np.concatenate(ids) if ids else np.zeros(0, np.int64)
        if flat.size and (flat.min() < 0 or flat.max() >= self._vocab_rows):
            raise ValueError("id outside the model vocabulary; wrong tokenizer for this config")
        self._bin.write(flat.astype(self.dtype).tobytes())
        self._mask.write((np.concatenate(mask) if mask else np.zeros(0, np.uint8)).tobytes())
        self._offsets.append(self._offsets[-1] + flat.size)

    def add_ids(self, ids: np.ndarray, mask: np.ndarray) -> None:
        """Append one item given directly as model ids and a target mask."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self._vocab_rows):
            raise ValueError("id outside the model vocabulary; wrong tokenizer for this config")
        self._bin.write(ids.astype(self.dtype).tobytes())
        self._mask.write(np.asarray(mask, dtype=np.uint8).tobytes())
        self._offsets.append(self._offsets[-1] + ids.size)

    def close(self) -> dict:
        self._bin.close()
        self._mask.close()
        np.asarray(self._offsets, dtype=np.int64).tofile(self.prefix.with_suffix(".idx"))
        meta = {**self.meta, "family": self.family, "items": len(self._offsets) - 1,
                "tokens": int(self._offsets[-1]), "dtype": np.dtype(self.dtype).name,
                "vocab_rows": self._vocab_rows, "format": 1}
        self.prefix.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return meta


def write_items(prefix, family: str, vocab_rows: int, samples: Iterable[Sample],
                meta: Optional[dict] = None) -> dict:
    writer = ShardWriter(prefix, family, vocab_rows, meta)
    for sample in samples:
        writer.add(sample)
    return writer.close()


class Shard:
    """Read-only, memory-mapped view of one shard."""

    def __init__(self, prefix: str | Path) -> None:
        self.prefix = Path(prefix)
        self.meta = json.loads(self.prefix.with_suffix(".json").read_text())
        self.family = self.meta["family"]
        dtype = np.dtype(self.meta["dtype"])
        n = self.meta["tokens"]
        # mode "r": pages come from the file on demand and are never copied
        # into process memory wholesale.
        self.ids = np.memmap(self.prefix.with_suffix(".bin"), dtype=dtype, mode="r",
                             shape=(n,)) if n else np.zeros(0, dtype)
        self.mask = np.memmap(self.prefix.with_suffix(".mask"), dtype=np.uint8, mode="r",
                              shape=(n,)) if n else np.zeros(0, np.uint8)
        self.offsets = np.fromfile(self.prefix.with_suffix(".idx"), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def sample(self, i: int) -> Sample:
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        ids = np.asarray(self.ids[lo:hi], dtype=np.int64)
        mask = np.asarray(self.mask[lo:hi], dtype=bool)
        return Sample(_spans(ids, mask), meta={"family": self.family})


def _spans(ids: np.ndarray, mask: np.ndarray) -> list[Span]:
    """Split a flat id run back into spans: a new span wherever the modality
    (control below ``TEXT_ID_OFFSET``, text at or above it) or the
    supervision flag changes."""
    if ids.size == 0:
        return []
    kind = ids >= TEXT_ID_OFFSET
    change = np.flatnonzero((kind[1:] != kind[:-1]) | (mask[1:] != mask[:-1])) + 1
    bounds = [0, *change.tolist(), ids.size]
    return [Span("text" if kind[a] else "control", ids[a:b].copy(), supervised=bool(mask[a]))
            for a, b in zip(bounds[:-1], bounds[1:])]


@dataclass
class _Source:
    family: str
    count: int
    get: Callable[[int], object]


class _LazyItems(Sequence):
    def __init__(self, sources: list[_Source]) -> None:
        self.sources = sources
        self.bounds = np.cumsum([0] + [s.count for s in sources])

    def __len__(self) -> int:
        return int(self.bounds[-1])

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        if i < 0:
            i += len(self)
        k = int(np.searchsorted(self.bounds, i, side="right") - 1)
        return self.sources[k].get(i - int(self.bounds[k]))


class MixedCorpus:
    """Duck-types :class:`~iridium.training.datasets.Corpus` without holding items.

    Each source contributes ``count`` item slots. Shard sources wrap around
    their shard (a repeat is an epoch, and the budget audit says so); synthetic
    sources generate item ``i`` from a generator seeded by ``(seed, family, i)``,
    so the same index is always the same example and nothing is stored.
    """

    def __init__(self, sources: list[_Source], split: str = "train") -> None:
        self.items = _LazyItems(sources)
        self.split = split
        self._counts = {}
        for s in sources:
            self._counts[s.family] = self._counts.get(s.family, 0) + s.count

    def __len__(self) -> int:
        return len(self.items)

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    @classmethod
    def build(cls, quotas: dict[str, int], shard_dir: Optional[str | Path] = None,
              seed: int = 0, split: str = "train") -> "MixedCorpus":
        from ..training.tasks import GENERATORS, Item, make_item

        sources: list[_Source] = []
        for family, count in sorted(quotas.items()):
            if count <= 0:
                continue
            if family in GENERATORS:
                def synth(i, family=family):
                    rng = np.random.default_rng([seed, zlib.crc32(family.encode()), i])
                    return make_item(family, rng, split)
                sources.append(_Source(family, count, synth))
                continue
            if shard_dir is None:
                raise ValueError(f"{family!r} needs a prepared shard; run "
                                 "`python -m iridium data prepare` first")
            shard = Shard(Path(shard_dir) / f"{family}-{split}")
            if not len(shard):
                raise ValueError(f"shard for {family!r} is empty")

            def real(i, shard=shard, family=family):
                return Item(sample=shard.sample(i % len(shard)), family=family)
            sources.append(_Source(family, count, real))
        return cls(sources, split)
