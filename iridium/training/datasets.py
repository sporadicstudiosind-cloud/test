"""Corpus assembly: mixtures, splits, and batching that does not lie.

Two rules this module enforces, both of which are easy to get wrong in a way
that inflates every downstream number:

1. **Splits are by generator parameter, not by random subset.** An
   interpolation item and an extrapolation item are drawn from disjoint
   discharge bands, so a held-out score cannot be earned by having seen a
   neighbouring example.
2. **Padding is never supervised.** ``collate`` marks padded positions invalid
   and every loss divides by its own valid-token count, so a mixture that
   happens to contain long items does not quietly reweight itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
import torch

from ..codecs.bank import TensorBatch, continuous_dims
from ..codecs.spans import Sample, collate
from ..config import CodecConfig
from .tasks import GENERATORS, Item, make_item

DEFAULT_MIXTURE: dict[str, float] = {
    "channel_depth": 0.32,
    "channel_intervention": 0.24,
    "field_rollout": 0.16,
    "scene_goal": 0.12,
    "false_premise": 0.16,
}


@dataclass
class Corpus:
    items: list[Item]
    split: str

    def __len__(self) -> int:
        return len(self.items)

    def by_family(self) -> dict[str, list[Item]]:
        out: dict[str, list[Item]] = {}
        for item in self.items:
            out.setdefault(item.family, []).append(item)
        return out

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.by_family().items()}


def build_corpus(
    n_items: int,
    seed: int = 0,
    split: str = "train",
    mixture: Optional[dict[str, float]] = None,
) -> Corpus:
    mixture = mixture or DEFAULT_MIXTURE
    unknown = set(mixture) - set(GENERATORS)
    if unknown:
        raise ValueError(f"unknown families in mixture: {sorted(unknown)}")
    total = sum(mixture.values())
    rng = np.random.default_rng(seed)
    families = list(mixture)
    weights = np.array([mixture[f] for f in families]) / total
    items: list[Item] = []
    for _ in range(n_items):
        family = families[int(rng.choice(len(families), p=weights))]
        items.append(make_item(family, rng, split))
    return Corpus(items, split)


class BatchLoader:
    """Shuffled batches, with length bucketing available and off by default.

    Bucketing by length is the usual way to cut padding waste, and here it is a
    trap. Item length correlates strongly with task family, so length buckets
    come out family-homogeneous — and the router's load-balancing objective is
    computed *per batch*. On a batch drawn entirely from one family, that
    objective demands the stacks be used equally *for that family*, which is
    the exact opposite of the specialisation phase 2 is trying to produce. The
    balance term only means "spread the corpus across the bank" when the batch
    is a representative sample of the corpus.

    So the default is off, and turning it on is a deliberate choice that should
    be paired with a balance coefficient of zero.
    """

    def __init__(
        self,
        corpus: Corpus,
        codecs: CodecConfig,
        batch_size: int = 8,
        seed: int = 0,
        bucket: bool = False,
        device: str = "cpu",
        max_length: Optional[int] = None,
    ) -> None:
        self.corpus = corpus
        self.dims = continuous_dims(codecs)
        self.n_scalars = codecs.action_scalars
        self.batch_size = batch_size
        self.bucket = bucket
        self.device = device
        self.max_length = max_length
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return (len(self.corpus) + self.batch_size - 1) // self.batch_size

    def batches(self) -> Iterator[tuple[TensorBatch, list[Item]]]:
        order = np.arange(len(self.corpus))
        self.rng.shuffle(order)
        if self.bucket:
            order = sorted(order, key=lambda i: len(self.corpus.items[i].sample))
            groups = [
                order[i : i + self.batch_size]
                for i in range(0, len(order), self.batch_size)
            ]
            self.rng.shuffle(groups)
        else:
            groups = [
                order[i : i + self.batch_size]
                for i in range(0, len(order), self.batch_size)
            ]
        for group in groups:
            items = [self.corpus.items[int(i)] for i in group]
            batch = collate(
                [it.sample for it in items], self.dims, self.n_scalars, self.max_length
            )
            yield TensorBatch(batch, device=self.device), items


def describe(corpus: Corpus) -> str:
    lengths = [len(it.sample) for it in corpus.items]
    lines = [
        f"corpus[{corpus.split}] n={len(corpus)} "
        f"len min/mean/max={min(lengths)}/{np.mean(lengths):.1f}/{max(lengths)}"
    ]
    for family, count in sorted(corpus.counts().items()):
        lines.append(f"  {family:<22} {count:5d}")
    return "\n".join(lines)
