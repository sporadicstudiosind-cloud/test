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


#: The mixture for a model meant to *talk* as well as compute. Half natural
#: language -- streamed licensed prose (``text_lm``) and human-written
#: conversations supervised on the assistant's turns (``chat``) -- and half the
#: exactly-checkable synthetic families, which carry the physics, quantity,
#: scene and false-premise skills. ``DEFAULT_MIXTURE`` has no natural language
#: at all, which is right for exercising the mechanisms and is why a model
#: trained on it cannot write (``docs/small-model-coherence.md``). Needs
#: network access to the text and chat sources.
OMNI_MIXTURE: dict[str, float] = {
    "text_lm": 0.35,
    "chat": 0.15,
    "channel_depth": 0.12,
    "channel_intervention": 0.10,
    "field_rollout": 0.10,
    "scene_goal": 0.08,
    "false_premise": 0.10,
}


def allocate_mixture(n_items: int, mixture: dict[str, float]) -> dict[str, int]:
    """Allocate an exact item budget with deterministic largest remainders."""
    if n_items < 1 or not mixture or any(
        not np.isfinite(weight) or weight < 0 for weight in mixture.values()
    ) or sum(mixture.values()) <= 0:
        raise ValueError("positive item count and nonnegative finite mixture weights required")
    names = sorted(name for name, weight in mixture.items() if weight > 0)
    total = float(sum(mixture.values()))
    ideal = {name: n_items * mixture[name] / total for name in names}
    counts = {name: int(np.floor(ideal[name])) for name in names}
    remainder = n_items - sum(counts.values())
    for name in sorted(names, key=lambda key: (-(ideal[key] - counts[key]), key))[:remainder]:
        counts[name] += 1
    return counts


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
    text_mix: Optional[dict[str, float]] = None,
    text_window: int = 256,
    chat_mix: Optional[dict[str, float]] = None,
    tokenizer=None,
    text_skip_docs: int = 0,
) -> Corpus:
    """Assemble a corpus. ``text_lm`` in the mixture streams *real* text.

    The synthetic families are generated locally and are exactly checkable;
    ``text_lm`` pulls from the licensed corpora in ``iridium.data.text_corpus``
    and is graded in bits per byte rather than by an exact checker, because
    there is no right answer to score against — only a likelihood. ``chat``
    does the same with real conversations, supervised on the assistant's turns
    only, and is what makes the model answer a question rather than continue
    the paragraph the question was written in.

    ``tokenizer`` is the subword vocabulary text is encoded with; ``None`` keeps
    the byte-level path. Build it from a config with
    :func:`iridium.training.tokenizer_bridge.tokenizer_for_config`, and pass the
    *same* one to training and to serving — a model read through a different
    vocabulary than it was written with does not error, it produces fluent
    nonsense.

    Note that :data:`DEFAULT_MIXTURE` contains **no** ``text_lm`` or ``chat``
    weight, so the default corpus is entirely synthetic structured families and
    contains no natural language at all. That is a reasonable default for
    exercising the physics and action machinery and a disastrous one for
    training something that talks; :mod:`iridium.training.budget` reports it
    rather than leaving it to be discovered from the model's output.
    """
    mixture = dict(DEFAULT_MIXTURE) if mixture is None else dict(mixture)
    unknown = set(mixture) - set(GENERATORS) - {"text_lm", "chat"}
    if unknown:
        raise ValueError(f"unknown families in mixture: {sorted(unknown)}")
    quotas = allocate_mixture(n_items, mixture)
    n_text = quotas.pop("text_lm", 0)
    n_chat = quotas.pop("chat", 0)
    rng = np.random.default_rng(seed)
    items: list[Item] = []
    if n_text:
        from ..data.text_corpus import text_items
        text = text_items(n_text, window=text_window, mix=text_mix,
                          seed=seed, split=split, tokenizer=tokenizer,
                          skip_docs=text_skip_docs)
        if len(text) != n_text:
            raise RuntimeError(f"text source returned {len(text)}/{n_text} requested items")
        items.extend(text)
    if n_chat:
        from ..data.chat_corpus import chat_items
        chat = chat_items(n_chat, mix=chat_mix, seed=seed, split=split)
        if len(chat) != n_chat:
            raise RuntimeError(f"chat source returned {len(chat)}/{n_chat} requested items")
        items.extend(chat)
    for family, count in sorted(quotas.items()):
        for _ in range(count):
            items.append(make_item(family, rng, split))
    rng.shuffle(items)
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
            if not (batch.valid[:, 1:] & batch.supervised[:, 1:]).any(axis=1).all():
                raise ValueError("example has no targets within context; shorten/rebuild the corpus")
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
