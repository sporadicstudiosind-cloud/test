"""Real text, streamed, with its licence recorded.

"Ethically sourced" is not a vibe, it is a property you can check, and it has
three parts this module makes concrete:

1. **Provenance.** Every source below names a dataset, a licence and what the
   licence obliges you to do. Nothing is scraped here; these are published
   corpora whose terms are stated.
2. **Attribution carried forward.** :class:`SourceSpec` travels into the run
   manifest, so a checkpoint records which corpora produced it. A CC BY-SA
   source obliges attribution and share-alike on derivatives, and a model that
   cannot say what it was trained on cannot honour that.
3. **Streaming, not hoarding.** Sources stream by default, so a notebook reads
   the few hundred megabytes it will actually consume rather than downloading
   terabytes it will not.

What this module deliberately does **not** do is claim the resulting model is
free of the problems its data has. Public-domain books are old and carry the
prejudices of their period; an encyclopedia has systemic coverage gaps; a
web crawl filtered for "educational" is still a web crawl. Recording the
source is the part that can be done honestly, and it is the part that makes
the rest auditable.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class SourceSpec:
    key: str
    dataset: str
    config: Optional[str]
    split: str
    text_field: str
    licence: str
    obligation: str
    description: str
    homepage: str = ""
    #: Rows held for the streaming shuffle. Sized by *row length*, not by taste:
    #: a Gutenberg row is a whole book (~1 MB), so a 1000-row buffer would pull
    #: a gigabyte before yielding anything. A web-crawl row is a few kilobytes
    #: and can afford a real buffer.
    shuffle_buffer: int = 1000
    #: Rough bytes per row, for documentation and for sizing decisions.
    typical_doc_bytes: int = 4_000
    #: Approximate row count, from the Hub's own dataset viewer. Used to sanity
    #: check a request against what the corpus can supply, never to index into it.
    approx_rows: int = 1_000_000
    #: Windows this source may contribute from a single document. A tension,
    #: not a constant: too low and a book-length corpus downloads a megabyte to
    #: use two kilobytes of it; too high and one work dominates the sample.
    #: Sized so that a full run still draws from dozens of distinct documents.
    windows_per_doc: int = 8

    def as_dict(self) -> dict:
        return asdict(self)


#: Corpora with stated, permissive licences. Ordered roughly by how clean the
#: provenance is rather than by how large the corpus is.
SOURCES: dict[str, SourceSpec] = {
    "gutenberg": SourceSpec(
        key="gutenberg",
        dataset="manu/project_gutenberg",
        config=None,
        split="en",          # language is a *split* here, not a config
        text_field="text",
        licence="Public domain (US)",
        obligation=(
            "None on the works themselves — they are out of copyright. The "
            "header Project Gutenberg prepends is a different matter: it carries "
            "the PG trademark and its own licence terms, so it is stripped "
            "before training rather than learned. See strip_gutenberg_boilerplate."
        ),
        description=(
            "Project Gutenberg English books. The cleanest provenance available: "
            "these works are out of copyright outright. They are also old, and "
            "carry the assumptions of their period into anything trained on them."
        ),
        homepage="https://www.gutenberg.org/policy/permission.html",
        shuffle_buffer=16,          # rows are whole books
        typical_doc_bytes=500_000,
        windows_per_doc=192,
        approx_rows=61_300,      # English books in this dump
    ),
    "wikipedia": SourceSpec(
        key="wikipedia",
        dataset="wikimedia/wikipedia",
        config="20231101.en",
        split="train",
        text_field="text",
        licence="CC BY-SA 4.0",
        obligation=(
            "Attribution to Wikipedia and its contributors, and share-alike on "
            "derivative works. Recorded in the run manifest for exactly this reason."
        ),
        description=(
            "English Wikipedia. Broad factual coverage, explicit licence, and "
            "well-documented systemic gaps in what it covers and who wrote it."
        ),
        homepage="https://huggingface.co/datasets/wikimedia/wikipedia",
        shuffle_buffer=512,
        typical_doc_bytes=12_000,   # mean; the median article is far shorter
        windows_per_doc=24,
        approx_rows=6_400_000,   # English articles, 20231101 dump
    ),
    "fineweb_edu": SourceSpec(
        key="fineweb_edu",
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        split="train",
        text_field="text",
        licence="ODC-By 1.0",
        obligation="Attribution to the dataset; underlying pages keep their own terms.",
        description=(
            "CommonCrawl filtered by an educational-quality classifier. Openly "
            "licensed as a dataset, but it is still a web crawl: the filter "
            "raises the average, it does not establish consent from every page."
        ),
        homepage="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
        shuffle_buffer=1000,
        typical_doc_bytes=3_000,
        windows_per_doc=8,
        approx_rows=9_700_000,   # sample-10BT
    ),
    "openwebmath": SourceSpec(
        key="openwebmath",
        dataset="open-web-math/open-web-math",
        config=None,
        split="train",
        text_field="text",
        licence="ODC-By 1.0",
        obligation="Attribution to the dataset.",
        description=(
            "Mathematical web text. Useful here because the model's numeric "
            "competence is the capability actually being measured."
        ),
        homepage="https://huggingface.co/datasets/open-web-math/open-web-math",
        shuffle_buffer=1000,
        typical_doc_bytes=5_000,
        windows_per_doc=8,
        approx_rows=6_300_000,
    ),
}

DEFAULT_MIX = {"gutenberg": 0.4, "wikipedia": 0.4, "fineweb_edu": 0.2}


def licence_notice(keys: Iterable[str]) -> str:
    """The attribution block a derivative work owes. Put it in the manifest."""
    lines = ["Training corpora and their terms:", ""]
    for key in keys:
        spec = SOURCES[key]
        lines += [
            f"* {spec.dataset}" + (f" ({spec.config})" if spec.config else ""),
            f"    licence:     {spec.licence}",
            f"    obligation:  {spec.obligation}",
            f"    homepage:    {spec.homepage}",
        ]
    lines += [
        "",
        "Recording this is the auditable part. It does not claim the resulting",
        "model is free of the biases these corpora carry.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------

#: Percentile bands, disjoint by construction.
SPLIT_SHARES: dict[str, tuple[int, int]] = {
    "train": (0, 80),
    "test": (80, 90),
    "extrapolation": (90, 100),
}


def split_bucket(doc: str) -> int:
    """A stable bucket in ``[0, 100)`` derived from the document's own bytes."""
    digest = hashlib.blake2b(doc.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 100


def in_split(doc: str, split: str) -> bool:
    """Does this document belong to this split?

    Content-hashed rather than positional, and that choice matters. The obvious
    alternative — give each split a different random seed — does **not**
    separate them: a streaming shuffle permutes within a buffer while
    traversing the file in order, so every split still sees the same early
    documents, and the test set quietly contains text the model trained on.
    Bits per byte then measures memorisation and reports it as generalisation.

    Hashing the content instead makes membership a property of the document.
    The same book lands in the same split under any seed, any buffer size, any
    corpus ordering, and on any machine — so a result computed here is
    comparable with a result computed in a notebook a month later.

    Measured on FineWeb-Edu, the seed-only scheme put **10%** of an 80-document
    test set inside the training set. Content hashing puts zero there, which
    ``tests/unit/test_text_corpus.py`` checks against the live corpus.
    """
    lo, hi = SPLIT_SHARES.get(split, SPLIT_SHARES["train"])
    return lo <= split_bucket(doc) < hi


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------

# Project Gutenberg's markers are not one format, they are a family of them,
# accumulated over decades of transcription. Observed in this corpus:
#
#   *** START OF THIS PROJECT GUTENBERG EBOOK <title> ***
#   ***START OF THE PROJECT GUTENBERG EBOOK <title>***      (no spaces)
#   *** END OF THIS PROJECT GUTENBERG EBOOK <title> ***
#   End of the Project Gutenberg EBook of <title>, by <author>   (no asterisks)
#   End of Project Gutenberg's <title>, by <author>              (no "the")
#
# The last two sit *before* the asterisk footer, so matching only the asterisk
# form leaves a licence line in every book. Matching the earliest footer is the
# fix; the position guard below is what keeps that from being dangerous.
_PG_START = re.compile(
    r"\*\*\*\s*START OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
_PG_END = re.compile(
    r"\*\*\*\s*END OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK.*?\*\*\*"
    r"|End of (?:the )?Project Gutenberg(?:'s|\u2019s)?[^\n]*",
    re.IGNORECASE | re.DOTALL,
)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Drop the header and footer Project Gutenberg wraps around each book.

    Two reasons, and the licence one is the one that matters. The book is
    public domain; the wrapper is not — it carries the Project Gutenberg
    trademark and the terms attached to it. Training on tens of thousands of
    copies of the same licence notice would also teach the model to recite it,
    which is both useless and precisely the text you least want reproduced.

    **The position guard.** A footer is only honoured in the back half of the
    document and a header only in the front half. Without it, a plain-text
    match like "End of Project Gutenberg's ..." occurring in a preface or a
    bibliography would truncate the book there, silently, and the corpus would
    quietly lose most of its content with nothing to indicate it. Losing a
    licence line is a small cost; losing nine tenths of a book without noticing
    is not.

    Conservative throughout: if the markers are missing, out of order, or would
    leave nothing behind, the document is returned unchanged.
    """
    n = len(text)
    half = n // 2

    body_start = 0
    for m in _PG_START.finditer(text):
        if m.start() <= half:
            body_start = m.end()
    body_end = n
    for m in _PG_END.finditer(text):
        if m.start() >= max(body_start, half):
            body_end = m.start()
            break

    body = text[body_start:body_end].strip()
    return body or text.strip()


#: ``source key -> cleaner``. Applied in :func:`stream_documents`.
CLEANERS = {"gutenberg": strip_gutenberg_boilerplate}


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------


def stream_documents(
    key: str,
    limit: Optional[int] = None,
    seed: int = 0,
    buffer: Optional[int] = None,
    shuffle: bool = True,
    split: Optional[str] = None,
    max_scanned: Optional[int] = None,
) -> Iterator[str]:
    """Yield raw documents from one source. Requires ``datasets``.

    ``buffer`` defaults to the source's own :attr:`SourceSpec.shuffle_buffer`,
    because a shuffle buffer is measured in *rows* and a row is a whole book in
    one corpus and a paragraph of web text in another. Passing 1000 everywhere
    downloads a gigabyte of Gutenberg before the first document appears.

    ``limit`` counts documents **yielded**, not rows scanned, so a split filter
    cannot silently shorten the result. ``max_scanned`` bounds the work anyway,
    so a filter that matches nothing terminates instead of reading the corpus.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:                          # pragma: no cover
        raise RuntimeError(
            "pip install datasets  — required for real-text training"
        ) from exc
    spec = SOURCES[key]
    ds = load_dataset(
        spec.dataset, spec.config, split=spec.split, streaming=True,
    )
    if shuffle:
        ds = ds.shuffle(seed=seed, buffer_size=buffer or spec.shuffle_buffer)
    clean = CLEANERS.get(key, lambda t: t)
    if max_scanned is None and limit is not None:
        max_scanned = limit * 50 + 200
    produced = 0
    for scanned, row in enumerate(ds):
        if max_scanned is not None and scanned >= max_scanned:
            return
        text = row.get(spec.text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        if split is not None and not in_split(text, split):
            continue
        yield clean(text)
        produced += 1
        if limit is not None and produced >= limit:
            return


def collect_documents(
    n_docs: int,
    mix: Optional[dict[str, float]] = None,
    seed: int = 0,
) -> list[tuple[str, str]]:
    """Draw ``n_docs`` documents across the mixture, **one source at a time**.

    The obvious implementation interleaves three live generators, and that is
    what this replaced. It failed in practice: three concurrent HTTP streams
    stall behind connection limits and download gates — a corporate proxy, a
    Kaggle container, this build environment — and the symptom is a notebook
    that hangs on cell 4 with no error to read. Draining one source before
    opening the next keeps exactly one connection live, and shuffling at the
    end recovers the interleaving that mattered.

    The quota per source is exact (``round(weight * n_docs)``), so the realised
    mixture matches the requested one rather than approaching it in
    expectation. A source that runs dry short of its quota is reported by
    :func:`realised_mixture` rather than silently backfilled from elsewhere.
    """
    mix = mix or DEFAULT_MIX
    total = sum(mix.values())
    rng = np.random.default_rng(seed)
    out: list[tuple[str, str]] = []
    for i, (key, weight) in enumerate(sorted(mix.items())):
        quota = int(round(n_docs * weight / total))
        if quota <= 0:
            continue
        for doc in stream_documents(key, limit=quota, seed=seed + i):
            out.append((key, doc))
    rng.shuffle(out)
    return out


def stream_mixture(
    mix: Optional[dict[str, float]] = None,
    limit: Optional[int] = None,
    seed: int = 0,
) -> Iterator[tuple[str, str]]:
    """Interleave sources by weight, holding one connection open per source.

    Kept for callers that genuinely want a lazy interleaved stream. Prefer
    :func:`collect_documents` in a notebook: this version needs every source's
    connection live at once, which is the thing that stalls behind a proxy.
    """
    mix = mix or DEFAULT_MIX
    keys = list(mix)
    weights = np.array([mix[k] for k in keys], dtype=float)
    weights = weights / weights.sum()
    streams = {k: stream_documents(k, seed=seed + i) for i, k in enumerate(keys)}
    rng = np.random.default_rng(seed)
    produced = 0
    while limit is None or produced < limit:
        key = keys[int(rng.choice(len(keys), p=weights))]
        try:
            yield key, next(streams[key])
            produced += 1
        except StopIteration:
            streams.pop(key, None)
            idx = keys.index(key)
            keys.pop(idx)
            weights = np.delete(weights, idx)
            if not keys:
                return
            weights = weights / weights.sum()


# --------------------------------------------------------------------------
# turning documents into training items
# --------------------------------------------------------------------------


def text_items(
    n_items: int,
    window: int = 256,
    mix: Optional[dict[str, float]] = None,
    seed: int = 0,
    offset: int = 16,
    max_windows_per_doc: Optional[int] = None,
    split: str = "train",
):
    """Build byte-level language-modelling items from streamed real text.

    Bytes rather than a learned tokenizer: the model's text codec is
    byte-level, so there is no vocabulary to train, nothing to go stale, and
    no silent mismatch between a tokenizer trained on one corpus and a model
    trained on another.

    Two things here are deliberate, and both were wrong in the obvious version:

    * ``max_windows_per_doc``, defaulting per source to
      :attr:`SourceSpec.windows_per_doc`. Taking every consecutive window of a
      document looks like it respects the mixture weights and does not: one
      Gutenberg book is a megabyte, so four thousand consecutive windows arrive
      from a single author before any other source is touched. Windows are
      capped per document and spread across it. The cap is per source because
      the right answer differs by an order of magnitude: eight windows from a
      3 kB web page is most of it, and eight windows from a 1 MB book means
      downloading five hundred times what you keep.
    * **The quota is counted in items, per source, and each source is drained
      before the next opens.** Drawing documents and stopping when the item
      budget fills leaves the tail of the document list unused, and the
      realised mixture then misses the requested one by ten points or more on
      a short run. Counting items makes it exact, and opening one connection
      at a time is what survives a proxy.

    Check the result rather than trusting it: :func:`realised_mixture` reports
    what the items actually contain.
    """
    from ..codecs.spans import Sample, text_span
    from ..training.tasks import Item, control_span, BOS, EOS

    mix = mix or DEFAULT_MIX
    total = sum(mix.values())
    rng = np.random.default_rng(seed)
    items = []

    for i, (key, weight) in enumerate(sorted(mix.items())):
        quota = int(round(n_items * weight / total))
        if quota <= 0:
            continue
        cap = max_windows_per_doc or SOURCES[key].windows_per_doc
        got = 0
        # Document allowance, deliberately generous. Sizing it by the *mean*
        # document length under-draws badly: Wikipedia's mean article is ~12 kB
        # and its median is a small fraction of that, so a budget computed from
        # the mean runs out of documents at three-quarters of the quota. The
        # loop breaks the moment the quota is met, so headroom is free.
        budget = max(8, -(-quota // max(cap, 1)) * 8 + 8)
        for doc in stream_documents(key, limit=budget, seed=seed + i, split=split):
            raw = doc.encode("utf-8", errors="ignore")
            n_windows = max(len(raw) // window, 1)
            take = min(cap, n_windows, quota - got)
            starts = (
                rng.choice(n_windows, size=take, replace=False) * window
                if n_windows > take
                else np.arange(take) * window
            )
            for start in sorted(int(x) for x in starts):
                chunk = raw[start : start + window]
                if len(chunk) < window // 2:
                    continue
                items.append(Item(
                    sample=Sample(
                        [
                            control_span(BOS, supervised=False),
                            text_span(chunk, supervised=True, offset=offset),
                            control_span(EOS),
                        ],
                        meta={"family": "text_lm", "source": key},
                    ),
                    family="text_lm",
                    prompt=chunk[:40].decode("utf-8", errors="replace"),
                    truth={"source": key, "bytes": len(chunk)},
                ))
                got += 1
            if got >= quota:
                break

    rng.shuffle(items)
    if len(items) < n_items:
        warnings.warn(
            f"asked for {n_items} text items, produced {len(items)}; "
            f"realised mixture {realised_mixture(items)}",
            stacklevel=2,
        )
    return items[:n_items]


def realised_mixture(items) -> dict[str, float]:
    """What fraction of the items each source actually produced.

    Worth printing next to the requested mixture: the two agreeing is a claim
    that can be checked, and they disagree more easily than you would expect.
    """
    counts: dict[str, int] = {}
    for it in items:
        key = it.truth.get("source", "?")
        counts[key] = counts.get(key, 0) + 1
    total = max(sum(counts.values()), 1)
    return {k: v / total for k, v in sorted(counts.items())}


def probe_availability(keys: Optional[Iterable[str]] = None) -> dict[str, str]:
    """Which sources this environment can actually reach. Network-dependent."""
    out: dict[str, str] = {}
    for key in (keys or SOURCES):
        try:
            next(stream_documents(key, limit=1, shuffle=False))
            out[key] = "ok"
        except Exception as exc:
            out[key] = f"unavailable: {type(exc).__name__}: {str(exc)[:90]}"
    return out
