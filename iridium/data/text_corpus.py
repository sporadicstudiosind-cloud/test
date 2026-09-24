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
            "carry the assumptions of their period into anything trained on them. "
            "Not in any default mix: streaming this dump peaks near 12 GB of host "
            "RAM before the first book (measured), more than a free Colab session "
            "has."
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
    # Synthetic textbooks and stories (Cosmopedia, generated by the Apache-2.0
    # Mixtral-8x7B-Instruct) are what SmolLM-class small models lean on for
    # fluency; FineMath is the cleanest openly licensed maths prose. Licences
    # read from the dataset cards, 2026-09.
    "cosmopedia_stories": SourceSpec(
        key="cosmopedia_stories",
        dataset="HuggingFaceTB/cosmopedia",
        config="stories",
        split="train",
        text_field="text",
        licence="Apache 2.0",
        obligation="Retain the licence and attribution notices.",
        description=(
            "Synthetic short stories written for varied audiences. Simple, "
            "complete narratives are what a 30-100M model can learn to produce "
            "end to end, the lesson of TinyStories at a broader vocabulary."
        ),
        homepage="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia",
        shuffle_buffer=1000,
        typical_doc_bytes=3_500,
        windows_per_doc=4,
        approx_rows=5_000_000,
    ),
    "cosmopedia_textbooks": SourceSpec(
        key="cosmopedia_textbooks",
        dataset="HuggingFaceTB/cosmopedia",
        config="stanford",
        split="train",
        text_field="text",
        licence="Apache 2.0",
        obligation="Retain the licence and attribution notices.",
        description=(
            "Synthetic textbook sections seeded from university course outlines: "
            "explanatory prose with worked examples, dense in the connective "
            "reasoning web text rarely spells out. Synthetic, so it carries its "
            "generator's errors; keep it a minority share."
        ),
        homepage="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia",
        shuffle_buffer=1000,
        typical_doc_bytes=4_500,
        windows_per_doc=6,
        approx_rows=1_000_000,
    ),
    "finemath": SourceSpec(
        key="finemath",
        dataset="HuggingFaceTB/finemath",
        config="finemath-4plus",
        split="train",
        text_field="text",
        licence="ODC-By 1.0",
        obligation="Attribution to the dataset; underlying pages keep their own terms.",
        description=(
            "CommonCrawl maths pages kept by an educational-maths classifier at "
            "its stricter threshold: step-by-step explanations rather than bare "
            "formulas."
        ),
        homepage="https://huggingface.co/datasets/HuggingFaceTB/finemath",
        shuffle_buffer=1000,
        typical_doc_bytes=5_000,
        windows_per_doc=8,
        approx_rows=6_700_000,
    ),
}

DEFAULT_MIX = {"wikipedia": 0.45, "fineweb_edu": 0.35, "cosmopedia_stories": 0.20}

#: Prose for the talking presets: the default sources plus synthetic stories
#: and textbooks, which small models learn fluency from fastest.
TALK_MIX = {"fineweb_edu": 0.35, "wikipedia": 0.25, "cosmopedia_stories": 0.20,
            "cosmopedia_textbooks": 0.15, "finemath": 0.05}

#: Prose for the STEM preset.
STEM_MIX = {"finemath": 0.35, "openwebmath": 0.15, "fineweb_edu": 0.25,
            "cosmopedia_textbooks": 0.15, "wikipedia": 0.10}


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
    skip: int = 0,
) -> Iterator[str]:
    """Yield raw documents from one source. Requires ``datasets``.

    ``skip`` drops that many rows, in file order, before the shuffle buffer
    fills. It is what lets training proceed in *rounds* of fresh documents:
    a shuffle buffer only permutes a window of the stream, so re-seeding it
    each round re-reads mostly the same opening documents -- the same flaw
    that once leaked 10% of a test split into training. Skipping is paid in
    reading (the rows still stream past), not in memory.

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
    if skip:
        ds = ds.skip(int(skip))
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

    The quota per source uses largest-remainder allocation, so the realised
    mixture matches the requested one rather than approaching it in
    expectation. A source that runs dry short of its quota is reported by
    :func:`realised_mixture` rather than silently backfilled from elsewhere.
    """
    from ..training.datasets import allocate_mixture

    mix = mix or DEFAULT_MIX
    quotas = allocate_mixture(n_docs, mix)
    rng = np.random.default_rng(seed)
    out: list[tuple[str, str]] = []
    for i, (key, quota) in enumerate(sorted(quotas.items())):
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


# --------------------------------------------------------------------------
# quality filters
#
# Each one is switchable independently because each catches a different,
# unrelated failure mode of scraped/crowd-sourced text, and a corpus that is
# clean on one axis can be filthy on another. Bundling them into one
# all-or-nothing flag would make it impossible to, say, keep the dedup filter
# (cheap, never wrong) while turning off the printable-ratio filter to
# inspect what it is actually rejecting.
# --------------------------------------------------------------------------


def _printable_ratio(text: str) -> float:
    """Fraction of characters that are printable text (plus newline/tab)
    rather than control bytes, private-use codepoints, or other debris that
    an HTML-to-text or PDF-to-text extractor leaves behind. A document that
    is mostly binary noise costs the same shuffle-buffer slot and download as
    a clean one but teaches the model nothing about language — worse, it
    teaches a byte-level model that these particular non-text byte sequences
    are worth predicting accurately, which is capacity spent on precisely
    the wrong thing."""
    if not text:
        return 0.0
    printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
    return printable / len(text)


def _repeat_run_pattern(min_run: int):
    return re.compile(r"(.)\1{%d,}" % min_run)


def collapse_repeated_runs(text: str, max_repeat: int = 8) -> str:
    """Collapse a run of more than ``max_repeat`` identical characters down to
    exactly ``max_repeat`` of them.

    Divider lines ("--------------------------------"), decorative ASCII-art
    borders, and repeated-character spam ("hahahahaha...") are common in
    scraped web text and teach a byte-level model a genuinely useless skill:
    predicting "same byte again" hundreds of times in a row. That is trivial
    loss to minimise and buys the model nothing, so long runs are truncated
    rather than deleted outright — the document still shows *that* a
    separator was there, just not how long it happened to be. ``.`` does not
    match a newline by default, so a run of blank lines (often meaningful
    paragraph spacing) is left alone; only a single repeated non-newline
    character triggers this.
    """
    return _repeat_run_pattern(max_repeat).sub(lambda m: m.group(1) * max_repeat, text)


def _prefix_hash(text: str, n: int = 512) -> bytes:
    """A hash of a document's first ``n`` bytes, not the whole document.

    Hashing everything is the obviously-more-thorough version and also the
    expensive one: at Gutenberg's half-megabyte documents, hashing every byte
    of every candidate before deciding whether to keep it undoes the point of
    *streaming* text in the first place. Templated boilerplate — mirrored
    articles, a scraper's repeated page header, a duplicated disclaimer —
    overwhelmingly repeats in a document's opening even when the body differs
    further down, so the prefix is where duplication actually shows up; this
    catches that at a fixed, small cost per document regardless of its length.
    """
    head = text.encode("utf-8", "ignore")[:n]
    return hashlib.blake2b(head, digest_size=8).digest()


def _clean_doc(
    doc: str,
    *,
    filter_printable: bool,
    min_printable_ratio: float,
    collapse_repeats: bool,
    max_repeat_run: int,
    dedupe: bool,
    seen_hashes: set,
) -> Optional[str]:
    """Run the enabled filters in order; ``None`` means "drop this document".

    Order is not arbitrary. Printable-ratio is checked on the document
    *before* collapsing repeats, so a document that is mostly one repeated
    control byte cannot dodge the ratio check by first being tidied up into
    something that looks like eight bytes of it. The dedup hash is taken
    *after* collapsing, so two copies of the same template differing only in
    how a divider line got mangled by extraction still hash identically.
    """
    if filter_printable and _printable_ratio(doc) < min_printable_ratio:
        return None
    if collapse_repeats:
        doc = collapse_repeated_runs(doc, max_repeat_run)
    if dedupe:
        h = _prefix_hash(doc)
        if h in seen_hashes:
            return None
        seen_hashes.add(h)
    return doc


def _encode_doc(doc: str, tokenizer) -> bytes:
    """Raw UTF-8 bytes when ``tokenizer`` is ``None`` (today's behaviour,
    unchanged), or token ids from ``tokenizer`` otherwise. Both are returned
    as a sequence supporting ``len``, slicing and concatenation, so the
    packing logic below never needs to know which case it is in — see
    ``text_items`` for where that distinction actually matters (building the
    ``Span``)."""
    if tokenizer is None:
        return doc.encode("utf-8", errors="ignore")
    return tokenizer.encode(doc)


def _preview(chunk, tokenizer) -> str:
    """A short human-readable prefix of a window, for :attr:`Item.prompt`."""
    head = bytes(chunk[:40]) if tokenizer is None else list(chunk[:40])
    if tokenizer is None:
        return head.decode("utf-8", errors="replace")
    return tokenizer.decode(head)


def text_items(
    n_items: int,
    window: int = 256,
    mix: Optional[dict[str, float]] = None,
    seed: int = 0,
    offset: int = 16,
    max_windows_per_doc: Optional[int] = None,
    split: str = "train",
    pack: bool = True,
    tokenizer=None,
    filter_printable: bool = True,
    min_printable_ratio: float = 0.85,
    collapse_repeats: bool = True,
    max_repeat_run: int = 8,
    dedupe: bool = True,
    skip_docs: int = 0,
):
    """Build language-modelling items from streamed real text.

    ``skip_docs`` starts every source that many rows in, so successive calls
    with growing offsets yield *new* documents -- the round-based training in
    :mod:`iridium.training.run_preset` depends on it.

    ``tokenizer=None`` keeps the original byte-level behaviour (id == byte
    value): there is no vocabulary to train, nothing to go stale, and no
    silent mismatch between a tokenizer trained on one corpus and a model
    trained on another. Passing a trained :class:`~iridium.data.tokenizer.BytePairTokenizer`
    switches windows to be measured in *tokens* rather than bytes, which at a
    fixed sequence length lets the model see several times more text — see
    ``iridium/data/tokenizer.py`` for why that gap exists and how large it is.

    **Packing (the actual fix this function exists for).** The previous
    version cut ``window``-byte slices at *random* byte offsets inside each
    document and wrapped every single one in BOS/EOS as though it were a
    whole document. That is wrong in a way that is easy to miss and expensive
    once trained on: the overwhelming majority of a 500 kB book's 2,000
    windows start and end mid-word, mid-sentence, sometimes mid-UTF-8-
    sequence, and the model is told, a few hundred thousand times, "a
    document begins here" about a position that is nothing of the sort. A
    model trained that way has no working notion of what a document boundary
    even is, which is a large part of why free-running generation "says a
    bunch of crap": it was never shown a real one to imitate.

    With ``pack=True`` (the default), encoded documents for a source are
    concatenated into one long token stream, EOS-free at the token level —
    the boundary is tracked out-of-band as a set of stream positions, not
    spliced into the text vocabulary — and windows are cut *contiguously*
    from that stream. A window gets a leading, unsupervised BOS control span
    only when its first token is genuinely the first token of some document;
    it gets a trailing EOS control span only when its last token is genuinely
    a document's last token. Most packed windows are interior and get
    neither, which is the correct thing to model: the middle of a book is not
    a place where a fresh document starts or an old one ends, and no longer
    claims to be either. This also fixes the tail-dropping the random-window
    version had: every byte/token of every fetched document is used exactly
    once (modulo at most one partial window at each end of the packed
    stream), instead of one random window in twenty and the rest of the
    document discarded.

    ``pack=False`` reproduces the old random-window behaviour byte-for-byte
    (modulo the new quality filters, which apply either way) — kept for
    direct comparison, not for training a model you want to be coherent.
    ``max_windows_per_doc`` (see :attr:`SourceSpec.windows_per_doc`) only
    means something in that mode: under packing, the "one book must not
    dominate" problem it solved is already solved by fetching many documents
    into one packed stream, and a per-document window cap on top of that
    would just throw away already-packed tokens for no benefit.

    **Quality filters**, applied before packing (see the functions above for
    what each one catches and why it is worth the cost): printable-character
    ratio, repeated-character-run collapsing, and prefix-hash de-duplication.
    Each is independently switchable via its own flag.

    **The quota is still counted in items, per source, with each source
    drained before the next opens** — unchanged from before, and
    :func:`realised_mixture` still reports what was actually produced.
    """
    from ..codecs.spans import Sample, Span, text_span
    from ..training.tasks import Item, control_span, BOS, EOS

    from ..training.datasets import allocate_mixture

    mix = mix or DEFAULT_MIX
    quotas = allocate_mixture(n_items, mix)
    rng = np.random.default_rng(seed)
    items = []

    for i, (key, quota) in enumerate(sorted(quotas.items())):
        if quota <= 0:
            continue
        seen_hashes: set = set()

        def clean(doc: str) -> Optional[str]:
            return _clean_doc(
                doc,
                filter_printable=filter_printable,
                min_printable_ratio=min_printable_ratio,
                collapse_repeats=collapse_repeats,
                max_repeat_run=max_repeat_run,
                dedupe=dedupe,
                seen_hashes=seen_hashes,
            )

        if pack:
            # Enough tokens for `quota` windows, plus one window of slack so
            # a random starting phase (see below) never leaves the packed
            # stream a few tokens short of a full quota.
            needed = (quota + 1) * window
            max_scanned = max(quota * 200 + 500, 500)
            stream = bytearray() if tokenizer is None else []
            doc_starts: list[int] = []
            for doc in stream_documents(
                key, limit=None, seed=seed + i, split=split, max_scanned=max_scanned,
            ):
                doc = clean(doc)
                if doc is None:
                    continue
                ids = _encode_doc(doc, tokenizer)
                if not ids:
                    continue
                doc_starts.append(len(stream))
                stream.extend(ids)
                if len(stream) >= needed:
                    break

            n = len(stream)
            n_windows = n // window
            if n_windows > 0:
                # A random phase, rather than always starting at index 0,
                # keeps "is this window a real document start" from
                # correlating with "which window index is this" across many
                # calls with different seeds — the packed stream's own
                # document lengths, not the caller's seed, should decide that.
                phase = int(rng.integers(0, window)) if n_windows > 1 else 0
                doc_start_set = set(doc_starts)
                idx = phase
                got = 0
                while idx + window <= n and got < quota:
                    chunk = stream[idx : idx + window]
                    is_start = idx in doc_start_set
                    end = idx + window
                    is_end = end in doc_start_set or end == n
                    spans = []
                    if is_start:
                        spans.append(control_span(BOS, supervised=False))
                    if tokenizer is None:
                        spans.append(text_span(bytes(chunk), supervised=True, offset=offset))
                    else:
                        ids_arr = np.asarray(chunk, dtype=np.int64) + offset
                        spans.append(Span("text", ids_arr, supervised=True))
                    if is_end:
                        spans.append(control_span(EOS))
                    items.append(Item(
                        sample=Sample(spans, meta={"family": "text_lm", "source": key}),
                        family="text_lm",
                        prompt=_preview(chunk, tokenizer),
                        truth={"source": key, "bytes": len(chunk)},
                    ))
                    got += 1
                    idx += window
        else:
            # Legacy path: random windows, always BOS/EOS-wrapped, capped and
            # spread per document. Kept only for comparing against `pack=True`
            # — see the docstring above for why this window is a documented
            # falsehood the model was trained to believe.
            cap = max_windows_per_doc or SOURCES[key].windows_per_doc
            got = 0
            budget = max(8, -(-quota // max(cap, 1)) * 8 + 8)
            for doc in stream_documents(key, limit=budget, seed=seed + i, split=split,
                                        skip=skip_docs):
                doc = clean(doc)
                if doc is None:
                    continue
                raw = _encode_doc(doc, tokenizer)
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
                    if tokenizer is None:
                        text_span_obj = text_span(bytes(chunk), supervised=True, offset=offset)
                    else:
                        ids_arr = np.asarray(chunk, dtype=np.int64) + offset
                        text_span_obj = Span("text", ids_arr, supervised=True)
                    items.append(Item(
                        sample=Sample(
                            [
                                control_span(BOS, supervised=False),
                                text_span_obj,
                                control_span(EOS),
                            ],
                            meta={"family": "text_lm", "source": key},
                        ),
                        family="text_lm",
                        prompt=_preview(chunk, tokenizer),
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
