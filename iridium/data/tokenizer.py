"""A byte-level BPE tokenizer with no third-party dependencies.

**Why this exists.** The model's text codec (``iridium/codecs/spans.py::text_span``)
is byte-level: 256 raw byte values plus a handful of reserved control ids. That
choice was originally "free" — no vocabulary to train, nothing to go stale, no
mismatch between a tokenizer trained on one corpus and a model trained on
another — but it has a real cost that only shows up at small model sizes: at a
fixed sequence length, a byte-level model sees roughly a quarter of the *text*
that a subword model sees, and spends a large share of its limited capacity
re-deriving orthography (that "the" is common, that "ing" ends a word) instead
of learning anything about what the text says. Every competitive small open
model (Qwen, Llama, Gemma, SmolLM) uses a 32k-256k subword vocabulary for
exactly this reason. This module is the smallest thing that closes that gap
without adding a dependency (``tokenizers``/``sentencepiece``/``tiktoken`` are
all off the table): a from-scratch, GPT-2-style byte-pair encoder.

**Losslessness is the non-negotiable property.** A BPE vocabulary trained on
English web text will never contain a merge for, say, the raw bytes of a
Shift-JIS document or a PNG that leaked into a text field, and a tokenizer
that falls back to "UNK" there silently destroys data and — worse — does it
unevenly across languages and scripts. This tokenizer never emits UNK: ids
``0..255`` are permanently reserved for the 256 raw byte values, every
learned merge is built *on top of* that base, and encoding a piece of text
that shares no vocabulary with training data just falls back to spelling it
out byte-by-byte. ``decode(encode(s)) == s`` therefore holds for arbitrary
unicode *and* arbitrary bytes (the latter round-tripped through ``str` via
``surrogateescape``, the same trick :pep:`383` uses for filenames), not only
for text that resembles the training corpus.

**The pre-tokenizer** is GPT-2's regex family: contractions, then runs of
letters, runs of punctuation, and whitespace are split into separate chunks
*before* BPE ever sees them, so a merge can never straddle a word boundary
into an unrelated word or glue a word to trailing punctuation. Digits are the
one deliberate departure from GPT-2: each digit becomes its own pre-token
rather than joining a run. A model that has memorised a byte-pair merge for
"197" and a different one for "198" has learned two unrelated symbols where
arithmetic needs it to see "coefficient 1, then 9, then 7"; keeping digits
separate is a small, well-documented lever (used by LLaMA and others) for
small-model arithmetic, and it costs nothing here since digits are rare
enough that merging them would not have bought much vocabulary efficiency
anyway.

**Training cost.** BPE training is inherently vocabulary-sized-many merges
over a corpus, and the naive implementation — rescan the whole corpus to find
the next most frequent pair, for every merge — is exactly the "O(corpus) per
merge" approach this module does *not* use, because it does not finish in a
useful amount of time: 8,000 merges times a 20 MB corpus is on the order of
160 GB of character comparisons. Instead this trains on a bag of *unique
pre-tokens with counts* (English text has far fewer distinct words than
total words) and maintains an incremental pair-frequency count plus an
inverted index of which words contain which pair, using a lazily-invalidated
max-heap to find the next merge in ``O(log n)``. Net cost is close to
``O(corpus)`` **total**, not per merge — training an 8k vocabulary over ~20 MB
of text is a low-single-digit-minutes job on one core, not hours. The one
deliberate cap: pre-tokens are truncated to the ``max_words`` most frequent
distinct ones (default 200k) before training starts, since a merge is only
ever chosen from a *frequent* pair and the long tail of hapax legomena (or,
on pathological input, unique noise) cannot win a frequency contest anyway —
dropping it up front just saves the bookkeeping cost of carrying it.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

N_BYTES = 256

# GPT-2's split, minus \p{L}/\p{N} (the `regex` package is a dependency this
# project does not take): Python's stdlib `re` has no Unicode property
# classes, so letters and digits are approximated with `\w` (Unicode-aware by
# default for `str` patterns) minus digits. Digits are split into single
# characters rather than runs — see the module docstring for why. Underscore
# is deliberately grouped with letters rather than left to fend for itself:
# `\w` counts it as a word character, so excluding it from *both* the
# letter-run class and the punctuation-run class (which is defined as
# "non-space, non-word") would leave it matched by nothing at all — a
# concrete bug this module's own tests caught, where every underscore in a
# round-tripped byte string silently vanished. The final two alternatives
# (`\s+(?!\S)` then `\s+`) are GPT-2's trick for putting trailing whitespace
# before a word with that word rather than after the previous one, which
# keeps "the boundary between words" mid-whitespace consistent regardless of
# how much whitespace separates them.
_PRETOKEN_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d]+| ?\d| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def _pretokenize(text: str) -> list[str]:
    """Split text into BPE's atomic units. Concatenating the result recovers
    the input exactly — no character is ever dropped by this step, which
    matters because it is what keeps the tokenizer lossless even on text the
    regex was not designed with in mind (emoji, combining marks, lone
    surrogates standing in for raw bytes — see ``_to_bytes``)."""
    return _PRETOKEN_PATTERN.findall(text)


def _to_bytes(text: str) -> bytes:
    """UTF-8 encode with ``surrogateescape`` rather than the usual strict/
    ignore/replace error handlers. Strict raises on a lone surrogate (which
    is how :func:`_from_bytes` represents a byte that was not valid UTF-8);
    ignore/replace both destroy information. Surrogateescape is the one
    handler that is a true inverse of :func:`_from_bytes`, which is what
    makes "arbitrary bytes" (not just arbitrary *unicode*) round-trip through
    this tokenizer: a caller who has raw, possibly-non-UTF-8 bytes can get a
    losslessly-representable ``str`` via ``data.decode("utf-8",
    "surrogateescape")`` and hand that to :meth:`BytePairTokenizer.encode`.
    """
    return text.encode("utf-8", "surrogateescape")


def _from_bytes(data: bytes) -> str:
    return data.decode("utf-8", "surrogateescape")


class BytePairTokenizer:
    """A trained-or-untrained byte-level BPE codec.

    With no merges and no special tokens (the state a fresh instance starts
    in, and what :func:`byte_tokenizer` returns explicitly), this is exactly
    the identity byte mapping the rest of the codebase already uses: id ==
    byte value. Training only ever *adds* merges on top of ids 256+, so
    nothing that already depends on the untrained byte behaviour breaks.
    """

    def __init__(self) -> None:
        self.special_tokens: dict[str, int] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self._merge_order: list[tuple[int, int]] = []
        self.id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(N_BYTES)}
        self.vocab_size: int = N_BYTES
        self._rank: dict[tuple[int, int], int] = {}

    # -- training ------------------------------------------------------

    def train(
        self,
        iterable_of_str: Iterable[str],
        vocab_size: int,
        special_tokens: Sequence[str] = (),
        max_words: int = 200_000,
    ) -> "BytePairTokenizer":
        """Learn merges from a corpus of strings.

        ``vocab_size`` bounds the *total* vocabulary (256 raw bytes +
        specials + merges), matching how ``vocab_size`` is used everywhere
        else in this codebase (``iridium.config.CodecConfig.vocab_size``) — a
        caller does not have to separately account for the reserved ids.
        Requesting a ``vocab_size`` at or below ``256 + len(special_tokens)``
        is valid and simply trains zero merges (a pure byte+specials
        tokenizer), rather than raising, since a caller sweeping vocabulary
        sizes should not have to special-case the smallest one.
        """
        specials = list(dict.fromkeys(special_tokens))  # de-dup, keep order
        self.special_tokens = {s: N_BYTES + i for i, s in enumerate(specials)}
        base = N_BYTES + len(specials)
        n_merges = max(0, vocab_size - base)

        self.id_to_bytes = {i: bytes([i]) for i in range(N_BYTES)}
        for tok, tid in self.special_tokens.items():
            self.id_to_bytes[tid] = _to_bytes(tok)

        # Unique pre-tokens with counts. Training BPE on *unique words* rather
        # than the raw token stream is what makes this tractable: English
        # text has orders of magnitude fewer distinct words than total words,
        # and a merge's frequency only needs the count, not every occurrence.
        freq: dict[tuple[int, ...], int] = {}
        for text in iterable_of_str:
            for piece in _pretokenize(text):
                if not piece:
                    continue
                key = tuple(_to_bytes(piece))
                freq[key] = freq.get(key, 0) + 1

        if len(freq) > max_words:
            freq = dict(sorted(freq.items(), key=lambda kv: -kv[1])[:max_words])

        words: list[tuple[int, ...]] = list(freq.keys())
        counts: list[int] = list(freq.values())

        pair_counts: dict[tuple[int, int], int] = {}
        pair_index: dict[tuple[int, int], set[int]] = {}
        for idx, word in enumerate(words):
            c = counts[idx]
            for a, b in zip(word, word[1:]):
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + c
                pair_index.setdefault((a, b), set()).add(idx)

        heap = [(-c, pair) for pair, c in pair_counts.items() if c > 0]
        heapq.heapify(heap)

        self.merges = {}
        self._merge_order = []
        next_id = base
        for _ in range(n_merges):
            pair = None
            # Lazy deletion: an entry is stale once its recorded count no
            # longer matches the live count (because some other merge
            # touched it since it was pushed). Discarding stale entries here,
            # instead of eagerly updating every heap entry on every change,
            # is what keeps each merge close to O(affected words) rather than
            # O(distinct pairs).
            while heap:
                neg_c, cand = heap[0]
                if pair_counts.get(cand, 0) == -neg_c and -neg_c > 0:
                    pair = cand
                    break
                heapq.heappop(heap)
            if pair is None:
                break
            heapq.heappop(heap)

            a, b = pair
            self.id_to_bytes[next_id] = self.id_to_bytes[a] + self.id_to_bytes[b]
            self.merges[pair] = next_id
            self._merge_order.append(pair)

            touched: set[tuple[int, int]] = set()
            affected = pair_index.pop(pair, set())
            for idx in affected:
                word = words[idx]
                c = counts[idx]
                for x, y in zip(word, word[1:]):
                    pair_counts[(x, y)] = pair_counts.get((x, y), 0) - c
                    touched.add((x, y))
                new_word: list[int] = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and word[j] == a and word[j + 1] == b:
                        new_word.append(next_id)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                word = tuple(new_word)
                words[idx] = word
                for x, y in zip(word, word[1:]):
                    pair_counts[(x, y)] = pair_counts.get((x, y), 0) + c
                    touched.add((x, y))
                    pair_index.setdefault((x, y), set()).add(idx)

            for p in touched:
                if pair_counts.get(p, 0) > 0:
                    heapq.heappush(heap, (-pair_counts[p], p))

            next_id += 1

        self.vocab_size = next_id
        self._rank = {p: i for i, p in enumerate(self._merge_order)}
        return self

    # -- encode / decode -------------------------------------------------

    def _split_specials(self, text: str):
        if not self.special_tokens:
            yield False, text
            return
        pattern = re.compile(
            "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True))
        )
        pos = 0
        for m in pattern.finditer(text):
            if m.start() > pos:
                yield False, text[pos:m.start()]
            yield True, m.group(0)
            pos = m.end()
        if pos < len(text):
            yield False, text[pos:]

    def _encode_piece(self, piece: str) -> list[int]:
        ids: list[int] = list(_to_bytes(piece))
        if not self._rank or len(ids) < 2:
            return ids
        # Repeatedly apply whichever *learnable* adjacent pair was learned
        # earliest (lowest rank), same rule GPT-2 uses: earlier merges were
        # chosen because they were more frequent overall, so applying them
        # first is what reproduces the training-time merge order instead of
        # some arbitrary one that happens to produce a different, untrained
        # tokenization of the same string.
        while len(ids) >= 2:
            best_rank = None
            best_i = -1
            for i in range(len(ids) - 1):
                r = self._rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            a, b = ids[best_i], ids[best_i + 1]
            new_id = self.merges[(a, b)]
            ids = ids[:best_i] + [new_id] + ids[best_i + 2:]
        return ids

    def encode(self, text: str) -> list[int]:
        """Text to token ids. Never raises and never emits an out-of-vocabulary
        marker — see the module docstring for why that property is load-bearing."""
        ids: list[int] = []
        for is_special, chunk in self._split_specials(text):
            if is_special:
                ids.append(self.special_tokens[chunk])
                continue
            for piece in _pretokenize(chunk):
                if piece:
                    ids.extend(self._encode_piece(piece))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Token ids back to text. Concatenating each id's stored bytes and
        decoding once (rather than decoding id-by-id) is what makes a
        multi-byte UTF-8 character split across two merged tokens decode
        correctly instead of producing a mid-character replacement char."""
        out = bytearray()
        for i in ids:
            out += self.id_to_bytes.get(int(i), b"")
        return _from_bytes(bytes(out))

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """JSON, and deliberately small: only the special-token strings (in
        order — their ids are ``256 + position``) and the merge pairs (in
        learned order — their ids are ``256 + n_specials + position``) are
        stored. Everything else (``id_to_bytes``, ``vocab_size``, the rank
        table) is a deterministic function of those two lists and is
        rebuilt by :meth:`load`, rather than duplicated on disk where it
        could drift out of sync with the merges that actually produced it.
        """
        Path(path).write_text(json.dumps(self.to_dict()))

    def to_dict(self) -> dict:
        """The persisted form (see :meth:`save`): plain lists, so it can ride
        inside a checkpoint manifest that is loaded with ``weights_only``."""
        return {
            "version": 1,
            "special_tokens": list(self.special_tokens.keys()),
            "merges": [list(p) for p in self._merge_order],
        }

    @classmethod
    def load(cls, path: str | Path) -> "BytePairTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data: dict) -> "BytePairTokenizer":
        tok = cls()
        specials = data.get("special_tokens", [])
        tok.special_tokens = {s: N_BYTES + i for i, s in enumerate(specials)}
        base = N_BYTES + len(specials)
        tok.id_to_bytes = {i: bytes([i]) for i in range(N_BYTES)}
        for s, i in tok.special_tokens.items():
            tok.id_to_bytes[i] = _to_bytes(s)
        tok.merges = {}
        tok._merge_order = []
        next_id = base
        for a, b in data.get("merges", []):
            pair = (int(a), int(b))
            tok.merges[pair] = next_id
            tok._merge_order.append(pair)
            tok.id_to_bytes[next_id] = tok.id_to_bytes[pair[0]] + tok.id_to_bytes[pair[1]]
            next_id += 1
        tok.vocab_size = next_id
        tok._rank = {p: i for i, p in enumerate(tok._merge_order)}
        return tok


def byte_tokenizer() -> BytePairTokenizer:
    """The identity byte mapping: id == byte value, no merges, no specials.

    This is what every caller in this codebase gets today, and it is what a
    caller gets from :func:`tokenizer_for` when no trained artifact exists
    and none can be trained (no cached file, no network, ``datasets`` not
    installed) — a training run should degrade to byte-level text rather than
    hard-fail on tokenizer construction, since byte-level text is still a
    correct, if capacity-inefficient, way to train.
    """
    return BytePairTokenizer()


def train_from_sources(
    vocab_size: int,
    sources: Optional[dict[str, float]] = None,
    seed: int = 0,
    n_docs: int = 2000,
    special_tokens: Sequence[str] = (),
):
    """Train a tokenizer by streaming documents from ``iridium.data.text_corpus``.

    Reuses that module's own mixture machinery (:data:`SOURCES`,
    :data:`DEFAULT_MIX`) so the tokenizer is trained on the same licensed,
    provenance-tracked corpora the model itself trains on — training a
    subword vocabulary on a *different* distribution of text than the model
    sees (a generic web-scale BPE vocab, say) systematically under-serves
    whatever the model's own corpus emphasises (here: Gutenberg-heavy
    English prose and Wikipedia, not, say, code or forum chat).

    Requires ``datasets`` (the same dependency :mod:`iridium.data.text_corpus`
    already requires for anything beyond this module's own tests) and a
    network connection; callers that may run offline should go through
    :func:`tokenizer_for`, which catches exactly that failure.
    """
    from .text_corpus import DEFAULT_MIX, stream_documents

    mix = sources or DEFAULT_MIX
    total = sum(mix.values())
    docs: list[str] = []
    for i, (key, weight) in enumerate(sorted(mix.items())):
        quota = max(1, int(round(n_docs * weight / total)))
        docs.extend(stream_documents(key, limit=quota, seed=seed + i))
    tok = BytePairTokenizer()
    tok.train(docs, vocab_size=vocab_size, special_tokens=special_tokens)
    return tok


def tokenizer_for(
    vocab_size: int,
    cache_dir: str | Path,
    sources: Optional[dict[str, float]] = None,
    seed: int = 0,
    n_docs: int = 2000,
) -> BytePairTokenizer:
    """Train-and-cache: the tokenizer a training run should actually call.

    Deterministic in its cache key (``vocab_size``, ``seed``, ``n_docs``), so
    the same call always returns the same tokenizer without retraining, and a
    swept vocabulary size or seed gets its own artifact rather than silently
    overwriting a previous one mid-sweep.

    Never raises for a reason a training run should not die over: if training
    fails (no network, ``datasets`` not installed, the corpus is
    unreachable), this falls back to :func:`byte_tokenizer` rather than
    propagating the exception, so a notebook that cannot reach the Hub still
    trains — on byte-level text, which is strictly worse but never wrong.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"bpe_v{vocab_size}_s{seed}_n{n_docs}.json"
    if path.exists():
        return BytePairTokenizer.load(path)
    try:
        tok = train_from_sources(vocab_size=vocab_size, sources=sources, seed=seed, n_docs=n_docs)
    except Exception:
        return byte_tokenizer()
    tok.save(path)
    return tok


def compression_ratio(tokenizer: BytePairTokenizer, sample: str) -> float:
    """Bytes of UTF-8 text per token, on ``sample``. 1.0 is the byte-level
    floor (every tokenizer is at least this good, by construction); a
    trained English BPE tokenizer typically lands in the 3.5-4.5 range on
    English prose, which is the multiple of "text seen per training step"
    this module exists to buy back.
    """
    n_bytes = len(_to_bytes(sample))
    n_tokens = len(tokenizer.encode(sample))
    return n_bytes / max(n_tokens, 1)
