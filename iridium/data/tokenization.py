"""One tokenizer interface for every text path.

Before 1.0 the codebase had two text encodings that met by accident: raw UTF-8
bytes (``tokenizer=None``, the original design) and a subword BPE bolted on
later. Every function that touched text had to be taught about the second, and
each place that was missed failed silently -- a model read through the wrong
vocabulary produces fluent nonsense, not an error.

Now there is one protocol, and byte level is simply the smallest tokenizer:

* ``encode(str) -> list[int]``, ``encode_batch(list[str]) -> list[list[int]]``
* ``decode(ids) -> str``
* ``vocab_size``
* ``to_dict()`` -- a plain-JSON state stored *inside* every checkpoint, so a
  model always travels with the exact vocabulary it was trained on
* ``kind`` -- ``"byte"``, ``"hf-bpe"`` or ``"bpe"``

Three implementations:

* :class:`ByteTokenizer` -- id == byte value. Identical to the historical
  ``tokenizer=None`` path, which remains accepted everywhere and means this.
* :class:`FastBPETokenizer` -- byte-level BPE on Hugging Face ``tokenizers``
  (Rust). Trains on hundreds of thousands of documents in seconds and encodes
  batches across all cores; the default whenever the package is installed.
* :class:`iridium.data.tokenizer.BytePairTokenizer` -- the dependency-free
  pure-Python BPE, kept as the fallback. Correct, and roughly two orders of
  magnitude slower, which is why it is no longer the default.

:func:`from_state` rebuilds any of them from ``to_dict()`` output, and
:func:`check_fits` refuses a tokenizer whose ids would not fit the model's
embedding table.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

__all__ = ["ByteTokenizer", "FastBPETokenizer", "fast_available", "from_state",
           "check_fits", "as_tokenizer"]


class ByteTokenizer:
    """UTF-8 bytes as ids. Lossless for any string; 256 ids."""

    kind = "byte"
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", "surrogateescape"))

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: Iterable[int]) -> str:
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {"kind": "byte"}


def fast_available() -> bool:
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return True


class FastBPETokenizer:
    """Byte-level BPE via Hugging Face ``tokenizers``.

    Byte level means no unknown token is possible: the 256 byte symbols are
    the initial alphabet and every merge is built on top of them, so any
    string -- any language, emoji, code, stray binary -- round-trips.
    No special tokens: Iridium's control tokens (BOS, roles, tool markers)
    live *below* ``TEXT_ID_OFFSET`` in model-id space, outside the
    tokenizer, where no user text can ever produce them.
    """

    kind = "hf-bpe"

    def __init__(self, backend) -> None:
        self._tok = backend

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> "FastBPETokenizer":
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=2, show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[])
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(tok)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch(list(texts))]

    def decode(self, ids: Iterable[int]) -> str:
        return self._tok.decode([int(i) for i in ids])

    def to_dict(self) -> dict:
        return {"kind": self.kind, "json": self._tok.to_str()}

    @classmethod
    def from_dict(cls, data: dict) -> "FastBPETokenizer":
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_str(data["json"]))


def from_state(state: Optional[dict]):
    """Rebuild a tokenizer from ``to_dict()`` output (``None`` -> bytes)."""
    if not state:
        return ByteTokenizer()
    kind = state.get("kind", "bpe")       # pre-1.0 BytePairTokenizer dicts had no kind
    if kind == "byte":
        return ByteTokenizer()
    if kind == "hf-bpe":
        return FastBPETokenizer.from_dict(state)
    if kind == "bpe":
        from .tokenizer import BytePairTokenizer
        return BytePairTokenizer.from_dict(state)
    raise ValueError(f"unknown tokenizer kind {kind!r}")


def as_tokenizer(tokenizer):
    """``None`` is the byte tokenizer; anything else is returned as is."""
    return ByteTokenizer() if tokenizer is None else tokenizer


def check_fits(tokenizer, cfg) -> None:
    """Refuse a tokenizer whose ids overflow the model's embedding table.

    Model ids are ``TEXT_ID_OFFSET + token id``; the table has
    ``cfg.codecs.vocab_size`` rows. A tokenizer *smaller* than the table
    leaves rows that never train, which is waste; one *larger* indexes past
    the table, which is a crash on GPU and garbage on some backends.
    """
    from ..config import TEXT_ID_OFFSET
    need = as_tokenizer(tokenizer).vocab_size + TEXT_ID_OFFSET
    have = cfg.codecs.vocab_size
    if need > have:
        raise ValueError(f"tokenizer needs {need} embedding rows (vocab "
                         f"{need - TEXT_ID_OFFSET} + {TEXT_ID_OFFSET} control ids) but "
                         f"{cfg.name} has {have}")
