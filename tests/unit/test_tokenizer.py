"""The tokenizer's one hard requirement is losslessness — see the module
docstring in ``iridium/data/tokenizer.py`` for why a byte-level fallback
that never emits UNK is the whole point of building this instead of taking
a dependency. These tests exist to pin that property under adversarial
input, not just on well-behaved English prose, plus the vocabulary-size and
persistence contracts callers depend on.
"""

import json
import random
import string

import pytest

from iridium.data.tokenizer import (
    BytePairTokenizer,
    byte_tokenizer,
    compression_ratio,
    train_from_sources,
)

WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and",
    "runs", "away", "into", "forest", "where", "many", "other", "animals",
    "live", "peacefully", "together", "in", "harmony", "with", "nature",
]


def _corpus(n_docs=400, seed=0):
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        n = rng.randint(15, 50)
        docs.append(" ".join(rng.choice(WORDS) for _ in range(n)) + ".")
    return docs


@pytest.fixture(scope="module")
def trained():
    tok = BytePairTokenizer()
    tok.train(_corpus(), vocab_size=600, special_tokens=["<|endoftext|>"])
    return tok


# ---------------------------------------------------------------------------
# round-trip / losslessness
# ---------------------------------------------------------------------------

UNICODE_SAMPLES = [
    "hello world",
    "",
    "café naïve résumé",
    "日本語です",          # Japanese
    "\U0001F600\U0001F601\U0001F602 emoji stack",
    "under_score and snake_case_identifiers",
    "digits 0123456789 and arithmetic 42+17=59",
    "  leading and trailing whitespace   \n\t",
    "mixed\ttabs\nand\nnewlines",
    "punctuation!!! -- ??? ... :;,.",
]


@pytest.mark.parametrize("text", UNICODE_SAMPLES)
def test_round_trips_arbitrary_unicode(trained, text):
    assert trained.decode(trained.encode(text)) == text


def test_round_trips_arbitrary_bytes(trained):
    """Bytes that are not valid UTF-8 at all — the case a naive tokenizer
    turns into UNK or a mangled replacement character. Represented as a
    ``str`` via ``surrogateescape`` (the same trick :func:`os.fsdecode` uses),
    which is the one encoding that makes this a true round trip."""
    raw = bytes(range(256)) + b"\xff\xfe\xfd\x00\x01\x02 mixed with text \x80\x81\xc0"
    s = raw.decode("utf-8", "surrogateescape")
    ids = trained.encode(s)
    out = trained.decode(ids)
    assert out == s
    assert out.encode("utf-8", "surrogateescape") == raw


def test_round_trips_on_the_untrained_identity_tokenizer():
    """With no merges at all, this must reduce to the plain byte mapping the
    rest of the codebase already relies on."""
    tok = byte_tokenizer()
    for text in UNICODE_SAMPLES:
        assert tok.decode(tok.encode(text)) == text
    assert tok.encode("abc") == [97, 98, 99]
    assert tok.vocab_size == 256


def test_never_produces_an_id_outside_the_trained_vocabulary(trained):
    for text in UNICODE_SAMPLES:
        for i in trained.encode(text):
            assert 0 <= i < trained.vocab_size


def test_random_byte_soup_round_trips(trained):
    """Fuzz a bit: text that shares essentially no vocabulary with training
    data (the corpus above is English words; this is not)."""
    rng = random.Random(3)
    for _ in range(20):
        raw = bytes(rng.randrange(256) for _ in range(rng.randint(0, 200)))
        s = raw.decode("utf-8", "surrogateescape")
        assert trained.decode(trained.encode(s)) == s


# ---------------------------------------------------------------------------
# vocabulary size
# ---------------------------------------------------------------------------

def test_vocab_size_is_respected():
    tok = BytePairTokenizer()
    tok.train(_corpus(600), vocab_size=1000)
    assert tok.vocab_size <= 1000


def test_vocab_size_grows_with_more_merges_requested():
    small = BytePairTokenizer()
    small.train(_corpus(600), vocab_size=300)
    big = BytePairTokenizer()
    big.train(_corpus(600), vocab_size=900)
    assert big.vocab_size > small.vocab_size


def test_vocab_size_at_or_below_the_byte_floor_trains_zero_merges():
    """A caller sweeping vocab sizes should not have to special-case the
    smallest one: asking for no more than the reserved ids is valid, not an
    error, and simply yields a pure byte(+specials) tokenizer."""
    tok = BytePairTokenizer()
    tok.train(_corpus(50), vocab_size=256, special_tokens=["<pad>"])
    assert tok.vocab_size == 257
    assert tok.merges == {}


def test_special_tokens_get_their_own_ids_and_are_atomic():
    tok = BytePairTokenizer()
    tok.train(_corpus(50), vocab_size=400, special_tokens=["<|endoftext|>", "<pad>"])
    ids = tok.encode("hello<|endoftext|>world<pad>")
    assert tok.special_tokens["<|endoftext|>"] in ids
    assert tok.special_tokens["<pad>"] in ids
    assert tok.decode(ids) == "hello<|endoftext|>world<pad>"


# ---------------------------------------------------------------------------
# compression
# ---------------------------------------------------------------------------

def test_compression_ratio_beats_the_byte_floor(trained):
    sample = " ".join(_corpus(50, seed=99))
    ratio = compression_ratio(trained, sample)
    assert ratio > 1.5, f"trained tokenizer should beat 1 byte/token, got {ratio}"


def test_compression_ratio_is_exactly_one_on_the_byte_identity_tokenizer():
    tok = byte_tokenizer()
    assert compression_ratio(tok, "any text at all") == 1.0


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def test_save_load_round_trips_encoding(trained, tmp_path):
    path = tmp_path / "tok.json"
    trained.save(path)
    loaded = BytePairTokenizer.load(path)
    sample = " ".join(_corpus(20, seed=7))
    assert loaded.encode(sample) == trained.encode(sample)
    assert loaded.vocab_size == trained.vocab_size


def test_saved_artifact_is_small_json(trained, tmp_path):
    path = tmp_path / "tok.json"
    trained.save(path)
    data = json.loads(path.read_text())
    assert "merges" in data and "special_tokens" in data
    # id_to_bytes / vocab_size / rank table are NOT stored — they are
    # recomputed by load(), so the file holds only the two lists that
    # actually determine the tokenizer.
    assert set(data) <= {"version", "special_tokens", "merges"}


# ---------------------------------------------------------------------------
# train_from_sources: cannot hit the network in this environment, so this
# only pins that it fails the way tokenizer_for expects (an exception, not a
# hang or a silent empty tokenizer), letting tokenizer_for's fallback do its
# job. A real compression-ratio measurement against the licensed corpora
# needs IRIDIUM_NETWORK_TESTS=1 and is not exercised here.
# ---------------------------------------------------------------------------

def _offline(monkeypatch):
    """Make the corpus unreachable whatever the host has installed."""
    import sys
    monkeypatch.setitem(sys.modules, "datasets", None)


def test_train_from_sources_raises_without_datasets_or_network(monkeypatch):
    _offline(monkeypatch)
    with pytest.raises(Exception):
        train_from_sources(vocab_size=300, n_docs=5)


def test_tokenizer_for_falls_back_to_byte_tokenizer_offline(tmp_path, monkeypatch):
    from iridium.data.tokenizer import tokenizer_for
    _offline(monkeypatch)
    tok = tokenizer_for(vocab_size=300, cache_dir=tmp_path)
    assert tok.vocab_size == 256  # fell back to byte_tokenizer()
    assert tok.decode(tok.encode("hello")) == "hello"
