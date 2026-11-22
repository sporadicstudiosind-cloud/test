"""Faster data preparation: batched chat tokenizing and parallel families."""
import numpy as np

from iridium.data import chat_corpus
from iridium.data.chat_corpus import _blocks, _Prefetched
from iridium.data.tokenization import FastBPETokenizer
from iridium.training import prepare as prep


def _tokenizer():
    texts = ["Once upon a time a fox met a crow. " * 20, "def add(a, b):\n    return a + b\n" * 20]
    return FastBPETokenizer.train(texts, vocab_size=300)


def test_prefetched_ids_equal_direct_encoding():
    tok = _tokenizer()
    texts = ["a fox met a crow", "", "def add(a, b): return a + b", "a fox met a crow"]
    fast = _Prefetched(tok, texts)
    for t in texts + ["a text outside the batch"]:
        assert fast.encode(t) == tok.encode(t)
    assert fast.vocab_size == tok.vocab_size
    assert _Prefetched(None, texts) is None


def test_blocks_keep_order_and_remainder():
    assert list(_blocks(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]


def test_batched_chat_items_match_unbatched(monkeypatch):
    from iridium.runtime.chat import Turn
    convs = [[Turn("user", f"Tell me story {i}."), Turn("assistant", f"A fox met crow {i}. " * (i % 5 + 1))]
             for i in range(600)]
    monkeypatch.setitem(chat_corpus.CONVERSATION_LOADERS, "story_requests",
                        lambda limit=None, seed=0: iter(convs))
    tok = _tokenizer()

    def ids(block):
        monkeypatch.setattr(chat_corpus, "_ENCODE_BLOCK", block)
        items = chat_corpus.iter_chat_items({"story_requests": 1.0}, tokenizer=tok, split="train")
        return [np.concatenate([s.payload for s in it.sample.spans]) for it in items]

    one, many = ids(1), ids(256)
    assert len(one) == len(many) > 0
    assert all(np.array_equal(a, b) for a, b in zip(one, many))


def _fake_family(family, split, need, **ctx):
    return {"family": family, "split": split, "items": need, "pid_ok": True}


def test_parallel_jobs_return_every_family(monkeypatch):
    monkeypatch.setattr(prep, "_write_family", _fake_family)
    jobs = [("a", "train", 5), ("b", "train", 9), ("a", "test", 1)]
    serial = prep._run_jobs(jobs, {}, workers=1)
    assert set(serial) == {("a", "train"), ("b", "train"), ("a", "test")}
    assert prep.default_workers() >= 1
