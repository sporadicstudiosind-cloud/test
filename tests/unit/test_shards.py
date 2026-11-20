"""Disk shards: items round-trip exactly, training reads them by memory map."""

import dataclasses

import numpy as np

from iridium.codecs.spans import Sample, Span
from iridium.data.shards import MixedCorpus, Shard, write_items
from iridium.runtime.chat import Turn, conversation_sample


def _conv(i):
    return conversation_sample([Turn("user", f"hi {i}"), Turn("assistant", f"hello {i}")])


def test_items_round_trip_ids_and_supervision(tmp_path):
    samples = [_conv(i) for i in range(5)]
    meta = write_items(tmp_path / "chat-train", "chat", 400, samples)
    assert meta["items"] == 5 and meta["dtype"] == "uint16"
    shard = Shard(tmp_path / "chat-train")
    for i, original in enumerate(samples):
        back = shard.sample(i)
        want_ids = np.concatenate([s.payload for s in original.spans])
        got_ids = np.concatenate([s.payload for s in back.spans])
        assert (want_ids == got_ids).all()
        want_mask = np.concatenate([[s.supervised] * len(s.payload) for s in original.spans])
        got_mask = np.concatenate([[s.supervised] * len(s.payload) for s in back.spans])
        assert (want_mask == got_mask).all()


def test_shard_refuses_ids_outside_the_vocabulary(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        write_items(tmp_path / "x", "chat", 100, [_conv(0)])


def test_mixed_corpus_is_lazy_deterministic_and_batches(tmp_path):
    from iridium.config import get_config
    from iridium.training.datasets import BatchLoader

    write_items(tmp_path / "chat-train", "chat", 400, [_conv(i) for i in range(3)])
    corpus = MixedCorpus.build({"chat": 10, "channel_depth": 6}, tmp_path, seed=3)
    assert len(corpus) == 16 and corpus.counts() == {"chat": 10, "channel_depth": 6}
    a = corpus.items[12].sample.spans[0].payload
    again = MixedCorpus.build({"chat": 10, "channel_depth": 6}, tmp_path, seed=3)
    assert np.array_equal(a, again.items[12].sample.spans[0].payload)
    loader = BatchLoader(corpus, get_config("tiny").codecs, batch_size=4)
    batch, items = next(loader.batches())
    assert len(items) == 4


def test_prepare_then_train_from_shards_offline(tmp_path, monkeypatch):
    from iridium.config import get_config
    from iridium.data import chat_corpus, text_corpus
    from iridium.data.tokenization import ByteTokenizer
    from iridium.presets import get_preset
    from iridium.training import prepare as prep, run_preset, tokenizer_bridge

    monkeypatch.setattr(tokenizer_bridge, "tokenizer_for_config", lambda cfg: ByteTokenizer())

    def fake_text(mix, window, tokenizer, seed=0, split="train", limit=None):
        for i in range(10_000):
            ids = np.full(window, 16 + (i % 200), np.int64)
            ids[0] = 1
            yield ids, np.r_[0, np.ones(window - 1)].astype(np.uint8)

    def fake_chat(mix, **kw):
        from iridium.training.tasks import Item
        for i in range(10_000):
            yield Item(sample=_conv(i), family="chat")

    monkeypatch.setattr(text_corpus, "iter_text_windows", fake_text)
    monkeypatch.setattr(chat_corpus, "iter_chat_items", fake_chat)
    cfg = get_config("tiny")
    preset = dataclasses.replace(get_preset("chat-34m"), config=cfg,
                                 mixture={"text_lm": 0.5, "chat": 0.3, "false_premise": 0.2},
                                 steps=3, batch_size=4, window=64, rounds=1, micro_batch=4)
    monkeypatch.setattr(run_preset, "get_preset", lambda name: preset, raising=False)
    out = prep.prepare(preset, tmp_path / "data")
    assert (out / "text_lm-train.bin").exists() and (out / "chat-train.idx").exists()
    assert (out / "text_lm-test.bin").exists() and (out / "chat-test.bin").exists()
    preset = dataclasses.replace(preset, steps=4)
    final = run_preset.train_preset(preset, device="cpu", out=str(tmp_path / "runs"),
                                    data=str(tmp_path / "data"))
    assert final.exists()
    import torch
    history = torch.load(final, weights_only=False)["history"]
    evals = [h["eval"] for h in history if "eval" in h]
    assert evals and "val_text_lm_bpb" in evals[-1] and "val_chat_loss" in evals[-1]


def test_packing_fills_windows_and_drops_duplicates(tmp_path):
    from iridium.data.shards import PackingWriter, ShardWriter

    inner = ShardWriter(tmp_path / "chat-train", "chat", 400)
    w = PackingWriter(inner, window=64)
    for i in range(30):
        w.add(_conv(i % 20))              # 10 exact repeats
    meta = w.close()
    assert meta["duplicates_dropped"] == 10 and meta["packed_items"] == 20
    shard = Shard(tmp_path / "chat-train")
    lengths = np.diff(shard.offsets)
    assert len(shard) < 20 and lengths.max() <= 64
