"""Tests for sequence packing in BatchLoader, Muon parameter routing, and Preset fields."""

from dataclasses import fields

import torch
import torch.nn as nn

from iridium.config import get_config
from iridium.presets import Preset
from iridium.training.datasets import BatchLoader, Corpus, build_corpus
from iridium.training.muon import _native_bf16, muon_param_groups

MAX_LEN = 256
SUBJECTS = ("alpha", "beta", "gamma")


# Families whose payload widths fit the tiny codec. The default mixture also
# draws field_rollout and scene_goal/image items, which do not.
TINY_MIXTURE = {"arithmetic_drill": 1.0, "channel_depth": 1.0, "channel_intervention": 1.0,
                "code_exec": 1.0, "false_premise": 1.0, "unknowable": 1.0}


def _tagged_corpus(n=120, seed=0):
    """A corpus whose items carry varied subject tags (the generators leave them None)."""
    corpus = build_corpus(n, seed=seed, split="train", mixture=TINY_MIXTURE)
    # Items longer than MAX_LEN alone are truncated by collate in both modes, which
    # would break the row-length invariant for reasons unrelated to packing. Drop them.
    corpus = Corpus([it for it in corpus.items if len(it.sample) <= MAX_LEN], corpus.split)
    for i, item in enumerate(corpus.items):
        item.sample.meta = dict(item.sample.meta or {}, subject=SUBJECTS[i % len(SUBJECTS)])
    return corpus


def _loader(corpus, pack, batch_size=4, max_length=MAX_LEN, seed=3):
    return BatchLoader(corpus, get_config("tiny").codecs, batch_size=batch_size,
                       seed=seed, max_length=max_length, pack=pack)


def _epoch(loader):
    """One full epoch from position 0: list of (batch, items) pairs in yield order."""
    loader.position = 0
    return list(loader.batches())


def _split_rows(items, row_lengths):
    """Recover row membership: each row is a contiguous run of items summing to its length."""
    rows, cursor = [], 0
    for length in row_lengths:
        row, total = [], 0
        while total < length:
            assert cursor < len(items), "row lengths overrun the item list"
            row.append(items[cursor])
            total += len(items[cursor].sample)
            cursor += 1
        assert total == length, "row length does not match the items it holds"
        rows.append(row)
    assert cursor == len(items), "items left over after all rows"
    return rows


# --- 1. BatchLoader(pack=True) -------------------------------------------------

def test_packed_epoch_covers_every_item_exactly_once():
    corpus = _tagged_corpus()
    pairs = _epoch(_loader(corpus, pack=True))
    ids = [id(it) for _, items in pairs for it in items]
    assert len(ids) == len(corpus)
    assert len(set(ids)) == len(corpus), "an item was dropped or yielded twice in one epoch"
    assert set(ids) == {id(it) for it in corpus.items}


def test_packed_rows_fit_max_length_and_share_one_subject():
    corpus = _tagged_corpus()
    multi_item_rows = 0
    for batch, items in _epoch(_loader(corpus, pack=True)):
        row_tokens = batch.valid.sum(dim=1).tolist()
        assert batch.valid.shape[1] <= MAX_LEN
        assert max(row_tokens) <= MAX_LEN
        for row in _split_rows(items, row_tokens):
            subjects = {it.sample.meta.get("subject") for it in row}
            assert len(subjects) == 1, f"mixed subjects in one packed row: {subjects}"
            multi_item_rows += len(row) > 1
    assert multi_item_rows > 0, "no row held more than one item; packing did not happen"


def test_packing_reduces_padding_versus_unpacked():
    corpus = _tagged_corpus()

    def padding(pack):
        total = 0
        for batch, _ in _epoch(_loader(corpus, pack=pack)):
            total += int((~batch.valid).sum())
        return total

    assert padding(pack=True) < padding(pack=False)


def test_resume_mid_epoch_restarts_at_chunk_boundary():
    corpus = _tagged_corpus()
    loader = _loader(corpus, pack=True, batch_size=4)
    chunk = loader.PACK_CHUNK * loader.batch_size          # 32 items
    order = loader._epoch_order(0)

    loader.position = 50                                    # mid-chunk
    pairs = list(loader.batches())                          # must not raise
    yielded = [it for _, items in pairs for it in items]

    # Restart is rounded down to the chunk boundary (32), so the first item
    # is the chunk's first item and everything from there on is covered.
    assert yielded[0] is corpus.items[int(order[chunk])]
    assert {id(it) for it in yielded} == {id(corpus.items[int(i)]) for i in order[chunk:]}
    assert loader.position == len(corpus)


# --- 2. Muon routing ----------------------------------------------------------

def test_native_bf16_is_false_on_cpu():
    assert _native_bf16(torch.device("cpu")) is False


class _RoutedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 32)
        self.proj = nn.Linear(32, 32)            # hidden matrix: Muon
        self.router = nn.Linear(32, 4)           # excluded by name
        self.text_head = nn.Linear(32, 100)      # excluded by name
        self.narrow = nn.Linear(32, 8)           # excluded by min dim < 16


def test_muon_group_holds_only_large_hidden_matrices():
    model = _RoutedModel()
    groups = muon_param_groups(model)

    muon = [g for g in groups if g.get("use_muon")]
    assert len(muon) == 1
    assert [id(p) for p in muon[0]["params"]] == [id(model.proj.weight)]

    adamw_ids = {id(p) for g in groups if not g.get("use_muon") for p in g["params"]}
    for name in ("embedding.weight", "router.weight", "text_head.weight", "narrow.weight"):
        param = dict(model.named_parameters())[name]
        assert id(param) in adamw_ids, f"{name} was routed to Muon"


# --- 3. Preset fields ----------------------------------------------------------

def test_preset_pack_and_anneal_mixture_defaults():
    field_map = {f.name: f for f in fields(Preset)}
    assert field_map["pack"].default is False
    assert field_map["anneal_mixture"].default is None

    preset = Preset(name="t", priority=1, goal="g", config=get_config("tiny"),
                    mixture={"false_premise": 1.0}, steps=1, batch_size=1,
                    window=8, lr=1e-3)
    assert preset.pack is False
    assert preset.anneal_mixture is None
