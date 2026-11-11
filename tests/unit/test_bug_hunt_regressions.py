"""Regressions for bugs found by the repository-wide audit."""

import json

import numpy as np
import torch

from iridium.config import get_config
from iridium.data.agent_corpus import held_out_arithmetic, is_test_pair
from iridium.model.iridium1 import Iridium1
from iridium.runtime.chat import Turn
from iridium.runtime.tools import _fit_agent_context, run_tool_loop
from iridium.training.datasets import BatchLoader, build_corpus
from iridium.training.tasks import _KEYS, UNKNOWN, unknowable_item


def test_loader_continues_one_permutation_across_calls():
    corpus = build_corpus(40, seed=0, split="train", mixture={"false_premise": 1.0})
    loader = BatchLoader(corpus, get_config("tiny").codecs, batch_size=4, seed=3)
    seen = []
    for _ in range(10):                          # ten "rounds" of one batch each
        batch, items = next(iter(loader.batches()))
        seen += [id(it) for it in items]
    assert len(set(seen)) == 40, "a new batches() call reshuffled instead of continuing"
    again = BatchLoader(corpus, get_config("tiny").codecs, batch_size=4, seed=3)
    again.position = 12                          # what a resume sets
    _, items = next(iter(again.batches()))
    assert [id(it) for it in items] == seen[12:16]


def test_agent_episodes_never_use_held_out_sums():
    from iridium.data import agent_corpus
    rng = np.random.default_rng(0)
    for _ in range(3000):
        a, b = agent_corpus._operands(rng, "*", 2, 200)
        assert not is_test_pair("*", a, b)
    assert held_out_arithmetic(5)


def test_unknowable_uses_the_chat_layout_and_disjoint_keys():
    item = unknowable_item(np.random.default_rng(1))
    ids = [int(s.payload[0]) for s in item.sample.spans if s.modality == "control"]
    assert ids[:2] == [1, 7] and 8 in ids             # BOS USER ... ASSISTANT
    rng = np.random.default_rng(0)
    train = {it.truth["key"] for it in (unknowable_item(rng, "train") for _ in range(300))}
    test = {it.truth["key"] for it in (unknowable_item(rng, "test") for _ in range(300))}
    assert not train & test
    assert set(_KEYS) >= train | test


def test_agent_context_keeps_the_request_when_a_result_is_huge():
    turns = [Turn("user", "What does notes.txt say?"), Turn("tool_call", '{"name":"read_file"}'),
             Turn("tool_result", "x" * 5000)]
    kept = _fit_agent_context(turns, 1500)
    assert [t.role for t in kept] == ["user", "tool_call", "tool_result"]
    assert sum(len(t.text) for t in kept) < 1500


def test_tool_loop_answers_after_its_last_allowed_call(tmp_path):
    from iridium.runtime.agent_tools import build_registry
    registry = build_registry(tmp_path, groups=("files",))
    model = Iridium1(get_config("tiny"))
    replies = iter(["CALL", "the folder is empty"])

    class Out:
        def __init__(self, text):
            self.text, self.ids = text, []

    def fake(model, sample, **kw):
        return Out(next(replies))

    import iridium.runtime.tools as tools
    original = tools.extract_tool_call
    tools.extract_tool_call = lambda ids, *a, **k: (
        json.dumps({"name": "list_dir", "arguments": {"path": "."}}) if fake.calls == 1 else None)
    fake.calls = 0

    def counted(model, sample, **kw):
        out = fake(model, sample, **kw)
        fake.calls += 1
        return out
    try:
        result = run_tool_loop(model, [Turn("user", "list files")], registry, max_calls=1,
                               generate_fn=counted)
    finally:
        tools.extract_tool_call = original
    assert result.stopped == "answer" and len(result.calls) == 1
