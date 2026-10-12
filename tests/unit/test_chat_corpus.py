"""Offline checks for exact chat-source allocation and assistant targets."""

from iridium.data import chat_corpus, text_corpus
from iridium.runtime.chat import Turn


def test_one_chat_item_is_not_lost_to_rounding(monkeypatch):
    def conversations(**_kwargs):
        yield [Turn("user", "hello"), Turn("assistant", "hi")]

    monkeypatch.setattr(chat_corpus, "CONVERSATION_LOADERS", {
        "dolly": conversations, "oasst": conversations,
    })
    monkeypatch.setattr(text_corpus, "in_split", lambda *_args: True)

    items = chat_corpus.chat_items(
        1, mix={"dolly": 0.5, "oasst": 0.5}, max_bytes=128,
    )

    assert len(items) == 1
    assert items[0].truth["source"] == "dolly"
    assert items[0].sample.spans[-2].supervised
    assert items[0].sample.spans[-1].supervised


class _FakeStream(list):
    def shuffle(self, **_kwargs):
        return self


def test_messages_loader_maps_roles_and_drops_unanswered_tails(monkeypatch):
    rows = _FakeStream([
        {"messages": [{"role": "system", "content": "be brief"},
                      {"role": "user", "content": "2+2?"},
                      {"role": "assistant", "content": "4"},
                      {"role": "user", "content": "and 3+3?"}]},
        {"messages": [{"role": "tool", "content": "x"}, {"role": "assistant", "content": "y"}]},
        {"messages": [{"role": "user", "content": "only a question"}]},
    ])
    monkeypatch.setattr(chat_corpus, "_load", lambda spec, streaming=True: rows)
    got = list(chat_corpus.CONVERSATION_LOADERS["smol_smoltalk"](limit=5))
    assert [[(t.role, t.text) for t in c] for c in got] == [
        [("system", "be brief"), ("user", "2+2?"), ("assistant", "4")]]


def test_openr1_uses_solutions_not_traces(monkeypatch):
    rows = _FakeStream([{"problem": "Find x.", "solution": "x = 2.",
                         "generations": ["<think>" + "a" * 10_000]}])
    monkeypatch.setattr(chat_corpus, "_load", lambda spec, streaming=True: rows)
    (conv,) = chat_corpus.CONVERSATION_LOADERS["openr1_math"](limit=1)
    assert conv[-1].text == "x = 2."


def test_default_mix_is_commercially_usable_and_opt_ins_are_marked():
    for key in chat_corpus.DEFAULT_CHAT_MIX:
        spec = chat_corpus.CHAT_SOURCES[key]
        assert spec.commercial_ok and not spec.opt_in, key
    assert set(chat_corpus.MAX_CHAT_MIX) == set(chat_corpus.CHAT_SOURCES)
    assert all(not chat_corpus.CHAT_SOURCES[k].commercial_ok
               for k in chat_corpus.CHAT_SOURCES if chat_corpus.CHAT_SOURCES[k].opt_in)
    assert set(chat_corpus.CONVERSATION_LOADERS) == set(chat_corpus.CHAT_SOURCES)


def test_an_underfilling_source_is_topped_up_from_the_rest(monkeypatch):
    def too_long(**_kwargs):
        for _ in range(50):
            yield [Turn("user", "q"), Turn("assistant", "x" * 5000)]

    def plenty(limit=None, seed=0):
        for i in range(limit or 100):
            yield [Turn("user", f"question {seed}-{i}"), Turn("assistant", "short")]

    monkeypatch.setattr(chat_corpus, "CONVERSATION_LOADERS", {"a": too_long, "b": plenty})
    monkeypatch.setattr(text_corpus, "in_split", lambda *_args: True)
    items = chat_corpus.chat_items(10, mix={"a": 0.5, "b": 0.5}, max_bytes=256)
    assert len(items) == 10
    assert {it.truth["source"] for it in items} == {"b"}
    assert len({it.prompt for it in items}) == 10          # no duplicates
