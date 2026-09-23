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
