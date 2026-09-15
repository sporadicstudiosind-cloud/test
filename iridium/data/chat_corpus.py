"""Conversations to train on, with their licences recorded.

A model only chats if something taught it the shape of a conversation. Byte
prediction over books and encyclopedia articles produces a model that continues
prose, which is a different skill: prompted with "what is a weir?" it writes the
*next paragraph of an article about weirs* rather than answering. The turn-taking
has to be in the data.

Two human-written, permissively licensed corpora:

* **Dolly 15k** (CC BY-SA 3.0) — single-turn instruction/response written by
  people, some with a supplied context passage.
* **OASST1** (Apache 2.0) — real multi-turn assistant conversations, released
  as a message *forest*: rows carry ``parent_id``, and a conversation is a root
  and one chain of replies. Reconstructed here rather than flattened, because
  multi-turn context is the part single-turn data cannot teach.

What this does not do is make the model's answers true. These corpora teach the
*form* of a reply. Everything this project says about grading — that a number
means nothing without the baseline beside it — applies with more force here,
because fluent conversational text is the most convincing possible disguise for
a model that knows nothing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterator, Optional

import numpy as np

from ..runtime.chat import Turn


@dataclass(frozen=True)
class ChatSourceSpec:
    key: str
    dataset: str
    config: Optional[str]
    split: str
    licence: str
    obligation: str
    description: str
    homepage: str = ""
    approx_rows: int = 10_000

    def as_dict(self) -> dict:
        return asdict(self)


CHAT_SOURCES: dict[str, ChatSourceSpec] = {
    "dolly": ChatSourceSpec(
        key="dolly",
        dataset="databricks/databricks-dolly-15k",
        config=None,
        split="train",
        licence="CC BY-SA 3.0",
        obligation=(
            "Attribution to Databricks and share-alike on derivative datasets. "
            "Recorded in the run manifest for exactly that reason."
        ),
        description=(
            "15k instruction/response pairs written by Databricks employees. "
            "Human-written throughout, which is why it is here: a corpus of one "
            "model's outputs would teach this model to imitate that model's "
            "mistakes, with no way to tell which are which."
        ),
        homepage="https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        approx_rows=15_000,
    ),
    "oasst": ChatSourceSpec(
        key="oasst",
        dataset="OpenAssistant/oasst1",
        config=None,
        split="train",
        licence="Apache 2.0",
        obligation="Retain the licence and attribution notices.",
        description=(
            "Crowd-sourced multi-turn assistant conversations, human written and "
            "human ranked. Filtered here to English, undeleted, review-passed "
            "messages, keeping the highest-ranked reply at each step."
        ),
        homepage="https://huggingface.co/datasets/OpenAssistant/oasst1",
        approx_rows=84_400,
    ),
}

DEFAULT_CHAT_MIX = {"dolly": 0.5, "oasst": 0.5}


def chat_licence_notice(keys=None) -> str:
    lines = ["Conversation corpora and their terms:", ""]
    for key in (keys or DEFAULT_CHAT_MIX):
        spec = CHAT_SOURCES[key]
        lines += [
            f"* {spec.dataset}",
            f"    licence:     {spec.licence}",
            f"    obligation:  {spec.obligation}",
            f"    homepage:    {spec.homepage}",
        ]
    lines += [
        "",
        "These teach the form of a reply, not the truth of one. A fluent answer",
        "from a model this size is still an unverified answer.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _load(spec: ChatSourceSpec, streaming: bool = True):
    try:
        from datasets import load_dataset
    except ImportError as exc:                          # pragma: no cover
        raise RuntimeError(
            "pip install datasets  — required for conversational training"
        ) from exc
    return load_dataset(spec.dataset, spec.config, split=spec.split,
                        streaming=streaming)


def dolly_conversations(limit: Optional[int] = None, seed: int = 0
                        ) -> Iterator[list[Turn]]:
    """Instruction (+ optional context) -> response, as a two-turn exchange."""
    spec = CHAT_SOURCES["dolly"]
    ds = _load(spec).shuffle(seed=seed, buffer_size=1000)
    produced = 0
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        context = (row.get("context") or "").strip()
        response = (row.get("response") or "").strip()
        if not instruction or not response:
            continue
        prompt = f"{instruction}\n\n{context}" if context else instruction
        yield [Turn("user", prompt), Turn("assistant", response)]
        produced += 1
        if limit is not None and produced >= limit:
            return


def oasst_conversations(limit: Optional[int] = None, seed: int = 0,
                        lang: str = "en", max_turns: int = 6,
                        ) -> Iterator[list[Turn]]:
    """Rebuild conversation threads from the message forest.

    The rows are a forest, not a list: each carries ``parent_id``, and a usable
    conversation is a root plus one chain of replies. Treating the rows as
    independent examples — which the flat schema invites — trains the model on
    replies whose questions it never saw.

    The whole split is materialised (about 40 MB of parquet) because a thread
    cannot be assembled from a stream that has not yet reached its children.
    """
    spec = CHAT_SOURCES["oasst"]
    rows = [
        r for r in _load(spec, streaming=False)
        if r.get("lang") == lang and not r.get("deleted")
        and (r.get("text") or "").strip()
    ]
    by_parent: dict[Optional[str], list[dict]] = {}
    for row in rows:
        by_parent.setdefault(row.get("parent_id"), []).append(row)

    def best(children: list[dict]) -> dict:
        # rank 0 is the top-ranked reply; unranked sorts last.
        return min(children, key=lambda r: (r.get("rank") is None, r.get("rank") or 0))

    roots = [r for r in rows if not r.get("parent_id") and r.get("role") == "prompter"]
    rng = np.random.default_rng(seed)
    rng.shuffle(roots)

    produced = 0
    for root in roots:
        turns: list[Turn] = []
        node = root
        while node is not None and len(turns) < max_turns:
            role = "user" if node.get("role") == "prompter" else "assistant"
            turns.append(Turn(role, (node["text"] or "").strip()))
            children = by_parent.get(node["message_id"])
            node = best(children) if children else None
        # A thread that ends on the user is an unanswered question: there is
        # nothing for the model to learn to say.
        while turns and turns[-1].role != "assistant":
            turns.pop()
        if len(turns) < 2:
            continue
        yield turns
        produced += 1
        if limit is not None and produced >= limit:
            return


CONVERSATION_LOADERS = {
    "dolly": dolly_conversations,
    "oasst": oasst_conversations,
}


# --------------------------------------------------------------------------
# training items
# --------------------------------------------------------------------------


def chat_items(
    n_items: int,
    mix: Optional[dict[str, float]] = None,
    seed: int = 0,
    max_bytes: int = 1024,
    split: str = "train",
):
    """Conversations as training items, supervised on the assistant only."""
    from ..runtime.chat import conversation_sample, fit_to_budget
    from ..training.tasks import Item
    from .text_corpus import in_split

    mix = mix or DEFAULT_CHAT_MIX
    total = sum(mix.values())
    items = []
    for i, (key, weight) in enumerate(sorted(mix.items())):
        quota = int(round(n_items * weight / total))
        if quota <= 0:
            continue
        got = 0
        # Generous: the split filter rejects most of what it sees for the
        # smaller splits, and a conversation is cheap to skip.
        budget = quota * 8 + 32
        for turns in CONVERSATION_LOADERS[key](limit=budget, seed=seed + i):
            joined = "\n".join(t.text for t in turns)
            if not in_split(joined, split):
                continue
            turns = fit_to_budget(turns, max_bytes)
            if len(turns) < 2 or turns[-1].role != "assistant":
                continue
            items.append(Item(
                sample=conversation_sample(turns, supervise_assistant=True,
                                           meta={"family": "chat", "source": key}),
                family="chat",
                prompt=turns[-2].text[:80],
                answer=turns[-1].text,
                truth={"source": key, "turns": len(turns)},
            ))
            got += 1
            if got >= quota:
                break

    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    return items[:n_items]
