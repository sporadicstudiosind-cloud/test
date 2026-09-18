"""Multi-turn conversation: the format, and a session that holds one.

**One definition, used by both sides.** The role markers and the turn layout
below are what the training data is built from *and* what generation conditions
on. A chat format that differs between training and inference by a single token
produces a model that answers fluently in training and emits noise in the chat
box, with nothing in the loss curve to indicate why — so
``data/chat_corpus.py`` imports this module rather than restating the layout.

The layout::

    BOS  USER "how deep is the channel"  ASSISTANT "about 1.5 m"  EOS
         USER "and if I double it"       ASSISTANT "…"            EOS

Only the assistant's bytes are supervised. The user's turns are context the
model must condition on and must never be scored for predicting — training on
them teaches the model to write the human's next message, which is the single
most common way a small chat model ends up interviewing itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..codecs.spans import Sample, Span, text_span

#: Role markers, in the control-token block reserved below ``TEXT_OFFSET``.
#: 1–6 are taken (BOS, EOS, SEP, PAD, VERDICT_TRUE, VERDICT_FALSE).
USER, ASSISTANT, SYSTEM = 7, 8, 9
ROLE_TOKEN = {"user": USER, "assistant": ASSISTANT, "system": SYSTEM, "tool": 10}
TOKEN_ROLE = {v: k for k, v in ROLE_TOKEN.items()}

BOS, EOS = 1, 2
TEXT_OFFSET = 16

#: Generation stops on either: the assistant ending its turn, or the assistant
#: starting a *user* turn — a small model does the latter constantly, and
#: without this the transcript fills with invented dialogue.
STOP_IDS = (EOS, USER, SYSTEM)


@dataclass
class Turn:
    role: str
    text: str

    def __post_init__(self) -> None:
        if self.role not in ROLE_TOKEN:
            raise ValueError(f"unknown role {self.role!r}; known: {sorted(ROLE_TOKEN)}")


def _control(code: int, supervised: bool = False) -> Span:
    import numpy as np
    return Span("control", np.array([code], dtype=np.int64), supervised=supervised)


def conversation_spans(
    turns: Sequence[Turn],
    supervise_assistant: bool = True,
    open_for_reply: bool = False,
) -> list[Span]:
    """The exact span layout, for training and for inference alike.

    ``open_for_reply`` appends a trailing ASSISTANT marker and nothing after
    it, which is how generation is asked for a reply: the model is positioned
    exactly where its own bytes begin during training.
    """
    spans: list[Span] = [_control(BOS)]
    for turn in turns:
        spans.append(_control(ROLE_TOKEN[turn.role]))
        supervised = supervise_assistant and turn.role == "assistant"
        if turn.text:
            spans.append(text_span(turn.text, supervised=supervised,
                                   offset=TEXT_OFFSET))
        if turn.role == "assistant":
            spans.append(_control(EOS, supervised=supervised))
    if open_for_reply:
        spans.append(_control(ASSISTANT))
    return spans


def conversation_sample(
    turns: Sequence[Turn],
    supervise_assistant: bool = True,
    open_for_reply: bool = False,
    meta: Optional[dict] = None,
) -> Sample:
    return Sample(
        conversation_spans(turns, supervise_assistant, open_for_reply),
        meta=meta or {"family": "chat"},
    )


def fit_to_budget(turns: Sequence[Turn], max_bytes: int) -> list[Turn]:
    """Drop whole turns from the front until the history fits.

    Whole turns, not truncated bytes: a half-sentence of someone else's message
    is worse context than no message, and cutting mid-turn leaves a role marker
    describing text that is no longer there.
    """
    kept: list[Turn] = []
    total = 0
    for turn in reversed(turns):
        cost = len(turn.text.encode("utf-8")) + 2
        if total + cost > max_bytes and kept:
            break
        kept.append(turn)
        total += cost
    return list(reversed(kept))


@dataclass
class ChatSession:
    """A conversation with one model, held across turns.

    Sampling defaults are chosen for a *small* model. Greedy decoding on an
    undertrained network loops within a sentence or two; temperature alone
    makes it incoherent. Nucleus sampling with a repetition penalty is what
    keeps a model this size producing something readable, and it is still the
    training that decides whether the text means anything.
    """

    model: object
    system: Optional[str] = None
    temperature: float = 0.8
    top_p: float = 0.92
    top_k: int = 0
    repetition_penalty: float = 1.15
    max_new_tokens: int = 160
    n_loops: Optional[int] = None
    seed: int = 0
    turns: list[Turn] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.system:
            self.turns.insert(0, Turn("system", self.system))

    # -- history -----------------------------------------------------------

    def reset(self) -> None:
        self.turns = [Turn("system", self.system)] if self.system else []

    def history(self) -> list[tuple[str, str]]:
        return [(t.role, t.text) for t in self.turns]

    def _budget(self) -> int:
        cfg = self.model.cfg
        # Leave room for the reply and the role markers around it.
        return max(64, cfg.max_seq_len - self.max_new_tokens - 8)

    # -- the actual exchange ----------------------------------------------

    def send(self, message: str, **overrides) -> str:
        """Append a user message, generate a reply, append and return it."""
        from .generate import generate

        self.turns.append(Turn("user", message))
        context = fit_to_budget(self.turns, self._budget())
        sample = conversation_sample(context, supervise_assistant=False,
                                     open_for_reply=True)

        params = dict(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stop_ids=STOP_IDS,
            n_loops=self.n_loops,
            seed=self.seed,
            text_offset=TEXT_OFFSET,
            text_only=True,
        )
        params.update(overrides)
        reply = generate(self.model, sample, **params).text.strip()

        if not reply:
            # An untrained or barely-trained model ends its turn immediately.
            # Printing a blank line looks like a bug in the notebook; saying so
            # points at the actual cause, which is the training, not the chat.
            reply = ("(no output — the model ended its turn immediately. That is "
                     "what an undertrained model does; train for more steps, or "
                     "raise CHAT_WEIGHT so more of the corpus is conversation.)")
            self.turns.pop()    # don't poison the history with the explanation
            return reply

        self.seed += 1          # so a repeated question is not a repeated answer
        self.turns.append(Turn("assistant", reply))
        return reply

    def transcript(self) -> str:
        lines = []
        for turn in self.turns:
            if turn.role == "system":
                continue
            who = "you" if turn.role == "user" else "iridium"
            lines.append(f"{who}: {turn.text}")
        return "\n".join(lines)
