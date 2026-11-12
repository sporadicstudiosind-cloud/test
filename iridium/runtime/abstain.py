"""Answer, escalate, or say "I don't know" -- never guess.

A hallucination is a confident-sounding answer the model had no basis for.
The cure is not a refusal filter bolted on after generation; it is asking,
before committing to an answer, whether the model actually has one, and
spending every resource it has before admitting it does not. The ladder, in
order of cost:

1. **Answer** at the requested effort.
2. **Think harder** -- every ponder loop, no early halting.
3. **Widen routing** -- send each token through more superstacks (``top_k``
   raised, up to all of them): more specialist capacity on the same question.
4. **Look it up** -- if a search tool is available, retrieve and answer from
   the result (a ``tool_result`` turn, never supervised, never trusted as the
   model's own knowledge).
5. **Abstain** -- "I don't know", with the reason (what was tried).

"Does it have an answer" is measured, not asked: an answer is accepted when

* the model's own ``UNKNOWN`` token (trained by the ``unknowable`` family) is
  not what it chose, **and**
* its **confidence** clears a threshold: the geometric-mean probability the
  model assigns to its own answer, re-scored teacher-forced, **and**
* its **last two ponder loops agree** on how the answer starts -- a model
  whose conclusion flips between passes has not converged on one.

All three are cheap (one extra forward), and all three come from the model
itself, so they work offline. They are only as calibrated as training makes
them: the ``unknowable`` and ``false_premise`` families exist so that low
confidence and ``UNKNOWN`` mean something. The threshold is a dial between
"answers more, wrong more" and "abstains more"; tune it on held-out data.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import torch

from ..codecs.bank import TensorBatch, continuous_dims
from ..codecs.spans import Sample, collate, text_span
from .chat import STOP_IDS, TEXT_OFFSET, Turn, conversation_sample
from .generate import generate

__all__ = ["Answerer", "Answer", "Attempt", "UNKNOWN_TEXT", "ASK_TO_EXTRAPOLATE",
           "EXTRAPOLATION_LABEL"]

UNKNOWN_TOKEN = 13
UNKNOWN_TEXT = "I don't know."
ASK_TO_EXTRAPOLATE = ("I couldn't find this, even after thinking it through and searching. "
                      "Do you want me to extrapolate? I'll label the answer as an unverified "
                      "estimate.")
EXTRAPOLATION_LABEL = "Extrapolated, not verified: "


@dataclass
class Attempt:
    stage: str
    text: str
    confidence: float
    loops_agree: bool
    said_unknown: bool

    @property
    def accepted(self) -> bool:
        return False  # set by Answerer, which knows the threshold


@dataclass
class Answer:
    text: str
    abstained: bool
    attempts: list[Attempt] = field(default_factory=list)
    reason: str = ""
    #: True when every rung failed and the model is asking permission to
    #: extrapolate rather than stating an answer; see Answerer.extrapolate.
    needs_permission: bool = False
    evidence: list[str] = field(default_factory=list)


class Answerer:
    """The escalation ladder around one model.

    ``search`` is optional: a callable ``query -> str | None`` (a thin wrapper
    around whatever web or document search the deployment has). Without it,
    step 4 is skipped and the ladder goes straight from widening to abstaining.
    """

    def __init__(self, model, tokenizer=None, threshold: float = 0.5,
                 search: Optional[Callable[..., Optional[str]]] = None,
                 max_new_tokens: int = 64, search_depths: int = 2) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.search = search
        self.search_depths = search_depths
        self.max_new_tokens = max_new_tokens

    def ask(self, question: str) -> Answer:
        """Climb the ladder; return an answer, or a request to extrapolate.

        Rungs: answer (adaptive depth) -> think harder (every loop) -> more
        superstacks -> search depth 1 -> search depth 2 ... Each search rung
        keeps the earlier results in context. Only when every rung fails does
        the model stop -- and then it asks whether to extrapolate instead of
        either guessing or flatly refusing.
        """
        from .thinking import ThinkingBudget

        cfg = self.model.cfg
        max_loops = cfg.router.max_loops
        n_stacks = cfg.stacks.n_stacks
        wide = n_stacks if cfg.router.top_k < n_stacks else None
        attempts: list[Attempt] = []
        turns = [Turn("user", question)]

        rungs = [("answer", dict(n_loops=max_loops, thinking=ThinkingBudget("balanced"))),
                 ("think_harder", dict(n_loops=max_loops))]
        if wide:
            rungs.append(("more_superstacks", dict(n_loops=max_loops, top_k=wide)))
        for stage, kw in rungs:
            a = self._attempt(stage, turns, **kw)
            attempts.append(a)
            if self._accept(a):
                return Answer(a.text, False, attempts)

        evidence: list[str] = []
        if self.search is not None:
            for depth in range(1, self.search_depths + 1):
                found = self._search(question, depth)
                if not found:
                    continue
                evidence.append(found)
                turns = [*turns, Turn("tool_result", found)]
                a = self._attempt(f"search_depth_{depth}", turns, n_loops=max_loops, top_k=wide)
                attempts.append(a)
                if self._accept(a):
                    return Answer(a.text, False, attempts, evidence=evidence)

        tried = ", ".join(a.stage for a in attempts)
        why = ("the model marked it unknowable" if any(a.said_unknown for a in attempts)
               else f"confidence stayed below {self.threshold:.2f}")
        return Answer(ASK_TO_EXTRAPOLATE, True, attempts,
                      reason=f"{why} after: {tried}"
                             + ("" if self.search else "; no search tool was available"),
                      needs_permission=True, evidence=evidence)

    def extrapolate(self, question: str, previous: Optional[Answer] = None) -> Answer:
        """The user said yes: best effort at full depth and width, labelled.

        Uses whatever the search rungs found as context. The label is part of
        the returned text so it cannot be dropped by a caller that only
        forwards ``.text``.
        """
        cfg = self.model.cfg
        turns = [Turn("user", question)]
        for found in (previous.evidence if previous else []):
            turns.append(Turn("tool_result", found))
        wide = cfg.stacks.n_stacks if cfg.router.top_k < cfg.stacks.n_stacks else None
        a = self._attempt("extrapolate", turns, n_loops=cfg.router.max_loops, top_k=wide,
                          allow_unknown=False)
        return Answer(EXTRAPOLATION_LABEL + (a.text or "(no estimate produced)"), False,
                      [*(previous.attempts if previous else []), a],
                      reason="user asked for an extrapolation")

    def _search(self, query: str, depth: int) -> Optional[str]:
        try:
            return self.search(query, depth=depth)
        except TypeError:
            # A one-argument search has one depth; do not repeat it.
            return self.search(query) if depth == 1 else None

    # -- internals -------------------------------------------------------------

    def _accept(self, a: Attempt) -> bool:
        return (not a.said_unknown) and a.loops_agree and a.confidence >= self.threshold \
            and bool(a.text.strip())

    @torch.no_grad()
    def _attempt(self, stage: str, turns, n_loops: int, top_k: Optional[int] = None,
                 thinking=None, allow_unknown: bool = True) -> Attempt:
        router = self.model.router
        original = router.cfg
        if top_k is not None:
            router.cfg = replace(original, top_k=min(top_k, self.model.cfg.stacks.n_stacks))
        try:
            prompt = conversation_sample(turns, supervise_assistant=False, open_for_reply=True,
                                         tokenizer=self.tokenizer)
            out = generate(self.model, prompt, max_new_tokens=self.max_new_tokens,
                           temperature=0.0,
                           stop_ids=(*STOP_IDS, UNKNOWN_TOKEN) if allow_unknown else STOP_IDS,
                           thinking=thinking,
                           n_loops=n_loops, text_offset=TEXT_OFFSET, text_only=True,
                           tokenizer=self.tokenizer)
            said_unknown = UNKNOWN_TOKEN in out.ids[:1]
            text = out.text.strip()
            confidence, agree = self._score(prompt, out.ids, n_loops)
        finally:
            router.cfg = original
        return Attempt(stage, text, confidence, agree, said_unknown)

    def _score(self, prompt: Sample, ids: list[int], n_loops: int) -> tuple[float, bool]:
        """Teacher-forced geometric-mean probability of the answer, and whether
        the final two ponder loops pick the same first token."""
        answer = [i for i in ids if i >= TEXT_OFFSET]
        if not answer:
            return 0.0, True
        from ..codecs.spans import Span
        import numpy as np
        full = Sample([*prompt.spans, Span("text", np.asarray(answer, np.int64), supervised=False)])
        param = next(self.model.parameters())
        batch = TensorBatch(collate([full], continuous_dims(self.model.cfg.codecs)),
                            device=param.device, dtype=param.dtype)
        out = self.model(batch, n_loops=n_loops)
        n_prompt = len(prompt)
        head = self.model.codecs.text_head
        logits = head(out.hidden)[0]                               # [T, V]
        logp = torch.log_softmax(logits.float(), -1)
        targets = torch.tensor(answer, device=logits.device)
        picked = logp[n_prompt - 1:n_prompt - 1 + len(answer)].gather(1, targets[:, None])
        confidence = float(picked.mean().exp())
        agree = True
        per_loop = out.per_loop_hidden
        if len(per_loop) >= 2:
            a = head(per_loop[-1])[0, n_prompt - 1].argmax()
            b = head(per_loop[-2])[0, n_prompt - 1].argmax()
            agree = bool(a == b)
        return confidence, agree
