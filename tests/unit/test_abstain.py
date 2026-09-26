"""Answer, escalate, or abstain: the ladder and the unknowable family."""

import numpy as np
import torch

from iridium.config import get_config
from iridium.model.iridium1 import Iridium1
from iridium.runtime.abstain import ASK_TO_EXTRAPOLATE, EXTRAPOLATION_LABEL, Answerer
from iridium.training.tasks import UNKNOWN, unknowable_item


def test_unknowable_family_is_balanced_and_graded_exactly():
    rng = np.random.default_rng(0)
    items = [unknowable_item(rng) for _ in range(400)]
    answerable = sum(it.truth["answerable"] for it in items)
    assert 150 < answerable < 250
    for it in items[:50]:
        tail = it.sample.spans[-2]
        if it.truth["answerable"]:
            assert it.grade(it.answer) and not it.grade("UNKNOWN")
        else:
            assert tail.modality == "control" and int(tail.payload[0]) == UNKNOWN
            assert it.grade("UNKNOWN")


def test_untrained_model_climbs_every_rung_then_asks_to_extrapolate():
    torch.manual_seed(0)
    model = Iridium1(get_config("tiny"))
    calls = []

    def search(q, depth=1):
        calls.append(depth)
        return f"results at depth {depth}"

    answerer = Answerer(model, threshold=0.99, max_new_tokens=4, search=search)
    q = "what is the airspeed of a laden swallow?"
    out = answerer.ask(q)
    assert out.abstained and out.needs_permission and out.text == ASK_TO_EXTRAPOLATE
    stages = [a.stage for a in out.attempts]
    assert stages[:2] == ["answer", "think_harder"]
    assert stages[-2:] == ["search_depth_1", "search_depth_2"] and calls == [1, 2]
    assert out.evidence == ["results at depth 1", "results at depth 2"]
    assert "after:" in out.reason
    guess = answerer.extrapolate(q, out)
    assert guess.text.startswith(EXTRAPOLATION_LABEL) and not guess.needs_permission
    assert guess.attempts[-1].stage == "extrapolate"
    # The router's configuration is restored after widening.
    assert model.router.cfg.top_k == model.cfg.router.top_k


def test_a_one_argument_search_is_used_once():
    torch.manual_seed(0)
    model = Iridium1(get_config("tiny"))
    seen = []
    out = Answerer(model, threshold=0.99, max_new_tokens=2,
                   search=lambda q: seen.append(q) or "x").ask("q?")
    assert seen == ["q?"] and out.attempts[-1].stage == "search_depth_1"


def test_a_confident_first_answer_is_returned_without_escalating():
    torch.manual_seed(0)
    model = Iridium1(get_config("tiny"))
    out = Answerer(model, threshold=0.0, max_new_tokens=3).ask("hi")
    if out.attempts[0].loops_agree and out.attempts[0].text:
        assert not out.abstained and len(out.attempts) == 1


def test_the_model_saying_unknown_is_respected_immediately():
    torch.manual_seed(0)
    model = Iridium1(get_config("tiny"))

    class Head(torch.nn.Module):
        def forward(self, h):
            logits = torch.full((*h.shape[:-1], model.cfg.codecs.vocab_size), -1e9)
            logits[..., UNKNOWN] = 0.0
            return logits

    model.codecs.text_head = Head()
    out = Answerer(model, threshold=0.0, max_new_tokens=3).ask("what is zeta?")
    assert out.abstained and all(a.said_unknown for a in out.attempts)
    assert "unknowable" in out.reason
