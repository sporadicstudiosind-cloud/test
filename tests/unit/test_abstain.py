"""Answer, escalate, or abstain: the ladder and the unknowable family."""

import numpy as np
import torch

from iridium.config import get_config
from iridium.model.iridium1 import Iridium1
from iridium.runtime.abstain import UNKNOWN_TEXT, Answerer
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


def test_untrained_model_climbs_the_whole_ladder_then_says_it_does_not_know():
    torch.manual_seed(0)
    model = Iridium1(get_config("tiny"))
    queries = []
    answerer = Answerer(model, threshold=0.99, max_new_tokens=4,
                        search=lambda q: queries.append(q) or "nothing relevant")
    out = answerer.ask("what is the airspeed of a laden swallow?")
    assert out.abstained and out.text == UNKNOWN_TEXT
    stages = [a.stage for a in out.attempts]
    assert stages[:2] == ["answer", "think_harder"] and stages[-1] == "search"
    assert queries == ["what is the airspeed of a laden swallow?"]
    assert "after:" in out.reason
    # The router's configuration is restored after widening.
    assert model.router.cfg.top_k == model.cfg.router.top_k


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
