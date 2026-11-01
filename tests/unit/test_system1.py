"""System-1 / System-2 routing.

The router's contract is small and every clause of it is a way to ship a wrong
answer silently: take the one-pass path only when the typed head is willing and
sure, escalate otherwise, never change the model underneath, and read the same
hidden state that generation would start from.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from iridium.codecs.spans import Sample, text_span
from iridium.config import get_config
from iridium.model.iridium1 import Iridium1
from iridium.model.typed_head import BoolType, EnumType, StructType
from iridium.runtime.system1 import System1Router, encode_prompt
from iridium.training.tasks import BOS, TEXT_OFFSET, control_span


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Iridium1(get_config("tiny")).eval()


def _sample(text: str) -> Sample:
    return Sample([control_span(BOS, supervised=False),
                   text_span(text, supervised=False, offset=TEXT_OFFSET)])


class _FakeGenerate:
    def __init__(self):
        self.calls = 0

    def __call__(self, model, sample, **kwargs):
        self.calls += 1
        return SimpleNamespace(text="yes", ids=[1, 2, 3])


def _force(head, abstain_logit: float, value_bias: float = 12.0):
    """Pin a head's outputs: decisive value, chosen abstention."""
    with torch.no_grad():
        for out in head.field_out.values():
            out.weight.zero_()
            out.bias.zero_()
            out.bias[0] = value_bias
        head.abstain_head[1].weight.zero_()
        head.abstain_head[1].bias.fill_(abstain_logit)


def test_confident_head_answers_in_one_pass_without_generating(model):
    fake = _FakeGenerate()
    router = System1Router(model, generate_fn=fake)
    _force(router.register("cat", EnumType(["a", "b", "c"])), abstain_logit=-10.0)
    answer = router.answer(_sample("pick one"), "cat")
    assert answer.route == "system1"
    assert answer.value == "a"
    assert answer.forward_passes == 1
    assert answer.confidence > 0.99
    assert fake.calls == 0


def test_abstaining_head_escalates_to_generation(model):
    fake = _FakeGenerate()
    router = System1Router(model, generate_fn=fake)
    _force(router.register("flag", BoolType()), abstain_logit=10.0)
    answer = router.answer(_sample("is it?"), "flag", parse=lambda t: t == "yes")
    assert answer.route == "system2"
    assert answer.value is True and answer.text == "yes"
    assert "abstained" in answer.reason
    assert answer.forward_passes == 1 + 3
    assert fake.calls == 1


def test_confidence_floor_escalates_on_the_weakest_struct_field(model):
    """One shaky field sends the whole struct to System 2."""
    fake = _FakeGenerate()
    router = System1Router(model, min_confidence=0.9, generate_fn=fake)
    head = router.register("pair", StructType({"x": BoolType(), "y": BoolType()}))
    _force(head, abstain_logit=-10.0)
    with torch.no_grad():
        head.field_out["y"].bias.zero_()          # y becomes a coin flip
    answer = router.answer(_sample("two things"), "pair")
    assert answer.route == "system2"
    assert "weakest field confidence" in answer.reason
    assert answer.field_confidence["x"] > 0.99 and answer.field_confidence["y"] < 0.9


def test_fallback_can_be_refused(model):
    router = System1Router(model, generate_fn=_FakeGenerate())
    _force(router.register("flag", BoolType()), abstain_logit=10.0)
    answer = router.answer(_sample("?"), "flag", fallback=False)
    assert answer.route == "system1" and answer.value is None


def test_registering_heads_leaves_the_model_parameter_set_untouched(model):
    """Heads live on the router: the model still matches its config's accounting."""
    before = sum(p.numel() for p in model.parameters())
    router = System1Router(model, generate_fn=_FakeGenerate())
    router.register("a", BoolType())
    router.register("b", EnumType(["x", "y"]))
    assert sum(p.numel() for p in model.parameters()) == before == model.cfg.n_params


def test_encode_prompt_is_the_state_generation_starts_from(model):
    """Same prefill as generate(): cached, chunked, same loop count."""
    from iridium.codecs.bank import TensorBatch, continuous_dims
    from iridium.codecs.spans import collate

    sample = _sample("hello there")
    state = encode_prompt(model, sample)
    batch = TensorBatch(collate([sample], continuous_dims(model.cfg.codecs)))
    with torch.no_grad():
        full = model(batch, n_loops=1).hidden[0, -1]
    torch.testing.assert_close(state, full, rtol=1e-5, atol=1e-5)


def test_fit_learns_a_separable_task_and_calibrates_on_the_holdout(model):
    router = System1Router(model, generate_fn=_FakeGenerate())
    router.register("polarity", BoolType(), d_hidden=32)
    texts = [f"{w} {i}" for i in range(12) for w in ("good", "bad")]
    labels = {"value": torch.tensor([1 if t.startswith("good") else 0 for t in texts])}
    report = router.fit("polarity", [_sample(t) for t in texts], labels,
                        steps=150, lr=1e-2)
    assert report["holdout"] > 0 and report["train"] + report["holdout"] == len(texts)
    assert report["loss_last"] < report["loss_first"]
    assert "value" in report["temperatures"]


def test_state_dict_round_trip_restores_answers(model):
    router = System1Router(model, generate_fn=_FakeGenerate())
    _force(router.register("cat", EnumType(["a", "b"])), abstain_logit=-10.0)
    router.heads["cat"].temperatures["value"] = 1.7
    restored = System1Router(model, generate_fn=_FakeGenerate())
    restored.load_state_dict(router.state_dict())
    assert restored.heads["cat"].temperatures["value"] == 1.7
    assert restored.answer(_sample("x"), "cat").value == router.answer(_sample("x"), "cat").value


def test_unknown_task_is_an_error_not_a_silent_fallback(model):
    with pytest.raises(KeyError):
        System1Router(model, generate_fn=_FakeGenerate()).answer(_sample("x"), "missing")
