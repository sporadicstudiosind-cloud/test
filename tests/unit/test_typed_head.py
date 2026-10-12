"""TypedHead: schema masking, calibration, parallel emission, abstention.

Calibration is the one claim in this module that is worthless unimplemented
and worse than nothing if implemented wrong (a badly-calibrated confidence is
actively misleading in a way that no confidence at all is not). So the
calibration test here does the thing the module docstring insists on: builds
a model that is deliberately, measurably miscalibrated, fits
``calibrate_temperature`` on a held-out split, and checks ECE and Brier score
actually improve on a *third*, disjoint evaluation split -- not the same data
the temperature was fit on, which would make the check circular.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from iridium.model.typed_head import (
    BoolType,
    EnumType,
    IntType,
    RealType,
    StructType,
    TypedHead,
    brier_score,
    calibrate_temperature,
    expected_calibration_error,
)


D_MODEL = 16


def _schema():
    return StructType({
        "ok": BoolType(),
        "color": EnumType(["red", "green", "blue"]),
        "count": IntType(0, 5),
        "score": RealType(0.0, 1.0),
    })


def test_schema_rejects_bad_shapes():
    with pytest.raises(ValueError):
        EnumType(["only_one"])
    with pytest.raises(ValueError):
        EnumType(["a", "a"])
    with pytest.raises(ValueError):
        IntType(5, 5)
    with pytest.raises(ValueError):
        RealType(1.0, 0.0)
    with pytest.raises(ValueError):
        StructType({})
    with pytest.raises(ValueError):
        StructType({"nested": StructType({"x": BoolType()})})


def test_output_is_always_a_valid_typed_value():
    """The type mask, not luck: every decoded value must be in the type's
    support, checked across many random hidden states and an untrained
    (hence structurally arbitrary) network, so this is a property of the
    output layer's shape, not of anything learned."""
    torch.manual_seed(0)
    schema = _schema()
    head = TypedHead(D_MODEL, schema)
    h = torch.randn(200, D_MODEL) * 5.0  # wide spread, including extreme activations
    pred = head(h)
    assert set(pred.fields["ok"].value) <= {True, False}
    assert set(pred.fields["color"].value) <= {"red", "green", "blue"}
    assert all(0 <= c <= 5 for c in pred.fields["count"].value)
    assert all(0.0 <= s <= 1.0 for s in pred.fields["score"].value)


def test_single_field_schema_uses_value_key():
    head = TypedHead(D_MODEL, BoolType())
    pred = head(torch.randn(3, D_MODEL))
    assert list(pred.fields.keys()) == ["value"]


def test_decide_interface_for_router_integration():
    torch.manual_seed(0)
    head = TypedHead(D_MODEL, _schema())
    out = head.decide(torch.randn(D_MODEL))
    assert isinstance(out, dict) and set(out) == {"ok", "color", "count", "score"}
    for name, (value, confidence) in out.items():
        assert isinstance(confidence, float) and 0.0 <= confidence <= 1.0

    # A head forced to always abstain must return None through this interface.
    always_abstain = TypedHead(D_MODEL, BoolType(), abstain_init_logit=10.0)
    assert always_abstain.decide(torch.randn(D_MODEL)) is None


def test_parallel_emission_one_forward_pass_no_autoregression():
    """All fields must come from a single call to the trunk: patching the
    trunk to count invocations and checking it fires once per ``forward``
    call, regardless of field count, is the operational meaning of
    "parallel" here (as opposed to one trunk call per field, which is what a
    naive per-field-autoregressive implementation would do)."""
    schema = StructType({f"f{i}": BoolType() for i in range(6)})
    head = TypedHead(D_MODEL, schema)
    calls = []
    original = head.trunk.forward

    def counting_forward(x):
        calls.append(1)
        return original(x)

    head.trunk.forward = counting_forward
    head(torch.randn(4, D_MODEL))
    assert len(calls) == 1, "expected exactly one trunk evaluation for all 6 fields"


def test_dependency_mechanism_changes_but_does_not_autoregress():
    """``dependency=True`` should measurably change one field's logits when
    another field's *branch input* changes (the light coupling working), via
    exactly one extra linear layer -- not by adding forward-pass calls."""
    torch.manual_seed(0)
    schema = StructType({"a": BoolType(), "b": EnumType(["x", "y", "z"])})
    coupled = TypedHead(D_MODEL, schema, dependency=True)
    with torch.no_grad():
        coupled.context_mix.weight.normal_(std=0.5)
        coupled.context_mix.bias.zero_()
    h = torch.randn(5, D_MODEL)
    logits_a_1 = coupled.field_logits(h)["a"].clone()
    with torch.no_grad():
        coupled.branches["b"][1].weight.add_(torch.randn_like(coupled.branches["b"][1].weight))
    logits_a_2 = coupled.field_logits(h)["a"]
    assert not torch.allclose(logits_a_1, logits_a_2), (
        "field a's logits should react to field b's branch under dependency=True"
    )


def test_abstention_trained_against_correctness():
    """The abstain head should learn to raise its probability on examples
    where the value head is wrong, given enough of a training signal --
    mirroring ConfidenceHead's contract in heads.py (trained against realized
    correctness, not a self-reported feeling)."""
    torch.manual_seed(0)
    head = TypedHead(D_MODEL, BoolType(), d_hidden=16)
    opt = torch.optim.Adam(head.parameters(), lr=5e-2)
    g = torch.Generator().manual_seed(0)
    # Construct an easy separable task: h[:, 0] > 0 <=> label True, and make
    # "hard" (near-zero h[:,0]) examples the ones we mark should-abstain.
    for _ in range(40):
        h = torch.randn(256, D_MODEL, generator=g)
        labels = {"value": (h[:, 0] > 0).long()}
        should_abstain = (h[:, 0].abs() < 0.15).float()
        loss = head.loss(h, labels, abstain_label=should_abstain, abstain_weight=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Probes are drawn from the training distribution with only the deciding
    # coordinate pinned. An earlier version used [0.02, 0, 0, ...] against
    # [3.0, 0, 0, ...]: vectors that differ only in length, which the head's
    # input RMSNorm maps to the *same* vector -- a probe no head could pass.
    probe = torch.Generator().manual_seed(1)
    with torch.no_grad():
        h_hard = torch.randn(256, D_MODEL, generator=probe)
        h_hard[:, 0] = 0.02  # near the decision boundary
        h_easy = torch.randn(256, D_MODEL, generator=probe)
        h_easy[:, 0] = 3.0 * torch.sign(torch.randn(256, generator=probe))  # far from it, both sides
        p_hard = head(h_hard).abstain_prob.mean().item()
        p_easy = head(h_easy).abstain_prob.mean().item()
    assert p_hard > p_easy, (p_hard, p_easy)


def test_calibration_improves_ece_and_brier_on_miscalibrated_model():
    """The central claim, measured end to end.

    Build a synthetic categorical model that is deliberately overconfident
    (true class probabilities scaled up before softmax -- the canonical
    miscalibration failure mode reported by Guo et al., 2017), fit
    ``calibrate_temperature`` on one held-out split, and check both ECE and
    Brier score improve on a third, disjoint evaluation split.
    """
    torch.manual_seed(0)
    n, k = 6000, 4
    true_logits = torch.randn(n, k) * 1.2
    labels = torch.distributions.Categorical(logits=true_logits).sample()
    overconfident = true_logits * 6.0  # miscalibration: correct ranking, wrong sharpness

    cal_logits, eval_logits = overconfident[:3000], overconfident[3000:]
    cal_labels, eval_labels = labels[:3000], labels[3000:]

    t = calibrate_temperature(cal_logits, cal_labels)
    assert t > 1.5, "a genuinely overconfident model should need T > 1 to flatten it"

    probs_raw = F.softmax(eval_logits, dim=-1)
    probs_cal = F.softmax(eval_logits / t, dim=-1)
    correct = (probs_raw.argmax(-1) == eval_labels).float()  # T doesn't change argmax

    ece_raw = expected_calibration_error(probs_raw.max(-1).values, correct)
    ece_cal = expected_calibration_error(probs_cal.max(-1).values, correct)
    brier_raw = brier_score(probs_raw, eval_labels)
    brier_cal = brier_score(probs_cal, eval_labels)

    assert ece_cal < ece_raw * 0.5, (ece_raw, ece_cal)
    assert brier_cal < brier_raw, (brier_raw, brier_cal)
    # Calibration must not have changed which class is predicted -- it can
    # only reshape confidence, never accuracy, by construction.
    assert torch.equal(probs_raw.argmax(-1), probs_cal.argmax(-1))


def test_typed_head_calibrate_uses_held_out_batch():
    torch.manual_seed(0)
    schema = StructType({"ok": BoolType()})
    head = TypedHead(D_MODEL, schema)
    with torch.no_grad():
        head.field_out["ok"].weight.mul_(8.0)  # force overconfidence
    g = torch.Generator().manual_seed(1)
    h = torch.randn(2000, D_MODEL, generator=g)
    with torch.no_grad():
        true_bit = (h[:, 0] > 0).long()
    fitted = head.calibrate(h, {"ok": true_bit})
    assert "ok" in fitted
    assert head.temperatures["ok"] == fitted["ok"]
