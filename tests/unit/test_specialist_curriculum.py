"""The specialist routing curriculum: forced subject stack, solo and any-stack windows."""

import dataclasses

import torch

from iridium.codecs.bank import TensorBatch, continuous_dims
from iridium.codecs.spans import Sample, collate, text_span
from iridium.config import ConfigError, get_config
from iridium.model.iridium1 import Iridium1

NAMES = ("science", "math")


def _model(**router):
    cfg = get_config("tiny")
    assert cfg.stacks.n_stacks >= 2
    names = NAMES + tuple(f"s{i}" for i in range(cfg.stacks.n_stacks - 2))
    cfg = dataclasses.replace(
        cfg, stacks=dataclasses.replace(cfg.stacks, specializations=names),
        router=dataclasses.replace(cfg.router, top_k=min(2, cfg.stacks.n_stacks), **router))
    torch.manual_seed(0)
    return Iridium1(cfg), cfg


def _batch(cfg, subjects):
    samples = [Sample([text_span("the quick brown fox jumps", offset=16)],
                      meta={"subject": s} if s else {}) for s in subjects]
    return TensorBatch(collate(samples, continuous_dims(cfg.codecs)))


def test_forced_subject_stack_is_always_visited_and_keeps_a_partner():
    model, cfg = _model(subject_force=True)
    model.train()
    out = model(_batch(cfg, ["math", "science", None, "math"]))
    index = out.decisions[0].stack_index
    assert bool((index[0] == 1).any(-1).all()) and bool((index[3] == 1).any(-1).all())
    assert bool((index[1] == 0).any(-1).all())
    if index.shape[-1] > 1 and cfg.stacks.n_stacks > 1:
        assert bool((index[0, :, 0] != index[0, :, 1]).all()), "partner must be another stack"
    weight = out.decisions[0].stack_weight
    assert torch.allclose(weight.sum(-1), torch.ones_like(weight.sum(-1)), atol=1e-5)


def test_solo_windows_use_one_stack_and_any_windows_ignore_the_subject():
    model, cfg = _model(subject_force=True, solo_p=1.0)
    model.train()
    out = model(_batch(cfg, ["math"] * 3))
    assert bool((out.decisions[0].stack_index == 1).all())
    assert out.stats["stack_stats"][0]["per_stack_tokens"][0] == 0

    model, cfg = _model(subject_force=True, any_p=1.0)
    model.train()
    torch.manual_seed(1)
    seen = set()
    for _ in range(12):
        index = model(_batch(cfg, ["math"] * 4)).decisions[0].stack_index
        for row in index:
            assert len(row.unique()) == 1, "an any-stack window visits exactly one stack"
            seen.add(int(row.unique()))
    assert len(seen) > 1, "the random stack never varied"


def test_curriculum_is_training_only_and_trains():
    model, cfg = _model(subject_force=True, solo_p=0.5, any_p=0.5)
    batch = _batch(cfg, ["math", "science"])
    model.eval()
    torch.manual_seed(0)
    a = model(batch).decisions[0].stack_index
    plain, _ = _model()
    plain.load_state_dict(model.state_dict())
    plain.eval()
    assert torch.equal(a, plain(batch).decisions[0].stack_index)
    model.train()
    losses, _ = model.losses(batch)
    total = sum(losses.values())
    total.backward()
    assert torch.isfinite(total)
    assert model.router.gate.weight.grad is not None


def test_bad_probabilities_are_rejected():
    for bad in (dict(solo_p=-0.1), dict(solo_p=0.7, any_p=0.6)):
        try:
            _model(**bad)
        except ConfigError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_stack_order_and_every_language_family_has_a_stack():
    from iridium.data.domains import DOMAIN_FAMILIES, SUBJECTS, subject_of
    from iridium.presets import get_preset

    assert SUBJECTS == ("math", "chat", "code", "science")
    preset = get_preset("iridium-50m")
    assert preset.config.stacks.specializations == SUBJECTS
    assert preset.config.router.subject_force and preset.config.router.any_p == 0
    for family in preset.mixture:
        if family.endswith("_foundation"):
            assert subject_of(family) is None
        else:
            assert subject_of(family) in SUBJECTS, family
    assert subject_of("chat_general") == subject_of("writing_qa") == "chat"
    assert all(spec.subject in SUBJECTS for spec in DOMAIN_FAMILIES.values())


def test_specialist_matrix_overrides_routing_and_restores_it():
    from iridium.evaluation.specialists import describe, specialist_matrix

    model, cfg = _model()
    model.train()
    batches = {"science": [_batch(cfg, ["science"] * 2)], "math": [_batch(cfg, ["math"] * 2)],
               "unknown": [_batch(cfg, [None])]}
    report = specialist_matrix(model, batches)
    assert set(report["subjects"]) == {"science", "math"}
    row = report["subjects"]["math"]
    assert {"routed", "core", "science", "math", "margin", "passes", "best_rival"} <= set(row)
    assert row["core"] != row["math"], "the core-only route still used a stack"
    assert row["passes"] == (row["margin"] > 0)
    assert model.training and model.route_override is None
    assert "verdict" in describe(report)

    # The override really does send every token to the one stack, or to none.
    model.eval()
    batch = _batch(cfg, ["math"])
    model.route_override = 0
    stats = model(batch).stats["stack_stats"][0]["per_stack_tokens"]
    assert stats[0] > 0 and sum(stats[1:]) == 0
    model.route_override = "core"
    assert sum(model(batch).stats["stack_stats"][0]["per_stack_tokens"]) == 0
    model.route_override = None
