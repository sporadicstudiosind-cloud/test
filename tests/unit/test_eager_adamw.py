"""EagerAdamW: the foreach path is the per-tensor path, and groups are honoured.

The claim worth testing here is not "the optimizer runs". It is that the two
code paths compute the *same* update and that weight decay reaches exactly the
tensors it should. Both are the kind of thing that is invisible in a loss curve
and expensive to discover months later.
"""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from iridium.runtime.memory import build_optimizer, decay_groups
from iridium.training.eager_adamw import EagerAdamW


def _model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(8, 16), nn.LayerNorm(16), nn.GELU(), nn.Linear(16, 4)
    )


def _run(model, optimizer, steps=6, seed=1):
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = torch.randn(4, 8, generator=generator)
        loss = model(x).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return [p.detach().clone() for p in model.parameters()]


def test_foreach_matches_per_tensor_exactly_enough():
    """Same arithmetic, different kernel count. Differences are fp32 round-off."""
    a, b = _model(), _model()
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    slow = _run(a, EagerAdamW(a.parameters(), lr=1e-2, foreach=False))
    fast = _run(b, EagerAdamW(b.parameters(), lr=1e-2, foreach=True))
    for x, y in zip(slow, fast):
        torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-7)


def test_foreach_default_follows_the_device():
    """CPU weights get the per-tensor path; there is no launch overhead to remove."""
    model = _model()
    assert EagerAdamW(model.parameters()).foreach is False


def test_decay_groups_split_by_rank_not_by_name():
    model = _model()
    groups = decay_groups(model.named_parameters(), weight_decay=0.1)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in groups[0]["params"])
    assert all(p.ndim < 2 for p in groups[1]["params"])
    # Every trainable tensor lands in exactly one group.
    total = sum(p.numel() for g in groups for p in g["params"])
    assert total == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_decay_groups_skips_frozen_parameters():
    model = _model()
    model[0].weight.requires_grad_(False)
    groups = decay_groups(model.named_parameters())
    flat = {id(p) for g in groups for p in g["params"]}
    assert id(model[0].weight) not in flat
    assert id(model[3].weight) in flat


def test_decay_groups_are_always_both_present():
    """A stable group layout is what lets one run load another's optimizer state."""
    only_matrices = nn.Linear(4, 4, bias=False)
    groups = decay_groups(only_matrices.named_parameters())
    assert len(groups) == 2 and groups[1]["params"] == []


def test_norms_and_biases_are_not_decayed():
    """The point of the split: a LayerNorm gain must not shrink under decay.

    Driven with zero gradient so the only force acting on the parameters is the
    decay term itself. With the split in place the gain and the biases are
    untouched; the matrices are not.
    """
    model = _model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = EagerAdamW(decay_groups(model.named_parameters(), weight_decay=0.5),
                           lr=1e-1)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()
    for name, param in model.named_parameters():
        if param.ndim >= 2:
            assert not torch.equal(param, before[name]), f"{name} should decay"
        else:
            assert torch.equal(param, before[name]), f"{name} must not decay"


def test_groups_survive_a_state_dict_round_trip():
    model = _model()
    optimizer = EagerAdamW(decay_groups(model.named_parameters(), 0.1), lr=1e-2)
    _run(model, optimizer, steps=3)
    state = copy.deepcopy(optimizer.state_dict())

    restored = _model()
    other = EagerAdamW(decay_groups(restored.named_parameters(), 0.1), lr=1e-2)
    other.load_state_dict(state)
    assert len(other.param_groups) == 2
    assert other.param_groups[1]["weight_decay"] == 0.0
    assert len(other.state) == len(optimizer.state)


def test_a_parameter_in_two_groups_is_refused():
    shared = nn.Parameter(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="more than one group"):
        EagerAdamW([{"params": [shared]}, {"params": [shared]}])


def test_lagging_parameter_gets_its_own_bias_correction():
    """A tensor that starts late must not inherit the leader's step count.

    Two parameters, one of which receives no gradient for the first few steps.
    Under the foreach path both live in the same group, so the bucketing by step
    count is the only thing keeping the latecomer's first update correctly
    bias-corrected. Checked against a solo optimizer that saw the same history.
    """
    early = nn.Parameter(torch.ones(2, 2))
    late = nn.Parameter(torch.ones(2, 2))
    solo = nn.Parameter(torch.ones(2, 2))
    together = EagerAdamW([early, late], lr=1e-2, weight_decay=0.0, foreach=True)
    alone = EagerAdamW([solo], lr=1e-2, weight_decay=0.0, foreach=False)

    for step in range(5):
        early.grad = torch.full((2, 2), 0.3)
        late.grad = torch.full((2, 2), 0.7) if step >= 3 else None
        together.step()
        if step >= 3:
            solo.grad = torch.full((2, 2), 0.7)
            alone.step()
    torch.testing.assert_close(late.detach(), solo.detach(), rtol=1e-6, atol=1e-7)


def test_build_optimizer_accepts_groups_and_a_flat_list():
    model = _model()
    flat = build_optimizer([p for p in model.parameters()], kind="eager_adamw")
    assert len(flat.param_groups) == 1
    grouped = build_optimizer(decay_groups(model.named_parameters()),
                              kind="eager_adamw")
    assert len(grouped.param_groups) == 2

    torch_flat = build_optimizer(list(model.parameters()), kind="adamw")
    assert isinstance(torch_flat, torch.optim.AdamW)
    torch_grouped = build_optimizer(decay_groups(model.named_parameters(), 0.1),
                                    kind="adamw")
    assert [g["weight_decay"] for g in torch_grouped.param_groups] == [0.1, 0.0]


def test_embedding_tables_are_not_decayed_even_though_they_are_two_dimensional():
    """OLMo 2's finding. A rank rule alone decays them; a module walk does not."""
    model = nn.Sequential(nn.Embedding(10, 4), nn.Linear(4, 4))
    groups = decay_groups(model, weight_decay=0.1)
    decayed = {id(p) for p in groups[0]["params"]}
    assert id(model[0].weight) not in decayed
    assert id(model[1].weight) in decayed


def test_tied_parameters_are_grouped_once():
    emb = nn.Embedding(10, 4)
    head = nn.Linear(4, 10, bias=False)
    head.weight = emb.weight
    model = nn.ModuleDict({"emb": emb, "head": head})
    groups = decay_groups(model)
    flat = [p for g in groups for p in g["params"]]
    assert len(flat) == len({id(p) for p in flat}) == 1
    assert groups[1]["params"][0] is emb.weight
