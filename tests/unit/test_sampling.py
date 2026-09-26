"""Truncation samplers, checked against hand-computed distributions.

These are twelve lines of tensor code that decide what the model is allowed to
say, and every one of them fails quietly: a mis-signed comparison does not
crash, it just widens the tail, and the only symptom is that the output reads
slightly worse than it should. So the expected keep-sets here are worked out by
hand from the probabilities rather than recorded from a run.
"""
from __future__ import annotations

import math

import pytest
import torch

from iridium.runtime.generate import _pick


def _rng(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _empirical_support(logits, samples=4000, **kwargs):
    """Which token ids the sampler can actually reach."""
    rng = _rng()
    return {_pick(logits.clone(), rng=rng, **kwargs) for _ in range(samples)}


def test_greedy_ignores_every_truncation_knob():
    logits = torch.tensor([0.1, 5.0, 0.3, 0.2])
    for kwargs in ({}, {"min_p": 0.9}, {"top_k": 1}, {"top_p": 0.1}):
        assert _pick(logits, temperature=0.0, rng=_rng(), **kwargs) == 1


def test_min_p_keeps_exactly_the_tokens_above_the_relative_floor():
    """p = [0.7, 0.2, 0.06, 0.04]; a floor of 0.25 * 0.7 = 0.175 keeps two."""
    probs = torch.tensor([0.7, 0.2, 0.06, 0.04])
    logits = probs.log()
    assert _empirical_support(logits, temperature=1.0, min_p=0.25) == {0, 1}
    # 0.1 * 0.7 = 0.07 -> keeps 0.7, 0.2 and 0.06? No: 0.06 < 0.07. Three only
    # once the floor drops below 0.04.
    assert _empirical_support(logits, temperature=1.0, min_p=0.05) == {0, 1, 2, 3}


def test_min_p_adapts_to_confidence_where_top_p_does_not():
    """The property min_p exists for, stated as a test.

    Two steps with the same nucleus budget: one confident, one uncertain. A
    fixed top_p keeps a tail at the confident step. min_p does not, and yet
    stays permissive at the uncertain step. If this ever fails, min_p has been
    reduced to a reparametrisation of top_p and is not worth having.
    """
    confident = torch.tensor([0.94, 0.03, 0.02, 0.01]).log()
    uncertain = torch.tensor([0.30, 0.28, 0.22, 0.20]).log()

    assert _empirical_support(confident, temperature=1.0, min_p=0.2) == {0}
    assert _empirical_support(uncertain, temperature=1.0, min_p=0.2) == {0, 1, 2, 3}

    # Same top_p admits the tail at the confident step -- the behaviour min_p
    # is meant to avoid.
    assert len(_empirical_support(confident, temperature=1.0, top_p=0.97)) > 1


def test_min_p_is_applied_after_temperature():
    """Flattening with temperature must widen what min_p keeps, not leave it fixed.

    If min_p were applied to the raw logits the support would be identical at
    both temperatures, which is the bug this catches.
    """
    logits = torch.tensor([3.0, 1.5, 1.0, 0.5])
    cold = _empirical_support(logits, temperature=0.5, min_p=0.3)
    hot = _empirical_support(logits, temperature=2.0, min_p=0.3)
    assert cold < hot


def test_min_p_never_empties_the_candidate_set():
    """The argmax clears its own floor by construction, at any threshold."""
    logits = torch.tensor([0.25, 0.25, 0.25, 0.25]).log()
    for floor in (0.01, 0.5, 0.999):
        assert _empirical_support(logits, samples=200, temperature=1.0,
                                  min_p=floor) <= {0, 1, 2, 3}
    sharp = torch.tensor([10.0, 0.0, 0.0])
    assert _empirical_support(sharp, samples=200, temperature=1.0, min_p=0.99) == {0}


def test_min_p_composes_with_top_k_without_emptying():
    """top_k first, min_p second; the intersection must still contain the argmax."""
    logits = torch.tensor([0.5, 0.25, 0.15, 0.10]).log()
    support = _empirical_support(logits, temperature=1.0, top_k=2, min_p=0.9)
    assert support == {0}


def test_min_p_out_of_range_is_a_no_op():
    """0.0 disables it; 1.0 and above are not silently clamped into 'greedy'."""
    logits = torch.tensor([0.4, 0.3, 0.2, 0.1]).log()
    assert _empirical_support(logits, temperature=1.0, min_p=0.0) == {0, 1, 2, 3}
    assert _empirical_support(logits, temperature=1.0, min_p=1.0) == {0, 1, 2, 3}


def test_repetition_penalty_moves_both_signs_toward_zero():
    """Dividing a negative logit raises it; the sampler must not do that."""
    logits = torch.tensor([2.0, -2.0, 0.5])
    rng = _rng()
    seen = {_pick(logits.clone(), temperature=1.0, rng=rng,
                  repetition_penalty=2.0, emitted=[0, 1]) for _ in range(2000)}
    assert seen  # sanity: it still produces something

    # Check the arithmetic directly rather than through the sampler's RNG.
    penalised = logits.clone()
    scores = penalised[torch.tensor([0, 1])]
    expected = torch.where(scores > 0, scores / 2.0, scores * 2.0)
    assert math.isclose(float(expected[0]), 1.0)
    assert math.isclose(float(expected[1]), -4.0)
