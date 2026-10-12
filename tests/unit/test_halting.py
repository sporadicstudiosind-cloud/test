"""Stopping-time semantics. Covers §3's "ACT averaging and PonderNet stopping
are combined without full semantics" correction and §6.4."""

import numpy as np
import pytest

from iridium.model.halting import (
    HaltingError,
    expected_depth,
    geometric_prior,
    halting_loss,
    kl_divergence,
    sample_stop,
    stopping_distribution,
    threshold_stop,
)


@pytest.mark.parametrize("n", [1, 2, 5, 16])
def test_distribution_sums_to_one(n):
    rng = np.random.default_rng(n)
    lam = rng.uniform(0.05, 0.95, size=n)
    p = stopping_distribution(lam)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(p >= 0.0)


def test_last_step_absorbs_remaining_mass():
    lam = np.array([0.1, 0.1, 0.1])
    p = stopping_distribution(lam)
    assert p[0] == pytest.approx(0.1)
    assert p[1] == pytest.approx(0.09)
    assert p[2] == pytest.approx(0.81)          # 0.9 * 0.9, not 0.1 * 0.81
    assert p.sum() == pytest.approx(1.0)


def test_certain_stop_puts_all_mass_on_first_step():
    p = stopping_distribution(np.array([1.0, 0.5, 0.5]))
    assert p[0] == pytest.approx(1.0)
    assert p[1:].sum() == pytest.approx(0.0)


def test_never_stop_puts_all_mass_on_last_step():
    p = stopping_distribution(np.array([0.0, 0.0, 0.0]))
    assert p[-1] == pytest.approx(1.0)


def test_invalid_lambdas_rejected():
    with pytest.raises(HaltingError):
        stopping_distribution(np.array([1.5]))
    with pytest.raises(HaltingError):
        stopping_distribution(np.array([]))
    with pytest.raises(HaltingError):
        stopping_distribution(np.zeros((2, 2)))


def test_geometric_prior_normalized():
    prior = geometric_prior(8, 0.25)
    assert prior.sum() == pytest.approx(1.0)
    assert prior[0] == pytest.approx(0.25)
    assert prior[-1] == pytest.approx(0.75 ** 7)


def test_geometric_prior_matches_forced_final_stop():
    prior = geometric_prior(3, 0.25)
    np.testing.assert_allclose(prior, [0.25, 0.1875, 0.5625])
    np.testing.assert_allclose(geometric_prior(1, 0.25), [1.0])
    with pytest.raises(HaltingError, match="positive"):
        geometric_prior(0, 0.25)


def test_kl_is_zero_for_identical_distributions():
    prior = geometric_prior(6, 0.3)
    assert kl_divergence(prior, prior) == pytest.approx(0.0, abs=1e-12)


def test_kl_is_positive_otherwise():
    a = geometric_prior(6, 0.3)
    b = geometric_prior(6, 0.7)
    assert kl_divergence(a, b) > 0.0


def test_loss_weights_task_loss_at_every_stop():
    """The defining PonderNet property: the task loss is trained at each stop.

    Halting early on a step whose task loss is high must cost more than halting
    early on a step whose task loss is low.
    """
    lam_early = np.array([0.9, 0.5, 0.5])
    lam_late = np.array([0.01, 0.01, 0.5])
    losses = np.array([5.0, 1.0, 0.2])         # deeper is better here

    loss_early, p_early = halting_loss(lam_early, losses, beta=0.0)
    loss_late, p_late = halting_loss(lam_late, losses, beta=0.0)

    assert loss_early > loss_late
    assert expected_depth(lam_early) < expected_depth(lam_late)


def test_loss_rejects_mismatched_lengths():
    with pytest.raises(HaltingError):
        halting_loss(np.array([0.5, 0.5]), np.array([1.0]))


def test_prior_term_penalises_extreme_depth_preference():
    losses = np.zeros(8)                       # task loss says nothing
    never = halting_loss(np.full(8, 1e-6), losses, prior_p_stop=0.3, beta=1.0)[0]
    matched = halting_loss(np.full(8, 0.3), losses, prior_p_stop=0.3, beta=1.0)[0]
    assert never > matched                     # the prior pulls toward itself


def test_sampled_serving_policy_is_reproducible():
    lam = np.array([0.2, 0.3, 0.4, 0.5])
    assert sample_stop(lam, seed=7) == sample_stop(lam, seed=7)


def test_sampled_policy_matches_training_distribution():
    """Serving must not silently use a different estimator from training."""
    lam = np.array([0.25, 0.4, 0.6, 0.5])
    p = stopping_distribution(lam)
    counts = np.zeros(lam.size)
    for seed in range(20_000):
        counts[sample_stop(lam, seed)] += 1
    empirical = counts / counts.sum()
    assert np.max(np.abs(empirical - p)) < 0.02


def test_threshold_policy_is_deterministic_and_distinct():
    lam = np.array([0.25, 0.4, 0.6, 0.5])
    a = threshold_stop(lam, 0.9)
    b = threshold_stop(lam, 0.9)
    assert a == b
    # It is a separately calibrated rule, not the same estimator as sampling.
    assert threshold_stop(lam, 0.1) != threshold_stop(lam, 0.99)


def test_threshold_bounds_checked():
    with pytest.raises(HaltingError):
        threshold_stop(np.array([0.5]), 0.0)
    with pytest.raises(HaltingError):
        threshold_stop(np.array([0.5]), 1.5)
