"""Stopping-time semantics for the recurrent core.

Architecture §6.4. The archived ORRERY specification combined ACT-style hidden
state averaging with a PonderNet-style geometric prior and called the result one
algorithm. They are different: PonderNet predicts a *distribution over stopping
times* and trains the task loss at each stop; ACT averages intermediate states.

This module implements the stopping distribution, the training loss, and two
serving policies whose behaviour matches the training semantics. It contains no
learned parameters; the halting head lives in the model, this is the contract it
must satisfy.
"""

from __future__ import annotations

import numpy as np


class HaltingError(ValueError):
    pass


def stopping_distribution(lambdas: np.ndarray) -> np.ndarray:
    """Convert conditional stopping probabilities to a distribution over stops.

    ``p_r = lambda_r * prod_{j<r} (1 - lambda_j)`` for ``r < R``
    ``p_R = prod_{j<R} (1 - lambda_j)``  — the last step absorbs the remainder.

    The result sums to exactly 1 by construction, which the tests assert.
    """
    lam = np.asarray(lambdas, dtype=np.float64)
    if lam.ndim != 1 or lam.size == 0:
        raise HaltingError("lambdas must be a non-empty 1-D array")
    if np.any(lam < 0.0) or np.any(lam > 1.0):
        raise HaltingError("lambdas must lie in [0, 1]")

    n = lam.size
    p = np.empty(n, dtype=np.float64)
    remaining = 1.0
    for r in range(n - 1):
        p[r] = lam[r] * remaining
        remaining *= 1.0 - lam[r]
    p[n - 1] = remaining
    return p


def geometric_prior(n_steps: int, p_stop: float) -> np.ndarray:
    """Truncated geometric prior over ``n_steps``, renormalized to sum to 1."""
    if not 0.0 < p_stop <= 1.0:
        raise HaltingError("p_stop must lie in (0, 1]")
    r = np.arange(n_steps, dtype=np.float64)
    prior = p_stop * (1.0 - p_stop) ** r
    total = prior.sum()
    if total <= 0:
        raise HaltingError("degenerate prior")
    return prior / total


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise HaltingError("distributions must have the same shape")
    mask = p > eps
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(np.maximum(q[mask], eps)))))


def halting_loss(
    lambdas: np.ndarray,
    per_step_task_loss: np.ndarray,
    *,
    prior_p_stop: float = 0.2,
    beta: float = 0.01,
) -> tuple[float, np.ndarray]:
    """``L = sum_r p_r * L_task(Y_r) + beta * KL(p || geometric_prior)``.

    Returns ``(loss, p)``. The task loss is evaluated at *every* potential
    stopping point, which is what makes an early stop a trained behaviour rather
    than a truncation.
    """
    p = stopping_distribution(lambdas)
    losses = np.asarray(per_step_task_loss, dtype=np.float64)
    if losses.shape != p.shape:
        raise HaltingError(
            f"expected {p.size} per-step losses, got {losses.size}"
        )
    prior = geometric_prior(p.size, prior_p_stop)
    return float(np.sum(p * losses) + beta * kl_divergence(p, prior)), p


def sample_stop(lambdas: np.ndarray, seed: int) -> int:
    """Serving policy A: sample the conditional decision with a recorded seed.

    This matches training semantics exactly, and the seed makes the run
    reproducible, which Architecture §6.5's cache-parity gate requires.
    """
    lam = np.asarray(lambdas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for r in range(lam.size - 1):
        if rng.random() < lam[r]:
            return r
    return lam.size - 1


def threshold_stop(lambdas: np.ndarray, cumulative_threshold: float = 0.9) -> int:
    """Serving policy B: deterministic cumulative-probability rule.

    Stops at the first ``r`` where the cumulative stopping mass reaches the
    threshold. This is a *separately calibrated* policy, not the same estimator
    as sampling: the tests assert it is deterministic and that its expected
    depth differs from the sampled mean, so the two are never conflated.
    """
    if not 0.0 < cumulative_threshold <= 1.0:
        raise HaltingError("threshold must lie in (0, 1]")
    p = stopping_distribution(lambdas)
    cumulative = np.cumsum(p)
    idx = int(np.searchsorted(cumulative, cumulative_threshold))
    return min(idx, p.size - 1)


def expected_depth(lambdas: np.ndarray) -> float:
    """Mean stopping index (0-based) under the stopping distribution."""
    p = stopping_distribution(lambdas)
    return float(np.sum(p * np.arange(p.size)))
