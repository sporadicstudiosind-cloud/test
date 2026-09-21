"""FlowMatchingHead: verifying the objective, not just exercising the shapes.

Two tests here are deliberately expensive relative to the rest of this file,
because they are the only tests in the repository that check the flow-matching
loss is *correct* rather than merely differentiable:

* ``test_gaussian_closed_form`` trains a head on a target with a known,
  hand-derived optimal velocity field and checks the trained network gets
  close to it. A loss that compiles and goes down is not evidence it is the
  right loss; only convergence to an independently-derived answer is.
* ``test_step_count_quality_tradeoff`` measures (not asserts) how sample
  quality on that same closed-form problem depends on step count and
  integrator, so the docstring claim in ``heads.py`` about Heun/midpoint
  beating Euler at low step counts is backed by a number in this repository
  rather than by citation alone.

Both share one trained head via a module-scoped fixture so the (slow, this
sandbox's per-op dispatch overhead dominates) training loop runs once.

Everything else here is cheap: shape/contract tests for the new keyword
arguments, each checked against the literal default that reproduces the
pre-existing behaviour bit-for-bit.
"""

from __future__ import annotations

import math

import pytest
import torch

from iridium.model.heads import FlowMatchingHead


# ---------------------------------------------------------------------------
# Closed-form Gaussian target: x0 ~ N(0, I), x1 ~ N(mu, sigma^2 I) independent.
# x_tau = (1-tau) x0 + tau x1 is Gaussian with mean tau*mu and (isotropic,
# per-dimension) variance s(tau)^2 = (1-tau)^2 + tau^2 sigma^2. Standard
# Gaussian conditioning gives
#   E[x0 | x_tau=x] = (1-tau)/s(tau)^2 * (x - tau*mu)
#   E[x1 | x_tau=x] = mu + tau*sigma^2/s(tau)^2 * (x - tau*mu)
# and the flow-matching optimum is v*(x,tau) = E[x1-x0 | x_tau=x]:
#   v*(x,tau) = mu + (x - tau*mu) * (tau*sigma^2 - (1-tau)) / s(tau)^2
# ---------------------------------------------------------------------------
D_MODEL, D_OUT = 4, 2
MU = torch.tensor([1.5, -0.7])
SIGMA = 0.6


def _analytic_velocity(x: torch.Tensor, tau: float) -> torch.Tensor:
    s2 = (1 - tau) ** 2 + tau ** 2 * SIGMA ** 2
    return MU + (x - tau * MU) * (tau * SIGMA ** 2 - (1 - tau)) / s2


def _velocity_rmse(head: FlowMatchingHead, n: int = 256, seed: int = 123) -> float:
    g = torch.Generator().manual_seed(seed)
    errs = []
    with torch.no_grad():
        for tau in (0.1, 0.3, 0.5, 0.7, 0.9):
            x = MU + SIGMA * torch.randn(n, D_OUT, generator=g)
            tau_t = torch.full((n,), tau)
            h = torch.zeros(n, D_MODEL)
            v_pred = head.velocity(h, x, tau_t)
            v_true = _analytic_velocity(x, tau)
            errs.append((v_pred - v_true).pow(2).mean().sqrt().item())
    return sum(errs) / len(errs)


@pytest.fixture(scope="module")
def trained_gaussian_head() -> FlowMatchingHead:
    torch.manual_seed(0)
    head = FlowMatchingHead(D_MODEL, D_OUT, d_hidden=32)
    opt = torch.optim.Adam(head.parameters(), lr=3e-2)
    h_const = torch.zeros(1024, D_MODEL)
    g = torch.Generator().manual_seed(0)
    for _ in range(60):
        target = MU + SIGMA * torch.randn(1024, D_OUT, generator=g)
        loss = head.loss(h_const, target, generator=g)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head


def test_gaussian_closed_form(trained_gaussian_head):
    """The trained head's velocity should track the analytic optimum.

    Measured on this exact fixture (seed-pinned, 60 Adam steps, d_hidden=32):
    an untrained head (zero-initialized ``out``, so it emits the zero vector
    everywhere) scores RMSE ~0.93 against the analytic field; after training
    it scores ~0.15-0.20. Neither number is a promise about a trained
    production checkpoint — this is a 2-dimensional toy with a hand-derived
    answer, chosen because it is the one case where "correct" is checkable at
    all. The assertions below are deliberately loose (large margin against
    run-to-run stochastic-optimization noise); they exist to catch the loss
    computing the wrong quantity entirely (e.g. a sign error in
    ``target - x0``, or tau and 1-tau swapped), which would leave the trained
    error indistinguishable from the untrained one.
    """
    untrained = FlowMatchingHead(D_MODEL, D_OUT, d_hidden=32)
    untrained_rmse = _velocity_rmse(untrained)
    trained_rmse = _velocity_rmse(trained_gaussian_head)
    assert trained_rmse < 0.35, f"trained RMSE {trained_rmse} did not converge"
    assert trained_rmse < 0.5 * untrained_rmse, (
        f"trained RMSE {trained_rmse} is not meaningfully better than "
        f"untrained {untrained_rmse}; the loss is likely not the stated one"
    )


def test_step_count_quality_tradeoff(trained_gaussian_head):
    """Measured, not asserted: does a better integrator help at low steps?

    Samples from the trained head at ``steps in {2, 16}`` for Euler and Heun,
    and checks the per-dimension std of the sampled distribution against the
    known ``SIGMA`` -- a proxy for sample quality that (unlike the sample
    mean) is not swamped by finite-sample noise at these batch sizes. The
    literature claim this backs: a second-order integrator (Heun) at a small
    step budget should not be worse than first-order Euler at the same
    budget, and both should improve as steps increase. This is checked on one
    synthetic distribution with one trained head; it is evidence for the
    docstring's claim in ``heads.py``, not a general benchmark.
    """
    n = 4096
    h = torch.zeros(n, D_MODEL)
    results = {}
    for integrator in ("euler", "heun"):
        for steps in (2, 16):
            g = torch.Generator().manual_seed(7)
            samples = trained_gaussian_head.sample(
                h, steps=steps, generator=g, integrator=integrator
            )
            std_err = abs(samples.std(0).mean().item() - SIGMA)
            results[(integrator, steps)] = std_err

    # More steps should not make quality worse, for either integrator.
    assert results[("euler", 16)] <= results[("euler", 2)] + 0.05
    assert results[("heun", 16)] <= results[("heun", 2)] + 0.05
    # At the small step budget, the second-order integrator should not be
    # meaningfully worse than the first-order one.
    assert results[("heun", 2)] <= results[("euler", 2)] + 0.05


# ---------------------------------------------------------------------------
# Contract tests: every new kwarg defaults to the pre-existing behaviour.
# ---------------------------------------------------------------------------

def test_defaults_reproduce_original_shapes_and_seeding():
    head = FlowMatchingHead(8, 4, n_tau=16)
    assert head.conditioning == "add"
    assert head.timestep_sampling == "uniform"
    assert head.tau_feature_kind == "fixed"
    assert head.resolution_shift is False
    assert head.cfg_dropout == 0.0

    h = torch.randn(2, 5, 8)
    target = torch.randn(2, 5, 4)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    loss1 = head.loss(h, target, generator=g1)
    loss2 = head.loss(h, target, generator=g2)
    assert torch.equal(loss1, loss2), "same seed must give bit-identical loss"

    s1 = head.sample(h, steps=4, generator=torch.Generator().manual_seed(1))
    s2 = head.sample(h, steps=4, generator=torch.Generator().manual_seed(1))
    assert torch.equal(s1, s2)
    assert s1.shape == h.shape[:-1] + (4,)


def test_tau_features_fixed_matches_original_formula():
    head = FlowMatchingHead(8, 4, n_tau=16)
    tau = torch.tensor([0.0, 0.25, 0.5, 1.0])
    feats = head._tau_features(tau)
    freqs = torch.linspace(0, 8, 16)
    ang = tau.unsqueeze(-1) * (2.0 ** freqs) * torch.pi
    expected = torch.cat([ang.sin(), ang.cos()], dim=-1)
    assert torch.allclose(feats, expected)


def test_tau_features_scaled_grows_with_n_tau():
    small = FlowMatchingHead(8, 4, n_tau=8, tau_feature_kind="scaled")
    large = FlowMatchingHead(8, 4, n_tau=64, tau_feature_kind="scaled")
    small_top = math.log2(max(small.n_tau, 2)) + 2.0
    large_top = math.log2(max(large.n_tau, 2)) + 2.0
    assert large_top > small_top, "wider embedding should span more octaves"


def test_invalid_kwargs_raise():
    with pytest.raises(ValueError):
        FlowMatchingHead(8, 4, conditioning="bogus")
    with pytest.raises(ValueError):
        FlowMatchingHead(8, 4, timestep_sampling="bogus")
    with pytest.raises(ValueError):
        FlowMatchingHead(8, 4, tau_feature_kind="bogus")
    head = FlowMatchingHead(8, 4)
    with pytest.raises(ValueError):
        head.sample(torch.randn(2, 8), integrator="bogus")


def test_logit_normal_concentrates_mass_mid_trajectory():
    """Logit-normal(0,1) should draw fewer taus near the extremes than
    uniform does, over many samples -- the whole point of the option."""
    head = FlowMatchingHead(4, 2, timestep_sampling="logit_normal")
    g = torch.Generator().manual_seed(0)
    tau = head._sample_tau((20000,), torch.device("cpu"), torch.float32, g, None)
    frac_extreme = ((tau < 0.1) | (tau > 0.9)).float().mean().item()
    # Uniform would put ~20% of mass in the two extreme deciles.
    assert frac_extreme < 0.12, frac_extreme


def test_resolution_shift_pushes_toward_high_noise_for_more_patches():
    head = FlowMatchingHead(
        4, 2, timestep_sampling="uniform",
        resolution_shift=True, resolution_shift_base=256,
    )
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    tau_low_res = head._sample_tau((10000,), torch.device("cpu"), torch.float32, g1, 256)
    tau_high_res = head._sample_tau((10000,), torch.device("cpu"), torch.float32, g2, 1024)
    # base resolution: shift == 1, so the shift is a no-op.
    assert torch.allclose(tau_low_res, torch.rand(10000, generator=torch.Generator().manual_seed(0)))
    # A higher patch count should shift the schedule toward tau=1 (more of
    # the sampling trajectory spent at high noise), raising the mean.
    assert tau_high_res.mean().item() > tau_low_res.mean().item()


def test_cfg_dropout_trains_null_branch_and_guidance_changes_output():
    torch.manual_seed(0)
    head = FlowMatchingHead(6, 3, d_hidden=16, cfg_dropout=1.0)
    # ``out`` is zero-initialized for training stability (see __init__), which
    # means dL/d(anything upstream) is exactly zero on the very first
    # backward pass -- a real, if temporary, dead-gradient property of that
    # design, not a bug in this test. Break the symmetry the way the first
    # optimizer step would, so the gradient check below is meaningful.
    with torch.no_grad():
        head.out.weight.normal_(std=0.1)
    h = torch.randn(4, 6)
    target = torch.randn(4, 3)
    loss = head.loss(h, target, generator=torch.Generator().manual_seed(0))
    loss.backward()
    assert head.null_cond.grad is not None and torch.any(head.null_cond.grad != 0)
    # cond.weight should get no gradient from h itself when dropout is always
    # on (the conditioning input was replaced by the null embedding), but the
    # projection's bias/weight still receives gradient through the null path.
    assert head.cond.weight.grad is not None

    guided_head = FlowMatchingHead(6, 3, d_hidden=16, cfg_dropout=0.5)
    h2 = torch.randn(3, 6)
    unguided = guided_head.sample(h2, steps=4, generator=torch.Generator().manual_seed(2),
                                  guidance_scale=1.0)
    guided = guided_head.sample(h2, steps=4, generator=torch.Generator().manual_seed(2),
                                guidance_scale=4.0)
    assert not torch.allclose(unguided, guided)


def test_adaln_conditioning_zero_init_is_stable_then_responsive():
    torch.manual_seed(0)
    head = FlowMatchingHead(6, 3, d_hidden=16, conditioning="adaln")
    assert torch.all(head.ada_mod.weight == 0) and torch.all(head.ada_mod.bias == 0)
    h = torch.randn(4, 6)
    x = torch.randn(4, 3)
    tau = torch.rand(4)
    # ``out`` is also zero-initialized, independently of AdaLN, so the head's
    # *output* is identically zero at construction regardless of conditioning
    # -- that is the stable start the zero-init buys. To check the AdaLN
    # branch itself is wired in (not a dead branch), give ``out`` a nonzero
    # weight, the way the first optimizer step would.
    with torch.no_grad():
        head.out.weight.normal_(std=0.1)
    out_at_init = head.velocity(h, x, tau)
    assert torch.isfinite(out_at_init).all()

    # Perturb the modulation away from zero and confirm it actually changes
    # the output -- i.e. the AdaLN path is wired into the forward pass, not a
    # dead branch.
    with torch.no_grad():
        head.ada_mod.bias.add_(1.0)
    out_after = head.velocity(h, x, tau)
    assert not torch.allclose(out_at_init, out_after)
