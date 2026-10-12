"""DyT, Derf, AdaptiveRMSNorm and the make_norm factory.

The factory's "rms" path must be bit-identical to ``layers.RMSNorm`` (not
merely equivalent), since that is what keeps a config that does not opt into
a new norm kind behaving exactly as it does today.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from iridium.model.layers import RMSNorm
from iridium.model.norms import (
    AdaptiveRMSNorm,
    Derf,
    DyT,
    make_norm,
    norm_param_count,
)

torch.manual_seed(0)


# --------------------------------------------------------------------------
# DyT
# --------------------------------------------------------------------------


def test_dyt_matches_closed_form():
    d = 6
    dyt = DyT(d, alpha_init=0.7)
    x = torch.randn(3, 4, d)
    expected = dyt.gamma * torch.tanh(0.7 * x) + dyt.beta
    assert torch.equal(dyt(x), expected)


def test_dyt_param_count_matches_module():
    d = 10
    dyt = DyT(d)
    assert sum(p.numel() for p in dyt.parameters()) == DyT.param_count(d)


def test_dyt_gradients_finite():
    dyt = DyT(8)
    x = torch.randn(2, 3, 8, requires_grad=True)
    dyt(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    for p in dyt.parameters():
        assert torch.isfinite(p.grad).all()


# --------------------------------------------------------------------------
# Derf
# --------------------------------------------------------------------------


def test_derf_matches_closed_form():
    d = 6
    derf = Derf(d, alpha_init=1.3, s_init=0.2)
    x = torch.randn(3, 4, d)
    expected = derf.gamma * torch.erf(1.3 * x + 0.2) + derf.beta
    assert torch.equal(derf(x), expected)


def test_derf_param_count_matches_module():
    d = 10
    derf = Derf(d)
    assert sum(p.numel() for p in derf.parameters()) == Derf.param_count(d)


def test_derf_gradients_finite():
    derf = Derf(8)
    x = torch.randn(2, 3, 8, requires_grad=True)
    derf(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    for p in derf.parameters():
        assert torch.isfinite(p.grad).all()


def test_derf_is_bounded_in_minus_one_one():
    # erf saturates to +-1; gamma=1, beta=0 at init, so the raw output must
    # lie in (-1, 1) regardless of how large x is.
    derf = Derf(4)
    x = torch.randn(2, 5, 4) * 1000.0
    out = derf(x)
    assert (out.abs() <= 1.0 + 1e-6).all()


# --------------------------------------------------------------------------
# make_norm factory
# --------------------------------------------------------------------------


def test_factory_rms_is_bit_identical_to_layers_rmsnorm():
    d = 12
    torch.manual_seed(3)
    a = make_norm("rms", d, eps=1e-5)
    torch.manual_seed(3)
    b = RMSNorm(d, eps=1e-5)
    assert type(a) is RMSNorm
    x = torch.randn(2, 3, d)
    assert torch.equal(a(x), b(x))


def test_factory_rms_returns_the_actual_class_not_a_copy():
    assert make_norm("rms", 8) .__class__ is RMSNorm


@pytest.mark.parametrize("kind,extra", [
    ("dyt", {}), ("derf", {}),
])
def test_factory_builds_requested_kind(kind, extra):
    norm = make_norm(kind, 8, **extra)
    x = torch.randn(2, 3, 8)
    out = norm(x)
    assert out.shape == x.shape


def test_factory_adaptive_needs_cond_dim_or_self_conditioned():
    norm = make_norm("adaptive", 8, cond_dim=4)
    x = torch.randn(2, 3, 8)
    c = torch.randn(2, 3, 4)
    assert norm(x, c).shape == x.shape


def test_factory_unknown_kind_raises():
    with pytest.raises(ValueError):
        make_norm("nonsense", 8)


def test_norm_param_count_matches_each_module():
    d, cond = 8, 5
    assert norm_param_count("rms", d) == d
    assert norm_param_count("dyt", d) == sum(p.numel() for p in DyT(d).parameters())
    assert norm_param_count("derf", d) == sum(p.numel() for p in Derf(d).parameters())
    assert norm_param_count("adaptive", d, cond_dim=cond) == sum(
        p.numel() for p in AdaptiveRMSNorm(d, cond_dim=cond).parameters()
    )


# --------------------------------------------------------------------------
# AdaptiveRMSNorm
# --------------------------------------------------------------------------


def test_adaptive_rmsnorm_at_init_equals_rmsnorm_exactly():
    d, cond_dim, eps = 16, 6, 1e-5
    torch.manual_seed(5)
    adaptive = AdaptiveRMSNorm(d, cond_dim=cond_dim, eps=eps)
    rms = RMSNorm(d, eps=eps)   # default weight is all-ones, matching gain=1
    x = torch.randn(2, 5, d)
    cond = torch.randn(2, 5, cond_dim)
    assert torch.allclose(adaptive(x, cond), rms(x), atol=1e-6)


def test_adaptive_rmsnorm_self_conditioned_at_init_equals_rmsnorm():
    d, eps = 16, 1e-5
    adaptive = AdaptiveRMSNorm(d, self_conditioned=True, eps=eps)
    rms = RMSNorm(d, eps=eps)
    x = torch.randn(2, 5, d)
    assert torch.allclose(adaptive(x), rms(x), atol=1e-6)


def test_adaptive_rmsnorm_param_count_matches_module():
    d, cond_dim = 10, 3
    a = AdaptiveRMSNorm(d, cond_dim=cond_dim)
    assert sum(p.numel() for p in a.parameters()) == AdaptiveRMSNorm.param_count(d, cond_dim)


def test_adaptive_rmsnorm_requires_cond_unless_self_conditioned():
    with pytest.raises(ValueError):
        AdaptiveRMSNorm(8)   # no cond_dim, not self_conditioned
    norm = AdaptiveRMSNorm(8, cond_dim=4)
    with pytest.raises(ValueError):
        norm(torch.randn(1, 1, 8))   # forward with no cond


def test_adaptive_rmsnorm_bound_limits_modulation():
    d, cond_dim, bound = 8, 4, 0.3
    norm = AdaptiveRMSNorm(d, cond_dim=cond_dim, bound=bound)
    # Push the gain/shift heads hard away from zero so tanh saturates, then
    # recompute gain/shift exactly as forward() does to check the bound
    # directly rather than through the normalised output.
    with torch.no_grad():
        nn.init.constant_(norm.to_gain.weight, 10.0)
        nn.init.constant_(norm.to_shift.weight, 10.0)
    cond = torch.randn(2, 3, cond_dim) * 5.0
    gain = 1.0 + bound * torch.tanh(norm.to_gain(cond))
    shift = bound * torch.tanh(norm.to_shift(cond))
    assert (gain >= 1 - bound - 1e-6).all() and (gain <= 1 + bound + 1e-6).all()
    assert (shift.abs() <= bound + 1e-6).all()


def test_adaptive_rmsnorm_gradients_finite_including_self_conditioned():
    for self_cond in (False, True):
        kwargs = {"self_conditioned": True} if self_cond else {"cond_dim": 4}
        norm = AdaptiveRMSNorm(8, **kwargs)
        x = torch.randn(2, 3, 8, requires_grad=True)
        cond = None if self_cond else torch.randn(2, 3, 4)
        out = norm(x) if self_cond else norm(x, cond)
        out.sum().backward()
        assert torch.isfinite(x.grad).all()
        for p in norm.parameters():
            assert torch.isfinite(p.grad).all()


# --------------------------------------------------------------------------
# autocast finiteness
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_all_norms_finite_under_autocast(dtype):
    d, cond_dim = 16, 4
    x = torch.randn(2, 5, d)
    cond = torch.randn(2, 5, cond_dim)
    norms_and_args = [
        (make_norm("rms", d), ()),
        (make_norm("dyt", d), ()),
        (make_norm("derf", d), ()),
        (make_norm("adaptive", d, cond_dim=cond_dim), (cond,)),
        (make_norm("adaptive", d, self_conditioned=True), ()),
    ]
    with torch.autocast(device_type="cpu", dtype=dtype):
        for norm, args in norms_and_args:
            out = norm(x, *args)
            assert torch.isfinite(out.float()).all(), norm.__class__.__name__
