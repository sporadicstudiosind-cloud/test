"""Muon variants: Polar Express coefficients, NorMuon row normalisation, the
WSD decay shapes, and the global ponder KL.

Each test checks a claim that would be wrong in a plausible but broken
implementation: Polar Express must actually tighten the singular-value band
(not just be a different table), row normalisation must not change the
update's overall size, the WSD decay must start where the schedule says and
reject unknown shapes, and the global KL must be the KL of the batch-mean
distribution (not the mean of per-token KLs).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from iridium.model.router import geometric_prior, global_ponder_kl, ponder_kl
from iridium.training.muon import (
    POLAR_EXPRESS,
    Muon,
    _quintic_coefficients,
    _row_normalise,
    muon_param_groups,
    zeropower_via_newtonschulz5,
)
from iridium.training.trainer import MUON_VARIANTS, TrainConfig, wsd_lr


def _scalar_poly(x: np.ndarray, coefficients) -> np.ndarray:
    """Apply the Newton-Schulz quintic ``a x + (b x^3 + c x^5)`` as the matrix
    iteration does, on singular values ``x`` (i.e. ``X X^T`` -> ``x^2``)."""
    y = x.copy()
    for a, b, c in coefficients:
        A = y * y
        B = b * A + c * (A * A)
        y = a * y + B * y
    return y


# --------------------------------------------------------------------------
# 1. Polar Express coefficients
# --------------------------------------------------------------------------

def test_polar_express_table_shape_and_fallback():
    assert len(POLAR_EXPRESS) == 8
    # Iterations past the table reuse its last entry (scaled by the same margin).
    coeffs = _quintic_coefficients("polar_express", 10)
    assert len(coeffs) == 10
    assert coeffs[7] == coeffs[8] == coeffs[9]


def test_unknown_coefficient_kind_raises():
    with pytest.raises(ValueError):
        _quintic_coefficients("nope", 5)
    with pytest.raises(ValueError):
        zeropower_via_newtonschulz5(torch.randn(8, 8), steps=5, coefficients="nope")


def test_scalar_band_polar_express_tighter_than_jordan():
    # The claim in the POLAR_EXPRESS comment: on singular values in [1e-3, 1]
    # after five steps, Polar Express stays above 0.8 while Jordan's fixed
    # polynomial dips below 0.6.
    x = np.logspace(-3, 0, 200)
    pe = _scalar_poly(x, _quintic_coefficients("polar_express", 5))
    jordan = _scalar_poly(x, _quintic_coefficients("jordan", 5))
    assert pe.min() > 0.8
    assert jordan.min() < 0.6
    assert pe.max() < 1.15


def test_polar_express_matrix_singular_values_in_band_and_tighter():
    torch.manual_seed(0)
    G = torch.randn(32, 64, dtype=torch.float32)
    pe = zeropower_via_newtonschulz5(G, steps=5, coefficients="polar_express")
    jo = zeropower_via_newtonschulz5(G, steps=5, coefficients="jordan")
    assert pe.dtype == torch.float32 and pe.shape == G.shape
    s_pe = torch.linalg.svdvals(pe)
    s_jo = torch.linalg.svdvals(jo)
    assert float(s_pe.min()) >= 0.8 and float(s_pe.max()) <= 1.15
    spread_pe = float(s_pe.max() - s_pe.min())
    spread_jo = float(s_jo.max() - s_jo.min())
    assert spread_pe < spread_jo


# --------------------------------------------------------------------------
# 2. Muon(coefficients="polar_express", row_norm=True)
# --------------------------------------------------------------------------

def _tiny_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32))


def _quadratic_loss(model, x, target):
    return ((model(x) - target) ** 2).mean()


def test_muon_normuon_steps_reduce_loss_and_track_row_sq():
    model = _tiny_model()
    groups = muon_param_groups(model, lr=0.02, weight_decay=0.0)
    opt = Muon(groups, lr=0.02, coefficients="polar_express", row_norm=True)

    torch.manual_seed(1)
    x = torch.randn(16, 32)
    target = torch.randn(16, 32)

    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = _quadratic_loss(model, x, target)
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    with torch.no_grad():
        losses.append(float(_quadratic_loss(model, x, target)))
    assert losses[-1] < losses[0]

    muon_params = [p for g in opt.param_groups if g["use_muon"] for p in g["params"]]
    assert muon_params, "muon_param_groups should route the two Linear weights to Muon"
    for p in muon_params:
        state = opt.state[p]
        assert state["row_sq"].shape == (32,)
        assert torch.isfinite(state["row_sq"]).all()
        assert "momentum_buffer" in state


def test_muon_state_dict_round_trip():
    model = _tiny_model()
    opt = Muon(muon_param_groups(model, lr=0.02), lr=0.02,
               coefficients="polar_express", row_norm=True)
    x = torch.randn(8, 32)
    target = torch.randn(8, 32)
    for _ in range(2):
        opt.zero_grad()
        _quadratic_loss(model, x, target).backward()
        opt.step()

    saved = opt.state_dict()

    other_model = _tiny_model()
    other = Muon(muon_param_groups(other_model, lr=0.02), lr=0.02,
                 coefficients="polar_express", row_norm=True)
    other.load_state_dict(saved)

    assert other.coefficients == "polar_express" and other.row_norm is True
    assert len(other.state) == len(opt.state)
    own = [p for g in opt.param_groups for p in g["params"]]
    loaded = [p for g in other.param_groups for p in g["params"]]
    for p_src, p_dst in zip(own, loaded):
        s_src, s_dst = opt.state[p_src], other.state[p_dst]
        assert s_src.keys() == s_dst.keys()
        for k in s_src:
            if torch.is_tensor(s_src[k]):
                assert torch.allclose(s_src[k], s_dst[k].to(s_src[k].device))
            else:
                assert s_src[k] == s_dst[k]


def test_row_normalise_equalises_row_scale():
    """NorMuon divides each row by its running RMS (no bias correction), so
    rows that start at very different scales come out comparable."""
    torch.manual_seed(2)
    update = torch.randn(32, 32) * torch.logspace(-2, 1, 32).unsqueeze(1)
    row_sq = torch.zeros(32)
    out = _row_normalise(update, row_sq, beta2=0.0)
    rms = out.pow(2).mean(1).sqrt()
    assert out.shape == update.shape
    assert torch.allclose(rms, torch.ones(32), rtol=1e-3)
    torch.testing.assert_close(row_sq, update.pow(2).mean(1))


# --------------------------------------------------------------------------
# 3. MUON_VARIANTS and wsd_lr decay shapes
# --------------------------------------------------------------------------

def test_muon_variants_keys_and_settings():
    assert set(MUON_VARIANTS) == {"muon", "muon_pe", "normuon"}
    assert MUON_VARIANTS["muon"] == {}
    assert MUON_VARIANTS["muon_pe"] == {"coefficients": "polar_express"}
    assert MUON_VARIANTS["normuon"] == {"coefficients": "polar_express", "row_norm": True}


def _wsd_cfg(**kw):
    base = dict(steps=100, lr=1.0, decay_ratio=0.2, min_lr_ratio=0.0,
                decay_shape="linear", warmup_ratio=0.0)
    base.update(kw)
    return TrainConfig(**base)


def test_wsd_linear_decay_points():
    cfg = _wsd_cfg()
    # Decay starts at round(100 * 0.8) = 80: still at the peak before it.
    assert wsd_lr(79, cfg) == pytest.approx(1.0)
    assert wsd_lr(90, cfg) == pytest.approx(0.5, abs=1e-6)
    assert wsd_lr(100, cfg) == pytest.approx(0.0, abs=1e-12)


def test_wsd_sqrt_decay_is_one_minus_sqrt_t():
    cfg = _wsd_cfg(decay_shape="sqrt")
    # t = (90 - 80) / (100 - 80) = 0.5, so 1 - sqrt(0.5).
    assert wsd_lr(90, cfg) == pytest.approx(1.0 - math.sqrt(0.5), abs=1e-6)
    assert wsd_lr(79, cfg) == pytest.approx(1.0)
    assert wsd_lr(100, cfg) == pytest.approx(0.0, abs=1e-12)


def test_wsd_unknown_decay_shape_raises_in_decay_phase_only():
    cfg = _wsd_cfg(decay_shape="cubic")
    # Stable phase (after the default 50-step warmup, before step 80) never
    # consults the shape.
    assert wsd_lr(60, cfg) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        wsd_lr(90, cfg)


# --------------------------------------------------------------------------
# 4. global_ponder_kl
# --------------------------------------------------------------------------

def test_global_kl_zero_when_every_token_equals_prior():
    prior = geometric_prior(3, 0.4, "cpu", torch.float32)
    p = prior.expand(2, 5, 3).clone()
    assert float(global_ponder_kl(p, prior)) == pytest.approx(0.0, abs=1e-6)


def test_global_kl_equals_kl_of_mean_distribution():
    torch.manual_seed(3)
    prior = geometric_prior(3, 0.4, "cpu", torch.float32)
    p = torch.softmax(torch.randn(2, 5, 3), dim=-1)
    mean = p.reshape(-1, 3).mean(0)
    expected = float((mean * (mean.log() - prior.log())).sum())
    assert float(global_ponder_kl(p, prior)) == pytest.approx(expected, rel=1e-5)


def test_global_kl_ignores_padding():
    torch.manual_seed(4)
    prior = geometric_prior(3, 0.4, "cpu", torch.float32)
    p = torch.softmax(torch.randn(2, 5, 3), dim=-1)
    valid = torch.ones(2, 5, dtype=torch.bool)
    valid[:, 3:] = False
    # Garbage in the padded positions must not change the result.
    corrupted = p.clone()
    corrupted[:, 3:] = torch.softmax(torch.randn(2, 2, 3) * 10, dim=-1)
    clean = global_ponder_kl(p[:, :3], prior)
    masked = global_ponder_kl(corrupted, prior, valid=valid)
    assert float(masked) == pytest.approx(float(clean), rel=1e-5)


def test_global_kl_bounded_above_by_per_token_kl():
    # Jensen: KL is convex in its first argument, so KL(mean p || q) <=
    # mean_t KL(p_t || q).
    torch.manual_seed(5)
    prior = geometric_prior(3, 0.4, "cpu", torch.float32)
    for _ in range(5):
        p = torch.softmax(torch.randn(2, 5, 3), dim=-1)
        g = float(global_ponder_kl(p, prior))
        per_token = float(ponder_kl(p, prior))
        assert g <= per_token + 1e-6
