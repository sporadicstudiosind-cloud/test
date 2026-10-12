"""DynamicLowRankLinear / DynamicSwiGLU: exact param accounting, exact
identity to a plain Linear/SwiGLU at init, and a measured (not claimed)
FLOPs/params overhead table.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from iridium.model.dynamic import DynamicLowRankLinear, DynamicSwiGLU
from iridium.model.layers import SwiGLU


@pytest.mark.parametrize("cond_dim", [None, 5])
def test_param_count_matches_numel_exactly(cond_dim):
    m = DynamicLowRankLinear(in_features=10, out_features=14, rank=3, cond_dim=cond_dim)
    formula = DynamicLowRankLinear.param_count(10, 14, 3, cond_dim=cond_dim)
    actual = sum(p.numel() for p in m.parameters())
    assert formula == actual


def test_output_shape():
    m = DynamicLowRankLinear(10, 14, rank=3)
    x = torch.randn(2, 7, 10)
    assert m(x).shape == (2, 7, 14)


def test_identical_to_plain_linear_at_init():
    """U is zero-init, so y = W x + U(...) == W x, for ANY V/gate init."""
    torch.manual_seed(0)
    m = DynamicLowRankLinear(10, 14, rank=3)
    reference = nn.Linear(10, 14, bias=False)
    with torch.no_grad():
        reference.weight.copy_(m.base.weight)
    x = torch.randn(5, 10)
    torch.testing.assert_close(m(x), reference(x), rtol=0, atol=0)


def test_default_conditioning_is_the_layer_input():
    """cond=None must use x itself, not some other default."""
    torch.manual_seed(0)
    m = DynamicLowRankLinear(6, 6, rank=2)
    with torch.no_grad():
        nn.init.normal_(m.U.weight, std=0.02)  # break the zero-init to exercise the gate
    x = torch.randn(4, 6)
    out_implicit = m(x)
    out_explicit = m(x, cond=x)
    torch.testing.assert_close(out_implicit, out_explicit, rtol=0, atol=0)


def test_dynamic_swiglu_param_count_matches_numel_exactly():
    m = DynamicSwiGLU(d_model=16, d_ff=40, rank=4)
    formula = DynamicSwiGLU.param_count(16, 40, 4)
    actual = sum(p.numel() for p in m.parameters())
    assert formula == actual


def test_dynamic_swiglu_matches_plain_swiglu_at_init():
    """Identical output to a SwiGLU built from the same base weights."""
    torch.manual_seed(0)
    d_model, d_ff = 16, 40
    dyn = DynamicSwiGLU(d_model, d_ff, rank=4)
    ref = SwiGLU(d_model, d_ff)
    with torch.no_grad():
        ref.gate.weight.copy_(dyn.gate.base.weight)
        ref.up.weight.copy_(dyn.up.base.weight)
        ref.down.weight.copy_(dyn.down.base.weight)

    x = torch.randn(3, 9, d_model)
    out_dyn = dyn(x)
    out_ref = ref(x)
    torch.testing.assert_close(out_dyn, out_ref, rtol=1e-6, atol=1e-6)


def test_dynamic_swiglu_forward_matches_reference_expression():
    torch.manual_seed(1)
    d_model, d_ff = 12, 20
    dyn = DynamicSwiGLU(d_model, d_ff, rank=3)
    x = torch.randn(2, 5, d_model)
    expected = dyn.down(F.silu(dyn.gate(x)) * dyn.up(x))
    torch.testing.assert_close(dyn(x), expected, rtol=0, atol=0)


def test_measured_overhead_vs_swiglu_at_d768_dff2048():
    """Report actual measured overhead, not an estimate, for rank 8/16/32."""
    d_model, d_ff = 768, 2048
    base_params = sum(p.numel() for p in SwiGLU(d_model, d_ff).parameters())
    rows = []
    for rank in (8, 16, 32):
        dyn = DynamicSwiGLU(d_model, d_ff, rank)
        dyn_params = sum(p.numel() for p in dyn.parameters())
        extra_params = dyn_params - base_params
        # Every extra parameter here belongs to a Linear applied once per
        # token, so extra forward FLOPs (multiply-add) are 2x the extra
        # weight count -- an exact consequence of the module shapes, not an
        # independent estimate.
        extra_flops_per_token = 2 * extra_params
        base_flops_per_token = 2 * base_params
        rows.append((rank, base_params, dyn_params, extra_params,
                     extra_params / base_params, extra_flops_per_token,
                     extra_flops_per_token / base_flops_per_token))
        assert dyn_params == DynamicSwiGLU.param_count(d_model, d_ff, rank)
        assert extra_params > 0

    print("\nDynamicSwiGLU overhead at d_model=768, d_ff=2048 (measured):")
    print(f"{'rank':>4} {'base params':>12} {'dyn params':>12} {'extra':>10} "
          f"{'extra %':>9} {'extra FLOPs/tok':>16} {'extra FLOPs %':>14}")
    for rank, base, dyn, extra, pct, fl, flpct in rows:
        print(f"{rank:>4} {base:>12,} {dyn:>12,} {extra:>10,} "
              f"{pct*100:>8.2f}% {fl:>16,} {flpct*100:>13.2f}%")
