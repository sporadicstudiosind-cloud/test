"""Hyper-Connections / mHC: exactness at n=1, doubly-stochastic constraint,
exact parameter accounting, and the measured signal-gain comparison the
mHC paper's central claim is about (unconstrained mixing blows up with
depth; Sinkhorn-projected mixing does not).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from iridium.model.residual import (
    HyperConnections,
    HyperStack,
    StreamCollapse,
    StreamMix,
    expand_streams,
    sinkhorn_knopp,
)

torch.manual_seed(0)


# --------------------------------------------------------------------------
# exact parameter accounting
# --------------------------------------------------------------------------


@pytest.mark.parametrize("d,n,dynamic", [(8, 1, True), (8, 3, True), (16, 4, False), (16, 4, True)])
def test_hyperconnections_param_count_matches_module(d, n, dynamic):
    hc = HyperConnections(d, n, constrained=True, dynamic=dynamic)
    actual = sum(p.numel() for p in hc.parameters())
    assert actual == HyperConnections.param_count(d, n, dynamic)


@pytest.mark.parametrize("d,n,dynamic", [(8, 1, True), (8, 3, False), (16, 4, True)])
def test_streammix_param_count_matches_module(d, n, dynamic):
    mix = StreamMix(d, n, constrained=False, dynamic=dynamic)
    assert sum(p.numel() for p in mix.parameters()) == StreamMix.param_count(d, n, dynamic)


@pytest.mark.parametrize("d,n,dynamic", [(8, 1, True), (8, 3, True)])
def test_collapse_param_count_matches_module(d, n, dynamic):
    c = StreamCollapse(d, n, dynamic)
    assert sum(p.numel() for p in c.parameters()) == StreamCollapse.param_count(d, n, dynamic)


@pytest.mark.parametrize("d,n,depth,dynamic,constrained", [
    (8, 1, 3, True, True), (8, 4, 5, True, False), (16, 3, 4, False, True),
])
def test_hyperstack_param_count_matches_module(d, n, depth, dynamic, constrained):
    sublayers = [nn.Linear(d, d) for _ in range(depth)]
    stack = HyperStack(sublayers, d, n, constrained=constrained, dynamic=dynamic)
    total = sum(p.numel() for p in stack.parameters())
    sublayer_params = sum(p.numel() for s in sublayers for p in s.parameters())
    wrapping = total - sublayer_params
    assert wrapping == HyperStack.param_count(d, n, depth, dynamic)


# --------------------------------------------------------------------------
# Sinkhorn-Knopp: doubly-stochastic constraint
# --------------------------------------------------------------------------


def test_sinkhorn_output_is_doubly_stochastic():
    logits = torch.randn(2, 3, 5, 5) * 2.0
    m = sinkhorn_knopp(logits, iters=80)
    row_sums = m.sum(dim=-1)
    col_sums = m.sum(dim=-2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4)
    assert (m >= 0).all()


def test_streammix_constrained_init_is_near_identity():
    d, n = 8, 4
    mix = StreamMix(d, n, constrained=True, dynamic=False)
    summary = torch.zeros(1, 1, d)
    m = mix(summary)[0, 0]
    assert torch.allclose(m, torch.eye(n), atol=1e-6)


def test_streammix_unconstrained_init_is_exact_identity():
    d, n = 8, 4
    mix = StreamMix(d, n, constrained=False, dynamic=False)
    summary = torch.zeros(1, 1, d)
    m = mix(summary)[0, 0]
    assert torch.equal(m, torch.eye(n))


# --------------------------------------------------------------------------
# exact reduction to the plain residual stack at n_streams=1
# --------------------------------------------------------------------------


@pytest.mark.parametrize("constrained", [True, False])
@pytest.mark.parametrize("dynamic", [True, False])
def test_reduces_to_plain_residual_at_n1(constrained, dynamic):
    d, depth = 12, 6
    torch.manual_seed(1)
    sublayers = [nn.Linear(d, d) for _ in range(depth)]
    x0 = torch.randn(2, 5, d)

    plain = x0
    for s in sublayers:
        plain = plain + s(plain)

    stack = HyperStack(sublayers, d, n_streams=1, constrained=constrained,
                        dynamic=dynamic, sinkhorn_iters=20)
    out = stack(x0)

    assert torch.equal(out, plain), (
        f"n_streams=1 must reduce EXACTLY (0.0) to the plain residual stack, "
        f"got max abs diff {(out - plain).abs().max().item()}"
    )


def test_reduces_to_plain_residual_wrapping_a_full_residual_block():
    """delta=False: wrapping something that already adds its own residual."""
    d, depth = 10, 5
    torch.manual_seed(2)

    class FullResidualSublayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(d, d)

        def forward(self, x):
            return x + self.lin(x)   # residual already inside

    sublayers = [FullResidualSublayer() for _ in range(depth)]
    x0 = torch.randn(2, 4, d)

    plain = x0
    for s in sublayers:
        plain = s(plain)   # NOT plain + s(plain): s already adds it

    stack = HyperStack(sublayers, d, n_streams=1, constrained=True)
    out = stack(x0, delta=False)
    assert torch.allclose(out, plain, atol=1e-6)


# --------------------------------------------------------------------------
# gradients reach every map, including through Sinkhorn
# --------------------------------------------------------------------------


def test_gradients_are_finite_through_constrained_mixing():
    d, n, depth = 8, 3, 4
    sublayers = [nn.Linear(d, d) for _ in range(depth)]
    stack = HyperStack(sublayers, d, n, constrained=True, dynamic=True)
    x0 = torch.randn(2, 3, d, requires_grad=True)
    out = stack(x0)
    out.sum().backward()
    assert torch.isfinite(x0.grad).all()
    for p in stack.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


# --------------------------------------------------------------------------
# the measured signal-gain table
# --------------------------------------------------------------------------


def _stream_norm_trace(sublayers, x0, n, constrained, dynamic, sigma, seed):
    """Run one HyperStack forward, tracking the streams tensor's norm after
    every layer (not just the final collapse), and return the trace.

    ``sigma`` perturbs each layer's ``StreamMix.logits`` in place with
    independent Gaussian noise before the run -- the "after training has
    moved the mixing parameters" condition the mHC paper measures against.
    ``sigma=0`` is "at init".
    """
    torch.manual_seed(seed)
    d = x0.shape[-1]
    stack = HyperStack(sublayers, d, n, constrained=constrained,
                        dynamic=dynamic, sinkhorn_iters=30)
    if sigma:
        gen = torch.Generator().manual_seed(seed + 1)
        for hc in stack.connections:
            noise = torch.randn(hc.mix.logits.shape, generator=gen) * sigma
            hc.mix.logits.data += noise

    with torch.no_grad():
        streams = expand_streams(x0, n)
        norms = [streams.norm().item()]
        for hc, sub in zip(stack.connections, stack.sublayers):
            streams = hc(streams, sub, delta=True)
            norms.append(streams.norm().item())
    return norms


def _plain_norm_trace(sublayers, x0):
    with torch.no_grad():
        x = x0
        norms = [x.norm().item()]
        for s in sublayers:
            x = x + s(x)
            norms.append(x.norm().item())
    return norms


def test_signal_gain_table():
    """48-layer toy stack, random (small, fixed) sublayers.

    Reports max(||signal||) / ||signal_0|| across depth for:
      * plain residual (single stream)
      * Hyper-Connections, unconstrained mixing (n=4)
      * mHC, Sinkhorn-constrained mixing (n=4)
    both at init (identity mixing) and after perturbing every layer's raw
    mixing parameters with independent Gaussian noise -- the condition the
    mHC paper measures its ~3000x-vs-~1.6x figures under. Numbers here are
    from a random 48-layer toy stack, not a trained model, and are not
    claimed to reproduce the paper's exact figures; they are reproduced to
    demonstrate that the *mechanism* (unconstrained repeated matrix
    multiplication has unbounded spectral radius; a doubly-stochastic
    matrix is non-expansive) is real and measurable, not asserted from the
    papers on faith.
    """
    d, depth, n = 16, 48, 4
    torch.manual_seed(42)
    sublayers = [nn.Linear(d, d) for _ in range(depth)]
    with torch.no_grad():
        for s in sublayers:
            s.weight.mul_(0.05)   # small, fixed, "random sublayer" contribution
            s.bias.zero_()
    x0 = torch.randn(2, 3, d) * 1.0

    plain_norms = _plain_norm_trace(sublayers, x0)
    plain_gain = max(plain_norms) / plain_norms[0]

    at_init = {}
    perturbed = {}
    for constrained, label in [(False, "hc_unconstrained"), (True, "mhc_constrained")]:
        norms0 = _stream_norm_trace(sublayers, x0, n, constrained, dynamic=False,
                                     sigma=0.0, seed=7)
        at_init[label] = max(norms0) / norms0[0]

        normsp = _stream_norm_trace(sublayers, x0, n, constrained, dynamic=False,
                                     sigma=0.6, seed=7)
        perturbed[label] = max(normsp) / normsp[0]

    report = (
        "\n--- signal-gain table (48 layers, d=16, n_streams=4) ---\n"
        f"plain residual                         : {plain_gain:10.3f}x\n"
        f"HC unconstrained, at init (mix = I)     : {at_init['hc_unconstrained']:10.3f}x\n"
        f"mHC constrained, at init (mix = I)      : {at_init['mhc_constrained']:10.3f}x\n"
        f"HC unconstrained, mix perturbed sigma=.6: {perturbed['hc_unconstrained']:10.3f}x\n"
        f"mHC constrained, mix perturbed sigma=.6 : {perturbed['mhc_constrained']:10.3f}x\n"
    )
    print(report)

    # At init, identity mixing makes both HC variants coincide with plain
    # residual to floating-point tolerance (see test_reduces_to_plain_residual).
    assert at_init["hc_unconstrained"] == pytest.approx(plain_gain, rel=1e-3)
    assert at_init["mhc_constrained"] == pytest.approx(plain_gain, rel=1e-3)

    # After perturbation: mHC's doubly-stochastic mixing is non-expansive by
    # construction (a convex combination cannot exceed the max of its
    # inputs' norms), so its gain must stay close to what the sublayers
    # alone would produce.
    assert perturbed["mhc_constrained"] < 5.0 * plain_gain

    # Unconstrained mixing has no such bound; 48 independent perturbations of
    # a free 4x4 matrix compound multiplicatively. This is a sanity floor,
    # not the paper's number (that was measured at 27B over real training),
    # chosen from the actual measured value with headroom below it so the
    # test documents "large and unbounded" without being tied to one seed's
    # exact float.
    assert perturbed["hc_unconstrained"] > 20.0 * perturbed["mhc_constrained"]
