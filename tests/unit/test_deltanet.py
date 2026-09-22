"""Gated DeltaNet: chunkwise and recurrent are the same function, exactly.

The one claim worth pinning here is that :func:`chunkwise_forward` (the
two-pass, zero-state-per-chunk reformulation described in
``iridium/model/deltanet.py``) and :func:`recurrent_forward` (a plain
token-by-token loop) compute *the same recurrence*, for every chunk size,
including ones that do not divide the sequence length. That is the KV-parity
gate's whole idea (``tests/integration/test_kv_parity.py``) applied to a
layer with no KV cache at all: whatever the fast training path does, decoding
must reproduce, one token at a time, to float precision -- not "close in
practice".
"""
from __future__ import annotations

import math

import pytest
import torch

from iridium.model.deltanet import (
    GatedDeltaNet,
    chunkwise_forward,
    recurrent_forward,
)
from iridium.model.layers import causal_keep


def _random_inputs(b, h, t, d, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, h, t, d, generator=g, dtype=torch.float32).to(dtype)
    k = torch.randn(b, h, t, d, generator=g, dtype=torch.float32).to(dtype)
    v = torch.randn(b, h, t, d, generator=g, dtype=torch.float32).to(dtype)
    alpha = torch.sigmoid(torch.randn(b, h, t, generator=g, dtype=torch.float32)).to(dtype)
    beta = torch.sigmoid(torch.randn(b, h, t, generator=g, dtype=torch.float32)).to(dtype)
    # q, k are L2-normalised by the module before either path ever sees them;
    # the free functions don't do that themselves, so this test does it, to
    # exercise the same regime (unit-norm keys/queries) the layer runs under.
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return q, k, v, alpha, beta


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 5, 7, 11, 13, 32, 64])
@pytest.mark.parametrize("t", [1, 5, 13, 29])
def test_chunkwise_matches_recurrent_fp32(chunk_size, t):
    """1e-5 fp32 agreement, every chunk size, including ones that don't divide T."""
    q, k, v, alpha, beta = _random_inputs(2, 3, t, 6, torch.float32, seed=t * 100 + chunk_size)
    out_chunk = chunkwise_forward(q, k, v, alpha, beta, chunk_size)
    out_rec, _ = recurrent_forward(q, k, v, alpha, beta)
    delta = (out_chunk - out_rec).abs().max().item()
    assert delta <= 1e-5, f"t={t} chunk_size={chunk_size} max delta {delta:.3e}"


@pytest.mark.parametrize("chunk_size", [1, 3, 4, 6, 11])
@pytest.mark.parametrize("t", [7, 17])
def test_chunkwise_matches_recurrent_fp64(chunk_size, t):
    """1e-9 float64 agreement -- the tight tolerance the module docstring promises."""
    q, k, v, alpha, beta = _random_inputs(2, 2, t, 5, torch.float64, seed=t * 10 + chunk_size)
    out_chunk = chunkwise_forward(q, k, v, alpha, beta, chunk_size)
    out_rec, _ = recurrent_forward(q, k, v, alpha, beta)
    delta = (out_chunk - out_rec).abs().max().item()
    assert delta <= 1e-9, f"t={t} chunk_size={chunk_size} max delta {delta:.3e}"


def test_chunkwise_rejects_bad_chunk_size():
    q, k, v, alpha, beta = _random_inputs(1, 1, 4, 4, torch.float32, seed=0)
    with pytest.raises(ValueError):
        chunkwise_forward(q, k, v, alpha, beta, 0)


# -- the module (projections, gates, conv, cache protocol) ------------------

def _layer(d_model=24, n_heads=3, d_head=8, conv_size=4, chunk_size=5, dtype=torch.float64):
    torch.manual_seed(0)
    return GatedDeltaNet(d_model, n_heads, d_head, conv_size=conv_size, chunk_size=chunk_size).to(dtype)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 5, 8, 16, 64])
def test_module_decode_matches_full_sequence(chunk_size):
    """Same standard the repo's own KV-parity gate holds attention layers to.

    A prompt processed as one chunkwise call, then continued one token at a
    time through the cache, must agree with a single chunkwise call over the
    whole sequence -- to fp64 tolerance, since that is what
    ``tests/integration/test_kv_parity.py`` demands of every other layer.
    """
    layer = _layer(chunk_size=chunk_size)
    b, t = 2, 11
    torch.manual_seed(3)
    x = torch.randn(b, t, 24, dtype=torch.float64)
    positions = torch.arange(t)[None].expand(b, t)
    with torch.no_grad():
        reference = layer(x, positions, causal_keep(t, t))

        cache: dict = {}
        outs = []
        for i in range(t):
            step_keep = causal_keep(1, i + 1)
            o = layer(x[:, i:i + 1], positions[:, i:i + 1], step_keep, cache=cache, cache_key=("l0",))
            outs.append(o)
        decoded = torch.cat(outs, dim=1)
    delta = (reference - decoded).abs().max().item()
    assert delta <= 1e-9, f"chunk_size={chunk_size} max delta {delta:.3e}"


def test_prefill_chunk_then_decode_matches_full_sequence():
    """A multi-token prefill through the cache, then single tokens, still agrees."""
    layer = _layer()
    b, t = 1, 15
    torch.manual_seed(4)
    x = torch.randn(b, t, 24, dtype=torch.float64)
    positions = torch.arange(t)[None]
    with torch.no_grad():
        reference = layer(x, positions, causal_keep(t, t))

        cache: dict = {}
        prefill = 9
        first = layer(x[:, :prefill], positions[:, :prefill], causal_keep(prefill, prefill),
                      cache=cache, cache_key=("l0",))
        rest = [first]
        for i in range(prefill, t):
            step_keep = causal_keep(1, i + 1)
            rest.append(layer(x[:, i:i + 1], positions[:, i:i + 1], step_keep, cache=cache, cache_key=("l0",)))
        combined = torch.cat(rest, dim=1)
    delta = (reference - combined).abs().max().item()
    assert delta <= 1e-9, f"max delta {delta:.3e}"


def test_param_count_matches_module():
    for d_model, n_heads, d_head, conv_size in [(24, 3, 8, 4), (16, 4, 4, 3), (64, 8, 8, 5)]:
        layer = GatedDeltaNet(d_model, n_heads, d_head, conv_size=conv_size)
        counted = sum(p.numel() for p in layer.parameters())
        formula = GatedDeltaNet.param_count(d_model, n_heads, d_head, conv_size)
        assert counted == formula, (d_model, n_heads, d_head, conv_size, counted, formula)


def test_state_bytes_independent_of_sequence_length():
    """The whole point of this layer: decode state size does not depend on T."""
    layer = _layer(chunk_size=4)
    b = 1
    cache: dict = {}
    torch.manual_seed(5)
    for t_total in (1, 50, 500):
        x = torch.randn(b, t_total, 24, dtype=torch.float64)
        positions = torch.arange(t_total)[None]
        cache.clear()
        with torch.no_grad():
            for i in range(t_total):
                step_keep = causal_keep(1, i + 1)
                layer(x[:, i:i + 1], positions[:, i:i + 1], step_keep, cache=cache, cache_key=("l0",))
        state, cq, ck, cv = cache[("l0",)]
        assert state.shape == (b, layer.n_heads, layer.d_head, layer.d_head)
        assert cq.shape == (b, layer.d_qkv, layer.conv_size - 1)
        reported = GatedDeltaNet.state_bytes(layer.n_heads, layer.d_head, layer.conv_size,
                                              bytes_per_element=8)
        actual = state.numel() * state.element_size() + sum(
            c.numel() * c.element_size() for c in (cq, ck, cv)
        )
        assert reported == actual, (t_total, reported, actual)


def test_padding_rows_are_zero_and_do_not_leak():
    """A padded key never enters the state; a padded query's own row is zero."""
    layer = _layer(dtype=torch.float32)
    b, t = 2, 10
    torch.manual_seed(6)
    x = torch.randn(b, t, 24)
    positions = torch.arange(t)[None].expand(b, t)
    valid = torch.ones(b, t, dtype=torch.bool)
    valid[1, 4:] = False  # second sample: real content in [0, 4), padding after
    keep = causal_keep(t, t) & valid.view(b, 1, 1, t)

    with torch.no_grad():
        out = layer(x, positions, keep)
        isolated = layer(x[0:1], positions[0:1], causal_keep(t, t))

    assert torch.equal(out[1, 4:], torch.zeros_like(out[1, 4:]))
    # Padding in sample 1 must not perturb sample 0's independent computation.
    assert torch.equal(out[0], isolated[0])


@pytest.mark.parametrize("t", [512, 4096])
def test_long_sequence_stays_finite(t):
    """Repeated sigmoid gates over thousands of steps: no NaN/Inf, tiny width."""
    layer = _layer(d_model=8, n_heads=2, d_head=4, chunk_size=32, dtype=torch.float32)
    torch.manual_seed(7)
    x = torch.randn(1, t, 8) * 0.5
    positions = torch.arange(t)[None]
    with torch.no_grad():
        out = layer(x, positions, causal_keep(t, t))
    assert torch.isfinite(out).all()
    assert out.abs().max().item() < 1e4


def test_positions_argument_is_ignored():
    """No RoPE: two different position tensors must give the same output."""
    layer = _layer(dtype=torch.float32)
    b, t = 1, 6
    torch.manual_seed(8)
    x = torch.randn(b, t, 24)
    keep = causal_keep(t, t)
    with torch.no_grad():
        out_a = layer(x, torch.arange(t)[None], keep)
        out_b = layer(x, torch.arange(100, 100 + t)[None], keep)
    assert torch.equal(out_a, out_b)
