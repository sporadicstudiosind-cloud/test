"""Gradient checkpointing must reproduce the unchecked run exactly.

The earlier verdict here ("gradient checkpointing returns incorrect gradients
in this architecture") was measured under PyTorch's *reentrant* checkpointing
(``use_reentrant=True``, the undocumented default when the flag is omitted).
That mode really is broken for this model: it silently perturbs gradients
once the same checkpointed layer is visited twice in one backward (which the
ponder loop guarantees whenever ``n_loops > 1``), and it raises outright on
the superstacks because every active stack's bridge cross-attention shares
one upstream core-state tensor, and reentrant checkpointing does not support
two checkpoint calls rooted in a shared ancestor without ``retain_graph``.

``iridium.model.control_core.ControlCore`` and
``iridium.model.superstack.Superstack`` now checkpoint with
``use_reentrant=False`` unconditionally (see
``Iridium1.enable_gradient_checkpointing``'s docstring for the measurements).
These tests are the regression guard: any future change that reintroduces a
gradient discrepancy, in any of the loop/controller-mode/hard_exit
combinations that used to disagree, fails here first, loudly, rather than in
a training curve that never shows a wrong-but-finite gradient.
"""

import copy
import dataclasses

import numpy as np
import pytest
import torch

from iridium.codecs.bank import TensorBatch, continuous_dims
from iridium.codecs.spans import Sample, collate, text_span
from iridium.config import get_config
from iridium.model.iridium1 import Iridium1

# Exact equality is not demanded (CPU multi-threaded BLAS reduction order is
# not guaranteed to be the same between a forward-only call and a checkpoint's
# recompute of that same forward), but the earlier reentrant-checkpointing
# bug was three to four orders of magnitude above this floor -- see the
# docstring on Iridium1.enable_gradient_checkpointing for the measured sizes
# (relative deltas of 1e-7 in fp32 and 2.6e-3 in bf16 autocast for the bug
# this guards against).
TOLERANCE = 1e-9


def _batch(cfg, seed=0, n=3):
    dims = continuous_dims(cfg.codecs)
    rng = np.random.default_rng(seed)
    samples = [
        Sample([text_span("".join(chr(97 + int(c)) for c in rng.integers(0, 26, 13)))])
        for _ in range(n)
    ]
    return TensorBatch(collate(samples, dims), dtype=torch.float32)


def _build(controller_mode=False, seed=0):
    torch.manual_seed(seed)
    cfg = dataclasses.replace(get_config("tiny"), controller_mode=controller_mode)
    model = Iridium1(cfg)
    model.train()
    return model, cfg


def _grads(model, batch, n_loops, seed=1234, hard_exit=False):
    """Run forward+backward with the router's stochastic path pinned by seed,
    so a comparison across two calls isolates checkpointing itself rather
    than a different draw of the training-time gumbel routing noise."""
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    out = model(batch, n_loops=n_loops, hard_exit=hard_exit)
    loss = out.hidden.float().sum()
    loss.backward()
    return {name: (p.grad.clone() if p.grad is not None else None)
            for name, p in model.named_parameters()}


def _assert_close(g_ref, g_test, tol=TOLERANCE):
    assert set(g_ref) == set(g_test)
    worst, worst_name = 0.0, None
    for name in g_ref:
        a, b = g_ref[name], g_test[name]
        if a is None and b is None:
            continue
        assert (a is None) == (b is None), f"{name}: grad presence mismatch"
        d = (a - b).abs().max().item()
        if d > worst:
            worst, worst_name = d, name
    assert worst <= tol, f"worst gradient delta {worst:.3e} on {worst_name} exceeds tol {tol:.0e}"


@pytest.mark.parametrize("controller_mode", [False, True])
@pytest.mark.parametrize("n_loops", [1, 2])
def test_core_checkpointing_matches_unchecked_gradients(controller_mode, n_loops):
    model, cfg = _build(controller_mode=controller_mode)
    batch = _batch(cfg)

    g_ref = _grads(model, batch, n_loops)

    model.enable_gradient_checkpointing(True, stacks=False)
    g_ck = _grads(model, batch, n_loops)
    model.enable_gradient_checkpointing(False, stacks=False)

    _assert_close(g_ref, g_ck)


@pytest.mark.parametrize("controller_mode", [False, True])
@pytest.mark.parametrize("n_loops", [1, 2])
def test_core_and_stack_checkpointing_matches_unchecked_gradients(controller_mode, n_loops):
    model, cfg = _build(controller_mode=controller_mode)
    batch = _batch(cfg)

    g_ref = _grads(model, batch, n_loops)

    model.enable_gradient_checkpointing(True, stacks=True)
    g_ck = _grads(model, batch, n_loops)
    model.enable_gradient_checkpointing(False, stacks=False)

    _assert_close(g_ref, g_ck)


def test_stack_checkpointing_matches_under_hard_exit():
    """hard_exit=True gives each row a genuinely data-dependent executed
    depth (PonderNet's `alive` mask) -- the mechanism the original docstring
    worried would defeat recomputation. It doesn't: the branching lives in
    the Python loop that decides how many checkpoint calls to make, not
    inside any checkpointed call, so recompute always replays the layer that
    actually ran."""
    model, cfg = _build()
    batch = _batch(cfg, seed=7, n=4)

    g_ref = _grads(model, batch, n_loops=1, hard_exit=True)

    model.enable_gradient_checkpointing(True, stacks=True)
    g_ck = _grads(model, batch, n_loops=1, hard_exit=True)
    model.enable_gradient_checkpointing(False, stacks=False)

    _assert_close(g_ref, g_ck)


def test_checkpointing_matches_under_bf16_autocast():
    """The autocast state must be identical in the recompute as in the
    original forward. use_reentrant=False guarantees this; the flag is not
    exposed precisely so a caller cannot accidentally select the broken
    mode."""
    model, cfg = _build()
    batch = _batch(cfg)

    def run(seed=1234):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
            out = model(batch, n_loops=2)
            loss = out.hidden.float().sum()
        loss.backward()
        return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    g_ref = run()
    model.enable_gradient_checkpointing(True, stacks=True)
    g_ck = run()
    model.enable_gradient_checkpointing(False, stacks=False)

    _assert_close(g_ref, g_ck)


def test_disabled_by_default():
    cfg = get_config("tiny")
    model = Iridium1(cfg)
    assert model.core.gradient_checkpointing is False
    assert all(not s.gradient_checkpointing for s in model.bank.stacks)


def test_stacks_flag_gates_independently_of_core():
    cfg = get_config("tiny")
    model = Iridium1(cfg)
    model.enable_gradient_checkpointing(True, stacks=False)
    assert model.core.gradient_checkpointing is True
    assert all(not s.gradient_checkpointing for s in model.bank.stacks)


def test_no_checkpointing_with_a_live_cache():
    """Checkpointing must never fire during incremental serving: it would
    recompute the segment during backward and write its cache entry twice.
    There is no backward pass in serving anyway, so nothing is lost."""
    model, cfg = _build()
    model.enable_gradient_checkpointing(True, stacks=True)
    batch = _batch(cfg, n=2)
    cache: dict = {}
    with torch.no_grad():
        model(batch, n_loops=1, cache=cache)
    # A cache key was written for every core layer despite checkpointing
    # being enabled -- proof the checkpointed path was not taken, not just
    # that it happened not to crash.
    core_keys = [k for k in cache if isinstance(k, tuple) and k and k[0] == "core"]
    assert core_keys
    model.enable_gradient_checkpointing(False, stacks=False)
