"""Routing decisions must be executable at sampling time.

Every test here is about causality or about the balance objective meaning what
it claims. A router that reads the future trains a policy that cannot be run.
"""

import pytest
import torch

from iridium.codecs.spans import Span
from iridium.config import get_config
from iridium.model.router import (
    MacroRouter,
    geometric_prior,
    pool_over_spans,
    stopping_distribution,
)


def build(top_k=2, n_stacks=4, d=32):
    from iridium.config import RouterConfig

    return MacroRouter(d, n_stacks, RouterConfig(top_k=top_k), min_depth=1, max_depth=8)


def test_focus_is_causal():
    """Perturbing the future must not change an earlier token's focus.

    The source plan computed focus from a chunk mean, which fails this.
    """
    torch.manual_seed(0)
    router = build().eval()
    h = torch.randn(2, 10, 32)
    base = router(h)
    perturbed = h.clone()
    perturbed[:, 5:] += 7.0
    after = router(perturbed)
    assert torch.equal(base.focus[:, :5], after.focus[:, :5])
    assert torch.equal(base.stack_index[:, :5], after.stack_index[:, :5])


def test_prefix_summary_survives_chunk_boundaries():
    """The running statistic must come from the cache, not restart per chunk."""
    torch.manual_seed(0)
    router = build().eval()
    h = torch.randn(1, 12, 32)
    whole = router.prefix_summary(h, 0, None)
    cache: dict = {}
    parts = [router.prefix_summary(h[:, i : i + 3], 0, cache) for i in range(0, 12, 3)]
    chunked = torch.cat(parts, dim=1)
    assert torch.allclose(whole, chunked, atol=1e-6)


def test_top_k_weights_sum_to_one():
    torch.manual_seed(0)
    router = build(top_k=2).eval()
    decision = router(torch.randn(3, 7, 32))
    assert torch.allclose(decision.stack_weight.sum(-1), torch.ones(3, 7), atol=1e-6)


def test_every_token_reaches_exactly_top_k_stacks():
    torch.manual_seed(0)
    n_stacks, k = 4, 2
    router = build(top_k=k, n_stacks=n_stacks).eval()
    decision = router(torch.randn(2, 9, 32))
    assert decision.stack_index.shape[-1] == k
    for b in range(2):
        for t in range(9):
            assert len(set(decision.stack_index[b, t].tolist())) == k


def test_balance_loss_penalises_dispatch_probability_correlation():
    """``alpha * N * sum_i f_i P_i`` penalises *agreement* between f and P.

    A detail worth pinning down, because it is easy to misread the objective
    as "penalise uneven dispatch": when P is uniform the loss is exactly
    ``alpha`` for *any* dispatch, since ``sum_i f_i P_i = (1/N) sum_i f_i``.
    The term only bites when the router is both confident and concentrated,
    which is the correct target - f carries no gradient anyway.
    """
    router = build(top_k=1, n_stacks=4).eval()
    confident = torch.tensor([[[0.70, 0.10, 0.10, 0.10]] * 8])
    collapsed = router._balance_loss(
        confident, torch.zeros(1, 8, 1, dtype=torch.long)
    )
    spread = router._balance_loss(
        confident, torch.arange(4).repeat(2).view(1, 8, 1)
    )
    assert collapsed > spread

    uniform = torch.full((1, 8, 4), 0.25)
    flat_a = router._balance_loss(uniform, torch.zeros(1, 8, 1, dtype=torch.long))
    flat_b = router._balance_loss(uniform, torch.arange(4).repeat(2).view(1, 8, 1))
    assert float(flat_a) == pytest.approx(float(flat_b))


def test_balance_loss_carries_its_coefficient():
    from iridium.config import RouterConfig

    strong = MacroRouter(32, 4, RouterConfig(balance_alpha=1.0), 1, 8).eval()
    weak = MacroRouter(32, 4, RouterConfig(balance_alpha=0.01), 1, 8).eval()
    probs = torch.full((1, 4, 4), 0.25)
    index = torch.zeros(1, 4, 1, dtype=torch.long)
    assert strong._balance_loss(probs, index) > weak._balance_loss(probs, index) * 50


def test_capacity_dropping_is_refused_as_non_causal():
    from iridium.config import RouterConfig

    router = MacroRouter(32, 4, RouterConfig(capacity_factor=1.25), 1, 8)
    with pytest.raises(NotImplementedError, match="causally"):
        router(torch.randn(1, 4, 32))


def test_span_pooling_makes_a_span_route_as_one():
    logits = torch.randn(1, 6, 4)
    span_id = torch.tensor([[-1, 0, 0, 0, -1, -1]])
    pooled = pool_over_spans(logits, span_id)
    assert torch.allclose(pooled[0, 1], pooled[0, 2])
    assert torch.allclose(pooled[0, 2], pooled[0, 3])
    assert torch.equal(pooled[0, 0], logits[0, 0])
    assert torch.equal(pooled[0, 4], logits[0, 4])


def test_span_pooling_requires_observed():
    """Pooling reads later positions; a span still being generated has none."""
    import numpy as np

    with pytest.raises(ValueError, match="causal"):
        Span("field", np.zeros((4, 8), dtype=np.float32), grid=(2, 2),
             observed=False, atomic=True)


def test_stopping_distribution_sums_to_one():
    lam = torch.rand(5, 9, dtype=torch.double)
    p = stopping_distribution(lam)
    assert torch.allclose(p.sum(-1), torch.ones(5, dtype=torch.double), atol=1e-12)
    assert bool((p >= 0).all())


def test_torch_stopping_matches_the_numpy_contract():
    """``model/halting.py`` is the contract; the torch version must obey it."""
    import numpy as np

    from iridium.model.halting import stopping_distribution as numpy_version

    lam = torch.rand(4, 7, dtype=torch.double)
    got = stopping_distribution(lam).numpy()
    for row in range(4):
        want = numpy_version(lam[row].numpy())
        assert np.allclose(got[row], want, atol=1e-15)


def test_geometric_prior_is_normalised():
    prior = geometric_prior(6, 0.3, dtype=torch.double)
    assert float(prior.sum()) == pytest.approx(1.0)
    assert bool((prior[:-1] >= prior[1:]).all())
