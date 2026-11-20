"""Dense recurrent core: masks, cache parity, isolation.

Gates: §18.3 "Causality", "Isolation", "Cache correctness".

The uncached path is the reference (§6.5). Every assertion about the cached
path is an assertion that it reproduces the uncached one.
"""

import pytest

torch = pytest.importorskip("torch")

from iridium.model.core import (  # noqa: E402
    IridiumCore,
    RecurrencePolicy,
    causal_mask,
    output_block_mask,
    workspace_mask,
)
from iridium.model.inventory import TransformerConfig  # noqa: E402

TINY = TransformerConfig(
    name="iridium-1-tiny-test",
    d_model=64,
    n_prelude=1,
    n_core=2,
    n_coda=1,
    d_ff=176,
    n_query_heads=4,
    n_kv_heads=2,
    d_head=16,
)


@pytest.fixture
def model() -> "IridiumCore":
    torch.manual_seed(0)
    m = IridiumCore(TINY, max_recurrence=6).double().eval()
    return m


def make_input(t: int = 12, b: int = 2) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(b, t, TINY.d_model, dtype=torch.float64)


def test_forward_runs_and_shapes_hold(model):
    x = make_input()
    out, _ = model(x, causal_mask(x.shape[1]), RecurrencePolicy(2))
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_backward_produces_gradients_for_every_parameter(model):
    """Every parameter must be reachable when the head that owns it is active.

    The halting head only participates when a stopping decision is requested,
    so the loss has to include it; a run with ``collect_halt=False`` legitimately
    leaves it unused.
    """
    x = make_input()
    out, lambdas = model(
        x, causal_mask(x.shape[1]), RecurrencePolicy(2), collect_halt=True
    )
    loss = out.square().mean() + sum(lam.mean() for lam in lambdas)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_halting_head_is_unused_when_no_stop_is_requested(model):
    x = make_input()
    out, lambdas = model(x, causal_mask(x.shape[1]), RecurrencePolicy(2))
    out.square().mean().backward()
    assert lambdas == []
    assert model.halt_head.weight.grad is None


def test_no_expert_router_exists(model):
    """Invariant 2, checked structurally rather than asserted in prose."""
    names = [n.lower() for n, _ in model.named_modules()]
    banned = ("expert", "router", "gate_network", "moe", "switch")
    assert not [n for n in names if any(b in n for b in banned)]


def test_recurrence_changes_the_output(model):
    """A weight-tied core must still behave differently at different depths."""
    x = make_input()
    mask = causal_mask(x.shape[1])
    a, _ = model(x, mask, RecurrencePolicy(1))
    b, _ = model(x, mask, RecurrencePolicy(3))
    assert not torch.allclose(a, b, atol=1e-6)


def test_depth_beyond_maximum_rejected(model):
    with pytest.raises(ValueError, match="exceeds max_recurrence"):
        model(make_input(), None, RecurrencePolicy(99))


def test_causal_mask_blocks_the_future(model):
    """Invariant 5: perturbing a future token cannot change a committed one."""
    x = make_input(t=10, b=1)
    mask = causal_mask(x.shape[1])
    base, _ = model(x, mask, RecurrencePolicy(2))

    perturbed = x.clone()
    perturbed[:, 7:, :] += 5.0
    after, _ = model(perturbed, mask, RecurrencePolicy(2))

    assert torch.allclose(base[:, :7], after[:, :7], atol=1e-10)
    assert not torch.allclose(base[:, 7:], after[:, 7:], atol=1e-6)


def test_output_block_mask_is_bidirectional_inside_and_closed_outside():
    """§6.3: a noised block sees itself and its prefix, never a clean future."""
    prefix, block = 5, 4
    mask = output_block_mask(prefix, block)[0, 0]

    # Prefix stays causal.
    assert mask[2, 3].item() < 0
    assert mask[2, 2].item() == 0.0
    # Block attends bidirectionally within itself.
    assert mask[prefix, prefix + block - 1].item() == 0.0
    assert mask[prefix + block - 1, prefix].item() == 0.0
    # Prefix never attends forward into the block.
    assert mask[0, prefix].item() < 0


def test_output_block_perturbation_does_not_reach_the_prefix(model):
    prefix, block = 6, 4
    x = make_input(t=prefix + block, b=1)
    mask = output_block_mask(prefix, block)
    base, _ = model(x, mask, RecurrencePolicy(2))

    perturbed = x.clone()
    perturbed[:, prefix:, :] += 3.0
    after, _ = model(perturbed, mask, RecurrencePolicy(2))
    assert torch.allclose(base[:, :prefix], after[:, :prefix], atol=1e-10)


def test_workspace_mask_isolates_concurrent_streams(model):
    """§8.1 / Invariant 4, for the attention component of isolation."""
    stream_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])
    mask = workspace_mask(stream_ids)
    x = make_input(t=stream_ids.shape[0], b=1)

    base, _ = model(x, mask, RecurrencePolicy(2))
    perturbed = x.clone()
    perturbed[:, 3:6, :] += 10.0                 # scribble on workspace 1
    after, _ = model(perturbed, mask, RecurrencePolicy(2))

    assert torch.allclose(base[:, 0:3], after[:, 0:3], atol=1e-10)
    assert torch.allclose(base[:, 6:8], after[:, 6:8], atol=1e-10)
    assert not torch.allclose(base[:, 3:6], after[:, 3:6], atol=1e-6)


def test_shared_prefix_is_readable_by_every_stream():
    stream_ids = torch.tensor([9, 0, 0, 1, 1])
    mask = workspace_mask(stream_ids, n_shared_prefix=1)[0, 0]
    assert mask[1, 0].item() == 0.0              # stream 0 reads shared record
    assert mask[3, 0].item() == 0.0              # stream 1 reads it too
    assert mask[3, 1].item() < 0                 # but not each other


def test_cached_incremental_decode_matches_uncached_reference(model, exact_attention):
    """§18.3 cache-correctness gate, at FP64 so the tolerance is meaningful.

    Pinned to the manual attention path: FP64 only makes the tolerance
    meaningful if the arithmetic is FP64 on both sides, and SDPA's fused
    kernels are not obliged to keep it there. See tests/conftest.py."""
    torch.manual_seed(3)
    full = make_input(t=9, b=1)
    depth = RecurrencePolicy(2)

    reference, _ = model(full, causal_mask(full.shape[1]), depth)

    cache: dict = {}
    outputs = []
    for i in range(full.shape[1]):
        step, _ = model(full[:, i: i + 1, :], None, depth, cache=cache)
        outputs.append(step)
    incremental = torch.cat(outputs, dim=1)

    assert torch.allclose(reference, incremental, rtol=1e-9, atol=1e-9)


def test_cache_is_keyed_by_recurrence_index(model):
    """Recurrences must not collapse into a single unlabeled KV array."""
    x = make_input(t=3, b=1)
    cache: dict = {}
    model(x, None, RecurrencePolicy(3), cache=cache)

    core_keys = sorted(k for k in cache if k[0] == "core")
    recurrence_indices = {k[1] for k in core_keys}
    assert recurrence_indices == {0, 1, 2}
    assert len(core_keys) == 3 * TINY.n_core


def test_changing_depth_invalidates_the_cache(model):
    """A cache built at one depth must not be reused at another."""
    x = make_input(t=4, b=1)
    cache_d2: dict = {}
    model(x, None, RecurrencePolicy(2), cache=cache_d2)
    cache_d3: dict = {}
    model(x, None, RecurrencePolicy(3), cache=cache_d3)

    keys_d2 = {k for k in cache_d2 if k[0] == "core"}
    keys_d3 = {k for k in cache_d3 if k[0] == "core"}
    assert keys_d2 != keys_d3
    assert keys_d2 < keys_d3                     # strict subset


def test_halting_lambdas_are_probabilities(model):
    x = make_input()
    _, lambdas = model(x, causal_mask(x.shape[1]), RecurrencePolicy(4),
                       collect_halt=True)
    assert len(lambdas) == 4
    for lam in lambdas:
        assert torch.all(lam >= 0.0) and torch.all(lam <= 1.0)
