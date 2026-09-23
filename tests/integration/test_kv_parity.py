"""The parity gate.

Cached incremental decoding must compute the *same function* as an uncached
forward pass over the whole sequence.

The standard is float64 agreement to ``1e-13`` absolute, not bit-equality.
Bit-equality is not achievable and demanding it would be a false gate: a
``[1, T, d] @ [1, d, T]`` matmul and a ``[1, 1, d] @ [1, d, T]`` matmul reduce
in different orders inside BLAS, so the two paths differ by one or two ULP
(observed: 4.4e-16) even when they are computing exactly the same expression.
Anything larger than round-off is a real divergence, and the tolerance is three
orders of magnitude above the observed noise and twelve below any mechanism
error this file is written to catch.

Three mechanisms in this architecture can break it, and each fails silently:

* superstacks keep sparse stack-local KV, so a token's stack-local history must
  be the same set whether gathered by a mask or accumulated by a cache;
* the bridge reads core stage-I states for the whole stream;
* the ponder loop indexes the cache, so a layer run twice must not alias.

A failure here means the serving path computes something other than what was
trained, and every downstream metric is describing a different model.
"""

import numpy as np
import pytest
import torch

from iridium.codecs.bank import TensorBatch, continuous_dims
from iridium.codecs.spans import Sample, Span, collate, text_span
from iridium.config import get_config
from iridium.model.iridium1 import Iridium1
from iridium.runtime.decode import atomic_chunks, run_atomic_chunked, run_chunked


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Iridium1(get_config("tiny")).double().eval()


def text_batch(seed=0, n=2):
    dims = continuous_dims(get_config("tiny").codecs)
    rng = np.random.default_rng(seed)
    samples = [
        Sample([text_span("".join(chr(97 + int(c)) for c in rng.integers(0, 26, 11)))])
        for _ in range(n)
    ]
    return TensorBatch(collate(samples, dims), dtype=torch.float64)


def mixed_batch(seed=0, n=2):
    cfg = get_config("tiny")
    dims = continuous_dims(cfg.codecs)
    samples = []
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        samples.append(Sample([
            text_span("abc"),
            Span("field", rng.normal(size=(4, dims["field"])), grid=(2, 2),
                 supervised=False, observed=True),
            text_span("defg"),
        ]))
    return TensorBatch(collate(samples, dims), dtype=torch.float64)


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 11])
def test_single_loop_decoding_is_bit_exact(model, chunk):
    """With one loop the two paths agree bit for bit, which is worth pinning."""
    batch = text_batch()
    with torch.no_grad():
        reference = model(batch, n_loops=1).hidden
        cached = run_chunked(model, batch, chunk=chunk, n_loops=1)
    assert torch.equal(reference, cached)


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 11])
@pytest.mark.parametrize("loops", [1, 2])
def test_text_decoding_matches_the_uncached_reference(model, chunk, loops):
    batch = text_batch()
    with torch.no_grad():
        reference = model(batch, n_loops=loops).hidden
        cached = run_chunked(model, batch, chunk=chunk, n_loops=loops)
    delta = float((reference - cached).abs().max())
    assert delta <= 1e-13, f"chunk={chunk} loops={loops} max delta {delta:.3e}"


@pytest.mark.parametrize("chunk", [1, 4])
@pytest.mark.parametrize("loops", [1, 2])
def test_mixed_modality_decoding_matches_with_atomic_chunks(model, chunk, loops):
    batch = mixed_batch()
    with torch.no_grad():
        reference = model(batch, n_loops=loops).hidden
        cached = run_atomic_chunked(model, batch, chunk=chunk, n_loops=loops)
    assert float((reference - cached).abs().max()) <= 1e-13


def test_splitting_a_field_span_across_chunks_changes_the_answer(model):
    """Not a bug to paper over: a spectral block needs the whole grid.

    This is the constraint that ``atomic_chunks`` exists to respect, and the
    test asserts the failure is loud rather than a quiet quality regression.
    """
    batch = mixed_batch()
    with torch.no_grad():
        reference = model(batch, n_loops=1).hidden
        naive = run_chunked(model, batch, chunk=1, n_loops=1)
    assert not torch.allclose(reference, naive, atol=1e-6)


def test_supervised_field_future_cannot_change_earlier_hidden_states(model):
    dims = continuous_dims(get_config("tiny").codecs)
    target = np.zeros((4, dims["field"]), dtype=np.float64)

    def batch_for(payload):
        sample = Sample([
            text_span("abc", supervised=False),
            Span("field", payload, grid=(2, 2), supervised=True),
            text_span("done"),
        ])
        return TensorBatch(collate([sample], dims), dtype=torch.float64)

    altered = target.copy()
    altered[-1] = 100.0
    with torch.no_grad():
        before = model(batch_for(target), n_loops=1).hidden
        after = model(batch_for(altered), n_loops=1).hidden
    # The first field patch predicts the next one. It cannot inspect the last
    # target patch through atomic routing or a bidirectional spectral block.
    torch.testing.assert_close(before[:, :4], after[:, :4], atol=0, rtol=0)
    assert not torch.equal(before[:, -1], after[:, -1])


def test_atomic_chunks_never_split_a_grid(model):
    batch = mixed_batch()
    bounds = atomic_chunks(batch, 1)
    starts = {lo for lo, _ in bounds}
    ends = {hi for _, hi in bounds}
    for b, start, shape in batch.grids:
        n = int(np.prod(shape))
        interior = set(range(start + 1, start + n))
        assert not (interior & starts)
        assert not (interior & ends)


def test_prefill_and_decode_agree(model):
    """A prompt processed in one shot then continued must match one pass."""
    batch = text_batch(seed=3, n=1)
    total = int(batch.modality.shape[1])
    with torch.no_grad():
        reference = model(batch, n_loops=1).hidden
        cache: dict = {}
        from iridium.runtime.decode import slice_batch

        first = model(slice_batch(batch, 0, total - 3), n_loops=1, cache=cache).hidden
        rest = [first]
        for t in range(total - 3, total):
            rest.append(
                model(slice_batch(batch, t, t + 1), n_loops=1, cache=cache).hidden
            )
        combined = torch.cat(rest, dim=1)
    assert float((reference - combined).abs().max()) <= 1e-13
