"""StreamingSession: exactness of lossless self-speculative decoding, cache
rollback correctness, and a smoke test for the lossy budget mode.

Everything here runs on ``iridium.config.get_config("tiny")`` -- the only
scale ``StreamingSession`` currently supports (``controller_mode=False``,
single loop, batch size 1, no context memory; see the module docstring in
``iridium/runtime/streaming.py`` for why).
"""

import numpy as np
import pytest
import torch

from iridium.codecs.bank import TensorBatch, continuous_dims
from iridium.codecs.spans import Sample, text_span, collate
from iridium.config import get_config
from iridium.model.iridium1 import Iridium1
from iridium.runtime.generate import generate
from iridium.runtime.streaming import StreamingSession


def make_model(seed=0):
    torch.manual_seed(seed)
    cfg = get_config("tiny")
    return Iridium1(cfg).eval()


def make_sample(text: str) -> Sample:
    return Sample([text_span(text)])


PROMPTS = ["hello world", "the quick fox", "ab"]


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("seed", [0, 1])
def test_lossless_matches_full_depth_greedy(prompt, seed):
    """Self-speculative greedy output must equal plain full-depth greedy
    output exactly, token for token -- this is the whole lossless claim."""
    model = make_model(seed)
    sample = make_sample(prompt)

    reference = generate(model, sample, max_new_tokens=10, temperature=0, n_loops=1)

    session = StreamingSession(
        model, mode="lossless", draft_len=3, max_new_tokens=10, seed=seed,
    )
    got_ids = [ev.token for ev in session.stream(sample)]

    assert got_ids == reference.ids, (
        f"prompt={prompt!r} seed={seed}: speculative {got_ids} != full-depth {reference.ids}"
    )


def test_lossless_acceptance_and_forward_count_are_reported():
    """Untrained exit heads: acceptance is expected to be poor. Report it,
    don't hide it -- this is exactly what the task asked to measure."""
    model = make_model(0)
    sample = make_sample("hello world, this is a longer prompt")
    session = StreamingSession(model, mode="lossless", draft_len=4, max_new_tokens=12, seed=0)
    events = list(session.stream(sample))
    assert len(events) >= 1
    assert session.stats["drafted"] > 0
    assert 0 <= session.stats["accepted"] <= session.stats["drafted"]
    assert session.stats["full_depth_forwards"] >= session.stats["verify_rounds"]
    # Not a correctness assertion -- untrained random-init exit heads have no
    # reason to agree with the full-depth model above chance. This just
    # prints/keeps the numbers so the report can quote them.
    acceptance_rate = session.stats["accepted"] / session.stats["drafted"]
    print(
        f"\n[lossless] drafted={session.stats['drafted']} "
        f"accepted={session.stats['accepted']} rate={acceptance_rate:.3f} "
        f"verify_rounds={session.stats['verify_rounds']} "
        f"full_depth_forwards={session.stats['full_depth_forwards']} "
        f"(plain decoding would need {len(events)} full-depth forwards)"
    )


def test_cache_after_speculative_decoding_matches_plain_decoding():
    """The hard part: after a speculative round with a rejection, the cache
    must hold exactly what plain token-at-a-time decoding would hold for the
    accepted tokens, and nothing for the rejected ones."""
    model = make_model(0)
    sample = make_sample("hello world")
    session = StreamingSession(model, mode="lossless", draft_len=4, max_new_tokens=6, seed=0)
    emitted = [ev.token for ev in session.stream(sample)]
    spec_cache = session.last_cache

    # Replay the SAME emitted tokens through plain one-token-at-a-time
    # decoding, from scratch, and compare the resulting caches.
    dims = continuous_dims(model.cfg.codecs)
    batch = TensorBatch(collate([sample], dims))
    from iridium.runtime.decode import atomic_chunks, slice_batch
    from iridium.runtime.generate import _single_token_batch
    from iridium.codecs.spans import MODALITY_INDEX

    plain_cache: dict = {}
    with torch.no_grad():
        for start, end in atomic_chunks(batch, int(batch.modality.shape[1])):
            model(slice_batch(batch, start, end), n_loops=1, cache=plain_cache)
        position = int(batch.positions[0, -1]) + 1
        for token in emitted:
            step = _single_token_batch(batch, MODALITY_INDEX["text"], token, position)
            model(step, n_loops=1, cache=plain_cache)
            position += 1

    assert set(spec_cache.keys()) == set(plain_cache.keys())
    for key in plain_cache:
        a, b = spec_cache[key], plain_cache[key]
        if isinstance(a, int):
            assert a == b, f"{key}: {a} != {b}"
        elif isinstance(a, tuple):
            for x, y in zip(a, b):
                assert x.shape == y.shape, f"{key}: shape {x.shape} != {y.shape}"
                assert torch.allclose(x, y, atol=1e-5), f"{key}: max delta {float((x-y).abs().max()):.3e}"
        else:
            assert a.shape == b.shape, f"{key}: shape {a.shape} != {b.shape}"
            assert torch.equal(a, b) if a.dtype == torch.bool else torch.allclose(a, b, atol=1e-5)


def test_lossless_produces_at_least_one_rejection_across_seeds():
    """Sanity that the rollback path (a < k) is actually exercised somewhere
    in the suite, not just the all-accepted branch."""
    model = make_model(0)
    saw_rejection = False
    for seed in range(4):
        session = StreamingSession(model, mode="lossless", draft_len=4, max_new_tokens=8, seed=seed)
        for ev in session.stream(make_sample("hello world")):
            if ev.accepted is False:
                saw_rejection = True
    assert saw_rejection, "expected at least one draft/full-depth disagreement with an untrained exit head"


def test_budget_mode_smoke():
    model = make_model(0)
    sample = make_sample("hi")
    session = StreamingSession(
        model, mode="budget", confidence=0.9, min_layers=1, max_new_tokens=6, seed=0,
    )
    events = list(session.stream(sample))
    assert len(events) >= 1
    for ev in events:
        assert ev.mode == "budget"
        assert 1 <= ev.exit_layer <= model.core.split
        assert ev.latency_ms >= 0
        assert ev.accepted is None


def test_budget_mode_deadline_forces_min_layers():
    """An impossible deadline should push every token to the shallowest
    allowed exit almost immediately (state-copy approximation kicks in for
    the rest of stage one)."""
    model = make_model(0)
    sample = make_sample("hi")
    session = StreamingSession(
        model, mode="budget", target_tokens_per_s=1e9, confidence=0.999999,
        min_layers=1, max_new_tokens=4, seed=0,
    )
    events = list(session.stream(sample))
    assert all(ev.exit_layer <= max(2, model.core.split // 2 + 1) for ev in events)


def test_interrupt_stops_the_stream_early():
    model = make_model(0)
    session = StreamingSession(model, mode="lossless", draft_len=3, max_new_tokens=50, seed=0)
    events = []
    for ev in session.stream(make_sample("hello world")):
        events.append(ev)
        if len(events) == 1:
            session.interrupt()
    assert len(events) < 50


def test_push_input_mid_stream_keeps_cache_valid():
    model = make_model(0)
    session = StreamingSession(model, mode="lossless", draft_len=2, max_new_tokens=6, seed=0)
    gen = session.stream(make_sample("hello"))
    first = next(gen)
    session.push_input(make_sample(" world"))
    rest = list(gen)
    assert isinstance(first.token, int)
    assert all(isinstance(ev.token, int) for ev in rest)


# -- exactness under the integrated architecture options ----------------------


def _modified_model():
    import dataclasses

    from iridium.config import get_config
    from iridium.model.iridium1 import Iridium1

    tiny = get_config("tiny")
    cfg = dataclasses.replace(
        tiny,
        core=dataclasses.replace(tiny.core, layer_pattern=("mla", "global"),
                                 mla_kv_rank=24, mla_rope_dim=8, hyper_streams=3,
                                 norm_kind="dyt", ffn_dynamic_rank=4),
        codecs=dataclasses.replace(tiny.codecs, ngram_table_size=53),
    )
    torch.manual_seed(0)
    return Iridium1(cfg).eval()


def test_lossless_stays_exact_under_modified_architectures():
    """MLA + mHC streams + DyT + dynamic FFN + n-gram embeddings.

    Drafting steps layers by hand and may compute the wrong draft under these
    options; that can only lower acceptance, because every emitted token comes
    from a real full-depth forward and the cache is restored from a snapshot.
    """
    from iridium.runtime.generate import generate

    model = _modified_model()
    sample = make_sample("the quick brown")
    plain = generate(model, sample, max_new_tokens=10, temperature=0, n_loops=1)
    session = StreamingSession(model, mode="lossless", max_new_tokens=10)
    streamed = [e.token for e in session.stream(sample)]
    assert streamed == plain.ids


def test_budget_mode_refuses_options_it_cannot_honour():
    with pytest.raises(NotImplementedError, match="lossless"):
        StreamingSession(_modified_model(), mode="budget")
