"""Adaptive thinking: per-token ponder depth from the halting head, effort and latency."""

from dataclasses import replace

import torch

from iridium.codecs.bank import TensorBatch, continuous_dims
from iridium.codecs.spans import Sample, collate, text_span
from iridium.config import get_config
from iridium.model.iridium1 import Iridium1
from iridium.runtime.generate import generate
from iridium.runtime.thinking import ThinkingBudget


def _model(halt_bias: float):
    cfg = get_config("tiny")
    cfg = replace(cfg, router=replace(cfg.router, max_loops=3))
    torch.manual_seed(0)
    m = Iridium1(cfg).eval()
    with torch.no_grad():
        m.core.loop_halt_head.weight.zero_()
        m.core.loop_halt_head.bias.fill_(halt_bias)
    return m


def _batch(cfg, text="hello there"):
    return TensorBatch(collate([Sample([text_span(text, offset=16)])], continuous_dims(cfg.codecs)))


def test_confident_tokens_stop_after_one_loop_and_caches_stay_aligned():
    m = _model(+20.0)
    cache = {}
    with torch.no_grad():
        out = m(_batch(m.cfg), n_loops=3, cache=cache, halt_threshold=0.9)
    assert out.stats["halted_at"] == 1
    lengths = {cache[("core", loop, 0)][0].shape[2] for loop in range(3)}
    assert lengths == {cache[("stream", "n")]}           # every loop covers every position
    assert torch.equal(cache[("core", 2, 0)][0], cache[("core", 0, 0)][0])


def test_unsure_tokens_use_every_loop_and_no_threshold_changes_nothing():
    m = _model(-20.0)
    with torch.no_grad():
        out = m(_batch(m.cfg), n_loops=3, cache={}, halt_threshold=0.9)
        assert out.stats["halted_at"] == 3
        a = m(_batch(m.cfg), n_loops=3, cache={}).hidden
        b = m(_batch(m.cfg), n_loops=3, cache={}, halt_threshold=None).hidden
    assert torch.equal(a, b)


def test_generation_adapts_depth_per_token():
    easy = _model(+20.0)
    out = generate(easy, Sample([text_span("hi", offset=16)]), max_new_tokens=5,
                   stop_ids=(), thinking=ThinkingBudget("fast"))
    assert out.loops[1:] and all(l == 1.0 for l in out.loops[1:])
    hard = _model(-20.0)
    out = generate(hard, Sample([text_span("hi", offset=16)]), max_new_tokens=5,
                   stop_ids=(), thinking=ThinkingBudget("deep"))
    assert all(l == 3.0 for l in out.loops[1:])
    out = generate(hard, Sample([text_span("hi", offset=16)]), max_new_tokens=3,
                   stop_ids=(), thinking=ThinkingBudget("instant"))
    assert all(l == 1.0 for l in out.loops[1:])


def test_budget_reacts_to_latency_and_difficulty():
    b = ThinkingBudget("balanced", latency_ms=10.0)
    _, base = b.plan(3)
    b.observe(elapsed_ms=50.0)                          # too slow: think less
    _, slower = b.plan(3)
    assert slower < base
    hard = ThinkingBudget("balanced")
    hard.observe(logits=torch.zeros(100))              # uniform: maximally unsure
    easy = ThinkingBudget("balanced")
    easy.observe(logits=torch.tensor([50.0] + [0.0] * 99))
    assert hard.plan(3)[1] > easy.plan(3)[1]
    assert ThinkingBudget("instant").plan(3) == (1, None)
    assert ThinkingBudget("max").plan(3) == (3, None)
