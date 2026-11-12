import torch

from iridium.runtime.thinking import ThinkingBudget


def test_user_base_moves_within_spread():
    b = ThinkingBudget(loops=12, spread=10)
    assert b.plan(3, ceiling=40)[0] == 12             # no signal yet: the base
    b.observe(logits=torch.zeros(256))               # uniform -> hardest
    assert b.plan(3, ceiling=40)[0] == 22
    peaked = torch.full((256,), -30.0); peaked[0] = 30.0
    b.observe(logits=peaked)                         # certain -> easiest
    assert b.plan(3, ceiling=40)[0] == 2
    assert b.cap(3, 40) == 22 and b.cap(3, 15) == 15
    assert ThinkingBudget(loops=3).plan(3, ceiling=13)[0] == 3
    b.observe(logits=peaked)
    assert ThinkingBudget(loops=1, spread=10).plan(3, 13)[0] == 1


def test_model_runs_past_trained_cap_in_eval_only():
    import pytest
    from dataclasses import replace

    from iridium.codecs.bank import TensorBatch, continuous_dims
    from iridium.codecs.spans import Sample, collate, text_span
    from iridium.config import get_config
    from iridium.model.iridium1 import Iridium1
    from iridium.runtime.generate import generate

    cfg = get_config("tiny")
    cfg = replace(cfg, router=replace(cfg.router, max_loops=2))
    assert cfg.router.loop_ceiling == 12
    torch.manual_seed(0)
    model = Iridium1(cfg).eval()
    with torch.no_grad():                            # never halt early
        model.core.loop_halt_head.weight.zero_()
        model.core.loop_halt_head.bias.fill_(-20.0)
    out = generate(model, Sample([text_span("hi", offset=16)]), max_new_tokens=3,
                   stop_ids=(), thinking=ThinkingBudget("deep", loops=8, spread=10))
    assert out.loops[1:] and all(2 < l <= 12 for l in out.loops[1:])
    batch = TensorBatch(collate([Sample([text_span("hi", offset=16)])],
                                continuous_dims(cfg.codecs)))
    with torch.no_grad():
        model(batch, n_loops=12)
    model.train()
    with pytest.raises(ValueError):
        model(batch, n_loops=3)                      # training stays within max_loops
