"""ExitHead parameter accounting, forward shape, and early-exit-loss sanity.

Mirrors the config ladder's own convention (see ``iridium/config.py``): the
exact parameter count is re-derived from real tensor shapes, not transcribed,
so it cannot silently drift from :class:`~iridium.model.exits.ExitHead`.
"""

import numpy as np
import torch

from iridium.codecs.bank import TensorBatch, continuous_dims
from iridium.codecs.spans import Sample, text_span, collate
from iridium.config import get_config
from iridium.model.exits import (
    ExitHead,
    curriculum_weights,
    early_exit_loss,
    layer_dropout_schedule,
)
from iridium.model.iridium1 import Iridium1


def text_batch(seed=0, n=2):
    dims = continuous_dims(get_config("tiny").codecs)
    rng = np.random.default_rng(seed)
    samples = [
        Sample([text_span("".join(chr(97 + int(c)) for c in rng.integers(0, 26, 11)))])
        for _ in range(n)
    ]
    return TensorBatch(collate(samples, dims))


def test_exit_head_param_count_matches_formula():
    for d_model in (16, 64, 512):
        head = ExitHead(d_model)
        exact = sum(p.numel() for p in head.parameters())
        assert exact == ExitHead.count_params(d_model)


def test_exit_head_forward_shape_and_reuses_text_head():
    torch.manual_seed(0)
    cfg = get_config("tiny")
    model = Iridium1(cfg).eval()
    head = ExitHead(cfg.core.d_model)
    h = torch.randn(2, 5, cfg.core.d_model)
    logits, conf = head(h, model.codecs.text_head)
    assert logits.shape == (2, 5, cfg.codecs.vocab_size)
    assert conf.shape == (2, 5)
    assert bool(((conf >= 0) & (conf <= 1)).all())
    # The big matrix is the reused, tied text head -- ExitHead adds nothing
    # of that size. Its own parameters are exactly norm + confidence.
    assert sum(p.numel() for p in head.parameters()) == cfg.core.d_model + (cfg.core.d_model + 1)


def test_layer_dropout_schedule_monotonic_and_bounded():
    sched = layer_dropout_schedule(6, max_rate=0.3)
    assert len(sched) == 6
    assert sched[0] == 0.0
    assert abs(sched[-1] - 0.3) < 1e-9
    assert all(a <= b + 1e-12 for a, b in zip(sched, sched[1:]))
    assert layer_dropout_schedule(1) == [0.0]


def test_curriculum_weights_sum_to_one_and_shift_with_progress():
    layers = [1, 2, 3]
    early = curriculum_weights(layers, progress=0.0)
    late = curriculum_weights(layers, progress=1.0)
    assert abs(sum(early) - 1.0) < 1e-9
    assert abs(sum(late) - 1.0) < 1e-9
    # progress=0 favors the deepest listed exit (train mostly as normal);
    # progress=1 favors the shallowest (push capability into early layers).
    assert early[-1] > early[0]
    assert late[0] > late[-1]


def test_early_exit_loss_runs_and_is_finite_and_trainable():
    torch.manual_seed(0)
    cfg = get_config("tiny")
    model = Iridium1(cfg)
    model.train()
    batch = text_batch()
    exit_layers = [1, model.core.split]
    heads = [ExitHead(cfg.core.d_model) for _ in exit_layers]
    params = [p for h in heads for p in h.parameters()]
    opt = torch.optim.SGD(params, lr=0.5)

    losses0 = early_exit_loss(model, batch, exit_layers, exit_heads=heads)
    assert torch.isfinite(losses0["early_exit"])
    assert set(losses0.keys()) == {"early_exit", f"early_exit_layer_1", f"early_exit_layer_{model.core.split}"}

    loss0 = float(losses0["early_exit"].detach())
    for _ in range(20):
        opt.zero_grad()
        out = early_exit_loss(model, batch, exit_layers, exit_heads=heads)
        out["early_exit"].backward()
        opt.step()
    out = early_exit_loss(model, batch, exit_layers, exit_heads=heads)
    assert float(out["early_exit"].detach()) < loss0, "exit heads should learn something from repeated fitting"


def test_early_exit_loss_rejects_stage_two_layers():
    cfg = get_config("tiny")
    model = Iridium1(cfg)
    batch = text_batch()
    try:
        early_exit_loss(model, batch, [cfg.core.n_layers])
        assert False, "expected a ValueError for an exit layer beyond stage one"
    except ValueError:
        pass
