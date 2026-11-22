"""Dropping empty rows before a superstack runs changes nothing but the cost."""
import dataclasses

import numpy as np
import torch

from iridium.codecs.bank import TensorBatch, continuous_dims
from iridium.codecs.spans import Sample, Span, collate
from iridium.config import TEXT_ID_OFFSET, get_config
from iridium.model.iridium1 import Iridium1


def _setup():
    cfg = get_config("tiny")
    names = ("math", "chat") + tuple(f"s{i}" for i in range(cfg.stacks.n_stacks - 2))
    cfg = dataclasses.replace(
        cfg, stacks=dataclasses.replace(cfg.stacks, specializations=names),
        router=dataclasses.replace(cfg.router, top_k=min(2, cfg.stacks.n_stacks),
                                   subject_force=True, solo_p=1.0))
    rng = np.random.default_rng(0)
    samples = [Sample([Span("control", np.array([1]), supervised=False),
                       Span("text", rng.integers(TEXT_ID_OFFSET, TEXT_ID_OFFSET + 200, 30), supervised=True)],
                      meta={"subject": s}) for s in ("math", "chat", "math", "chat")]
    return cfg, TensorBatch(collate(samples, continuous_dims(cfg.codecs)))


def _run(compact, n_loops):
    cfg, batch = _setup()
    torch.manual_seed(0)
    model = Iridium1(cfg).double().train()
    model.bank.compact_rows = compact
    torch.manual_seed(1)
    losses, out = model.losses(batch, n_loops=n_loops)
    total = sum(v for v in losses.values() if torch.is_tensor(v) and v.requires_grad)
    total.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return out.hidden.detach(), float(total.detach()), grads


def test_compaction_is_exact_in_forward_and_backward():
    for loops in (1, 2):
        h0, l0, g0 = _run(False, loops)
        h1, l1, g1 = _run(True, loops)
        torch.testing.assert_close(h1, h0, rtol=1e-10, atol=1e-10)
        assert abs(l1 - l0) < 1e-9
        assert g0.keys() == g1.keys()
        for name in g0:
            torch.testing.assert_close(g1[name], g0[name], rtol=1e-9, atol=1e-10, msg=name)
