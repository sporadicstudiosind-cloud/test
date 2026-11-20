"""Training-memory fixes must not change what the model computes."""

import numpy as np
import torch

from iridium.codecs import bank
from iridium.codecs.bank import TensorBatch, continuous_dims
from iridium.codecs.spans import Sample, Span, collate
from iridium.config import TEXT_ID_OFFSET, get_config
from iridium.model import layers
from iridium.model.iridium1 import Iridium1


def _run(chunk, monkeypatch):
    monkeypatch.setattr(bank, "TEXT_LOSS_CHUNK", chunk)
    cfg = get_config("tiny")
    rng = np.random.default_rng(0)
    vocab = cfg.codecs.vocab_size
    batch = TensorBatch(collate([Sample([Span("text", rng.integers(TEXT_ID_OFFSET, vocab, 96))])
                                 for _ in range(3)], continuous_dims(cfg.codecs)))
    torch.manual_seed(0)
    model = Iridium1(cfg).train()
    losses, _ = model.losses(batch, n_loops=2, generator=torch.Generator().manual_seed(1))
    sum(losses.values()).backward()
    grads = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
    return float(losses["text"].detach()), float(losses["text_z"].detach()), grads


def test_chunked_text_loss_matches_the_unchunked_loss(monkeypatch):
    whole = _run(10**9, monkeypatch)
    chunked = _run(32, monkeypatch)
    assert abs(whole[0] - chunked[0]) < 1e-5 and abs(whole[1] - chunked[1]) < 1e-4
    assert torch.allclose(whole[2], chunked[2], atol=1e-6, rtol=1e-5)


def test_cpu_attention_still_uses_the_gqa_flag_and_matches_manual():
    q = torch.randn(2, 6, 16, 8)
    k = torch.randn(2, 2, 16, 8)
    v = torch.randn(2, 2, 16, 8)
    keep = torch.ones(16, 16, dtype=torch.bool).tril().view(1, 1, 16, 16)
    fused = layers._attend(q, k, v, keep, 8, n_rep=3)
    previous = layers.set_attention_backend("manual")
    try:
        manual = layers._attend(q, k, v, keep, 8, n_rep=3)
    finally:
        layers.set_attention_backend(previous)
    assert torch.allclose(fused, manual, atol=1e-5)


def test_lean_rmsnorm_and_swiglu_match_plain_autograd():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 16, dtype=torch.float64, requires_grad=True)
    w = (torch.randn(16, dtype=torch.float64) + 1).requires_grad_()
    g = torch.randn(3, 5, 16, dtype=torch.float64)
    (layers._RMSNormFn.apply(x, w, 1e-5) * g).sum().backward()
    lean = (x.grad.clone(), w.grad.clone())
    x.grad = w.grad = None
    (layers._rms_norm_reference(x, w, 1e-5) * g).sum().backward()
    assert torch.allclose(lean[0], x.grad, atol=1e-12) and torch.allclose(lean[1], w.grad, atol=1e-12)

    a = torch.randn(4, 9, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, 9, dtype=torch.float64, requires_grad=True)
    (layers.silu_mul(a, b) * g[0, :4, :9]).sum().backward()
    lean = (a.grad.clone(), b.grad.clone())
    a.grad = b.grad = None
    (torch.nn.functional.silu(a) * b * g[0, :4, :9]).sum().backward()
    assert torch.allclose(lean[0], a.grad, atol=1e-12) and torch.allclose(lean[1], b.grad, atol=1e-12)
