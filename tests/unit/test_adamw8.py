"""8-bit AdamW tracks fp32 AdamW closely and trains a model."""

import torch

from iridium.training.adamw8 import AdamW8bit, _dequant_signed, _dequant_sqrt, _quant_signed, _quant_sqrt


def test_quantization_keeps_small_values_nonzero_and_close():
    x = torch.randn(1000) * torch.logspace(-3, 1, 1000)
    q, s = _quant_signed(x, 256)
    back = _dequant_signed(q, s, x.shape, x.numel())
    assert ((back - x).abs() <= 0.02 * s.repeat_interleave(256)[:1000] + 1e-12).all()
    v = x.square()
    q, s = _quant_sqrt(v, 256)
    vb = _dequant_sqrt(q, s, v.shape, v.numel())
    big = v > 1e-4 * s.repeat_interleave(256)[:1000] ** 4
    assert (vb[big] > 0).all()


def test_matches_fp32_adamw_on_a_regression():
    torch.manual_seed(0)
    X, w_true = torch.randn(256, 64), torch.randn(64, 1)
    y = X @ w_true
    losses = {}
    for name, cls in (("fp32", torch.optim.AdamW), ("int8", AdamW8bit)):
        torch.manual_seed(1)
        lin = torch.nn.Linear(64, 1, bias=False)
        kw = dict(min_8bit_size=1) if cls is AdamW8bit else {}
        opt = cls(lin.parameters(), lr=1e-2, weight_decay=0.0, betas=(0.9, 0.95), **kw)
        for _ in range(600):
            opt.zero_grad()
            loss = ((lin(X) - y) ** 2).mean()
            loss.backward()
            opt.step()
        losses[name] = float(loss)
    assert losses["int8"] < 1e-3 and losses["int8"] < 5 * losses["fp32"] + 1e-4


def test_state_is_about_two_bytes_per_parameter():
    lin = torch.nn.Linear(1024, 1024, bias=False)
    opt = AdamW8bit(lin.parameters())
    lin(torch.randn(2, 1024)).sum().backward()
    opt.step()
    st = opt.state[lin.weight]
    nbytes = sum(t.numel() * t.element_size() for t in st.values() if torch.is_tensor(t))
    assert nbytes / lin.weight.numel() < 2.1
