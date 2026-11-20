"""Sub-1-bit LittleBit layers (iridium.quant.littlebit)."""
import torch
import torch.nn as nn

from iridium.quant.littlebit import (LittleBitLinear, compress, dual_svid, joint_itq,
                                     layer_bits, rank_for_bpw, smooth_sign)


def _weight(out=48, inp=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    low = torch.randn(out, 8, generator=g) @ torch.randn(8, inp, generator=g) / 3
    return low + 0.2 * torch.randn(out, inp, generator=g)


def test_bit_budget_formula_and_inverse():
    for out, inp, bpw, res in [(384, 1024, 1.0, True), (384, 384, 0.55, False), (1024, 384, 0.3, True)]:
        r = rank_for_bpw(out, inp, bpw, res)
        assert layer_bits(out, inp, r, res) <= bpw * out * inp
        assert layer_bits(out, inp, r + 1, res) > bpw * out * inp


def test_small_matrices_have_a_floor_set_by_their_scales():
    # iridium-50m's widest matrices are 1024 x 384. Two paths of fp16 scales
    # alone cost 32 * (1024 + 384) bits, about 0.115 bits per weight, so the
    # papers' 0.1 bpw setting cannot exist at this size: rank bottoms out at 1.
    assert rank_for_bpw(1024, 384, 0.1, True) == 1
    assert layer_bits(1024, 384, 1, True) > 0.1 * 1024 * 384


def test_smooth_sign_forward_is_sign_and_gradient_is_local():
    x = torch.tensor([-2.0, -0.001, 0.0, 0.001, 2.0], requires_grad=True)
    y = smooth_sign(x)
    assert y.tolist() == [-1.0, -1.0, 1.0, 1.0, 1.0]
    y.sum().backward()
    assert x.grad[0] < 1e-6 and x.grad[-1] < 1e-6 and x.grad[2] > 50


def test_joint_itq_is_orthogonal_and_keeps_the_product():
    u, v = torch.randn(20, 6), torch.randn(30, 6)
    r = joint_itq(u, v, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(r @ r.T, torch.eye(6), atol=1e-5, rtol=0)
    torch.testing.assert_close((u @ r) @ (v @ r).T, u @ v.T, atol=1e-4, rtol=1e-4)


def test_joint_itq_start_reconstructs_better_than_plain_dual_svid():
    w = _weight()
    errs = {}
    for itq in (False, True):
        u, v, h, g, l = dual_svid(w, 12, itq=itq, generator=torch.Generator().manual_seed(0))
        approx = (h[:, None] * torch.sign(u) * l) @ (torch.sign(v) * g[:, None]).T
        errs[itq] = float((approx - w).norm() / w.norm())
    assert errs[True] < errs[False]


def test_layer_matches_its_dense_weight_and_residual_helps():
    w = _weight()
    lin = nn.Linear(96, 48, bias=True)
    with torch.no_grad():
        lin.weight.copy_(w)
    errs = {}
    for residual in (False, True):
        layer = LittleBitLinear.from_linear(lin, 12, residual=residual).eval()
        x = torch.randn(4, 96)
        with torch.no_grad():
            torch.testing.assert_close(layer(x), x @ layer.weight().T + lin.bias, atol=1e-4, rtol=1e-4)
            errs[residual] = float((layer.weight() - w).norm() / w.norm())
    assert errs[True] < errs[False]


def test_training_moves_the_latents_and_packing_is_compact():
    lin = nn.Linear(64, 32, bias=False)
    layer = LittleBitLinear.from_linear(lin, 8).train()
    x = torch.randn(16, 64)
    loss = (layer(x) - lin(x).detach()).pow(2).mean()
    loss.backward()
    assert layer.u_latent.grad is not None and layer.h.grad.abs().sum() > 0
    packed = layer.packed()
    sign_bytes = sum(packed[k].nbytes for k in ("u0", "v0", "u1", "v1"))
    assert sign_bytes == 2 * ((32 * 8 + 7) // 8 + (64 * 8 + 7) // 8)


def test_compress_replaces_only_hidden_matrices():
    model = nn.Sequential()
    model.add_module("embedding", nn.Embedding(50, 64))
    model.add_module("mid", nn.Linear(64, 64))
    model.add_module("text_head", nn.Linear(64, 50))
    report = compress(model, bpw=2.0)
    assert isinstance(model.mid, LittleBitLinear)
    assert isinstance(model.text_head, nn.Linear) and isinstance(model.embedding, nn.Embedding)
    assert report["layers"] == 1 and report["bits_per_weight"] <= 2.0
