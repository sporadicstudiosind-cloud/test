"""Muon: orthogonalization is approximate by design, the split is by role not rank.

The claims worth testing are the ones that are wrong in the "obvious" version
of this optimizer and invisible in a loss curve: that Newton-Schulz lands in
its known non-convergent band rather than either doing nothing or blowing up,
that the update RMS is actually shape-independent after Moonlight's scaling
(the whole reason that scaling exists), that an embedding table never
silently ends up on the Muon path, and that state_dict round-trips. The toy
training comparison reports real numbers, not a claim that Muon "wins" --
at this toy scale, matched lr, few hundred steps, there is no reason to
expect it to.
"""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from iridium.training.eager_adamw import EagerAdamW
from iridium.training.muon import Muon, muon_param_groups, zeropower_via_newtonschulz5


# --------------------------------------------------------------------------
# Newton-Schulz
# --------------------------------------------------------------------------

def _singular_values(mat: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(mat)


@pytest.mark.parametrize("shape", [(64, 64), (32, 128), (128, 32), (17, 5), (5, 17)])
def test_newton_schulz_lands_in_known_band(shape):
    """Tall, wide, and square: most singular values land in Jordan's ~[0.7, 1.2] band.

    Not [0.99, 1.01]: these coefficients are deliberately tuned to *not*
    converge all the way to an exact orthogonal matrix (see module
    docstring). Not "every single one", either: a random Gaussian matrix's
    smallest singular values, after the initial Frobenius-norm normalization,
    start close enough to the map's other fixed point at 0 that 5 steps is
    not always enough to pull them out (measured: for a 64x64 standard
    normal matrix the smallest of 64 singular values can still be ~0.04
    after 5 steps, while the top value and the bulk sit in-band). This is
    the same effective-rank sensitivity arXiv:2606.00371 ("How Much
    Orthogonalization Does Muon Need?") studies, not a bug in this
    implementation, so the test checks the bulk and the top value, which is
    the part that matters for a real momentum-gradient's update direction.
    """
    torch.manual_seed(0)
    G = torch.randn(*shape)
    O = zeropower_via_newtonschulz5(G, steps=5)
    assert O.shape == G.shape
    assert torch.isfinite(O).all()
    svals = _singular_values(O)
    assert svals.max() < 1.3
    in_band = (svals > 0.7) & (svals < 1.2)
    assert in_band.float().mean() >= 0.5, f"only {in_band.float().mean():.2f} of singular values in band"


def test_newton_schulz_ill_conditioned_input():
    """A matrix with a huge condition number orthogonalizes without blowing up.

    The initial spectral normalization step (dividing by the Frobenius norm,
    an upper bound on the spectral norm) exists exactly for this case:
    without it the iteration's polynomial overshoots its convergent basin
    for the largest singular direction and diverges. What it does not
    promise is pulling every near-zero singular value up to 1 in 5 steps --
    a direction contributing 1e-8 of the original matrix's energy is exactly
    the kind of noise direction a real (i.e. non-adversarial) gradient
    matrix would not have much of, so the test checks the well-conditioned
    top of the spectrum, and that nothing overflows or NaNs on the badly
    conditioned tail.
    """
    torch.manual_seed(1)
    m, n = 40, 20
    U, _ = torch.linalg.qr(torch.randn(m, m))
    V, _ = torch.linalg.qr(torch.randn(n, n))
    s = torch.logspace(0, -8, n)  # condition number 1e8
    G = U[:, :n] @ torch.diag(s) @ V.T
    O = zeropower_via_newtonschulz5(G, steps=5)
    assert torch.isfinite(O).all()
    svals = _singular_values(O)
    assert svals.max() < 1.3
    # the well-conditioned quarter of the spectrum (largest singular values)
    # should land in-band; the tail near the 1e-8 end is not expected to.
    top_quarter = svals[: max(1, len(svals) // 4)]
    assert torch.all((top_quarter > 0.7) & (top_quarter < 1.3))


def test_newton_schulz_finite_under_bf16_autocast_forward():
    """A bf16 autocast forward pass still produces a finite fp32 Muon update.

    Mirrors AMP practice: the forward runs in bf16, the backward and the
    master weights stay fp32, and it is the fp32 gradient that reaches the
    optimizer -- Muon never sees a bf16 parameter.
    """
    torch.manual_seed(2)
    model = nn.Linear(16, 16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(torch.randn(8, 16))
        loss = out.pow(2).mean()
    loss.backward()
    assert model.weight.grad.dtype == torch.float32
    O = zeropower_via_newtonschulz5(model.weight.grad, steps=5)
    assert torch.isfinite(O).all()


# --------------------------------------------------------------------------
# update RMS matches the Moonlight target
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(32, 512), (512, 32), (128, 128)])
def test_update_rms_matches_moonlight_scale_regardless_of_shape(shape):
    """0.2 * sqrt(max(m, n)) cancels Lemma 1's 1/sqrt(max(m, n)) update RMS.

    This is the entire point of the scale factor: without it, a fat matrix
    (large max(m, n)) gets a systematically smaller per-element update than a
    square one. We check the *applied* update (post weight-decay-free step),
    not just the orthogonalized matrix, since that's what a shared lr with
    AdamW actually has to match.
    """
    torch.manual_seed(3)
    m, n = shape
    p = nn.Parameter(torch.randn(m, n) * 0.02)
    p.grad = torch.randn(m, n)
    opt = Muon([{"params": [p], "use_muon": True}], lr=1.0, weight_decay=0.0, momentum=0.0)
    before = p.detach().clone()
    opt.step()
    update = before - p.detach()
    rms = update.pow(2).mean().sqrt().item()
    # lr=1, weight_decay=0, momentum=0 isolates the scale: update == lr * scale * O,
    # and O's per-element RMS is ~1/sqrt(m*n) for an orthogonalized matrix, so the
    # applied update's RMS should sit near 0.2 (Moonlight's target band, 0.2-0.4)
    # times a roughly-shape-independent factor rather than swinging with max(m, n).
    assert 0.1 < rms < 0.5, f"shape {shape}: update RMS {rms} outside AdamW-matched band"


# --------------------------------------------------------------------------
# muon_param_groups: role, not rank
# --------------------------------------------------------------------------

class _ToyTransformerish(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(50, 16)
        self.proj_in = nn.Linear(16, 32)
        self.norm = nn.LayerNorm(32)
        self.proj_out = nn.Linear(32, 16)
        self.text_head = nn.Linear(16, 50)

    def forward(self, idx):
        x = self.embedding(idx)
        x = self.proj_in(x)
        x = self.norm(x)
        x = torch.relu(x)
        x = self.proj_out(x)
        return self.text_head(x)


def test_muon_param_groups_splits_by_module_role():
    model = _ToyTransformerish()
    groups = muon_param_groups(model, lr=1e-3, weight_decay=0.05)
    muon_ids = {id(p) for g in groups if g["use_muon"] for p in g["params"]}
    adamw_ids = {id(p) for g in groups if not g["use_muon"] for p in g["params"]}

    assert id(model.embedding.weight) in adamw_ids
    assert id(model.embedding.weight) not in muon_ids
    assert id(model.text_head.weight) in adamw_ids, "name contains text_head -> adamw"
    assert id(model.norm.weight) in adamw_ids and id(model.norm.bias) in adamw_ids
    assert id(model.proj_in.bias) in adamw_ids and id(model.proj_out.bias) in adamw_ids

    assert id(model.proj_in.weight) in muon_ids
    assert id(model.proj_out.weight) in muon_ids

    # every parameter is accounted for exactly once
    all_ids = muon_ids | adamw_ids
    assert all_ids == {id(p) for p in model.parameters()}
    assert len(muon_ids) + len(adamw_ids) == sum(1 for _ in model.parameters())


def test_muon_rejects_non_2d_in_muon_group():
    bias = nn.Parameter(torch.randn(8))
    with pytest.raises(ValueError):
        Muon([{"params": [bias], "use_muon": True}])


def test_muon_requires_use_muon_key():
    w = nn.Parameter(torch.randn(8, 8))
    with pytest.raises(ValueError):
        Muon([{"params": [w]}])


def test_muon_param_groups_handles_flat_named_parameters_without_module():
    """No module structure available: falls back to rank + name based split."""
    named = [("layer.weight", nn.Parameter(torch.randn(8, 8))),
             ("layer.bias", nn.Parameter(torch.randn(8))),
             ("embedding.weight", nn.Parameter(torch.randn(50, 8)))]
    groups = muon_param_groups(named, lr=1e-3, weight_decay=0.01)
    muon_ids = {id(p) for g in groups if g["use_muon"] for p in g["params"]}
    adamw_ids = {id(p) for g in groups if not g["use_muon"] for p in g["params"]}
    assert id(named[0][1]) in muon_ids
    assert id(named[1][1]) in adamw_ids
    assert id(named[2][1]) in adamw_ids, "name contains 'embedding' -> adamw even without a module"


# --------------------------------------------------------------------------
# state_dict round trip
# --------------------------------------------------------------------------

def test_state_dict_round_trip():
    model = _ToyTransformerish()
    groups = muon_param_groups(model, lr=1e-2, weight_decay=0.01)
    opt = Muon(groups, lr=1e-2)

    generator = torch.Generator().manual_seed(0)
    idx = torch.randint(0, 50, (4, 5), generator=generator)
    for _ in range(3):
        loss = model(idx).float().pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    dumped = copy.deepcopy(opt.state_dict())

    model2 = _ToyTransformerish()
    model2.load_state_dict(model.state_dict())
    groups2 = muon_param_groups(model2, lr=1e-2, weight_decay=0.01)
    opt2 = Muon(groups2, lr=1e-2)
    opt2.load_state_dict(dumped)

    for p, p2 in zip(model.parameters(), model2.parameters()):
        s, s2 = opt.state.get(p), opt2.state.get(p2)
        assert (s is None) == (s2 is None)
        if s is None:
            continue
        for key in s:
            if torch.is_tensor(s[key]):
                torch.testing.assert_close(s[key], s2[key])
            else:
                assert s[key] == s2[key]

    # and the resumed optimizer can keep stepping
    loss = model2(idx).float().pow(2).mean()
    opt2.zero_grad()
    loss.backward()
    opt2.step()


def test_load_state_dict_rejects_layout_mismatch():
    model = _ToyTransformerish()
    groups = muon_param_groups(model, lr=1e-2)
    opt = Muon(groups, lr=1e-2)
    bad = opt.state_dict()
    bad["param_groups"] = bad["param_groups"][:-1]
    other = Muon(muon_param_groups(_ToyTransformerish(), lr=1e-2), lr=1e-2)
    with pytest.raises(ValueError):
        other.load_state_dict(bad)


# --------------------------------------------------------------------------
# toy training comparison
# --------------------------------------------------------------------------

class _ToyRegressionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def _toy_regression_data(n=512, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 10, generator=generator)
    true_w = torch.randn(10, 1, generator=generator)
    y = x @ true_w + 0.1 * torch.randn(n, 1, generator=generator)
    return x, y


def _train(model, optimizer, x, y, steps=300, batch=64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), generator=generator)
        pred = model(x[idx])
        loss = (pred - y[idx]).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return model(x).sub(y).pow(2).mean().item()


def test_toy_regression_adamw_vs_muon_matched_lr():
    """Report both final losses honestly; this is not a claim that Muon wins.

    Moonlight's ~2x compute-efficiency result is measured on multi-billion-
    parameter LLM pretraining with a MoE and trillions of tokens (arXiv
    2502.16982, Fig. 1a) -- a three-layer MLP on synthetic linear-regression
    data at a few hundred steps is not that regime, and this test does not
    pretend otherwise. It only checks that both optimizers make real progress
    from the same random init at the same shared lr, and prints what each one
    actually reached.
    """
    torch.manual_seed(42)
    x, y = _toy_regression_data()
    init_state = _ToyRegressionMLP().state_dict()

    lr = 2e-2

    model_adamw = _ToyRegressionMLP()
    model_adamw.load_state_dict(init_state)
    with torch.no_grad():
        init_loss = model_adamw(x).sub(y).pow(2).mean().item()
    opt_adamw = EagerAdamW(model_adamw.parameters(), lr=lr, weight_decay=0.0)
    loss_adamw = _train(model_adamw, opt_adamw, x, y)

    model_muon = _ToyRegressionMLP()
    model_muon.load_state_dict(init_state)
    groups = muon_param_groups(model_muon, lr=lr, weight_decay=0.0)
    opt_muon = Muon(groups, lr=lr)
    loss_muon = _train(model_muon, opt_muon, x, y)

    print(f"\ntoy regression MLP, matched lr={lr}, 300 steps, batch 64:")
    print(f"  init loss:  {init_loss:.6f}")
    print(f"  AdamW loss: {loss_adamw:.6f}")
    print(f"  Muon loss:  {loss_muon:.6f}")

    assert loss_adamw < init_loss * 0.5, "AdamW made no real progress"
    assert loss_muon < init_loss * 0.5, "Muon made no real progress"
    assert torch.isfinite(torch.tensor(loss_adamw))
    assert torch.isfinite(torch.tensor(loss_muon))
