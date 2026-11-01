"""ParallelBlock (GPT-J/PaLM parallel attn+FFN) and function-preserving
depth growth (grow_depth).

No quality claim is made or tested here: PaLM's own ablation found a small
degradation at 8B and none at 62B, and nothing about a forward pass on
random weights can measure "quality" at all. What is tested is shape
compatibility with TransformerBlock, exact parameter accounting, and that
depth growth is bit-identical (to float tolerance) at the moment of growth.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from iridium.model.blocks import ParallelBlock, grow_depth
from iridium.model.layers import TransformerBlock, causal_keep
from iridium.model.norms import norm_param_count
from iridium.model.rope import RotaryEmbedding

torch.manual_seed(0)

D_MODEL, N_Q, N_KV, D_HEAD, D_FF = 32, 4, 2, 8, 64


def _rope():
    return RotaryEmbedding(D_HEAD)


def _inputs(batch=2, seq=5):
    x = torch.randn(batch, seq, D_MODEL)
    positions = torch.arange(seq).unsqueeze(0).expand(batch, seq)
    keep = causal_keep(seq, seq)
    return x, positions, keep


# --------------------------------------------------------------------------
# ParallelBlock: drop-in shape/signature compatibility
# --------------------------------------------------------------------------


def test_parallel_block_matches_transformer_block_signature_and_shape():
    block = ParallelBlock(D_MODEL, N_Q, N_KV, D_HEAD, D_FF, _rope())
    x, positions, keep = _inputs()
    out = block(x, positions, keep, None, None)
    assert out.shape == x.shape


def test_parallel_block_formula():
    d, n = 6, 7
    x, positions, keep = _inputs()
    n_q, n_kv, d_head, d_ff = 4, 2, 8, 32
    d_model = n_q * d_head
    x = torch.randn(2, 5, d_model)
    block = ParallelBlock(d_model, n_q, n_kv, d_head, d_ff, _rope())
    a = block.attn(block.norm(x), positions, keep, None, None)
    f = block.ffn(block.norm(x))
    expected = x + a + f
    assert torch.equal(block(x, positions, keep, None, None), expected)


def test_parallel_block_uses_one_norm_not_two():
    """The parameter saving PaLM describes: one shared norm, not two."""
    block = ParallelBlock(D_MODEL, N_Q, N_KV, D_HEAD, D_FF, _rope())
    reference = TransformerBlock(D_MODEL, N_Q, N_KV, D_HEAD, D_FF, _rope())
    n_norms_parallel = sum(1 for _ in [block.norm])
    n_norms_reference = sum(1 for _ in [reference.norm_attn, reference.norm_ffn])
    assert n_norms_parallel == 1
    assert n_norms_reference == 2

    parallel_total = sum(p.numel() for p in block.parameters())
    reference_total = sum(p.numel() for p in reference.parameters())
    # Same attn/ffn shapes, one fewer RMSNorm's worth of parameters (d_model).
    assert reference_total - parallel_total == D_MODEL


@pytest.mark.parametrize("norm_kind", ["rms", "dyt", "derf"])
def test_parallel_block_param_count_matches_module(norm_kind):
    block = ParallelBlock(D_MODEL, N_Q, N_KV, D_HEAD, D_FF, _rope(), norm_kind=norm_kind)
    actual = sum(p.numel() for p in block.parameters())
    expected = ParallelBlock.param_count(D_MODEL, N_Q, N_KV, D_HEAD, D_FF, norm_kind)
    assert actual == expected


def test_parallel_block_rejects_adaptive_norm():
    with pytest.raises(ValueError):
        ParallelBlock(D_MODEL, N_Q, N_KV, D_HEAD, D_FF, _rope(), norm_kind="adaptive")


def test_parallel_block_gradients_finite():
    block = ParallelBlock(D_MODEL, N_Q, N_KV, D_HEAD, D_FF, _rope())
    x, positions, keep = _inputs()
    x.requires_grad_(True)
    out = block(x, positions, keep, None, None)
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    for p in block.parameters():
        assert torch.isfinite(p.grad).all()


def test_parallel_block_cache_roundtrip_matches_full_forward():
    """Same KV-cache contract as TransformerBlock: incremental decode agrees
    with a single full forward pass (manual attention backend for exactness,
    matching how TransformerBlock's own parity gate is run)."""
    from iridium.model.layers import use_attention_backend

    with use_attention_backend("manual"):
        block = ParallelBlock(D_MODEL, N_Q, N_KV, D_HEAD, D_FF, _rope())
        block.eval()
        x = torch.randn(1, 4, D_MODEL)
        positions = torch.arange(4).unsqueeze(0)
        keep = causal_keep(4, 4)
        full = block(x, positions, keep, None, None)

        cache = {}
        outs = []
        for t in range(4):
            xt = x[:, t:t + 1, :]
            pt = positions[:, t:t + 1]
            kt = causal_keep(1, t + 1)
            outs.append(block(xt, pt, kt, cache, ("k",)))
        incremental = torch.cat(outs, dim=1)
        assert torch.allclose(full, incremental, atol=1e-5)


# --------------------------------------------------------------------------
# grow_depth: function-preserving depth growth
# --------------------------------------------------------------------------


def _stack(depth, cls=TransformerBlock, **kw):
    rope = _rope()
    return nn.ModuleList(cls(D_MODEL, N_Q, N_KV, D_HEAD, D_FF, rope, **kw) for _ in range(depth))


def _run(stack, x, positions, keep):
    for block in stack:
        x = block(x, positions, keep, None, None)
    return x


@pytest.mark.parametrize("method", ["append", "interleave", "duplicate"])
@pytest.mark.parametrize("block_cls", [TransformerBlock, ParallelBlock])
def test_grow_depth_is_function_preserving(method, block_cls):
    torch.manual_seed(11)
    stack = _stack(4, cls=block_cls)
    x, positions, keep = _inputs()

    before = _run(stack, x, positions, keep)
    grown = grow_depth(stack, new_depth=8, method=method)
    after = _run(grown, x, positions, keep)

    assert len(grown) == 8
    assert torch.allclose(before, after, atol=1e-6), (
        f"{method}: max abs diff {(before - after).abs().max().item()}"
    )


def test_grow_depth_by_factor():
    stack = _stack(3)
    grown = grow_depth(stack, factor=2.0, method="append")
    assert len(grown) == 6


def test_grow_depth_original_modulelist_is_untouched():
    stack = _stack(3)
    ids_before = [id(b) for b in stack]
    grow_depth(stack, new_depth=6, method="append")
    assert [id(b) for b in stack] == ids_before
    assert len(stack) == 3


def test_grow_depth_new_blocks_receive_gradients():
    torch.manual_seed(12)
    stack = _stack(2)
    grown = grow_depth(stack, new_depth=4, method="append")
    x, positions, keep = _inputs()
    out = _run(grown, x, positions, keep)
    out.sum().backward()
    new_block = grown[2]
    # attn.wo and ffn.down start at exactly zero grad *value* is not the
    # claim; the claim is that a gradient reaches them at all (nonzero grad
    # tensor allocated, finite), which is what lets training move them away
    # from the identity they were inserted at.
    assert new_block.attn.wo.weight.grad is not None
    assert torch.isfinite(new_block.attn.wo.weight.grad).all()
    assert new_block.ffn.down.weight.grad is not None
    assert torch.isfinite(new_block.ffn.down.weight.grad).all()


def test_grow_depth_rejects_shrinking_and_bad_args():
    stack = _stack(4)
    with pytest.raises(ValueError):
        grow_depth(stack, new_depth=2)
    with pytest.raises(ValueError):
        grow_depth(stack)   # neither new_depth nor factor
    with pytest.raises(ValueError):
        grow_depth(stack, new_depth=6, factor=2.0)   # both
    with pytest.raises(ValueError):
        grow_depth(nn.ModuleList(), new_depth=3)   # empty
    with pytest.raises(ValueError):
        grow_depth(stack, new_depth=6, method="bogus")


def test_grow_depth_inserted_blocks_have_zeroed_output_projections():
    stack = _stack(2)
    grown = grow_depth(stack, new_depth=4, method="append")
    for new_block in list(grown)[2:]:
        assert torch.equal(new_block.attn.wo.weight, torch.zeros_like(new_block.attn.wo.weight))
        assert torch.equal(new_block.ffn.down.weight, torch.zeros_like(new_block.ffn.down.weight))


def test_grow_depth_duplicate_copies_neighbor_weights_before_zeroing():
    torch.manual_seed(13)
    stack = _stack(2)
    grown = grow_depth(stack, new_depth=3, method="duplicate")
    new_block = grown[2]
    src = stack[1]
    # Everything upstream of the zeroed output projections is copied, not
    # freshly initialised.
    assert torch.equal(new_block.attn.wq.weight, src.attn.wq.weight)
    assert torch.equal(new_block.norm_attn.weight, src.norm_attn.weight)
