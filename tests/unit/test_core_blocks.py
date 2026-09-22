"""Every control-core architecture option, checked through the real model.

The modules each have their own tests. What those cannot show is that they
still hold *inside* Iridium-1, which is where three invariants live that any
one of these options could quietly break:

* **Exact parameter accounting.** ``config.py`` restates every module's
  parameter formula as arithmetic, because it may not import torch. The real
  model must match it to the parameter, or the ladder's cost reports are
  fiction.
* **Cached decoding equals teacher forcing.** The repo's parity gate. Sliding
  windows, latent caches, recurrent state, and residual streams each change
  what a cache entry is; any of them can make the serving path compute a
  different function from the one that was trained.
* **Function-preserving growth.** Zeroing a block's output projections must
  make it an exact identity, for every block type the factory can build.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest
import torch

from iridium.codecs.bank import TensorBatch, continuous_dims
from iridium.codecs.spans import Sample, collate, text_span
from iridium.config import ConfigError, IridiumConfig, get_config
from iridium.model.blocks import grow_depth
from iridium.model.core_blocks import output_projections
from iridium.model.iridium1 import Iridium1
from iridium.model.layers import causal_keep
from iridium.runtime.decode import run_chunked

VARIANTS = {
    "default": {},
    "local_global": dict(layer_pattern=("local", "global"), local_window=3),
    "mla": dict(layer_pattern=("mla",), mla_kv_rank=24, mla_rope_dim=8),
    "mla_q_lora": dict(layer_pattern=("mla",), mla_kv_rank=24, mla_q_rank=12,
                       mla_rope_dim=16),
    # Sixteen heads of width 16 on a d_model of 64: only legal because every
    # layer is MLA, whose per-head width is independent of d_model.
    "mla_many_heads": dict(layer_pattern=("mla",), n_query_heads=16, mla_kv_rank=24,
                           mla_rope_dim=8),
    "deltanet_hybrid": dict(layer_pattern=("deltanet", "deltanet", "deltanet", "global")),
    "dyt": dict(norm_kind="dyt"),
    "derf_parallel": dict(norm_kind="derf", block_kind="parallel"),
    "dynamic_ffn": dict(ffn_dynamic_rank=4),
    "mhc": dict(hyper_streams=4),
    "mhc_static": dict(hyper_streams=3, hyper_dynamic=False),
    "everything": dict(layer_pattern=("deltanet", "mla", "local", "global"),
                       local_window=3, mla_kv_rank=24, mla_rope_dim=8,
                       norm_kind="dyt", block_kind="parallel", ffn_dynamic_rank=4,
                       hyper_streams=4),
}


def _config(options: dict) -> IridiumConfig:
    tiny = get_config("tiny")
    return dataclasses.replace(tiny, core=dataclasses.replace(tiny.core, **options))


def _model(options: dict, dtype=torch.float64) -> Iridium1:
    torch.manual_seed(0)
    return Iridium1(_config(options)).to(dtype).eval()


def _batch(cfg, seed=0, n=2, dtype=torch.float64) -> TensorBatch:
    rng = np.random.default_rng(seed)
    samples = [Sample([text_span("".join(chr(97 + int(c)) for c in rng.integers(0, 26, 11)))])
               for _ in range(n)]
    return TensorBatch(collate(samples, continuous_dims(cfg.codecs)), dtype=dtype)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_real_model_matches_the_config_formula_exactly(name):
    model = _model(VARIANTS[name], dtype=torch.float32)
    cfg = model.cfg
    assert sum(p.numel() for p in model.parameters()) == cfg.n_params
    assert model.parameter_inventory()["core"] == cfg.core.params


@pytest.mark.parametrize("name", sorted(VARIANTS))
@pytest.mark.parametrize("chunk", [1, 3])
def test_cached_decoding_equals_the_uncached_forward(name, chunk):
    model = _model(VARIANTS[name])
    batch = _batch(model.cfg)
    with torch.no_grad():
        reference = model(batch, n_loops=1).hidden
        cached = run_chunked(model, batch, chunk=chunk, n_loops=1)
    torch.testing.assert_close(cached, reference, rtol=0, atol=1e-10)


@pytest.mark.parametrize("name", ["local_global", "mla", "deltanet_hybrid", "mhc", "everything"])
def test_cached_decoding_holds_through_the_ponder_loop(name):
    """With two loops the same core layers run twice per token; caches must not alias."""
    model = _model(VARIANTS[name])
    batch = _batch(model.cfg, seed=1)
    with torch.no_grad():
        reference = model(batch, n_loops=2).hidden
        cached = run_chunked(model, batch, chunk=1, n_loops=2)
    torch.testing.assert_close(cached, reference, rtol=0, atol=1e-10)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_zeroed_output_projections_make_every_block_an_identity(name):
    model = _model(VARIANTS[name])
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 5, model.cfg.core.d_model, generator=g, dtype=torch.float64)
    positions = torch.arange(5).expand(2, 5)
    keep = causal_keep(5, 5)
    for block in model.core.layers:
        silent = copy.deepcopy(block)
        projections = output_projections(silent)
        assert len(projections) >= 2
        with torch.no_grad():
            for proj in projections:
                proj.weight.zero_()
        torch.testing.assert_close(silent(x, positions, keep), x, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["dynamic_ffn", "derf_parallel", "mla", "deltanet_hybrid"])
def test_depth_growth_is_function_preserving_for_new_block_types(name):
    model = _model(VARIANTS[name])
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 4, model.cfg.core.d_model, generator=g, dtype=torch.float64)
    positions = torch.arange(4).expand(1, 4)
    keep = causal_keep(4, 4)

    def run(layers):
        h = x
        for layer in layers:
            h = layer(h, positions, keep)
        return h

    before = run(model.core.layers)
    after = run(grow_depth(model.core.layers, factor=2, method="interleave"))
    torch.testing.assert_close(after, before, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_a_training_step_produces_finite_gradients_everywhere(name):
    model = _model(VARIANTS[name], dtype=torch.float32).train()
    batch = _batch(model.cfg, dtype=torch.float32)
    losses, _ = model.losses(batch, n_loops=1)
    total = sum(v for v in losses.values())
    total.backward()
    # Every parameter of every core *layer* must learn. (The loop gate and the
    # loop-halting head sit outside the layers and take no part in a one-loop
    # forward, so they legitimately receive no gradient here.)
    for name_, param in model.core.layers.named_parameters():
        assert param.grad is not None, f"{name_} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name_} has a non-finite gradient"
    if model.core.hyper is not None:
        for name_, param in model.core.hyper.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name_


def test_default_options_build_the_original_block_stack():
    """No option set: every core layer is the plain TransformerBlock, no streams."""
    from iridium.model.layers import GroupedQueryAttention, SwiGLU, TransformerBlock

    model = _model({})
    assert model.core.hyper is None
    for block in model.core.layers:
        assert type(block) is TransformerBlock
        assert type(block.attn) is GroupedQueryAttention and block.attn.window is None
        assert type(block.ffn) is SwiGLU


def test_gqa_layers_still_require_heads_to_tile_d_model():
    with pytest.raises(ConfigError, match="d_model"):
        _config(dict(layer_pattern=("mla", "global"), n_query_heads=16))


@pytest.mark.parametrize("bad", [dict(layer_pattern=("flash",)), dict(norm_kind="adaptive"),
                                 dict(block_kind="moe"), dict(hyper_streams=0),
                                 dict(mla_rope_dim=7)])
def test_invalid_options_are_refused_at_config_time(bad):
    with pytest.raises(ConfigError):
        _config(bad)


def test_new_options_and_tokenizer_fields_survive_serialisation():
    cfg = dataclasses.replace(_config(VARIANTS["everything"]),
                              text_vocab_size=8192,
                              codecs=dataclasses.replace(get_config("tiny").codecs,
                                                         vocab_size=8208))
    restored = IridiumConfig.from_dict(cfg.to_dict())
    assert restored.core == cfg.core
    assert restored.text_vocab_size == 8192
    assert restored.n_params == cfg.n_params
    import json
    assert IridiumConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))).core == cfg.core


def _ngram_model(dtype=torch.float64) -> Iridium1:
    tiny = get_config("tiny")
    cfg = dataclasses.replace(tiny, codecs=dataclasses.replace(tiny.codecs, ngram_table_size=97))
    torch.manual_seed(0)
    return Iridium1(cfg).to(dtype).eval()


def test_ngram_embeddings_are_costed_exactly():
    model = _ngram_model(torch.float32)
    assert sum(p.numel() for p in model.parameters()) == model.cfg.n_params
    assert model.cfg.n_params - get_config("tiny").n_params == 2 * 97 * model.cfg.core.d_model


@pytest.mark.parametrize("chunk", [1, 2, 5])
def test_ngram_embeddings_keep_cached_decoding_exact(chunk):
    """A decode step's n-gram reads tokens only the cache still has."""
    model = _ngram_model()
    batch = _batch(model.cfg, seed=3)
    with torch.no_grad():
        reference = model(batch, n_loops=1).hidden
        cached = run_chunked(model, batch, chunk=chunk, n_loops=1)
    torch.testing.assert_close(cached, reference, rtol=0, atol=1e-10)


def test_ngram_orders_survive_serialisation():
    import json
    cfg = _ngram_model(torch.float32).cfg
    restored = IridiumConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert restored.codecs == cfg.codecs and restored.n_params == cfg.n_params
