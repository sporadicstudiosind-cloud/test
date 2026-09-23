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
    # Windows 4 and 5 exercise the cache while it is still shorter than the
    # window; an early trimming bug kept too few keys there and window 3 hid it.
    "local_w4": dict(layer_pattern=("local",), local_window=4),
    "local_w5": dict(layer_pattern=("local", "global"), local_window=5),
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


# -- the camera modality -------------------------------------------------------


def _camera_model(dtype=torch.float64) -> Iridium1:
    tiny = get_config("tiny")
    cfg = dataclasses.replace(tiny, codecs=dataclasses.replace(
        tiny.codecs, camera_features=6, n_modalities=10))
    torch.manual_seed(0)
    return Iridium1(cfg).to(dtype).eval()


def _camera_sample(cfg, seed=0):
    from iridium.codecs.spans import Span
    from iridium.world.camera import Camera, look_at
    from iridium.world.tokens import camera_span

    patch = cfg.codecs.image_patch
    cam = Camera.from_fov(60.0, 4 * patch, 2 * patch,
                          look_at((0.0, -1.0, -4.0), (0.0, 0.0, 0.0), dtype=torch.float64))
    rng = np.random.default_rng(seed)
    image_dim = continuous_dims(cfg.codecs)["image"]
    return Sample([text_span("see"),
                   camera_span(cam, patch=patch),
                   Span("image", rng.normal(size=(8, image_dim))),
                   text_span("ok")])


def test_camera_modality_is_opt_in_and_leaves_the_default_model_unchanged():
    from iridium.codecs.spans import MODALITIES, MODALITY_INDEX

    assert MODALITIES[:9] == ("control", "text", "image", "video", "audio", "field",
                              "geometry", "action", "quantity")
    assert MODALITY_INDEX["camera"] == 9
    assert get_config("tiny").codecs.n_modalities == 9
    assert "camera" not in Iridium1(get_config("tiny")).codecs.encoders


def test_camera_modality_is_costed_exactly_and_has_no_decoder():
    model = _camera_model(torch.float32)
    assert sum(p.numel() for p in model.parameters()) == model.cfg.n_params
    assert "camera" in model.codecs.encoders and "camera" not in model.codecs.decoders


def test_camera_modality_requires_a_modality_slot():
    tiny = get_config("tiny")
    with pytest.raises(ConfigError, match="n_modalities"):
        dataclasses.replace(tiny.codecs, camera_features=6)


def test_camera_tokens_change_the_representation_and_keep_decoding_exact():
    model = _camera_model()
    cfg = model.cfg
    sample = _camera_sample(cfg)
    batch = TensorBatch(collate([sample], continuous_dims(cfg.codecs)), dtype=torch.float64)
    with torch.no_grad():
        reference = model(batch, n_loops=1).hidden
        cached = run_chunked(model, batch, chunk=1, n_loops=1)
        # Move the camera: the final state must depend on where it looks.
        moved = batch.continuous["camera"].clone()
        moved[..., :3] = -moved[..., :3]
        batch.continuous["camera"] = moved
        other = model(batch, n_loops=1).hidden
    torch.testing.assert_close(cached, reference, rtol=0, atol=1e-10)
    assert not torch.allclose(other[0, -1], reference[0, -1])


def test_a_camera_span_without_the_modality_enabled_is_refused():
    cfg = get_config("tiny")
    with pytest.raises(ValueError, match="camera_features"):
        collate([_camera_sample(cfg)], continuous_dims(cfg.codecs))


def test_generation_never_emits_a_camera_token():
    from iridium.runtime.generate import generate

    model = _camera_model(torch.float32)
    with torch.no_grad():
        model.codecs.slot_type_head.proj.bias.zero_()
        model.codecs.slot_type_head.proj.bias[9] = 1e4   # make "camera" the argmax
    out = generate(model, _camera_sample(model.cfg), max_new_tokens=3, allow_continuous=True)
    assert "camera" not in out.modalities


# -- per-layer embeddings ------------------------------------------------------


def _ple_model(dtype=torch.float64, **core) -> Iridium1:
    tiny = get_config("tiny")
    cfg = dataclasses.replace(tiny, ple_dim=8,
                              core=dataclasses.replace(tiny.core, **core))
    torch.manual_seed(0)
    return Iridium1(cfg).to(dtype).eval()


def test_per_layer_embeddings_are_costed_exactly():
    model = _ple_model(torch.float32)
    cfg = model.cfg
    assert sum(p.numel() for p in model.parameters()) == cfg.n_params
    n_layers, vocab, d = cfg.core.n_layers, cfg.codecs.vocab_size, cfg.core.d_model
    assert model.parameter_inventory()["per_layer_embedding"] == vocab * n_layers * 8 + n_layers * 8 * d


def test_per_layer_embeddings_are_function_preserving_until_trained():
    """Zero-initialised projections: switching PLE on changes nothing at step 0."""
    torch.manual_seed(0)
    plain = Iridium1(get_config("tiny")).double().eval()
    with_ple = _ple_model()
    state = {k: v for k, v in with_ple.state_dict().items() if not k.startswith("ple.")}
    plain.load_state_dict(state)
    batch = _batch(plain.cfg, seed=4)
    with torch.no_grad():
        torch.testing.assert_close(with_ple(batch, n_loops=1).hidden,
                                   plain(batch, n_loops=1).hidden, rtol=0, atol=0)


@pytest.mark.parametrize("core", [{}, {"hyper_streams": 3, "layer_pattern": ("mla", "global"),
                                       "mla_kv_rank": 24, "mla_rope_dim": 8}])
def test_per_layer_embeddings_keep_cached_decoding_exact(core):
    model = _ple_model(**core)
    with torch.no_grad():
        for proj in model.ple.projections:     # make them matter
            proj.weight.normal_(std=0.1)
    batch = _batch(model.cfg, seed=5)
    with torch.no_grad():
        reference = model(batch, n_loops=1).hidden
        cached = run_chunked(model, batch, chunk=1, n_loops=1)
    torch.testing.assert_close(cached, reference, rtol=0, atol=1e-10)


def test_per_layer_embeddings_train():
    model = _ple_model(torch.float32).train()
    losses, _ = model.losses(_batch(model.cfg, dtype=torch.float32), n_loops=1)
    sum(losses.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.ple.parameters())


# -- superstack options --------------------------------------------------------

STACK_VARIANTS = {
    "deltanet": dict(layer_pattern=("deltanet", "global")),
    "mla": dict(layer_pattern=("mla",), mla_kv_rank=24, mla_rope_dim=8),
    "local": dict(layer_pattern=("local",), local_window=2),
    "dyt_parallel_dynamic": dict(norm_kind="dyt", block_kind="parallel", ffn_dynamic_rank=4),
}


def _stack_model(options: dict, dtype=torch.float64, core: dict | None = None,
                 controller: bool = False) -> Iridium1:
    tiny = get_config("tiny")
    cfg = dataclasses.replace(
        tiny, controller_mode=controller,
        stacks=dataclasses.replace(tiny.stacks, **options),
        core=dataclasses.replace(tiny.core, **(core or {})),
    )
    torch.manual_seed(0)
    return Iridium1(cfg).to(dtype).eval()


@pytest.mark.parametrize("name", sorted(STACK_VARIANTS))
def test_superstack_options_are_costed_exactly(name):
    model = _stack_model(STACK_VARIANTS[name], torch.float32)
    assert sum(p.numel() for p in model.parameters()) == model.cfg.n_params
    assert model.parameter_inventory()["superstacks"] == model.cfg.stacks.params


@pytest.mark.parametrize("name", sorted(STACK_VARIANTS))
@pytest.mark.parametrize("loops", [1, 2])
def test_superstack_options_keep_cached_decoding_exact(name, loops):
    """Stack caches are keyed on each stack's packed routed sequence; every
    layer kind must still reproduce the uncached forward."""
    model = _stack_model(STACK_VARIANTS[name])
    batch = _batch(model.cfg, seed=6)
    with torch.no_grad():
        reference = model(batch, n_loops=loops).hidden
        cached = run_chunked(model, batch, chunk=1, n_loops=loops)
    torch.testing.assert_close(cached, reference, rtol=0, atol=1e-10)


def test_bridges_stay_gqa_whatever_the_stack_options():
    from iridium.model.layers import BridgeCrossAttention

    model = _stack_model(STACK_VARIANTS["mla"])
    bridges = [layer.bridge for stack in model.bank.stacks for layer in stack.layers
               if layer.bridge is not None]
    assert bridges and all(type(b) is BridgeCrossAttention for b in bridges)


def test_everything_everywhere_in_controller_mode():
    """Core and stack options together, under controller mode's full-core cycles."""
    model = _stack_model(STACK_VARIANTS["deltanet"], controller=True,
                         core=VARIANTS["everything"])
    assert sum(p.numel() for p in model.parameters()) == model.cfg.n_params
    batch = _batch(model.cfg, seed=7)
    with torch.no_grad():
        reference = model(batch, n_loops=2).hidden
        cached = run_chunked(model, batch, chunk=1, n_loops=2)
    torch.testing.assert_close(cached, reference, rtol=0, atol=1e-10)


# -- cache accounting against the real cache -----------------------------------


def _tensor_bytes(entry) -> int:
    if torch.is_tensor(entry):
        return entry.numel() * entry.element_size()
    if isinstance(entry, (tuple, list)):
        return sum(_tensor_bytes(e) for e in entry)
    return 0


@pytest.mark.parametrize("pattern", [("global",), ("mla",), ("deltanet", "global"),
                                     ("local", "deltanet", "mla", "global")])
def test_core_cache_formula_matches_the_bytes_actually_cached(pattern):
    """Sum the core's real cache entries after decoding T tokens and compare
    with CoreConfig.cache_bytes(T). Priced per layer kind: GQA K/V, MLA latent,
    local capped at its window, DeltaNet a fixed state."""
    options = dict(layer_pattern=pattern, local_window=4, mla_kv_rank=24, mla_rope_dim=8)
    model = _model(options)
    batch = _batch(model.cfg, n=1, seed=8)
    cache: dict = {}
    with torch.no_grad():
        run_chunked(model, batch, chunk=1, n_loops=1, cache=cache)
    t = batch.modality.shape[1]
    held = sum(_tensor_bytes(v) for k, v in cache.items()
               if isinstance(k, tuple) and k[:2] == ("core", 0))
    assert held == model.cfg.core.cache_bytes(t, bytes_per_element=8)


# -- M-RoPE and YaRN -------------------------------------------------------------


def _mrope_model(dtype=torch.float64, **extra) -> Iridium1:
    tiny = get_config("tiny")               # core d_head 16 -> 8 rotary pairs
    cfg = dataclasses.replace(tiny, mrope_sections=(2, 3, 3), **extra)
    torch.manual_seed(0)
    return Iridium1(cfg).to(dtype).eval()


def _image_sample(cfg, grid, seed=0):
    from iridium.codecs.spans import Span
    rng = np.random.default_rng(seed)
    dim = continuous_dims(cfg.codecs)["image"]
    n = int(np.prod(grid))
    return Sample([text_span("look"), Span("image", rng.normal(size=(n, dim)), grid=grid),
                   text_span("done")])


def test_rope_positions_follow_the_grid():
    from iridium.codecs.spans import collate as _collate
    cfg = get_config("tiny")
    batch = _collate([_image_sample(cfg, (2, 3))], continuous_dims(cfg.codecs))
    rp, pos = batch.rope_positions[0], batch.positions[0]
    s = 4                                     # "look" is 4 tokens
    assert (rp[:s] == np.stack([pos[:s]] * 3, -1)).all()           # text on the diagonal
    assert rp[s + 4].tolist() == [s, s + 1, s + 1]                   # row 1, col 1
    assert (rp[s + 6:] == np.stack([pos[s + 6:]] * 3, -1)).all()     # text resumes 1-D


def test_mrope_leaves_text_bit_identical():
    plain = _model({})
    mrope = _mrope_model()
    mrope.load_state_dict(plain.state_dict())
    batch = _batch(plain.cfg, seed=9)
    with torch.no_grad():
        torch.testing.assert_close(mrope(batch, n_loops=1).hidden,
                                   plain(batch, n_loops=1).hidden, rtol=0, atol=0)


def test_mrope_makes_image_layout_matter():
    """A 2x4 and a 4x2 image of the same patches differ only in layout; with
    M-RoPE the model can tell them apart, without it it cannot."""
    def final_state(model, grid):
        batch = TensorBatch(collate([_image_sample(model.cfg, grid)],
                                    continuous_dims(model.cfg.codecs)), dtype=torch.float64)
        with torch.no_grad():
            return model(batch, n_loops=1).hidden[0, -1]
    plain, mrope = _model({}), _mrope_model()
    assert torch.equal(final_state(plain, (2, 4)), final_state(plain, (4, 2)))
    assert not torch.allclose(final_state(mrope, (2, 4)), final_state(mrope, (4, 2)))


@pytest.mark.parametrize("chunk", [1, 3])
def test_mrope_keeps_cached_decoding_exact_with_images(chunk):
    model = _mrope_model()
    batch = TensorBatch(collate([_image_sample(model.cfg, (2, 3), seed=1)],
                                continuous_dims(model.cfg.codecs)), dtype=torch.float64)
    with torch.no_grad():
        reference = model(batch, n_loops=1).hidden
        from iridium.runtime.decode import run_atomic_chunked
        cached = run_atomic_chunked(model, batch, chunk=chunk, n_loops=1)
    torch.testing.assert_close(cached, reference, rtol=0, atol=1e-10)


def test_mrope_and_yarn_validation_and_serialisation():
    import json
    tiny = get_config("tiny")
    with pytest.raises(ConfigError, match="mrope_sections"):
        dataclasses.replace(tiny, mrope_sections=(2, 2, 2))
    with pytest.raises(ConfigError, match="rope_original_max_position"):
        dataclasses.replace(tiny, rope_yarn_factor=4.0)
    cfg = dataclasses.replace(tiny, mrope_sections=(2, 3, 3), rope_yarn_factor=4.0,
                              rope_original_max_position=256)
    restored = IridiumConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert restored.mrope_sections == (2, 3, 3) and restored.rope_yarn_factor == 4.0
    model = Iridium1(cfg)
    assert model.rope.scaling["type"] == "yarn"
