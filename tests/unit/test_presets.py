"""Every preset on the ladder must cost exactly what the builder claims.

Three layers of check, weakest to strongest:

1. **Nameplate honesty.** A preset called ``8b`` has to be near 8 B. The
   tolerance is stated, not implied, so a geometry that drifts fails loudly
   instead of quietly shipping a "24b" that is really 19 B.
2. **Independent arithmetic.** The transformer body is recomputed here from
   the architecture description — retyped, not imported — and demanded equal
   to ``config.py`` to the parameter. A shared helper would make this
   tautological; the duplication is the point.
3. **Real modules.** At the rungs small enough to instantiate, the formula is
   checked against ``sum(p.numel())`` of the actual torch graph, delta zero.

The giant rungs get (1) and (2) only. Nothing here claims they are trainable:
that is :func:`~iridium.config_builder.fits`, tested separately below.
"""

import pytest
import torch

from iridium.config_builder import (
    PRESET_SPECS,
    SPECIALIZATIONS,
    TRAINING_BYTES_PER_PARAM,
    all_presets,
    build,
    fits,
    preset,
    swiglu_ff,
)
from iridium.codecs.bank import TensorBatch, continuous_dims
from iridium.codecs.spans import Sample, collate, text_span
from iridium.model.iridium1 import Iridium1

#: preset -> nameplate parameter count.
TARGETS = {
    "50m": 50e6, "100m": 100e6, "500m": 500e6, "1b": 1e9, "8b": 8e9,
    "16b": 16e9, "24b": 24e9, "100b": 100e9, "200b": 200e9, "1t": 1e12,
}

#: Rungs cheap enough to actually build on a test runner.
INSTANTIABLE = ("50m", "100m")

NAMEPLATE_TOLERANCE = 0.05


def test_every_preset_has_a_target():
    assert set(PRESET_SPECS) == set(TARGETS), (
        "a preset was added or renamed without a nameplate target"
    )


@pytest.mark.parametrize("name", sorted(TARGETS))
def test_nameplate_is_honest(name):
    cfg = preset(name)
    ratio = cfg.n_params / TARGETS[name]
    assert abs(ratio - 1.0) <= NAMEPLATE_TOLERANCE, (
        f"{name}: {cfg.n_params:,} params is {ratio:.3f}x its {TARGETS[name]:,.0f} "
        f"nameplate, outside the {NAMEPLATE_TOLERANCE:.0%} tolerance"
    )


# ---------------------------------------------------------------------------
# independent arithmetic
# ---------------------------------------------------------------------------

def _layer_params(d: int, d_kv: int, d_ff: int) -> int:
    """Q/O at full width, K/V at GQA width, SwiGLU's three matrices, two norms."""
    return 2 * d * d + 2 * d * d_kv + 3 * d * d_ff + 2 * d


def _bridge_params(d: int, d_kv: int) -> int:
    """Cross-attention onto the core cache: Q and O full width, K/V from the core."""
    return 2 * d * d + 2 * d * d_kv + d


def _spectral_params(d: int, c: int, m: int, dims: int) -> int:
    """Lift, ``2**(dims-1)`` complex corner blocks, pointwise path, project, norm."""
    corners = 2 ** (dims - 1)
    return (
        (d * c + c)                         # lift
        + (c * d + d)                       # project
        + corners * c * c * m ** dims * 2   # complex weights, two reals each
        + (c * c + c)                       # pointwise residual path
        + d                                 # norm
    )


def _expected_body(spec: dict) -> tuple[int, int]:
    """Recompute (core, superstacks) from the spec alone. No config.py arithmetic."""
    d = spec["d_model"]
    d_head = spec.get("d_head", 64)
    d_kv = spec.get("n_kv_heads", 4) * d_head
    d_ff = spec.get("d_ff") or swiglu_ff(d)
    core_layers = spec["core_layers"]
    n_stacks = spec["n_superstacks"]
    stack_layers = spec["superstack_layers"]
    cross_stride = spec.get("cross_stride", 4)
    spectral_stride = spec.get("spectral_stride", 8)
    spectral_stacks = tuple(
        i for i in spec.get("spectral_stacks", (0,)) if i < n_stacks
    )

    per_layer = _layer_params(d, d_kv, d_ff)
    core = core_layers * per_layer + d + d + (d + 1)   # layers, out-norm, gate, halt head

    n_cross = len(range(0, stack_layers, cross_stride))
    n_spec = len(range(0, stack_layers, spectral_stride)) if spectral_stride > 0 else 0
    one_spectral = _spectral_params(
        d,
        spec.get("spectral_channels", 64),
        spec.get("spectral_modes", 16),
        dims=2,
    )
    plain = (
        stack_layers * per_layer
        + n_cross * _bridge_params(d, d_kv)
        + stack_layers * (d + 1)   # per-layer halting head (PonderNet in depth)
        + d                        # exit norm
        + 1                        # focus gain
    )
    stacks = n_stacks * plain + len(spectral_stacks) * n_spec * one_spectral
    return core, stacks


@pytest.mark.parametrize("name", sorted(TARGETS))
def test_body_matches_independent_arithmetic(name):
    cfg = preset(name)
    core, stacks = _expected_body(PRESET_SPECS[name])
    assert cfg.core.params == core, (
        f"{name} core: config says {cfg.core.params:,}, retyped formula says {core:,}"
    )
    assert cfg.stacks.params == stacks, (
        f"{name} stacks: config says {cfg.stacks.params:,}, retyped formula says {stacks:,}"
    )


@pytest.mark.parametrize("name", sorted(TARGETS))
def test_total_is_the_sum_of_its_parts(name):
    cfg = preset(name)
    parts = cfg.parameter_parts()
    assert sum(parts.values()) == cfg.n_params
    assert parts["control_core"] == cfg.core.params
    assert parts["superstacks"] == cfg.stacks.params


@pytest.mark.parametrize("name", INSTANTIABLE)
def test_small_presets_match_real_modules_exactly(name):
    cfg = preset(name)
    model = Iridium1(cfg)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == cfg.n_params, (
        f"{name}: formula says {cfg.n_params:,}, torch graph has {actual:,} "
        f"(delta {actual - cfg.n_params:+,})"
    )


@pytest.mark.parametrize("name", INSTANTIABLE)
def test_small_presets_run_a_forward_pass(name):
    """A geometry that costs out correctly but cannot run is not a model."""
    torch.manual_seed(0)
    cfg = preset(name)
    model = Iridium1(cfg).eval()
    dims = continuous_dims(cfg.codecs)
    batch = TensorBatch(collate([Sample([text_span("routing carries the token")])], dims))
    with torch.no_grad():
        out = model(batch)
        logits = model.codecs.text_head(out.hidden)
    t = batch.shape[1]
    assert out.hidden.shape == (1, t, cfg.core.d_model)
    assert logits.shape == (1, t, cfg.codecs.vocab_size)
    assert torch.isfinite(out.hidden).all() and torch.isfinite(logits).all()
    assert out.decisions, "the router made no decision at all"
    for decision in out.decisions:
        assert decision.stack_index.shape == (1, t, cfg.router.top_k)
        assert int(decision.stack_index.max()) < cfg.stacks.n_stacks
        assert torch.allclose(
            decision.stack_weight.sum(-1), torch.ones(1, t), atol=1e-5
        )
        depth = decision.target_depth
        assert int(depth.min()) >= cfg.stacks.min_depth
        assert int(depth.max()) <= cfg.stacks.n_layers


# ---------------------------------------------------------------------------
# structure the architecture insists on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(TARGETS))
def test_superstacks_are_deeper_than_the_core(name):
    """The core dispatches; the stacks do the work. Invert that and it is a
    plain transformer with a routing gimmick bolted on."""
    cfg = preset(name)
    assert cfg.stacks.n_layers > cfg.core.n_layers, (
        f"{name}: stacks {cfg.stacks.n_layers}L are not deeper than core "
        f"{cfg.core.n_layers}L"
    )


@pytest.mark.parametrize("name", sorted(TARGETS))
def test_head_geometry_is_valid(name):
    cfg = preset(name)
    assert cfg.core.d_model == cfg.core.n_query_heads * cfg.core.d_head
    assert cfg.core.n_query_heads % cfg.core.n_kv_heads == 0
    assert cfg.stacks.d_model == cfg.core.d_model


@pytest.mark.parametrize("name", sorted(TARGETS))
def test_routing_and_specialisation_are_consistent(name):
    cfg = preset(name)
    assert 1 <= cfg.router.top_k <= cfg.stacks.n_stacks
    assert len(cfg.stacks.specializations) == cfg.stacks.n_stacks
    assert len(set(cfg.stacks.specializations)) == cfg.stacks.n_stacks
    assert all(i < cfg.stacks.n_stacks for i in cfg.stacks.spectral_stacks)
    assert 1 <= cfg.stacks.min_depth <= cfg.stacks.n_layers


@pytest.mark.parametrize("name", sorted(TARGETS))
def test_active_parameters_are_a_strict_subset_of_total(name):
    """Routing is only worth anything if a token touches less than everything."""
    cfg = preset(name)
    lo, hi = cfg.active_parameters()
    assert 0 < lo <= hi < cfg.n_params
    assert hi < cfg.n_params, f"{name}: every parameter is active — routing bought nothing"


def test_ladder_is_monotone():
    """Sorting by nameplate must sort by real count, width and depth alike."""
    order = sorted(TARGETS, key=TARGETS.get)
    counts = [preset(n).n_params for n in order]
    assert counts == sorted(counts), dict(zip(order, counts))
    widths = [preset(n).core.d_model for n in order]
    assert widths == sorted(widths), dict(zip(order, widths))


# ---------------------------------------------------------------------------
# the builder's guard rails
# ---------------------------------------------------------------------------

def test_build_rejects_indivisible_width():
    with pytest.raises(ValueError, match="divisible"):
        build(d_model=100, core_layers=2, n_superstacks=2, superstack_layers=4, d_head=64)


def test_build_rejects_impossible_gqa():
    with pytest.raises(ValueError, match="kv heads"):
        build(d_model=384, core_layers=2, n_superstacks=2, superstack_layers=4,
              d_head=64, n_kv_heads=5)


def test_build_clamps_top_k_and_spectral_stacks():
    cfg = build(d_model=256, core_layers=2, n_superstacks=1, superstack_layers=4,
                top_k=4, spectral_stacks=(0, 3, 9))
    assert cfg.router.top_k == 1
    assert cfg.stacks.spectral_stacks == (0,)


def test_custom_build_is_costed_exactly():
    spec = dict(d_model=512, core_layers=4, n_superstacks=3, superstack_layers=6,
                n_kv_heads=2, vocab_size=8192, max_seq_len=512)
    cfg = build(**spec)
    core, stacks = _expected_body(spec)
    assert (cfg.core.params, cfg.stacks.params) == (core, stacks)
    assert sum(p.numel() for p in Iridium1(cfg).parameters()) == cfg.n_params


def test_swiglu_ff_rounds_up_to_a_multiple():
    for d in (384, 512, 1024, 1280, 2048, 5120):
        ff = swiglu_ff(d)
        assert ff % 128 == 0
        assert ff >= 8 * d / 3


def test_specialisations_cover_the_widest_preset():
    widest = max(PRESET_SPECS.values(), key=lambda s: s["n_superstacks"])
    assert widest["n_superstacks"] <= len(SPECIALIZATIONS)


def test_all_presets_builds_the_whole_ladder():
    presets = all_presets()
    assert set(presets) == set(PRESET_SPECS)
    assert all(cfg.name.startswith("iridium-1-") for cfg in presets.values())


# ---------------------------------------------------------------------------
# fitting on real hardware
# ---------------------------------------------------------------------------

def test_fit_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="strategy"):
        fits(preset("50m"), 16e9, strategy="wishful")


def test_cheaper_strategies_never_need_more():
    cfg = preset("1b")
    by_cost = sorted(TRAINING_BYTES_PER_PARAM.items(), key=lambda kv: kv[1])
    needs = [fits(cfg, 1e12, strategy=s).bytes_needed for s, _ in by_cost]
    assert needs == sorted(needs)


def test_a_trillion_does_not_fit_on_one_card():
    report = fits(preset("1t"), 80e9, strategy="adamw_fp32")
    assert not report.fits
    assert "shard" in report.note or "Try strategy" in report.note
    assert "DOES NOT FIT" in report.render()


def test_small_preset_fits_a_free_tier_card():
    """16 GB is a T4 or a Colab-class allocation; 50m must train there."""
    report = fits(preset("50m"), 16e9, strategy="adamw_fp32")
    assert report.fits, report.render()


def test_headroom_is_actually_reserved():
    cfg = preset("500m")
    exact = cfg.n_params * TRAINING_BYTES_PER_PARAM["adamw_fp32"]
    assert not fits(cfg, exact, strategy="adamw_fp32", activation_headroom=0.25).fits
    assert fits(cfg, exact, strategy="adamw_fp32", activation_headroom=0.0).fits


def test_suggested_fallback_strategy_really_fits():
    cfg = preset("8b")
    report = fits(cfg, 24e9, strategy="adamw_fp32")
    if not report.fits and "Try strategy=" in report.note:
        suggested = report.note.split("Try strategy=")[1].strip("'.\"")
        assert fits(cfg, 24e9, strategy=suggested).fits, (
            f"{suggested!r} was recommended but does not fit either"
        )
