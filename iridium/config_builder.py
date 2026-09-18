"""Build any Iridium-1 geometry from parameters, and cost it honestly.

Two things live here:

* :func:`build` — every architectural knob as an argument, so a notebook can
  expose sliders for core depth, superstack count, superstack depth, width,
  head layout, routing and the ponder budget, and get a real
  :class:`~iridium.config.IridiumConfig` back.
* :data:`PRESETS` — named geometries from 50 M to 1 T, each *measured* against
  the parameter formulae rather than asserted, and each carrying the honest
  note about what it takes to train.

The distinction that matters: **describing a geometry and being able to train
it are different things.** A 1 T preset is a real configuration whose cost this
module computes exactly; it is not a promise that anything you own can fit it.
:func:`fits` answers that separately, from the device's actual memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import (
    CodecConfig,
    CoreConfig,
    IridiumConfig,
    RouterConfig,
    SuperstackConfig,
)

SPECIALIZATIONS = (
    "science_physics_simulation",
    "mathematics_symbolic_proof",
    "code_systems_tools",
    "language_reasoning_intent",
    "perception_geometry_action",
    "chemistry_materials",
    "biology_medicine",
    "earth_climate_astronomy",
    "design_ui_interaction",
    "audio_speech_music",
    "video_spatiotemporal",
    "retrieval_citation_grounding",
    "logic_proof_verification",
    "statistics_uncertainty",
    "optimization_control",
    "economics_operations",
    "law_policy_structure",
    "linguistics_translation",
    "planning_long_horizon",
    "tool_protocol_orchestration",
    "safety_refusal_boundary",
    "calibration_critique",
    "signal_estimation",
    "numerical_analysis",
    "graphics_rendering",
    "robotics_kinematics",
    "networks_distributed",
    "security_cryptography",
    "knowledge_encyclopedic",
    "commonsense_pragmatics",
    "creative_generation",
    "meta_reasoning",
)


def swiglu_ff(d_model: int, multiple_of: int = 128) -> int:
    """``8/3 * d`` rounded up to a hardware-friendly multiple."""
    raw = int(8 * d_model / 3)
    return ((raw + multiple_of - 1) // multiple_of) * multiple_of


def build(
    *,
    d_model: int,
    core_layers: int,
    n_superstacks: int,
    superstack_layers: int,
    d_head: int = 64,
    n_kv_heads: int = 4,
    d_ff: Optional[int] = None,
    top_k: int = 2,
    max_loops: int = 3,
    min_depth: Optional[int] = None,
    cross_stride: int = 4,
    spectral_stride: int = 8,
    spectral_modes: int = 16,
    spectral_channels: int = 64,
    spectral_stacks: tuple[int, ...] = (0,),
    vocab_size: int = 384,      # byte-level codec: 256 bytes + control ids
    max_seq_len: int = 4096,
    name: str = "iridium-1-custom",
    notes: str = "",
) -> IridiumConfig:
    """Assemble a configuration from the knobs a notebook would expose."""
    if d_model % d_head:
        raise ValueError(f"d_model {d_model} must be divisible by d_head {d_head}")
    n_query_heads = d_model // d_head
    if n_query_heads % n_kv_heads:
        raise ValueError(
            f"{n_query_heads} query heads must be a multiple of {n_kv_heads} kv heads"
        )
    d_ff = d_ff or swiglu_ff(d_model)
    min_depth = min_depth if min_depth is not None else max(1, superstack_layers // 4)
    spectral_stacks = tuple(i for i in spectral_stacks if i < n_superstacks)

    core = CoreConfig(
        d_model=d_model, n_layers=core_layers, n_query_heads=n_query_heads,
        n_kv_heads=n_kv_heads, d_head=d_head, d_ff=d_ff,
    )
    stacks = SuperstackConfig(
        n_stacks=n_superstacks,
        n_layers=superstack_layers,
        d_model=d_model,
        n_query_heads=n_query_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        d_ff=d_ff,
        cross_stride=cross_stride,
        spectral_stride=spectral_stride if spectral_stacks else 0,
        spectral_modes=spectral_modes,
        spectral_channels=spectral_channels,
        spectral_stacks=spectral_stacks,
        min_depth=min_depth,
        specializations=SPECIALIZATIONS[:n_superstacks]
        if n_superstacks <= len(SPECIALIZATIONS) else (),
    )
    # Routing to *every* stack is not routing. When ``top_k`` reaches the stack
    # count the gate is decorative: every token visits every bank, the balance
    # loss is satisfied by construction, and the model is a dense ensemble
    # wearing a router. Leave at least one stack unselected.
    top_k = min(top_k, max(1, n_superstacks - 1))
    return IridiumConfig(
        name=name,
        core=core,
        stacks=stacks,
        router=RouterConfig(top_k=top_k, max_loops=max_loops),
        codecs=CodecConfig(vocab_size=vocab_size),
        max_seq_len=max_seq_len,
        notes=notes,
    )


#: ``preset -> kwargs``. Counts are verified by tests/unit/test_presets.py.
#:
#: Every rung sets ``vocab_size=384`` and that is not a typo. The text codec is
#: byte-level: :func:`~iridium.codecs.spans.text_span` emits raw UTF-8 bytes
#: shifted past the control tokens, so 256 byte values plus room for control
#: ids is the whole reachable vocabulary. A 32,000- or 128,000-row embedding
#: would be a promise about a tokenizer that does not exist here — rows that
#: never receive a gradient, and a softmax over classes the data cannot
#: produce. At the 1 T rung that mistake was 1.3 B dead parameters. The budget
#: they were consuming now buys depth instead.
PRESET_SPECS: dict[str, dict] = {
    "50m": dict(d_model=384, core_layers=6, n_superstacks=2, superstack_layers=9, n_kv_heads=2, d_ff=896, vocab_size=384, max_seq_len=1024),
    "100m": dict(d_model=512, core_layers=6, n_superstacks=4, superstack_layers=8, n_kv_heads=2, d_ff=1024, vocab_size=384, max_seq_len=2048),
    "500m": dict(d_model=1024, core_layers=10, n_superstacks=2, superstack_layers=15, n_kv_heads=4, d_ff=2816, vocab_size=384, max_seq_len=4096),
    "1b": dict(d_model=1280, core_layers=12, n_superstacks=3, superstack_layers=14, n_kv_heads=4, d_ff=3456, vocab_size=384, max_seq_len=4096),
    "8b": dict(d_model=2048, core_layers=24, n_superstacks=4, superstack_layers=36, d_head=128, n_kv_heads=4, vocab_size=384, max_seq_len=8192),
    "16b": dict(d_model=2560, core_layers=28, n_superstacks=5, superstack_layers=38, d_head=128, n_kv_heads=4, vocab_size=384, max_seq_len=8192),
    "24b": dict(d_model=3072, core_layers=32, n_superstacks=4, superstack_layers=48, d_head=128, n_kv_heads=8, vocab_size=384, max_seq_len=8192),
    "100b": dict(d_model=5120, core_layers=48, n_superstacks=5, superstack_layers=60, d_head=128, n_kv_heads=8, vocab_size=384, max_seq_len=16384),
    "200b": dict(d_model=6144, core_layers=56, n_superstacks=5, superstack_layers=86, d_head=128, n_kv_heads=8, vocab_size=384, max_seq_len=32768),
    "1t": dict(d_model=10240, core_layers=64, n_superstacks=8, superstack_layers=103, d_head=128, n_kv_heads=16, cross_stride=8, spectral_stacks=(0, 1, 2, 3), vocab_size=384, max_seq_len=131072),
}


def preset(name: str) -> IridiumConfig:
    key = name.lower().replace(" ", "")
    if key not in PRESET_SPECS:
        raise ValueError(f"unknown preset {name!r}; known: {sorted(PRESET_SPECS)}")
    spec = dict(PRESET_SPECS[key])
    return build(name=f"iridium-1-{key}", **spec)


def all_presets() -> dict[str, IridiumConfig]:
    return {k: preset(k) for k in PRESET_SPECS}


def intelligence_preset(name='consumer') -> IridiumConfig:
    """Controller-first geometries; names describe use, not unverified quality.

    All specialist stacks are deeper than the general core. Larger geometries
    require substantial hardware; defining them does not implement sharding.
    """
    from dataclasses import replace
    choices = {
        'consumer_tiny': (256, 4, 4, 6, 2, 32),
        'consumer': (384, 8, 4, 12, 2, 64),
        'workstation': (768, 16, 6, 28, 4, 128),
        'research_large': (2048, 32, 12, 64, 8, 256),
        'frontier_design': (8192, 64, 32, 128, 16, 512),
    }
    if name not in choices:
        raise ValueError(f'unknown intelligence preset; choose {tuple(choices)}')
    width, core, banks, depth, kv, slots = choices[name]
    cfg = build(d_model=width, core_layers=core, n_superstacks=banks,
                superstack_layers=depth, n_kv_heads=kv, max_loops=8,
                max_seq_len=1024 if name.startswith('consumer') else 4096,
                name='iridium-controller-' + name)
    return replace(cfg, controller_mode=True, qk_norm=True, loop_identity=True,
                   memory_slots=slots, memory_stride=32, memory_rank=min(64, width // 4),
                   perception_layers=2 if name.startswith('consumer') else 4, perception_rank=64)


# --------------------------------------------------------------------------
# what it takes to train
# --------------------------------------------------------------------------

#: Bytes of optimizer + gradient + weight state per parameter, by strategy.
#: The weights themselves are included in each figure.
TRAINING_BYTES_PER_PARAM = {
    "adamw_fp32":   16.0,   # bf16 w + bf16 grad + fp32 master + 2 fp32 moments
    "adamw_bf16":   12.0,   # bf16 moments
    "adamw_8bit":    6.0,   # bnb 8-bit moments
    "adafactor":     6.0,   # factored second moment, no first moment
    "sgd":           4.0,   # bf16 weights + bf16 grads, no moments
    "lora":          2.2,   # frozen bf16 base + adapter state only
    "inference_bf16": 2.0,
    "inference_mxfp4": 0.53125,   # 4.25 bits, shared scale included
}


@dataclass(frozen=True)
class FitReport:
    config_name: str
    parameters: int
    strategy: str
    bytes_needed: float
    bytes_available: float
    fits: bool
    note: str

    def render(self) -> str:
        verdict = "FITS" if self.fits else "DOES NOT FIT"
        return (
            f"{self.config_name}: {self.parameters:,} params, {self.strategy} needs "
            f"{self.bytes_needed / 1e9:.1f} GB against {self.bytes_available / 1e9:.1f} GB "
            f"available — {verdict}. {self.note}"
        )


def fits(
    cfg: IridiumConfig,
    available_bytes: float,
    strategy: str = "adamw_fp32",
    activation_headroom: float = 0.25,
) -> FitReport:
    """Does this geometry train, on this much memory, under this strategy?

    ``activation_headroom`` reserves a fraction of the device for activations,
    workspace and fragmentation. It is a rule of thumb, not a measurement, and
    a long context or a large batch will exceed it.
    """
    if strategy not in TRAINING_BYTES_PER_PARAM:
        raise ValueError(f"unknown strategy {strategy!r}")
    need = cfg.n_params * TRAINING_BYTES_PER_PARAM[strategy]
    usable = available_bytes * (1.0 - activation_headroom)
    ok = need <= usable
    if ok:
        note = "state fits; activations still depend on batch and context."
    else:
        ratio = need / max(usable, 1)
        cheaper = [s for s, b in sorted(TRAINING_BYTES_PER_PARAM.items(), key=lambda kv: kv[1])
                   if cfg.n_params * b <= usable and not s.startswith("inference")]
        note = (f"short by {ratio:.1f}x. "
                + (f"Try strategy={cheaper[0]!r}." if cheaper
                   else "No single-device strategy fits; this needs sharding across devices."))
    return FitReport(cfg.name, cfg.n_params, strategy, need, usable, ok, note)


def ladder_table() -> str:
    header = (f"{'preset':<8}{'params':>16}{'core':>10}{'stacks':>12}{'d_model':>9}"
              f"{'active/tok':>13}{'bf16 wts':>11}{'AdamW fp32':>12}")
    lines = [header, "-" * len(header)]
    for key, cfg in all_presets().items():
        _, hi = cfg.active_parameters()
        lines.append(
            f"{key:<8}{cfg.n_params:>16,}{cfg.core.n_layers:>10}"
            f"{f'{cfg.stacks.n_stacks}x{cfg.stacks.n_layers}':>12}{cfg.core.d_model:>9}"
            f"{_si(hi):>13}{_si(cfg.n_params * 2):>11}{_si(cfg.n_params * 16):>12}"
        )
    return "\n".join(lines)


def _si(x: float) -> str:
    for unit, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= scale:
            return f"{x / scale:.2f}{unit}"
    return f"{x:.0f}"


if __name__ == "__main__":                              # pragma: no cover
    print(ladder_table())
