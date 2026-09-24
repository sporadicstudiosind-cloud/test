"""Named, ready-to-train configurations, ordered by what Iridium 1.0 is for.

The priority order is a product decision and the presets follow it:

1. **talking + reasoning** -- ``chat-34m``, ``chat-100m``
2. **tool use** -- ``tools-100m``
3. **omnimodality** -- ``omni-100m``
4. **physics / STEM** -- ``stem-100m``
5. **world model** -- ``world-100m``

plus two that are *costed, not trainable on free tiers*: ``modern-744m`` and
``8b``. Every preset is a complete recipe -- model config, data mixture,
tokenizer size, schedule, optimizer, step and batch budget -- so
``python -m iridium train --preset chat-34m`` is the whole instruction.

The budget column is sized for **free compute**: Colab's free T4, Kaggle's
30 GPU-hours a week (P100 or 2xT4), and Colab's free TPU v5e-1. Hosted inference
tiers (Groq, NVIDIA NIM and the like) serve models; they cannot train one.
Their use here is generating and grading data, and each hosted model's own
licence decides whether its outputs may train another model.

**Status labels are claims, and they are narrow.** ``verified in theory``
means the configuration builds, its parameter count matches the formula, and
the invariants in the test suite (cached decoding equals the full forward,
exact accounting, finite gradients) hold for it. It does **not** mean the
preset has been trained, or that training it will produce a good model.
Nothing in this repository has been trained at these recipes yet.

Time estimates are arithmetic, not measurements: training FLOPs are taken as
``3 x`` the config's own forward FLOPs per token (forward plus backward),
divided by the device's published peak at an *assumed* 30% utilisation. Real
small-model runs are often worse than that on free GPUs (data loading,
Python overhead, fp32 on cards without bf16), so read them as optimistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from .config import TEXT_ID_OFFSET, IridiumConfig, get_config

#: Duplicated from iridium.data (which imports numpy) so presets stay importable
#: with the standard library alone; a test holds the copies equal.
_TALK_MIX = {"fineweb_edu": 0.35, "wikipedia": 0.25, "cosmopedia_stories": 0.20,
             "cosmopedia_textbooks": 0.15, "finemath": 0.05}
_STEM_MIX = {"finemath": 0.35, "openwebmath": 0.15, "fineweb_edu": 0.25,
             "cosmopedia_textbooks": 0.15, "wikipedia": 0.10}

__all__ = ["Preset", "PRESETS", "get_preset", "FREE_TIERS", "estimate_hours", "preset_table"]


#: Published peak dense throughput (FLOP/s) and memory of free-tier devices.
#: ``precision`` is what the trainer would actually use there: the T4 and P100
#: have no bf16, and this codebase trains in fp32 rather than fp16 on them
#: (routed softmaxes overflow fp16; see docs/gpu.md), so the fp32 peak applies.
FREE_TIERS: dict[str, dict] = {
    "colab_t4":   {"label": "Colab free T4",        "peak": 8.1e12,  "memory_gb": 15, "precision": "fp32",
                   "quota": "~4-12 h sessions, availability varies"},
    "kaggle_p100": {"label": "Kaggle P100",         "peak": 9.3e12,  "memory_gb": 16, "precision": "fp32",
                    "quota": "30 GPU-h/week, 12 h sessions"},
    "kaggle_2xt4": {"label": "Kaggle 2xT4",         "peak": 16.2e12, "memory_gb": 30, "precision": "fp32",
                    "quota": "30 GPU-h/week; needs the multi-GPU placement path"},
    "tpu_v5e1":   {"label": "Colab free TPU v5e-1", "peak": 197e12, "memory_gb": 16, "precision": "bf16",
                   "quota": "availability varies; XLA path (Kaggle's is v5e-8, 20 h/week)"},
}
ASSUMED_UTILISATION = 0.30


@dataclass(frozen=True)
class Preset:
    name: str
    priority: int                      # 1 = talking ... 5 = world model; 0 = costed only
    goal: str
    config: IridiumConfig
    mixture: dict[str, float]
    steps: int
    batch_size: int
    window: int                        # training sequence length, in tokens
    lr: float
    optimizer: str = "eager_adamw"
    schedule: str = "wsd"
    ema_decay: float = 0.999
    loss_balance: str = "none"
    rounds: int = 4                    # fresh-data rounds; see run_preset
    free_tier: Optional[str] = "kaggle_p100"
    status: str = "verified in theory"
    notes: str = ""
    #: Source weights inside the ``text_lm`` and ``chat`` families; ``None``
    #: means the module defaults. See iridium.data.text_corpus / chat_corpus.
    text_mix: Optional[dict[str, float]] = None
    chat_mix: Optional[dict[str, float]] = None

    @property
    def tokens(self) -> int:
        return self.steps * self.batch_size * self.window

    @property
    def trainable_on_free_tier(self) -> bool:
        return self.free_tier is not None


def _with_vocab(cfg: IridiumConfig, text_vocab: int, **extra) -> IridiumConfig:
    """A rung with a subword vocabulary sized so no embedding row is dead."""
    return replace(cfg, text_vocab_size=text_vocab,
                   codecs=replace(cfg.codecs, vocab_size=text_vocab + TEXT_ID_OFFSET),
                   qk_norm=True, **extra)


def _presets() -> dict[str, Preset]:
    nano, n100 = get_config("nano"), get_config("nano100m")
    out: dict[str, Preset] = {}

    chat34 = replace(_with_vocab(nano, 8_192), name="iridium-1-chat-34m", max_seq_len=1024)
    out["chat-34m"] = Preset(
        "chat-34m", 1,
        "A small model that holds a conversation: prose plus human-written chats.",
        chat34, {"text_lm": 0.60, "chat": 0.40},
        steps=20_000, batch_size=32, window=512, lr=1.5e-3, free_tier="colab_t4", rounds=5,
        text_mix=_TALK_MIX,
        notes="The first thing to train. ~330M tokens, about half of compute-"
              "optimal for 36M params: roughly one free Colab session, optimistically.")

    chat100 = replace(_with_vocab(n100, 16_384), name="iridium-1-chat-100m", max_seq_len=2048)
    out["chat-100m"] = Preset(
        "chat-100m", 1,
        "The main talking model: prose, chat, and a little false-premise reasoning.",
        chat100, {"text_lm": 0.55, "chat": 0.35, "false_premise": 0.10},
        steps=20_000, batch_size=32, window=1024, lr=8e-4, rounds=10, text_mix=_TALK_MIX,
        notes="~655M tokens: about a third of compute-optimal for 104M params, "
              "spread over several weeks of Kaggle quota.")

    out["tools-100m"] = Preset(
        "tools-100m", 2,
        "chat-100m plus tool calling: licensed tool-use data and exactly-graded "
        "synthetic tool tasks, decoded under schema constraints.",
        replace(chat100, name="iridium-1-tools-100m"),
        {"text_lm": 0.40, "chat": 0.30, "tools": 0.30},
        steps=20_000, batch_size=32, window=1024, lr=8e-4, rounds=10, text_mix=_TALK_MIX,
        notes="Best started from a chat-100m checkpoint rather than from scratch.")

    omni = replace(chat100, name="iridium-1-omni-100m", mrope_sections=(8, 12, 12),
                   codecs=replace(chat100.codecs, continuous_conditioning="adaln",
                                  flow_timestep_sampling="logit_normal"))
    out["omni-100m"] = Preset(
        "omni-100m", 3,
        "Language plus the synthetic multimodal families (scenes, fields), with "
        "grid-aware positions for media and SD3-style flow heads.",
        omni, {"text_lm": 0.35, "chat": 0.20, "scene_goal": 0.15, "field_rollout": 0.15,
               "channel_depth": 0.15},
        steps=20_000, batch_size=32, window=1024, lr=8e-4, loss_balance="ema", rounds=10,
        notes="Loss balancing is on here because flow MSE and cross-entropy share "
              "the objective; it is a heuristic -- compare against 'none'.")

    out["stem-100m"] = Preset(
        "stem-100m", 4,
        "Physics and quantitative reasoning: the exactly-checkable fluid, channel "
        "and quantity families, kept anchored with language.",
        replace(chat100, name="iridium-1-stem-100m"),
        {"text_lm": 0.25, "chat": 0.10, "channel_depth": 0.20, "channel_intervention": 0.20,
         "field_rollout": 0.15, "false_premise": 0.10},
        steps=15_000, batch_size=32, window=1024, lr=8e-4, rounds=6, text_mix=_STEM_MIX)

    world = replace(omni, name="iridium-1-world-100m",
                    codecs=replace(omni.codecs, camera_features=6, n_modalities=10,
                                   point_features=14))
    out["world-100m"] = Preset(
        "world-100m", 5,
        "Camera-conditioned scenes: the camera modality and splat geometry enabled.",
        world, {"text_lm": 0.30, "chat": 0.20, "scene_goal": 0.25, "field_rollout": 0.25},
        steps=10_000, batch_size=32, window=1024, lr=8e-4, loss_balance="ema",
        status="verified in theory; no camera-posed training data yet",
        notes="The architecture is ready; the data is not. Nothing in the mixture "
              "carries camera tokens until a posed-video source is added.")

    out["modern-744m"] = Preset(
        "modern-744m", 0,
        "Every option with published evidence at scale; 1M-token-capable cache.",
        get_config("modern"), {"text_lm": 0.55, "chat": 0.30, "tools": 0.15},
        steps=100_000, batch_size=64, window=4096, lr=3e-4, optimizer="muon",
        free_tier=None, rounds=50,
        notes="Costed, not free-tier trainable: fp32 Adam state alone is ~12 GB and "
              "the token budget is weeks of a single A100.")

    return out


PRESETS: dict[str, Preset] = _presets()


def get_preset(name: str) -> Preset:
    if name == "8b":
        raise KeyError("8b is a rung, not a trainable preset; see `iridium report 8b` "
                       "and docs/training-8b.md")
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown preset {name!r}; known: {sorted(PRESETS)}") from None


def estimate_hours(preset: Preset, tier: str, tokens: Optional[int] = None) -> float:
    """Optimistic wall-clock hours to train ``preset`` on a free tier."""
    device = FREE_TIERS[tier]
    _, hi = preset.config.flops_per_token()
    forward = hi / preset.config.router.max_loops        # one loop, full routed depth
    total = 3.0 * forward * (tokens if tokens is not None else preset.tokens)
    return total / (device["peak"] * ASSUMED_UTILISATION) / 3600.0


def preset_table() -> str:
    head = (f"{'preset':<13} {'pri':>3} {'params':>12} {'tokens':>8}  "
            f"{'T4 h':>6} {'P100 h':>7} {'v5e h':>6}  status")
    lines = [head, "-" * len(head)]
    for p in sorted(PRESETS.values(), key=lambda p: (p.priority or 99, p.config.n_params, p.name)):
        hours = [estimate_hours(p, t) for t in ("colab_t4", "kaggle_p100", "tpu_v5e1")]
        lines.append(
            f"{p.name:<13} {p.priority or '-':>3} {p.config.n_params:>12,} "
            f"{p.tokens / 1e6:>7.0f}M  {hours[0]:>6.1f} {hours[1]:>7.1f} {hours[2]:>6.1f}  {p.status}")
    lines.append(f"(hours assume {ASSUMED_UTILISATION:.0%} of published peak; optimistic)")
    return "\n".join(lines)
