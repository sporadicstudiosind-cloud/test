"""Presets: every one builds, costs exactly, and the round-based runner trains."""
from __future__ import annotations

import dataclasses
import json

import pytest

from iridium.presets import FREE_TIERS, PRESETS, estimate_hours, get_preset, preset_table
from iridium.training import run_preset


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_builds_and_matches_its_formula(name):
    assert run_preset.dry_run(get_preset(name))["match"]


def test_presets_follow_the_release_priority_order():
    order = {p.name: p.priority for p in PRESETS.values()}
    assert order["chat-34m"] == order["chat-100m"] == 1
    assert order["tools-100m"] == 2 and order["omni-100m"] == 3
    assert order["stem-100m"] == 4 and order["world-100m"] == 5


def test_free_tier_presets_fit_free_tier_memory():
    """fp32 Adam state (16 B/param) must fit the target device with headroom."""
    for p in PRESETS.values():
        if p.free_tier is None:
            continue
        state_gb = p.config.training_state_bytes() / 1e9
        assert state_gb < 0.5 * FREE_TIERS[p.free_tier]["memory_gb"], p.name


def test_every_preset_vocabulary_has_no_dead_rows():
    for p in PRESETS.values():
        if p.config.text_vocab_size:
            assert p.config.codecs.vocab_size == p.config.text_vocab_size + 16, p.name


def test_estimates_scale_with_tokens_and_device():
    p = get_preset("chat-34m")
    assert estimate_hours(p, "colab_t4", p.tokens * 2) == pytest.approx(
        2 * estimate_hours(p, "colab_t4"))
    assert estimate_hours(p, "tpu_v5e1") < estimate_hours(p, "colab_t4")
    assert "optimistic" in preset_table()


def test_8b_is_not_a_trainable_preset():
    with pytest.raises(KeyError, match="8b"):
        get_preset("8b")


def test_round_based_training_streams_fresh_corpora_and_checkpoints(tmp_path, monkeypatch):
    """Offline: synthetic families only, tokenizer bypassed, tiny budget.

    Checks the part that matters for free tiers: several rounds, one schedule,
    a checkpoint per round, the step count carried across rounds.
    """
    from iridium.training import tokenizer_bridge

    monkeypatch.setattr(tokenizer_bridge, "tokenizer_for_config", lambda cfg: None)
    monkeypatch.setattr(tokenizer_bridge, "tokenizer_manifest",
                        lambda tok, cfg=None: {"kind": "test", "fell_back": False})
    base = get_preset("chat-34m")
    preset = dataclasses.replace(
        base, mixture={"channel_depth": 1.0, "false_premise": 1.0},
        steps=4, batch_size=2, window=256, rounds=2, ema_decay=0.9)
    seen = []

    import iridium.training.datasets as datasets
    original = datasets.build_corpus

    def recording_build(n, **kw):
        seen.append(kw["seed"])
        return original(n, **kw)

    monkeypatch.setattr(datasets, "build_corpus", recording_build)
    final = run_preset.train_preset(preset, device="cpu", out=str(tmp_path))
    assert final is not None and final.exists()
    assert len(seen) == 2 and seen[0] != seen[1]          # a fresh corpus per round
    out = tmp_path / "chat-34m"
    assert (out / "chat-34m-round0.pt").exists() and (out / "chat-34m-round1.pt").exists()
    meta = json.loads((out / "preset.json").read_text())
    assert meta["steps"] == 4 and meta["rounds"] == 2


def test_a_trained_preset_checkpoint_chats_through_its_own_tokenizer(tmp_path, monkeypatch):
    """train --preset -> chat --checkpoint must round-trip the vocabulary.

    The checkpoint carries the tokenizer's merges; the chat loader (which
    refuses arbitrary pickles) must accept a trainer checkpoint and rebuild
    the exact tokenizer from it.
    """
    from iridium import cli
    from iridium.codecs.spans import Sample, text_span
    from iridium.data.tokenizer import BytePairTokenizer
    from iridium.config import get_config
    from iridium.presets import _with_vocab
    from iridium.runtime.generate import generate
    from iridium.training import tokenizer_bridge

    text = ["Tools answer questions; models call tools when facts are needed. "
            "A calculator adds numbers, a search engine finds pages."] * 30
    tok = BytePairTokenizer().train(text, 290)
    monkeypatch.setattr(tokenizer_bridge, "tokenizer_for_config", lambda cfg: tok)
    cfg = _with_vocab(get_config("tiny"), tok.vocab_size)
    preset = dataclasses.replace(
        get_preset("chat-34m"), config=cfg, mixture={"false_premise": 1.0},
        steps=2, batch_size=2, window=128, rounds=1)
    final = run_preset.train_preset(preset, device="cpu", out=str(tmp_path))

    model, manifest = cli._load_chat_checkpoint(final, "cpu")
    again = tokenizer_bridge.tokenizer_from_manifest(manifest)
    assert again is not None and again.encode(text[0]) == tok.encode(text[0])
    out = generate(model, Sample([text_span("A calculator", offset=16, tokenizer=again)]),
                   max_new_tokens=3, text_only=True, tokenizer=again)
    assert isinstance(out.text, str)


def test_preset_text_mixes_match_the_data_module_and_name_real_sources():
    from iridium import presets
    from iridium.data import text_corpus

    assert presets._TALK_MIX == text_corpus.TALK_MIX
    assert presets._STEM_MIX == text_corpus.STEM_MIX
    for p in presets.PRESETS.values():
        if p.text_mix:
            assert set(p.text_mix) <= set(text_corpus.SOURCES), p.name
            assert abs(sum(p.text_mix.values()) - 1) < 1e-9


def test_tools_family_builds_offline_from_synthetic_tasks(monkeypatch):
    from iridium.data import tool_corpus
    from iridium.training.datasets import build_corpus

    real = tool_corpus.tool_items
    monkeypatch.setattr(tool_corpus, "tool_items",
                        lambda n, **kw: real(n, **{**kw, "mix": {"synthetic": 1.0}}))
    corpus = build_corpus(6, mixture={"tools": 1.0}, text_window=1024)
    assert len(corpus.items) == 6
