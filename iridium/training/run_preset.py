"""Train a named preset end to end: ``python -m iridium train --preset chat-34m``.

What this adds over the phase scripts is the thing free-tier training needs
most: **rounds of fresh data**. The corpus is built as an in-memory list of
items, which is fine for a few megabytes and impossible for the hundreds of
millions of tokens a preset asks for. So a run is split into ``rounds``: each
round streams a new slice of the text sources (skipping the documents earlier
rounds consumed), trains on it, and hands the trainer the next slice. The
optimizer, the step counter and the learning-rate schedule are never reset --
the schedule spans the whole run -- so the rounds are invisible to training
and only bound peak memory.

A round is also a natural checkpoint. A free Colab session can end without
notice; each round saves, and ``--resume`` continues from the last one.

``--dry-run`` does everything that needs no network and no training: builds
the model on the meta device, checks the parameter count against the formula,
prints the data-budget audit and the free-tier time estimates.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

from ..presets import FREE_TIERS, Preset, estimate_hours, get_preset


def describe(preset: Preset) -> str:
    cfg = preset.config
    lines = [
        f"preset {preset.name} (priority {preset.priority or 'costed-only'}): {preset.goal}",
        f"  model     {cfg.name}: {cfg.n_params:,} parameters, "
        f"text vocab {cfg.text_vocab_size or 'byte-level'}, context {cfg.max_seq_len}",
        f"  data      " + ", ".join(f"{k} {v:.0%}" for k, v in preset.mixture.items()),
        f"  budget    {preset.steps:,} steps x {preset.batch_size} x {preset.window} tokens "
        f"= {preset.tokens / 1e6:,.0f}M tokens in {preset.rounds} rounds",
        f"  recipe    {preset.optimizer}, {preset.schedule}, lr {preset.lr:g}, "
        f"EMA {preset.ema_decay or 'off'}, loss balance {preset.loss_balance}",
        f"  status    {preset.status}",
    ]
    for tier, spec in FREE_TIERS.items():
        lines.append(f"  est.      {spec['label']:<24} {estimate_hours(preset, tier):8.1f} h "
                     f"({spec['quota']})")
    if preset.notes:
        lines.append(f"  note      {preset.notes}")
    return "\n".join(lines)


def dry_run(preset: Preset) -> dict:
    """No network, no training: build on the meta device and cost it."""
    import torch

    from ..model.iridium1 import Iridium1
    from .budget import audit

    with torch.device("meta"):
        model = Iridium1(preset.config)
    built = sum(p.numel() for p in model.parameters())
    report = audit(preset.config, preset.tokens // preset.window, preset.steps,
                   preset.batch_size, window=preset.window, mixture=preset.mixture)
    print(describe(preset))
    print(report.describe())
    return {"parameters_built": built, "parameters_formula": preset.config.n_params,
            "match": built == preset.config.n_params}


def train_preset(preset: Preset, *, steps: Optional[int] = None, rounds: Optional[int] = None,
                 device: Optional[str] = None, out: str = "runs", init: Optional[str] = None,
                 resume: Optional[str] = None, seed: int = 0, max_data: bool = False) -> Path:
    """Train ``preset``; returns the final checkpoint path.

    ``max_data`` swaps in every registered chat source, including opt-in ones
    whose terms are non-commercial or unclear (see ``chat_corpus.MAX_CHAT_MIX``).
    The sources actually used, with their licences, are written to the
    checkpoint manifest either way.
    """
    import torch

    from ..model.iridium1 import Iridium1
    from ..runtime.checkpoint_compat import load_compatible
    from ..runtime.device import detect
    from .datasets import build_corpus
    from .tokenizer_bridge import tokenizer_for_config, tokenizer_manifest
    from .trainer import TrainConfig, Trainer

    if not preset.trainable_on_free_tier:
        print(f"warning: {preset.name} is costed for larger hardware than a free tier",
              file=sys.stderr)
    steps = steps or preset.steps
    rounds = max(1, min(rounds or preset.rounds, steps))
    info = detect(device)
    cfg = preset.config
    print(describe(replace(preset, steps=steps, rounds=rounds)))
    print(f"device: {info.describe()}")

    tokenizer = tokenizer_for_config(cfg)
    manifest = tokenizer_manifest(tokenizer, cfg)
    print("tokenizer:", {k: v for k, v in manifest.items() if k != "state"})
    if manifest.get("fell_back"):
        raise RuntimeError(
            "the preset's subword tokenizer could not be trained (no network or no "
            "`datasets` package); training it byte-level would silently mismatch its "
            "vocabulary. Install `datasets` and retry with network access.")

    torch.manual_seed(seed)
    model = Iridium1(cfg)
    if init:
        blob = torch.load(init, map_location="cpu", weights_only=False)
        report = load_compatible(model, blob["state_dict"])
        print(f"initialised from {init}: {report.summary()}")

    from ..data.chat_corpus import CHAT_SOURCES, DEFAULT_CHAT_MIX, MAX_CHAT_MIX
    from ..data.text_corpus import DEFAULT_MIX, SOURCES

    chat_mix = MAX_CHAT_MIX if max_data else (preset.chat_mix or DEFAULT_CHAT_MIX)
    text_mix = preset.text_mix or DEFAULT_MIX
    families = preset.mixture
    sources = {
        "text": [SOURCES[k].as_dict() for k in text_mix] if families.get("text_lm") else [],
        "chat": [CHAT_SOURCES[k].as_dict() for k in chat_mix] if families.get("chat") else [],
        "max_data": max_data,
    }
    if max_data:
        print("warning: --max-data includes opt-in sources whose terms are "
              "non-commercial or unclear; keep the result private", file=sys.stderr)
    # A subword token averages roughly four bytes of English, so a window of
    # N tokens holds about 4N bytes of conversation; bytes map one to one.
    chat_max_bytes = preset.window * (4 if tokenizer is not None else 1)

    items_per_round = max(1, math.ceil(steps * preset.batch_size / rounds))
    text_share = preset.mixture.get("text_lm", 0.0) / max(sum(preset.mixture.values()), 1e-9)

    def corpus_for(round_index: int):
        # Each earlier round consumed at most one document per text window, so
        # skipping that many rows guarantees fresh documents (over-skipping
        # costs streaming time, never correctness).
        skip = round_index * math.ceil(items_per_round * text_share)
        return build_corpus(items_per_round, seed=seed + round_index, split="train",
                            mixture=preset.mixture, tokenizer=tokenizer,
                            text_window=preset.window, text_skip_docs=skip,
                            text_mix=text_mix, chat_mix=chat_mix,
                            chat_max_bytes=chat_max_bytes)

    micro = max(1, min(preset.micro_batch, preset.batch_size))
    tcfg = TrainConfig(
        steps=steps, batch_size=micro, accumulate=math.ceil(preset.batch_size / micro),
        lr=preset.lr, seed=seed,
        optimizer=preset.optimizer, schedule=preset.schedule, ema_decay=preset.ema_decay,
        loss_balance=preset.loss_balance, max_length=min(preset.window, cfg.max_seq_len),
        log_every=max(steps // 100, 1), label=preset.name, warmup_ratio=0.02,
    )
    out_dir = Path(out) / preset.name
    trainer = Trainer(model, corpus_for(0), tcfg, out_dir=out_dir, device=info.device)
    start_round = 0
    if resume:
        trainer.resume(resume)
        start_round = min(rounds - 1, trainer.completed_steps * rounds // steps)
        print(f"resumed at step {trainer.completed_steps} (round {start_round})")
        if start_round:
            trainer.set_corpus(corpus_for(start_round))

    path: Optional[Path] = None
    for r in range(start_round, rounds):
        if r > start_round:
            trainer.set_corpus(corpus_for(r))
        until = steps if r == rounds - 1 else (r + 1) * steps // rounds
        trainer.train(until=until)
        path = trainer.save(f"round{r}", extra={"preset": preset.name, "round": r,
                                                "tokenizer": manifest, "sources": sources})
        print(f"round {r + 1}/{rounds} done at step {trainer.completed_steps}: {path}",
              flush=True)
    final = trainer.save("final", extra={"preset": preset.name, "tokenizer": manifest,
                                         "sources": sources})
    (out_dir / "preset.json").write_text(json.dumps(
        {"preset": preset.name, "steps": steps, "rounds": rounds, "status": preset.status,
         "trained": True, "tokenizer": manifest, "sources": sources}, indent=2), encoding="utf-8")
    return final or path
