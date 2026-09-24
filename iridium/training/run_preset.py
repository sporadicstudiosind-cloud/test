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
    from ..presets import reference_hparams
    ref = reference_hparams(preset)
    lines.append(f"  reference DeepSeek-LLM fit for this compute: lr {ref['lr']:.2g}, batch "
                 f"{ref['batch_tokens'] / 1e3:,.0f}k tokens (preset: lr {preset.lr:g}, "
                 f"{preset.batch_size * preset.window / 1e3:,.0f}k)")
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
                 resume: Optional[str] = None, seed: int = 0, max_data: bool = False,
                 data: Optional[str] = None) -> Path:
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

    shard_dir = None
    if data:
        # Prepared shards carry the tokenizer that wrote them; use exactly that
        # one rather than retraining a possibly different vocabulary.
        from .tokenizer_bridge import tokenizer_from_manifest
        shard_dir = Path(data) / preset.name if (Path(data) / preset.name).is_dir() else Path(data)
        prepared = json.loads((shard_dir / "prepared.json").read_text())
        if prepared["preset"] != preset.name:
            raise ValueError(f"{shard_dir} was prepared for {prepared['preset']}, not {preset.name}")
        first = next(iter(prepared["families"].values()))
        tokenizer = tokenizer_from_manifest(first["tokenizer"])
    else:
        tokenizer = tokenizer_for_config(cfg)
    manifest = tokenizer_manifest(tokenizer, cfg)
    print("tokenizer:", {k: v for k, v in manifest.items() if k != "state"})
    if manifest.get("fell_back"):
        raise RuntimeError(
            "the preset's subword tokenizer could not be trained (no network or no "
            "`datasets` package); training it byte-level would silently mismatch its "
            "vocabulary. Install `datasets` and retry with network access.")

    from ..data.tokenization import check_fits
    check_fits(tokenizer, cfg)
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
        eval_every=max(steps // 20, 1) if shard_dir is not None else 0,
    )
    out_dir = Path(out) / preset.name
    if shard_dir is not None:
        # One corpus for the whole run, memory-mapped; rounds become nothing
        # more than checkpoint intervals.
        from ..data.shards import MixedCorpus
        from .datasets import allocate_mixture
        corpus = MixedCorpus.build(allocate_mixture(steps * preset.batch_size, preset.mixture),
                                   shard_dir, seed=seed, tokenizer=tokenizer)
        corpus_for = lambda r: corpus  # noqa: E731
    on_eval = None
    if shard_dir is not None:
        on_eval = held_out_evaluator(shard_dir, cfg, info.device, micro, preset.window)
    trainer = Trainer(model, corpus_for(0), tcfg, out_dir=out_dir, device=info.device)
    if on_eval is not None:
        on_eval.model = trainer.model
    start_round = 0
    if resume:
        trainer.resume(resume)
        start_round = min(rounds - 1, trainer.completed_steps * rounds // steps)
        print(f"resumed at step {trainer.completed_steps} (round {start_round})")
        if start_round and shard_dir is None:
            trainer.set_corpus(corpus_for(start_round))

    path: Optional[Path] = None
    for r in range(start_round, rounds):
        if r > start_round and shard_dir is None:
            trainer.set_corpus(corpus_for(r))
        until = steps if r == rounds - 1 else (r + 1) * steps // rounds
        trainer.train(until=until, on_eval=on_eval)
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


def held_out_evaluator(shard_dir, cfg, device, batch_size: int, window: int,
                       max_batches: int = 16):
    """Validation loss and bits per byte on the held-out shards, or ``None``.

    Bits per byte, not loss per token, is the number to compare: it does not
    depend on the tokenizer, so a 8k- and a 32k-vocabulary run (or a byte-level
    one) are on the same scale. ``bpb = loss_per_token / ln 2 * tokens / bytes``
    with tokens and bytes counted over the held-out text when it was written.
    """
    import torch

    from ..data.shards import MixedCorpus, Shard
    from .datasets import BatchLoader

    prepared = json.loads((Path(shard_dir) / "prepared.json").read_text())
    held = prepared.get("held_out") or {}
    if not held:
        return None
    loaders = {}
    for family, meta in held.items():
        n = len(Shard(Path(shard_dir) / f"{family}-test"))
        corpus = MixedCorpus.build({family: n}, shard_dir, split="test")
        loaders[family] = (BatchLoader(corpus, cfg.codecs, batch_size, seed=0, device=device,
                                       max_length=window),
                           meta.get("text_tokens", 0) / max(meta.get("utf8_bytes", 1), 1))

    @torch.no_grad()
    def evaluate(step: int) -> dict:
        out = {}
        for family, (loader, tokens_per_byte) in loaders.items():
            total, count = 0.0, 0
            for i, (batch, _) in enumerate(loader.batches()):
                if i >= max_batches:
                    break
                losses, _ = evaluate.model.losses(batch)
                total += float(losses["text"])
                count += 1
            loss = total / max(count, 1)
            out[f"val_{family}_loss"] = round(loss, 4)
            if tokens_per_byte:
                out[f"val_{family}_bpb"] = round(loss / math.log(2) * tokens_per_byte, 4)
        return out

    return evaluate
