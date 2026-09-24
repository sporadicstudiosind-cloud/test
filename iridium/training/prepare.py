"""Write a preset's language data to disk once: ``python -m iridium data prepare``.

Streams the preset's text, chat and tool sources, tokenizes them with the
preset's tokenizer (Rust ``tokenizers`` when installed, batched across
cores), and writes shards (see :mod:`iridium.data.shards`). Training then
reads by memory map, so RAM use is independent of the token budget and the
network is touched once, not once per round.

This is the step to run on a free CPU session (Kaggle's CPU notebooks do not
spend GPU quota); the shard directory is then attached to the GPU/TPU run.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

from ..presets import Preset

SHARDED_FAMILIES = ("text_lm", "chat", "tools")
_CHUNK = 4096


def plan(preset: Preset, tokens: Optional[int] = None) -> dict[str, int]:
    """Items each sharded family needs for the run's token budget.

    A packed text window and a conversation each fill (at most) one sequence,
    so items = sequences consumed x family share. Budgets beyond this repeat
    the shard, which the budget audit reports as epochs.
    """
    from .datasets import allocate_mixture

    sequences = math.ceil((tokens or preset.tokens) / preset.window)
    quotas = allocate_mixture(sequences, preset.mixture)
    return {k: v for k, v in quotas.items() if k in SHARDED_FAMILIES and v > 0}


def prepare(preset: Preset, out: str | Path = "data", *, tokens: Optional[int] = None,
            seed: int = 0, max_data: bool = False, split: str = "train") -> Path:
    from ..config import TEXT_ID_OFFSET
    from ..data.chat_corpus import CHAT_SOURCES, DEFAULT_CHAT_MIX, MAX_CHAT_MIX, iter_chat_items
    from ..data.shards import ShardWriter
    from ..data.text_corpus import DEFAULT_MIX, SOURCES, iter_text_windows
    from ..data.tokenization import check_fits
    from .tokenizer_bridge import tokenizer_for_config, tokenizer_manifest

    cfg = preset.config
    out_dir = Path(out) / preset.name
    tokenizer = tokenizer_for_config(cfg)
    manifest = tokenizer_manifest(tokenizer, cfg)
    if manifest.get("fell_back"):
        raise RuntimeError("the preset's subword tokenizer could not be trained (no network "
                           "or no `datasets`); install `datasets` and retry online")
    check_fits(tokenizer, cfg)
    rows = cfg.codecs.vocab_size
    text_mix = preset.text_mix or DEFAULT_MIX
    chat_mix = MAX_CHAT_MIX if max_data else (preset.chat_mix or DEFAULT_CHAT_MIX)
    chat_bytes = preset.window * (4 if tokenizer is not None else 1)
    base_meta = {"preset": preset.name, "split": split, "window": preset.window,
                 "tokenizer": manifest}

    written = {}
    for family, need in plan(preset, tokens).items():
        prefix = out_dir / f"{family}-{split}"
        started = time.time()
        if family == "text_lm":
            meta = {**base_meta, "sources": [SOURCES[k].as_dict() for k in text_mix]}
            w = ShardWriter(prefix, family, rows, meta)
            done = 0
            for ids, mask in iter_text_windows(text_mix, preset.window, tokenizer,
                                               seed=seed, split=split, limit=need):
                w.add_ids(ids, mask)
                done += 1
                if done % _CHUNK == 0:
                    _progress(family, done, need, started)
                if done >= need:
                    break
        elif family == "chat":
            meta = {**base_meta, "sources": [CHAT_SOURCES[k].as_dict() for k in chat_mix],
                    "max_data": max_data}
            w = ShardWriter(prefix, family, rows, meta)
            done = 0
            for it in iter_chat_items(chat_mix, seed=seed, max_bytes=chat_bytes, split=split,
                                      tokenizer=tokenizer, max_tokens=preset.window):
                w.add(it.sample)
                done += 1
                if done % _CHUNK == 0:
                    _progress(family, done, need, started)
                if done >= need:
                    break
            if done < need:
                print(f"note: chat sources hold {done:,} fitting conversations, fewer than "
                      f"the {need:,} planned; training will repeat them "
                      f"({need / max(done, 1):.1f} epochs)", file=sys.stderr)
        else:  # tools
            from ..data.tool_corpus import tool_items
            w = ShardWriter(prefix, family, rows, base_meta)
            done, k = 0, 0
            while done < need:
                n = min(_CHUNK, need - done)
                for it in tool_items(n + n // 2 + 8, seed=seed + k, split=split,
                                     tokenizer=tokenizer, max_bytes=chat_bytes):
                    if len(it.sample) <= preset.window and done < need:
                        w.add(it.sample)
                        done += 1
                k += 1
                _progress(family, done, need, started)
        written[family] = w.close()
    (out_dir / "prepared.json").write_text(json.dumps(
        {"preset": preset.name, "split": split, "tokens_planned": tokens or preset.tokens,
         "families": written, "id_offset": TEXT_ID_OFFSET}, indent=2, default=str))
    return out_dir


def _progress(family: str, done: int, need: int, started: float) -> None:
    rate = done / max(time.time() - started, 1e-6)
    print(f"[prepare] {family:<8} {done:>10,}/{need:,} items  ({rate:,.0f}/s)", flush=True)
