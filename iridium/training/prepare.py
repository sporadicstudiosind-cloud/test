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

import numpy as np
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

    ctx = dict(out_dir=out_dir, rows=rows, tokenizer=tokenizer, window=preset.window,
               text_mix=text_mix, chat_mix=chat_mix, chat_bytes=chat_bytes, seed=seed,
               base_meta=base_meta, max_data=max_data)
    written = {}
    for family, need in plan(preset, tokens).items():
        written[family] = _write_family(family, split, need, **ctx)
    # A held-out split for validation loss and bits per byte: disjoint from
    # train by the sources' own content-hash split, small enough to evaluate
    # often. Tools are exactly graded elsewhere, so only language is held out.
    held_out = {}
    for family, need in plan(preset, tokens).items():
        if family in ("text_lm", "chat"):
            held_out[family] = _write_family(family, "test", min(max(need // 200, 64), 1024),
                                             **ctx)
    (out_dir / "prepared.json").write_text(json.dumps(
        {"preset": preset.name, "split": split, "tokens_planned": tokens or preset.tokens,
         "families": written, "held_out": held_out, "id_offset": TEXT_ID_OFFSET},
        indent=2, default=str))
    return out_dir


def _write_family(family, split, need, *, out_dir, rows, tokenizer, window, text_mix,
                  chat_mix, chat_bytes, seed, base_meta, max_data) -> dict:
    from ..config import TEXT_ID_OFFSET
    from ..data.chat_corpus import CHAT_SOURCES, iter_chat_items
    from ..data.shards import PackingWriter, ShardWriter
    from ..data.text_corpus import SOURCES, iter_text_windows
    from ..data.tokenization import as_tokenizer

    tok = as_tokenizer(tokenizer)
    prefix = out_dir / f"{family}-{split}"
    started = time.time()
    meta = {**base_meta, "split": split}
    measure = split != "train"           # count UTF-8 bytes for bits-per-byte
    text_tokens = n_bytes = 0

    def account(ids):
        nonlocal text_tokens, n_bytes
        if measure:
            t = [int(i) - TEXT_ID_OFFSET for i in ids if i >= TEXT_ID_OFFSET]
            text_tokens += len(t)
            n_bytes += len(tok.decode(t).encode("utf-8"))

    if family == "text_lm":
        meta["sources"] = [SOURCES[k].as_dict() for k in text_mix]
        w = ShardWriter(prefix, family, rows, meta)
        seen: set[int] = set()
        dropped = 0
        for ids, mask in iter_text_windows(text_mix, window, tokenizer, seed=seed,
                                           split=split, limit=need + need // 50 + 8):
            h = hash(ids.tobytes())       # exact duplicate windows (mirrored pages)
            if h in seen:
                dropped += 1
                continue
            seen.add(h)
            w.add_ids(ids, mask)
            account(ids)
            if w.n_items % _CHUNK == 0:
                _progress(family, w.n_items, need, started)
            if w.n_items >= need:
                break
        w.meta["duplicates_dropped"] = dropped
        inner = w
    else:
        if family == "chat":
            meta.update(sources=[CHAT_SOURCES[k].as_dict() for k in chat_mix], max_data=max_data)
        inner = ShardWriter(prefix, family, rows, meta)
        w = PackingWriter(inner, window)

        def stream():
            if family == "chat":
                yield from iter_chat_items(chat_mix, seed=seed, max_bytes=chat_bytes, split=split,
                                           tokenizer=tokenizer, max_tokens=window)
                return
            from ..data.tool_corpus import tool_items
            k = 0
            while True:
                batch = tool_items(_CHUNK, seed=seed + k, split=split, tokenizer=tokenizer,
                                   max_bytes=chat_bytes)
                if not batch:
                    return
                yield from batch
                k += 1

        for it in stream():
            if w.add(it.sample):
                account(np.concatenate([s.payload for s in it.sample.spans]))
            if inner.n_items and inner.n_items % _CHUNK == 0:
                _progress(family, inner.n_items, need, started)
            if inner.n_items >= need:
                break
        if inner.n_items < need:
            print(f"note: {family} sources filled {inner.n_items:,} of {need:,} planned "
                  f"windows; training repeats them ({need / max(inner.n_items, 1):.1f} epochs)",
                  file=sys.stderr)
    if measure:
        (w if family == "text_lm" else inner).meta.update(
            text_tokens=text_tokens, utf8_bytes=n_bytes)
    return w.close()


def _progress(family: str, done: int, need: int, started: float) -> None:
    rate = done / max(time.time() - started, 1e-6)
    print(f"[prepare] {family:<8} {done:>10,}/{need:,} items  ({rate:,.0f}/s)", flush=True)
