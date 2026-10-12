"""One place that turns a config into the tokenizer a run must use.

Kept separate from both :mod:`iridium.config` (which may not import torch, and
must stay costable on a laptop) and :mod:`iridium.data.tokenizer` (which knows
nothing about model configs). The join has to live somewhere, and scattering it
across the phase scripts is how a run ends up training against one vocabulary
and serving against another.

That failure is worth naming, because it is the expensive one. A tokenizer
mismatch does not raise. Ids are still in range, the forward pass still runs,
the loss on the wrong vocabulary is merely bad rather than infinite — and the
model emits confident, fluent, entirely wrong text. The only defence is that
exactly one function decides which tokenizer a config implies, and that the
answer is written into the run manifest. Both are here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

__all__ = ["tokenizer_for_config", "tokenizer_manifest", "tokenizer_from_manifest"]


def tokenizer_for_config(cfg, cache_dir: Optional[str | Path] = None, seed: int = 0,
                         n_docs: int = 2000):
    """The tokenizer ``cfg`` implies, or ``None`` for the byte-level path.

    ``None`` rather than a byte-level tokenizer object, because the byte path
    is what every caller in this codebase already does when handed ``None``,
    and returning an equivalent-but-different object would mean two code paths
    that have to stay numerically identical forever. One path is cheaper to
    keep honest than two.

    Training the vocabulary needs the corpus, so this can be slow on first call
    and is cached thereafter; see :func:`iridium.data.tokenizer.tokenizer_for`
    for the cache key. If the corpus cannot be reached, that function falls back
    to byte level rather than failing the run — which is the right call for
    availability and the wrong one to leave silent, so check
    :func:`tokenizer_manifest` on the object you get back before concluding a
    run used the vocabulary it asked for.
    """
    if not getattr(cfg, "text_vocab_size", 0):
        return None
    from ..data.tokenizer import tokenizer_for
    directory = Path(cache_dir or cfg.text_tokenizer_cache)
    return tokenizer_for(cfg.text_vocab_size, directory, seed=seed, n_docs=n_docs)


def tokenizer_manifest(tokenizer, cfg=None) -> dict:
    """What a run manifest should record about the vocabulary it used.

    Includes ``requested`` alongside ``vocab_size`` on purpose. A BPE trainer
    that runs out of distinct merges on a small or repetitive sample settles
    below the size it was asked for, and it does so quietly. The two fields
    disagreeing is the signal that the tokenizer was trained on less text than
    intended — which is recoverable if it is noticed at the start of a run and
    not if it is noticed at the end.
    """
    requested = int(getattr(cfg, "text_vocab_size", 0) or 0) if cfg is not None else 0
    if tokenizer is None or getattr(tokenizer, "kind", "") == "byte":
        return {"kind": "byte", "vocab_size": 256, "requested": requested,
                "fell_back": bool(requested), "state": {"kind": "byte"}}
    size = int(getattr(tokenizer, "vocab_size", 0) or len(getattr(tokenizer, "vocab", ()) or ()))
    manifest = {
        "kind": getattr(tokenizer, "kind", "bpe"),
        "vocab_size": size,
        "requested": requested,
        "fell_back": bool(requested) and size < requested,
    }
    if hasattr(tokenizer, "to_dict"):
        # The merges themselves, not a cache path: a checkpoint must carry the
        # exact vocabulary it was trained with, because retraining the
        # tokenizer later (new data, new library) yields a different one.
        manifest["state"] = tokenizer.to_dict()
    return manifest


def tokenizer_from_manifest(manifest) -> Optional[object]:
    """Rebuild the tokenizer a checkpoint was trained with, or ``None``.

    Accepts a checkpoint manifest (looks under ``"tokenizer"``) or the
    tokenizer manifest itself. ``None`` means byte level: either the run was
    byte level, or it predates tokenizers being stored — and a subword config
    without a stored tokenizer is refused, since guessing is the silent
    mismatch this module exists to prevent.
    """
    if not isinstance(manifest, dict):
        return None
    tok = manifest.get("tokenizer", manifest)
    if isinstance(tok, dict) and isinstance(tok.get("state"), dict):
        from ..data.tokenization import from_state
        return from_state(tok["state"])
    config = manifest.get("model_config") or {}
    if config.get("text_vocab_size"):
        raise ValueError(
            "checkpoint was trained with a subword vocabulary "
            f"({config['text_vocab_size']} ids) but does not store its tokenizer; "
            "re-save it with the tokenizer, or it will be read as bytes")
    return None
