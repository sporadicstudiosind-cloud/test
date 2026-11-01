"""Continuous perception: the model watches a screen (or camera) as it changes.

Computer-use agents today poll: take a screenshot, send the whole image as a
new prompt, wait for a reply, act, repeat. Every poll re-encodes a mostly
identical image, the model is blind between polls, and latency is one full
round trip per look.

Iridium's architecture makes a different loop possible, and this module is it:

1. **Deltas, not screenshots.** :class:`FrameDelta` patchifies each frame
   and emits only the patches that changed since the last frame, each tagged
   with its own ``(t, y, x)`` M-RoPE position. A static screen costs zero
   tokens; a blinking cursor costs one; scrolling costs what scrolled. The
   model still knows *where* each patch is and *when* it changed, because the
   position is in the rotary embedding, not in a re-sent full frame.

2. **One persistent context, read and write interleaved.** :class:`LiveSession`
   appends each delta straight into the KV cache -- no re-prefill -- and the
   model can decode between frames (text, tool calls, actions) from the same
   cache. Perception and action share a timeline, full duplex: a frame that
   arrives while it is "thinking" is simply the next thing it reads.

3. **Bounded memory for an unbounded stream.** Old frame tokens are evicted
   from attention caches once a token budget is exceeded (instructions and the
   model's own words are kept). This is exact for attention -- positions are
   baked into cached keys, so dropping one never changes another -- and the
   Gated DeltaNet layers (``layer_pattern`` with ``"deltanet"``, as in the
   ``modern`` rung) carry a constant-size recurrent summary of everything
   evicted, which is the long-horizon memory a pure-attention model lacks.

What this is not: trained. The mechanics are exact and tested (incremental
observation equals one full forward over the same tokens; eviction bounds
memory); whether a model learns to *use* a delta stream well depends on
training it on one, which the ``world-100m`` and ``omni-100m`` presets would
need posed video / screen recordings for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from ..codecs.bank import TensorBatch, continuous_dims
from ..codecs.spans import MODALITY_INDEX, Sample, Span, collate, text_span

__all__ = ["FrameDelta", "LiveSession"]


class FrameDelta:
    """Changed patches between consecutive frames.

    ``threshold`` is the mean absolute difference, in the frame's own units,
    above which a patch counts as changed. ``keyframe_every`` forces a full
    frame every N frames so accumulated sub-threshold drift is corrected.
    """

    def __init__(self, patch: int, threshold: float = 0.02, keyframe_every: int = 0,
                 scene_change: float = 0.5) -> None:
        self.patch = patch
        self.threshold = threshold
        self.keyframe_every = keyframe_every
        #: Fraction of changed patches above which the whole frame is sent as
        #: a keyframe: past about half, a delta costs nearly a full frame and
        #: leaves a fragmented picture (a new window, a page load).
        self.scene_change = scene_change
        self._last: Optional[np.ndarray] = None
        self.frames = 0
        self.force_next = False
        self.last_was_key = False

    def patches(self, frame: np.ndarray) -> np.ndarray:
        c, h, w = frame.shape
        p = self.patch
        if h % p or w % p:
            raise ValueError(f"frame {h}x{w} is not divisible by patch {p}")
        grid = frame.reshape(c, h // p, p, w // p, p).transpose(1, 3, 0, 2, 4)
        return grid.reshape(h // p, w // p, c * p * p)

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(features [n, C*p*p], yx [n, 2])`` for the patches to send."""
        cur = self.patches(np.asarray(frame, dtype=np.float32))
        key = (self._last is None or self.force_next
               or bool(self.keyframe_every and self.frames % self.keyframe_every == 0))
        if not key:
            changed = np.abs(cur - self._last).mean(axis=-1) > self.threshold
            key = changed.mean() >= self.scene_change
        if key:
            changed = np.ones(cur.shape[:2], dtype=bool)
        self.force_next = False
        self.last_was_key = key
        self._last = cur
        self.frames += 1
        ys, xs = np.nonzero(changed)
        return cur[ys, xs], np.stack([ys, xs], axis=-1)


@dataclass
class _Entry:
    kind: str          # "frame" (delta) | "keyframe" | "text" | "generated"
    start: int         # cache index at insertion
    length: int


@dataclass
class LiveStats:
    frames: int = 0
    tokens_sent: int = 0
    tokens_full_frames: int = 0
    evicted: int = 0
    keyframes: int = 0

    @property
    def savings(self) -> float:
        return 1.0 - self.tokens_sent / max(self.tokens_full_frames, 1)


class LiveSession:
    """A persistent, full-duplex context over a live visual stream.

    ``max_cache_tokens`` bounds attention memory: past it, the oldest frame
    tokens are evicted (never text). ``None`` keeps everything, which is what
    the parity test uses.
    """

    def __init__(self, model, tokenizer=None, threshold: float = 0.02,
                 max_cache_tokens: Optional[int] = None, keyframe_every: int = 0,
                 n_loops: int = 1, keyframe_seconds: Optional[float] = None,
                 scene_change: float = 0.5, supersede: bool = True) -> None:
        self.model = model.eval()
        self.cfg = model.cfg
        self.tokenizer = tokenizer
        self.delta = FrameDelta(self.cfg.codecs.image_patch, threshold, keyframe_every,
                                scene_change)
        #: Wall-clock keyframe interval (the "periodic screenshot"), on top of
        #: the frame-count interval; ``observe(frame, now=...)`` supplies time.
        self.keyframe_seconds = keyframe_seconds
        #: A keyframe restates the whole screen, so every earlier frame token
        #: is redundant for "what is on screen now"; drop them from attention
        #: (DeltaNet state still remembers the history).
        self.supersede = supersede
        self._last_key_time: Optional[float] = None
        self.dims = continuous_dims(self.cfg.codecs)
        self.max_cache_tokens = max_cache_tokens
        self.n_loops = n_loops
        self.cache: dict = {}
        self.position = 0            # next stream position (never reused, even after eviction)
        self.ledger: list[_Entry] = []
        self.stats = LiveStats()
        self.hidden: Optional[torch.Tensor] = None
        self._template: Optional[TensorBatch] = None

    # -- input ----------------------------------------------------------------

    def refresh(self) -> None:
        """Ask for a full screenshot on the next frame (the agent's own call,
        e.g. before an irreversible click, or when it is unsure what it sees)."""
        self.delta.force_next = True

    def observe(self, frame: np.ndarray, now: Optional[float] = None) -> int:
        """Feed one frame; returns how many tokens it cost (0 if nothing changed).

        Frames arrive as deltas, with a full keyframe when: it is the first
        frame, ``refresh()`` was called, ``keyframe_every`` frames or
        ``keyframe_seconds`` have passed, or most of the screen changed.
        """
        if (self.keyframe_seconds is not None and now is not None
                and (self._last_key_time is None
                     or now - self._last_key_time >= self.keyframe_seconds)):
            self.delta.force_next = True
        feats, yx = self.delta(frame)
        is_key = self.delta.last_was_key
        if is_key:
            self.stats.keyframes += 1
            if now is not None:
                self._last_key_time = now
            if self.supersede:
                self._drop(lambda e: e.kind in ("frame", "keyframe"))
        per_frame = int(np.prod(self.delta._last.shape[:2]))
        self.stats.frames += 1
        self.stats.tokens_full_frames += per_frame
        if len(feats) == 0:
            return 0
        # Every patch of one frame shares a time coordinate: (0, y, x) offsets
        # from the span start, so M-RoPE sees "same instant, this location".
        coords = np.concatenate([np.zeros((len(yx), 1), np.int64), yx], axis=1)
        span = Span("image", feats, supervised=False, atomic=False, meta={"coords": coords})
        self._forward(Sample([span]), "keyframe" if is_key else "frame")
        self.stats.tokens_sent += len(feats)
        return len(feats)

    def hear(self, text: str) -> int:
        """Instructions or speech transcript into the same timeline."""
        span = text_span(text, supervised=False, offset=16, tokenizer=self.tokenizer)
        self._forward(Sample([span]), "text")
        return len(span)

    # -- output ---------------------------------------------------------------

    @torch.no_grad()
    def think(self, max_new_tokens: int = 16, temperature: float = 0.0, stop_ids=(2, 7, 9),
              seed: int = 0) -> str:
        """Decode from the live context without re-reading it."""
        from ..data.tokenization import as_tokenizer
        from .device import generator_for
        from .generate import _pick, _single_token_batch

        if self.hidden is None:
            raise RuntimeError("nothing observed yet")
        rng = generator_for(self.hidden.device, seed)
        ids: list[int] = []
        start = self._length()
        for _ in range(max_new_tokens):
            logits = self.model.codecs.text_head(self.hidden[:, -1:])[0, -1]
            token = _pick(logits, temperature, rng, 0.0, 0, 1.0, ids, 0.0)
            ids.append(token)
            if token in stop_ids:
                break
            step = _single_token_batch(self._template, MODALITY_INDEX["text"], token,
                                       self.position)
            out = self.model(step, n_loops=self.n_loops, cache=self.cache)
            self.hidden = out.hidden
            self.position += 1
        self.ledger.append(_Entry("generated", start, self._length() - start))
        text_ids = [t - 16 for t in ids if t >= 16]
        return as_tokenizer(self.tokenizer).decode(text_ids)

    def step(self, frame: np.ndarray, max_new_tokens: int = 0, **kw) -> tuple[int, str]:
        """One tick of the loop: see the frame, optionally say or do something."""
        cost = self.observe(frame)
        said = self.think(max_new_tokens, **kw) if max_new_tokens else ""
        return cost, said

    # -- internals -------------------------------------------------------------

    @torch.no_grad()
    def _forward(self, sample: Sample, kind: str) -> None:
        raw = collate([sample], self.dims)
        raw.positions = raw.positions + self.position
        raw.rope_positions = raw.rope_positions + self.position
        param = next(self.model.parameters())
        batch = TensorBatch(raw, device=param.device, dtype=param.dtype)
        start = self._length()
        out = self.model(batch, n_loops=self.n_loops, cache=self.cache)
        n = int(raw.valid.sum())
        self.position += n
        self.hidden = out.hidden
        self._template = batch
        self.ledger.append(_Entry(kind, start, n))
        self._evict()

    def _length(self) -> int:
        return int(self.cache.get(("stream", "n"), 0))

    def _evict(self) -> None:
        """Budget eviction: oldest deltas first, then old keyframes; never the
        latest keyframe, never text."""
        budget = self.max_cache_tokens
        if budget is None or self._length() <= budget:
            return
        excess = self._length() - budget
        frames = [e for e in self.ledger if e.kind == "frame" and e.length]
        keys = [e for e in self.ledger if e.kind == "keyframe" and e.length][:-1]
        chosen = []
        for e in frames + keys:
            if excess <= 0:
                break
            chosen.append(e)
            excess -= e.length
        ids = {id(e) for e in chosen}
        self._drop(lambda e: id(e) in ids)

    def _drop(self, predicate) -> None:
        total = self._length()
        drop = np.zeros(total, dtype=bool)
        for entry in self.ledger:
            if entry.length and predicate(entry):
                drop[entry.start:entry.start + entry.length] = True
                entry.length = 0
        if not drop.any():
            return
        keep_idx = torch.from_numpy(np.flatnonzero(~drop))
        for key, value in list(self.cache.items()):
            self.cache[key] = _slice_history(value, total, keep_idx)
        kept = int((~drop).sum())
        self.cache[("stream", "n")] = kept
        self.stats.evicted += int(drop.sum())
        new_start = np.cumsum(~drop) - 1
        live = []
        for e in self.ledger:
            if e.length:
                e.start = int(new_start[e.start])
                live.append(e)
        self.ledger = live


def _slice_history(value, total: int, keep_idx: torch.Tensor):
    """Drop evicted positions from any cached tensor indexed by history.

    Attention K/V are ``[B, H, T, D]`` (history on dim 2); bridge states,
    positions and validity masks are ``[B, T, ...]`` (dim 1). Tensors whose
    history dimension is shorter -- sliding-window caches that already forgot,
    recurrent DeltaNet states, running router sums -- are left alone: they
    either never held the evicted tokens or hold only a summary of them.
    """
    if isinstance(value, tuple):
        return tuple(_slice_history(v, total, keep_idx) for v in value)
    if isinstance(value, list):
        return [_slice_history(v, total, keep_idx) for v in value]
    if not torch.is_tensor(value):
        return value
    idx = keep_idx.to(value.device)
    if value.dim() == 4 and value.shape[2] == total:
        return value.index_select(2, idx)
    if value.dim() in (2, 3) and value.shape[1] == total:
        return value.index_select(1, idx)
    return value
