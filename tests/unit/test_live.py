"""Continuous perception: deltas, exact incremental context, bounded memory."""

import numpy as np
import torch

from iridium.codecs.bank import TensorBatch, continuous_dims
from iridium.codecs.spans import Sample, Span, collate, text_span
from iridium.config import get_config
from iridium.model.iridium1 import Iridium1
from iridium.runtime.live import FrameDelta, LiveSession


def _model():
    torch.manual_seed(0)
    return Iridium1(get_config("tiny")).double().eval()


def _frame(cfg, seed=0):
    c, p = cfg.codecs.image_channels, cfg.codecs.image_patch
    return np.random.default_rng(seed).random((c, 4 * p, 4 * p))


def test_static_screen_costs_nothing_and_one_change_costs_one_patch():
    cfg = get_config("tiny")
    delta = FrameDelta(cfg.codecs.image_patch)
    f = _frame(cfg)
    assert len(delta(f)[0]) == 16            # first frame: every patch
    assert len(delta(f)[0]) == 0             # nothing changed
    g = f.copy()
    p = cfg.codecs.image_patch
    g[:, :p, p:2 * p] += 0.5                 # one patch (row 0, col 1) changes
    feats, yx = delta(g)
    assert len(feats) == 1 and yx.tolist() == [[0, 1]]


def test_incremental_observation_equals_one_full_forward():
    model = _model()
    cfg = model.cfg
    live = LiveSession(model)
    f0 = _frame(cfg, 0)
    f1 = f0.copy()
    f1[:, :cfg.codecs.image_patch, :] += 0.3   # top row of patches changes
    live.hear("watch the screen")
    live.observe(f0)
    live.observe(f1)

    # The same tokens as one sample: text, then each frame's delta span.
    d = FrameDelta(cfg.codecs.image_patch)
    spans = [text_span("watch the screen", supervised=False, offset=16)]
    for f in (f0, f1):
        feats, yx = d(f)
        coords = np.concatenate([np.zeros((len(yx), 1), np.int64), yx], 1)
        spans.append(Span("image", feats, supervised=False, atomic=False,
                          meta={"coords": coords}))
    batch = TensorBatch(collate([Sample(spans)], continuous_dims(cfg.codecs)),
                        dtype=torch.float64)
    with torch.no_grad():
        full = model(batch, n_loops=1)
    assert torch.allclose(live.hidden[0, -1].double(), full.hidden[0, -1].double(), atol=1e-8)


def test_eviction_bounds_the_cache_and_keeps_running():
    model = _model()
    cfg = model.cfg
    live = LiveSession(model, max_cache_tokens=40)
    live.hear("go")
    rng = np.random.default_rng(1)
    for i in range(12):
        live.observe(rng.random(_frame(cfg).shape))   # every frame all-new: 16 tokens
        assert live._length() <= 40 + 16
    assert live.stats.evicted > 0
    kv = live.cache[("core", 0, 0)][0]
    assert kv.shape[2] == live._length()
    text = live.think(max_new_tokens=3)
    assert isinstance(text, str)


def test_savings_on_a_mostly_static_stream():
    model = _model()
    live = LiveSession(model)
    f = _frame(model.cfg)
    for i in range(10):
        g = f.copy()
        g[:, :model.cfg.codecs.image_patch, :model.cfg.codecs.image_patch] += 0.1 * i  # cursor blink
        live.observe(g)
    assert live.stats.savings > 0.8


def test_keyframes_periodic_on_demand_and_on_scene_change_supersede_old_frames():
    model = _model()
    cfg = model.cfg
    p = cfg.codecs.image_patch
    live = LiveSession(model, keyframe_seconds=10.0)
    live.hear("task")
    f = _frame(cfg)
    assert live.observe(f, now=0.0) == 16                 # first frame: keyframe
    g = f.copy(); g[:, :p, :p] += 0.5
    assert live.observe(g, now=1.0) == 1                  # delta
    assert live.observe(g, now=11.0) == 16                # 10 s elapsed: periodic keyframe
    # The keyframe superseded both earlier frames: only text + this keyframe remain.
    kinds = [e.kind for e in live.ledger]
    assert kinds.count("keyframe") == 1 and "frame" not in kinds
    assert live._length() == live.ledger[0].length + 16
    live.refresh()
    assert live.observe(g, now=12.0) == 16                # asked for a full screenshot
    h = np.random.default_rng(9).random(f.shape)
    assert live.observe(h, now=13.0) == 16                # most of the screen changed
    assert live.stats.keyframes == 4
    assert isinstance(live.think(2), str)
