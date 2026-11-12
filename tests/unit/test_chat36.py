"""chat-36m: the rebalanced chat preset and the two embedding options it turns on."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from iridium.codecs.bank import TensorBatch, continuous_dims
from iridium.codecs.spans import Sample, collate, text_span
from iridium.config import IridiumConfig
from iridium.model.iridium1 import Iridium1
from iridium.presets import get_preset
from iridium.runtime.decode import run_chunked


def test_budget_is_compute_optimal_on_the_body():
    preset = get_preset("chat-36m")
    cfg = preset.config
    assert 35e6 < cfg.body_params < 37e6
    assert 5e6 < cfg.embedding_params < 7e6
    assert preset.tokens / cfg.body_params == pytest.approx(20, rel=0.01)
    # No Fourier blocks on a text model, and routing still leaves a stack out.
    assert cfg.stacks.spectral_stacks == () and cfg.router.top_k < cfg.stacks.n_stacks
    # The shared core is no longer the small part.
    assert cfg.core.params > cfg.stacks.params / 2


def test_config_round_trips_and_counts_match_the_module():
    cfg = get_preset("chat-36m").config
    assert IridiumConfig.from_dict(cfg.to_dict()) == cfg
    model = Iridium1(cfg)
    assert sum(p.numel() for p in model.parameters()) == cfg.n_params


def _small(embed_scale: bool):
    """chat-36m's options on a geometry small enough for fp64 parity."""
    cfg = get_preset("chat-36m").config
    small_core = replace(cfg.core, d_model=64, n_layers=3, n_query_heads=4, d_head=16,
                         n_kv_heads=2, d_ff=128)
    small_stacks = replace(cfg.stacks, d_model=64, core_d_model=64, n_layers=2, n_query_heads=4,
                           d_head=16, n_kv_heads=2, d_ff=128, min_depth=1)
    return replace(cfg, core=small_core, stacks=small_stacks, ple_dim=8, max_seq_len=64,
                   codecs=replace(cfg.codecs, embed_scale=embed_scale))


def _text_batch(cfg, n=2):
    rng = np.random.default_rng(0)
    samples = [Sample([text_span("".join(chr(97 + int(c)) for c in rng.integers(0, 26, 11)))])
               for _ in range(n)]
    return TensorBatch(collate(samples, continuous_dims(cfg.codecs)), dtype=torch.float64)


@pytest.mark.usefixtures("exact_attention")
@pytest.mark.parametrize("chunk", [1, 3])
def test_cached_decoding_matches_with_ple_and_scaled_embeddings(chunk):
    cfg = _small(embed_scale=True)
    torch.manual_seed(0)
    model = Iridium1(cfg).double().eval()
    # PLE projections start at zero; move them so the parity check sees them.
    with torch.no_grad():
        for proj in model.ple.projections:
            proj.weight.normal_(std=0.05)
    batch = _text_batch(cfg)
    with torch.no_grad():
        reference = model(batch, n_loops=1).hidden
        cached = run_chunked(model, batch, chunk=chunk, n_loops=1)
    assert float((reference - cached).abs().max()) <= 1e-13


def test_embed_scale_multiplies_text_embeddings_by_sqrt_d():
    plain, scaled = _small(False), _small(True)
    torch.manual_seed(0)
    a = Iridium1(plain).double().eval()
    b = Iridium1(scaled).double().eval()
    b.load_state_dict(a.state_dict())
    batch = _text_batch(plain, n=1)
    ea, eb = a.codecs.embed(batch), b.codecs.embed(batch)
    modality = a.codecs.modality_embedding(batch.modality)
    assert torch.allclose(eb - modality, (ea - modality) * math.sqrt(64))


def test_trains_end_to_end(tmp_path):
    from iridium.training.datasets import build_corpus
    from iridium.training.trainer import TrainConfig, Trainer
    torch.manual_seed(0)
    model = Iridium1(_small(embed_scale=True))
    trainer = Trainer(model, build_corpus(16, seed=0),
                      TrainConfig(steps=3, batch_size=2, log_every=1), out_dir=tmp_path)
    history = trainer.train()
    assert all(math.isfinite(h["total"]) for h in history if "total" in h)
