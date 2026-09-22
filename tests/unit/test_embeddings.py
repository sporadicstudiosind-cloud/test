"""PerLayerEmbedding (Gemma 3n PLE) and HashedNgramEmbedding (Over-Tokenized
Transformer's Over-Encoding): shapes, exact parameter accounting, zero-init
function preservation, strict causality, and exact masking.
"""

from __future__ import annotations

import pytest
import torch

from iridium.model.embeddings import HashedNgramEmbedding, PerLayerEmbedding


# --------------------------------------------------------------------------
# PerLayerEmbedding
# --------------------------------------------------------------------------


def _ple():
    return PerLayerEmbedding(vocab_size=37, n_layers=5, d_ple=6, d_model=16)


def test_ple_param_count_matches_numel_exactly():
    m = _ple()
    formula = PerLayerEmbedding.param_count(vocab_size=37, n_layers=5, d_ple=6, d_model=16)
    actual = sum(p.numel() for p in m.parameters())
    assert formula == actual


def test_ple_output_shape():
    m = _ple()
    ids = torch.randint(0, 37, (3, 9))
    out = m(ids, layer_index=2)
    assert out.shape == (3, 9, 16)


def test_ple_is_function_preserving_at_init():
    """Zero-init projection: every layer's contribution starts at exactly zero."""
    torch.manual_seed(0)
    m = _ple()
    ids = torch.randint(0, 37, (2, 5))
    for layer in range(m.n_layers):
        out = m(ids, layer_index=layer)
        assert torch.equal(out, torch.zeros_like(out))


def test_ple_is_actually_wired_once_trained():
    """Sanity: after the projection moves, output is no longer trivially zero."""
    torch.manual_seed(0)
    m = _ple()
    with torch.no_grad():
        for proj in m.projections:
            torch.nn.init.normal_(proj.weight, std=0.02)
    ids = torch.randint(0, 37, (2, 5))
    out = m(ids, layer_index=0)
    assert not torch.equal(out, torch.zeros_like(out))


def test_ple_layers_read_disjoint_slices():
    """Different layer_index values must not collapse onto the same lookup."""
    torch.manual_seed(0)
    m = _ple()
    with torch.no_grad():
        for proj in m.projections:
            torch.nn.init.normal_(proj.weight, std=0.02)
    ids = torch.randint(0, 37, (2, 5))
    out0 = m(ids, layer_index=0)
    out1 = m(ids, layer_index=1)
    assert not torch.allclose(out0, out1)


# --------------------------------------------------------------------------
# HashedNgramEmbedding
# --------------------------------------------------------------------------


def _ngram():
    return HashedNgramEmbedding(vocab_size=64, d_model=12, table_size=97, n_values=(2, 3))


def test_ngram_param_count_matches_numel_exactly():
    m = _ngram()
    formula = HashedNgramEmbedding.param_count(table_size=97, d_model=12, n_values=(2, 3))
    actual = sum(p.numel() for p in m.parameters())
    assert formula == actual


def test_ngram_output_shape():
    m = _ngram()
    ids = torch.randint(0, 64, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.bool)
    out = m(ids, mask)
    assert out.shape == (2, 10, 12)


def test_ngram_masked_positions_are_exactly_zero():
    torch.manual_seed(0)
    m = _ngram()
    ids = torch.randint(0, 64, (2, 10))
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, ::2] = True  # every other position is "text"
    out = m(ids, mask)
    masked_out = out[~mask]
    assert torch.equal(masked_out, torch.zeros_like(masked_out))
    unmasked_out = out[mask]
    assert not torch.equal(unmasked_out, torch.zeros_like(unmasked_out))


def test_ngram_hashing_is_deterministic():
    torch.manual_seed(0)
    m = _ngram()
    ids = torch.randint(0, 64, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.bool)
    out1 = m(ids, mask)
    out2 = m(ids, mask)
    assert torch.equal(out1, out2)


def test_ngram_is_strictly_causal():
    """Changing a FUTURE token must never change an earlier position's embedding."""
    torch.manual_seed(0)
    m = _ngram()
    ids = torch.randint(0, 64, (1, 10))
    mask = torch.ones(1, 10, dtype=torch.bool)
    out_before = m(ids, mask)

    changed = ids.clone()
    changed[0, 5:] = (changed[0, 5:] + 1) % 64  # perturb positions 5..9 only
    out_after = m(changed, mask)

    assert torch.equal(out_before[:, :5], out_after[:, :5])
    # And the perturbation must actually have moved at least one later position,
    # or this test would pass vacuously.
    assert not torch.equal(out_before[:, 5:], out_after[:, 5:])


def test_ngram_early_positions_use_a_fixed_sentinel_not_future_data():
    """Position 0 (no history at all) must still be a deterministic function."""
    torch.manual_seed(0)
    m = _ngram()
    ids_a = torch.randint(0, 64, (1, 4))
    ids_b = ids_a.clone()
    ids_b[0, 1:] = (ids_b[0, 1:] + 7) % 64  # only positions >=1 differ
    mask = torch.ones(1, 4, dtype=torch.bool)
    out_a = m(ids_a, mask)
    out_b = m(ids_b, mask)
    assert torch.equal(out_a[:, :1], out_b[:, :1])


def test_ngram_missing_history_does_not_collide_with_token_zero():
    """Real id 0 and 'no token here' must hash differently.

    The earlier encoding zeroed missing positions and then added 1, so both
    became 1. Position 1 of [0, 5] has a real predecessor (id 0); position 0 of
    [5] has none. Their 2-gram hashes must differ.
    """
    torch.manual_seed(0)
    emb = HashedNgramEmbedding(vocab_size=16, d_model=8, table_size=997, n_values=(2,))
    with_zero = emb._ngram_hash(emb.encode(torch.tensor([[0, 5]]), torch.ones(1, 2, dtype=torch.bool)), 2)
    alone = emb._ngram_hash(emb.encode(torch.tensor([[5]]), torch.ones(1, 1, dtype=torch.bool)), 2)
    assert int(with_zero[0, 1]) != int(alone[0, 0])


def test_ngram_does_not_read_non_text_slots_as_words():
    """A text token after an image patch hashes as if its predecessor were missing."""
    emb = HashedNgramEmbedding(vocab_size=64, d_model=8, table_size=997, n_values=(2,))
    ids_a = torch.tensor([[7, 9]])
    ids_b = torch.tensor([[33, 9]])
    mask = torch.tensor([[False, True]])            # position 0 is not text
    h_a = emb._ngram_hash(emb.encode(ids_a, mask), 2)
    h_b = emb._ngram_hash(emb.encode(ids_b, mask), 2)
    assert int(h_a[0, 1]) == int(h_b[0, 1])


@pytest.mark.parametrize("chunk", [1, 2, 3, 5])
def test_ngram_streamed_in_chunks_equals_one_shot(chunk):
    """Incremental decoding embeds one token at a time; the result must not change.

    Carrying the last max(n)-1 encoded ids between chunks is what makes this
    hold -- without it every chunk boundary reads 'missing' history and cached
    decoding computes a different function from teacher forcing.
    """
    torch.manual_seed(0)
    emb = HashedNgramEmbedding(vocab_size=50, d_model=8, table_size=1009, n_values=(2, 3))
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 50, (2, 11), generator=g)
    mask = torch.rand(2, 11, generator=g) > 0.2
    full = emb(ids, mask)
    history = None
    pieces = []
    for start in range(0, 11, chunk):
        out, history = emb.forward_with_history(ids[:, start:start + chunk],
                                                mask[:, start:start + chunk], history)
        pieces.append(out)
    assert torch.equal(torch.cat(pieces, dim=1), full)
