"""Two evidenced ways to add "more embedding layers", neither of them FLOPs.

Both bullets below trade *lookup-table* parameters for capacity, which is a
different resource from the matrices the rest of this file's siblings add:
an embedding table is read, never multiplied against, so its cost is memory
and (for the hashed table) a hash, not a matmul. That is what makes both of
these cheap enough to add "more" of without also adding compute.

* :class:`PerLayerEmbedding` -- Gemma 3n's Per-Layer Embeddings. Gemma 3n
  gives every transformer layer its own small per-token lookup, projected
  into that layer's residual stream. The payoff Google reports is not
  quality per se but where the parameters *live*: a PLE table needs no
  matmul to produce its layer's contribution beyond one small projection, so
  it can sit in host RAM and be streamed in per layer, keeping a model with
  many nominal parameters at the accelerator-memory footprint of a much
  smaller one (reported: a 5B-parameter E2B variant fits in about the VRAM a
  2B model would need). This module implements the lookup + per-layer
  projection; the offload-to-host-memory placement decision belongs to
  ``iridium/runtime/placement.py``, not here.
* :class:`HashedNgramEmbedding` -- the Over-Tokenized Transformer's
  Over-Encoding (ByteDance Seed, Jan 2025, arXiv:2501.16975). The paper's
  finding is that loss falls log-linearly as the *input*-side vocabulary
  grows, largely independent of the output vocabulary, and gets there by
  hashing short token n-grams into a large embedding table (up to ~1-12M
  rows) added to the ordinary token embedding, at negligible extra FLOPs
  because the table is looked up, not multiplied. This module implements the
  hashed-n-gram half (Over-Encoding); the paper's other half, Over-Decoding
  (a larger output vocabulary), is a head-sizing change in ``codecs/bank.py``
  and out of scope here.

Both must be safe to add to a codec bank that also carries non-text
modalities: an n-gram computed from a field patch's or an image patch's
discrete id is not a real n-gram (patches are not drawn from a shared
sequential vocabulary the way subword ids are), so :class:`HashedNgramEmbedding`
takes an explicit boolean mask and is exactly zero everywhere that mask is
False.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class PerLayerEmbedding(nn.Module):
    """One small per-token embedding per layer, projected into the residual.

    ``forward(token_ids, layer_index)`` returns ``[B, T, d_model]``, meant to
    be added into that layer's input (or output) alongside the ordinary
    token/modality embedding the codec bank already provides. The projection
    is zero-initialised, so a model that already exists and is not using PLE
    yet can have it switched on -- ``self.ple = PerLayerEmbedding(...)`` added
    to an existing checkpoint -- and produce bit-identical outputs at that
    instant, changing behaviour only once the projection weights move under
    gradient descent. That is the same zero-init contract
    ``DynamicLowRankLinear`` uses for the same reason: a module that is a
    strict *addition* of capacity should not also be free to change what the
    model already computes the moment it is wired in.

    The lookup table is a single ``nn.Embedding`` of width ``n_layers *
    d_ple`` rather than ``n_layers`` separate embeddings of width ``d_ple``,
    because one wide table is one gather instead of ``n_layers`` of them and
    the per-layer split is a free ``view`` on the result -- the parameter
    count is identical either way.
    """

    def __init__(self, vocab_size: int, n_layers: int, d_ple: int, d_model: int) -> None:
        super().__init__()
        if d_ple < 1 or n_layers < 1:
            raise ValueError("n_layers and d_ple must be >= 1")
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.d_ple = d_ple
        self.d_model = d_model
        self.table = nn.Embedding(vocab_size, n_layers * d_ple)
        nn.init.normal_(self.table.weight, std=0.02)
        self.projections = nn.ModuleList(
            nn.Linear(d_ple, d_model, bias=False) for _ in range(n_layers)
        )
        for proj in self.projections:
            nn.init.zeros_(proj.weight)

    @staticmethod
    def param_count(vocab_size: int, n_layers: int, d_ple: int, d_model: int) -> int:
        return vocab_size * n_layers * d_ple + n_layers * d_ple * d_model

    def forward(self, token_ids: torch.Tensor, layer_index: int) -> torch.Tensor:
        if not 0 <= layer_index < self.n_layers:
            raise ValueError(f"layer_index {layer_index} out of range [0, {self.n_layers})")
        whole = self.table(token_ids)  # [B, T, n_layers * d_ple]
        lo = layer_index * self.d_ple
        per_layer = whole[..., lo : lo + self.d_ple]
        return self.projections[layer_index](per_layer)


#: Large odd multipliers for the polynomial rolling hash below. Odd so each is
#: coprime with any power-of-two table size; distinct per n-gram order so a
#: 2-gram and a 3-gram sharing a suffix do not collide by construction (they
#: can still collide by table-size reduction, same as any hash table).
_HASH_PRIMES: dict[int, int] = {2: 1_000_003, 3: 999_983, 4: 999_961}


class HashedNgramEmbedding(nn.Module):
    """Adds an embedding of the hashed preceding n-gram, causally, text-only.

    For each configured order ``n`` in ``n_values`` (default ``(2, 3)``,
    matching the paper's own default), position ``i`` looks up a hash of
    token ids ``(i - n + 1, ..., i)`` -- i.e. the n-gram *ending at* the
    current position, which the causal LM at position ``i`` is entitled to
    know before it emits position ``i + 1``. Positions before ``n - 1`` use a
    fixed sentinel id (0) for the missing history rather than reading out of
    bounds; that sentinel is itself a deterministic function of position and
    never of anything at or after ``i``, so it does not weaken causality.

    The hash is a plain polynomial rolling hash mod ``table_size`` -- no
    input-dependent state, so identical inputs always land on identical rows
    (``tests/unit/test_embeddings.py::test_hashing_is_deterministic``) and
    collisions are exactly what "hashed into a table of size M" in the paper
    means: multiple n-grams sharing one row, which is the compression that
    makes an M-row table representable at all for M far smaller than
    ``vocab_size ** n``.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        table_size: int,
        n_values: Sequence[int] = (2, 3),
    ) -> None:
        super().__init__()
        if not n_values or any(n < 2 for n in n_values):
            raise ValueError("n_values must be non-empty orders >= 2")
        unknown = set(n_values) - set(_HASH_PRIMES)
        if unknown:
            raise ValueError(f"no hash multiplier configured for n_values {sorted(unknown)}")
        self.vocab_size = vocab_size
        self.table_size = table_size
        self.n_values = tuple(n_values)
        self.tables = nn.ModuleDict(
            {str(n): nn.Embedding(table_size, d_model) for n in self.n_values}
        )
        for table in self.tables.values():
            nn.init.normal_(table.weight, std=0.02)

    @staticmethod
    def param_count(table_size: int, d_model: int, n_values: Sequence[int] = (2, 3)) -> int:
        return len(n_values) * table_size * d_model

    def _ngram_hash(self, token_ids: torch.Tensor, n: int) -> torch.Tensor:
        """``[B, T]`` long hash of the n-gram ending at each position, mod table_size.

        Built with plain shifts (``torch.roll`` + masking), not a Python loop
        over ``T``: this runs once per forward, over the whole sequence, and a
        per-position loop would be the slowest part of an otherwise O(1)-per-
        token embedding lookup.
        """
        b, t = token_ids.shape
        multiplier = _HASH_PRIMES[n]
        acc = torch.zeros_like(token_ids)
        for offset in range(n):
            # offset 0 is the current token, offset (n-1) is the oldest one in
            # the window; shifting right by `offset` and zeroing the wrapped
            # tail is how position i reads position i - offset without ever
            # reading a real token from i + 1 or later.
            shifted = torch.roll(token_ids, shifts=offset, dims=1)
            idx = torch.arange(t, device=token_ids.device)
            valid = idx >= offset
            shifted = torch.where(valid.unsqueeze(0), shifted, torch.zeros_like(shifted))
            acc = acc * multiplier + (shifted + 1)  # +1 so id 0 and "missing" differ
        return acc.remainder(self.table_size)

    def forward(self, token_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        """``token_ids``/``text_mask`` are ``[B, T]``; returns ``[B, T, d_model]``.

        Exactly zero wherever ``text_mask`` is False -- a field/image/action
        position contributes nothing, by construction (multiplied by the mask
        before the sum, not merely "usually small").
        """
        total = None
        for n in self.n_values:
            h = self._ngram_hash(token_ids, n)
            contribution = self.tables[str(n)](h)
            total = contribution if total is None else total + contribution
        return total * text_mask.to(total.dtype).unsqueeze(-1)


__all__ = ["PerLayerEmbedding", "HashedNgramEmbedding"]
