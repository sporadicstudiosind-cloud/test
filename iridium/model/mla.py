"""Multi-head Latent Attention (DeepSeek-V2/V3).

The request "more heads for attention" has an obvious wrong answer and this
module exists because of it. Grouped-query attention (``layers.py``) keeps
``n_kv_heads`` full-width key/value heads and grows only the query side, so
adding query heads at a fixed ``d_model`` just narrows ``d_head`` (heads below
roughly 32-64 dims stop having enough room to represent a distinct relation),
and adding *kv* heads to keep per-head width up explodes the cache linearly in
head count. Neither knob lets "many heads" pay for itself.

MLA (Multi-head Latent Attention, DeepSeek-V2 -- arXiv:2405.04434, kept in
DeepSeek-V3) sidesteps both failure modes by changing what the cache holds.
Keys and values for every head are produced by up-projecting one shared,
low-rank latent ``c_kv`` (``kv_lora_rank`` wide, e.g. 512) that is *itself*
what gets cached -- not the per-head K/V. So caching cost is
``O(kv_lora_rank)`` per token, independent of ``n_heads``: doubling the head
count doubles compute and representational capacity without moving the cache
size at all. Queries go through an analogous, optional low-rank bottleneck
(``q_lora_rank``, 0 disables it) purely to cut *parameter* count, since the
query side is never cached and so never touches the memory argument.

Decoupled RoPE, and why it exists
----------------------------------
RoPE cannot be applied to the shared content latent, because doing so would
destroy the one property MLA exists to provide: weight absorption. The
content part of the attention score factors as

    q_content_i . k_content_i,j = c_q^T (W_UQ_i^T W_UK_i) c_kv_j

which is a *fixed* bilinear form in the two latents -- ``W_UQ_i^T W_UK_i`` is
a constant matrix computable once, so the score can be read directly off the
cached latents (see ``_absorbed_forward``) without ever materialising a
per-head key. Insert a position-dependent rotation ``R(t)`` between the two
up-projections -- i.e. rotate the shared latent instead of a per-head vector
-- and the "constant" matrix becomes ``W_UQ_i^T R(t_q - t_k) W_UK_i``: a
different matrix for every relative offset. The absorption trick, and with it
the entire reason to cache a compressed latent instead of full K/V, is gone.

DeepSeek's fix is to keep RoPE positional information *out* of the shared
latent entirely and carry it in a small side channel instead: a decoupled key
head ``k_rope`` of width ``rope_head_dim`` (e.g. 64), computed straight from
the residual stream, shared across all query heads, and rotated by ordinary
RoPE; and, on the query side, one decoupled RoPE vector per head, produced
from the (possibly compressed) query latent. Each head's full key/query is
the concatenation of its content part and the (shared or per-head) RoPE part,
and the score is content-score + rope-score. Only ``c_kv`` and the rotated
``k_rope`` need to be cached; both are position-free with respect to the
absorption trick because the rotation lives entirely in the small side
channel, not in ``c_kv``.

Two inference paths, same numbers
----------------------------------
``forward`` (the default) decompresses the cached latents back to full
per-head K/V every call and runs ordinary attention through
``layers._attend`` -- simple, and what every non-absorbed reference
implementation does. ``forward(..., absorbed=True)`` instead runs the
algebra above: it folds ``W_UK`` into the query (producing a
``kv_lora_rank``-wide "absorbed query" per head) and ``W_UV`` into the output
projection, so the attention matmuls run directly against the cached latents
and no ``[*, n_heads, T, d_head]`` key or value tensor is ever built. This is
the path DeepSeek actually serves with -- it is what turns "cache is small"
into "decoding is also fast", since materialising full K/V from the latent on
every decode step is itself ``O(n_heads * d_head * kv_lora_rank)`` work that
absorption skips. ``tests/unit/test_mla.py`` pins the two paths to agree at
float64 precision, and the *sacred* incremental-vs-full-forward invariant
(``forward``, non-absorbed) is pinned the same way ``GroupedQueryAttention``
is pinned in ``tests/integration/test_kv_parity.py``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .layers import RMSNorm, _attend, _masked_softmax, neg_inf, sliding_window_keep
from .rope import RotaryEmbedding


class MultiHeadLatentAttention(nn.Module):
    """Drop-in replacement for ``GroupedQueryAttention``: same forward, same cache protocol.

    ``forward(x, positions, keep, cache, cache_key)`` matches
    ``GroupedQueryAttention.forward`` exactly, so ``TransformerBlock`` can hold
    either as ``self.attn`` unmodified. Unlike GQA, there is no
    ``n_kv_heads``: every head reads the same cached latent, which is the
    entire point.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        kv_lora_rank: int,
        q_lora_rank: int,
        rope_head_dim: int,
        rope: Optional[RotaryEmbedding] = None,
        *,
        window: Optional[int] = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if kv_lora_rank < 1:
            raise ValueError("kv_lora_rank must be >= 1")
        if q_lora_rank < 0:
            raise ValueError("q_lora_rank must be >= 0 (0 disables q compression)")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rope_head_dim = rope_head_dim
        self.rope = rope if rope is not None else RotaryEmbedding(rope_head_dim)
        self.window = window
        self._scale = 1.0 / math.sqrt(d_head + rope_head_dim)

        # -- KV side: always compressed. ----------------------------------
        self.w_dkv = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank, eps)
        self.w_uk = nn.Linear(kv_lora_rank, n_heads * d_head, bias=False)
        self.w_uv = nn.Linear(kv_lora_rank, n_heads * d_head, bias=False)
        # Decoupled key: ONE head, shared across all n_heads query heads.
        self.w_kr = nn.Linear(d_model, rope_head_dim, bias=False)

        # -- Q side: compressed iff q_lora_rank > 0. ----------------------
        if q_lora_rank > 0:
            self.w_dq = nn.Linear(d_model, q_lora_rank, bias=False)
            self.q_norm = RMSNorm(q_lora_rank, eps)
            self.w_uq = nn.Linear(q_lora_rank, n_heads * d_head, bias=False)
            self.w_qr = nn.Linear(q_lora_rank, n_heads * rope_head_dim, bias=False)
            self._q_in_dim = q_lora_rank
        else:
            self.w_dq = None
            self.q_norm = None
            self.w_q = nn.Linear(d_model, n_heads * d_head, bias=False)
            self.w_qr = nn.Linear(d_model, n_heads * rope_head_dim, bias=False)
            self._q_in_dim = d_model

        self.wo = nn.Linear(n_heads * d_head, d_model, bias=False)

    # -- parameter accounting ---------------------------------------------

    @staticmethod
    def param_count(
        d_model: int,
        n_heads: int,
        d_head: int,
        kv_lora_rank: int,
        q_lora_rank: int,
        rope_head_dim: int,
    ) -> int:
        """Exact parameter count, matching ``__init__`` module-for-module.

        Every linear here is bias-free (repo convention). ``q_lora_rank == 0``
        replaces the ``W_DQ`` + ``q_norm`` + ``W_UQ`` triple with a single
        plain ``W_Q``, and the decoupled query-RoPE projection then reads
        ``d_model`` directly instead of the compressed latent.
        """
        d, h, dh, kvr, qr, rr = d_model, n_heads, d_head, kv_lora_rank, q_lora_rank, rope_head_dim
        if qr > 0:
            q_part = d * qr + qr + qr * (h * dh) + qr * (h * rr)
        else:
            q_part = d * (h * dh) + d * (h * rr)
        kv_part = d * kvr + kvr + kvr * (h * dh) + kvr * (h * dh)
        rope_key = d * rr
        out = (h * dh) * d
        return q_part + kv_part + rope_key + out

    @staticmethod
    def kv_cache_bytes_per_token(
        kv_lora_rank: int, rope_head_dim: int, bytes_per_element: int = 2
    ) -> int:
        """Cache footprint for ONE layer, one token: the latent plus the shared rope key.

        Independent of ``n_heads`` and of ``d_head`` -- the entire structural
        claim of MLA. Contrast with GQA's ``2 * n_kv_heads * d_head *
        bytes_per_element``, which scales with the kv head count directly.
        """
        return (kv_lora_rank + rope_head_dim) * bytes_per_element

    # -- forward ------------------------------------------------------------

    def _project_new(self, x: torch.Tensor, positions: torch.Tensor):
        b, t, _ = x.shape
        c_kv = self.kv_norm(self.w_dkv(x))                      # [B, T, kvr]
        k_rope = self.w_kr(x).view(b, t, 1, self.rope_head_dim).transpose(1, 2)
        k_rope = self.rope(k_rope, positions)                    # [B, 1, T, rr]
        return c_kv, k_rope

    def _project_q(self, x: torch.Tensor, positions: torch.Tensor):
        b, t, _ = x.shape
        if self.q_lora_rank > 0:
            c_q = self.q_norm(self.w_dq(x))
            q_content = self.w_uq(c_q).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
            q_rope = self.w_qr(c_q).view(b, t, self.n_heads, self.rope_head_dim).transpose(1, 2)
            q_in = c_q
        else:
            q_content = self.w_q(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
            q_rope = self.w_qr(x).view(b, t, self.n_heads, self.rope_head_dim).transpose(1, 2)
            q_in = x
        q_rope = self.rope(q_rope, positions)
        return q_content, q_rope, q_in

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        keep: torch.Tensor,
        cache: Optional[dict] = None,
        cache_key: Optional[tuple] = None,
        *,
        absorbed: bool = False,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        c_kv_new, k_rope_new = self._project_new(x, positions)

        if cache is not None and cache_key is not None:
            prev = cache.get(cache_key)
            if prev is not None:
                c_kv_new = torch.cat([prev[0], c_kv_new], dim=1)
                k_rope_new = torch.cat([prev[1], k_rope_new], dim=2)
            cache[cache_key] = (c_kv_new, k_rope_new)

        c_kv_full, k_rope_full = c_kv_new, k_rope_new  # [B, Tk, kvr], [B, 1, Tk, rr]
        q_content, q_rope, q_in = self._project_q(x, positions)

        if self.window is not None:
            tk = c_kv_full.shape[1]
            keep = keep & sliding_window_keep(t, tk, self.window, x.device)

        if absorbed:
            out = self._absorbed_attend(q_in, q_rope, c_kv_full, k_rope_full, keep)
        else:
            out = self._materialized_attend(q_content, q_rope, c_kv_full, k_rope_full, keep)
        return self.wo(out.transpose(1, 2).reshape(b, t, -1))

    def _materialized_attend(self, q_content, q_rope, c_kv_full, k_rope_full, keep):
        b, tk, _ = c_kv_full.shape
        k_content = self.w_uk(c_kv_full).view(b, tk, self.n_heads, self.d_head).transpose(1, 2)
        v_content = self.w_uv(c_kv_full).view(b, tk, self.n_heads, self.d_head).transpose(1, 2)
        k_rope_b = k_rope_full.expand(-1, self.n_heads, -1, -1)
        q_full = torch.cat([q_content, q_rope], dim=-1)
        k_full = torch.cat([k_content, k_rope_b], dim=-1)
        return _attend(q_full, k_full, v_content, keep, self.d_head + self.rope_head_dim)

    def _absorb_matrices(self):
        """``M_i = W_in_i^T W_UK_i`` and ``W_UV_i`` reshaped per head.

        ``W_UK``/``W_UV`` weight tensors are ``[n_heads * d_head, kv_lora_rank]``
        (torch's ``nn.Linear`` convention: ``out_features x in_features``);
        reshaping the leading axis splits per head without touching memory.
        ``W_in`` is ``W_UQ`` when queries are compressed, ``W_Q`` otherwise;
        both are ``[n_heads * d_head, q_in_dim]`` for the same reason.
        """
        h, dh, kvr = self.n_heads, self.d_head, self.kv_lora_rank
        w_uk = self.w_uk.weight.view(h, dh, kvr)          # [h, dh, kvr]
        w_uv = self.w_uv.weight.view(h, dh, kvr)          # [h, dh, kvr]
        w_in = (self.w_uq if self.q_lora_rank > 0 else self.w_q).weight
        w_in = w_in.view(h, dh, self._q_in_dim)            # [h, dh, q_in_dim]
        # M_i = W_in_i^T @ W_UK_i : [q_in_dim, kvr]
        m = torch.einsum("hdi,hdk->hik", w_in, w_uk)
        # W_UV_i as [kvr, dh] so a latent-space vector maps straight to d_head.
        w_uv_t = w_uv.transpose(1, 2)                      # [h, kvr, dh]
        return m, w_uv_t

    def _absorbed_attend(self, q_in, q_rope, c_kv_full, k_rope_full, keep):
        """Score and mix entirely in latent space; no per-head K/V is built.

        ``q_in`` is ``c_q`` (compressed) or ``x`` (uncompressed) -- whichever
        fed ``W_UQ``/``W_Q`` in the materialized path -- so ``q' = q_in @ M``
        reproduces ``q_content . k_content`` exactly (see module docstring).
        """
        m, w_uv_t = self._absorb_matrices()
        q_prime = torch.einsum("bti,hik->bhtk", q_in, m)          # [B,h,Tq,kvr]
        score_content = torch.einsum("bhtk,bsk->bhts", q_prime, c_kv_full)
        score_rope = torch.einsum("bhtd,bosd->bhts", q_rope, k_rope_full)
        scores = (score_content + score_rope) * self._scale
        probs = _masked_softmax(scores, keep)
        out_latent = torch.einsum("bhts,bsk->bhtk", probs, c_kv_full)  # [B,h,Tq,kvr]
        return torch.einsum("bhtk,hkd->bhtd", out_latent, w_uv_t)      # [B,h,Tq,dh]


__all__ = ["MultiHeadLatentAttention"]
