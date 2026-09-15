"""Transformer primitives shared by the control core and the superstacks.

Attention is written out rather than delegated to
``F.scaled_dot_product_attention`` for three reasons that matter here:

1. the cache-parity gate (docs/architecture.md) compares incremental decoding
   against an uncached reference in float64, and the fused kernels do not offer
   a stable float64 path on every backend;
2. superstack attention needs an additive mask built from *original stream
   positions*, not from packed indices;
3. a fully-masked query row must be defined, not NaN. Padded slots in a packed
   superstack batch are exactly that case, and a NaN there would propagate into
   the core residual and poison every other token in the batch.

Point 3 is the one that bites. ``_masked_softmax`` handles it explicitly.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding


def neg_inf(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        norm = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * norm).to(dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


#: How attention is computed. ``"sdpa"`` calls PyTorch's fused kernels, which
#: never materialise the ``[B, H, Tq, Tk]`` score matrix; ``"manual"`` builds it
#: explicitly. Override with ``IRIDIUM_ATTENTION=manual`` or
#: :func:`set_attention_backend`.
#:
#: The score matrix is what runs a GPU out of memory. At batch 32, 8 heads and
#: a 2048-token context it is 4.3 GB in fp32 **per layer**, and autograd holds
#: every one of them until the backward pass. Nothing else in this model comes
#: close. The fused kernels compute the same result in tiles and keep only what
#: the backward pass genuinely needs.
#:
#: The cost is exactness: a fused kernel's reduction order depends on sequence
#: length, so a cached single-token step and a full forward pass no longer agree
#: to the last bit — they agree to floating-point tolerance. ``"manual"`` is
#: what the bit-exact parity gate runs under, and is why it is still here.
_ATTENTION_BACKEND = os.environ.get("IRIDIUM_ATTENTION", "sdpa").lower()


def set_attention_backend(backend: str) -> str:
    """Set ``"sdpa"`` or ``"manual"``; returns the previous value."""
    global _ATTENTION_BACKEND
    if backend not in ("sdpa", "manual"):
        raise ValueError(f"backend must be 'sdpa' or 'manual', got {backend!r}")
    previous, _ATTENTION_BACKEND = _ATTENTION_BACKEND, backend
    return previous


def attention_backend() -> str:
    return _ATTENTION_BACKEND


class use_attention_backend:
    """Context manager form, so a test can pin exactness locally."""

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.previous = ""

    def __enter__(self) -> str:
        self.previous = set_attention_backend(self.backend)
        return self.backend

    def __exit__(self, *exc) -> None:
        set_attention_backend(self.previous)


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            keep: torch.Tensor, d_head: int) -> torch.Tensor:
    """Attention with an explicit boolean mask, both ways.

    Both paths agree on the awkward case: a query row with *no* admissible key
    returns zeros rather than NaN. Softmax over an all-``-inf`` row is NaN, and
    a fused kernel propagates that into the residual stream, where it silently
    turns the whole batch into NaN several layers later.
    """
    if _ATTENTION_BACKEND == "sdpa":
        mask = keep.expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])
        dead = ~mask.any(dim=-1, keepdim=True)
        # Give a dead row one admissible key so the kernel produces a finite
        # number, then zero the row afterwards. Masking it back out is what
        # keeps this identical to the manual path rather than merely close.
        safe = mask.masked_fill(dead.expand_as(mask) & _first_column(mask), True)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=safe)
        return out.masked_fill(dead, 0.0)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)
    probs = _masked_softmax(scores, keep)
    return probs @ v


def _first_column(mask: torch.Tensor) -> torch.Tensor:
    """A mask selecting only key position 0, broadcast over ``mask``'s shape."""
    column = torch.zeros_like(mask)
    column[..., :1] = True
    return column


def _masked_softmax(scores: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Softmax over ``scores`` where ``keep`` (bool, broadcastable) is True.

    A query row with no admissible key returns all zeros rather than NaN. The
    caller is responsible for making sure such rows are padding.
    """
    filled = scores.masked_fill(~keep, neg_inf(scores.dtype))
    probs = torch.softmax(filled.float(), dim=-1).to(scores.dtype)
    dead = ~keep.any(dim=-1, keepdim=True)
    return probs.masked_fill(dead, 0.0)


class GroupedQueryAttention(nn.Module):
    """Self-attention with fewer KV heads than query heads.

    ``forward`` takes an explicit boolean ``keep`` mask of shape broadcastable
    to ``[B, H, Tq, Tk]``. Nothing here builds a causal mask on its own: the
    caller knows whether the sequence is the stream, a routed subsequence, or a
    noised output block, and those need different masks.
    """

    def __init__(
        self,
        d_model: int,
        n_query_heads: int,
        n_kv_heads: int,
        d_head: int,
        rope: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        if n_query_heads % n_kv_heads:
            raise ValueError("query heads must be a multiple of kv heads")
        self.n_q = n_query_heads
        self.n_kv = n_kv_heads
        self.d_head = d_head
        self.repeat = n_query_heads // n_kv_heads
        d_kv = n_kv_heads * d_head
        self.wq = nn.Linear(d_model, n_query_heads * d_head, bias=False)
        self.wk = nn.Linear(d_model, d_kv, bias=False)
        self.wv = nn.Linear(d_model, d_kv, bias=False)
        self.wo = nn.Linear(n_query_heads * d_head, d_model, bias=False)
        self.rope = rope

    def project_kv(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        k = self.wk(x).view(b, t, self.n_kv, self.d_head).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv, self.d_head).transpose(1, 2)
        if self.rope is not None:
            k = self.rope(k, positions)
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        keep: torch.Tensor,
        cache: Optional[dict] = None,
        cache_key: Optional[tuple] = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_q, self.d_head).transpose(1, 2)
        if self.rope is not None:
            q = self.rope(q, positions)
        k, v = self.project_kv(x, positions)

        if cache is not None and cache_key is not None:
            prev = cache.get(cache_key)
            if prev is not None:
                k = torch.cat([prev[0], k], dim=2)
                v = torch.cat([prev[1], v], dim=2)
            cache[cache_key] = (k, v)

        kr = k.repeat_interleave(self.repeat, dim=1)
        vr = v.repeat_interleave(self.repeat, dim=1)
        out = _attend(q, kr, vr, keep, self.d_head)
        return self.wo(out.transpose(1, 2).reshape(b, t, -1))


class BridgeCrossAttention(nn.Module):
    """Superstack queries attending to the control core's global KV.

    This is the mechanism that resolves what the source plan called the
    "KV-cache asynchrony paradox". The superstack never owns the stream's
    history; the core does. A stack layer reads it through ``W_K_bridge`` and
    ``W_V_bridge``, which project the *core* hidden width into this stack's
    head layout.

    The K/V projections are applied to the core states supplied by the caller.
    In deployment those come from the core's residual stream at the dispatch
    layer, cached once per stream and broadcast; recomputing them per stack
    layer is the memory/compute trade the planner costs out.
    """

    def __init__(
        self,
        d_model: int,
        d_core: int,
        n_query_heads: int,
        n_kv_heads: int,
        d_head: int,
        rope: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        self.n_q = n_query_heads
        self.n_kv = n_kv_heads
        self.d_head = d_head
        self.repeat = n_query_heads // n_kv_heads
        d_kv = n_kv_heads * d_head
        self.wq = nn.Linear(d_model, n_query_heads * d_head, bias=False)
        self.wk = nn.Linear(d_core, d_kv, bias=False)
        self.wv = nn.Linear(d_core, d_kv, bias=False)
        self.wo = nn.Linear(n_query_heads * d_head, d_model, bias=False)
        self.rope = rope

    def forward(
        self,
        x: torch.Tensor,
        q_positions: torch.Tensor,
        core_states: torch.Tensor,
        core_positions: torch.Tensor,
        keep: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        _, s, _ = core_states.shape
        q = self.wq(x).view(b, t, self.n_q, self.d_head).transpose(1, 2)
        k = self.wk(core_states).view(b, s, self.n_kv, self.d_head).transpose(1, 2)
        v = self.wv(core_states).view(b, s, self.n_kv, self.d_head).transpose(1, 2)
        if self.rope is not None:
            q = self.rope(q, q_positions)
            k = self.rope(k, core_positions)
        kr = k.repeat_interleave(self.repeat, dim=1)
        vr = v.repeat_interleave(self.repeat, dim=1)
        out = _attend(q, kr, vr, keep, self.d_head)
        return self.wo(out.transpose(1, 2).reshape(b, t, -1))


def causal_keep(t_q: int, t_k: int, device=None) -> torch.Tensor:
    """``[1, 1, t_q, t_k]`` boolean: query ``i`` (offset by history) sees key ``j``."""
    offset = t_k - t_q
    qi = torch.arange(t_q, device=device).view(-1, 1) + offset
    kj = torch.arange(t_k, device=device).view(1, -1)
    return (kj <= qi).view(1, 1, t_q, t_k)


def position_keep(
    q_positions: torch.Tensor, k_positions: torch.Tensor, strict: bool = False
) -> torch.Tensor:
    """Causal mask from *stream positions* rather than tensor indices.

    ``q_positions`` is ``[B, Tq]``, ``k_positions`` is ``[B, Tk]``. Returns
    ``[B, 1, Tq, Tk]``. With ``strict`` the query may not see its own position,
    which is what a cross-attention onto a *pre-dispatch* core state wants when
    the core state for that position was produced by the same token.
    """
    qp = q_positions.unsqueeze(-1)
    kp = k_positions.unsqueeze(-2)
    keep = (kp < qp) if strict else (kp <= qp)
    return keep.unsqueeze(1)


class TransformerBlock(nn.Module):
    """Pre-norm self-attention + SwiGLU. The unit both stacks are built from."""

    def __init__(
        self,
        d_model: int,
        n_query_heads: int,
        n_kv_heads: int,
        d_head: int,
        d_ff: int,
        rope: Optional[RotaryEmbedding] = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(d_model, eps)
        self.attn = GroupedQueryAttention(
            d_model, n_query_heads, n_kv_heads, d_head, rope
        )
        self.norm_ffn = RMSNorm(d_model, eps)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        keep: torch.Tensor,
        cache: Optional[dict] = None,
        cache_key: Optional[tuple] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), positions, keep, cache, cache_key)
        return x + self.ffn(self.norm_ffn(x))
