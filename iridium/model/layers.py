"""Transformer primitives with SDPA by default and a manual reference backend.

Masks stay broadcast over heads. Fully masked rows are zeroed explicitly.
Optional parameter-free Q/K normalization precedes RoPE; positions remain the
original stream positions in both core and packed superstack attention.

Three further knobs, all keyword-only, all defaulting to today's behaviour:
``GroupedQueryAttention``/``TransformerBlock`` take ``window`` (a per-layer
sliding-window size, for interleaving local and global layers a la Gemma 3 /
Mistral) and ``softcap`` (Gemma 2 attention-logit softcapping), and grouped
query attention now expands kv heads via SDPA's ``enable_gqa`` on backends
that support it instead of always calling ``repeat_interleave`` first. See
``_attend`` for what forces the manual backend and why, and ``sliding_window_keep``
for how a window composes with an existing causal/keep mask.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding


def head_rms(x):
    work = x.float()
    return (work * (work.square().mean(-1, keepdim=True) + 1e-6).rsqrt()).to(x.dtype)


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
        return (x32 * norm).to(dtype) * self.weight.to(dtype)


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

#: Cached result of probing whether this torch build's SDPA accepts
#: ``enable_gqa`` (added in torch 2.5). Probed once with a throwaway tensor
#: rather than parsed from ``torch.__version__``, because printed versions
#: differ across ROCm/nightly/vendor builds in ways a string match eventually
#: gets wrong, and a 1-element attention call costs nothing next to a real one.
_ENABLE_GQA_SUPPORTED: Optional[bool] = None


def _enable_gqa_supported() -> bool:
    global _ENABLE_GQA_SUPPORTED
    if _ENABLE_GQA_SUPPORTED is None:
        try:
            q = torch.zeros(1, 2, 1, 4)
            kv = torch.zeros(1, 1, 1, 4)
            F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
            _ENABLE_GQA_SUPPORTED = True
        except TypeError:
            _ENABLE_GQA_SUPPORTED = False
    return _ENABLE_GQA_SUPPORTED


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
            keep: torch.Tensor, d_head: int, n_rep: int = 1,
            softcap: Optional[float] = None) -> torch.Tensor:
    """Attention with an explicit boolean mask, both ways.

    ``k``/``v`` carry ``n_rep`` fewer heads than ``q`` when this is grouped
    query attention (``n_rep = n_query_heads // n_kv_heads``); the two
    backends expand that grouping differently, both described below.

    Both paths agree on the awkward case: a query row with *no* admissible key
    returns zeros rather than NaN. Softmax over an all-``-inf`` row is NaN, and
    a fused kernel propagates that into the residual stream, where it silently
    turns the whole batch into NaN several layers later.

    ``softcap`` (Gemma 2's ``tanh(logits / c) * c`` before the softmax) always
    takes the manual path, on either backend setting. SDPA's fused kernels
    only ever combine an *additive* bias or a boolean ``-inf`` into the score
    matrix — that is the entire vocabulary of ``attn_mask`` and ``is_causal``
    — and there is no bias that turns ``x`` into ``c * tanh(x / c)``, because
    that reshapes the whole score distribution rather than shifting it. So
    softcapping is exactly the case the module docstring means by "some masks
    ... may not map onto SDPA": it is not the mask that fails to map, it is
    the score transform, and enabling it deliberately trades the fused
    kernel's memory/speed win for Gemma 2's cure for logit blowup.
    """
    if softcap is not None:
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)
        scores = softcap * torch.tanh(scores / softcap)
        probs = _masked_softmax(scores, keep)
        return probs @ v

    if _ATTENTION_BACKEND == "sdpa":
        mask = keep
        dead = ~mask.any(dim=-1, keepdim=True)
        # Give a dead row one admissible key so the kernel produces a finite
        # number, then zero the row afterwards. Masking it back out is what
        # keeps this identical to the manual path rather than merely close.
        safe = mask.masked_fill(dead.expand_as(mask) & _first_column(mask), True)
        if n_rep > 1 and _enable_gqa_supported():
            # Let the kernel broadcast kv heads internally instead of us
            # materializing ``n_rep`` copies of k/v first. repeat_interleave
            # is the correct fallback (below) but it is real memory: at 8 kv
            # heads repeated 4x over a long cache, that is 4x the KV cache's
            # footprint held twice during the copy, for a broadcast the
            # kernel is willing to do for free.
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=safe, enable_gqa=True)
        else:
            if n_rep > 1:
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=safe)
        return out.masked_fill(dead, 0.0)

    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
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
        *,
        window: Optional[int] = None,
        softcap: Optional[float] = None,
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
        self.qk_norm = False
        # Per-layer local attention window (Mistral: uniform; Gemma 3: 5
        # local layers per global one). None keeps today's full-causal
        # behaviour; a caller wanting the interleave sets it per-instance
        # after construction, the same way ``qk_norm`` is toggled today,
        # since window is a keyword-only default-preserving knob and this
        # module is not the place that decides the per-layer schedule.
        self.window = window
        # Gemma 2 attention-logit softcapping; None keeps SDPA in play.
        self.softcap = softcap

    def project_kv(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        k = self.wk(x).view(b, t, self.n_kv, self.d_head).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv, self.d_head).transpose(1, 2)
        if self.qk_norm:
            k = head_rms(k)
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
        if self.qk_norm:
            q = head_rms(q)
        if self.rope is not None:
            q = self.rope(q, positions)
        k, v = self.project_kv(x, positions)

        if cache is not None and cache_key is not None:
            prev = cache.get(cache_key)
            if prev is not None:
                k = torch.cat([prev[0], k], dim=2)
                v = torch.cat([prev[1], v], dim=2)
            cache[cache_key] = (k, v)

        if self.window is not None:
            window_mask = sliding_window_keep(q.shape[2], k.shape[2], self.window, q.device)
            keep = keep & window_mask

        out = _attend(q, k, v, keep, self.d_head, n_rep=self.repeat, softcap=self.softcap)
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
        *,
        softcap: Optional[float] = None,
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
        self.qk_norm = False
        # No ``window`` here: a stack-local window narrows self-attention over
        # the stack's own packed sequence, but this attends onto the core's
        # global KV, whose positions are the *original stream* positions
        # (see the module docstring), not the packed axis a window index would
        # need to mean anything against. Softcapping still applies unchanged.
        self.softcap = softcap

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
        if self.qk_norm:
            q, k = head_rms(q), head_rms(k)
        if self.rope is not None:
            q = self.rope(q, q_positions)
            k = self.rope(k, core_positions)
        out = _attend(q, k, v, keep, self.d_head, n_rep=self.repeat, softcap=self.softcap)
        return self.wo(out.transpose(1, 2).reshape(b, t, -1))


def causal_keep(t_q: int, t_k: int, device=None) -> torch.Tensor:
    """``[1, 1, t_q, t_k]`` boolean: query ``i`` (offset by history) sees key ``j``."""
    offset = t_k - t_q
    qi = torch.arange(t_q, device=device).view(-1, 1) + offset
    kj = torch.arange(t_k, device=device).view(1, -1)
    return (kj <= qi).view(1, 1, t_q, t_k)


def sliding_window_keep(t_q: int, t_k: int, window: int, device=None) -> torch.Tensor:
    """``[1, 1, t_q, t_k]`` boolean: query ``i`` sees key ``j`` iff ``i - j < window``.

    Same index convention as :func:`causal_keep` — ``t_k - t_q`` offsets the
    query row so this is correct against a KV cache, not just a fresh
    sequence — because it is meant to be AND-ed with a causal mask the caller
    already built the same way, never used standalone. A window is a *cost*
    property (Mistral's uniform local attention, Gemma 3's 5-local:1-global
    interleave), not a correctness one, so this never decides on its own
    whether key ``j`` is in the past; it only ever narrows a mask that already
    decided that. That is also why this composes safely with the KV-parity
    invariant: it is a deterministic function of the same absolute-position
    offsets ``causal_keep`` uses, so a cached decode step and a full forward
    pass narrow their causal mask identically, token by token.
    """
    offset = t_k - t_q
    qi = torch.arange(t_q, device=device).view(-1, 1) + offset
    kj = torch.arange(t_k, device=device).view(1, -1)
    return ((qi - kj) < window).view(1, 1, t_q, t_k)


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
        *,
        window: Optional[int] = None,
        softcap: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(d_model, eps)
        self.attn = GroupedQueryAttention(
            d_model, n_query_heads, n_kv_heads, d_head, rope,
            window=window, softcap=softcap,
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
