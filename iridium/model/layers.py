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


def at_least_fp32(x: torch.Tensor) -> torch.Tensor:
    """Widen fp16/bf16 to fp32 for a reduction; leave fp32 and fp64 alone.

    Normalisers and softmaxes accumulate, and accumulating in half precision
    loses the sum. The usual cure is ``x.float()`` — which is a *downcast*
    when the caller asked for fp64, and a silent one: the result still has
    dtype float64, carrying fp32 rounding inside it. That is worth naming
    precisely because of where it lands. This repo's cache-parity gate runs
    the model in fp64 and demands cached decoding match the uncached forward
    to 1e-10; a hardcoded ``.float()`` anywhere under it caps the achievable
    agreement at fp32 eps, so the gate passes only while both sides happen to
    round identically, and fails the moment anything makes them differ —
    which is what a fused attention kernel on one CI runner did, producing
    2**-22 and 2**-23 discrepancies in every parity test at once while the
    fp64 physics tests were unaffected.

    ``deltanet.py`` reached this conclusion first and worked around it
    locally; this is that fix, in one place, for every site that needs it.
    """
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def head_rms(x):
    work = at_least_fp32(x)
    return (work * (work.square().mean(-1, keepdim=True) + 1e-6).rsqrt()).to(x.dtype)


def neg_inf(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and (x.requires_grad or self.weight.requires_grad):
            return _RMSNormFn.apply(x, self.weight, self.eps)
        return _rms_norm_reference(x, self.weight, self.eps)


def _rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x32 = at_least_fp32(x)
    norm = x32.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (x32 * norm).to(dtype) * weight.to(dtype)


class _RMSNormFn(torch.autograd.Function):
    """RMSNorm that keeps only its input and one scalar per row for backward.

    Autograd on the plain expression kept the fp32 input, its normalised copy
    and the product before the weight -- two to three full fp32 copies per norm,
    and a superstack runs several norms per layer per token. Same forward
    result; the backward is the closed-form RMSNorm gradient, in fp32.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        dtype = x.dtype
        x32 = at_least_fp32(x)
        rstd = x32.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        ctx.save_for_backward(x, weight, rstd)
        return (x32 * rstd).to(dtype) * weight.to(dtype)

    @staticmethod
    def backward(ctx, grad):
        x, weight, rstd = ctx.saved_tensors
        x32 = at_least_fp32(x)
        xhat = x32 * rstd
        g32 = at_least_fp32(grad)
        grad_weight = None
        if ctx.needs_input_grad[1]:
            # The forward rounds xhat to x's dtype before the weight multiply.
            xr = xhat.to(x.dtype)
            grad_weight = (g32 * at_least_fp32(xr)).reshape(-1, xhat.shape[-1]).sum(0)
            grad_weight = grad_weight.to(weight.dtype)
        grad_x = None
        if ctx.needs_input_grad[0]:
            gw = g32 * at_least_fp32(weight.to(x.dtype))
            dot = (gw * xhat).mean(-1, keepdim=True)
            grad_x = ((gw - xhat * dot) * rstd).to(x.dtype)
        return grad_x, grad_weight, None


class _SiluMulFn(torch.autograd.Function):
    """``silu(g) * u`` keeping only ``g`` and ``u``; silu(g) is recomputed in
    backward instead of stored (one fewer [tokens, d_ff] tensor per FFN)."""

    @staticmethod
    def forward(ctx, g, u):
        ctx.save_for_backward(g, u)
        return F.silu(g) * u

    @staticmethod
    def backward(ctx, grad):
        g, u = ctx.saved_tensors
        sig = torch.sigmoid(g)
        silu = g * sig
        grad_u = grad * silu if ctx.needs_input_grad[1] else None
        grad_g = grad * u * (sig * (1 + g * (1 - sig))) if ctx.needs_input_grad[0] else None
        return grad_g, grad_u


def silu_mul(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and (g.requires_grad or u.requires_grad):
        return _SiluMulFn.apply(g, u)
    return F.silu(g) * u


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(silu_mul(self.gate(x), self.up(x)))


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
        if q.is_cuda:
            _report_cuda_kernel(q, k, v, safe, n_rep)
        # On CUDA the GQA flag is avoided whenever a mask is given (always,
        # here): FlashAttention cannot take an arbitrary mask, and on several
        # torch versions the memory-efficient kernel cannot take enable_gqa, so
        # the flag silently selected the math fallback, which stores the full
        # fp32 score matrix per head and layer. Expanding k/v costs a few MB;
        # the fallback cost gigabytes and most of a T4's time.
        if n_rep > 1 and _enable_gqa_supported() and not q.is_cuda:
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


_CUDA_KERNEL_REPORTED = False


def _report_cuda_kernel(q, k, v, mask, n_rep) -> None:
    """Say once, in the training log, whether masked attention gets a fused
    kernel. A silent fallback to the math kernel is the failure this guards."""
    global _CUDA_KERNEL_REPORTED
    if _CUDA_KERNEL_REPORTED:
        return
    _CUDA_KERNEL_REPORTED = True
    try:
        from torch.backends.cuda import SDPAParams, can_use_efficient_attention
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        try:
            params = SDPAParams(q, k, v, mask, 0.0, False, False)
        except TypeError:                      # torch < 2.5: no enable_gqa argument
            params = SDPAParams(q, k, v, mask, 0.0, False)
        ok = can_use_efficient_attention(params, False)
    except Exception as exc:                  # a diagnostic must never stop training
        print(f"attention: could not check the CUDA kernel ({exc})", flush=True)
        return
    print("attention: memory-efficient kernel" if ok else
          "attention: WARNING math fallback (slow, memory-hungry); report this line",
          flush=True)


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
    probs = torch.softmax(at_least_fp32(filled), dim=-1).to(scores.dtype)
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
        #: Output gate (``enable_gate``); None is the ungated layer.
        self.gate: Optional[nn.Linear] = None
        self.gate_kind = ""

    def enable_gate(self, kind: str, d_model: int) -> None:
        """Gated attention (Qiu et al., "Gated Attention for Large Language
        Models", NeurIPS 2025, arXiv 2505.06708): multiply each head's output by
        ``sigmoid(x W)`` before ``wo``. Their best placement, after the
        attention, so the fast SDPA kernel still runs. "head" is one scalar
        per head; "elem" one per channel. Reported to remove attention sinks
        and allow higher learning rates; shown at 1.7B dense and 15B MoE."""
        if kind not in ("head", "elem"):
            raise ValueError("gate kind must be 'head' or 'elem'")
        self.gate_kind = kind
        self.gate = nn.Linear(d_model, self.n_q if kind == "head" else self.n_q * self.d_head,
                              bias=False)

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
            if self.window is not None:
                # A sliding window is a memory saving only if the cache forgets.
                # Every later query needs at most the ``window - 1`` keys before
                # it, and anything older is masked out -- contributing exactly
                # zero to the softmax -- so dropping it cannot change a result.
                # Keeping it (as an earlier version did) paid full-attention
                # memory for local-attention layers, which is the one cost a
                # window exists to cut. ``tests/unit/test_core_blocks.py``
                # checks the bytes actually cached against the config formula.
                # Start index clamped at 0: while the cache is still shorter than
                # the window, ``len - keep`` is negative, and a negative slice
                # start counts from the end -- silently dropping keys the next
                # query needs. (That was the first version of this line.)
                start = max(k.shape[2] - max(self.window - 1, 0), 0)
                cache[cache_key] = (k[:, :, start:], v[:, :, start:])
            else:
                cache[cache_key] = (k, v)

        if self.window is not None:
            # The caller's mask spans the whole history; the keys held here are
            # its most recent ``k.shape[2]`` columns.
            keep = keep[..., keep.shape[-1] - k.shape[2]:]
            window_mask = sliding_window_keep(q.shape[2], k.shape[2], self.window, q.device)
            keep = keep & window_mask

        out = _attend(q, k, v, keep, self.d_head, n_rep=self.repeat, softcap=self.softcap)
        out = out.transpose(1, 2)                      # [B, T, H, d_head]
        if self.gate is not None:
            g = torch.sigmoid(self.gate(x))
            g = g.unsqueeze(-1) if self.gate_kind == "head" else g.view(b, t, self.n_q, self.d_head)
            out = out * g.to(out.dtype)
        return self.wo(out.reshape(b, t, -1))


class CanonConv(nn.Module):
    """Canon layer (Allen-Zhu, "Physics of Language Models: Part 4.1",
    arXiv 2512.17351): ``h_t + sum_{i<4} w_i * h_{t-i}``, a residual causal
    depthwise convolution of width four.

    It hands each position a mix of its three predecessors, the "horizontal"
    information flow the paper finds attention does poorly on its own; the
    paper reports 2-4x deeper synthetic reasoning and about 0.5% more
    parameters, and an independent 4M-parameter TinyStories run saw ~2.8%
    lower loss. Initialised as the official code does (PyTorch's default
    depthwise Conv1d init, kaiming_uniform a=sqrt(5): U(-1/2, 1/2) for width
    four), no bias, no activation; the paper found identity-like init gave
    no benefit.

    Incremental decoding keeps the last three inputs in the cache under
    ``key``, so a token fed one at a time sees the same window as in a full
    forward pass.
    """

    WIDTH = 4

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_model, self.WIDTH))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.WIDTH)       # = kaiming_uniform(a=sqrt(5)), fan_in 4
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: torch.Tensor, cache: Optional[dict] = None,
                key: Optional[tuple] = None) -> torch.Tensor:
        b, t, d = x.shape
        history = self.WIDTH - 1
        past = cache.get(key) if cache is not None and key is not None else None
        if past is None:
            past = x.new_zeros(b, history, d)
        full = torch.cat([past.to(x.dtype), x], dim=1)          # [B, 3 + T, D]
        if cache is not None and key is not None:
            cache[key] = full[:, -history:].detach()
        w = self.weight.to(x.dtype)
        # out_t = sum_i w[:, i] * x_{t-i}; full[:, history - i + t] is x_{t-i}.
        out = x
        for i in range(self.WIDTH):
            out = out + w[:, i] * full[:, history - i:history - i + t]
        return out


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
        #: Canon-A (before attention) and Canon-C (before the MLP); see
        #: CanonConv. None unless ``enable_canon`` was called.
        self.canon_a: Optional[CanonConv] = None
        self.canon_c: Optional[CanonConv] = None

    def enable_canon(self, d_model: int) -> None:
        self.canon_a = CanonConv(d_model)
        self.canon_c = CanonConv(d_model)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        keep: torch.Tensor,
        cache: Optional[dict] = None,
        cache_key: Optional[tuple] = None,
    ) -> torch.Tensor:
        a = self.norm_attn(x)
        if self.canon_a is not None:
            a = self.canon_a(a, cache, None if cache_key is None else (*cache_key, "canon_a"))
        x = x + self.attn(a, positions, keep, cache, cache_key)
        c = self.norm_ffn(x)
        if self.canon_c is not None:
            c = self.canon_c(c, cache, None if cache_key is None else (*cache_key, "canon_c"))
        return x + self.ffn(c)
