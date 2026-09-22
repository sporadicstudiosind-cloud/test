"""Parallel transformer blocks and function-preserving depth growth.

Two independent ideas, both about restructuring the *stack* rather than any
single sublayer:

* :class:`ParallelBlock` computes attention and the FFN from the same normed
  input and sums both into the residual, instead of running them one after
  the other: ``x + attn(norm(x)) + ffn(norm(x))`` in place of
  ``x = x + attn(norm(x)); x = x + ffn(norm(x))``. This is the GPT-J /
  PaLM formulation (Chowdhery et al., "PaLM: Scaling Language Modeling with
  Pathways", arXiv:2204.02311, JMLR 2023, §2); PaLM report it gives ~15%
  faster *training* throughput at their largest scales because the two
  sublayers now read the same normed tensor, so their input matmuls (Q/K/V
  projection and the FFN's up/gate projection) can be fused into one larger
  matmul, and likewise the two output projections (attention's ``W_O`` and
  the FFN's down-projection) can fuse on the way back into the residual --
  concurrency the sequential form cannot offer because the FFN's input
  literally does not exist until attention's output has been added to the
  residual. This codebase does not implement that fusion (it would require
  concatenating ``wq``/``wk``/``wv`` with the FFN's ``gate``/``up`` into one
  ``nn.Linear``, which is an integration decision for whoever wires this in,
  not a change this module makes on its own); what is provided here is the
  *shape* of the computation the fusion needs, i.e. one norm read by two
  independent sublayers rather than two norms in sequence.

  PaLM's own ablation is a real tradeoff, not a free win, and is repeated
  here rather than summarised away: they measured a small quality
  degradation from the parallel formulation at 8B parameters, no measurable
  degradation at 62B, and extrapolated (did not measure) neutrality at
  540B. Nothing in this file measures quality at any scale -- these are
  architectural changes whose benefit or cost only shows up after training,
  and claiming otherwise from a forward-pass unit test would be exactly the
  kind of unmeasured claim this codebase's tests are supposed to prevent.
  What *is* measured, in ``tests/unit/test_blocks.py``, is that a
  ``ParallelBlock`` costs one fewer norm's worth of parameters than
  ``TransformerBlock`` (one shared norm instead of two) and produces finite
  gradients.

* :func:`grow_depth` inserts new blocks into an existing ``nn.ModuleList`` in
  a way that leaves the network's *function* unchanged at the instant of
  growth: each inserted block's output projections (``attn.wo``,
  ``ffn.down``) are zero-initialised, so the block computes a zero residual
  update no matter what its input-side weights are, and it is functionally
  absent until training moves those projections away from zero. This is the
  standard trick behind function-preserving network growth (zero/near-zero
  output initialisation for a newly added path is also how ReZero and
  DeepNorm keep a deepened network's forward pass numerically unchanged at
  init) applied to *inserted*, not just appended, layers, which is the part
  that makes "dynamically stackable" mean something beyond "add to the end".
  The broader motivation -- growing depth partway through training rather
  than fixing it up front -- follows a real progressive-training literature:
  Gong et al. ("Efficient Training of BERT by Progressively Stacking", ICML
  2019) double a shallow, already-trained BERT by stacking two copies of it
  and continuing training, and report matching from-scratch quality for
  much less compute; Agarwal, Awasthi, Kale & Zhao ("Stacking as Accelerated
  Gradient Descent", arXiv:2403.04978, 2024) give a theoretical account of
  *why* copying trained shallow layers into new deep ones behaves like a
  momentum/Nesterov step rather than a random restart; Du et al.
  ("Stacking Your Transformers: A Closer Look at Model Growth for Efficient
  LLM Pre-training", NeurIPS 2024) run a systematic comparison of growth
  operators (stacking among them) for LLM pre-training specifically. What
  is implemented here is a simplified, function-preserving instantiation in
  that spirit -- zero-init the grown block's output path -- not a
  reproduction of any one paper's exact copying schedule or of Gong et al.'s
  specific double-then-finetune recipe.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from .layers import GroupedQueryAttention, SwiGLU
from .norms import norm_param_count, make_norm
from .rope import RotaryEmbedding


class ParallelBlock(nn.Module):
    """``x + attn(norm(x)) + ffn(norm(x))``. Drop-in for ``layers.TransformerBlock``.

    Same constructor signature as ``TransformerBlock`` plus ``norm_kind``
    (one of ``"rms"``, ``"dyt"``, ``"derf"`` -- not ``"adaptive"``, which
    needs a conditioning tensor this block's ``forward`` has no slot for)
    and the identical ``forward(x, positions, keep, cache, cache_key)``
    signature, so ``ControlCore`` can hold a mix of ``TransformerBlock`` and
    ``ParallelBlock`` instances in the same ``nn.ModuleList`` without any
    other change.
    """

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
        norm_kind: str = "rms",
    ) -> None:
        super().__init__()
        if norm_kind == "adaptive":
            raise ValueError(
                "ParallelBlock.forward has no conditioning input; "
                "'adaptive' needs one and cannot be selected here"
            )
        self.norm_kind = norm_kind
        self.norm = make_norm(norm_kind, d_model, eps)
        self.attn = GroupedQueryAttention(
            d_model, n_query_heads, n_kv_heads, d_head, rope,
            window=window, softcap=softcap,
        )
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        keep: torch.Tensor,
        cache: Optional[dict] = None,
        cache_key: Optional[tuple] = None,
    ) -> torch.Tensor:
        n = self.norm(x)
        a = self.attn(n, positions, keep, cache, cache_key)
        f = self.ffn(n)
        return x + a + f

    @staticmethod
    def param_count(
        d_model: int, n_query_heads: int, n_kv_heads: int, d_head: int, d_ff: int,
        norm_kind: str = "rms",
    ) -> int:
        d_kv = n_kv_heads * d_head
        attn = 2 * d_model * d_model + 2 * d_model * d_kv
        ffn = 3 * d_model * d_ff
        norm = norm_param_count(norm_kind, d_model)   # one shared norm, not two
        return attn + ffn + norm


def _zero_init_delta(block: nn.Module) -> nn.Module:
    """Deep-copy ``block`` and zero its output projections in place.

    Works for both ``TransformerBlock`` and ``ParallelBlock`` because both
    expose ``.attn.wo`` (``GroupedQueryAttention``) and ``.ffn.down``
    (``SwiGLU``) under those exact names. Zeroing both means the copy's
    total contribution to the residual stream is exactly zero regardless of
    the (otherwise ordinarily-initialised, or copied-and-then-zeroed)
    weights upstream of them -- upstream weights only ever reach the
    residual stream through one of these two projections.
    """
    new_block = copy.deepcopy(block)
    with torch.no_grad():
        nn.init.zeros_(new_block.attn.wo.weight)
        if new_block.attn.wo.bias is not None:
            nn.init.zeros_(new_block.attn.wo.bias)
        nn.init.zeros_(new_block.ffn.down.weight)
    return new_block


def grow_depth(
    blocks: nn.ModuleList,
    new_depth: Optional[int] = None,
    factor: Optional[float] = None,
    method: str = "append",
) -> nn.ModuleList:
    """Return a NEW ``nn.ModuleList`` with more blocks, function-preserving at growth.

    Exactly one of ``new_depth`` (target length) or ``factor`` (multiply the
    current length, rounded) must be given. ``method``:

    * ``"append"`` -- add zero-init copies of the last block at the end.
      Simplest, and the right default for "the network needs more capacity,
      it does not matter where".
    * ``"interleave"`` -- insert a zero-init copy immediately after every
      existing block (so depth at most doubles in one call; call again to
      grow further), closest in spirit to Gong et al.'s stack-a-copy-of-the
      -whole-network doubling, but inserted throughout rather than appended
      as one contiguous second half.
    * ``"duplicate"`` -- like ``"append"``, but each new block is grown from
      the block that will be its immediate predecessor at insertion time
      (so consecutive new blocks are copies of a *chain*, each one derived
      from the previous new one, not all copies of the original last block).
      This is the one to use when the intent is specifically "give the new
      layers a trained starting point close to their neighbour" rather than
      "give them an architecturally valid but functionally inert start" --
      both are function-preserving at the moment of growth (both zero the
      output projections), the difference is only what they look like once
      training starts moving them away from zero.

    Every new block is a deep copy (architecture and, other than the zeroed
    output projections, weights) of an existing block, so this requires
    ``blocks`` to be non-empty and assumes -- as the identity-preservation
    claim requires -- that every existing block already has zero-mean-safe
    output projections and nothing else in the block reads the *position*
    of a layer in the stack (RoPE and windowing are per-call arguments, not
    per-layer state, so this holds for ``TransformerBlock``/``ParallelBlock``
    as given).
    """
    old = list(blocks)
    if not old:
        raise ValueError("grow_depth needs at least one existing block to copy")
    n = len(old)
    if (new_depth is None) == (factor is None):
        raise ValueError("pass exactly one of new_depth or factor")
    if new_depth is None:
        new_depth = max(n, round(n * factor))
    if new_depth < n:
        raise ValueError(f"new_depth {new_depth} is smaller than the current depth {n}")

    if method == "append":
        grown = [_zero_init_delta(old[-1]) for _ in range(new_depth - n)]
        return nn.ModuleList(old + grown)

    if method == "duplicate":
        out = list(old)
        for _ in range(new_depth - n):
            out.append(_zero_init_delta(out[-1]))
        return nn.ModuleList(out)

    if method == "interleave":
        out: list = []
        i = 0
        while len(out) < new_depth and i < n:
            out.append(old[i])
            if len(out) < new_depth:
                out.append(_zero_init_delta(old[i]))
            i += 1
        while len(out) < new_depth:
            out.append(_zero_init_delta(out[-1]))
        return nn.ModuleList(out)

    raise ValueError(f"unknown growth method {method!r}; choose append, interleave or duplicate")
