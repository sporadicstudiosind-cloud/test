"""Build one control-core layer from its config: the join between options and modules.

Every architecture option added to the control core -- latent attention,
local/global/linear-attention layer schedules, normalisation-free norms,
parallel blocks, token-conditioned FFNs -- is a separate module with its own
tests. This file is the only place that decides which of them a given layer
gets. Keeping that decision in one function is what keeps three things in
agreement that otherwise drift apart silently:

* the modules :class:`~iridium.model.control_core.ControlCore` instantiates,
* the parameter count :class:`~iridium.config.CoreConfig` reports, and
* the projections the model's depth-scaled init and ``grow_depth`` zero or
  rescale.

The third is the one that bit first. ``Iridium1.__init__`` rescales
``attn.wo`` and ``ffn.down`` on every block, and ``grow_depth`` zeroes them to
make an inserted block an exact identity. A token-conditioned FFN has no
``ffn.down.weight`` -- its output lives in two linears, ``down.base`` and
``down.U`` -- and a parallel block is not a ``TransformerBlock`` subclass, so
the rescale skipped it entirely. :func:`output_projections` is the single
answer to "which linears write this block's output into the residual stream".

Lives in its own module because ``norms``, ``mla``, ``blocks`` and ``deltanet``
all import from ``layers``; a factory inside ``layers`` would be an import
cycle.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from ..config import CoreConfig, SuperstackConfig
from .layers import GroupedQueryAttention, SwiGLU, TransformerBlock
from .rope import RotaryEmbedding

__all__ = ["build_block", "build_core_block", "block_rope", "output_projections"]


def _attention(cfg, kind: str, rope: RotaryEmbedding) -> nn.Module:
    if kind in ("global", "local"):
        window = cfg.local_window if kind == "local" else None
        return GroupedQueryAttention(cfg.d_model, cfg.n_query_heads, cfg.n_kv_heads,
                                     cfg.d_head, rope, window=window)
    if kind == "mla":
        from .mla import MultiHeadLatentAttention
        return MultiHeadLatentAttention(cfg.d_model, cfg.n_query_heads, cfg.d_head,
                                        cfg.mla_kv_rank, cfg.mla_q_rank,
                                        cfg.mla_rope_dim, rope)
    if kind == "deltanet":
        from .deltanet import GatedDeltaNet
        return GatedDeltaNet(cfg.d_model, cfg.n_query_heads, cfg.d_head,
                             conv_size=cfg.deltanet_conv)
    raise ValueError(f"unknown layer kind {kind!r}")


def _ffn(cfg) -> nn.Module:
    if cfg.ffn_dynamic_rank:
        from .dynamic import DynamicSwiGLU
        return DynamicSwiGLU(cfg.d_model, cfg.d_ff, cfg.ffn_dynamic_rank)
    return SwiGLU(cfg.d_model, cfg.d_ff)


def block_rope(cfg, rope: RotaryEmbedding) -> RotaryEmbedding:
    """The rotary table MLA layers need, given the stack's shared one.

    MLA rotates only an ``mla_rope_dim``-wide slice; handing it a table of
    another width rotates the wrong number of channels. Same theta as the
    shared table, so positions mean the same thing in every layer.
    """
    if "mla" in cfg.layer_kinds() and cfg.mla_rope_dim != rope.d_head:
        return RotaryEmbedding(cfg.mla_rope_dim, getattr(rope, "theta", 500_000.0))
    return rope


def build_block(cfg, rope: RotaryEmbedding, index: int, eps: Optional[float] = None,
                mla_rope: Optional[RotaryEmbedding] = None) -> nn.Module:
    """The block for layer ``index`` of a core or superstack config.

    Both :class:`~iridium.config.CoreConfig` and
    :class:`~iridium.config.SuperstackConfig` carry the same per-layer
    options, so one factory serves both. The all-defaults config yields the
    exact ``TransformerBlock`` these stacks have always built, constructed in
    the same order, so default initialisation draws the same random numbers.
    """
    eps = getattr(cfg, "norm_eps", 1e-5) if eps is None else eps
    kind = cfg.layer_kinds()[index]
    if kind == "mla":
        rope = mla_rope if mla_rope is not None else block_rope(cfg, rope)
    default_parts = kind == "global" and not cfg.ffn_dynamic_rank
    if cfg.block_kind == "parallel":
        from .blocks import ParallelBlock
        block = ParallelBlock(cfg.d_model, cfg.n_query_heads, cfg.n_kv_heads,
                              cfg.d_head, cfg.d_ff, rope, eps,
                              norm_kind=cfg.norm_kind)
    else:
        block = TransformerBlock(cfg.d_model, cfg.n_query_heads, cfg.n_kv_heads,
                                 cfg.d_head, cfg.d_ff, rope, eps)
        if cfg.norm_kind != "rms":
            from .norms import make_norm
            block.norm_attn = make_norm(cfg.norm_kind, cfg.d_model, eps)
            block.norm_ffn = make_norm(cfg.norm_kind, cfg.d_model, eps)
    if not default_parts:
        # Replacing the default parts after construction, rather than
        # threading every option through two block constructors, keeps the
        # default path byte-for-byte what it was. The discarded modules cost a
        # little init time on non-default configs only.
        block.attn = _attention(cfg, kind, rope)
        block.ffn = _ffn(cfg)
    return block


def build_core_block(cfg: CoreConfig, rope: RotaryEmbedding, index: int,
                     mla_rope: Optional[RotaryEmbedding] = None) -> nn.Module:
    """Core layer ``index``; see :func:`build_block`."""
    return build_block(cfg, rope, index, mla_rope=mla_rope)


def output_projections(block: nn.Module) -> list[nn.Linear]:
    """Every linear through which ``block`` writes into the residual stream.

    Zeroing all of them makes the block an exact identity (its residual
    contribution is zero whatever the upstream weights), which is what
    function-preserving depth growth needs; scaling them is what depth-scaled
    init needs. Anything a block contributes that is not listed here would
    survive both, so this list must be complete for every block type the
    factory can build -- ``tests/unit/test_core_blocks.py`` checks it is.
    """
    out: list[nn.Linear] = []
    attn = getattr(block, "attn", None)
    for name in ("wo", "out_proj", "o_proj"):
        proj = getattr(attn, name, None)
        if isinstance(proj, nn.Linear):
            out.append(proj)
            break
    down = getattr(getattr(block, "ffn", None), "down", None)
    if isinstance(down, nn.Linear):
        out.append(down)
    elif down is not None:
        out.extend(m for m in (getattr(down, "base", None), getattr(down, "U", None))
                   if isinstance(m, nn.Linear))
    return out
