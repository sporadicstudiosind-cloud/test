"""Superstacks: deep domain banks with sparse stack-local memory.

The problem the source plan named
-------------------------------
If token ``t`` is routed to stack A and token ``t+1`` to stack B, causal
self-attention inside B is undefined: B never saw ``t``, so it has no key or
value for it. The plan's answer was to remove self-attention from the stacks
entirely and have them cross-attend to the core's cache. That works, but it
throws away something real — a stack cannot relate a token to the *other tokens
it specialises in*, which is most of what a domain bank is for.

What is implemented here instead
--------------------------------
Two memories, and the pair is what makes incremental decoding exact.

1. **Stack-local sparse KV.** Inside stack ``s``, a token attends to the subset
   of earlier tokens that were *also* routed to ``s``. That set is always
   available: it is exactly the set of tokens that ever ran these layers, so
   their keys and values were computed and cached when they did. There is no
   hole to fill.
2. **Bridge cross-attention onto the core.** Every ``cross_stride`` layers the
   stack reads the control core's stage-I states for all positions ``<= t``.
   The core is the only component that sees the whole stream, and it stays the
   single writer of stream history.

The consequence that a test can check: teacher-forced batch evaluation and
token-by-token decoding produce the same numbers, because both define
attention over the same two sets. ``tests/integration/test_kv_parity.py``
asserts this in float64.

Packing
-------
Routed tokens are gathered into a dense ``[B, M, d]`` batch. Packing preserves
stream order, so an ordinary causal mask is correct over packed indices — but
**rotary positions use the original stream positions**, never the packed ones.
Tokens 3, 7, 11 packed adjacently are four apart, not one, and encoding them as
adjacent would make a stack's geometry depend on how busy the router was.

Depth
-----
Depth is a PonderNet ladder over layers: every layer emits a halting
probability, the training objective is the stopping-distribution-weighted
mixture, and serving takes a hard early exit. The router's focus scalar biases
the halting logits, so "more focus" means "halt later" through a learned,
differentiable path rather than through an index that has no gradient.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint

from ..config import IridiumConfig, SuperstackConfig
from .fno import FNOBlock
from .layers import (
    BridgeCrossAttention,
    RMSNorm,
    TransformerBlock,
    causal_keep,
    position_keep,
)
from .rope import RotaryEmbedding
from .router import geometric_prior, ponder_kl, stopping_distribution


@dataclass
class PackedBatch:
    """Routed tokens gathered into a dense batch, with the maps to undo it."""

    hidden: torch.Tensor         # [B, M, d]
    positions: torch.Tensor      # [B, M] original stream positions
    valid: torch.Tensor          # [B, M] bool
    batch_index: torch.Tensor    # [n] source batch index
    token_index: torch.Tensor    # [n] source token index
    slot_index: torch.Tensor     # [n] destination packed slot
    weight: torch.Tensor         # [n] gate weight for each member
    slot_of: torch.Tensor        # [B, T] packed slot of each stream token
    member: torch.Tensor         # [B, T] bool

    @property
    def empty(self) -> bool:
        return self.batch_index.numel() == 0


def pack_routed(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    member: torch.Tensor,
    weight: torch.Tensor,
) -> PackedBatch:
    """Gather ``member`` tokens into ``[B, M, d]``, preserving stream order."""
    b, t, d = hidden.shape
    counts = member.sum(dim=1)
    m = int(counts.max().item()) if counts.numel() else 0
    bi, ti = member.nonzero(as_tuple=True)
    rank = (member.cumsum(dim=1) - 1).clamp_min(0)
    si = rank[bi, ti]

    packed = hidden.new_zeros(b, max(m, 1), d)
    packed_pos = positions.new_zeros(b, max(m, 1))
    valid = torch.zeros(b, max(m, 1), dtype=torch.bool, device=hidden.device)
    if bi.numel():
        packed[bi, si] = hidden[bi, ti]
        packed_pos[bi, si] = positions[bi, ti]
        valid[bi, si] = True
    return PackedBatch(
        hidden=packed,
        positions=packed_pos,
        valid=valid,
        batch_index=bi,
        token_index=ti,
        slot_index=si,
        weight=weight[bi, ti],
        slot_of=rank,
        member=member,
    )


class SuperstackLayer(nn.Module):
    def __init__(
        self,
        cfg: SuperstackConfig,
        rope: RotaryEmbedding,
        bridge: bool,
        spectral: bool,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.block = TransformerBlock(
            cfg.d_model,
            cfg.n_query_heads,
            cfg.n_kv_heads,
            cfg.d_head,
            cfg.d_ff,
            rope,
            eps,
        )
        self.bridge: Optional[BridgeCrossAttention] = None
        self.bridge_norm: Optional[RMSNorm] = None
        if bridge:
            self.bridge_norm = RMSNorm(cfg.d_model, eps)
            self.bridge = BridgeCrossAttention(
                cfg.d_model,
                cfg.core_d_model or cfg.d_model,
                cfg.n_query_heads,
                cfg.n_kv_heads,
                cfg.d_head,
                rope,
            )
        self.spectral: Optional[FNOBlock] = None
        if spectral:
            modes = (cfg.spectral_modes, cfg.spectral_modes)
            self.spectral = FNOBlock(cfg.d_model, cfg.spectral_channels, modes, eps)
        self.halt_head = nn.Linear(cfg.d_model, 1)
        nn.init.zeros_(self.halt_head.bias)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        self_keep: torch.Tensor,
        core_states: Optional[torch.Tensor],
        core_positions: Optional[torch.Tensor],
        bridge_keep: Optional[torch.Tensor],
        grids: Sequence[tuple[int, int, tuple[int, ...]]] = (),
        cache: Optional[dict] = None,
        cache_key: Optional[tuple] = None,
    ) -> torch.Tensor:
        x = self.block(x, positions, self_keep, cache, cache_key)
        if self.bridge is not None and core_states is not None:
            x = x + self.bridge(
                self.bridge_norm(x),
                positions,
                core_states,
                core_positions,
                bridge_keep,
            )
        if self.spectral is not None:
            x = self.spectral(x, grids)
        return x


@dataclass
class StackOutput:
    hidden: torch.Tensor             # [B, M, d] mixture over stopping depths
    stopping: torch.Tensor           # [B, M, L] stopping distribution
    expected_depth: torch.Tensor     # [B, M] float
    executed_layers: int             # layers actually run (serving metric)


class Superstack(nn.Module):
    """One deep domain bank."""

    def __init__(
        self,
        cfg: SuperstackConfig,
        stack_index: int,
        rope: RotaryEmbedding,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stack_index = stack_index
        self.specialization = (
            cfg.specializations[stack_index] if cfg.specializations else f"stack_{stack_index}"
        )
        has_spectral = cfg.has_spectral(stack_index)
        self.layers = nn.ModuleList(
            SuperstackLayer(
                cfg,
                rope,
                bridge=(i % cfg.cross_stride == 0),
                spectral=(
                    has_spectral
                    and cfg.spectral_stride > 0
                    and i % cfg.spectral_stride == 0
                ),
                eps=eps,
            )
            for i in range(cfg.n_layers)
        )
        self.exit_norm = RMSNorm(cfg.d_model, eps)
        # Focus -> halting bias. Positive gain means "more focus, halt later".
        self.focus_gain = nn.Parameter(torch.tensor(4.0))
        # See Iridium1.enable_gradient_checkpointing for what this does and
        # does not cover, and the measurements behind why it defaults off.
        self.gradient_checkpointing = False

    def forward(
        self,
        packed: PackedBatch,
        focus: torch.Tensor,                   # [B, M]
        core_states: Optional[torch.Tensor],
        core_positions: Optional[torch.Tensor],
        grids: Sequence[tuple[int, int, tuple[int, ...]]] = (),
        cache: Optional[dict] = None,
        cache_prefix: tuple = (),
        hard_exit: bool = False,
        exit_threshold: float = 0.9,
        depth_cap: Optional[torch.Tensor] = None,
    ) -> StackOutput:
        x = packed.hidden
        b, m, _ = x.shape
        positions = packed.positions
        valid = packed.valid

        if cache is not None:
            past_valid = cache.get((cache_prefix, "valid"))
            if past_valid is None:
                key_valid = valid
            else:
                key_valid = torch.cat([past_valid, valid], dim=1)
            history = key_valid.shape[1] - m
            self_keep = causal_keep(m, history + m, x.device) & key_valid.view(
                b, 1, 1, history + m
            )
            cache[(cache_prefix, "valid")] = key_valid
        else:
            self_keep = causal_keep(m, m, x.device) & valid.view(b, 1, 1, m)

        bridge_keep = None
        if core_states is not None:
            bridge_keep = position_keep(positions, core_positions)

        focus_bias = self.focus_gain * (0.5 - focus)       # [B, M]
        lambdas: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        alive = valid.clone()
        executed = 0

        # Checkpointing recomputes each layer during backward, which would
        # write its cache entry a second time if a KV cache were live here.
        # cache is only non-None during incremental serving, where there is
        # no backward pass anyway. hard_exit's per-row early stop lives in
        # this Python loop, outside the checkpointed call, so recompute
        # replays exactly the layer that was actually run — see the
        # docstring on Iridium1.enable_gradient_checkpointing for the
        # measurement that this holds even with hard_exit=True.
        use_checkpoint = (
            self.gradient_checkpointing
            and self.training
            and cache is None
            and x.requires_grad
            and torch.is_grad_enabled()
        )

        for depth, layer in enumerate(self.layers):
            if hard_exit and not bool(alive.any()):
                break
            executed += 1
            key = (cache_prefix, depth) if cache is not None else None
            if use_checkpoint:
                def run_layer(
                    xx: torch.Tensor,
                    layer: SuperstackLayer = layer,
                ) -> torch.Tensor:
                    return layer(
                        xx, positions, self_keep, core_states, core_positions,
                        bridge_keep, grids, None, None,
                    )

                x = torch_checkpoint.checkpoint(
                    run_layer, x, use_reentrant=False, preserve_rng_state=True,
                )
            else:
                x = layer(
                    x,
                    positions,
                    self_keep,
                    core_states,
                    core_positions,
                    bridge_keep,
                    grids,
                    cache,
                    key,
                )
            logit = layer.halt_head(x).squeeze(-1).to(torch.float64 if x.dtype == torch.float64 else torch.float32) + focus_bias
            lam = torch.sigmoid(logit)
            if depth == len(self.layers) - 1:
                lam = torch.ones_like(lam)
            if depth_cap is not None:
                forced = (depth + 1) >= depth_cap
                lam = torch.where(forced, torch.ones_like(lam), lam)
            if depth + 1 < self.cfg.min_depth:
                lam = torch.zeros_like(lam)
            lambdas.append(lam)
            states.append(self.exit_norm(x))
            if hard_exit:
                p_so_far = stopping_distribution(torch.stack(lambdas, -1))
                alive = alive & (p_so_far.sum(-1) < exit_threshold)

        lam_stack = torch.stack(lambdas, dim=-1)           # [B, M, L']
        p = stopping_distribution(lam_stack)
        h_stack = torch.stack(states, dim=-1)              # [B, M, d, L']
        mixed = (h_stack * p.unsqueeze(-2)).sum(-1)
        steps = torch.arange(
            1, p.shape[-1] + 1, device=p.device, dtype=p.dtype
        )
        return StackOutput(
            hidden=mixed,
            stopping=p,
            expected_depth=(p * steps).sum(-1),
            executed_layers=executed,
        )


class SuperstackBank(nn.Module):
    """All stacks, plus the dispatch and gather that connect them to the core."""

    def __init__(self, cfg: IridiumConfig, rope: RotaryEmbedding) -> None:
        super().__init__()
        self.cfg = cfg
        self.stacks = nn.ModuleList(
            Superstack(cfg.stacks, i, rope, cfg.core.norm_eps)
            for i in range(cfg.stacks.n_stacks)
        )
        st = cfg.stacks
        if st.uses_io_projection:
            self.entry = nn.Linear(cfg.core.d_model, st.d_model, bias=False)
            self.exit = nn.Linear(st.d_model, cfg.core.d_model, bias=False)
        else:
            self.entry = None
            self.exit = None

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        decision,
        core_states: Optional[torch.Tensor],
        core_positions: Optional[torch.Tensor],
        grids: Sequence[tuple[int, int, tuple[int, ...]]] = (),
        cache: Optional[dict] = None,
        cache_prefix: tuple = (),
        hard_exit: bool = False,
        exit_threshold: float = 0.9,
        use_depth_cap: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        b, t, d = hidden.shape
        out = torch.zeros_like(hidden)
        stats: dict = {
            "per_stack_tokens": [],
            "per_stack_expected_depth": [],
            "per_stack_executed_layers": [],
            "depth_kl": hidden.new_zeros(()),
        }
        grid_intact = 0
        grid_total = 0
        prior = geometric_prior(
            self.cfg.stacks.n_layers,
            self.cfg.router.depth_prior_p_stop,
            hidden.device,
            torch.float32,
        )
        n_active_stacks = 0
        bridge_by_device = {}

        for s, stack in enumerate(self.stacks):
            member = (decision.stack_index == s).any(dim=-1)      # [B, T]
            if getattr(decision, "valid", None) is not None:
                member = member & decision.valid
            n_members = int(member.sum().item())
            stats["per_stack_tokens"].append(n_members)
            if n_members == 0:
                stats["per_stack_expected_depth"].append(0.0)
                stats["per_stack_executed_layers"].append(0)
                continue

            gate = (
                decision.stack_weight
                * (decision.stack_index == s).to(decision.stack_weight.dtype)
            ).sum(-1)                                             # [B, T]
            packed = pack_routed(hidden, positions, member, gate)
            focus_packed = _pack_scalar(decision.focus, packed)
            depth_cap = (
                _pack_scalar(decision.target_depth.float(), packed)
                if use_depth_cap
                else None
            )
            x = packed.hidden if self.entry is None else self.entry(packed.hidden)
            packed = PackedBatch(
                hidden=x,
                positions=packed.positions,
                valid=packed.valid,
                batch_index=packed.batch_index,
                token_index=packed.token_index,
                slot_index=packed.slot_index,
                weight=packed.weight,
                slot_of=packed.slot_of,
                member=packed.member,
            )
            local_grids, kept, total = translate_grids(grids, packed)
            grid_intact += kept
            grid_total += total
            stack_device = next(stack.parameters()).device
            if stack_device not in bridge_by_device:
                bridge_by_device[stack_device] = (
                    core_states.to(stack_device) if core_states is not None else None,
                    core_positions.to(stack_device) if core_positions is not None else None)
            local_core, local_positions = bridge_by_device[stack_device]
            local = replace(packed, **{name: value.to(stack_device)
                            for name, value in vars(packed).items() if torch.is_tensor(value)})
            result = stack(
                local,
                focus_packed.to(stack_device),
                local_core,
                local_positions,
                local_grids,
                cache,
                cache_prefix + (s,),
                hard_exit,
                exit_threshold,
                depth_cap.to(stack_device) if depth_cap is not None else None,
            )
            result.hidden = result.hidden.to(hidden.device)
            result.stopping = result.stopping.to(hidden.device)
            result.expected_depth = result.expected_depth.to(hidden.device)
            y = result.hidden if self.exit is None else self.exit(result.hidden)
            contribution = y[packed.batch_index, packed.slot_index]
            out.index_put_(
                (packed.batch_index, packed.token_index),
                (contribution * packed.weight.unsqueeze(-1)).to(out.dtype),
                accumulate=True,
            )

            valid = packed.valid
            depth_p = result.stopping[valid]
            if depth_p.numel():
                padded = torch.zeros(
                    depth_p.shape[0], self.cfg.stacks.n_layers, device=depth_p.device,
                    dtype=depth_p.dtype,
                )
                padded[:, : depth_p.shape[-1]] = depth_p
                stats["depth_kl"] = stats["depth_kl"] + ponder_kl(padded, prior)
                n_active_stacks += 1
            stats["per_stack_expected_depth"].append(
                float(result.expected_depth[valid].detach().mean())
            )
            stats["per_stack_executed_layers"].append(result.executed_layers)

        if n_active_stacks:
            stats["depth_kl"] = stats["depth_kl"] / n_active_stacks
        stats["grid_intact"] = grid_intact
        stats["grid_total"] = grid_total
        stats["grid_intact_fraction"] = (
            grid_intact / grid_total if grid_total else 1.0
        )
        return out, stats


def translate_grids(
    grids: Sequence[tuple[int, int, tuple[int, ...]]], packed: PackedBatch
) -> tuple[list[tuple[int, int, tuple[int, ...]]], int, int]:
    """Map stream-coordinate field grids into packed coordinates.

    A spectral operator needs a *complete* grid. If the router sent only part
    of a field span to this stack, the remaining patches are not here and no
    amount of reshaping will conjure them: that grid is dropped rather than
    applied to a fragment. The count of intact versus total grids is reported
    so the loss of spectral coverage is visible instead of silent — see
    ``MacroRouter(span_id=...)`` for the span-coherent routing that keeps the
    fraction at 1.0.
    """
    out: list[tuple[int, int, tuple[int, ...]]] = []
    kept = 0
    considered = 0
    for b, start, shape in grids:
        n = 1
        for s_ in shape:
            n *= s_
        window = packed.member[b, start : start + n]
        if window.numel() != n or not bool(window.any()):
            continue                    # this stack saw none of the grid
        considered += 1                 # it saw some: completeness is now at stake
        if not bool(window.all()):
            continue
        slots = packed.slot_of[b, start : start + n]
        if int(slots[-1] - slots[0]) != n - 1:      # must stay contiguous
            continue
        out.append((b, int(slots[0]), shape))
        kept += 1
    return out, kept, considered


def _pack_scalar(values: torch.Tensor, packed: PackedBatch) -> torch.Tensor:
    """Gather a ``[B, T]`` per-token scalar into the packed ``[B, M]`` layout."""
    out = values.new_zeros(packed.hidden.shape[0], packed.hidden.shape[1])
    if packed.batch_index.numel():
        out[packed.batch_index, packed.slot_index] = values[
            packed.batch_index, packed.token_index
        ]
    return out
