"""Small dense recurrent transformer reference for Iridium-1.

Architecture §6. This is the *correctness-first* implementation: it is slow,
uncached by default, and its outputs define the reference that every cache
optimization must reproduce (§6.5).

Properties this file is responsible for:

* dense shared weights, no expert bank and no router (Invariant 2)
* a weight-tied core applied a variable number of times, with ``H0`` reinjected
  at every recurrence so refinement cannot drift away from the observation
* grouped-query attention, which reduces cache size without introducing experts
* a recurrence-indexed cache, so a core layer run ``R`` times occupies ``R``
  cache slots rather than silently reusing one
* explicit mask construction: causal for committed symbolic output,
  bidirectional *within* a noised continuous block, never onto a clean target
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .inventory import TransformerConfig
from .layers import at_least_fp32


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        work = at_least_fp32(x)
        norm = work.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (work * norm).type_as(x) * self.weight


class GroupedQueryAttention(nn.Module):
    """Multi-head attention with fewer KV heads than query heads."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.n_q = cfg.n_query_heads
        self.n_kv = cfg.n_kv_heads
        self.d_head = cfg.d_head
        self.repeat = cfg.n_query_heads // cfg.n_kv_heads
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_kv, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_kv, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        cache: dict | None = None,
        cache_key: tuple | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_q, self.d_head).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv, self.d_head).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv, self.d_head).transpose(1, 2)

        if cache is not None and cache_key is not None:
            prev = cache.get(cache_key)
            if prev is not None:
                k = torch.cat([prev[0], k], dim=2)
                v = torch.cat([prev[1], v], dim=2)
            cache[cache_key] = (k, v)

        k = k.repeat_interleave(self.repeat, dim=1)
        v = v.repeat_interleave(self.repeat, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores + mask
        out = torch.softmax(at_least_fp32(scores), dim=-1).type_as(q) @ v
        return self.wo(out.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(cfg)
        self.norm_ffn = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        cache: dict | None = None,
        cache_key: tuple | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), mask, cache, cache_key)
        return x + self.ffn(self.norm_ffn(x))


@dataclass
class RecurrencePolicy:
    """Fixed-depth policy. §6.4 requires a correct fixed baseline before adaptive."""

    depth: int = 2

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("recurrence depth must be >= 1")


class IridiumCore(nn.Module):
    """Dense prelude, weight-tied recurrent core, dense coda."""

    def __init__(self, cfg: TransformerConfig, max_recurrence: int = 8) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_recurrence = max_recurrence
        self.prelude = nn.ModuleList(Block(cfg) for _ in range(cfg.n_prelude))
        self.core = nn.ModuleList(Block(cfg) for _ in range(cfg.n_core))
        self.coda = nn.ModuleList(Block(cfg) for _ in range(cfg.n_coda))
        self.norm_out = RMSNorm(cfg.d_model)
        # Conditioning on the recurrence index, so the tied core knows which
        # pass it is on. Without this a weight-tied stack cannot behave
        # differently at different depths.
        self.recurrence_embed = nn.Embedding(max_recurrence + 1, cfg.d_model)
        # Reinjection gate for H0 (§6.1).
        self.h0_gate = nn.Parameter(torch.zeros(cfg.d_model))
        self.halt_head = nn.Linear(cfg.d_model, 1)

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
        policy: RecurrencePolicy | None = None,
        cache: dict | None = None,
        collect_halt: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        policy = policy or RecurrencePolicy()
        if policy.depth > self.max_recurrence:
            raise ValueError(
                f"depth {policy.depth} exceeds max_recurrence {self.max_recurrence}"
            )

        for i, block in enumerate(self.prelude):
            h = block(h, mask, cache, ("pre", i) if cache is not None else None)

        h0 = h
        lambdas: list[torch.Tensor] = []
        for r in range(policy.depth):
            h = h + torch.sigmoid(self.h0_gate) * h0
            h = h + self.recurrence_embed(
                torch.full(h.shape[:2], r, dtype=torch.long, device=h.device)
            )
            for i, block in enumerate(self.core):
                key = ("core", r, i) if cache is not None else None
                h = block(h, mask, cache, key)
            if collect_halt:
                lambdas.append(torch.sigmoid(self.halt_head(h.mean(dim=1))).squeeze(-1))

        for i, block in enumerate(self.coda):
            h = block(h, mask, cache, ("post", i) if cache is not None else None)
        return self.norm_out(h), lambdas

    def parameter_inventory(self) -> dict[str, int]:
        """Count parameters by role so the checkpoint manifest is auditable."""
        counts: dict[str, int] = {}
        for name, param in self.named_parameters():
            role = name.split(".")[0]
            counts[role] = counts.get(role, 0) + param.numel()
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["transformer_matmul_only"] = sum(
            p.numel()
            for n, p in self.named_parameters()
            if p.dim() == 2 and n.split(".")[0] in {"prelude", "core", "coda"}
        )
        return counts


NEG_INF = torch.finfo(torch.float32).min


def causal_mask(length: int, device=None) -> torch.Tensor:
    """Standard causal mask for committed symbolic output."""
    m = torch.full((length, length), NEG_INF, device=device)
    return torch.triu(m, diagonal=1).view(1, 1, length, length)


def output_block_mask(
    prefix_len: int, block_len: int, device=None
) -> torch.Tensor:
    """Mask for a noised continuous output block (§6.3).

    The block attends bidirectionally *within itself* and to its conditioning
    prefix. The prefix stays causal. Nothing attends forward out of the block,
    so a clean future target can never leak in.
    """
    total = prefix_len + block_len
    mask = torch.full((total, total), NEG_INF, device=device)
    # Causal prefix.
    for i in range(prefix_len):
        mask[i, : i + 1] = 0.0
    # Block sees the whole prefix and all of itself.
    mask[prefix_len:, :total] = 0.0
    return mask.view(1, 1, total, total)


def workspace_mask(stream_ids: torch.Tensor, n_shared_prefix: int = 0) -> torch.Tensor:
    """Block-diagonal mask isolating concurrent workspaces (§8.1).

    Positions ``< n_shared_prefix`` are immutable shared records readable by
    every stream. Everything else is visible only within its own stream.

    §8.1 is explicit that this mask is *necessary but not sufficient* for
    isolation: cache allocation, memory retrieval, logging, object storage and
    action handles need their own enforcement. This function covers attention
    only.
    """
    t = stream_ids.shape[0]
    same = stream_ids.view(t, 1) == stream_ids.view(1, t)
    if n_shared_prefix:
        shared = torch.zeros(t, dtype=torch.bool, device=stream_ids.device)
        shared[:n_shared_prefix] = True
        same = same | shared.view(1, t)
    mask = torch.where(
        same, torch.zeros(t, t, device=stream_ids.device), torch.full((t, t), NEG_INF)
    )
    return mask.view(1, 1, t, t)
