"""The control core: the only component that sees the whole stream.

Two stages and a loop.

* **Stage I** (layers ``0 .. split-1``) builds the representation the router
  reads. It owns the canonical causal KV cache.
* **Stage II** (layers ``split .. L-1``) integrates whatever came back from the
  superstacks and decides whether the token is finished.
* **The ponder loop** re-enters at ``loop_entry`` with a loop-index embedding
  and the original stage input re-injected through a learned gate, so a second
  pass refines the observation instead of drifting away from it.

The cache is keyed by ``(stage, loop_index, layer)``. A layer executed twice by
the loop occupies two cache slots, because it is two different conditional
distributions — the second pass sees a residual stream the first one had not
produced yet. Collapsing them into one slot is the subtle cache bug that makes
a recurrent model's sampled output disagree with its teacher-forced output, and
``tests/integration/test_kv_parity.py`` exists to catch it.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import CoreConfig
from .layers import RMSNorm, TransformerBlock, causal_keep
from .rope import RotaryEmbedding


class ControlCore(nn.Module):
    def __init__(self, cfg: CoreConfig, rope: RotaryEmbedding, max_loops: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_loops = max_loops
        self.layers = nn.ModuleList(
            TransformerBlock(
                cfg.d_model,
                cfg.n_query_heads,
                cfg.n_kv_heads,
                cfg.d_head,
                cfg.d_ff,
                rope,
                cfg.norm_eps,
            )
            for _ in range(cfg.n_layers)
        )
        self.out_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        # Re-injection gate for the loop entry state (sigmoid(0) = 0.5).
        self.loop_gate = nn.Parameter(torch.zeros(cfg.d_model))
        self.loop_halt_head = nn.Linear(cfg.d_model, 1)
        nn.init.zeros_(self.loop_halt_head.bias)

    @property
    def split(self) -> int:
        return self.cfg.split

    def _run(
        self,
        h: torch.Tensor,
        positions: torch.Tensor,
        keep: torch.Tensor,
        layer_range: range,
        loop_index: int,
        cache: Optional[dict],
    ) -> torch.Tensor:
        for i in layer_range:
            key = ("core", loop_index, i) if cache is not None else None
            h = self.layers[i](h, positions, keep, cache, key)
        return h

    def stage_one(
        self,
        h: torch.Tensor,
        positions: torch.Tensor,
        keep: torch.Tensor,
        loop_index: int = 0,
        cache: Optional[dict] = None,
        start: int = 0,
    ) -> torch.Tensor:
        return self._run(h, positions, keep, range(start, self.split), loop_index, cache)

    def stage_two(
        self,
        h: torch.Tensor,
        positions: torch.Tensor,
        keep: torch.Tensor,
        loop_index: int = 0,
        cache: Optional[dict] = None,
    ) -> torch.Tensor:
        h = self._run(
            h, positions, keep, range(self.split, self.cfg.n_layers), loop_index, cache
        )
        return h

    def reinject(self, h: torch.Tensor, entry: torch.Tensor) -> torch.Tensor:
        return h + torch.sigmoid(self.loop_gate) * entry

    def halt_logit(self, h: torch.Tensor) -> torch.Tensor:
        return self.loop_halt_head(self.out_norm(h)).squeeze(-1)

    def finalize(self, h: torch.Tensor) -> torch.Tensor:
        return self.out_norm(h)

    @staticmethod
    def stream_keep(
        n_query: int, n_key: int, device=None, valid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        keep = causal_keep(n_query, n_key, device)
        if valid is not None:
            keep = keep & valid.view(valid.shape[0], 1, 1, -1)
        return keep
