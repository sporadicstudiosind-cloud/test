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
import torch.utils.checkpoint as torch_checkpoint

from ..config import CoreConfig
from .layers import RMSNorm, TransformerBlock, causal_keep
from .rope import RotaryEmbedding


class ControlCore(nn.Module):
    def __init__(self, cfg: CoreConfig, rope: RotaryEmbedding, max_loops: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_loops = max_loops
        from .core_blocks import build_core_block
        # MLA rotates only an ``mla_rope_dim``-wide slice, so it needs a table
        # of that width; handing it the core's ``d_head`` table would rotate
        # the wrong number of channels whenever the two differ.
        mla_rope = (RotaryEmbedding(cfg.mla_rope_dim, cfg.rope_theta)
                    if "mla" in cfg.layer_kinds() and cfg.mla_rope_dim != cfg.d_head
                    else rope)
        self.layers = nn.ModuleList(
            build_core_block(cfg, mla_rope if kind == "mla" else rope, i)
            for i, kind in enumerate(cfg.layer_kinds())
        )
        if cfg.hyper_streams > 1:
            from .residual import HyperConnections, StreamCollapse
            self.hyper = nn.ModuleList(
                HyperConnections(cfg.d_model, cfg.hyper_streams, constrained=True,
                                 sinkhorn_iters=cfg.hyper_sinkhorn_iters,
                                 dynamic=cfg.hyper_dynamic)
                for _ in range(cfg.n_layers)
            )
            self.hyper_collapse = StreamCollapse(cfg.d_model, cfg.hyper_streams,
                                                 dynamic=cfg.hyper_dynamic)
        else:
            self.hyper = None
        self.out_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        # Re-injection gate for the loop entry state (sigmoid(0) = 0.5).
        self.loop_gate = nn.Parameter(torch.zeros(cfg.d_model))
        self.loop_halt_head = nn.Linear(cfg.d_model, 1)
        nn.init.zeros_(self.loop_halt_head.bias)
        # See Iridium1.enable_gradient_checkpointing for what this does and
        # does not cover, and the measurements behind why it defaults off.
        self.gradient_checkpointing = False

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
        layer_bias=None,
    ) -> torch.Tensor:
        # ``layer_bias(i)``, when given, is added to layer ``i``'s input: the
        # per-layer embeddings (Gemma 3n PLE). It depends only on each token's
        # own id, never on history, so it needs nothing from the cache.
        # Checkpointing recomputes each layer during backward, which would
        # write its cache entry a second time if a KV cache were live here.
        # cache is only non-None during incremental serving, where there is
        # no backward pass anyway, so this excludes nothing checkpointing
        # could otherwise help with.
        use_checkpoint = (
            self.gradient_checkpointing
            and self.training
            and cache is None
            and h.requires_grad
            and torch.is_grad_enabled()
        )
        def apply(i: int, x: torch.Tensor) -> torch.Tensor:
            key = ("core", loop_index, i) if cache is not None else None
            if layer_bias is not None:
                x = x + layer_bias(i)
            if use_checkpoint:
                layer = self.layers[i]

                def run_layer(hh: torch.Tensor, layer: nn.Module = layer) -> torch.Tensor:
                    return layer(hh, positions, keep, None, None)

                return torch_checkpoint.checkpoint(
                    run_layer, x, use_reentrant=False, preserve_rng_state=True,
                )
            return self.layers[i](x, positions, keep, cache, key)

        if self.hyper is None:
            for i in layer_range:
                h = apply(i, h)
            return h

        # mHC: n residual streams for the duration of this run of layers.
        # Each block is a full-residual function (it adds its own input), so
        # it is wrapped with delta=False and only its increment is written
        # back into the streams. The per-layer closure is what keeps each
        # layer's KV-cache key its own -- one shared *args tuple would give
        # every layer the same key. Stream mixing is computed per token from
        # that token's own state, so cached decoding stays exact.
        from .residual import expand_streams
        if len(layer_range) == 0:
            return h
        streams = expand_streams(h, self.cfg.hyper_streams)
        for i in layer_range:
            streams = self.hyper[i](streams, lambda x, i=i: apply(i, x), delta=False)
        return self.hyper_collapse(streams)

    def stage_one(
        self,
        h: torch.Tensor,
        positions: torch.Tensor,
        keep: torch.Tensor,
        loop_index: int = 0,
        cache: Optional[dict] = None,
        start: int = 0,
        layer_bias=None,
    ) -> torch.Tensor:
        return self._run(h, positions, keep, range(start, self.split), loop_index, cache,
                         layer_bias)

    def stage_two(
        self,
        h: torch.Tensor,
        positions: torch.Tensor,
        keep: torch.Tensor,
        loop_index: int = 0,
        cache: Optional[dict] = None,
        layer_bias=None,
    ) -> torch.Tensor:
        h = self._run(
            h, positions, keep, range(self.split, self.cfg.n_layers), loop_index, cache,
            layer_bias,
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
