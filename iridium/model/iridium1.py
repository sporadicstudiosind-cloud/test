"""Iridium-1: control core, macro-router, superstack bank, omnimodal codecs.

    tokens -> codecs -> core stage I -> router -> superstacks -> core stage II
                             ^                                        |
                             +------------- ponder loop --------------+
                                                                      |
                                                          heads -> emissions

One forward pass of the whole thing, with a cache that supports exact
incremental decoding.

The loop-granularity constraint
-------------------------------
The ponder loop raises the same coherence problem one level up from the
superstacks. If token ``t`` stops after one loop and token ``t+1`` takes two,
then at loop 2 the core's cache holds keys only for tokens that reached loop 2,
and ``t+1`` attends to a history with holes in it. Three ways out exist:

1. *copy-through*: a halted token's final state is re-projected at every later
   loop, so the history is dense. Correct, and costs a projection per halted
   token per loop.
2. *sparse loop history*: attention at loop ``L`` covers exactly the tokens that
   also reached loop ``L``. Cacheable and causal — this is what the superstacks
   do — but teacher forcing with uniform loops and sampling with variable loops
   then compute different functions.
3. *chunk-uniform loops*: every token in a chunk takes the same number of
   loops, chosen by the chunk's halting statistic.

This implementation uses **(3)**, and the halting distribution is still learned
per token and trained through the PonderNet objective, so the policy is a
serving choice rather than a training assumption. (1) is specified and not
implemented; it is listed as such in docs/capability-register.md rather than
quietly assumed to work. Choosing (2) silently is the failure this note exists
to prevent — it is the version that passes every unit test and then samples
differently from how it was trained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..codecs.bank import CodecBank, TensorBatch
from ..config import IridiumConfig
from .control_core import ControlCore
from .layers import causal_keep
from .rope import RotaryEmbedding
from .router import MacroRouter, RoutingDecision, geometric_prior, ponder_kl, stopping_distribution
from .superstack import SuperstackBank


@dataclass
class ModelOutput:
    hidden: torch.Tensor                      # [B, T, d] mixture or final state
    per_loop_hidden: list[torch.Tensor]
    loop_stopping: torch.Tensor               # [B, T, n_loops]
    decisions: list[RoutingDecision]
    stats: dict = field(default_factory=dict)

    @property
    def expected_loops(self) -> torch.Tensor:
        steps = torch.arange(
            1, self.loop_stopping.shape[-1] + 1,
            device=self.loop_stopping.device, dtype=self.loop_stopping.dtype,
        )
        return (self.loop_stopping * steps).sum(-1)


class Iridium1(nn.Module):
    def __init__(self, cfg: IridiumConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.rope = RotaryEmbedding(cfg.core.d_head, cfg.core.rope_theta)
        self.codecs = CodecBank(cfg.codecs, cfg.core.d_model)
        self.core = ControlCore(cfg.core, self.rope, cfg.router.max_loops)
        self.router = MacroRouter(
            cfg.core.d_model,
            cfg.stacks.n_stacks,
            cfg.router,
            cfg.stacks.min_depth,
            cfg.stacks.n_layers,
        )
        self.bank = SuperstackBank(cfg, self.rope)

    # -- introspection ----------------------------------------------------

    def parameter_inventory(self) -> dict[str, int]:
        groups = {"codecs": self.codecs, "core": self.core, "router": self.router,
                  "superstacks": self.bank}
        counts = {k: sum(p.numel() for p in m.parameters()) for k, m in groups.items()}
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    # -- forward ----------------------------------------------------------

    def enable_gradient_checkpointing(self, enabled: bool = True,
                                      stacks: bool = False) -> None:
        """Not available in this architecture. Measured, not assumed.

        Gradient checkpointing recomputes a layer's activations during the
        backward pass instead of storing them, and is normally free apart from
        the time. Here it is not correct, in two separate ways:

        * **Superstacks** raise ``CheckpointError: a different number of tensors
          was saved during the original forward and recomputation``. Their
          layers branch on data — which field grids survived routing, which
          tokens are still alive on the depth ladder — so the recomputed pass is
          a different graph.
        * **The control core** accepts checkpointing and silently returns
          *different gradients*: with the stochastic path pinned and the
          unchecked run reproducible to 0.0, enabling it moved the modality
          embedding's gradient by **2.75**. The loss is unchanged, so nothing
          in a training curve would ever show it.

        PyTorch's ``determinism_check="none"`` silences the first symptom and
        would have shipped the second. A wrong gradient degrades a run
        invisibly; an OOM at least announces itself. So this raises instead.

        The memory it would have saved is available elsewhere, and cheaply:
        fused attention removed the dominant allocation already, and
        ``TrainConfig.accumulate`` trades steps for peak memory exactly.
        """
        raise NotImplementedError(
            "gradient checkpointing returns incorrect gradients in this "
            "architecture (see the docstring; measured, not assumed). Use "
            "TrainConfig.accumulate to cut peak memory instead."
        )

    def forward(
        self,
        batch: TensorBatch,
        n_loops: Optional[int] = None,
        cache: Optional[dict] = None,
        hard_exit: bool = False,
        exit_threshold: float = 0.9,
        use_depth_cap: bool = False,
        embedded: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        n_loops = n_loops or 1
        if n_loops > self.cfg.router.max_loops:
            raise ValueError(
                f"n_loops {n_loops} exceeds max_loops {self.cfg.router.max_loops}"
            )

        h = self.codecs.embed(batch) if embedded is None else embedded
        positions = batch.positions
        b, t, _ = h.shape

        history = cache.get(("stream", "n"), 0) if cache is not None else 0
        keep = self._stream_keep(batch, history)
        core_states_cache = cache.get(("bridge", "states")) if cache is not None else None
        core_pos_cache = cache.get(("bridge", "positions")) if cache is not None else None

        entry = h
        summary: Optional[torch.Tensor] = None
        per_loop: list[torch.Tensor] = []
        halt_logits: list[torch.Tensor] = []
        decisions: list[RoutingDecision] = []
        stats: dict = {"stack_stats": [], "balance_loss": h.new_zeros(()),
                       "z_loss": h.new_zeros(()), "depth_kl": h.new_zeros(())}

        for loop in range(n_loops):
            start = 0 if loop == 0 else self.cfg.router.loop_entry
            h1 = self.core.stage_one(h, positions, keep, loop, cache, start)

            if loop == 0:
                # The focus summary, like the bridge states, is a loop-0
                # property of the stream. Recomputing it per loop would need
                # loop-L states for history tokens that never ran loop L.
                summary = self.router.prefix_summary(h1, 0, cache)
            decision = self.router(
                h1,
                loop_index=loop,
                span_id=getattr(batch, "span_id", None),
                summary=summary,
            )
            decisions.append(decision)

            # The bridge reads the core's *loop-0* stage-I states for the
            # whole stream, at every loop. Two reasons, and the second is the
            # one that has teeth:
            #
            # 1. semantically, the bridge is a stack's window onto the stream's
            #    perception, not onto the current refinement of it;
            # 2. mechanically, later loops only ever run for the tokens in the
            #    current chunk, so loop-L states do not exist for the history.
            #    Concatenating cached loop-0 history with fresh loop-L current
            #    states would make decoding compute something teacher forcing
            #    never computes, and the parity gate would fail — which is
            #    exactly how this was found.
            if loop == 0:
                if cache is not None:
                    bridge_states = (
                        h1 if core_states_cache is None
                        else torch.cat([core_states_cache, h1], dim=1)
                    )
                    bridge_positions = (
                        positions if core_pos_cache is None
                        else torch.cat([core_pos_cache, positions], dim=1)
                    )
                    cache[("bridge", "states")] = bridge_states
                    cache[("bridge", "positions")] = bridge_positions
                else:
                    bridge_states, bridge_positions = h1, positions

            stack_out, stack_stats = self.bank(
                h1,
                positions,
                decision,
                bridge_states,
                bridge_positions,
                batch.grids,
                cache,
                (loop,),
                hard_exit,
                exit_threshold,
                use_depth_cap,
            )
            stats["stack_stats"].append(stack_stats)
            stats["balance_loss"] = stats["balance_loss"] + decision.balance_loss
            stats["z_loss"] = stats["z_loss"] + decision.z_loss
            stats["depth_kl"] = stats["depth_kl"] + stack_stats["depth_kl"]

            h2 = self.core.stage_two(h1 + stack_out, positions, keep, loop, cache)
            per_loop.append(self.core.finalize(h2))
            halt_logits.append(self.core.halt_logit(h2))
            h = self.core.reinject(h2, entry)

        lam = torch.sigmoid(torch.stack(halt_logits, dim=-1))
        lam = torch.cat([lam[..., :-1], torch.ones_like(lam[..., -1:])], dim=-1)
        loop_p = stopping_distribution(lam)
        stacked = torch.stack(per_loop, dim=-1)                  # [B, T, d, L]
        mixed = (stacked * loop_p.unsqueeze(-2)).sum(-1)

        for key in ("balance_loss", "z_loss", "depth_kl"):
            stats[key] = stats[key] / n_loops
        prior = geometric_prior(
            n_loops, self.cfg.router.ponder_prior_p_stop, h.device, loop_p.dtype
        )
        stats["loop_kl"] = ponder_kl(loop_p, prior) if n_loops > 1 else h.new_zeros(())

        if cache is not None:
            cache[("stream", "n")] = history + t

        return ModelOutput(
            hidden=mixed,
            per_loop_hidden=per_loop,
            loop_stopping=loop_p,
            decisions=decisions,
            stats=stats,
        )

    def _stream_keep(self, batch: TensorBatch, history: int) -> torch.Tensor:
        b, t = batch.modality.shape
        keep = causal_keep(t, history + t, batch.modality.device)
        if history:
            past = torch.ones(b, history, dtype=torch.bool, device=batch.valid.device)
            valid = torch.cat([past, batch.valid], dim=1)
        else:
            valid = batch.valid
        return keep & valid.view(b, 1, 1, -1)

    # -- convenience ------------------------------------------------------

    def losses(
        self,
        batch: TensorBatch,
        n_loops: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[dict[str, torch.Tensor], ModelOutput]:
        """Per-loop task losses weighted by the stopping distribution.

        PonderNet's objective evaluates the task loss at *every* potential stop,
        which is what makes an early stop a trained behaviour instead of a
        truncation. Averaging the hidden states first and scoring once — ACT's
        recipe — is a different algorithm and is not what this is.
        """
        out = self.forward(batch, n_loops=n_loops)
        per_loop_losses: list[dict[str, torch.Tensor]] = []
        for hidden in out.per_loop_hidden:
            per_loop_losses.append(self.codecs.losses(hidden, batch, generator))

        keys = per_loop_losses[0].keys()
        weight = out.loop_stopping.mean(dim=(0, 1))              # [L]
        losses = {
            k: sum(w * d[k] for w, d in zip(weight, per_loop_losses)) for k in keys
        }
        losses["router_balance"] = out.stats["balance_loss"]
        losses["router_z"] = out.stats["z_loss"]
        losses["depth_kl"] = self.cfg.router.depth_beta * out.stats["depth_kl"]
        losses["loop_kl"] = self.cfg.router.ponder_beta * out.stats["loop_kl"]
        return losses, out
