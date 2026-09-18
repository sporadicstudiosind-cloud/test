"""Iridium: one native multimodal model with routed subject stacks.

Legacy mode splits the core around the bank. Controller mode executes the full
core on every cycle, optionally dispatches specialists, then returns their states
to the next core cycle for integration. Emission is chosen only from core states.
The last budgeted cycle cannot dispatch work that has no return/integration pass.

Training scores a learned distribution over stopping cycles. Inference selects
the first confident stop, with a forced final-budget stop. Both still unroll core
cycles to maintain dense cache histories: this is NOT compute-saving per-token
halting. Learned input memory is causal and optional. LongContextSession is a
separate lossy serving path whose long-range recall must be trained and measured.
This revision has not been executed or tested.
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
        self.codecs.spatial_coordinates = cfg.controller_mode
        if cfg.perception_layers:
            from .perception import PerceptualEncoder
            for name, encoder in list(self.codecs.encoders.items()):
                self.codecs.encoders[name] = PerceptualEncoder(encoder, cfg.core.d_model,
                                                              cfg.perception_rank, cfg.perception_layers)
        self.core = ControlCore(cfg.core, self.rope, cfg.router.max_loops)
        self.router = MacroRouter(
            cfg.core.d_model,
            cfg.stacks.n_stacks,
            cfg.router,
            cfg.stacks.min_depth,
            cfg.stacks.n_layers,
        )
        self.bank = SuperstackBank(cfg, self.rope)
        if cfg.controller_mode:
            self.core.dispatch_head = nn.Linear(cfg.core.d_model, 1)
            nn.init.constant_(self.core.dispatch_head.bias, 1.)
            nn.init.constant_(self.core.loop_halt_head.bias, -2.)
        from .context_memory import ContextMemory
        self.context_memory = ContextMemory(cfg.core.d_model, cfg.memory_slots,
                                            cfg.memory_stride, cfg.memory_rank) if cfg.memory_slots else None
        self.bank_gate = nn.Parameter(torch.zeros(cfg.core.d_model)) if cfg.gated_bank else None
        from .layers import GroupedQueryAttention, BridgeCrossAttention
        for module in self.modules():
            if isinstance(module, (GroupedQueryAttention, BridgeCrossAttention)):
                module.qk_norm = cfg.qk_norm
        # Scale residual-producing projections as depth grows. Existing checkpoint
        # weights override this initialization on load.
        import math
        with torch.no_grad():
            for module in self.modules():
                from .layers import TransformerBlock, BridgeCrossAttention
                if isinstance(module, TransformerBlock):
                    depth = cfg.core.n_layers + cfg.stacks.n_layers
                    std = 1.0 / math.sqrt(2 * depth * cfg.core.d_model)
                    nn.init.normal_(module.attn.wo.weight, std=std)
                    nn.init.normal_(module.ffn.down.weight, std=std)
                elif isinstance(module, BridgeCrossAttention):
                    nn.init.normal_(module.wo.weight, std=1.0 / math.sqrt(
                        2 * cfg.stacks.n_layers * cfg.stacks.d_model))

    # -- introspection ----------------------------------------------------

    def parameter_inventory(self) -> dict[str, int]:
        groups = {"codecs": self.codecs, "core": self.core, "router": self.router,
                  "superstacks": self.bank}
        if self.context_memory is not None:
            groups["context_memory"] = self.context_memory
        counts = {k: sum(p.numel() for p in m.parameters()) for k, m in groups.items()}
        if self.bank_gate is not None:
            counts["bank_gate"] = self.bank_gate.numel()
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
        if n_loops is None:
            n_loops = min(3, self.cfg.router.max_loops) if self.cfg.controller_mode else 1
        if n_loops < 1:
            raise ValueError("n_loops must be positive")
        if n_loops > self.cfg.router.max_loops:
            raise ValueError(
                f"n_loops {n_loops} exceeds max_loops {self.cfg.router.max_loops}"
            )

        h = self.codecs.embed(batch) if embedded is None else embedded
        memory_loss = h.sum() * 0
        if self.context_memory is not None:
            memory_state = cache.get(("context", "state")) if cache is not None else None
            h, memory_state, memory_loss = self.context_memory(h, batch.valid, memory_state)
            if cache is not None:
                cache[("context", "state")] = memory_state.detach() if not self.training else memory_state
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

        stats["memory_reconstruction"] = memory_loss
        stats["core_dispatch"] = []
        stats["subject_loss"] = h.sum() * 0
        for loop in range(n_loops):
            start = 0 if loop == 0 else self.cfg.router.loop_entry
            h1 = (self.core._run(h, positions, keep, range(self.cfg.core.n_layers), loop, cache)
                  if self.cfg.controller_mode else self.core.stage_one(h, positions, keep, loop, cache, start))

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
                valid=batch.valid,
            )
            if self.cfg.controller_mode:
                dispatch = torch.sigmoid(self.core.dispatch_head(self.core.finalize(h1)).float()).squeeze(-1)
                selected = (dispatch >= .5) & batch.valid
                # Full unroll preserves dense per-cycle core cache histories.
                # No dispatch on the final cycle: results must return to the core.
                if loop == n_loops - 1:
                    selected = torch.zeros_like(selected)
                decision.valid = selected
                straight_through = selected.to(dispatch.dtype) + dispatch - dispatch.detach()
                decision.stack_weight = decision.stack_weight * straight_through[..., None]
                stats["core_dispatch"].append(dispatch)
            # Real subject labels supervise domain selection; they are optional.
            # During specialist-only training they also force the selected domain.
            forced = getattr(self, "training_subject", None)
            if forced is not None:
                # Router gather backward retains its original index tensor.
                # Do not mutate that tensor when imposing curriculum routes.
                decision.stack_index = decision.stack_index.clone()
                decision.stack_weight = decision.stack_weight.clone()
                decision.valid = decision.valid.clone() if decision.valid is not None else batch.valid.clone()
            for row, metadata in enumerate(batch.meta):
                subject = metadata.get("subject")
                names = self.cfg.stacks.specializations
                if subject in names:
                    index = names.index(subject)
                    mask = batch.valid[row]
                    probs = decision.gate_probs[row, :, index].float().clamp_min(1e-8)
                    stats["subject_loss"] = stats["subject_loss"] - probs[mask].log().mean() / len(batch.meta)
                    if self.cfg.controller_mode and loop < n_loops - 1:
                        # Labelled specialist work also teaches the dispatch
                        # head; otherwise a collapsed hard bypass has no task
                        # gradient with which to discover an unused specialist.
                        stats["subject_loss"] = stats["subject_loss"] - dispatch[row][mask].float().clamp_min(1e-8).log().mean() / len(batch.meta)
                    if forced == subject and (not self.cfg.controller_mode or loop < n_loops - 1):
                        decision.stack_index[row] = index
                        decision.stack_weight[row] = 1. / decision.stack_index.shape[-1]
                        decision.valid[row] = mask
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

            if self.bank_gate is not None:
                # A per-channel learned integration strength; gate starts at 0.5.
                stack_out = stack_out * torch.sigmoid(self.bank_gate)
            h2 = h1 if self.cfg.controller_mode else self.core.stage_two(h1 + stack_out, positions, keep, loop, cache)
            per_loop.append(self.core.finalize(h2))
            halt_logits.append(self.core.halt_logit(h2))
            h = self.core.reinject(h2 + stack_out if self.cfg.controller_mode else h2, entry)
            if self.cfg.loop_identity:
                h = h + self.router.loop_embed.weight[loop + 1]

        lam = torch.sigmoid(torch.stack(halt_logits, dim=-1).to(torch.float64 if h.dtype == torch.float64 else torch.float32))
        lam = torch.cat([lam[..., :-1], torch.ones_like(lam[..., -1:])], dim=-1)
        loop_p = stopping_distribution(lam)
        stacked = torch.stack(per_loop, dim=-1)                  # [B, T, d, L]
        mixed = (stacked * loop_p.unsqueeze(-2)).sum(-1)
        if self.cfg.controller_mode and not self.training:
            # Select a completed core state while still maintaining later caches.
            stops = lam >= exit_threshold
            stops[..., -1] = True
            first = stops.to(torch.int64).argmax(-1)
            mixed = stacked.gather(-1, first[..., None, None].expand(-1, -1, stacked.shape[-2], 1)).squeeze(-1)
            stats["chosen_cycle"] = first + 1
        stats["subject_loss"] = stats["subject_loss"] / n_loops

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
        for index, hidden in enumerate(out.per_loop_hidden):
            per_loop_losses.append(self.codecs.losses(
                hidden, batch, generator, token_weight=out.loop_stopping[..., index]))
        # Each token's error is weighted by that token's own stopping probability.
        # Averaging stopping probabilities across the batch first loses this link.
        keys = per_loop_losses[0].keys()
        losses = {k: sum(d[k] for d in per_loop_losses) for k in keys}
        losses["memory_reconstruction"] = .01 * out.stats["memory_reconstruction"]
        losses["subject_routing"] = .1 * out.stats["subject_loss"]
        losses["router_balance"] = out.stats["balance_loss"]
        losses["router_z"] = out.stats["z_loss"]
        losses["depth_kl"] = self.cfg.router.depth_beta * out.stats["depth_kl"]
        losses["loop_kl"] = self.cfg.router.ponder_beta * out.stats["loop_kl"]
        return losses, out
