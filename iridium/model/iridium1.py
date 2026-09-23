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
        scaling = ({"type": "yarn", "factor": cfg.rope_yarn_factor,
                    "original_max_position": cfg.rope_original_max_position}
                   if cfg.rope_yarn_factor > 1.0 else None)
        self.rope = RotaryEmbedding(cfg.core.d_head, cfg.core.rope_theta, scaling=scaling,
                                    sections=tuple(cfg.mrope_sections) or None)
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
        if cfg.ple_dim:
            from .embeddings import PerLayerEmbedding
            self.ple = PerLayerEmbedding(cfg.codecs.vocab_size, cfg.core.n_layers,
                                         cfg.ple_dim, cfg.core.d_model)
        else:
            self.ple = None
        from .layers import GroupedQueryAttention, BridgeCrossAttention
        for module in self.modules():
            if isinstance(module, (GroupedQueryAttention, BridgeCrossAttention)):
                module.qk_norm = cfg.qk_norm
        # Scale residual-producing projections as depth grows. Existing checkpoint
        # weights override this initialization on load.
        import math
        with torch.no_grad():
            from .blocks import ParallelBlock
            from .core_blocks import output_projections
            for module in self.modules():
                from .layers import TransformerBlock, BridgeCrossAttention
                if isinstance(module, (TransformerBlock, ParallelBlock)):
                    depth = cfg.core.n_layers + cfg.stacks.n_layers
                    std = 1.0 / math.sqrt(2 * depth * cfg.core.d_model)
                    for proj in output_projections(module):
                        # A token-conditioned FFN's low-rank U starts at zero
                        # on purpose -- that is what makes it exactly a plain
                        # FFN at init -- so it is left alone; only its base
                        # projection takes the depth-scaled draw.
                        if proj is getattr(getattr(module.ffn, "down", None), "U", None):
                            continue
                        nn.init.normal_(proj.weight, std=std)
                elif isinstance(module, BridgeCrossAttention):
                    nn.init.normal_(module.wo.weight, std=1.0 / math.sqrt(
                        2 * cfg.stacks.n_layers * cfg.stacks.d_model))

    # -- introspection ----------------------------------------------------

    def parameter_inventory(self) -> dict[str, int]:
        groups = {"codecs": self.codecs, "core": self.core, "router": self.router,
                  "superstacks": self.bank}
        if self.context_memory is not None:
            groups["context_memory"] = self.context_memory
        if self.ple is not None:
            groups["per_layer_embedding"] = self.ple
        counts = {k: sum(p.numel() for p in m.parameters()) for k, m in groups.items()}
        if self.bank_gate is not None:
            counts["bank_gate"] = self.bank_gate.numel()
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    # -- forward ----------------------------------------------------------

    def enable_gradient_checkpointing(self, enabled: bool = True,
                                      stacks: bool = False) -> None:
        """Re-measured. The earlier "returns incorrect gradients" verdict was
        real but mis-attributed: it was PyTorch's *reentrant* checkpointing
        (``use_reentrant=True``, still the undocumented default when the flag
        is omitted), not gradient checkpointing per se. ``use_reentrant=False``
        is exact here; this method uses it unconditionally and does not expose
        the choice.

        What was actually going on, measured on the ``tiny`` config with the
        router's gumbel noise pinned by seed (so the stochastic path is
        identical between runs) and gradients compared to an unchecked
        reference at the same seed:

        * **The control core.** Checkpointing each core layer and comparing to
          the unchecked run gave *exact* (0.0 max-abs-delta) gradients with
          ``n_loops=1``, under both ``use_reentrant`` settings. The earlier
          "moved the modality embedding's gradient by 2.75" symptom only
          appears with ``n_loops=2`` — i.e. once the *same* core layer is
          checkpointed twice in one backward pass, once per ponder-loop
          iteration — and only under ``use_reentrant=True``: fp32 gives a
          reentrant/non-reentrant delta of 3.7e-4 (relative 1.4e-7) and bf16
          autocast gives 1.875 (relative 2.6e-3) on ``core.layers.0.attn.wo``.
          ``use_reentrant=False`` gives 0.0 in every one of these cases,
          including ``controller_mode`` True and False. Reentrant
          checkpointing's known restriction is exactly this: correctness is
          not guaranteed when the same checkpointed parameters are visited by
          more than one checkpoint call in a single backward. The ponder loop
          guarantees that whenever ``n_loops > 1``.
        * **Superstacks.** ``use_reentrant=True`` does not silently corrupt
          gradients here — it raises outright: ``RuntimeError: Trying to
          backward through the graph a second time``. Every active stack's
          bridge cross-attention reads the *same* upstream core-state tensor
          (see ``SuperstackBank.forward``'s ``bridge_by_device`` sharing), so
          two or more checkpointed stacks root their nested backward calls in
          a shared ancestor — unsupported by reentrant checkpointing without
          ``retain_graph=True``, which per-call checkpointing does not set.
          ``use_reentrant=False`` has no such restriction (it saves via hooks
          instead of nested ``autograd.backward`` calls) and reproduced the
          unchecked gradients exactly, including with ``hard_exit=True``,
          where each row's executed depth is a genuine function of activation
          values (PonderNet's ``alive`` mask) — the data-dependence the
          original docstring worried about lives entirely inside the Python
          loop that decides *how many* checkpoint calls to make, not inside
          any one checkpointed call, so recompute of a call that did happen
          always replays the same layer it originally ran.

        Both are therefore correct and both default off, gated separately
        (``enabled`` for the core, ``stacks`` additionally for the
        superstacks) so a training run opts in deliberately. Checkpointing is
        skipped whenever a KV cache is live (incremental serving): the
        segment would otherwise write its cache entry twice, once in the
        checkpoint's own no-grad forward and once in the backward recompute,
        and serving never runs backward anyway so there is nothing to gain
        there. See ``tests/unit/test_checkpointing.py`` for the gradient-
        equality regression test and the measured peak-memory saving on the
        ``nano`` config (CPU): modest but real for the core, and additionally
        real for the stacks once both are enabled together.

        ``TrainConfig.accumulate`` still is not a substitute: it cuts the
        batch dimension, this cuts the depth dimension, and a deep model
        trained at small batch is bounded by the latter.
        """
        self.core.gradient_checkpointing = enabled
        for stack in self.bank.stacks:
            stack.gradient_checkpointing = enabled and stacks

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

        h = self.codecs.embed(batch, cache) if embedded is None else embedded
        memory_loss = h.sum() * 0
        if self.context_memory is not None:
            memory_state = cache.get(("context", "state")) if cache is not None else None
            h, memory_state, memory_loss = self.context_memory(h, batch.valid, memory_state)
            if cache is not None:
                cache[("context", "state")] = memory_state.detach() if not self.training else memory_state
        positions = batch.positions
        b, t, _ = h.shape
        # The core alone reads grid positions (M-RoPE); its masks are
        # index-based, so a [B, T, 3] tensor passes through unchanged. The
        # router, superstacks, bridge and cache keep the scalar order.
        core_positions = positions
        if self.cfg.mrope_sections and getattr(batch, "rope_positions", None) is not None:
            core_positions = batch.rope_positions
        layer_bias = None
        if self.ple is not None:
            from ..codecs.spans import MODALITY_INDEX
            ple_ids = batch.discrete.clamp(0, self.cfg.codecs.vocab_size - 1)
            # Text positions only: a discrete id at an image or field slot is
            # not a token, and its "embedding" would be noise.
            ple_mask = (batch.modality == MODALITY_INDEX["text"]).unsqueeze(-1)
            layer_bias = lambda i: self.ple(ple_ids, i).to(h.dtype) * ple_mask  # noqa: E731

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
            h1 = (self.core._run(h, core_positions, keep, range(self.cfg.core.n_layers), loop,
                                 cache, layer_bias)
                  if self.cfg.controller_mode
                  else self.core.stage_one(h, core_positions, keep, loop, cache, start, layer_bias))

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
            h2 = h1 if self.cfg.controller_mode else self.core.stage_two(
                h1 + stack_out, core_positions, keep, loop, cache, layer_bias)
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
