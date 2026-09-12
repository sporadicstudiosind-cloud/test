"""The codec bank: input projections and output heads for every modality.

One embedding space, ``d_model`` wide. Text ids and action opcodes come in
through embeddings; continuous modalities come in through a linear patch
projection. Every token additionally carries a modality embedding, so the core
always knows what kind of thing it is looking at — a field patch and an image
patch of identical width are not interchangeable and the model should never
have to infer which it received.

Output is symmetric. Discrete modalities get softmax heads, continuous ones get
a flow-matching head (default) or a regression head (the honest blurry
baseline), and a slot-type head predicts the modality of the next position so
generation can run free.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CodecConfig
from ..model.heads import (
    ActionHead,
    ConfidenceHead,
    FlowMatchingHead,
    RegressionHead,
    SlotTypeHead,
    TextHead,
)
from .spans import CONTINUOUS, MODALITIES, MODALITY_INDEX, Batch


def continuous_dims(cfg: CodecConfig) -> dict[str, int]:
    return {
        "image": cfg.image_channels * cfg.image_patch ** 2,
        "video": cfg.image_channels * cfg.video_patch_t * cfg.image_patch ** 2,
        "audio": cfg.audio_mels * cfg.audio_frames,
        "field": cfg.field_channels * cfg.field_patch ** 2,
        "geometry": cfg.point_features,
    }


class TensorBatch:
    """A :class:`~iridium.codecs.spans.Batch` moved onto a torch device."""

    def __init__(self, batch: Batch, device=None, dtype: torch.dtype | None = None) -> None:
        dtype = dtype or torch.get_default_dtype()
        t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
        self.modality = t(batch.modality, torch.long)
        self.discrete = t(batch.discrete, torch.long)
        self.scalars = t(batch.scalars, dtype)
        self.positions = t(batch.positions, torch.long)
        self.valid = t(batch.valid, torch.bool)
        self.supervised = t(batch.supervised, torch.bool)
        self.span_id = t(batch.span_id, torch.long)
        self.continuous = {k: t(v, dtype) for k, v in batch.continuous.items()}
        self.grids = list(batch.grids)
        self.meta = list(batch.meta)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.modality.shape)  # type: ignore[return-value]


class CodecBank(nn.Module):
    def __init__(self, cfg: CodecConfig, d_model: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.dims = continuous_dims(cfg)

        self.text_embedding = nn.Embedding(cfg.vocab_size, d_model)
        self.modality_embedding = nn.Embedding(cfg.n_modalities, d_model)
        nn.init.normal_(self.text_embedding.weight, std=0.02)
        nn.init.normal_(self.modality_embedding.weight, std=0.02)

        self.encoders = nn.ModuleDict(
            {name: nn.Linear(dim, d_model) for name, dim in self.dims.items()}
        )
        self.action_encoder = nn.Linear(
            cfg.action_ops + cfg.action_scalars, d_model
        )

        head_cls = FlowMatchingHead if cfg.continuous_head == "flow" else RegressionHead
        if cfg.continuous_head == "flow":
            self.decoders = nn.ModuleDict(
                {
                    name: FlowMatchingHead(
                        d_model, dim, n_tau=cfg.flow_tau_features
                    )
                    for name, dim in self.dims.items()
                }
            )
        else:
            self.decoders = nn.ModuleDict(
                {name: RegressionHead(d_model, dim) for name, dim in self.dims.items()}
            )
        self.head_kind = cfg.continuous_head

        self.text_head = TextHead(
            d_model,
            cfg.vocab_size,
            self.text_embedding if cfg.tie_text_embedding else None,
        )
        self.action_head = ActionHead(d_model, cfg.action_ops, cfg.action_scalars)
        self.slot_type_head = SlotTypeHead(d_model, cfg.n_modalities)
        self.confidence_head = ConfidenceHead(d_model)

    # -- input ------------------------------------------------------------

    def embed(self, batch: TensorBatch) -> torch.Tensor:
        b, t = batch.modality.shape
        h = self.modality_embedding(batch.modality)

        text_mask = batch.modality == MODALITY_INDEX["text"]
        ctrl_mask = batch.modality == MODALITY_INDEX["control"]
        discrete_mask = text_mask | ctrl_mask
        if bool(discrete_mask.any()):
            ids = batch.discrete.clamp(0, self.cfg.vocab_size - 1)
            h = h + self.text_embedding(ids) * discrete_mask.unsqueeze(-1)

        act_mask = batch.modality == MODALITY_INDEX["action"]
        if bool(act_mask.any()):
            ops = F.one_hot(
                batch.discrete.clamp(0, self.cfg.action_ops - 1), self.cfg.action_ops
            ).to(h.dtype)
            act_in = torch.cat([ops, batch.scalars], dim=-1)
            h = h + self.action_encoder(act_in) * act_mask.unsqueeze(-1)

        for name in CONTINUOUS:
            mask = batch.modality == MODALITY_INDEX[name]
            if not bool(mask.any()):
                continue
            values = batch.continuous.get(name)
            if values is None:
                continue
            h = h + self.encoders[name](values) * mask.unsqueeze(-1)

        return h * batch.valid.unsqueeze(-1)

    # -- output -----------------------------------------------------------

    def next_slot_targets(self, batch: TensorBatch) -> dict[str, torch.Tensor]:
        """Shift everything left by one: position ``t`` predicts position ``t+1``."""
        shift = lambda x: x[:, 1:]  # noqa: E731
        out = {
            "modality": shift(batch.modality),
            "discrete": shift(batch.discrete),
            "scalars": shift(batch.scalars),
            "valid": shift(batch.valid) & shift(batch.supervised),
        }
        out.update(
            {f"continuous.{k}": shift(v) for k, v in batch.continuous.items()}
        )
        return out

    def losses(
        self,
        hidden: torch.Tensor,
        batch: TensorBatch,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        """Per-modality next-slot losses, each normalized by its own token count."""
        h = hidden[:, :-1]
        tgt = self.next_slot_targets(batch)
        valid = tgt["valid"]
        losses: dict[str, torch.Tensor] = {}
        zero = hidden.new_zeros(())

        text_mask = valid & (
            (tgt["modality"] == MODALITY_INDEX["text"])
            | (tgt["modality"] == MODALITY_INDEX["control"])
        )
        if bool(text_mask.any()):
            logits = self.text_head(h)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tgt["discrete"].reshape(-1).clamp(0, self.cfg.vocab_size - 1),
                reduction="none",
            ).view_as(text_mask)
            losses["text"] = (ce * text_mask).sum() / text_mask.sum().clamp_min(1)
        else:
            losses["text"] = zero

        act_mask = valid & (tgt["modality"] == MODALITY_INDEX["action"])
        if bool(act_mask.any()):
            op_logits, scalar_pred = self.action_head(h)
            ce = F.cross_entropy(
                op_logits.reshape(-1, op_logits.shape[-1]),
                tgt["discrete"].reshape(-1).clamp(0, self.cfg.action_ops - 1),
                reduction="none",
            ).view_as(act_mask)
            losses["action_op"] = (ce * act_mask).sum() / act_mask.sum().clamp_min(1)
            se = (scalar_pred - tgt["scalars"]).pow(2).mean(-1)
            losses["action_scalar"] = (se * act_mask).sum() / act_mask.sum().clamp_min(1)
        else:
            losses["action_op"] = zero
            losses["action_scalar"] = zero

        for name in CONTINUOUS:
            mask = valid & (tgt["modality"] == MODALITY_INDEX[name])
            key = f"continuous.{name}"
            if not bool(mask.any()) or key not in tgt:
                losses[name] = zero
                continue
            losses[name] = self.decoders[name].loss(
                h, tgt[key], mask.to(h.dtype), **({"generator": generator} if self.head_kind == "flow" else {})
            )

        slot_logits = self.slot_type_head(h)
        ce = F.cross_entropy(
            slot_logits.reshape(-1, slot_logits.shape[-1]),
            tgt["modality"].reshape(-1),
            reduction="none",
        ).view_as(valid)
        losses["slot_type"] = (ce * valid).sum() / valid.sum().clamp_min(1)
        return losses

    @torch.no_grad()
    def decode_continuous(
        self, hidden: torch.Tensor, modality: str, steps: int = 16,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        head = self.decoders[modality]
        if isinstance(head, FlowMatchingHead):
            return head.sample(hidden, steps=steps, generator=generator)
        return head(hidden)
