"""Free-running generation: the model chooses what kind of thing comes next.

Discrete slots are sampled from the text/action heads; continuous slots are
integrated out of the flow-matching head. Which one runs is decided by the
model's own slot-type head, so generation is not steered by an external
controller deciding "now emit an image".

Every step goes through the same cache the parity gate covers, so what is
sampled here is what teacher forcing trained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from ..codecs.bank import TensorBatch, continuous_dims
from ..codecs.spans import MODALITY_INDEX, MODALITIES, Sample, collate


@dataclass
class Generated:
    text: str = ""
    ids: list[int] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    continuous: list[tuple[str, np.ndarray]] = field(default_factory=list)
    focus: list[float] = field(default_factory=list)
    loops: list[float] = field(default_factory=list)
    stopped: str = "length"

    @property
    def mean_focus(self) -> float:
        return float(np.mean(self.focus)) if self.focus else 0.0


def _single_token_batch(
    template: TensorBatch,
    modality: int,
    token_id: int,
    position: int,
    continuous: Optional[dict[str, torch.Tensor]] = None,
) -> TensorBatch:
    view = object.__new__(TensorBatch)
    device = template.modality.device
    dtype = template.scalars.dtype
    one = lambda v, dt: torch.tensor([[v]], dtype=dt, device=device)  # noqa: E731
    view.modality = one(modality, torch.long)
    view.discrete = one(token_id, torch.long)
    view.positions = one(position, torch.long)
    view.valid = one(True, torch.bool)
    view.supervised = one(True, torch.bool)
    view.span_id = one(-1, torch.long)
    view.scalars = torch.zeros(1, 1, template.scalars.shape[-1], dtype=dtype, device=device)
    view.continuous = {
        k: torch.zeros(1, 1, v.shape[-1], dtype=dtype, device=device)
        for k, v in template.continuous.items()
    }
    if continuous:
        for k, v in continuous.items():
            view.continuous[k] = v.view(1, 1, -1).to(dtype)
    view.grids = []
    view.meta = template.meta
    return view


@torch.no_grad()
def generate(
    model,
    sample: Sample,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    stop_ids: Sequence[int] = (2,),
    n_loops: int = 1,
    flow_steps: int = 8,
    allow_continuous: bool = False,
    seed: int = 0,
    text_offset: int = 16,
) -> Generated:
    """Greedy (``temperature=0``) or sampled continuation of ``sample``."""
    dims = continuous_dims(model.cfg.codecs)
    batch = TensorBatch(collate([sample], dims))
    cache: dict = {}
    rng = torch.Generator().manual_seed(seed)

    from .decode import atomic_chunks, slice_batch

    hidden = None
    for start, end in atomic_chunks(batch, int(batch.modality.shape[1])):
        out = model(slice_batch(batch, start, end), n_loops=n_loops, cache=cache)
        hidden = out.hidden
    assert hidden is not None
    position = int(batch.positions[0, -1]) + 1

    result = Generated()
    codecs = model.codecs
    for _ in range(max_new_tokens):
        h = hidden[:, -1:]
        slot = int(codecs.slot_type_head(h).argmax(-1).item())
        name = MODALITIES[slot]
        if not allow_continuous and name not in ("text", "control", "action"):
            name = "text"
            slot = MODALITY_INDEX["text"]

        continuous_payload = None
        if name in ("text", "control"):
            logits = codecs.text_head(h)[0, -1]
            token = _pick(logits, temperature, rng)
        elif name == "action":
            op_logits, _ = codecs.action_head(h)
            token = _pick(op_logits[0, -1], temperature, rng)
        else:
            token = 0
            value = codecs.decode_continuous(h, name, steps=flow_steps, generator=rng)
            continuous_payload = {name: value[0, -1]}
            result.continuous.append((name, value[0, -1].cpu().numpy()))

        result.ids.append(token)
        result.modalities.append(name)
        result.focus.append(float(out.decisions[0].focus[0, -1]))
        result.loops.append(float(out.expected_loops[0, -1]))
        if name in ("text", "control") and token in stop_ids:
            result.stopped = "eos"
            break

        step = _single_token_batch(batch, slot, token, position, continuous_payload)
        out = model(step, n_loops=n_loops, cache=cache)
        hidden = out.hidden
        position += 1

    body = bytearray()
    for token, name in zip(result.ids, result.modalities):
        if name == "text":
            v = token - text_offset
            if 0 <= v < 256:
                body.append(v)
    result.text = body.decode("utf-8", errors="replace")
    return result


def _pick(logits: torch.Tensor, temperature: float, rng: torch.Generator) -> int:
    if temperature <= 0:
        return int(logits.argmax(-1).item())
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, 1, generator=rng).item())
