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
from ..codecs.spans import CONDITIONING, MODALITY_INDEX, MODALITIES, Sample, collate
from .device import device_of, generator_for


@dataclass
class Generated:
    text: str = ""
    ids: list[int] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    continuous: list[tuple[str, np.ndarray]] = field(default_factory=list)
    focus: list[float] = field(default_factory=list)
    loops: list[float] = field(default_factory=list)
    stopped: str = "length"
    actions: list[dict] = field(default_factory=list)

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
    # A generated token is placed on the 1-D diagonal (p, p, p). For
    # text that is exact; for generated media it is not the grid
    # position training used, which needs a layout-aware decoder.
    view.rope_positions = view.positions.unsqueeze(-1).expand(1, 1, 3).contiguous()
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
    top_p: float = 0.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    stop_ids: Sequence[int] = (2,),
    n_loops: Optional[int] = None,
    flow_steps: int = 8,
    allow_continuous: bool = False,
    seed: int = 0,
    text_offset: int = 16,
    text_only: bool = False,
    force_modality: Optional[str] = None,
    min_p: float = 0.0,
    tokenizer=None,
    thinking=None,
) -> Generated:
    """Greedy (``temperature=0``) or sampled continuation of ``sample``.

    See :func:`_pick` for what each truncation knob does and why ``min_p`` is
    the one to reach for first on an undertrained model.

    ``thinking`` (a :class:`~iridium.runtime.thinking.ThinkingBudget`) makes
    ponder depth adaptive per token: each step's loop cap and halting
    threshold come from the budget, which updates from the step's latency and
    the model's uncertainty. ``result.loops`` records the loops each token used.
    """
    import time as _time
    if thinking is not None:
        # The prompt is read at the budget's loop cap so every loop's cache
        # covers it; each generated token then halts wherever it is done.
        n_loops = thinking.plan(model.cfg.router.max_loops)[0]
    if n_loops is None:
        n_loops = min(3, model.cfg.router.max_loops) if model.cfg.controller_mode else 1
    if len(sample) < 1 or max_new_tokens < 1:
        raise ValueError("nonempty context and positive generation budget required")
    if len(sample) + max_new_tokens > model.cfg.max_seq_len:
        raise ValueError("context plus output exceeds max_seq_len; shorten/chunk the media")
    if force_modality is not None and force_modality not in MODALITY_INDEX:
        raise ValueError("unknown forced output modality")
    model.eval()
    dims = continuous_dims(model.cfg.codecs)
    device = device_of(model)
    batch = TensorBatch(collate([sample], dims), device=device)
    cache: dict = {}
    rng = generator_for(device, seed)

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
        if (not allow_continuous and name not in ("text", "control", "action")) \
                or name in CONDITIONING:
            # A camera is conditioning the model reads, never content it emits;
            # it has no decoder, so a slot head that predicts one is overruled.
            name = "text"
            slot = MODALITY_INDEX["text"]

        if text_only:
            name, slot = "text", MODALITY_INDEX["text"]
        if force_modality is not None:
            name, slot = force_modality, MODALITY_INDEX[force_modality]
        continuous_payload = None
        action_scalars = None
        if name in ("text", "control"):
            logits = codecs.text_head(h)[0, -1]
            token = _pick(logits, temperature, rng, top_p, top_k,
                          repetition_penalty, result.ids, min_p)
        elif name == "action":
            op_logits, action_scalars = codecs.action_head(h)
            token = _pick(op_logits[0, -1], temperature, rng, top_p, top_k,
                          min_p=min_p)
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
        if action_scalars is not None:
            step.scalars = action_scalars.to(step.scalars.dtype)
            result.actions.append({"op": token, "operands": action_scalars[0, 0].cpu().tolist()})
        if thinking is not None:
            step_loops, thr = thinking.plan(model.cfg.router.max_loops)
            step_loops = min(step_loops, n_loops)
            t0 = _time.perf_counter()
            out = model(step, n_loops=step_loops, cache=cache, halt_threshold=thr)
            used = int(out.stats.get("halted_at", step_loops))
            thinking.observe(logits if name in ("text", "control") else None,
                             (_time.perf_counter() - t0) * 1000, used)
            result.loops[-1] = float(used)
        else:
            out = model(step, n_loops=n_loops, cache=cache)
        hidden = out.hidden
        position += 1

    text_ids = [token - text_offset for token, name in zip(result.ids, result.modalities)
                if name == "text" and token >= text_offset]
    if tokenizer is not None:
        # Decode all ids at once so a character split across tokens survives.
        result.text = tokenizer.decode(text_ids)
    else:
        result.text = bytes(v for v in text_ids if v < 256).decode("utf-8", errors="replace")
    return result


def _pick(
    logits: torch.Tensor,
    temperature: float,
    rng: torch.Generator,
    top_p: float = 0.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    emitted: Optional[Sequence[int]] = None,
    min_p: float = 0.0,
) -> int:
    """Sample one token.

    Four knobs beyond temperature, and a small model needs all of them:

    * **repetition_penalty** divides the logit of anything already emitted.
      Undertrained networks fall into two-word cycles within a sentence, and
      no amount of temperature breaks a loop whose logit gap is large.
    * **top_k / top_p** cut the tail before sampling. Raising temperature
      without truncating makes the long tail of near-zero-probability bytes
      reachable, which is how "creative" turns into mojibake.
    * **min_p** cuts the tail *relative to the most likely token* — everything
      below ``min_p * p_max`` goes. This is the one that suits this model best
      and it is worth being precise about why, because it looks like a
      reparametrisation of top_p and is not. A fixed nucleus applies the same
      mass budget to every step, but the two kinds of step want opposite
      treatment: where the model is confident (mid-word, inside a number, after
      an opening brace) the correct next token holds nearly all the mass and
      top_p still admits a tail of plausible-looking alternatives that are
      simply wrong; where it is genuinely uncertain, the same budget truncates
      options it should keep. Scaling the floor by ``p_max`` makes the cut
      tighten exactly when the model is sure and loosen when it is not, which
      is why min_p tolerates far higher temperatures without degenerating. That
      matters more here than for most models: this one emits bytes or subwords
      into a *shared* discrete space with control and action slots, so a
      tail-sampled token is not merely a poor word choice, it can be a
      structurally invalid slot.

      Applied to the temperature-scaled distribution, as in the original
      formulation — applying it before temperature would make the threshold
      mean something different at every temperature setting and defeat the
      point.

    Greedy (``temperature <= 0``) ignores all of them by construction, and is
    still the right default for a grader that wants a reproducible answer.
    """
    logits = logits.float()
    if temperature <= 0:
        return int(logits.argmax(-1).item())

    if repetition_penalty != 1.0 and emitted:
        seen = torch.tensor(sorted(set(int(t) for t in emitted)),
                            device=logits.device, dtype=torch.long)
        seen = seen[seen < logits.shape[-1]]
        if seen.numel():
            scores = logits[seen]
            # Penalise toward zero from whichever side the logit sits on;
            # dividing a negative logit would *raise* it.
            logits[seen] = torch.where(
                scores > 0, scores / repetition_penalty, scores * repetition_penalty
            )

    logits = logits / temperature

    if top_k and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if 0.0 < top_p < 1.0:
        ordered, index = torch.sort(logits, descending=True)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        # Keep the first token whose cumulative mass crosses top_p, so the
        # nucleus is never empty even when one token holds all the mass.
        drop = cumulative - torch.softmax(ordered, dim=-1) >= top_p
        drop[0] = False
        ordered = ordered.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, index, ordered)

    if 0.0 < min_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        floor = min_p * float(probs.max())
        keep = probs >= floor
        # The argmax always clears its own floor, so `keep` cannot be empty --
        # but assert it rather than trust it, because an all -inf logit vector
        # (every candidate already masked by top_k/top_p) would make `max` NaN
        # and silently drop every token.
        if bool(keep.any()):
            logits = logits.masked_fill(~keep, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
        return int(logits.argmax(-1).item())
    return int(torch.multinomial(probs, 1, generator=rng).item())
