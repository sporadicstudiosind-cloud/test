"""System-1 / System-2 routing: answer in one pass when the answer is a typed value.

Most of what a program asks a model is not an essay. It is "which of these five
categories", "is this safe to run", "what quantity is this", "fill these four
fields". Answering those by autoregressive decoding means generating text, one
full forward pass per token, and then parsing the text back into the value the
caller wanted -- paying for fluency nobody reads, and adding a parser that can
fail. TypeSafe's Jev (released 15 Sept 2026) is the clearest recent statement
of the alternative: a model that returns typed values with calibrated
confidence, produced in parallel in one query rather than token by token. Its
architecture is not published and nothing here claims to reproduce it. What is
built here is the mechanism it is an instance of, inside this model:

* **System 1** -- a :class:`~iridium.model.typed_head.TypedHead` reads the
  model's final hidden state after one prefill of the prompt and emits the
  whole typed value at once, every field in parallel, masked so an invalid
  value is unrepresentable, with a temperature-calibrated confidence and an
  explicit abstention output. One forward pass, regardless of how many fields
  the answer has.
* **System 2** -- ordinary autoregressive generation, with the ponder loop,
  superstacks and whatever reasoning budget the caller allows.

The router tries System 1 and escalates to System 2 only when the typed head
abstains or its confidence falls below the caller's floor. That escalation is
the entire point, and it is why the head must be *calibrated* rather than
merely accurate: an overconfident System 1 never escalates, so every one of its
mistakes ships; an underconfident one always escalates, so it saves nothing.
Calibration is what makes the confidence a usable routing signal rather than a
number that happens to be between zero and one.

Where the heads live, and why not in the model: a typed head is specific to one
schema and one task, and a deployment accumulates them. Registering them as
submodules of :class:`~iridium.model.iridium1.Iridium1` would make the model's
parameter set depend on which tasks were registered -- so a checkpoint would no
longer load into a model built from the same config, and the exact parameter
accounting in :mod:`iridium.config` would stop being true. They are held here,
saved and loaded with :meth:`System1Router.state_dict`, and read the model's
hidden state without modifying it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from ..codecs.bank import TensorBatch, continuous_dims
from ..codecs.spans import Sample, collate
from .device import device_of

__all__ = ["Answer", "System1Router", "encode_prompt"]


@dataclass
class Answer:
    """What the router returns, whichever path produced it."""

    route: str                      # "system1" | "system2"
    value: Any = None               # typed value (system1) or parsed value (system2)
    confidence: Optional[float] = None
    text: str = ""                  # system2's raw text, empty for system1
    reason: str = ""                # why this route was taken
    forward_passes: int = 0         # full-model forwards spent, for cost accounting
    field_confidence: dict = field(default_factory=dict)


@torch.no_grad()
def encode_prompt(model, sample: Sample, n_loops: Optional[int] = None) -> torch.Tensor:
    """Final-position hidden state after a cached, atomic-chunk prefill: ``[d]``.

    Uses exactly the prefill :func:`iridium.runtime.generate.generate` uses --
    same chunking, same cache, same loop count -- so the state System 1 reads is
    the state System 2 would begin decoding from. Reading a different state (an
    uncached forward, a different loop count) would train and calibrate the head
    on one distribution and route on another.
    """
    from .decode import atomic_chunks, slice_batch

    if n_loops is None:
        n_loops = min(3, model.cfg.router.max_loops) if model.cfg.controller_mode else 1
    model.eval()
    batch = TensorBatch(collate([sample], continuous_dims(model.cfg.codecs)),
                        device=device_of(model))
    cache: dict = {}
    hidden = None
    for start, end in atomic_chunks(batch, int(batch.modality.shape[1])):
        hidden = model(slice_batch(batch, start, end), n_loops=n_loops, cache=cache).hidden
    assert hidden is not None
    return hidden[0, -1]


class System1Router:
    """Holds typed heads for named tasks and routes each request between the paths.

    ``min_confidence`` is a floor on the *least* confident field: a struct
    answer with one shaky field is escalated whole, because a caller consuming
    a struct cannot use most of one. ``abstain_threshold`` is the head's own
    abstention cut. Both are routing policy, not model properties, so they are
    set here and can be tightened per deployment without retraining.
    """

    def __init__(self, model, abstain_threshold: float = 0.5,
                 min_confidence: float = 0.0,
                 generate_fn: Optional[Callable[..., Any]] = None) -> None:
        if not 0.0 <= min_confidence <= 1.0 or not 0.0 < abstain_threshold < 1.0:
            raise ValueError("confidence floor must lie in [0, 1], threshold in (0, 1)")
        self.model = model
        self.heads: nn.ModuleDict = nn.ModuleDict()
        self.abstain_threshold = abstain_threshold
        self.min_confidence = min_confidence
        if generate_fn is None:
            from .generate import generate as generate_fn
        self._generate = generate_fn

    # -- registration --------------------------------------------------------

    def register(self, name: str, schema, **head_kwargs):
        """Create (or replace) the typed head for task ``name``."""
        from ..model.typed_head import TypedHead

        head = TypedHead(self.model.cfg.core.d_model, schema, **head_kwargs)
        self.heads[name] = head.to(device_of(self.model))
        return self.heads[name]

    # -- training ------------------------------------------------------------

    def fit(self, name: str, samples: Sequence[Sample], labels: Mapping[str, torch.Tensor],
            steps: int = 200, lr: float = 3e-3, holdout: float = 0.2,
            abstain_labels: Optional[torch.Tensor] = None, seed: int = 0) -> dict:
        """Train task ``name``'s head on a frozen model, then calibrate on a holdout.

        The model is frozen deliberately. Training the trunk through a typed
        head would move representations every other head and the text path
        depend on, for the benefit of one schema -- and the change would be
        invisible to everything except the one task that asked for it. Hidden
        states are encoded once up front, so a head trains in seconds.

        The holdout is carved off *before* training and used only for the
        temperature fit. Calibrating on training data measures how well the
        head memorised, not how well its confidence predicts correctness on
        inputs it has not seen, which is the only thing routing needs.
        """
        head = self.heads[name]
        states = torch.stack([encode_prompt(self.model, s) for s in samples])
        n = states.shape[0]
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(n, generator=g)
        n_hold = int(round(n * holdout)) if n >= 5 else 0
        hold, train = order[:n_hold], order[n_hold:]
        pick = lambda idx: {k: v[idx.to(v.device)] for k, v in labels.items()}  # noqa: E731

        head.train()
        opt = torch.optim.AdamW(head.parameters(), lr=lr)
        losses = []
        for _ in range(steps):
            kwargs = {}
            if abstain_labels is not None:
                kwargs["abstain_label"] = abstain_labels[train.to(abstain_labels.device)]
            loss = head.loss(states[train], pick(train), **kwargs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        head.eval()
        temperatures = head.calibrate(states[hold], pick(hold)) if n_hold else {}
        return {"loss_first": losses[0], "loss_last": losses[-1],
                "train": int(train.numel()), "holdout": int(n_hold),
                "temperatures": temperatures}

    # -- routing -------------------------------------------------------------

    @torch.no_grad()
    def answer(self, sample: Sample, name: str, *, fallback: bool = True,
               parse: Optional[Callable[[str], Any]] = None, **generate_kwargs) -> Answer:
        """Route one request. System 1 first; escalate if it declines or is unsure."""
        if name not in self.heads:
            raise KeyError(f"no typed head registered for task {name!r}")
        head = self.heads[name]
        pred = head(encode_prompt(self.model, sample).unsqueeze(0),
                    abstain_threshold=self.abstain_threshold)
        field_conf = {k: float(fp.confidence[0]) for k, fp in pred.fields.items()}
        weakest = min(field_conf.values()) if field_conf else 0.0

        if not bool(pred.abstained[0]) and weakest >= self.min_confidence:
            values = {k: fp.value[0] for k, fp in pred.fields.items()}
            value = values["value"] if list(values) == ["value"] else values
            return Answer("system1", value=value, confidence=weakest,
                          reason="typed head answered", forward_passes=1,
                          field_confidence=field_conf)

        reason = ("typed head abstained" if bool(pred.abstained[0])
                  else f"weakest field confidence {weakest:.3f} < {self.min_confidence:.3f}")
        if not fallback:
            return Answer("system1", value=None, confidence=weakest, reason=reason,
                          forward_passes=1, field_confidence=field_conf)
        out = self._generate(self.model, sample, **generate_kwargs)
        text = getattr(out, "text", str(out))
        steps = len(getattr(out, "ids", ())) or 1
        return Answer("system2", value=parse(text) if parse else None, text=text,
                      reason=reason, forward_passes=1 + steps,
                      field_confidence=field_conf)

    # -- persistence -----------------------------------------------------------

    def state_dict(self) -> dict:
        return {name: {"schema": head.schema, "state": head.state_dict(),
                       "temperatures": dict(head.temperatures),
                       "d_hidden": head.d_hidden, "dependency": head.dependency}
                for name, head in self.heads.items()}

    def load_state_dict(self, data: Mapping[str, Mapping]) -> None:
        for name, entry in data.items():
            head = self.register(name, entry["schema"], d_hidden=entry["d_hidden"],
                                 dependency=entry["dependency"])
            head.load_state_dict(entry["state"])
            head.temperatures.update(entry["temperatures"])
