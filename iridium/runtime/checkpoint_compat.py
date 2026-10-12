"""Load a checkpoint written by an older revision of the model, safely or not at all.

The model gains parameters as it evolves, and old checkpoints do not have
them. The two obvious responses are both wrong:

* ``strict=True`` refuses every old checkpoint, including ones that differ only
  by a parameter that is inert until a feature using it is switched on. The
  checkpoint shipped in ``serve/weights/`` predates the ``quantity`` modality,
  so a strict load has raised on it since that modality was added, and the
  server's ``nano`` entry could not serve its own trained weights.
* ``strict=False`` accepts every mismatch silently -- a missing attention
  projection and a missing unused vector look identical, and the first one
  produces a model that runs and emits garbage.

What this module does instead is name, per missing key, *why* it is safe to
leave at its initial value, and refuse anything it cannot name:

* ``*.null_cond`` -- the classifier-free-guidance "no conditioning" vector on
  each flow head. It is read only when guidance is used (``cfg_dropout > 0`` in
  training, a guidance scale at sampling); a checkpoint trained without it
  behaves identically with it at its zero init.
* the encoder/decoder of a modality the checkpoint was never trained on
  (e.g. ``quantity`` for a checkpoint that predates it). Those weights are
  untrained either way; leaving them at init changes nothing about the
  modalities the checkpoint *was* trained on. The report says which, so no one
  mistakes the model for one that handles them.

Unexpected keys and shape mismatches are always errors: they mean the
architecture itself differs, and no amount of filling makes that safe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

import torch

__all__ = ["CompatReport", "load_compatible", "CheckpointIncompatible"]


class CheckpointIncompatible(RuntimeError):
    """The checkpoint differs from the model in a way no filling can make safe."""


#: (pattern, reason). A missing key is accepted only if some pattern matches.
_SAFE_MISSING: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"^codecs\.decoders\.[a-z]+\.null_cond$"),
     "classifier-free-guidance null vector; inert unless guidance is used"),
)


def _untrained_modality(key: str, present: set[str]) -> str | None:
    """A missing codec key belonging to a modality with *no* weights in the file."""
    m = re.match(r"^codecs\.(encoders|decoders)\.([a-z]+)\.", key)
    if not m:
        return None
    modality = m.group(2)
    prefix_enc, prefix_dec = f"codecs.encoders.{modality}.", f"codecs.decoders.{modality}."
    if any(k.startswith((prefix_enc, prefix_dec)) for k in present):
        return None            # partially present: a real mismatch, not an absent modality
    return modality


@dataclass
class CompatReport:
    filled: dict[str, str] = field(default_factory=dict)      # key -> reason
    untrained_modalities: list[str] = field(default_factory=list)

    @property
    def exact(self) -> bool:
        return not self.filled

    def summary(self) -> str:
        if self.exact:
            return "checkpoint matches the model exactly"
        parts = [f"{len(self.filled)} parameter(s) left at initialisation"]
        if self.untrained_modalities:
            parts.append("untrained modalities: " + ", ".join(self.untrained_modalities))
        return "; ".join(parts)


def load_compatible(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> CompatReport:
    """Load ``state`` into ``model``, filling only keys that are provably safe to fill."""
    own = model.state_dict()
    present = set(state)
    unexpected = sorted(present - set(own))
    mismatched = sorted(k for k in present & set(own) if tuple(state[k].shape) != tuple(own[k].shape))
    report = CompatReport()
    unsafe: list[str] = []
    untrained: set[str] = set()
    for key in sorted(set(own) - present):
        reason = next((why for pattern, why in _SAFE_MISSING if pattern.match(key)), None)
        if reason is None:
            modality = _untrained_modality(key, present)
            if modality is not None:
                untrained.add(modality)
                reason = f"modality {modality!r} absent from the checkpoint; untrained either way"
        if reason is None:
            unsafe.append(key)
        else:
            report.filled[key] = reason
    if unexpected or mismatched or unsafe:
        lines = []
        if unsafe:
            lines.append(f"missing and not safe to fill: {unsafe[:8]}")
        if unexpected:
            lines.append(f"unexpected: {unexpected[:8]}")
        if mismatched:
            lines.append("shape mismatch: " + ", ".join(
                f"{k} {tuple(state[k].shape)} vs {tuple(own[k].shape)}" for k in mismatched[:8]))
        raise CheckpointIncompatible("; ".join(lines))
    report.untrained_modalities = sorted(untrained)
    model.load_state_dict(dict(state), strict=False)
    return report
