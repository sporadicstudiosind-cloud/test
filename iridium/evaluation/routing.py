"""Did the bank actually specialise, or is it just balanced?

Load balance and specialisation pull in opposite directions, and a router can
look healthy on either metric while failing the other:

* **balanced but undifferentiated** — every stack gets 1/N of every family.
  The entropy is maximal and the bank is a very expensive way to be one stack.
* **specialised but collapsed** — one stack takes everything. Perfect purity,
  no capacity.

The quantity that separates them is the **mutual information between task
family and stack choice**, ``I(family; stack)``, in nats. Zero means routing is
independent of the task. ``log(min(n_families, n_stacks))`` is the maximum, and
is reached only by a clean partition. Reporting it alongside the balance is the
only way to tell the two failure modes apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import torch

from ..codecs.bank import TensorBatch


@dataclass
class RoutingReport:
    counts: np.ndarray                     # [n_families, n_stacks]
    families: list[str]
    mutual_information: float
    max_mutual_information: float
    stack_entropy: float
    max_stack_entropy: float
    dominant_stack: dict[str, int] = field(default_factory=dict)
    purity: float = 0.0

    @property
    def normalized_mi(self) -> float:
        return self.mutual_information / max(self.max_mutual_information, 1e-12)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mutual_information_nats": round(self.mutual_information, 4),
            "max_mutual_information_nats": round(self.max_mutual_information, 4),
            "normalized_mi": round(self.normalized_mi, 4),
            "stack_entropy": round(self.stack_entropy, 4),
            "max_stack_entropy": round(self.max_stack_entropy, 4),
            "purity": round(self.purity, 4),
            "dominant_stack": self.dominant_stack,
        }

    def render(self) -> str:
        lines = ["family \\ stack     " + "".join(f"{i:>7d}" for i in range(self.counts.shape[1]))]
        totals = self.counts.sum(axis=1, keepdims=True).clip(min=1)
        share = self.counts / totals
        for i, family in enumerate(self.families):
            row = "".join(f"{share[i, j]:7.3f}" for j in range(self.counts.shape[1]))
            lines.append(f"{family:<18}{row}")
        lines.append(
            f"I(family;stack) = {self.mutual_information:.4f} nats "
            f"of {self.max_mutual_information:.4f} max "
            f"({self.normalized_mi:.1%})"
        )
        return "\n".join(lines)


def mutual_information(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = p * (np.log(p) - np.log(px) - np.log(py))
    return float(np.nansum(np.where(p > 0, term, 0.0)))


@torch.no_grad()
def analyse(
    model, loader, max_batches: int = 40, n_loops: int = 1
) -> RoutingReport:
    """Accumulate a family-by-stack dispatch table over real batches."""
    model.eval()
    n_stacks = model.cfg.stacks.n_stacks
    family_index: dict[str, int] = {}
    rows: list[np.ndarray] = []

    for i, (batch, items) in enumerate(loader.batches()):
        if i >= max_batches:
            break
        out = model(batch, n_loops=n_loops)
        index = out.decisions[0].stack_index          # [B, T, k]
        valid = batch.valid
        for b, item in enumerate(items):
            fam = item.family
            if fam not in family_index:
                family_index[fam] = len(rows)
                rows.append(np.zeros(n_stacks, dtype=np.float64))
            picks = index[b][valid[b]].reshape(-1)
            counts = np.bincount(picks.cpu().numpy(), minlength=n_stacks)
            rows[family_index[fam]] += counts

    counts = np.stack(rows) if rows else np.zeros((1, n_stacks))
    families = [f for f, _ in sorted(family_index.items(), key=lambda kv: kv[1])]
    mi = mutual_information(counts)
    max_mi = float(np.log(max(min(len(families), n_stacks), 1)))

    stack_totals = counts.sum(axis=0)
    p = stack_totals / max(stack_totals.sum(), 1)
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum()) if nz.size else 0.0

    dominant = {
        fam: int(np.argmax(counts[i])) for i, fam in enumerate(families)
    }
    purity = float(counts.max(axis=1).sum() / max(counts.sum(), 1))
    return RoutingReport(
        counts=counts, families=families, mutual_information=mi,
        max_mutual_information=max_mi, stack_entropy=entropy,
        max_stack_entropy=float(np.log(n_stacks)),
        dominant_stack=dominant, purity=purity,
    )
