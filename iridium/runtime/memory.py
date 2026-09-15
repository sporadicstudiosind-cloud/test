"""Fitting a training run into the GPU you actually have.

**The honest framing first.** PyTorch cannot generally spill CUDA tensors into
system RAM. There is no flag for it. An activation that does not fit does not
fit, and a "unified memory" mode that silently pages tensors over PCIe would
turn a 20-minute run into a multi-hour one without saying so. What *does* exist
is a set of specific trades, and one of them — a paged optimizer — really is
optimizer state living in host RAM and moving back on demand.

In descending order of what they buy on a 16 GB card:

1. **Fused attention** (``model/layers.py``). The ``[B, H, T, T]`` score matrix
   is the dominant allocation in this architecture: at batch 32, 8 heads and a
   2048-token context it is **4.3 GB per layer** in fp32, and autograd holds
   one per layer until the backward pass. The fused kernels never build it.
   This is on by default and is why a T4 run that used to die now starts.
2. **Gradient accumulation.** Four micro-batches of 8 cost the memory of 8 and
   the gradient of 32. Nothing is approximated: the only difference from a true
   batch of 32 is that BatchNorm-style cross-sample statistics would differ, and
   this model has none.
3. **Optimizer state off the GPU.** AdamW keeps two fp32 moments per parameter —
   8 bytes/param, more than the weights themselves. ``paged_adamw`` puts them in
   host RAM via CUDA unified memory and pages on demand (this is the literal
   "overflow into system RAM"); ``adamw_8bit`` quantises them to 2 bytes/param
   instead; ``adafactor`` factors the second moment and drops the first.

**Gradient checkpointing is not on this list, and that is a finding.** It is the
usual third lever, and in this architecture it is incorrect. The superstacks
refuse it outright — their layers branch on data, so the recomputed pass is a
different graph and PyTorch says so. The control core *accepts* it and silently
returns different gradients: with the stochastic path pinned and the unchecked
run reproducible to 0.0, enabling it moved one embedding's gradient by 2.75
while leaving the loss bit-identical. Nothing in a training curve would show
that. ``Iridium1.enable_gradient_checkpointing`` raises rather than offering it.

And the thing nobody tells you: **interrupting a cell does not free anything.**
The model, the optimizer and the autograd graph are still referenced by the
notebook kernel, and PyTorch's caching allocator holds freed blocks as reserve.
:func:`free_memory` is what actually gives it back.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch


# --------------------------------------------------------------------------
# reclaiming
# --------------------------------------------------------------------------


def free_memory(*objects: Any, verbose: bool = True) -> dict[str, float]:
    """Drop references, collect, and return the allocator's cache to the driver.

    Pass whatever is holding memory — ``free_memory(model, trainer, optimizer)``
    — then rebind those names to ``None`` yourself. This clears *its* references,
    collects, and empties the cache; it cannot clear the caller's variables, and
    a single surviving reference keeps the whole graph alive.

    The usual surprise after a failed run is that ``nvidia-smi`` still shows
    gigabytes in use. Two causes, both handled here: Python has not collected
    the autograd graph yet, and PyTorch keeps freed blocks reserved rather than
    returning them to the driver.
    """
    before = _snapshot()
    for obj in objects:
        if hasattr(obj, "zero_grad"):
            try:
                obj.zero_grad(set_to_none=True)
            except Exception:
                pass
        if isinstance(obj, torch.optim.Optimizer):
            obj.state.clear()
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    after = _snapshot()
    if verbose and after:
        print(f"allocated {before.get('allocated_gb', 0):.2f} -> "
              f"{after.get('allocated_gb', 0):.2f} GB, "
              f"reserved {before.get('reserved_gb', 0):.2f} -> "
              f"{after.get('reserved_gb', 0):.2f} GB")
    return after


def _snapshot() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def memory_report() -> str:
    if not torch.cuda.is_available():
        return "cpu — host RAM only"
    s = _snapshot()
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return (f"{torch.cuda.get_device_name(0)}: {s['allocated_gb']:.2f} GB live, "
            f"{s['reserved_gb']:.2f} GB reserved, peak {s['peak_gb']:.2f} GB, "
            f"of {total:.2f} GB")


def recommend_alloc_conf() -> str:
    """The one allocator setting worth having, and what it does not do.

    ``expandable_segments:True`` lets the caching allocator grow a segment
    instead of failing when free memory is fragmented into pieces that are each
    too small. It recovers the memory you already have; it does not add any.
    Must be set *before* the first CUDA allocation.
    """
    return "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"


# --------------------------------------------------------------------------
# optimizers
# --------------------------------------------------------------------------

#: Bytes of optimizer state per parameter, excluding the weights and gradients.
OPTIMIZER_STATE_BYTES = {
    "adamw": 8.0,          # two fp32 moments
    "adamw_8bit": 2.0,     # two quantised moments
    "paged_adamw": 0.0,    # two moments, in host RAM
    "adafactor": 4.0,      # factored second moment, no first moment
    "sgd": 0.0,
}


def build_optimizer(params, kind: str = "adamw", lr: float = 3e-4,
                    weight_decay: float = 0.01, betas=(0.9, 0.95)):
    """Build an optimizer, falling back loudly rather than silently.

    ``paged_adamw`` and ``adamw_8bit`` need ``bitsandbytes`` and a CUDA device.
    When either is missing this says so and uses plain AdamW, because a run that
    quietly used four times the memory you budgeted for is worse than one that
    told you it was going to.
    """
    params = list(params)
    kind = kind.lower()

    if kind in ("paged_adamw", "adamw_8bit"):
        try:
            import bitsandbytes as bnb
        except ImportError:
            print(f"[memory] {kind} needs bitsandbytes (pip install bitsandbytes); "
                  "using adamw, which holds 8 bytes/param of state on the GPU")
            kind = "adamw"
        else:
            if not torch.cuda.is_available():
                print(f"[memory] {kind} needs CUDA; using adamw")
                kind = "adamw"
            elif kind == "paged_adamw":
                return bnb.optim.PagedAdamW32bit(
                    params, lr=lr, weight_decay=weight_decay, betas=betas)
            else:
                return bnb.optim.AdamW8bit(
                    params, lr=lr, weight_decay=weight_decay, betas=betas)

    if kind == "adafactor":
        try:
            from torch.optim import Adafactor
            return Adafactor(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            print("[memory] this torch has no Adafactor; using adamw")
            kind = "adamw"

    if kind == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.0)

    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryPlan:
    parameters: int
    weights_gb: float
    gradients_gb: float
    optimizer_gb: float
    attention_gb: float
    fixed_gb: float
    available_gb: float
    optimizer_kind: str
    micro_batch: int
    accumulate: int

    @property
    def fits(self) -> bool:
        return self.fixed_gb + self.attention_gb <= self.available_gb

    def render(self) -> str:
        lines = [
            f"  weights            {self.weights_gb:7.2f} GB",
            f"  gradients          {self.gradients_gb:7.2f} GB",
            f"  optimizer state    {self.optimizer_gb:7.2f} GB  ({self.optimizer_kind})",
            f"  activations (est)  {self.attention_gb:7.2f} GB  "
            f"(micro-batch {self.micro_batch})",
            f"  {'-' * 40}",
            f"  total              {self.fixed_gb + self.attention_gb:7.2f} GB"
            f"  of {self.available_gb:.2f} GB available",
        ]
        if self.accumulate > 1:
            lines.append(f"  accumulating {self.accumulate} micro-batches "
                         f"-> effective batch {self.micro_batch * self.accumulate}")
        lines.append("  " + ("FITS" if self.fits else "DOES NOT FIT — see suggestions"))
        return "\n".join(lines)


def plan_training(
    n_params: int,
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_heads: int,
    available_bytes: float,
    optimizer_kind: str = "adamw",
    fused_attention: bool = True,
    bytes_per_element: int = 4,
    headroom: float = 0.15,
) -> MemoryPlan:
    """Estimate, then pick a micro-batch that fits.

    An estimate, not a measurement: activation memory depends on the
    implementation in ways no formula captures exactly. It is deliberately
    pessimistic, and the run is what settles it.
    """
    weights = n_params * bytes_per_element
    grads = n_params * bytes_per_element
    opt = n_params * OPTIMIZER_STATE_BYTES.get(optimizer_kind, 8.0)
    fixed = (weights + grads + opt) / 1e9
    usable = available_bytes * (1.0 - headroom) / 1e9

    def activations(micro: int) -> float:
        if fused_attention:
            # Residual stream and the block's internal tensors, per layer.
            per_layer = micro * seq_len * 8 * bytes_per_element
        else:
            # The score matrix dominates and is quadratic in sequence length.
            per_layer = micro * n_heads * seq_len * seq_len * bytes_per_element
        return per_layer * n_layers * 4 / 1e9    # 4x for the block's intermediates

    micro = batch_size
    while micro > 1 and fixed + activations(micro) > usable:
        micro //= 2
    accumulate = max(1, batch_size // micro) if micro else 1

    return MemoryPlan(
        parameters=n_params,
        weights_gb=weights / 1e9,
        gradients_gb=grads / 1e9,
        optimizer_gb=opt / 1e9,
        attention_gb=activations(micro),
        fixed_gb=fixed,
        available_gb=usable,
        optimizer_kind=optimizer_kind,
        micro_batch=micro,
        accumulate=accumulate,
    )


def suggestions(plan: MemoryPlan) -> list[str]:
    """What to change, most effective first, only what still applies."""
    out: list[str] = []
    if plan.fits:
        return ["it fits; the estimate is pessimistic, so the run is the real test"]
    if plan.optimizer_kind == "adamw":
        out.append(f"OPTIMIZER = 'paged_adamw'  — moves {plan.optimizer_gb:.1f} GB "
                   "of moments into host RAM (needs bitsandbytes)")
        out.append(f"OPTIMIZER = 'adamw_8bit'   — {plan.optimizer_gb * 0.75:.1f} GB "
                   "saved, state stays on the GPU")
    if plan.micro_batch > 1:
        out.append(f"BATCH_SIZE smaller than {plan.micro_batch} — accumulation keeps "
                   "the effective batch the same")
    out.append("MAX_SEQ_LEN shorter — activations scale linearly with it")
    out.append("a smaller PRESET — nothing else helps once weights alone exceed the card")
    return out
