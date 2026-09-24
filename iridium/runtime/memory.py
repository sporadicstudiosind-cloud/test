"""Conservative training memory estimates and explicit optimizer selection.

SDPA can select fused OR math kernels; masks and routing still consume memory.
AMP reduces activation storage, while master weights/gradients remain fp32.
Paged optimizers can migrate state, but are conservatively budgeted at full GPU
residency. These are estimates, never a promise that a chosen model fits.
Gradient checkpointing remains disabled because of the earlier implementation's
reported recomputation problems. free_memory cannot clear the caller's references.
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
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
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
    "eager_adamw": 8.0,
    "adamw": 8.0,          # two fp32 moments
    "adamw_8bit": 2.0,     # two quantised moments
    "paged_adamw": 8.0,    # conservative: unified memory may remain GPU-resident
    "adafactor": 4.0,      # factored second moment, no first moment
    "sgd": 0.0,
}


def decay_groups(params_or_module, weight_decay: float = 0.01):
    """Split parameters into the ones weight decay is correct for, and the rest.

    Decay is a prior that a *matrix* should be small, and it is only that. Every
    other kind of tensor in this model is harmed by it, each in its own way:

    * **RMSNorm gains** multiply the residual stream. Shrinking a gain is a
      global scale on everything downstream, so decay here is not regularisation
      but a slow, invisible attenuation of the network.
    * **Embedding tables** are updated only where a token appears, but decay
      applies every step regardless, so embedding norms shrink over training.
      OLMo 2 (arXiv 2501.00656, section 3) measured exactly this -- decayed
      embeddings drift small, early-layer gradients grow, and loss spikes
      follow -- and removed decay from embeddings. (With a tied text head the
      same tensor is also the output projection, so it is excluded too.)
    * **Biases and scalar gates** -- the halting heads, the ponder and depth
      priors, the bank gate, the focus gain. These encode a *calibrated* value.
      Decaying a halting bias of -2.0 toward zero is decaying the model's
      stopping prior toward "always stop", which shows up as a collapsed ponder
      loop and looks like an architecture problem rather than an optimizer one.

    An embedding table is a 2-D tensor, so a pure rank rule decays it -- which
    is what an earlier version of this function did while its own docstring
    said embeddings were exempt. Pass the *module* and embeddings are found by
    type (``nn.Embedding``), which survives renaming; the rank rule then
    handles norms, biases and scalars. Passing ``named_parameters()`` still
    works and falls back to rank plus a name check for embeddings.

    Returns two torch.optim-style group dicts, always both, even when one is
    empty -- an optimizer built from a stable group layout can load a checkpoint
    written by another run of the same model.
    """
    embedding_ids: set[int] = set()
    if isinstance(params_or_module, torch.nn.Module):
        for module in params_or_module.modules():
            if isinstance(module, torch.nn.Embedding):
                embedding_ids.add(id(module.weight))
        named = params_or_module.named_parameters()
    else:
        named = params_or_module
    decay, no_decay = [], []
    seen: set[int] = set()
    for name, param in named:
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        is_embedding = id(param) in embedding_ids or (
            not embedding_ids and ("embedding" in name or name.endswith(".table.weight")))
        (no_decay if is_embedding or param.ndim < 2 else decay).append(param)
    return [
        {"params": decay, "weight_decay": float(weight_decay), "group": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "group": "no_decay"},
    ]


def build_optimizer(params, kind: str = "adamw", lr: float = 3e-4,
                    weight_decay: float = 0.01, betas=(0.9, 0.95),
                    foreach: bool | None = None):
    """Build the requested optimizer; missing optional dependencies fail explicitly.

    EagerAdamW bypasses torch.optim's lazy Dynamo import. No fallback changes
    the memory budget behind the user's back.

    ``params`` may be a flat parameter iterable or a list of group dicts, so a
    caller that has split decay from no-decay (see :func:`decay_groups`) passes
    the groups straight through. ``foreach`` fuses the per-tensor update into
    list operations; ``None`` lets each optimizer choose by device, which is the
    right default because the launch overhead it removes exists only on a GPU.
    """
    params = list(params)
    kind = kind.lower()
    grouped = bool(params) and isinstance(params[0], dict)

    if kind == "eager_adamw":
        from ..training.eager_adamw import EagerAdamW
        return EagerAdamW(params, lr=lr, weight_decay=weight_decay, betas=betas,
                          foreach=foreach)
    if kind == "adamw8":
        from ..training.adamw8 import AdamW8bit
        return AdamW8bit(params, lr=lr, weight_decay=weight_decay, betas=betas)
    flat = [p for g in params for p in g["params"]] if grouped else params
    if kind in ("paged_adamw", "adamw_8bit"):
        if not all(p.is_cuda for p in flat):
            raise ValueError(f"{kind} requires CUDA parameters")
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError("Install bitsandbytes or explicitly choose eager_adamw; "
                               "automatic fallback would invalidate the memory budget") from exc
        cls = bnb.optim.PagedAdamW32bit if kind == "paged_adamw" else bnb.optim.AdamW8bit
        return cls(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if kind == "adafactor":
        return torch.optim.Adafactor(params, lr=lr, weight_decay=weight_decay)
    if kind == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.0)
    if kind != "adamw":
        raise ValueError(f"unknown optimizer {kind!r}")
    # foreach=None lets torch pick per device. The previous hard False was a
    # workaround for a Dynamo import in some releases; ``eager_adamw`` is the
    # supported route for those images, and forcing every run onto the
    # per-tensor path to protect that one case is a large, permanent, and
    # entirely avoidable cost on a model with thousands of small tensors.
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas,
                            foreach=foreach)


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
    headroom: float = 0.25,
    d_model: int = 512,
    d_ff: int = 1536,
    n_loops: int = 1,
) -> MemoryPlan:
    """Estimate, then pick a micro-batch that fits.

    An estimate, not a measurement: activation memory depends on the
    implementation in ways no formula captures exactly. It is deliberately
    pessimistic, and the run is what settles it.
    """
    if min(batch_size, seq_len, n_layers, n_heads, d_model, d_ff, n_loops) < 1:
        raise ValueError("memory plan dimensions must be positive")
    weights = n_params * bytes_per_element
    grads = n_params * bytes_per_element
    opt = n_params * OPTIMIZER_STATE_BYTES.get(optimizer_kind, 8.0)
    fixed = (weights + grads + opt) / 1e9
    usable = available_bytes * (1.0 - headroom) / 1e9

    def activations(micro: int) -> float:
        if fused_attention:
            # Residual stream and the block's internal tensors, per layer.
            per_layer = micro * seq_len * (8 * d_model + 3 * d_ff) * bytes_per_element
            # Explicit masks and SDPA math fallback can still be quadratic.
            per_layer += micro * n_heads * seq_len * seq_len * bytes_per_element
        else:
            # The score matrix dominates and is quadratic in sequence length.
            per_layer = micro * n_heads * seq_len * seq_len * bytes_per_element
        return per_layer * n_layers * n_loops / 1e9    # 4x for the block's intermediates

    micro = batch_size
    while micro > 1 and fixed + activations(micro) > usable:
        micro //= 2
    accumulate = max(1, (batch_size + micro - 1) // micro)

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
        return ["fits the estimate; actual runtime memory still needs measurement"]
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
