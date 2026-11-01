"""What a 1M-token context actually costs this architecture, computed from shapes.

Nothing here imports ``torch``. That is a promise, not an accident: the
question this module answers -- "does a 1M-token window fit in N GB, for this
config, with this layer schedule" -- is a question about tensor *shapes*
times element sizes, and every input already exists as plain Python numbers
in ``iridium.config`` (which itself carries the same "no torch" promise, for
the same reason: a multi-trillion-parameter rung's cost can be checked on a
laptop, or in this module's own tests, without allocating anything). The
obvious wrong alternative is to build the schedule's layers as real
``nn.Module``s and read ``.numel()`` off their KV caches after a forward
pass: that requires instantiating and running a model at the very sizes
(1M tokens, ``extreme``-rung widths) whose entire point is that nobody can
afford to instantiate and run them. A calculator has to work *before* that
model exists.

Three attention shapes carry KV cost differently as the context grows, and
this module keeps them as three separate functions rather than one
parameterised by a boolean, because their asymptotic behaviour is the
argument for a hybrid schedule and one function would bury it:

* :func:`full_attention_cache_bytes` -- exact, unbounded. Every token's K/V
  stays live forever, so the cache is ``O(tokens)`` with no ceiling. This is
  ``GroupedQueryAttention`` with ``window=None``, i.e. every layer in this
  codebase today.
* :func:`sliding_window_cache_bytes` -- exact, bounded. ``GroupedQueryAttention``
  with ``window=W`` (Mistral's uniform window; Gemma 3's 5-local:1-global
  interleave). Cache is ``O(min(tokens, W - 1))``: flat past ``W`` tokens, but
  every token more than ``W`` back is gone, not summarised -- there is no
  partial credit for "it was pretty recent".
* :func:`deltanet_state_bytes` -- constant. A :class:`~iridium.model.deltanet.
  GatedDeltaNet` layer's whole state (see that module) is one
  ``n_heads x d_head x d_head`` matrix plus a few conv-kernel columns, full
  stop, independent of how many tokens produced it. This is what a "1M
  context" needs at least some layers to be, because ``O(tokens)`` at
  ``tokens = 1_048_576`` is the number the table below exists to make
  concrete, not abstract.

:func:`mla_latent_cache_bytes` is included for completeness (DeepSeek's
multi-head latent attention, and the "MLA latent cache" the lead's brief
asks this module to be able to cost) but this codebase has no MLA layer, so
it takes the caller's own per-token-byte accounting rather than guessing at
DeepSeek's specific compression ratio -- inventing one from memory would
break the "never claim a capability you did not measure" rule as surely as
inventing a benchmark number would.

None of this includes attention's compute cost directly; :func:`schedule_flops`
is a *separate*, deliberately crude prefill-FLOPs estimate (``4 * T^2 * d``
for a full-attention layer, ignoring projections and the softmax itself) --
good enough to see that full attention's compute, like its cache, is the
term that runs away, not good enough to size a training budget from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from iridium.config import IridiumConfig

BYTES_PER_ELEMENT = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1}

_LAYER_TYPES = ("global", "local", "deltanet", "mla")


def _bytes(dtype_or_count) -> int:
    if isinstance(dtype_or_count, str):
        if dtype_or_count not in BYTES_PER_ELEMENT:
            raise ValueError(f"unknown dtype {dtype_or_count!r}; use one of {tuple(BYTES_PER_ELEMENT)}")
        return BYTES_PER_ELEMENT[dtype_or_count]
    return int(dtype_or_count)


def hybrid_schedule(n_layers: int, pattern: Sequence[str]) -> list[str]:
    """Tile ``pattern`` (e.g. ``("deltanet", "deltanet", "deltanet", "global")``,
    Qwen3-Next / Kimi Linear's 3:1 ratio) out to ``n_layers`` layers.

    Not required to divide evenly: the last tile is truncated, the same way
    an odd-length sequence's last chunk is in
    ``iridium.model.deltanet.chunkwise_forward`` -- a schedule generator that
    only worked for round numbers would be a trap for whichever rung's
    layer count doesn't happen to be a multiple of 4.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    if not pattern:
        raise ValueError("pattern must be non-empty")
    for layer_type in pattern:
        if layer_type not in _LAYER_TYPES:
            raise ValueError(f"unknown layer type {layer_type!r}; use one of {_LAYER_TYPES}")
    return [pattern[i % len(pattern)] for i in range(n_layers)]


def full_attention_cache_bytes(tokens: int, d_kv: int, n_layers: int = 1,
                                bytes_per_element="bf16") -> int:
    """Exact online K+V cache for ``n_layers`` full-attention layers at ``tokens``.

    ``2 * d_kv`` (K and V) times the element size times every layer times
    every token -- this is ``IridiumConfig.core.kv_bytes_per_token`` with the
    per-token multiply made explicit, because per-token rates are the wrong
    shape of number to compare against a device's total memory.
    """
    return 2 * d_kv * _bytes(bytes_per_element) * n_layers * tokens


def sliding_window_cache_bytes(tokens: int, window: int, d_kv: int, n_layers: int = 1,
                                bytes_per_element="bf16") -> int:
    """Same per-token rate as full attention, capped at ``window - 1`` live tokens.

    It is a ``min`` because past ``window`` tokens back, the layer's own
    ``sliding_window_keep`` (see ``model/layers.py``) has already made those
    keys unreachable, so keeping their K/V would be paying memory for
    something no query can read again. And it is ``window - 1``, not
    ``window``, because the next query's own key is computed fresh rather
    than read from the cache -- which is exactly what the layer keeps, and
    what an earlier version of this function overcounted by one.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    # window - 1: the cache keeps only the keys the *next* query can see; the
    # current token's own key is computed fresh (see GroupedQueryAttention).
    return 2 * d_kv * _bytes(bytes_per_element) * n_layers * min(tokens, max(window - 1, 0))


def deltanet_state_bytes(n_heads: int, d_head: int, n_layers: int = 1, conv_size: int = 4,
                          bytes_per_element="bf16") -> int:
    """Fixed per-layer state (see ``iridium.model.deltanet.GatedDeltaNet.state_bytes``);
    ``tokens`` does not appear in this signature on purpose."""
    per_layer = n_heads * d_head * d_head + 3 * n_heads * d_head * max(conv_size - 1, 0)
    return per_layer * _bytes(bytes_per_element) * n_layers


def mla_latent_cache_bytes(tokens: int, n_layers: int = 1, bytes_per_element="bf16", *,
                            latent_dim: Optional[int] = None,
                            formula: Optional[Callable[[int, int, int], int]] = None) -> int:
    """MLA-style compressed KV. Requires ``latent_dim`` (a bare per-token
    latent width times ``tokens``) or a caller-supplied ``formula(tokens,
    n_layers, bytes_per_element)`` for a real published compression scheme
    (e.g. DeepSeek-V4's CSA+HCA hybrid, which is not a fixed-width latent and
    so cannot be reduced to one number here -- see ``docs/long-context.md``).
    Raises rather than defaulting silently to zero or to a guessed ratio.
    """
    if formula is not None:
        return int(formula(tokens, n_layers, _bytes(bytes_per_element)))
    if latent_dim is None:
        raise ValueError("mla_latent_cache_bytes needs latent_dim or formula")
    return latent_dim * _bytes(bytes_per_element) * n_layers * tokens


def layer_cache_bytes(layer_type: str, tokens: int, *, d_kv: Optional[int] = None,
                       n_heads: Optional[int] = None, d_head: Optional[int] = None,
                       window: Optional[int] = None, conv_size: int = 4,
                       bytes_per_element="bf16", latent_dim: Optional[int] = None,
                       mla_formula=None) -> int:
    """Dispatch one layer's cache bytes at ``tokens`` by ``layer_type``."""
    if layer_type == "global":
        if d_kv is None:
            raise ValueError("'global' needs d_kv")
        return full_attention_cache_bytes(tokens, d_kv, 1, bytes_per_element)
    if layer_type == "local":
        if d_kv is None or window is None:
            raise ValueError("'local' needs d_kv and window")
        return sliding_window_cache_bytes(tokens, window, d_kv, 1, bytes_per_element)
    if layer_type == "deltanet":
        if n_heads is None or d_head is None:
            raise ValueError("'deltanet' needs n_heads and d_head")
        return deltanet_state_bytes(n_heads, d_head, 1, conv_size, bytes_per_element)
    if layer_type == "mla":
        return mla_latent_cache_bytes(tokens, 1, bytes_per_element, latent_dim=latent_dim,
                                       formula=mla_formula)
    raise ValueError(f"unknown layer type {layer_type!r}; use one of {_LAYER_TYPES}")


def schedule_cache_bytes(schedule: Sequence[str], tokens: int, **kwargs) -> int:
    """Sum :func:`layer_cache_bytes` over every layer in ``schedule``."""
    return sum(layer_cache_bytes(layer_type, tokens, **kwargs) for layer_type in schedule)


def full_attention_flops(tokens: int, d_model: int) -> int:
    """``4 * T^2 * d``: QK^T and the attention-value product, each ``T x T x d``
    multiply-adds. Ignores Q/K/V/O projections and the softmax itself -- an
    order-of-magnitude prefill estimate, not a FLOPs-accounted training budget."""
    return 4 * tokens * tokens * d_model


def sliding_window_flops(tokens: int, window: int, d_model: int) -> int:
    """Same accounting as :func:`full_attention_flops`, over ``min(tokens, window)``
    keys per query instead of every earlier token."""
    return 4 * tokens * min(tokens, window) * d_model


def deltanet_flops(tokens: int, n_heads: int, d_head: int) -> int:
    """State update and readout are each one ``d_head x d_head`` matrix-vector
    product per head per token: linear in ``tokens``, not quadratic."""
    return 4 * tokens * n_heads * d_head * d_head


def schedule_flops(schedule: Sequence[str], tokens: int, *, d_model: Optional[int] = None,
                    window: Optional[int] = None, n_heads: Optional[int] = None,
                    d_head: Optional[int] = None) -> int:
    total = 0
    for layer_type in schedule:
        if layer_type == "global":
            if d_model is None:
                raise ValueError("'global' needs d_model")
            total += full_attention_flops(tokens, d_model)
        elif layer_type == "local":
            if d_model is None or window is None:
                raise ValueError("'local' needs d_model and window")
            total += sliding_window_flops(tokens, window, d_model)
        elif layer_type == "deltanet":
            if n_heads is None or d_head is None:
                raise ValueError("'deltanet' needs n_heads and d_head")
            total += deltanet_flops(tokens, n_heads, d_head)
        elif layer_type == "mla":
            continue  # no published closed-form FLOPs formula supplied; see module docstring
        else:
            raise ValueError(f"unknown layer type {layer_type!r}; use one of {_LAYER_TYPES}")
    return total


@dataclass(frozen=True)
class BudgetReport:
    """One config's memory bill at one token count. Bytes, not GB -- the
    caller decides how to display it, this module only ever measures."""

    core_cache_bytes: int
    stack_cache_bytes: int
    weight_bytes: float
    total_bytes: float


def budget(cfg: IridiumConfig, core_schedule: Sequence[str], tokens: int, *,
           window: Optional[int] = None, bytes_per_element="bf16",
           weight_bits: float = 16.0, include_stacks: bool = True) -> BudgetReport:
    """Total memory at ``tokens`` for ``cfg``'s core run under ``core_schedule``.

    ``core_schedule`` replaces only the core's own attention layers (see
    ``docs/long-context.md`` for why the core's plain-causal ``keep`` is the
    part of this architecture DeltaNet can honestly stand in for). Superstack
    KV is priced with the config's own existing, unchanged
    ``kv_bytes_per_token`` -- ``top_k`` stacks per token, every one of the
    stack's layers -- because nothing here has redesigned the superstack or
    bridge path; see the module docstring on ``mla_latent_cache_bytes`` for
    the same "don't invent a number for a mechanism this file doesn't model"
    stance applied to the bridge's own KV read.

    ``weight_bytes`` is ``cfg.weight_bytes`` unmodified: swapping a GQA layer
    for ``GatedDeltaNet`` changes a layer's own parameter count by a few
    percent (see ``GatedDeltaNet.param_count`` vs.
    ``CoreConfig.params_per_layer``), which this approximation does not
    correct for -- stated here rather than silently absorbed, since weight
    bytes at these scales dwarf a percent-level layer-shape difference far
    less than the KV cache term this module exists to get right does.
    """
    if len(core_schedule) != cfg.core.n_layers:
        raise ValueError(
            f"core_schedule has {len(core_schedule)} layers, cfg.core has {cfg.core.n_layers}"
        )
    core_bytes = schedule_cache_bytes(
        core_schedule, tokens,
        d_kv=cfg.core.d_kv, n_heads=cfg.core.n_query_heads, d_head=cfg.core.d_head,
        window=window, bytes_per_element=bytes_per_element,
    )
    stack_bytes = 0
    if include_stacks:
        stack_bytes = cfg.router.top_k * cfg.stacks.kv_bytes_per_token(_bytes(bytes_per_element)) * tokens
    weight_bytes = cfg.weight_bytes(weight_bits)
    return BudgetReport(core_bytes, stack_bytes, weight_bytes, core_bytes + stack_bytes + weight_bytes)


def fits(cfg: IridiumConfig, core_schedule: Sequence[str], tokens: int, device_gb: float,
         **kwargs) -> bool:
    """``True`` iff :func:`budget`'s total fits in ``device_gb`` gigabytes (1e9 bytes)."""
    return budget(cfg, core_schedule, tokens, **kwargs).total_bytes <= device_gb * 1e9
