"""Rotary position embedding with *explicit* positions.

Every other implementation in this repository passes positions in rather than
assuming ``arange(T)``. That is not fastidiousness: superstacks see a
*subsequence* of the stream — tokens 3, 7 and 11 packed adjacently — and a
rotary embedding applied to their packed indices would encode a relative gap of
one where the real gap is four. The dispatcher would then be silently changing
the geometry a token is embedded in depending on how busy the router was.

``iridium/model/tests`` and ``tests/integration/test_superstack_kv.py`` pin the
consequence: packed and unpacked paths must agree.

Four additions on top of that baseline, all opt-in and all defaulting to
exactly today's numbers:

* **A cos/sin table cache** (see ``RotaryEmbedding._table_cache``). One
  ``RotaryEmbedding`` instance is shared by every layer in the core and every
  stack, and within a single forward pass most of those layers call it with
  the *same* ``positions`` tensor object — the stream's positions do not
  change layer to layer. Before this, every one of those calls recomputed
  ``cos``/``sin`` from scratch; the cache below makes the second call onward a
  dict lookup. It is keyed by the positions tensor's identity (``id``), not by
  its values, and only ever reuses an entry after confirming the stored
  tensor ``is`` the tensor just passed in — a value-based cache (hashing
  ``positions``) would need to materialize and compare the tensor to build the
  key, which is exactly the compute the cache exists to skip. The one
  intentional oddity is that the cache holds a strong reference to each
  positions tensor it has seen: `id` is only a safe cache key while the object
  it names cannot be garbage-collected and its address reused for something
  else, and a strong reference is what forecloses that. The capacity bound
  (4 slots, FIFO) caps how much that costs in retained memory. Identity is
  still a safe key for a ``[B, T, 3]`` M-RoPE positions tensor: nothing above
  changes when the *shape* of "positions" grows a trailing axis, only what
  ``angles`` does with it once looked up.
* **RoPE scaling for context extension** (``scaling=``). Three strategies are
  implemented, all static (chosen once at construction, not adapted per
  sequence length): ``"linear"`` position interpolation (Chen et al. 2023 —
  divide positions by the extension factor so the model never sees a position
  index larger than it was trained on), ``"ntk"`` NTK-aware base adjustment
  (raise ``theta`` so short-wavelength/high-frequency dimensions are barely
  stretched while long-wavelength ones absorb most of the extension), and now
  ``"yarn"`` (Peng, Quesnelle, Fan & Shippole, "YaRN: Efficient Context Window
  Extension of Large Language Models", arXiv:2309.00071), which is what
  ``"ntk"``'s docstring used to point at as the thing it deliberately wasn't.
  YaRN does two things neither ``"linear"`` nor ``"ntk"`` does alone:

  1. **Per-frequency-band ramp** ("NTK-by-parts", ``_yarn_inv_freq_and_mscale``
     below). ``"linear"`` interpolates every dimension uniformly, which
     needlessly blurs the short-wavelength dimensions that complete many full
     turns within the training range and were never the ones causing trouble.
     ``"ntk"`` fixes that by raising the base once, globally, which *does*
     spare the high-frequency dimensions but has no independent handle on
     where the transition happens — the single scalar ``theta`` multiplier
     that spares dimension 0 is the same one that governs every other
     dimension's stretch, so the transition band is wherever the exponential
     curve happens to put it, not a chosen wavelength. YaRN instead computes,
     per frequency index, the wavelength ``2*pi/inv_freq_i`` and classifies it
     against two rotation-count thresholds (``beta_fast``, ``beta_slow``)
     translated into a dimension-index range via the paper's
     ``find_correction_dim``: dimensions whose wavelength is much shorter
     than the trained context are left at their original (extrapolated)
     frequency untouched, dimensions whose wavelength is comparable to or
     longer than the trained context are linearly interpolated by ``1/factor``
     (Chen et al.'s treatment, but now applied only where it is the right
     call), and a linear ramp over the dimension index blends the two in
     between so there is no discontinuity a trained attention head could
     never have seen. This is strictly more work than either single-strategy
     alternative because it requires the wavelength accounting neither of
     them needs — that accounting is the entire reason YaRN is a separate
     scaling type instead of a flag on ``"ntk"``.
  2. **Attention temperature correction** (``mscale``, ``get_mscale`` in the
     original implementation). Interpolating any RoPE dimension flattens that
     dimension's contribution to the pre-softmax logits (rotating by a
     smaller effective angle per step shrinks ``|cos(delta*theta)|``'s spread
     across the sequence), which lengthens the entropy of the softmax
     distribution attention was tuned for at the original context length. The
     obvious fix — retune the softmax temperature by hand per context length —
     is what the paper's ablations reject in favor of a closed-form, len-
     independent correction: multiply the rotated ``q``/``k`` components by a
     constant ``mscale = 0.1 * ln(factor) + 1`` (so the ``q . k`` logit itself
     scales by ``mscale ** 2``), derived empirically to track the perplexity-
     minimizing temperature across the factors the paper measured. This is
     applied here by scaling the returned ``cos``/``sin`` tables, since every
     caller already multiplies ``x`` by exactly those tensors — no call site
     needs to change to pick up the correction.

  ``scaling=None`` (the default) reproduces today's ``theta``-only behaviour
  exactly, and ``"yarn"`` with ``factor=1`` reproduces it exactly too: with no
  extension requested, ``inv_freq_interpolation`` and ``inv_freq_extrapolation``
  are the same value, so the ramp has nothing to blend and ``mscale`` is
  exactly ``1``.
* **Multi-axis positions** (``sections=``, M-RoPE — Qwen2-VL, Bai et al. 2024,
  arXiv:2409.12191). ``codecs/spatial.py`` already implements this idea as its
  own module (``AxialRotaryEmbedding``) for the vision/native-resolution
  codecs; what is added here is the same partition-the-frequency-table trick,
  inside *this* class, so a stack that wants M-RoPE does not need a second
  rotary module living alongside this one with its own cache and scaling
  logic — one ``RotaryEmbedding`` instance still serves the whole layer.
  ``sections=(n_t, n_y, n_x)`` splits the ``d_head/2`` frequency slots into
  that many contiguous chunks, one per axis; ``forward``/``angles`` then
  accept ``positions`` shaped either ``[B, T]`` (today's contract — broadcast
  onto every axis) or ``[B, T, len(sections)]`` (one coordinate per axis).
  Because the frequency table itself is never rebuilt or reordered per axis —
  only *which coordinate* feeds a given frequency slot changes — feeding the
  same scalar into every axis reconstructs bit-for-bit the angle plain 1-D
  RoPE would have produced for that scalar (see ``angles``), which is what
  makes a text token an ordinary one-axis case of M-RoPE rather than a
  degenerate one. ``sections=None`` (the default) keeps ``positions`` strictly
  ``[B, T]``, exactly today's contract. M-RoPE composes with ``scaling``
  unchanged: scaling only ever changes ``inv_freq`` or rescales the resulting
  ``cos``/``sin``, never how axes are assigned to frequency slots.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Optional, Sequence

import torch

_SCALING_TYPES = ("linear", "ntk", "yarn")
_YARN_REQUIRED_KEYS = ("factor", "original_max_position")
_YARN_DEFAULT_BETA_FAST = 32
_YARN_DEFAULT_BETA_SLOW = 1


def _yarn_find_correction_dim(num_rotations: float, dim: int, base: float, max_position: float) -> float:
    """Dimension index whose wavelength complete exactly ``num_rotations`` turns over ``max_position``.

    Inverts ``wavelength_i = 2*pi * base**(2i/dim)``: solving
    ``max_position / wavelength_i = num_rotations`` for ``i`` gives this. Used
    with ``num_rotations = beta_fast`` and ``beta_slow`` to turn the paper's
    two rotation-count hyperparameters into a concrete dimension-index range
    (YaRN, arXiv:2309.00071, section 3.1).
    """
    return (dim * math.log(max_position / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(
    beta_fast: float, beta_slow: float, dim: int, base: float, max_position: float
) -> tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(beta_fast, dim, base, max_position))
    high = math.ceil(_yarn_find_correction_dim(beta_slow, dim, base, max_position))
    return max(low, 0), min(high, dim - 1)


def _yarn_ramp(low: float, high: float, n_freq: int) -> torch.Tensor:
    """Linear ramp from 0 to 1 over dimension index, clamped outside ``[low, high]``."""
    if low == high:
        high += 0.001  # avoid a divide-by-zero when both thresholds land on one dim
    linear = (torch.arange(n_freq, dtype=torch.float32) - low) / (high - low)
    return linear.clamp(0.0, 1.0)


def _yarn_inv_freq_and_mscale(
    d_head: int,
    theta: float,
    factor: float,
    original_max_position: float,
    beta_fast: float,
    beta_slow: float,
) -> tuple[torch.Tensor, float]:
    """Per-dimension NTK-by-parts ``inv_freq`` plus the scalar attention-temperature ``mscale``.

    Reproduces the reference implementation released alongside YaRN
    (arXiv:2309.00071): ``extrapolation_factor`` is 1 (frequency left exactly
    as trained) for the short-wavelength dimensions below ``beta_fast``
    rotations over ``original_max_position``, 0 (fully interpolated by
    ``1/factor``, i.e. Chen et al. 2023's linear position interpolation) for
    the long-wavelength dimensions above ``beta_slow`` rotations, and ramps
    linearly between. ``factor <= 1`` degenerates to the unscaled schedule
    exactly, because interpolation and extrapolation are then the same value
    regardless of the ramp, and ``mscale`` is exactly 1.
    """
    n_freq = d_head // 2
    pos_freqs = theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
    inv_freq_extrapolation = 1.0 / pos_freqs
    if factor <= 1.0:
        # interpolation and extrapolation are mathematically the same value
        # here (dividing by factor == 1 is a no-op), but computing the ramp
        # mix anyway would still round the result through an extra multiply
        # and add per dimension -- not bit-identical to the unscaled table
        # even though it is mathematically supposed to be. Short-circuiting
        # is what makes ``factor=1`` reproduce unscaled RoPE exactly rather
        # than merely to floating-point tolerance.
        return inv_freq_extrapolation, 1.0
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    low, high = _yarn_find_correction_range(beta_fast, beta_slow, d_head, theta, original_max_position)
    extrapolation_factor = 1.0 - _yarn_ramp(low, high, n_freq)
    inv_freq = (
        inv_freq_interpolation * (1.0 - extrapolation_factor)
        + inv_freq_extrapolation * extrapolation_factor
    )
    mscale = 0.1 * math.log(factor) + 1.0
    return inv_freq, mscale


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        d_head: int,
        theta: float = 500_000.0,
        scaling: Optional[dict] = None,
        cache_capacity: int = 4,
        sections: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if d_head % 2:
            raise ValueError("rotary embedding needs an even head dimension")
        if scaling is not None and scaling.get("type") not in _SCALING_TYPES:
            raise ValueError(
                f"scaling['type'] must be one of {_SCALING_TYPES!r}, got "
                f"{scaling.get('type')!r}"
            )
        if scaling is not None and scaling["type"] == "yarn":
            missing = [k for k in _YARN_REQUIRED_KEYS if k not in scaling]
            if missing:
                raise ValueError(f"yarn scaling missing required key(s): {missing!r}")
        n_freq = d_head // 2
        if sections is not None:
            if sum(sections) != n_freq:
                raise ValueError(
                    f"sections {tuple(sections)!r} must sum to d_head/2 ({n_freq})"
                )
        self.d_head = d_head
        self.theta = theta
        self.scaling = scaling
        self.sections = tuple(sections) if sections is not None else None

        # NTK-aware scaling changes the *base*, computed once here, so it is
        # baked into inv_freq like an unscaled model's theta would be. Linear
        # interpolation instead rescales the *positions* at lookup time (see
        # ``angles``), because it is a statement about where a given absolute
        # position lands in the trained range, not about the frequency ladder.
        # YaRN's per-dimension ramp is likewise baked into inv_freq once here
        # (see ``_yarn_inv_freq_and_mscale``): it is as static as NTK's base
        # adjustment, just computed per frequency index instead of globally.
        self._yarn_mscale = 1.0
        if scaling is not None and scaling["type"] == "ntk":
            factor = float(scaling["factor"])
            effective_theta = theta * (factor ** (d_head / (d_head - 2)))
            inv = 1.0 / (
                effective_theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
            )
        elif scaling is not None and scaling["type"] == "yarn":
            inv, self._yarn_mscale = _yarn_inv_freq_and_mscale(
                d_head,
                theta,
                float(scaling["factor"]),
                float(scaling["original_max_position"]),
                float(scaling.get("beta_fast", _YARN_DEFAULT_BETA_FAST)),
                float(scaling.get("beta_slow", _YARN_DEFAULT_BETA_SLOW)),
            )
        else:
            inv = 1.0 / (theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
        self.register_buffer("inv_freq", inv, persistent=False)

        if self.sections is not None:
            # Which frequency slot belongs to which axis. The table itself
            # (``inv_freq`` above) is never reordered or rebuilt per axis —
            # only this lookup, of which *coordinate* feeds a given slot,
            # changes — which is what makes the equal-axes-vs-1D reduction in
            # ``angles`` exact rather than approximate. See module docstring
            # and ``codecs/spatial.py`` (the sibling implementation this
            # mirrors for the vision codecs).
            sizes = torch.tensor(self.sections)
            # output_size is known, and passing it is what lets this be built
            # on the meta device (deferred / sharded initialisation), where a
            # data-dependent output length cannot be computed.
            axis_id = torch.repeat_interleave(torch.arange(len(self.sections)), sizes,
                                              output_size=sum(self.sections))
            self.register_buffer("axis_id", axis_id, persistent=False)
        else:
            self.axis_id = None

        # id(positions) -> (positions, cos, sin). See module docstring for why
        # identity rather than value is the key, and why that is safe.
        self._table_cache: "OrderedDict[int, tuple]" = OrderedDict()
        self._cache_capacity = cache_capacity

    def clear_cache(self) -> None:
        """Drop cached tables. Not needed for correctness; useful in tests."""
        self._table_cache.clear()

    def angles(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``positions`` is ``[B, T]``, or ``[B, T, len(sections)]`` when
        ``sections`` was given (integer or float). Returns cos, sin.

        Always computed in fp32 regardless of the model's compute dtype:
        ``inv_freq`` is an fp32 buffer and the multiply/cos/sin below stay in
        fp32 until the caller casts down in ``forward``. Computing the table
        in bf16 is a real bug some implementations have shipped — bf16 has 8
        bits of mantissa, so two adjacent large positions can round to the
        same angle, and every rotation from that point on encodes the wrong
        relative offset. Casting a correctly-rounded fp32 angle down to the
        working dtype afterwards does not have that failure mode.

        With ``sections`` set, ``positions`` may still be plain ``[B, T]``: it
        is then broadcast onto every axis, and because the same global
        ``inv_freq`` entry is used for a given frequency slot regardless of
        which axis feeds it, the resulting angle for every slot is identical
        to what unpartitioned 1-D RoPE would compute for that scalar — the
        partition changes nothing when every axis agrees. That equality is
        exact (same floating-point operations in the same order), not merely
        close, which is what lets a mixed text+image stream share one rotary
        table without text tokens picking up any M-RoPE-specific numerics.
        """
        key = id(positions)
        cached = self._table_cache.get(key)
        if cached is not None and cached[0] is positions:
            self._table_cache.move_to_end(key)
            return cached[1], cached[2]

        # Follow the table's own precision rather than pinning fp32. The
        # buffer is cast with the model, so an fp64 model has an fp64
        # ``inv_freq``; forcing fp32 here put fp32-rounded angles into every
        # position of every layer, and an angle quantised differently on two
        # code paths is a parity failure with no cache defect behind it. Half
        # precision still widens, which was the original intent.
        work_dtype = (self.inv_freq.dtype
                      if self.inv_freq.dtype in (torch.float32, torch.float64)
                      else torch.float32)
        inv_freq = self.inv_freq.to(device=positions.device, dtype=work_dtype)

        if self.sections is not None:
            n_axes = len(self.sections)
            if positions.dim() == 2:
                coords = positions.unsqueeze(-1).expand(*positions.shape, n_axes)
            elif positions.dim() == 3 and positions.shape[-1] == n_axes:
                coords = positions
            else:
                raise ValueError(
                    f"positions must be [B, T] or [B, T, {n_axes}] when "
                    f"sections={self.sections!r}, got shape {tuple(positions.shape)}"
                )
            axis_id = self.axis_id.to(positions.device)
            per_freq_coord = coords.to(work_dtype).index_select(-1, axis_id)  # [B, T, D/2]
            if self.scaling is not None and self.scaling["type"] == "linear":
                per_freq_coord = per_freq_coord / float(self.scaling["factor"])
            freqs = per_freq_coord * inv_freq
        else:
            if positions.dim() != 2:
                raise ValueError(
                    f"positions must be [B, T] when sections is None, got shape "
                    f"{tuple(positions.shape)}"
                )
            pos = positions.to(work_dtype)
            if self.scaling is not None and self.scaling["type"] == "linear":
                pos = pos / float(self.scaling["factor"])
            freqs = pos.unsqueeze(-1) * inv_freq

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        if self._yarn_mscale != 1.0:
            # Attention temperature correction: scale both cos and sin by the
            # same constant so every caller's q.k, which already multiplies
            # both by these tables, picks it up as a q.k *= mscale**2 without
            # any call site change. See module docstring point 2.
            cos = cos * self._yarn_mscale
            sin = sin * self._yarn_mscale

        self._table_cache[key] = (positions, cos, sin)
        self._table_cache.move_to_end(key)
        if len(self._table_cache) > self._cache_capacity:
            self._table_cache.popitem(last=False)
        return cos, sin

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Rotate ``x`` of shape ``[B, H, T, D]`` by ``positions``.

        ``positions`` is ``[B, T]``, or ``[B, T, len(sections)]`` when this
        instance was constructed with ``sections`` (M-RoPE) — see ``angles``.
        """
        cos, sin = self.angles(positions)
        cos = cos.unsqueeze(1).to(x.dtype)     # [B, 1, T, D/2]
        sin = sin.unsqueeze(1).to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


def rotate_pair(
    q: torch.Tensor, k: torch.Tensor, rope: RotaryEmbedding,
    q_pos: torch.Tensor, k_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate queries and keys by their own, possibly different, positions."""
    return rope(q, q_pos), rope(k, k_pos)
