"""Rotary position embedding with *explicit* positions.

Every other implementation in this repository passes positions in rather than
assuming ``arange(T)``. That is not fastidiousness: superstacks see a
*subsequence* of the stream — tokens 3, 7 and 11 packed adjacently — and a
rotary embedding applied to their packed indices would encode a relative gap of
one where the real gap is four. The dispatcher would then be silently changing
the geometry a token is embedded in depending on how busy the router was.

``iridium/model/tests`` and ``tests/integration/test_superstack_kv.py`` pin the
consequence: packed and unpacked paths must agree.

Two additions on top of that baseline, both opt-in and both defaulting to
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
  (4 slots, FIFO) caps how much that costs in retained memory.
* **RoPE scaling for context extension** (``scaling=``). Two strategies are
  implemented, both static (chosen once at construction, not adapted per
  sequence length): ``"linear"`` position interpolation (Chen et al. 2023 —
  divide positions by the extension factor so the model never sees a position
  index larger than it was trained on) and ``"ntk"`` NTK-aware base
  adjustment (raise ``theta`` so short-wavelength/high-frequency dimensions
  are barely stretched while long-wavelength ones absorb most of the
  extension, which is what actually costs YaRN its extra complexity to do
  per-frequency instead of globally). This is *not* full YaRN: YaRN additionally
  ramps between the NTK and linear treatments per frequency band with a
  wavelength threshold and applies a temperature correction to the attention
  logits themselves (``1/sqrt(t)`` scaling derived from the interpolation
  factor). Neither piece is implemented here. ``scaling=None`` (the default)
  reproduces today's ``theta``-only behaviour exactly.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch

_SCALING_TYPES = ("linear", "ntk")


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        d_head: int,
        theta: float = 500_000.0,
        scaling: Optional[dict] = None,
        cache_capacity: int = 4,
    ) -> None:
        super().__init__()
        if d_head % 2:
            raise ValueError("rotary embedding needs an even head dimension")
        if scaling is not None and scaling.get("type") not in _SCALING_TYPES:
            raise ValueError(
                f"scaling['type'] must be one of {_SCALING_TYPES!r}, got "
                f"{scaling.get('type')!r}"
            )
        self.d_head = d_head
        self.theta = theta
        self.scaling = scaling

        # NTK-aware scaling changes the *base*, computed once here, so it is
        # baked into inv_freq like an unscaled model's theta would be. Linear
        # interpolation instead rescales the *positions* at lookup time (see
        # ``angles``), because it is a statement about where a given absolute
        # position lands in the trained range, not about the frequency ladder.
        effective_theta = theta
        if scaling is not None and scaling["type"] == "ntk":
            factor = float(scaling["factor"])
            effective_theta = theta * (factor ** (d_head / (d_head - 2)))

        inv = 1.0 / (
            effective_theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
        )
        self.register_buffer("inv_freq", inv, persistent=False)

        # id(positions) -> (positions, cos, sin). See module docstring for why
        # identity rather than value is the key, and why that is safe.
        self._table_cache: "OrderedDict[int, tuple]" = OrderedDict()
        self._cache_capacity = cache_capacity

    def clear_cache(self) -> None:
        """Drop cached tables. Not needed for correctness; useful in tests."""
        self._table_cache.clear()

    def angles(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``positions`` is ``[B, T]`` (integer or float). Returns cos, sin.

        Always computed in fp32 regardless of the model's compute dtype:
        ``inv_freq`` is an fp32 buffer and the multiply/cos/sin below stay in
        fp32 until the caller casts down in ``forward``. Computing the table
        in bf16 is a real bug some implementations have shipped — bf16 has 8
        bits of mantissa, so two adjacent large positions can round to the
        same angle, and every rotation from that point on encodes the wrong
        relative offset. Casting a correctly-rounded fp32 angle down to the
        working dtype afterwards does not have that failure mode.
        """
        key = id(positions)
        cached = self._table_cache.get(key)
        if cached is not None and cached[0] is positions:
            self._table_cache.move_to_end(key)
            return cached[1], cached[2]

        pos = positions.to(torch.float32)
        if self.scaling is not None and self.scaling["type"] == "linear":
            pos = pos / float(self.scaling["factor"])
        freqs = pos.unsqueeze(-1) * self.inv_freq.to(device=positions.device, dtype=torch.float32)
        cos, sin = torch.cos(freqs), torch.sin(freqs)

        self._table_cache[key] = (positions, cos, sin)
        self._table_cache.move_to_end(key)
        if len(self._table_cache) > self._cache_capacity:
            self._table_cache.popitem(last=False)
        return cos, sin

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Rotate ``x`` of shape ``[B, H, T, D]`` by ``positions`` ``[B, T]``."""
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
