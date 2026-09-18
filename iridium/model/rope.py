"""Rotary position embedding with *explicit* positions.

Every other implementation in this repository passes positions in rather than
assuming ``arange(T)``. That is not fastidiousness: superstacks see a
*subsequence* of the stream — tokens 3, 7 and 11 packed adjacently — and a
rotary embedding applied to their packed indices would encode a relative gap of
one where the real gap is four. The dispatcher would then be silently changing
the geometry a token is embedded in depending on how busy the router was.

``iridium/model/tests`` and ``tests/integration/test_superstack_kv.py`` pin the
consequence: packed and unpacked paths must agree.
"""

from __future__ import annotations

import torch


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, d_head: int, theta: float = 500_000.0) -> None:
        super().__init__()
        if d_head % 2:
            raise ValueError("rotary embedding needs an even head dimension")
        self.d_head = d_head
        self.theta = theta
        inv = 1.0 / (
            theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
        )
        self.register_buffer("inv_freq", inv, persistent=False)

    def angles(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``positions`` is ``[B, T]`` (integer or float). Returns cos, sin."""
        freqs = positions.to(torch.float32).unsqueeze(-1) * self.inv_freq.to(positions.device)
        return torch.cos(freqs), torch.sin(freqs)

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
