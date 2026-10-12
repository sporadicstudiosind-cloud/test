"""Spectral (Fourier neural operator) blocks.

Li et al. 2021, *Fourier Neural Operator for Parametric Partial Differential
Equations*. The operator learns a kernel that is diagonal in Fourier space, so
one layer is global in the spatial domain at ``O(N log N)`` cost, and the
learned map is resolution-agnostic in the modes it keeps.

**Correction carried from the source plan.** The plan's ``SpectralConv3d``
allocated *two* complex weight tensors and wrote the ``[:m1, :m2, :m3]`` and
``[-m1:, :m2, :m3]`` corners. That is the 2-D pattern applied to a 3-D
transform. ``rfftn`` halves only the **last** axis, so a ``d``-dimensional real
spectral convolution has ``2**(d-1)`` independent corner blocks:

    1-D -> 1 block      2-D -> 2 blocks      3-D -> 4 blocks

With two blocks in 3-D the corners ``[:m1, -m2:, :m3]`` and ``[-m1:, -m2:, :m3]``
are written as zero, which discards every mode with a negative second-axis
wavenumber. The operator is then not merely under-parameterized, it is
*anisotropic*: it cannot represent a structure that a 90-degree rotation in the
first two axes maps onto another representable structure.
``tests/scientific/test_spectral.py::test_three_d_needs_four_corner_blocks``
measures the resulting asymmetry.

The parameter cost follows the same rule and is what
``SuperstackConfig.params_per_spectral`` counts:
``2**(d-1) * C_in * C_out * prod(modes) * 2`` reals.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def n_corner_blocks(dims: int) -> int:
    """Independent corner blocks of a real ``dims``-dimensional spectral kernel."""
    if dims < 1:
        raise ValueError("dims must be >= 1")
    return 2 ** (dims - 1)


class SpectralConv(nn.Module):
    """Truncated spectral convolution in 1, 2 or 3 dimensions.

    Input and output are ``[B, C, *spatial]``. ``modes`` keeps the lowest
    ``modes[i]`` wavenumbers on axis ``i``; the last axis is real-halved so it
    keeps ``modes[-1]`` of ``n//2 + 1`` available components.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Sequence[int],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = tuple(int(m) for m in modes)
        self.dims = len(self.modes)
        if self.dims not in (1, 2, 3):
            raise ValueError("SpectralConv supports 1, 2 or 3 spatial dimensions")

        scale = 1.0 / (in_channels * out_channels)
        n_blocks = n_corner_blocks(self.dims)
        # Stored as real tensors with a trailing (re, im) axis so that the
        # parameter count is transparent and optimizer state is ordinary.
        self.weight = nn.Parameter(
            scale
            * torch.randn(n_blocks, in_channels, out_channels, *self.modes, 2)
        )

    # -- corner bookkeeping ------------------------------------------------

    def effective_modes(self, spatial: Sequence[int]) -> tuple[int, ...]:
        """Modes actually usable at this resolution.

        A 2x2 field does not have 8 wavenumbers, and refusing to run on it
        would make the operator's applicability depend on the coarsest patch
        in the batch. The kept modes are ``min(configured, available)`` per
        axis, and the weight tensor is sliced to match — so the *same weights*
        serve every resolution, which is the property that makes a neural
        operator resolution-agnostic in the first place.
        """
        out = []
        for axis, n in enumerate(spatial):
            limit = (n // 2 + 1) if axis == self.dims - 1 else n
            # A negative-end corner and a positive-end corner must not overlap.
            if axis < self.dims - 1:
                limit = min(limit, max(1, n // 2))
            out.append(max(1, min(self.modes[axis], limit)))
        return tuple(out)

    def _corner_slices(self, modes: Sequence[int]) -> list[tuple[slice, ...]]:
        """One slice tuple per corner block, in a fixed canonical order.

        Every axis except the last contributes a sign (low modes from the
        positive end, or from the negative/wrapped end). The last axis is
        halved by ``rfft`` and only has a positive end.
        """
        corners: list[tuple[slice, ...]] = []
        signs = itertools.product(*[(1, -1)] * (self.dims - 1)) if self.dims > 1 else [()]
        for sign in signs:
            sl: list[slice] = []
            for axis, s in enumerate(sign):
                m = modes[axis]
                sl.append(slice(0, m) if s == 1 else slice(-m, None))
            sl.append(slice(0, modes[-1]))
            corners.append(tuple(sl))
        return corners

    def _complex_weight(
        self, block: int, dtype: torch.dtype, modes: Sequence[int]
    ) -> torch.Tensor:
        w = self.weight[block]
        for axis, m in enumerate(modes):
            w = w.narrow(2 + axis, 0, m)
        w = w.to(dtype)
        return torch.complex(w[..., 0], w[..., 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = tuple(x.shape[2:])
        if len(spatial) != self.dims:
            raise ValueError(
                f"expected {self.dims} spatial dimensions, got {len(spatial)}"
            )
        modes = self.effective_modes(spatial)

        axes = list(range(2, 2 + self.dims))
        # Keep float64 in float64: the conservation tests compare a spectral
        # path against an exact solution and a silent downcast would hide the
        # operator's own error inside float32 rounding.
        work_dtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
        x_ft = torch.fft.rfftn(x.to(work_dtype), dim=axes)
        out_shape = list(x_ft.shape)
        out_shape[1] = self.out_channels
        out_ft = torch.zeros(out_shape, dtype=x_ft.dtype, device=x.device)

        letters = "xyz"[: self.dims]
        eq = f"bi{letters},io{letters}->bo{letters}"
        for block, corner in enumerate(self._corner_slices(modes)):
            idx = (slice(None), slice(None)) + corner
            out_ft[idx] = torch.einsum(
                eq, x_ft[idx], self._complex_weight(block, work_dtype, modes)
            )

        out = torch.fft.irfftn(out_ft, s=spatial, dim=axes)
        return out.to(x.dtype)


class FNOBlock(nn.Module):
    """A residual spectral block that can sit inside a transformer stack.

    Tokens are not a grid. A spectral operator needs one. This block therefore
    takes an explicit ``grids`` description saying which token spans are field
    patches and what shape they form; spans that are not fields take the
    pointwise path only. Pretending an arbitrary token sequence is a
    2-D lattice is the failure mode this signature exists to prevent.
    """

    def __init__(
        self,
        d_model: int,
        channels: int,
        modes: Sequence[int],
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        from .layers import RMSNorm

        self.norm = RMSNorm(d_model, eps)
        self.lift = nn.Linear(d_model, channels)
        self.project = nn.Linear(channels, d_model)
        self.spectral = SpectralConv(channels, channels, modes)
        self.pointwise = nn.Linear(channels, channels)
        self.dims = len(tuple(modes))
        self.channels = channels
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(
        self,
        x: torch.Tensor,
        grids: Sequence[tuple[int, int, tuple[int, ...]]] = (),
    ) -> torch.Tensor:
        """``grids`` entries are ``(batch_index, token_start, spatial_shape)``."""
        h = self.lift(self.norm(x))
        mixed = self.pointwise(h)
        if grids:
            for b, start, shape in grids:
                if len(shape) != self.dims:
                    raise ValueError(
                        f"grid {shape} does not match a {self.dims}-D operator"
                    )
                n = 1
                for s in shape:
                    n *= s
                patch = h[b, start : start + n]                     # [n, C]
                if patch.shape[0] != n:
                    raise ValueError("field span runs past the end of the sequence")
                field = patch.transpose(0, 1).reshape(1, self.channels, *shape)
                out = self.spectral(field)
                flat = out.reshape(self.channels, n).transpose(0, 1)
                mixed = mixed.clone()
                mixed[b, start : start + n] = mixed[b, start : start + n] + flat
        return x + self.project(F.gelu(mixed))
