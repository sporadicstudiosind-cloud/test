"""Multi-axis rotary position encoding for native-resolution media.

The obvious wrong alternative is what the codebase does today: patches get a
flat 1-D stream index (``Batch.positions``) and the sequence rotary embedding
in ``model/rope.py`` treats an image the way it treats a sentence. Two patches
that are vertically adjacent in the image get a "distance" of ``grid_width``,
identical to two patches that are on opposite sides of the picture but happen
to sit ``grid_width`` tokens apart in raster order. The model cannot recover
"up" and "down" from that signal; it can only recover "earlier" and "later" in
an arbitrary flattening. Normalizing patch index to ``[0, 1]`` per image (the
other obvious alternative — a resolution-relative grid coordinate) is not
better: it makes patch ``(4, 4)`` of a 512x512 image and patch ``(8, 8)`` of a
1024x1024 image collide on the same coordinate even though they cover
different physical extents, so the model can never learn "this is an 8-pixel
edge" as a coordinate-local fact independent of what else was in the picture.

What is implemented here instead is **M-RoPE** (multi-axis rotary position
embedding), the scheme introduced for Qwen2-VL (Bai et al., 2024) and adopted
in one form or another by most native-multimodal transformers since: instead
of one rotary frequency table walking a 1-D index, the frequency table is
partitioned once, at construction time, into disjoint contiguous chunks, one
chunk per axis (``t``, ``y``, ``x``). A token's rotation angle for the
frequencies in a given chunk is that token's coordinate *on that axis only*,
in absolute native units — pixels for space, seconds (or a fixed frame
period) for time — never a normalized index, so a 512x512 tile and a
1024x768 tile share one coordinate system and a patch keeps the same meaning
regardless of what else is in the batch (the NaViT / native-resolution
packing property; Dehghani et al., 2023). A 2-D or 3-D axial split of RoPE
across separate frequency *ranges* (rather than a shared range with an
independent phase per axis) is the same family of idea used for vision
transformers' axial RoPE, generalized here to three axes and to text.

The key correctness property, proved in ``tests/unit/test_spatial.py`` rather
than asserted here, is the one that makes RoPE worth using at all: rotating
query and key by their own absolute coordinates and taking a dot product
yields a value that is a function of the *coordinate difference* alone. That
holds per-axis here because each axis only ever rotates its own frequency
chunk, so the standard RoPE relative-position algebra applies chunk-by-chunk.

**Reduction to 1-D RoPE for text.** Text does not get a separate, narrower
model: it gets the same partitioned frequency table, fed the *same* scalar
stream position on every axis (``text_coordinates``). Because the global
per-frequency table is identical to ``model/rope.py``'s (same ``theta``, same
``theta ** (-2i/d_head)`` schedule) and is only ever *partitioned*, never
*rebuilt* per axis, feeding one shared position `p` into every chunk
reconstructs the exact same angle, `p * inv_freq[i]`, for every frequency
index `i`, on every axis — which is by construction identical to standard
1-D RoPE applied to `p`. This is what makes a mixed text+image sequence
coherent: a text token is not a degenerate case with wasted axes, it is the
one-axis special case of the same rotation.

**Resolutions unseen in training.** Because axes are unbounded absolute
coordinates and not entries in a learned or bucketed table, there is nothing
to "look up" outside a trained range — extrapolation is RoPE's native
behavior, not a special mechanism bolted on. That native behavior has a
well-documented failure mode (the same one long-context LLMs hit): the
lowest-index, fastest-rotating frequency chunks complete many more full
turns at large coordinates than they ever did in training, and the network
has never seen what a phase beyond its training range "means" — attention
scores at large offsets can alias onto a phase pattern indistinguishable from
a much smaller offset, corrupting rather than merely blurring long-range
structure. ``interpolate_coordinates`` implements the mitigation with a name:
*position interpolation* (Chen et al., 2023) — linearly rescale coordinates
down into the trained extent before rotating. That trades away native
resolution at the extremes (many distinct large-image coordinates compress
onto frequencies the model only ever saw at higher density) for staying
in-distribution. Nothing here has been measured against unrescaled
extrapolation on this model; both are implemented, the tradeoff is not
resolved by this module.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn


def _axis_sizes(n_freq: int, n_axes: int) -> list[int]:
    """Split ``n_freq`` frequency slots into ``n_axes`` near-equal, contiguous chunks."""
    base, extra = divmod(n_freq, n_axes)
    return [base + (1 if i < extra else 0) for i in range(n_axes)]


class AxialRotaryEmbedding(nn.Module):
    """M-RoPE: one shared frequency table, partitioned across axes.

    ``axes`` names the coordinate axes, in the order a coordinate tensor's
    last dimension supplies them; ``("t",)`` reproduces ``model/rope.py``'s
    ``RotaryEmbedding`` exactly (see the module docstring), ``("t", "y", "x")``
    is the 3-D case used for video, ``("y", "x")`` for images with no
    temporal axis.
    """

    def __init__(
        self,
        d_head: int,
        axes: Sequence[str] = ("t", "y", "x"),
        theta: float = 500_000.0,
    ) -> None:
        super().__init__()
        if d_head % 2:
            raise ValueError("rotary embedding needs an even head dimension")
        if not axes:
            raise ValueError("at least one axis is required")
        self.d_head = d_head
        self.axes = tuple(axes)
        self.theta = theta
        n_freq = d_head // 2
        # Identical schedule to model/rope.py.RotaryEmbedding — partitioning
        # this table, rather than building a separate smaller one per axis, is
        # exactly what makes the text reduction exact (see module docstring).
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
        sizes = _axis_sizes(n_freq, len(self.axes))
        axis_id = torch.repeat_interleave(torch.arange(len(self.axes)), torch.tensor(sizes))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("axis_id", axis_id, persistent=False)

    def angles(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``coordinates`` is ``[..., len(axes)]``. Returns cos, sin ``[..., d_head/2]``."""
        if coordinates.shape[-1] != len(self.axes):
            raise ValueError(
                f"coordinates last dim {coordinates.shape[-1]} != {len(self.axes)} axes"
            )
        axis_id = self.axis_id.to(coordinates.device)
        # Gather each frequency's own axis coordinate, then scale by that
        # frequency's *global* inv_freq entry — never a per-axis-local one.
        per_freq_coord = coordinates.index_select(-1, axis_id)
        freqs = per_freq_coord.to(torch.float32) * self.inv_freq.to(coordinates.device)
        return torch.cos(freqs), torch.sin(freqs)

    def forward(self, x: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """Rotate ``x`` ``[B, H, T, D]`` by ``coordinates`` ``[B, T, len(axes)]``."""
        cos, sin = self.angles(coordinates)
        cos = cos.unsqueeze(1).to(x.dtype)  # [B, 1, T, D/2]
        sin = sin.unsqueeze(1).to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


def rotate_pair(
    q: torch.Tensor, k: torch.Tensor, rope: AxialRotaryEmbedding,
    q_coordinates: torch.Tensor, k_coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate queries and keys by their own, possibly different, coordinates."""
    return rope(q, q_coordinates), rope(k, k_coordinates)


def text_coordinates(positions: torch.Tensor, n_axes: int) -> torch.Tensor:
    """Broadcast a 1-D stream position onto every axis: ``[..., n_axes]``.

    This is the text side of the exact-reduction property: every axis'
    frequency chunk receives the same scalar, so the partitioned table
    reassembles into the original, unpartitioned one-axis rotation.
    """
    return positions.unsqueeze(-1).expand(*positions.shape, n_axes).to(torch.float32)


def interpolate_coordinates(
    coordinates: torch.Tensor, train_extent: float, native_extent: float,
) -> torch.Tensor:
    """Position-interpolation rescale (Chen et al., 2023) into the trained range.

    Multiplies every coordinate by ``train_extent / native_extent``. Use it
    when ``native_extent`` (the largest coordinate this input will present,
    e.g. an 8K image's pixel width) exceeds ``train_extent`` (the largest
    coordinate seen in training). See the module docstring for the failure
    mode this avoids and the resolution it trades away instead. A no-op when
    ``native_extent <= train_extent``.
    """
    if native_extent <= train_extent:
        return coordinates
    return coordinates * (train_extent / native_extent)


# -- native-resolution coordinate grids and token-count accounting ----------


def grid_coordinates(
    grid: Sequence[int], patch: int, origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Absolute ``(t, y, x)`` patch-center coordinates for a spatial patch grid.

    Matches the convention already used by ``codecs/media.py`` and
    ``codecs/high_resolution.py``: patch centers in native pixel units,
    ``origin`` shifts a tile's coordinates into the untiled image's frame.
    ``grid`` is ``(h, w)`` for an image or ``(t, h, w)`` for video (the ``t``
    entry there is a *frame count*, not used directly — ``origin[0]`` plus the
    caller's own per-frame-group time offset is what ends up on the time
    axis, exactly as ``high_resolution.video_tiles`` already computes it).
    """
    if len(grid) == 2:
        h, w = grid
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        t = np.full(yy.size, origin[0], dtype=np.float32)
    elif len(grid) == 3:
        _, h, w = grid
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        t = np.full(yy.size, origin[0], dtype=np.float32)
    else:
        raise ValueError(f"grid must have 2 or 3 axes, got {grid}")
    y = origin[1] + (yy.ravel().astype(np.float32) + 0.5) * patch
    x = origin[2] + (xx.ravel().astype(np.float32) + 0.5) * patch
    return np.stack([t, y, x], axis=-1).astype(np.float32)


def compressed_token_count(
    height: int, width: int, patch: int, compression: int = 1, n_registers: int = 0,
) -> int:
    """Token count for one image after patchify and pixel-shuffle compression.

    Raises rather than silently rounding, matching ``spans.patchify``'s
    contract: a compression factor that does not evenly divide the patch grid
    would otherwise silently drop edge patches, which is a content bug (lost
    pixels) dressed up as a shape convenience.
    """
    if height % patch or width % patch:
        raise ValueError(f"{height}x{width} not divisible by patch {patch}")
    gh, gw = height // patch, width // patch
    if gh % compression or gw % compression:
        raise ValueError(
            f"patch grid {gh}x{gw} not divisible by compression {compression}"
        )
    return (gh // compression) * (gw // compression) + n_registers


def pixel_shuffle_merge(
    x: torch.Tensor, grid: tuple[int, ...], factor: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Space-to-depth token merge (InternVL-style pixel shuffle), no learned weights.

    ``x`` is ``[..., n_tokens, width]`` with ``n_tokens == prod(grid)``, tokens
    in the row-major order ``spans.patchify`` produces. Merges each
    ``factor x factor`` block of the last two ``grid`` axes (space; a leading
    time axis, if present, is left alone) into one token by concatenating
    along the feature axis — the caller still owns projecting
    ``width * factor**2`` back down to ``width`` with a learned ``Linear``
    (kept in ``model/perception.py``, since this module has no parameters).
    The alternative of *pooling* (mean/max) the block instead of concatenating
    was rejected: pooling throws away exactly the fine-grained content this
    compression exists to keep tractable, leaving the model to spend capacity
    reconstructing detail a linear projection could have kept for free.
    """
    if factor == 1:
        return x, grid
    if len(grid) < 2:
        raise ValueError("pixel_shuffle_merge needs at least 2 spatial axes")
    *outer, h, w = grid
    if h % factor or w % factor:
        raise ValueError(f"grid {grid} not divisible by compression factor {factor}")
    lead_shape = x.shape[:-2]
    _, width = x.shape[-2:]
    outer_n = int(np.prod(outer)) if outer else 1
    b = int(np.prod(lead_shape)) if lead_shape else 1
    flat = x.reshape(b, outer_n, h // factor, factor, w // factor, factor, width)
    flat = flat.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    merged = flat.reshape(b, outer_n * (h // factor) * (w // factor), factor * factor * width)
    merged = merged.reshape(*lead_shape, outer_n * (h // factor) * (w // factor), factor * factor * width)
    return merged, tuple(outer) + (h // factor, w // factor)
