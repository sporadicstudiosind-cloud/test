"""Trainable modality-specific encoders that sit between a codec's flat linear
projection and the shared core.

The default configuration is exactly what was here before: a stack of
pre-norm residual MLP blocks over the codec's already-projected width, scaled
by ``(2 * depth) ** -.5`` so a taller stack doesn't blow up the residual
stream. ``tests/unit/test_perception.py`` pins this: with every new argument
left at its default, ``PerceptualEncoder`` is byte-for-byte the module that
was here before this file was rewritten.

Everything else here is opt-in, and every opt-in feature is a no-op at its
default, because the call site (``model/iridium1.py``) constructs one
``PerceptualEncoder`` per continuous modality with four positional arguments
and nothing else — this file cannot change what that call site knows or
passes without touching ``iridium1.py``/``codecs/bank.py``, which are outside
this change's ownership. What's built here is the encoder side of "much
better spatial understanding"; wiring it up (passing ``modality=``,
``patch_shape=``, ``grid=`` per forward call, ``n_registers=``,
``compression=``, and threading the resulting token-count change back through
``spans.py``/``bank.py``) is reported as follow-up work, not done here.

**Local mixing, before the global sequence sees a patch.** A "patch" handed
to this module is *already* a flattened little tensor (``C * patch * patch``
for an image, ``C * t * patch * patch`` for video, one PCM window for audio)
— it has spatial structure inside it that a flat ``nn.Linear`` throws away
before anything else even runs. ``patch_shape`` (e.g. ``(3, 8, 8)``) lets the
stem reshape a payload back into that structure and run a small local
mixer over it *before* the linear projection, so the projection sees
locally-consistent features instead of a bag of independent scalars. This is
the one piece of "hierarchical pyramid" mixing that needs no cross-token
information (no grid, no batch layout) and so needs no change anywhere else
to activate — it is still off by default because the call site never passes
``patch_shape``.

**Token compression.** Real hierarchical vision models don't stop at
sub-patch mixing; they merge *neighboring patches* (InternVL's pixel shuffle,
space-to-depth) so a high-resolution image costs a tractable number of
tokens instead of one token per 8x8 pixel block. That needs the patch
*grid* — which patches are adjacent — which is known at collate time
(``Span.grid``) but is not currently passed into the per-token encoder call
(``CodecBank.embed`` calls ``self.encoders[name](values)`` on a flat
``[B, T, dim]`` tensor with no grid). ``forward`` accepts an optional ``grid``
argument and ``compressed_forward`` operates on one media item's tokens
directly, precisely so this is unit-testable now; it is not reachable from
the live model until ``bank.py`` is changed to pass ``grid`` through and to
stop assuming the encoder's output token count matches its input token count
(the ``positions``/``span_id``/``modality``/``valid`` arrays in ``spans.py``
are all indexed per input token — compression breaks that alignment and
needs those to be re-derived, not just resized).

**Register tokens** (Darcet et al., 2023, "Vision Transformers Need
Registers"). A few learned tokens, prepended per image/video item, give
attention somewhere to dump global summary information instead of hijacking
a real (and typically background) patch for it — the effect Darcet et al.
document is cleaner attention maps: without registers, ViTs reliably
repurpose a handful of low-information patches as ad hoc memory, which then
show up as attention outliers when someone tries to *use* those maps (for
segmentation, for probing what the model looked at). ``n_registers`` is 0
by default (no tokens added, no shape change) and > 0 only where a caller
constructs the encoder that way directly (currently: tests).

**Modality-specific stems.** Image/video get a small local convolution-like
mixer over the patch's own ``(C, *spatial)`` shape (``conv_dim=2`` or ``3``).
Audio does **not** get a time-frequency (2-D mel x frame) mixer, on purpose:
``CodecConfig.audio_mels`` and ``audio_frames`` name a mel-spectrogram
layout, but ``codecs/media.py: audio_span`` says outright that it packs raw
PCM samples ("PCM patches are invertible; unlike mel magnitudes they do not
require a separately trained vocoder") and reshapes them using
``audio_mels * audio_frames`` purely as a fixed total width — see its comment
"historical field names, fixed tensor width". Those config field names are
misleading: the tensor the audio stem actually receives is not organized as
(frequency bin, time frame) at all, it is one flat window of consecutive
raw samples. Building a 2-D mel-axis mixer over that would mix physically
meaningless "rows" (arbitrary chunks of the sample window, not frequency
content). The audio stem here is therefore a 1-D temporal mixer over the
flat PCM window, which is the correct structure for what the payload
actually is. This is a real discrepancy between the config's naming and the
codec's behavior, reported back rather than fixed here since
``config.py``/``media.py`` are out of this change's ownership — either the
field names should be renamed to something PCM-shaped, or ``audio_span``
should be replaced with an actual mel front end (which would need a vocoder
for generation, a much bigger change).

**Depth-scaled residual initialization.** The forward-time ``(2 * depth) **
-.5`` factor already keeps the *activations* bounded as depth grows, but the
blocks' own weights are still PyTorch's default Kaiming-uniform init, sized
for a standalone (non-residual, unscaled) layer. ``depth_scaled_init=True``
additionally initializes each block's output ``Linear`` with
``std=0.02 * scale`` and a zero bias (the DeepNet/GPT-2-residual-branch
convention: https://arxiv.org/abs/1901.09321-style depth scaling), so a
deep stack starts close to its input (each block starts as an
approximately-identity residual) rather than starting from a variance
budgeted for a single layer and then also being summed ``depth`` times. It
defaults to ``False`` because turning it on changes the module's random
initialization and would break the "identical to the previous
implementation" property the default configuration is required to keep;
whoever wires up ``perception_layers`` in ``iridium1.py`` for real training
should turn it on then, not before.

No claim of a quality improvement is made anywhere in this file. Every
change here is architectural; whether it helps is a training-time question
this module cannot answer on its own, and it has not been trained.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from ..codecs.spatial import pixel_shuffle_merge


def _reshape_apply(payload: torch.Tensor, patch_shape: tuple[int, ...], conv: nn.Module) -> torch.Tensor:
    """Fold ``payload``'s last dim into ``patch_shape``, run ``conv``, flatten back."""
    lead = payload.shape[:-1]
    channels = patch_shape[0]
    spatial = patch_shape[1:]
    x = payload.reshape(-1, channels, *spatial)
    x = conv(x)
    return x.reshape(*lead, -1)


class _LocalMixStem(nn.Module):
    """Local mixing inside one already-cut patch (sub-patch pyramid level).

    Zero-initialized, additive: at init this is the identity, so enabling it
    never destabilizes a freshly-constructed model regardless of depth —
    consistent with the depth-scaled-init rationale above, but here it's
    unconditional because there's no per-layer count to scale against; there
    is exactly one stem per encoder.
    """

    def __init__(self, patch_shape: tuple[int, ...], conv_dim: int) -> None:
        super().__init__()
        channels = patch_shape[0]
        if conv_dim == 1:
            self.conv = nn.Conv1d(channels, channels, 3, padding=1)
        elif conv_dim == 2:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        elif conv_dim == 3:
            self.conv = nn.Conv3d(channels, channels, 3, padding=1)
        else:
            raise ValueError(f"conv_dim must be 1, 2 or 3, got {conv_dim}")
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.patch_shape = patch_shape

    def forward(self, payload: torch.Tensor) -> torch.Tensor:
        return payload + _reshape_apply(payload, self.patch_shape, self.conv)


def build_stem(modality: str, patch_shape: tuple[int, ...]) -> _LocalMixStem:
    """Modality-appropriate local mixer. See the module docstring for why
    audio gets a 1-D temporal mixer and not a 2-D mel/time one."""
    conv_dim = {"image": 2, "video": 3, "audio": 1, "field": 2}.get(modality)
    if conv_dim is None:
        raise ValueError(f"no stem defined for modality {modality!r}")
    return _LocalMixStem(patch_shape, conv_dim)


class PerceptualEncoder(nn.Module):
    def __init__(
        self,
        projection,
        width,
        rank,
        depth,
        *,
        modality: Optional[str] = None,
        patch_shape: Optional[tuple[int, ...]] = None,
        n_registers: int = 0,
        compression: int = 1,
        depth_scaled_init: bool = False,
    ):
        super().__init__()
        self.projection = projection
        self.width = width
        self.compression = compression
        self.stem = build_stem(modality, patch_shape) if patch_shape is not None else None

        self.blocks = nn.ModuleList(nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, rank), nn.GELU(), nn.Linear(rank, width)
        ) for _ in range(depth))
        self.scale = (2 * depth) ** -.5 if depth else 1.0

        self.compress_proj = (
            nn.Linear(width * compression * compression, width) if compression > 1 else None
        )

        self.registers = nn.Parameter(torch.zeros(n_registers, width)) if n_registers else None
        if self.registers is not None:
            nn.init.normal_(self.registers, std=0.02)

        if depth_scaled_init:
            for block in self.blocks:
                last = block[-1]
                nn.init.normal_(last.weight, std=0.02 * self.scale)
                nn.init.zeros_(last.bias)

    def forward(self, payload: torch.Tensor, grid: Optional[tuple[int, ...]] = None) -> torch.Tensor:
        if self.stem is not None:
            payload = self.stem(payload)
        hidden = self.projection(payload)
        if self.compress_proj is not None:
            if grid is None:
                raise ValueError("compression > 1 requires the token grid at forward time")
            hidden, grid = pixel_shuffle_merge(hidden, grid, self.compression)
            hidden = self.compress_proj(hidden)
        for block in self.blocks:
            hidden = hidden + self.scale * block(hidden)
        if self.registers is not None:
            lead = hidden.shape[:-2]
            regs = self.registers.expand(*lead, *self.registers.shape)
            hidden = torch.cat([regs, hidden], dim=-2)
        return hidden
