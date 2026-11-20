"""Typed spans, samples and batching.

A **span** is a contiguous run of tokens of one modality with one payload. A
**sample** is an ordered list of spans — an interleaved conversation of text,
fields, video and actions. Batching pads to the longest sample and records
everything the model needs to reconstruct geometry: original stream positions,
per-span grid shapes for spectral blocks, and which slots are supervised.

The next-slot objective is uniform across modalities: at position ``t`` the
model predicts the payload *and the modality* of position ``t + 1``. That is
what makes free-running generation possible without an external controller
deciding what kind of thing comes next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

MODALITIES: tuple[str, ...] = (
    "control",    # structural markers: turn boundaries, tool results, EOS
    "text",       # bytes / BPE ids
    "image",      # raster patches
    "video",      # spatio-temporal patches
    "audio",      # mel patches
    "field",      # physical field patches on a declared mesh
    "geometry",   # point / splat features
    "action",     # UI and tool actuation tuples
    "quantity",   # a dimensional scalar, carried as a value and a role
    "camera",     # per-patch Plucker rays: where each visual token is looking
)
MODALITY_INDEX = {name: i for i, name in enumerate(MODALITIES)}
CONTINUOUS = ("image", "video", "audio", "field", "geometry", "quantity")
#: Continuous modalities the model reads but never emits. A camera is
#: conditioning -- "render this from here" -- not content to generate, so it
#: gets an encoder and no decoder. Appended *after* every other modality so
#: that the indices of the original nine never move: a model built without it
#: (``CodecConfig.n_modalities == 9``) is byte-identical to one built before it
#: existed, and existing checkpoints still load.
CONDITIONING = ("camera",)
DISCRETE = ("control", "text", "action")


#: Roles a quantity can play. A number without a role is prose, and the whole
#: point of this modality is that a measurement is not prose.
QUANTITY_ROLES: tuple[str, ...] = (
    "slope", "manning", "discharge", "depth", "critical_depth", "velocity",
    "froude", "factor", "ratio", "viscosity", "time", "length",
    "count", "temperature", "pressure", "other",
)
QUANTITY_ROLE_INDEX = {r: i for i, r in enumerate(QUANTITY_ROLES)}
QUANTITY_FEATURES = 3 + len(QUANTITY_ROLES)


def quantity_vector(value: float, role: str) -> np.ndarray:
    """Encode one scalar as ``[log10|v|, sign, tanh v, role one-hot]``.

    The log is what makes the physics linear: Manning's law is a product of
    powers, so in log space the whole mapping is affine and a small network
    learns it immediately. Handing the model ``"0.0020"`` as four digit-bytes
    instead costs it a digit parser it has no reason to own — measured at 18.6%
    of answers inside tolerance against 100% for this encoding.
    """
    v = float(value)
    out = np.zeros(QUANTITY_FEATURES, dtype=np.float32)
    out[0] = np.log10(abs(v) + 1e-12)
    out[1] = np.sign(v)
    out[2] = np.tanh(v)
    try:
        out[3 + QUANTITY_ROLE_INDEX[role]] = 1.0
    except KeyError as exc:
        raise ValueError(f"unknown quantity role {role!r}") from exc
    return out


def quantity_span(pairs, supervised: bool = True) -> "Span":
    """``pairs`` is a sequence of ``(role, value)``; one token each."""
    payload = np.stack([quantity_vector(v, r) for r, v in pairs])
    return Span("quantity", payload, supervised=supervised, atomic=False)


def decode_quantity(log10_value: float) -> float:
    return float(10.0 ** log10_value)


def modality_id(name: str) -> int:
    try:
        return MODALITY_INDEX[name]
    except KeyError as exc:
        raise ValueError(f"unknown modality {name!r}") from exc


@dataclass
class Span:
    """One run of same-modality tokens.

    ``payload`` is an integer array for discrete modalities and a
    ``[n_tokens, dim]`` float array for continuous ones. ``grid`` records the
    spatial shape a continuous span forms, which spectral blocks require and
    which must never be inferred from the token count alone.
    """

    modality: str
    payload: np.ndarray
    grid: Optional[tuple[int, ...]] = None
    supervised: bool = True
    observed: bool = True
    atomic: Optional[bool] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.modality not in MODALITY_INDEX:
            raise ValueError(f"unknown modality {self.modality!r}")
        self.payload = np.asarray(self.payload)
        if (self.modality in CONTINUOUS or self.modality in CONDITIONING) and self.payload.ndim != 2:
            raise ValueError(
                f"{self.modality} span payload must be [n_tokens, dim], got "
                f"{self.payload.shape}"
            )
        if self.modality == "action" and self.payload.ndim != 2:
            raise ValueError("action span payload must be [n_tokens, 1 + n_scalars]")
        if self.grid is not None:
            n = int(np.prod(self.grid))
            if n != len(self):
                raise ValueError(
                    f"grid {self.grid} implies {n} tokens, span has {len(self)}"
                )
        if self.atomic is None:
            self.atomic = self.grid is not None and self.observed and not self.supervised
        if self.atomic and (not self.observed or self.supervised):
            # Span-coherent routing pools the router logits over the whole
            # span, which reads positions later than the token being routed.
            # This is safe only for an unsupervised observed input. Target
            # patches are present under teacher forcing but are unavailable
            # when the model emits them one by one.
            raise ValueError(
                "atomic routing requires observed=True and supervised=False: "
                "pooling over an output target span is not causal"
            )

    def __len__(self) -> int:
        return int(self.payload.shape[0])


@dataclass
class Sample:
    spans: list[Span]
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return sum(len(s) for s in self.spans)


@dataclass
class Batch:
    """Dense tensors. Field names are stable; ``iridium/model`` reads them."""

    modality: np.ndarray                    # [B, T] int64
    discrete: np.ndarray                    # [B, T] int64 (text ids / opcodes)
    continuous: dict[str, np.ndarray]       # name -> [B, T, dim] float32
    scalars: np.ndarray                     # [B, T, n_scalars] float32 (actions)
    positions: np.ndarray                   # [B, T] int64
    valid: np.ndarray                       # [B, T] bool
    supervised: np.ndarray                  # [B, T] bool
    span_id: np.ndarray                     # [B, T] int64, -1 = route per token
    grids: list[tuple[int, int, tuple[int, ...]]]
    meta: list[dict[str, Any]]
    media_coordinates: np.ndarray | None = None
    coordinate_valid: np.ndarray | None = None
    #: ``[B, T, 3]`` int64 multi-axis rotary positions (M-RoPE). A text token
    #: at stream position ``p`` is ``(p, p, p)`` -- which reduces M-RoPE to
    #: ordinary 1-D RoPE exactly -- and a media token at row ``y``, column ``x``
    #: (frame ``t``) of a span starting at ``s`` is ``(s + t, s + y, s + x)``,
    #: Qwen2-VL's layout. Only the control core reads it; everything that
    #: needs a *scalar* order (superstack packing, the bridge's position mask,
    #: the cache) keeps using ``positions``.
    rope_positions: np.ndarray | None = None

    @property
    def batch_size(self) -> int:
        return int(self.modality.shape[0])

    @property
    def length(self) -> int:
        return int(self.modality.shape[1])


def collate(
    samples: Sequence[Sample],
    continuous_dims: dict[str, int],
    n_scalars: int = 6,
    max_length: Optional[int] = None,
) -> Batch:
    lengths = [len(s) for s in samples]
    t = max(lengths) if not max_length else min(max(lengths), max_length)
    b = len(samples)

    modality = np.zeros((b, t), dtype=np.int64)
    discrete = np.zeros((b, t), dtype=np.int64)
    scalars = np.zeros((b, t, n_scalars), dtype=np.float32)
    positions = np.zeros((b, t), dtype=np.int64)
    valid = np.zeros((b, t), dtype=bool)
    supervised = np.zeros((b, t), dtype=bool)
    span_id = np.full((b, t), -1, dtype=np.int64)
    media_coordinates = np.zeros((b, t, 3), dtype=np.float32)
    rope_positions = np.zeros((b, t, 3), dtype=np.int64)
    coordinate_valid = np.zeros((b, t), dtype=bool)
    continuous = {
        name: np.zeros((b, t, dim), dtype=np.float32)
        for name, dim in continuous_dims.items()
        if any(span.modality == name for sample in samples for span in sample.spans)
    }
    grids: list[tuple[int, int, tuple[int, ...]]] = []

    for i, sample in enumerate(samples):
        cursor = 0
        next_span = 0
        for span in sample.spans:
            n = len(span)
            if cursor >= t:
                break
            n = min(n, t - cursor)
            sl = slice(cursor, cursor + n)
            if 'coordinates' in span.meta:
                coordinates = np.asarray(span.meta['coordinates'], dtype=np.float32)
                if coordinates.shape != (len(span), 3) or not np.isfinite(coordinates).all():
                    raise ValueError('media coordinates must be finite [tokens,3]')
                media_coordinates[i, sl] = coordinates[:n]
                coordinate_valid[i, sl] = True
            modality[i, sl] = modality_id(span.modality)
            valid[i, sl] = True
            supervised[i, sl] = span.supervised
            positions[i, sl] = np.arange(cursor, cursor + n)
            rope_positions[i, sl] = _rope_positions(span, cursor, n)
            if span.atomic and n == len(span):
                span_id[i, sl] = next_span
                next_span += 1
            if span.modality in CONTINUOUS or span.modality in CONDITIONING:
                if span.modality not in continuous:
                    raise ValueError(
                        f"{span.modality} span given, but this codec has no "
                        f"{span.modality} input (for cameras: CodecConfig.camera_features)"
                    )
                dim = continuous[span.modality].shape[-1]
                payload = span.payload[:n]
                if payload.shape[-1] != dim:
                    raise ValueError(
                        f"{span.modality} span has width {payload.shape[-1]}, "
                        f"codec expects {dim}"
                    )
                continuous[span.modality][i, sl] = payload
                if (span.modality == "field" and span.grid is not None
                        and n == len(span) and span.observed and not span.supervised):
                    grids.append((i, cursor, tuple(span.grid)))
            elif span.modality == "action":
                discrete[i, sl] = span.payload[:n, 0].astype(np.int64)
                width = min(n_scalars, span.payload.shape[1] - 1)
                scalars[i, sl, :width] = span.payload[:n, 1 : 1 + width]
            else:
                discrete[i, sl] = span.payload[:n].astype(np.int64)
            cursor += n

    return Batch(
        modality=modality,
        discrete=discrete,
        continuous=continuous,
        scalars=scalars,
        positions=positions,
        valid=valid,
        supervised=supervised,
        span_id=span_id,
        grids=grids,
        meta=[s.meta for s in samples],
        media_coordinates=media_coordinates, coordinate_valid=coordinate_valid,
        rope_positions=rope_positions,
    )


def _rope_positions(span: "Span", start: int, n: int) -> np.ndarray:
    """``[n, 3]`` M-RoPE positions for the first ``n`` tokens of ``span``.

    A span with a 2-D grid ``(h, w)`` is laid out row-major; a 3-D grid
    ``(t, h, w)`` frame by frame. Anything else -- text, actions, quantities,
    a gridless continuous span -- gets ``(p, p, p)``, i.e. plain 1-D RoPE.
    """
    p = np.arange(start, start + n, dtype=np.int64)
    out = np.stack([p, p, p], axis=-1)
    coords = span.meta.get("coords") if span.meta else None
    if coords is not None:
        # A sparse set of patches (only what changed on screen) has no
        # rectangular grid; each token carries its own (t, y, x) offset
        # from the span start instead. See iridium.runtime.live.
        c = np.asarray(coords, dtype=np.int64)[:n]
        return start + c
    grid = span.grid
    if grid is None or len(grid) not in (2, 3):
        return out
    shape = (1, *grid) if len(grid) == 2 else tuple(grid)
    k = np.arange(n, dtype=np.int64)
    t_idx, rem = np.divmod(k, shape[1] * shape[2])
    y_idx, x_idx = np.divmod(rem, shape[2])
    return np.stack([start + t_idx, start + y_idx, start + x_idx], axis=-1)


# -- convenience constructors ------------------------------------------------


def text_span(text: str | bytes, supervised: bool = True, offset: int = 0,
              tokenizer=None) -> Span:
    """Text as ids. ``offset`` reserves low ids for control tokens.

    Byte-level by default. Pass the run's subword ``tokenizer`` (anything with
    ``encode(str) -> list[int]``) and the same text is encoded with it instead:
    a model trained on one vocabulary and prompted in another does not fail,
    it reads fluent nonsense, so every span a subword model sees must go
    through the tokenizer it was trained with.
    """
    if tokenizer is not None:
        if isinstance(text, bytes):
            text = text.decode("utf-8", "surrogateescape")
        ids = np.asarray(tokenizer.encode(text), dtype=np.int64)
        return Span("text", ids + offset, supervised=supervised)
    raw = text.encode("utf-8") if isinstance(text, str) else text
    return Span("text", np.frombuffer(raw, dtype=np.uint8).astype(np.int64) + offset,
                supervised=supervised)


def patchify(array: np.ndarray, patch: tuple[int, ...]) -> tuple[np.ndarray, tuple[int, ...]]:
    """Split ``[C, *spatial]`` into ``[n_patches, C * prod(patch)]`` + grid shape.

    Raises rather than cropping when the shape does not divide: a silent crop
    changes the physical extent of a field, and every downstream flux integral
    would then be over a domain nobody declared.
    """
    c, *spatial = array.shape
    if len(spatial) != len(patch):
        raise ValueError(f"patch {patch} does not match spatial rank {len(spatial)}")
    grid = []
    for n, p in zip(spatial, patch):
        if n % p:
            raise ValueError(f"extent {n} is not divisible by patch {p}")
        grid.append(n // p)
    # [C, g0, p0, g1, p1, ...] -> [g0, g1, ..., C, p0, p1, ...]
    shape: list[int] = [c]
    for g, p in zip(grid, patch):
        shape += [g, p]
    reshaped = array.reshape(shape)
    order = [1 + 2 * i for i in range(len(grid))] + [0] + [2 + 2 * i for i in range(len(grid))]
    moved = np.transpose(reshaped, order)
    n_patches = int(np.prod(grid))
    return moved.reshape(n_patches, -1), tuple(grid)


def unpatchify(
    tokens: np.ndarray, grid: tuple[int, ...], channels: int, patch: tuple[int, ...]
) -> np.ndarray:
    """Exact inverse of :func:`patchify`."""
    shape = list(grid) + [channels] + list(patch)
    moved = tokens.reshape(shape)
    n = len(grid)
    order = [n]
    for i in range(n):
        order += [i, n + 1 + i]
    restored = np.transpose(moved, order)
    spatial = [g * p for g, p in zip(grid, patch)]
    return restored.reshape([channels] + spatial)
