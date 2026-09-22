"""Cameras and splat scenes as model input: spans, anchor coordinates, canonical order.

Three converters, each turning a piece of ``world`` state into the shape
``iridium/codecs/spans.py`` and the model consume:

* :func:`camera_span` -- a camera's per-patch Plücker rays, aligned patch for
  patch with an ``image``/``video`` span of the same view, so the model can
  condition each visual token on the exact line in space it depicts.
* :func:`anchor_coordinates` -- per-patch 3D *positions* (not rays) for an
  existing image/depth pair, the numbers a 3D positional encoding needs to
  place a visual token in the world rather than only in its own image.
* :func:`scene_span` -- a splat scene as a token sequence, ordered so the
  same scene always yields the same sequence regardless of how its splats
  happened to be listed (see :func:`scene_span`'s own docstring).

**Modality note, read before wiring either span in:** neither camera rays nor
splat parameters have a perfectly-fitting modality in
:data:`iridium.codecs.spans.MODALITIES` today. Per this module's build
instructions, ``spans.py`` is not edited here; instead each span carries its
true content and a ``meta["kind"]`` tag, using the *closest* existing
modality as a placeholder, and the exact config change each one actually
needs is spelled out below and in the final report:

* ``scene_span`` uses ``"geometry"`` -- an exact semantic match ("point /
  splat features" is this module's own description in ``spans.py``) but the
  wrong *width*: ``CodecConfig.point_features`` is 10 today: enough for
  position + a handful of scalars, not for the 14 floats
  (``means[3] + log_scales[3] + quats[4] + opacity[1] + sh0[3]``) one splat
  needs. This needs ``point_features`` raised to 14, or a new field (e.g.
  ``splat_features``) if 10-wide point clouds elsewhere must not change width.
* ``camera_span`` defaults to ``"field"`` as a stand-in modality, chosen
  because it is (like a camera's ray field) a structured multi-channel signal
  defined over a patch grid -- but ``CodecConfig.field_channels`` is 4
  (physical-field channels; unrelated content) against the 6 Plücker
  channels a ray needs, so this is a *placeholder*, not a real fit. The
  clean answer is a new ``"camera"`` modality, width 6/patch, added to
  ``MODALITIES``/``CONTINUOUS`` in ``spans.py`` -- ``camera_span`` takes
  ``modality`` as a parameter for exactly this reason, so switching to it
  later is a one-line call-site change, not a rewrite.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from iridium.codecs.spans import Span
from .camera import Camera

__all__ = ["camera_span", "anchor_coordinates", "scene_span"]


def camera_span(camera: Camera, patch: int = 8, modality: str = "field",
                supervised: bool = False, meta: Optional[dict] = None) -> Span:
    """Per-patch Plücker rays ``(d, o x d)`` for ``camera``, as a continuous span.

    Grid and patch count match exactly what an ``image``/``video`` span for
    the same view and the same ``patch`` produces (both come from dividing
    the same ``height x width`` by the same ``patch``), so the two spans can
    be interleaved token-for-token, or their payloads concatenated/added
    per-patch, without any resampling. ``supervised=False`` by default: a
    camera's own pose is normally given, not predicted, when it is used to
    condition generation of the *image*; pass ``True`` for a
    camera-prediction task (e.g. relocalisation) where the model emits the
    rays themselves.
    """
    plucker = camera.plucker(patch)                       # [H', W', 6]
    grid = tuple(plucker.shape[:-1])
    payload = plucker.reshape(-1, 6).to(torch.float32).detach().cpu().numpy()
    span_meta = {"kind": "camera_plucker", "patch": patch,
                "width_px": camera.width, "height_px": camera.height}
    if meta:
        span_meta.update(meta)
    return Span(modality, payload, grid=grid, atomic=False,
               supervised=supervised, observed=not supervised, meta=span_meta)


def anchor_coordinates(camera: Camera, depth: torch.Tensor, patch: int = 8) -> torch.Tensor:
    """World-space ``(x, y, z)`` of each image patch's centre pixel, ``[n_patches, 3]``.

    This is Atlas's "every visual token is anchored to a position in 3D
    space", made concrete: it is the *same* patch centre
    ``camera.rays``/``camera.plucker`` use (``(u + 0.5) * patch``), sampled
    from ``depth`` and unprojected, so index ``i`` here lines up with token
    ``i`` of an ``image`` span built from the same camera, depth and patch
    size. Depth is read at the nearest pixel to each patch centre, not
    averaged over the patch -- averaging depth across a patch that straddles
    a depth discontinuity (an object edge) produces a 3D point that is not on
    any real surface, floating between foreground and background; nearest
    sampling instead reads at least a *real* depth, just possibly not the
    modal one for that patch.

    Feeding this into ``codecs/spatial.py``'s M-RoPE: these are absolute
    world-frame coordinates, a different coordinate system from the
    ``(t, y, x)`` *pixel*-frame coordinates ``Batch.media_coordinates``
    already carries for the same tokens (see ``spatial.py``'s docstring on
    why pixel coordinates and normalized coordinates must not be conflated --
    world and pixel coordinates are the same failure mode one level up).
    They should not overwrite ``media_coordinates``; the two are
    complementary axes on the same token (one says "where in the picture",
    the other "where in the world"). The integration this module recommends
    to the config/model owner: a second ``AxialRotaryEmbedding(axes=("x",
    "y", "z"), ...)`` applied only to tokens carrying anchor coordinates,
    with its rotated contribution combined into the token the same way
    ``continuous_conditioning`` already combines the flow head's timestep
    embedding (``"add"``: project and add) -- not folded into the existing
    ``(t, y, x)`` table, which would require unifying two physically
    different units (pixels, world length) under one ``theta`` schedule.
    """
    if depth.shape != (camera.height, camera.width):
        raise ValueError(
            f"depth must be [{camera.height}, {camera.width}], got {tuple(depth.shape)}")
    dtype = camera.world_to_camera.dtype
    h, w = camera.height // patch, camera.width // patch
    v = (torch.arange(h, dtype=dtype) + 0.5) * patch
    u = (torch.arange(w, dtype=dtype) + 0.5) * patch
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    vi = vv.reshape(-1).long().clamp(0, camera.height - 1)
    ui = uu.reshape(-1).long().clamp(0, camera.width - 1)
    depth_at_centre = depth[vi, ui].reshape(-1, 1).to(dtype)
    uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)
    return camera.unproject(uv, depth_at_centre)


# -- canonical splat ordering -------------------------------------------------


def _spread_bits(x: np.ndarray, bits: int) -> np.ndarray:
    """Insert two zero bits after each bit of ``x`` (Morton/Z-order interleave step)."""
    x = x.astype(np.uint64)
    out = np.zeros_like(x)
    for i in range(bits):
        out |= ((x >> np.uint64(i)) & np.uint64(1)) << np.uint64(3 * i)
    return out


def _morton_codes(points: np.ndarray, bits: int = 21) -> np.ndarray:
    """Z-order code per point, quantised to ``bits`` per axis (fits in uint64 for bits<=21)."""
    mins = points.min(axis=0)
    extent = np.maximum(points.max(axis=0) - mins, 1e-8)
    scale = float((1 << bits) - 1)
    q = np.clip(np.round((points - mins) / extent * scale), 0, scale).astype(np.uint64)
    return (_spread_bits(q[:, 0], bits)
           | (_spread_bits(q[:, 1], bits) << np.uint64(1))
           | (_spread_bits(q[:, 2], bits) << np.uint64(2)))


def scene_span(scene, max_splats: Optional[int] = None, supervised: bool = False) -> Span:
    """A splat scene as a ``[n, 14]`` ``"geometry"`` span in a canonical order.

    14 floats per splat: ``means[3], log_scales[3], quats[4], opacity_logit[1],
    sh0[3]``. ``sh_rest`` (higher-order, view-dependent spherical-harmonic
    coefficients) is not included -- it is optional and variable-width on
    ``GaussianScene``, and folding a ragged extra block into a fixed-width
    per-token feature would either force every scene in a batch to the same
    SH degree or need its own separate span/config entry; out of scope for
    the "diffuse Gaussian as a token" case this covers.

    **Why canonical order matters** (same argument as
    ``codecs/geometry3d.canonicalize`` makes for mesh vertex order): a splat
    scene is a *set*, not a sequence -- ``GaussianScene.concat`` and a
    renderer both treat splat order as irrelevant to the geometry they
    describe. Left alone, N splats have N! equally valid token orderings for
    one scene, and a model conditioned on the sequence would have to learn
    that ordering is meaningless from data instead of never seeing the
    symmetry at all. Ordering is fixed here by Morton (Z-order) code on
    quantised ``means``: interleaving each axis's bits produces a 1-D key
    that keeps spatially nearby splats nearby in the sequence (unlike, say,
    sorting by a hash of the full feature vector, which would scatter a
    physically local cluster of splats across the whole sequence and destroy
    exactly the locality a spectral/windowed block over this span would want
    to exploit) while still being a pure function of splat content.

    When ``max_splats`` truncates the scene, the *selection* (which splats
    survive) is done by ``opacity_logits`` (highest first) before ordering --
    a value-based criterion, so it is itself permutation-invariant to input
    order, unlike e.g. "keep the first N" which is not a property of the
    scene at all.
    """
    means = scene.means.detach().to(torch.float32).cpu().numpy()
    log_scales = scene.log_scales.detach().to(torch.float32).cpu().numpy()
    quats = scene.quats.detach().to(torch.float32).cpu().numpy()
    opacity = scene.opacity_logits.detach().to(torch.float32).cpu().numpy()
    sh0 = scene.sh0.detach().to(torch.float32).cpu().numpy()

    n_total = means.shape[0]
    truncated = False
    if max_splats is not None and n_total > max_splats:
        keep = np.argsort(opacity)[::-1][:max_splats]
        means, log_scales, quats = means[keep], log_scales[keep], quats[keep]
        opacity, sh0 = opacity[keep], sh0[keep]
        truncated = True

    order = np.argsort(_morton_codes(means), kind="stable")
    payload = np.concatenate([
        means[order], log_scales[order], quats[order],
        opacity[order].reshape(-1, 1), sh0[order],
    ], axis=-1).astype(np.float32)

    return Span("geometry", payload, grid=None, atomic=False,
               supervised=supervised, observed=not supervised,
               meta={"kind": "gaussian_splat", "order": "morton_z",
                     "n_total": n_total, "n_kept": payload.shape[0], "truncated": truncated})
