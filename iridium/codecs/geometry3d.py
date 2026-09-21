"""Real mesh tokenisation: loading, canonicalisation, quantisation, tokens.

``CodecConfig.point_features = 10`` today buys the "geometry" modality a flat
per-point feature vector — position plus a handful of scalars, no faces, no
topology, no notion that two points are on the same triangle. That is enough
to describe a splat cloud; it is not enough to describe a mesh, because a mesh
*is* its connectivity. A model trained on point features alone can learn where
surface samples are, never what surface they bound.

This module is the alternative literature has converged on: **treat a mesh as
a sequence and let an autoregressive (or flow) head emit it**, the way
PolyGen (Nash et al. 2020) and MeshGPT (Siddiqui et al. 2023) do, rather than
regressing a fixed-size vertex buffer or a signed-distance field. The
constraint that makes this work is the one naive implementations skip:

    the SAME mesh must produce the SAME token sequence, no matter how its
    vertices and faces happened to be listed in the source file.

An OBJ exporter's vertex order is an implementation detail of whichever tool
wrote it — not a property of the shape. Left alone, a cube has 8! x 6!
equally valid token sequences for one geometry, and a sequence model spends
its capacity discovering that permutation symmetry from data instead of
learning shape. :func:`canonicalize` removes the symmetry before the model
ever sees a token, exactly as :func:`~iridium.codecs.spans.patchify` removes
the row/column-major ambiguity of a raster before the image codec sees a
patch.

Two representations are provided, mirroring the two kinds of head the bank
already has (see ``iridium/codecs/bank.py``):

* :func:`tokenize` / :func:`detokenize` — a discrete, exact, PolyGen-style
  face-list stream. Round-trips bit-for-bit at the chosen quantisation. Pairs
  with a softmax head, like ``text`` and ``action``.
* :func:`face_patches` — continuous, per-face local-frame neighbourhoods.
  Lossy (frames are a summary, not an inverse), approximate, but the right
  shape for a flow-matching head: nearby patches vary smoothly, which a
  discrete vertex-index stream deliberately does not (index 41 and index 42
  are not "close" in any sense the model should generalise across). See the
  module-level docstring section "Discrete vs. continuous" below for the
  tradeoff spelled out.

Nothing here touches ``spans.py``, ``bank.py`` or ``config.py``: integration
(a modality registration, a ``CodecConfig`` field for vertex bits and patch
ring size, a vocabulary slice for face tokens) is reported to the owner of
those files rather than made here.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------
# The raw mesh: whatever the file format handed us, before canonicalisation.
# --------------------------------------------------------------------------


@dataclass
class Mesh:
    """Vertices in arbitrary units, faces as tuples of vertex indices.

    Faces are kept as their native polygon (a quad stays a quad) because
    triangulating on load is itself a canonicalisation decision — fan
    triangulation of a non-planar quad is not unique and silently injects a
    diagonal the source file never specified. Triangulation, when a triangle
    stream is required, happens explicitly in :func:`tokenize` and is
    reported in the returned stats, not hidden inside the loader.
    """

    vertices: np.ndarray                     # [N, 3] float64
    faces: list[tuple[int, ...]]

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must be [N, 3], got {self.vertices.shape}")

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def validate(self) -> "MeshValidity":
        """Check index range, degeneracy and closure. Never raises."""
        n = self.n_vertices
        bad_index = 0
        degenerate = 0
        edge_count: dict[tuple[int, int], int] = {}
        for face in self.faces:
            if len(face) < 3:
                degenerate += 1
                continue
            if any(i < 0 or i >= n for i in face):
                bad_index += 1
                continue
            if len(set(face)) != len(face):
                degenerate += 1
                continue
            for a, b in zip(face, face[1:] + face[:1]):
                key = (a, b) if a < b else (b, a)
                edge_count[key] = edge_count.get(key, 0) + 1
        non_manifold_edges = sum(1 for c in edge_count.values() if c > 2)
        boundary_edges = sum(1 for c in edge_count.values() if c == 1)
        return MeshValidity(
            n_vertices=n,
            n_faces=self.n_faces,
            bad_index_faces=bad_index,
            degenerate_faces=degenerate,
            boundary_edges=boundary_edges,
            non_manifold_edges=non_manifold_edges,
            watertight=boundary_edges == 0 and non_manifold_edges == 0 and bad_index == 0,
        )


@dataclass(frozen=True)
class MeshValidity:
    n_vertices: int
    n_faces: int
    bad_index_faces: int
    degenerate_faces: int
    boundary_edges: int
    non_manifold_edges: int
    watertight: bool


# --------------------------------------------------------------------------
# Loaders. No third-party dependency: each format's binary/text layout is
# small enough to parse by hand, and pulling in trimesh to read a triangle
# list would hide exactly the format quirks (STL's per-triangle duplicate
# vertices, PLY's little/big-endian property lists) a mesh codec has to know
# about anyway.
# --------------------------------------------------------------------------


def load_obj(text: str) -> Mesh:
    """Wavefront OBJ: ``v x y z`` and ``f i j k ...`` lines.

    ``f`` indices are 1-based and OBJ allows negative (relative-to-end)
    indices and ``v/vt/vn`` slash groups; only the vertex index is geometry,
    so texture/normal references are read and discarded.
    """
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "v":
            x, y, z = (float(v) for v in parts[1:4])
            vertices.append((x, y, z))
        elif tag == "f":
            idx = []
            for token in parts[1:]:
                vi = int(token.split("/")[0])
                vi = vi - 1 if vi > 0 else len(vertices) + vi
                idx.append(vi)
            faces.append(tuple(idx))
    return Mesh(np.asarray(vertices, dtype=np.float64) if vertices else np.zeros((0, 3)), faces)


def load_ply(data: bytes) -> Mesh:
    """PLY (ASCII or binary_little/big_endian), vertex + face list elements.

    Only the ``x``, ``y``, ``z`` vertex properties and the face
    ``vertex_indices``/``vertex_index`` list property are decoded; any other
    declared property (colour, normal, confidence) is skipped by its declared
    byte width so the parser stays correct without needing to interpret it.
    """
    header_end = data.find(b"end_header\n")
    if header_end < 0:
        raise ValueError("PLY missing end_header")
    header = data[:header_end].decode("ascii", errors="replace")
    body = data[header_end + len(b"end_header\n"):]
    lines = [ln.strip() for ln in header.splitlines() if ln.strip()]
    if not lines or lines[0] != "ply":
        raise ValueError("not a PLY file")
    fmt = "ascii"
    elements: list[dict] = []
    cur = None
    for ln in lines[1:]:
        toks = ln.split()
        if toks[0] == "format":
            fmt = toks[1]
        elif toks[0] == "comment":
            continue
        elif toks[0] == "element":
            cur = {"name": toks[1], "count": int(toks[2]), "props": []}
            elements.append(cur)
        elif toks[0] == "property":
            if toks[1] == "list":
                cur["props"].append(("list", toks[2], toks[3], toks[4]))
            else:
                cur["props"].append(("scalar", toks[1], toks[2]))

    _PLY_TYPES = {
        "char": ("b", 1), "uchar": ("B", 1), "uint8": ("B", 1), "int8": ("b", 1),
        "short": ("h", 2), "ushort": ("H", 2), "uint16": ("H", 2), "int16": ("h", 2),
        "int": ("i", 4), "int32": ("i", 4), "uint": ("I", 4), "uint32": ("I", 4),
        "float": ("f", 4), "float32": ("f", 4), "double": ("d", 8), "float64": ("d", 8),
    }
    endian = "<" if fmt != "binary_big_endian" else ">"

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []

    if fmt == "ascii":
        cursor = body.decode("ascii", errors="replace").split()
        pos = 0

        def next_val(kind: str):
            nonlocal pos
            v = cursor[pos]
            pos += 1
            return float(v) if kind in ("float", "double", "float32", "float64") else int(v)

        for elem in elements:
            for _ in range(elem["count"]):
                record = {}
                for prop in elem["props"]:
                    if prop[0] == "scalar":
                        _, kind, name = prop
                        record[name] = next_val(kind)
                    else:
                        _, count_kind, val_kind, name = prop
                        n = int(next_val(count_kind))
                        record[name] = [next_val(val_kind) for _ in range(n)]
                if elem["name"] == "vertex":
                    vertices.append((record.get("x", 0.0), record.get("y", 0.0), record.get("z", 0.0)))
                elif elem["name"] == "face":
                    key = "vertex_indices" if "vertex_indices" in record else "vertex_index"
                    faces.append(tuple(int(i) for i in record.get(key, [])))
    else:
        off = 0
        for elem in elements:
            for _ in range(elem["count"]):
                record = {}
                for prop in elem["props"]:
                    if prop[0] == "scalar":
                        _, kind, name = prop
                        code, size = _PLY_TYPES[kind]
                        record[name] = struct.unpack_from(endian + code, body, off)[0]
                        off += size
                    else:
                        _, count_kind, val_kind, name = prop
                        ccode, csize = _PLY_TYPES[count_kind]
                        n = struct.unpack_from(endian + ccode, body, off)[0]
                        off += csize
                        vcode, vsize = _PLY_TYPES[val_kind]
                        vals = struct.unpack_from(endian + f"{n}{vcode}", body, off)
                        off += vsize * n
                        record[name] = list(vals)
                if elem["name"] == "vertex":
                    vertices.append((record.get("x", 0.0), record.get("y", 0.0), record.get("z", 0.0)))
                elif elem["name"] == "face":
                    key = "vertex_indices" if "vertex_indices" in record else "vertex_index"
                    faces.append(tuple(int(i) for i in record.get(key, [])))
    return Mesh(np.asarray(vertices, dtype=np.float64) if vertices else np.zeros((0, 3)), faces)


def load_stl(data: bytes) -> Mesh:
    """STL, ASCII or binary. Every STL triangle owns private vertices, so
    this loader welds coincident vertices (exact float equality after a
    fixed-precision round) — otherwise a cube would load as 12 disconnected
    triangles and 36 vertices instead of 8, and :meth:`Mesh.validate` would
    report every edge as a boundary edge.
    """
    text_head = data[:5]
    is_ascii = text_head == b"solid" and b"facet" in data[:2048]
    tris: list[tuple[tuple[float, float, float], ...]] = []
    if is_ascii:
        vals: list[float] = []
        cur: list[tuple[float, float, float]] = []
        for raw_line in data.decode("ascii", errors="replace").splitlines():
            line = raw_line.strip()
            if line.startswith("vertex"):
                x, y, z = (float(v) for v in line.split()[1:4])
                cur.append((x, y, z))
                if len(cur) == 3:
                    tris.append(tuple(cur))
                    cur = []
    else:
        if len(data) < 84:
            raise ValueError("binary STL too short")
        n_tri = struct.unpack_from("<I", data, 80)[0]
        off = 84
        for _ in range(n_tri):
            # normal(3f) + 3*vertex(3f) + attr(2 bytes)
            vals = struct.unpack_from("<9f", data, off + 12)
            tri = ((vals[0], vals[1], vals[2]), (vals[3], vals[4], vals[5]), (vals[6], vals[7], vals[8]))
            tris.append(tri)
            off += 50

    weld: dict[tuple[int, int, int], int] = {}
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for tri in tris:
        idx = []
        for v in tri:
            key = tuple(int(round(c * 1e6)) for c in v)
            j = weld.get(key)
            if j is None:
                j = len(vertices)
                weld[key] = j
                vertices.append(v)
            idx.append(j)
        faces.append(tuple(idx))
    return Mesh(np.asarray(vertices, dtype=np.float64) if vertices else np.zeros((0, 3)), faces)


def load_mesh(path_or_bytes, fmt: str) -> Mesh:
    """Dispatch by declared ``fmt`` in {"obj", "ply", "stl"}. Format is
    declared, not sniffed from a filename extension, because extensions lie
    and a codec that guesses wrong silently mis-parses instead of failing.
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = bytes(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as fh:
            data = fh.read()
    if fmt == "obj":
        return load_obj(data.decode("utf-8", errors="replace"))
    if fmt == "ply":
        return load_ply(data)
    if fmt == "stl":
        return load_stl(data)
    raise ValueError(f"unknown mesh format {fmt!r}")


# --------------------------------------------------------------------------
# Canonicalisation.
# --------------------------------------------------------------------------


@dataclass
class Canonicalization:
    """The affine map from source units to canonical space, kept so results
    can be reported back in the mesh's original units (a quantisation error
    in "units of the unit cube" is meaningless to a CAD user; in millimetres
    it is a spec)."""

    centroid: np.ndarray        # [3], source units
    rotation: np.ndarray        # [3, 3], orthonormal, det = +1
    scale: float                # source-units-per-canonical-unit


@dataclass
class CanonicalMesh:
    """Vertices sorted, faces sorted and index-rotated, coordinates quantised.

    ``vertices`` is the *dequantised* canonical position (float, in
    ``[-0.5, 0.5]``) — what the model would be trained to reconstruct in a
    continuous encoding. ``codes`` is the exact integer lattice point per
    vertex, ``bits`` wide per axis — what the discrete token stream carries.
    """

    vertices: np.ndarray             # [N, 3] float64, canonical + quantised
    codes: np.ndarray                # [N, 3] int64, in [0, 2**bits - 1]
    faces: list[tuple[int, ...]]
    bits: int
    xform: Canonicalization
    quant_error: "QuantError"


@dataclass(frozen=True)
class QuantError:
    """Discretisation error introduced by rounding to the ``bits``-bit lattice.

    Measured, not assumed: this is the actual per-vertex displacement caused
    by snapping continuous canonical coordinates to the nearest lattice
    point, in both canonical units (where the mesh occupies roughly one unit
    of extent) and back-projected into the mesh's original units via
    ``xform.scale``.
    """

    max_canonical: float
    mean_canonical: float
    max_source_units: float
    mean_source_units: float
    bits: int


def _pca_frame(centered: np.ndarray) -> np.ndarray:
    """Deterministic PCA rotation: eigenvectors ordered by descending
    variance, sign fixed by third-moment (skewness) so the *same* point set
    always yields the *same* rotation — plain PCA only fixes axes up to a
    sign flip per axis, which is precisely the ambiguity that would make
    otherwise-identical meshes canonicalise to mirror images of each other.
    """
    n = centered.shape[0]
    if n == 0:
        return np.eye(3)
    cov = (centered.T @ centered) / max(n, 1)
    # Symmetric matrix: eigh gives real, orthonormal eigenvectors.
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]  # columns are the ordered principal axes

    for k in range(3):
        proj = centered @ axes[:, k]
        skew = float(np.sum(proj ** 3))
        if skew < -1e-12:
            axes[:, k] = -axes[:, k]
        elif abs(skew) <= 1e-12:
            # Symmetric distribution along this axis (third moment vanishes
            # identically for e.g. a box or any point set with a mirror
            # plane through the centroid): fall back to the extreme-
            # magnitude vertex. Picking it by ``argmax`` alone is NOT
            # order-invariant when several vertices tie for that extreme
            # (a box's corners all sit at exactly the same |proj| on every
            # principal axis) — argmax would then silently return whichever
            # tied vertex happens to come first in the input array. Instead,
            # break the tie among the tied *set* by the lexicographically
            # largest full 3-D centered coordinate, a criterion that depends
            # only on the point set, never on its listed order.
            proj_abs = np.abs(proj)
            tol = max(1e-9, 1e-6 * float(proj_abs.max() if proj_abs.size else 0.0))
            tied = np.flatnonzero(proj_abs >= proj_abs.max() - tol)
            best = max(tied, key=lambda i: tuple(centered[i]))
            if proj[best] < 0:
                axes[:, k] = -axes[:, k]

    # Fixing each axis independently can leave a reflection (det = -1).
    # Flipping the *last* axis (least-variance, least visually significant)
    # restores a proper rotation without mirroring the shape.
    if np.linalg.det(axes) < 0:
        axes[:, 2] = -axes[:, 2]
    return axes


def canonicalize(mesh: Mesh, bits: int = 8) -> CanonicalMesh:
    """Centre, orient, scale, quantise, then sort into one canonical order.

    Steps, and why each one is load-bearing:

    1. **Centre on the centroid.** Translation carries no shape information;
       leaving it in means two placements of the same object tokenise
       differently for no modelling reason.
    2. **Orient by PCA**, sign-fixed as in :func:`_pca_frame`. A mesh and the
       same mesh rotated 37 degrees about an arbitrary axis are the same
       object; without a canonical frame the model must learn full SO(3)
       invariance from data instead of getting it for free.
    3. **Scale into a unit cube.** One uniform scale factor (not per-axis)
       so aspect ratio — a real shape property — survives; per-axis scaling
       into a cube would flatten a thin long object into a cube and throw
       the aspect information away.
    4. **Quantise** each axis to ``bits`` bits (7-9 is the usual range: 8
       bits is the PolyGen default and the choice below is measured, not
       assumed — see :class:`QuantError`).
    5. **Weld** vertices that quantise to an identical lattice point (two
       close-but-distinct source vertices are, at this resolution, the same
       vertex; keeping both would emit a duplicate-vertex, zero-length-edge
       degenerate face).
    6. **Sort vertices** lexicographically on ``(z, y, x)`` quantised
       coordinate. This key depends only on the point *set*, never on input
       order, so it is what actually kills the N! vertex-order symmetry.
    7. **Remap and cyclically rotate each face** to start at its lowest
       (post-sort) vertex index, **then sort the face list** by the
       resulting index tuple. Cyclic rotation is chosen deliberately over
       full re-sorting a face's own indices: an arbitrary permutation of a
       face's vertices would silently reverse its winding order and flip its
       normal — real geometry, not a labelling artefact — whereas a cyclic
       rotation changes only which vertex the list happens to start from.
    """
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1, 16] (7-9 is the recommended range)")
    v = mesh.vertices
    n = v.shape[0]
    centroid = v.mean(axis=0) if n else np.zeros(3)
    centered = v - centroid
    rotation = _pca_frame(centered)
    rotated = centered @ rotation

    extent = float(np.max(np.abs(rotated))) * 2.0 if n else 1.0
    scale = extent if extent > 1e-12 else 1.0
    normalized = rotated / scale  # now inside [-0.5, 0.5]

    levels = (1 << bits) - 1
    codes_all = np.clip(np.round((normalized + 0.5) * levels), 0, levels).astype(np.int64)
    dequant_all = codes_all.astype(np.float64) / levels - 0.5

    err_canon = np.abs(dequant_all - normalized)
    max_c = float(err_canon.max()) if n else 0.0
    mean_c = float(err_canon.mean()) if n else 0.0
    quant_error = QuantError(
        max_canonical=max_c,
        mean_canonical=mean_c,
        max_source_units=max_c * scale,
        mean_source_units=mean_c * scale,
        bits=bits,
    )

    # Weld by quantised code, first-seen order (arbitrary but irrelevant: the
    # lexicographic sort right after erases whatever order welding used).
    code_key = [tuple(int(x) for x in row) for row in codes_all]
    weld: dict[tuple[int, int, int], int] = {}
    remap = np.empty(n, dtype=np.int64)
    kept_codes: list[tuple[int, int, int]] = []
    for i, key in enumerate(code_key):
        j = weld.get(key)
        if j is None:
            j = len(kept_codes)
            weld[key] = j
            kept_codes.append(key)
        remap[i] = j

    order = sorted(range(len(kept_codes)), key=lambda i: (kept_codes[i][2], kept_codes[i][1], kept_codes[i][0]))
    rank = np.empty(len(kept_codes), dtype=np.int64)
    for new_i, old_i in enumerate(order):
        rank[old_i] = new_i

    final_index = rank[remap]  # per source vertex -> final sorted index
    n_final = len(kept_codes)
    codes = np.zeros((n_final, 3), dtype=np.int64)
    for old_i, key in enumerate(kept_codes):
        codes[rank[old_i]] = key
    vertices = codes.astype(np.float64) / levels - 0.5

    canonical_faces: list[tuple[int, ...]] = []
    for face in mesh.faces:
        remapped = tuple(int(final_index[i]) for i in face)
        if len(set(remapped)) < 3:
            continue  # collapsed by welding: no longer a face
        k = min(range(len(remapped)), key=lambda i: remapped[i])
        rotated_face = remapped[k:] + remapped[:k]
        canonical_faces.append(rotated_face)
    canonical_faces.sort()

    return CanonicalMesh(
        vertices=vertices,
        codes=codes,
        faces=canonical_faces,
        bits=bits,
        xform=Canonicalization(centroid=centroid, rotation=rotation, scale=scale),
        quant_error=quant_error,
    )


# --------------------------------------------------------------------------
# Discrete token stream (PolyGen-style: vertices, then a face index list).
# --------------------------------------------------------------------------

_HEADER_LEN = 3  # n_vertices, n_faces, bits


def tokenize(mesh: CanonicalMesh) -> np.ndarray:
    """Flatten a canonical mesh into one int64 array.

    Layout::

        [n_vertices, n_faces, bits,
         x0, y0, z0, x1, y1, z1, ...,             # n_vertices * 3 codes
         f0_v0, f0_v1, ..., END_FACE,
         f1_v0, ..., END_FACE,
         ...]

    Vertex codes are emitted in the already-sorted (canonical) order, so a
    downstream model reads "vertex 0" as an absolute lattice point, not a
    pointer — coordinates first, faces second, is the PolyGen ordering,
    chosen because it lets face tokens be pure back-references into a vertex
    table the model has already committed to, rather than re-emitting
    coordinates per face-corner as an OBJ file does (3-4x more tokens for a
    typical closed mesh, since each vertex is shared by several faces).

    ``END_FACE`` is the sentinel value ``n_vertices`` (one past the largest
    valid vertex id), so the vocabulary a model needs is exactly
    ``n_vertices + 1`` symbols for the face stream — no separate "end of
    mesh" token is needed because the header already declares ``n_faces``.
    """
    end_face = mesh.vertices.shape[0]
    out = [mesh.vertices.shape[0], len(mesh.faces), mesh.bits]
    out.extend(int(c) for row in mesh.codes for c in row)
    for face in mesh.faces:
        out.extend(face)
        out.append(end_face)
    return np.asarray(out, dtype=np.int64)


def detokenize(tokens: np.ndarray) -> CanonicalMesh:
    """Exact inverse of :func:`tokenize`.

    Exact up to the affine transform: this recovers the quantised canonical
    mesh, not the original source-unit mesh, because :func:`tokenize` never
    carried the centroid/rotation/scale (those are floats a discrete
    token stream has no slot for; a training pipeline that needs them back
    stores :class:`Canonicalization` alongside the tokens as sample metadata,
    the same way image codec tiles carry ``source_size`` in ``Span.meta``
    rather than in the payload itself).
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    n_vertices, n_faces, bits = (int(x) for x in tokens[:_HEADER_LEN])
    cursor = _HEADER_LEN
    codes = tokens[cursor:cursor + n_vertices * 3].reshape(n_vertices, 3)
    cursor += n_vertices * 3
    levels = (1 << bits) - 1
    vertices = codes.astype(np.float64) / levels - 0.5

    end_face = n_vertices
    faces: list[tuple[int, ...]] = []
    cur: list[int] = []
    while len(faces) < n_faces:
        tok = int(tokens[cursor])
        cursor += 1
        if tok == end_face:
            faces.append(tuple(cur))
            cur = []
        else:
            cur.append(tok)

    return CanonicalMesh(
        vertices=vertices,
        codes=codes,
        faces=faces,
        bits=bits,
        xform=Canonicalization(centroid=np.zeros(3), rotation=np.eye(3), scale=1.0),
        quant_error=QuantError(0.0, 0.0, 0.0, 0.0, bits),
    )


def to_source_units(mesh: CanonicalMesh) -> np.ndarray:
    """Map canonical (quantised) vertices back through ``xform`` into the
    coordinate system :func:`canonicalize` started from."""
    x = mesh.xform
    return (mesh.vertices * x.scale) @ x.rotation.T + x.centroid


# --------------------------------------------------------------------------
# Continuous patch encoding (the flow-matching-friendly alternative).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FacePatch:
    face: tuple[int, ...]
    frame_origin: np.ndarray       # [3]
    frame_axes: np.ndarray         # [3, 3], rows = (normal, u, v)
    features: np.ndarray           # [n_ring_verts, 3] local coordinates


def face_patches(mesh: CanonicalMesh, ring: int = 1) -> list[FacePatch]:
    """One local-frame neighbourhood patch per face.

    Discrete-vs-continuous tradeoff (why this function exists alongside
    :func:`tokenize` rather than instead of it):

    * The discrete stream (:func:`tokenize`) is **exact and combinatorial**.
      ``detokenize(tokenize(m))`` reproduces ``m`` bit-for-bit. It suits a
      softmax head because face structure is a discrete choice (this vertex
      or that one), not a continuous quantity — there is no meaningful
      interpolation between vertex index 41 and 42.
    * This patch encoding is **local, smooth, and lossy**. Each face gets its
      own frame (normal, plus an in-plane basis picked from the longest
      edge for determinism) and expresses its ring of neighbouring vertices
      as offsets in that frame — translation- and rotation-invariant at the
      patch level, the same reason ``field`` patches in
      ``iridium/codecs/high_resolution.py`` carry coordinates rather than
      raw pixel position. It suits a flow-matching head because nearby
      patches vary smoothly (a slightly bent hinge is a slightly perturbed
      patch, not a different token), which is exactly the property CFM
      needs to interpolate between samples during integration. What it does
      **not** give you back is exact connectivity beyond ``ring`` hops: two
      different meshes can produce numerically close patches, which is a
      feature for a similarity-seeking loss and a defect for round-tripping.

    ``ring`` is the topological radius (in shared-vertex hops) of the
    neighbourhood pulled in around each face; ``ring=1`` is the face's own
    three-or-more corners plus every vertex on a face that shares an edge
    with it.
    """
    if ring < 0:
        raise ValueError("ring must be >= 0")
    adjacency: dict[int, set[int]] = {}
    for face in mesh.faces:
        for v in face:
            adjacency.setdefault(v, set()).update(f for f in face if f != v)

    patches: list[FacePatch] = []
    for face in mesh.faces:
        verts = set(face)
        frontier = set(face)
        for _ in range(ring):
            nxt = set()
            for v in frontier:
                nxt |= adjacency.get(v, set())
            frontier = nxt - verts
            verts |= nxt
        ring_ids = sorted(verts)  # deterministic: vertex ids are already canonical

        pts = mesh.vertices[list(face)]
        origin = pts.mean(axis=0)
        e0 = pts[1] - pts[0]
        e1 = pts[2] - pts[0]
        normal = np.cross(e0, e1)
        norm_len = np.linalg.norm(normal)
        normal = normal / norm_len if norm_len > 1e-12 else np.array([0.0, 0.0, 1.0])
        # In-plane axis: the longest edge, projected out of the normal, for
        # a deterministic in-plane basis (an arbitrary "first edge" choice
        # would rotate the patch frame under a cyclic re-rotation of the
        # face's own index list, which is a symmetry canonicalize() already
        # removed and this must not reintroduce).
        edges = [(pts[(i + 1) % len(face)] - pts[i], i) for i in range(len(face))]
        longest, _ = max(edges, key=lambda e: np.linalg.norm(e[0]))
        u = longest - normal * np.dot(longest, normal)
        u_len = np.linalg.norm(u)
        u = u / u_len if u_len > 1e-12 else np.array([1.0, 0.0, 0.0])
        w = np.cross(normal, u)
        axes = np.stack([normal, u, w])

        local = (mesh.vertices[ring_ids] - origin) @ axes.T
        patches.append(FacePatch(face=face, frame_origin=origin, frame_axes=axes, features=local))
    return patches
