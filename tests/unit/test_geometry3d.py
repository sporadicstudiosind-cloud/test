"""Mesh loading, canonicalisation and tokenisation. Gate: ordering invariance.

The property under test throughout is the one the module docstring calls
load-bearing: :func:`canonicalize` (and therefore :func:`tokenize`) must not
care how the source file happened to list vertices or faces. Everything else
(loaders, quantisation error, patch encoding) is secondary to that.
"""

from __future__ import annotations

import random
import struct

import numpy as np
import pytest

from iridium.codecs.geometry3d import (
    CanonicalMesh,
    Mesh,
    canonicalize,
    detokenize,
    face_patches,
    load_obj,
    load_ply,
    load_stl,
    to_source_units,
    tokenize,
)

# -- fixtures ---------------------------------------------------------------

# An axis-aligned box with three DISTINCT side lengths. A cube (or any box
# with two equal sides) has a residual point-group symmetry that makes its
# PCA frame genuinely non-unique — not a bug in canonicalize, a fact about
# the shape (a sphere has no PCA frame at all, for the same reason). Real
# meshes are essentially never exactly symmetric at float precision, so the
# permutation tests below use shapes with no such symmetry.
_BOX_VERTICES = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 0.0), (0.0, 2.0, 0.0),
    (0.0, 0.0, 3.0), (1.0, 0.0, 3.0), (1.0, 2.0, 3.0), (0.0, 2.0, 3.0),
]
_BOX_FACES = [
    (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
    (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
]

_TETRA_VERTICES = [
    (0.1, 0.2, 0.3), (1.7, -0.4, 0.2), (-0.3, 1.1, -0.9), (0.4, -0.8, 1.5),
]
_TETRA_FACES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def _permute(vertices, faces, rng: random.Random):
    n = len(vertices)
    perm = list(range(n))
    rng.shuffle(perm)
    inverse = [0] * n
    for new_i, old_i in enumerate(perm):
        inverse[old_i] = new_i
    new_vertices = [vertices[old_i] for old_i in perm]
    new_faces = []
    for face in faces:
        remapped = tuple(inverse[i] for i in face)
        k = rng.randrange(len(remapped))
        new_faces.append(remapped[k:] + remapped[:k])
    rng.shuffle(new_faces)
    return new_vertices, new_faces


# -- Mesh / validate ----------------------------------------------------------


def test_mesh_validate_reports_watertight_box():
    m = Mesh(np.asarray(_BOX_VERTICES), _BOX_FACES)
    v = m.validate()
    assert v.n_vertices == 8 and v.n_faces == 6
    assert v.watertight
    assert v.boundary_edges == 0 and v.non_manifold_edges == 0


def test_mesh_validate_flags_bad_index_and_degenerate():
    m = Mesh(np.asarray(_TETRA_VERTICES), [(0, 1, 1), (0, 1, 99)])
    v = m.validate()
    assert v.degenerate_faces == 1
    assert v.bad_index_faces == 1
    assert not v.watertight


def test_mesh_rejects_non_nx3_vertices():
    with pytest.raises(ValueError):
        Mesh(np.zeros((4, 2)), [(0, 1, 2)])


# -- loaders ------------------------------------------------------------------


def test_load_obj_triangles_and_relative_indices():
    text = """
    # a unit right triangle prism, mixed 1-based/negative indices
    v 0 0 0
    v 1 0 0
    v 0 1 0
    v 0 0 1
    f 1 2 3
    f 1/1/1 2/2/1 4/3/1
    f -4 -3 -1
    """
    m = load_obj(text)
    assert m.n_vertices == 4
    assert m.faces == [(0, 1, 2), (0, 1, 3), (0, 1, 3)]


def _ply_ascii(vertices, faces) -> bytes:
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(vertices)}",
        "property float x", "property float y", "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices", "end_header",
    ]
    for v in vertices:
        lines.append(f"{v[0]} {v[1]} {v[2]}")
    for f in faces:
        lines.append(f"{len(f)} " + " ".join(str(i) for i in f))
    return ("\n".join(lines) + "\n").encode("ascii")


def _ply_binary(vertices, faces) -> bytes:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    body = b"".join(struct.pack("<3f", *v) for v in vertices)
    for f in faces:
        body += struct.pack("<B", len(f)) + struct.pack(f"<{len(f)}i", *f)
    return header + body


def test_load_ply_ascii_and_binary_agree():
    m_ascii = load_ply(_ply_ascii(_TETRA_VERTICES, _TETRA_FACES))
    m_binary = load_ply(_ply_binary(_TETRA_VERTICES, _TETRA_FACES))
    assert m_ascii.n_vertices == m_binary.n_vertices == 4
    assert m_ascii.faces == m_binary.faces == _TETRA_FACES
    assert np.allclose(m_ascii.vertices, m_binary.vertices)


def test_load_ply_skips_undeclared_properties():
    """Colour/normal properties must not desync the binary cursor."""
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "element face 0\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    body = struct.pack("<3f", 1.0, 2.0, 3.0) + bytes([10, 20, 30])
    m = load_ply(header + body)
    assert m.n_vertices == 1
    assert np.allclose(m.vertices[0], [1.0, 2.0, 3.0])


def _stl_ascii(triangles) -> bytes:
    lines = ["solid t"]
    for tri in triangles:
        lines.append("facet normal 0 0 0")
        lines.append("outer loop")
        for v in tri:
            lines.append(f"vertex {v[0]} {v[1]} {v[2]}")
        lines.append("endloop")
        lines.append("endfacet")
    lines.append("endsolid t")
    return ("\n".join(lines) + "\n").encode("ascii")


def _stl_binary(triangles) -> bytes:
    data = b"\x00" * 80 + struct.pack("<I", len(triangles))
    for tri in triangles:
        data += struct.pack("<3f", 0.0, 0.0, 0.0)
        for v in tri:
            data += struct.pack("<3f", *v)
        data += struct.pack("<H", 0)
    return data


_TET_TRIANGLES = [
    (_TETRA_VERTICES[a], _TETRA_VERTICES[b], _TETRA_VERTICES[c])
    for (a, b, c) in _TETRA_FACES
]


def test_load_stl_ascii_and_binary_weld_to_the_same_mesh():
    m_ascii = load_stl(_stl_ascii(_TET_TRIANGLES))
    m_binary = load_stl(_stl_binary(_TET_TRIANGLES))
    assert m_ascii.n_vertices == 4  # welded, not 12
    assert m_binary.n_vertices == 4
    v_ascii = m_ascii.validate()
    v_binary = m_binary.validate()
    assert v_ascii.watertight and v_binary.watertight


def test_stl_ascii_vs_binary_sniff_does_not_misfire_on_binary_containing_bytes():
    # A binary STL's 80-byte header can contain arbitrary bytes, including
    # ones that happen to spell "solid" — the loader must not misdetect it
    # as ASCII on header content alone. Regression guard: pad with the ASCII
    # word "solid" in the header and confirm triangle count still matches.
    raw = _stl_binary(_TET_TRIANGLES)
    poisoned = b"solid " + raw[6:]
    m = load_stl(poisoned)
    assert m.n_faces == 4


# -- canonicalisation: the core property ---------------------------------------


@pytest.mark.parametrize("bits", [6, 8, 9])
def test_canonical_vertices_are_sorted_z_y_x(bits):
    c = canonicalize(Mesh(np.asarray(_BOX_VERTICES), _BOX_FACES), bits=bits)
    keys = [tuple(row) for row in c.codes[:, [2, 1, 0]]]
    assert keys == sorted(keys)


def test_canonicalize_is_invariant_to_vertex_and_face_order():
    rng = random.Random(7)
    base = canonicalize(Mesh(np.asarray(_BOX_VERTICES), _BOX_FACES), bits=8)
    base_tokens = tokenize(base)
    for _ in range(25):
        v2, f2 = _permute(_BOX_VERTICES, _BOX_FACES, rng)
        c2 = canonicalize(Mesh(np.asarray(v2), f2), bits=8)
        assert np.array_equal(tokenize(c2), base_tokens)


def test_canonicalize_is_invariant_for_a_tetrahedron():
    rng = random.Random(11)
    base = canonicalize(Mesh(np.asarray(_TETRA_VERTICES), _TETRA_FACES), bits=8)
    base_tokens = tokenize(base)
    for _ in range(25):
        v2, f2 = _permute(_TETRA_VERTICES, _TETRA_FACES, rng)
        c2 = canonicalize(Mesh(np.asarray(v2), f2), bits=8)
        assert np.array_equal(tokenize(c2), base_tokens)


def test_face_cyclic_rotation_alone_does_not_change_tokens():
    """The narrowest form of the invariance claim, isolated from vertex
    permutation and face-list reordering: rotate one face's own index list
    and nothing else."""
    m1 = Mesh(np.asarray(_BOX_VERTICES), _BOX_FACES)
    rotated_faces = list(_BOX_FACES)
    rotated_faces[0] = rotated_faces[0][2:] + rotated_faces[0][:2]
    m2 = Mesh(np.asarray(_BOX_VERTICES), rotated_faces)
    assert np.array_equal(tokenize(canonicalize(m1)), tokenize(canonicalize(m2)))


def test_canonicalization_scales_into_unit_extent():
    c = canonicalize(Mesh(np.asarray(_BOX_VERTICES), _BOX_FACES), bits=9)
    assert np.all(c.vertices >= -0.5 - 1e-9) and np.all(c.vertices <= 0.5 + 1e-9)
    # The longest axis should span (close to) the full [-0.5, 0.5] extent.
    assert float(c.vertices.max() - c.vertices.min()) == pytest.approx(1.0, abs=1e-2)


def test_canonicalization_welds_coincident_quantized_vertices():
    # A vertex much closer to an existing corner than one quantisation cell
    # must merge into it. Bit depth is kept low (coarse cells) so the merge
    # is not a coin flip against a cell boundary: adding *any* extra vertex
    # perturbs the centroid (the mean over N points shifts by construction
    # when N grows, independent of how close the new point is to an
    # existing one), so this test needs cells much wider than that shift,
    # not merely wider than the source-space epsilon.
    verts = np.asarray(_BOX_VERTICES + [(1e-4, 1e-4, 1e-4)])
    faces = _BOX_FACES + [(0, 1, 8)]
    c = canonicalize(Mesh(verts, faces), bits=3)
    assert c.vertices.shape[0] == 8  # the near-duplicate merged with vertex 0


def test_bits_out_of_range_rejected():
    with pytest.raises(ValueError):
        canonicalize(Mesh(np.asarray(_TETRA_VERTICES), _TETRA_FACES), bits=0)
    with pytest.raises(ValueError):
        canonicalize(Mesh(np.asarray(_TETRA_VERTICES), _TETRA_FACES), bits=20)


# -- quantisation error ---------------------------------------------------


def test_quantization_error_shrinks_with_more_bits():
    rng = np.random.default_rng(3)
    verts = rng.normal(size=(30, 3))
    faces = [(0, 1, 2)]
    errs = []
    for bits in (5, 7, 9):
        c = canonicalize(Mesh(verts, faces), bits=bits)
        errs.append(c.quant_error.max_canonical)
    assert errs[0] > errs[1] > errs[2]
    # Halving the cell size each +1 bit halves worst-case error; check the
    # right order of magnitude rather than an exact factor (welding can
    # perturb the exact vertex set bit depth to bit depth).
    assert errs[0] < 1.0 / (1 << 5)
    assert errs[2] < 1.0 / (1 << 9)


def test_quantization_error_reported_in_source_units():
    verts = np.asarray(_BOX_VERTICES) * 100.0  # box is 100 x 200 x 300 units
    c = canonicalize(Mesh(verts, _BOX_FACES), bits=8)
    assert c.quant_error.max_source_units == pytest.approx(
        c.quant_error.max_canonical * c.xform.scale
    )
    assert c.quant_error.max_source_units > c.quant_error.max_canonical  # scale > 1


# -- tokenize / detokenize round trip -------------------------------------


def test_detokenize_is_exact_inverse_of_tokenize():
    c = canonicalize(Mesh(np.asarray(_BOX_VERTICES), _BOX_FACES), bits=8)
    tokens = tokenize(c)
    back = detokenize(tokens)
    assert np.array_equal(back.codes, c.codes)
    assert np.allclose(back.vertices, c.vertices)
    assert back.faces == c.faces
    assert back.bits == c.bits


def test_tokenize_vocabulary_bound():
    """Every face-stream token is a valid vertex id or the END_FACE sentinel:
    the vocabulary a model needs is exactly n_vertices + 1 symbols."""
    c = canonicalize(Mesh(np.asarray(_TETRA_VERTICES), _TETRA_FACES), bits=7)
    tokens = tokenize(c)
    n_vertices = c.vertices.shape[0]
    header, coords = 3, n_vertices * 3
    face_stream = tokens[header + coords:]
    assert set(np.unique(face_stream).tolist()) <= set(range(n_vertices + 1))
    assert int((face_stream == n_vertices).sum()) == len(c.faces)


def test_to_source_units_recovers_original_scale_and_position():
    verts = np.asarray(_BOX_VERTICES) + np.array([100.0, -50.0, 3.0])
    c = canonicalize(Mesh(verts, _BOX_FACES), bits=9)
    recovered = to_source_units(c)
    # Nearest-vertex round trip error should be within one quantisation cell.
    for i in range(recovered.shape[0]):
        d = np.min(np.linalg.norm(verts - recovered[i], axis=1))
        assert d < 2.0 * c.quant_error.max_source_units + 1e-6


# -- continuous patch encoding ----------------------------------------------


def test_face_patches_shapes_and_frame_orthonormal():
    c = canonicalize(Mesh(np.asarray(_TETRA_VERTICES), _TETRA_FACES), bits=8)
    patches = face_patches(c, ring=1)
    assert len(patches) == len(c.faces)
    for p in patches:
        assert p.frame_axes.shape == (3, 3)
        gram = p.frame_axes @ p.frame_axes.T
        assert np.allclose(gram, np.eye(3), atol=1e-6)
        assert p.features.shape[1] == 3
        assert p.features.shape[0] >= len(p.face)


def test_face_patches_ring_zero_is_just_the_face_corners():
    c = canonicalize(Mesh(np.asarray(_TETRA_VERTICES), _TETRA_FACES), bits=8)
    patches = face_patches(c, ring=0)
    for p in patches:
        assert p.features.shape[0] == len(p.face)


def test_face_patches_rejects_negative_ring():
    c = canonicalize(Mesh(np.asarray(_TETRA_VERTICES), _TETRA_FACES), bits=8)
    with pytest.raises(ValueError):
        face_patches(c, ring=-1)
