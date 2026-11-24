# 3-D geometry: meshes and STEP B-rep

This document covers `iridium/codecs/geometry3d.py` (mesh tokenisation) and
`iridium/codecs/step.py` (STEP/ISO-10303-21 B-rep parsing and tokenisation).
Neither file touches `spans.py`, `bank.py`, or `config.py`; §"Integration
needed" at the end lists exactly what the owner of those files has to add.

## Why the existing `geometry` modality is not enough

`CodecConfig.point_features = 10` gives the `geometry` modality a flat
per-point feature vector — a splat cloud. It has no faces, no edges, no
notion that several points bound one surface. A model trained on it can
learn *where* samples are; it cannot learn *what surface* they close, which
is the entire content of a mesh or a B-rep. The two modules here give the
codec bank something with actual topology to condition on.

## Mesh tokenisation (`geometry3d.py`)

### What is represented

A `Mesh` is vertices (`[N, 3]` float) plus faces (arbitrary-arity polygon
index tuples — a quad stays a quad; triangulating on load would silently
invent a diagonal a non-planar quad's source file never specified).
Loaders exist for OBJ, PLY (ASCII and binary, little/big-endian, arbitrary
skipped properties), and STL (ASCII and binary, with per-triangle vertex
welding since raw STL has no shared-vertex structure at all). None of them
depend on a third-party library.

`Mesh.validate()` reports vertex/face counts, out-of-range or degenerate
faces, boundary and non-manifold edge counts, and a `watertight` boolean —
a real check, not a name that always returns `True`.

### Canonicalisation, and why each rule exists

The same mesh, listed with a different vertex order or a different face
order, is the same *object* — but without canonicalisation it is a different
*token sequence*, and a model then spends capacity discovering the N! · F!
permutation symmetry from data instead of learning shape. `canonicalize()`
removes it in seven steps:

1. **Centre on the centroid.** Translation carries no shape information.
2. **Orient by PCA**, principal axes ordered by descending variance, sign
   fixed by third-moment (skewness) with a lexicographic-extreme-vertex
   fallback when skewness vanishes exactly (a mirror-symmetric point set,
   e.g. any axis-aligned box). Plain PCA only determines axes up to a sign
   flip per axis; without fixing it, two placements of the same object can
   canonicalise to mirror images of each other.
3. **One uniform scale into a unit cube.** Not per-axis: per-axis scaling
   would flatten a thin, long object into a cube and throw away aspect
   ratio, which is real shape information.
4. **Quantise** each axis to `bits` bits (7–9 is the recommended range; see
   the error budget below).
5. **Weld** vertices that quantise to an identical lattice point.
6. **Sort vertices** lexicographically on `(z, y, x)` quantised coordinate —
   a function of the point *set*, never of input order. This is what
   actually kills the vertex-order symmetry.
7. **Cyclically rotate each face** to start at its lowest (post-sort) vertex
   index, then **sort the face list** by the resulting index tuple. Cyclic
   rotation only — never a full re-sort of a face's own indices — because an
   arbitrary permutation of a face's vertices silently reverses its winding
   order and flips its normal, which is real geometry, not a labelling
   artefact.

**Known, honest limitation.** A shape with an exact point-group symmetry
(a perfect cube, a sphere, any object literally invariant under some
non-trivial rotation) has *no* unique PCA frame — not a bug here, a fact:
any position/order-invariant function of an exactly-symmetric point set must
itself respect that symmetry, so several equally valid frames exist and no
canonicalisation can pick a unique one from the point set alone. In practice
this only bites meshes with exact symmetry (rare outside primitive-shape
test cases); `tests/unit/test_geometry3d.py` uses shapes with distinct
principal-axis extents specifically to test the ordinary, non-degenerate
case, and documents the cube case as expected-degenerate rather than
asserting invariance for it.

### Quantisation error budget (measured)

`QuantError` reports max/mean discretisation error in canonical units (where
the mesh spans ≈ 1 unit) and back-projected into source units via the
recovered scale factor. Measured on a synthetic 30-vertex random point set
(`tests/unit/test_geometry3d.py::test_quantization_error_shrinks_with_more_bits`):

| bits | levels | max error (canonical units) |
|-----:|-------:|-----------------------------:|
| 5    | 31     | < 1/32 ≈ 0.031               |
| 7    | 127    | < 1/128 ≈ 0.0078             |
| 9    | 511    | < 1/512 ≈ 0.00195            |

Each `+1` bit halves the worst-case cell size, as expected for uniform
quantisation; the exact numbers are re-measured by the test on every run
rather than hard-coded, since welding can perturb the realised vertex set
bit-depth to bit-depth. For a real-world part of extent 300 mm, 8 bits
(255 levels) puts the worst-case discretisation error at ≈ 1.2 mm; 9 bits
halves that to ≈ 0.6 mm. Pick bits by dividing the smallest feature you need
to preserve into the mesh's largest extent.

### Discrete token stream

`tokenize()` / `detokenize()` implement the PolyGen ordering: vertex
coordinates first (as absolute lattice codes, in canonical sorted order),
then a face stream of back-references into that vertex table, each face
terminated by the sentinel value `n_vertices` (so the face-stream vocabulary
is exactly `n_vertices + 1` symbols, no separate end-of-mesh token needed).
This is exact: `detokenize(tokenize(m))` reproduces `m`'s quantised
vertices and faces bit-for-bit — checked directly in
`test_detokenize_is_exact_inverse_of_tokenize`, and the ordering-invariance
tests (`test_canonicalize_is_invariant_to_vertex_and_face_order`,
`test_canonicalize_is_invariant_for_a_tetrahedron`,
`test_face_cyclic_rotation_alone_does_not_change_tokens`) are the load-
bearing property: 25 random vertex/face-order/face-rotation permutations
per shape, all producing the identical token array.

Coordinates-then-faces (rather than a coordinate triple per face corner, as
raw OBJ stores it) is chosen because a closed mesh shares each vertex among
several faces — a per-corner encoding re-emits the same coordinate 3–4×
for a typical mesh, purely as a side effect of file layout, not because the
model needs to see it again.

### Continuous patch encoding — the flow-matching-friendly alternative

`face_patches()` gives each face a local frame (surface normal from its own
first two edges, an in-plane axis from its longest edge for a determinstic
tie-break, since an arbitrary "first edge" choice would reintroduce the
cyclic-rotation dependency canonicalisation just removed) and expresses its
`ring`-hop vertex neighbourhood as offsets in that frame.

**Tradeoff, spelled out:**

* The discrete stream is *exact and combinatorial* — right for a softmax
  head, because face structure is a discrete choice (this vertex or that
  one), and there is no meaningful interpolation between vertex index 41
  and 42.
* The patch encoding is *local, smooth, and lossy* — right for a
  flow-matching head, because nearby patches vary smoothly (a slightly bent
  hinge is a slightly perturbed patch, not a different token), which is
  exactly the property continuous flow-matching integration needs. What it
  does not give back is exact connectivity beyond `ring` hops: two
  different meshes can produce numerically close patches, a feature for a
  similarity loss and a defect for exact round-tripping. There is
  deliberately no `patches_to_mesh` inverse — it would not be exact and
  claiming otherwise would misrepresent the encoding.

## STEP / B-rep (`step.py`)

### Two layers, scoped separately

**The physical-file layer is fully general.** Any conformant ISO 10303-21
file parses: header, data section, `#N=TYPE(args);` instances (including the
`#N=(TYPE1(...)TYPE2(...))` multiple-inheritance form), the full value
grammar (references, nested lists, `.ENUM.` literals, `$` null, `*` derived,
simple-defined-type wrappers), and both string escape forms — `''` for a
literal quote, `\X\HH` for one Latin-1 byte, `\X2\HHHH...\X0\` for a run of
UCS-2 code points. Forward references (`#3` pointing at `#4` defined later)
need no two-pass fixup: references are kept as `Ref(id)` and resolved by
dictionary lookup after parsing, not inlined during it.

**The geometry layer understands a fixed, listed subset.** Everything
outside it is preserved as a generic record (type name plus ref-resolved
argument list) and counted in `StepModel.unsupported` — never dropped. This
is checked directly: `test_unsupported_entities_preserved_structurally_not_dropped`
confirms a supported `EDGE_CURVE` can still reference an unsupported
`VERTEX_POINT` and the reference still resolves.

### Supported entity subset (exhaustive — `SUPPORTED_ENTITIES` in code)

Geometry:
- `CARTESIAN_POINT`, `DIRECTION`, `VECTOR`
- `AXIS2_PLACEMENT_3D`
- `LINE`, `CIRCLE`, `PLANE`, `CYLINDRICAL_SURFACE`

Topology:
- `EDGE_CURVE`, `ORIENTED_EDGE`, `EDGE_LOOP`
- `FACE_BOUND`, `FACE_OUTER_BOUND`, `ADVANCED_FACE`
- `CLOSED_SHELL`, `MANIFOLD_SOLID_BREP`

Product/context chain (needed only to *find* a root, not geometry itself):
- `PRODUCT`, `PRODUCT_DEFINITION_FORMATION`, `PRODUCT_DEFINITION`,
  `PRODUCT_DEFINITION_SHAPE`
- `SHAPE_DEFINITION_REPRESENTATION`, `SHAPE_REPRESENTATION`,
  `ADVANCED_BREP_SHAPE_REPRESENTATION`
- `APPLICATION_CONTEXT`, `APPLICATION_PROTOCOL_DEFINITION`,
  `PRODUCT_CONTEXT`, `PRODUCT_DEFINITION_CONTEXT`

### Explicitly not supported

- **`VERTEX_POINT`** (and any other topology entity not listed above, e.g.
  `VERTEX_LOOP`, `SEAM_CURVE`, `SURFACE_OF_REVOLUTION`, B-spline surfaces/
  curves, `SPHERICAL_SURFACE`, `TOROIDAL_SURFACE`). Note in particular that
  a real `EDGE_CURVE`'s `edge_start`/`edge_end` point at `VERTEX_POINT`
  instances — this module deliberately does not interpret those, only
  preserves them, since the vertex position is already recoverable from the
  edge's own curve geometry for the bounded curve types it does support.
- Any curve/surface type beyond `LINE`/`CIRCLE` and `PLANE`/
  `CYLINDRICAL_SURFACE` — no B-splines, no NURBS, no swept or revolved
  surfaces.
- Assemblies (`NEXT_ASSEMBLY_USAGE_OCCURRENCE`, multiple `PRODUCT`s wired
  together), colour/layer/PMI annotation entities, units and measure
  entities (`MEASURE_WITH_UNIT`, `(NAMED_UNIT ...)` complex instances —
  handled generically, not interpreted), and multiple-inheritance complex
  instances in general (`#N=(TYPE1(...)TYPE2(...))`) — preserved, reported
  as `"COMPLEX"` in `unsupported_report()`, never interpreted.
- **Canonical (numbering-independent) instance ordering.** The token
  grammar orders instances by ascending *original file id* — a real
  limitation, not full canonicalisation: two STEP files describing the
  identical solid with different instance numbering yield different token
  sequences. A numbering-independent order would need a canonical
  topological sort of the reference graph plus a deterministic tie-break
  for unordered siblings — the same class of combinatorial-canonicalisation
  problem `geometry3d.canonicalize` solves for meshes — and was cut here to
  keep every claim checkable by the round-trip test rather than asserted
  without one.

### B-rep vs. construction-history (feature-tree)

This module implements the **B-rep route**: STEP already commits to
describing the boundary, not the modelling steps that produced it, so a
tokeniser for STEP has to speak B-rep or translate away information the file
never discarded in the first place.

A **feature-tree** representation (sketch → extrude → fillet, as in
DeepCAD / Fusion 360 Gallery / SkexGen) is the better target when the source
of truth is a parametric history. What it would buy over what is built
here: an editable, semantically meaningful, redo-able op sequence instead of
a frozen boundary; far fewer tokens for regular mechanical parts (a filleted
block is four numbers, not forty vertices); and much better generalisation
across sizes of the same *kind* of part. What it costs: a fundamentally
different and narrower input domain — it needs a CAD history log, or feature
recognition performed on a B-rep (itself an open research problem, not a
parsing one) — whereas the B-rep route here works on any STEP file using the
supported entities, including the overwhelming majority of STEP files in
the wild, which do not carry a feature tree at all.

### Token grammar

`tokenize_step()` flattens every instance (supported and unsupported alike)
into one flat token list: per instance, `ENTITY_START`, a `SIMPLE`/`COMPLEX`
kind tag, the type name(s), then each field encoded as a tagged primitive
(`REF` with a *local, contiguous* index — never the raw file id, so a model
never needs file-numbering-scale embeddings for a mesh numbered into the
thousands by some exporter — `NULL`, `DERIVED`, `ENUM`, `BOOL`, `INT`,
`FLOAT`, `STR`, or a length-prefixed `LIST`/`TYPED`), then `ENTITY_END`.
`detokenize_step()` is its exact inverse, and `write_step_file()` renders
the result back to valid ISO 10303-21 text. The chain is checked
end-to-end: parse → tokenize → detokenize → write → reparse must produce a
graph with the identical structural signature (type, resolved field values,
reference topology up to renumbering) as the original — see
`test_full_round_trip_reparses_to_equivalent_graph`, exercised against both
a plain-geometry fixture and one that combines escapes, a forward
reference, and an unsupported entity in the same file.

As with `geometry3d.tokenize`, integer/subword vocabulary assignment for
these tagged tokens (turning `("FLOAT", 3.5)` into a vocabulary id, or a
continuous slot for a flow head) is left to `iridium/codecs/bank.py`; this
module stops at the typed grammar.

## How this would be trained

Both modules produce token/feature streams, not a trained model — training
integration needs the codec-bank hookup below, but the intended objectives:

* **Discrete mesh stream** (`geometry3d.tokenize`): next-token cross-entropy
  over the `n_vertices + 1`-symbol face-stream vocabulary plus a coordinate
  cross-entropy over the `2**bits`-symbol-per-axis vertex stream — exactly
  the softmax-head treatment `text`/`action` already get from `bank.py`.
  Teacher forcing on the canonical order removes the permutation-symmetry
  problem at the loss level, not just at the representation level: without
  canonicalisation, cross-entropy against an arbitrary target ordering would
  penalize an equally-valid differently-ordered prediction as if it were
  wrong.
* **Continuous face patches** (`geometry3d.face_patches`): a flow-matching
  head per patch feature vector, the same treatment `image`/`video`/`field`
  get — appropriate because nearby patches are numerically close for
  similar local geometry, which is what conditional flow matching needs to
  interpolate between during sampling.
* **STEP token grammar** (`step.tokenize_step`): next-token cross-entropy
  over the tagged-token vocabulary, conditioned on the product/context
  chain up to the target `MANIFOLD_SOLID_BREP` root — the natural framing
  for "generate a CAD part matching this text/image description" once a
  modality slot exists for it (see Integration needed).

## Integration needed (for the owner of `config.py` / `spans.py` / `bank.py`)

Nothing in `geometry3d.py` or `step.py` touches these files; the following
is a report, not a change already made.

1. **A `CodecConfig` field for vertex quantisation bits**, e.g.
   `mesh_vertex_bits: int = 8`, feeding `geometry3d.canonicalize(mesh, bits=...)`.
2. **A `CodecConfig` field for the patch neighbourhood ring**, e.g.
   `mesh_patch_ring: int = 1`, feeding `geometry3d.face_patches(mesh, ring=...)`.
3. **Either a new `"mesh"` modality** (discrete face-stream tokens, akin to
   `action`/`text`, vocabulary size `max_vertices_per_mesh + 1`) **or
   reuse of the existing `"geometry"` modality** with `point_features`
   repurposed to the patch feature width from `face_patches` (currently
   `3` — local offset per neighbourhood vertex — times a padded/max ring
   size, or a per-token width matching one `FacePatch.features` row).
   The discrete stream does not fit `"geometry"`'s continuous-only slot in
   `bank.py`'s `continuous_dims()`/`CONTINUOUS` list without either adding
   it to `DISCRETE` in `spans.py` or introducing the new modality above.
4. **A vocabulary slice for the STEP token grammar** if STEP generation is
   wanted as a first-class output: `step.tokenize_step` tokens are typed
   tuples (`("REF", i)`, `("FLOAT", x)`, `("ENUM", name)`, ...), not yet
   mapped to a single integer id space; that mapping is a modelling
   decision (fixed vocabulary vs. hashed vs. per-tag sub-embeddings) this
   module deliberately leaves open.
