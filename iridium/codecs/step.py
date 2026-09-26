"""STEP (ISO 10303-21) B-rep support: a real Part 21 parser plus a bounded,
honest geometry subset — and the token grammar that subset needs to be
learnable.

Scope, stated up front because a CAD format invites scope creep that quietly
turns into silent data loss:

* The **physical file layer** (header, data section, ``#N=TYPE(args);``
  instances, the reference graph, list/enum/string syntax including the
  ``\\X2\\`` / ``\\X\\`` escapes) is fully general. Any conformant Part 21
  file parses, including files that use entity types this module has never
  heard of.
* The **geometry layer** understands a fixed, listed subset of AP203/214/242
  entities (see ``SUPPORTED_ENTITIES`` below) — enough for a single planar
  or single-curved face with one closed loop, wired up to the product
  structure needed to find it. Anything else is **kept, not dropped**: an
  unsupported instance is preserved as a generic record (its type name and
  raw, ref-resolved argument list) and counted in
  :attr:`StepModel.unsupported`. A round trip through this module therefore
  never silently deletes geometry it does not understand; it can fail loud
  (by the caller checking ``unsupported``) but it will not fail quiet.

Why B-rep tokens and not a construction-history (sketch + extrude + fillet)
representation: B-rep is what STEP *is* — a STEP file already commits to
describing the boundary, not the modelling steps that produced it, so a
tokeniser for STEP has to speak B-rep or translate away information the file
never discarded in the first place. A feature-tree representation (as in
DeepCAD, Fusion 360 Gallery, SkexGen) is the better target when the source
of truth is a *parametric history* — but that history is a different, richer
input STEP does not carry, and building it from a B-rep alone requires
feature recognition (a research problem in itself, not a parsing one). What
a feature-tree buys over what is built here: an editable, semantically
meaningful op sequence (a redo-able "extrude then fillet") instead of a
frozen boundary, far fewer tokens for regular mechanical shapes (a filleted
block is four numbers, not forty vertices), and much better generalisation
to unseen sizes of the same *kind* of part. What it costs: a fundamentally
different (and much narrower) input domain — it targets CAD history logs or
requires feature recognition on a B-rep, whereas the B-rep route here works
on any STEP file that uses the supported entities, including ones with no
recoverable history at all (most STEP files in the wild, since STEP itself
does not store the feature tree).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

# --------------------------------------------------------------------------
# Physical-file value model
# --------------------------------------------------------------------------


class Ref:
    """An unresolved ``#N`` reference. Kept as an id, not a pointer, so the
    forward-reference graph (``#3`` can point at ``#4`` defined later in the
    file — legal and common in STEP) never needs a two-pass fixup inside the
    value parser itself; resolution happens once, after parsing, in
    :func:`build_model`."""

    __slots__ = ("id",)

    def __init__(self, id: int) -> None:
        self.id = id

    def __repr__(self) -> str:
        return f"#{self.id}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Ref) and other.id == self.id

    def __hash__(self) -> int:
        return hash(("Ref", self.id))


class Enum:
    """A STEP enumeration or boolean literal, e.g. ``.T.`` or ``.PLANE.``.
    Kept as a distinct type rather than the bare string ``".T."`` so a
    tokeniser downstream can tell "this argument slot is an enum" from
    "this argument slot happens to contain a string that starts with a dot"
    without re-parsing."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Enum) and other.name == self.name

    def __hash__(self) -> int:
        return hash(("Enum", self.name))


class _Sentinel:
    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:
        return self.label


NULL = _Sentinel("$")       # an explicitly unset optional attribute
DERIVED = _Sentinel("*")    # an attribute EXPRESS declares as derived here


@dataclass
class TypedValue:
    """A "simple defined type" wrapper, e.g. ``LENGTH_MEASURE(3.5)`` or a
    select-type disambiguation. Rare in the bounded subset below but legal
    anywhere a value can appear, so the parser must not choke on it even
    where it is never produced by :func:`write_step_file`."""

    name: str
    args: list


# --------------------------------------------------------------------------
# Tokeniser
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
      (?P<WS>\s+)
    | (?P<STR>'(?:''|\\X2\\[0-9A-Fa-f]*\\X0\\|\\X\\[0-9A-Fa-f]{2}|[^'\\]|\\(?!X))*')
    | (?P<ENUM>\.[A-Za-z_][A-Za-z0-9_]*\.)
    | (?P<REF>\#\d+)
    | (?P<NUM>[+-]?(?:\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?))
    | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<LPAREN>\()
    | (?P<RPAREN>\))
    | (?P<COMMA>,)
    | (?P<SEMI>;)
    | (?P<EQ>=)
    | (?P<DOLLAR>\$)
    | (?P<STAR>\*)
    """,
    re.VERBOSE,
)

_X2_RE = re.compile(r"\\X2\\([0-9A-Fa-f]*)\\X0\\")
_X1_RE = re.compile(r"\\X\\([0-9A-Fa-f]{2})")


def decode_string(literal: str) -> str:
    """Part 21 §6.3.1.4 string unescaping: ``''`` is a literal quote,
    ``\\X\\HH`` is one Latin-1 byte by hex code, ``\\X2\\HHHH...\\X0\\`` is a
    run of one or more UCS-2 code points by hex code, ending the run."""
    inner = literal[1:-1].replace("''", "'")

    def _x2(m: "re.Match[str]") -> str:
        hexs = m.group(1)
        return "".join(chr(int(hexs[i:i + 4], 16)) for i in range(0, len(hexs), 4))

    inner = _X2_RE.sub(_x2, inner)
    inner = _X1_RE.sub(lambda m: chr(int(m.group(1), 16)), inner)
    return inner


def encode_string(value: str) -> str:
    """Exact inverse convention (not the only legal encoding — Part 21 does
    not mandate byte-vs-\\X2\\ choice for a given code point — but a fixed,
    deterministic one, which is what a writer needs)."""
    out: list[str] = []
    for ch in value:
        cp = ord(ch)
        if ch == "'":
            out.append("''")
        elif ch == "\\":
            out.append("\\")
        elif 0x20 <= cp <= 0x7E:
            out.append(ch)
        elif cp <= 0xFF:
            out.append(f"\\X\\{cp:02X}")
        else:
            out.append(f"\\X2\\{cp:04X}\\X0\\")
    return "'" + "".join(out) + "'"


class StepSyntaxError(ValueError):
    pass


def _lex(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    for m in _TOKEN_RE.finditer(text):
        if m.start() != pos:
            raise StepSyntaxError(f"cannot tokenize near {text[pos:pos + 40]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "WS":
            continue
        tokens.append((kind, m.group()))
    if pos != len(text):
        raise StepSyntaxError(f"trailing unparsed text {text[pos:pos + 40]!r}")
    return tokens


class _ValueParser:
    """Recursive-descent parser over the token stream for one statement's
    argument list. One instance per statement (or header entry); state is
    just a cursor into the shared token list."""

    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.toks = tokens
        self.pos = 0

    def _peek(self) -> tuple[Optional[str], Optional[str]]:
        return self.toks[self.pos] if self.pos < len(self.toks) else (None, None)

    def _eat(self, kind: Optional[str] = None) -> str:
        k, v = self._peek()
        if kind is not None and k != kind:
            raise StepSyntaxError(f"expected {kind}, got {k} {v!r} at token {self.pos}")
        self.pos += 1
        return v

    def value(self) -> Any:
        k, v = self._peek()
        if k == "LPAREN":
            return self.list_()
        if k == "DOLLAR":
            self._eat()
            return NULL
        if k == "STAR":
            self._eat()
            return DERIVED
        if k == "REF":
            self._eat()
            return Ref(int(v[1:]))
        if k == "ENUM":
            self._eat()
            return Enum(v)
        if k == "STR":
            self._eat()
            return decode_string(v)
        if k == "NUM":
            self._eat()
            return float(v) if any(c in v for c in ".eE") else int(v)
        if k == "IDENT":
            name = self._eat("IDENT")
            self._eat("LPAREN")
            args = self.arglist()
            self._eat("RPAREN")
            return TypedValue(name, args)
        raise StepSyntaxError(f"unexpected token {k} {v!r} at {self.pos}")

    def list_(self) -> list:
        self._eat("LPAREN")
        items = self.arglist()
        self._eat("RPAREN")
        return items

    def arglist(self) -> list:
        items: list[Any] = []
        if self._peek()[0] == "RPAREN":
            return items
        items.append(self.value())
        while self._peek()[0] == "COMMA":
            self._eat("COMMA")
            items.append(self.value())
        return items

    def at_end(self) -> bool:
        return self.pos >= len(self.toks)


def parse_statements(tokens: list[tuple[str, str]]) -> list[tuple[str, list]]:
    """``NAME(args); NAME(args); ...`` with no ``#N=`` prefix — the header
    section's grammar."""
    p = _ValueParser(tokens)
    out: list[tuple[str, list]] = []
    while not p.at_end():
        name = p._eat("IDENT")
        p._eat("LPAREN")
        args = p.arglist()
        p._eat("RPAREN")
        p._eat("SEMI")
        out.append((name, args))
    return out


@dataclass
class RawInstance:
    """One ``#N=...;`` line, before reference resolution or classification.

    ``kind`` is ``"complex"`` for the multiple-inheritance form
    ``#N=(TYPE1(...)TYPE2(...));`` (used by AP242 for e.g.
    ``GEOMETRIC_REPRESENTATION_ITEM`` mixins) — ``type`` is ``None`` and
    ``args`` holds the ``(subtype_name, subtype_args)`` pairs instead of one
    flat argument list. Complex instances are never in
    ``SUPPORTED_ENTITIES``; they are always preserved generically.
    """

    id: int
    kind: str                    # "simple" | "complex"
    type: Optional[str]
    args: list


def parse_data(tokens: list[tuple[str, str]]) -> list[RawInstance]:
    p = _ValueParser(tokens)
    out: list[RawInstance] = []
    while not p.at_end():
        ref_tok = p._eat("REF")
        inst_id = int(ref_tok[1:])
        p._eat("EQ")
        k, _ = p._peek()
        if k == "LPAREN":
            p._eat("LPAREN")
            parts: list[tuple[str, list]] = []
            while p._peek()[0] != "RPAREN":
                name = p._eat("IDENT")
                p._eat("LPAREN")
                args = p.arglist()
                p._eat("RPAREN")
                parts.append((name, args))
            p._eat("RPAREN")
            p._eat("SEMI")
            out.append(RawInstance(inst_id, "complex", None, parts))
        else:
            name = p._eat("IDENT")
            p._eat("LPAREN")
            args = p.arglist()
            p._eat("RPAREN")
            p._eat("SEMI")
            out.append(RawInstance(inst_id, "simple", name, args))
    return out


_SECTION_RE = {
    "iso": re.compile(r"ISO-10303-21\s*;"),
    "end": re.compile(r"END-ISO-10303-21\s*;"),
    "header": re.compile(r"HEADER\s*;"),
    "endsec": re.compile(r"ENDSEC\s*;"),
    "data": re.compile(r"DATA\s*;"),
}


def _split_sections(text: str) -> tuple[str, str]:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    m_iso = _SECTION_RE["iso"].search(text)
    m_end = _SECTION_RE["end"].search(text)
    if not m_iso or not m_end:
        raise StepSyntaxError("not an ISO-10303-21 physical file (missing markers)")
    body = text[m_iso.end():m_end.start()]
    m_h1 = _SECTION_RE["header"].search(body)
    m_h2 = _SECTION_RE["endsec"].search(body, m_h1.end() if m_h1 else 0)
    if not m_h1 or not m_h2:
        raise StepSyntaxError("missing HEADER section")
    header_text = body[m_h1.end():m_h2.start()]
    rest = body[m_h2.end():]
    m_d1 = _SECTION_RE["data"].search(rest)
    if m_d1:
        m_d2 = _SECTION_RE["endsec"].search(rest, m_d1.end())
        data_text = rest[m_d1.end():m_d2.start()] if m_d2 else rest[m_d1.end():]
    else:
        data_text = ""
    return header_text, data_text


@dataclass
class Header:
    description: list
    name: list
    schema: list
    other: list  # any other header-section entry, preserved verbatim

    def schema_names(self) -> list[str]:
        if not self.schema or not self.schema[0]:
            return []
        return list(self.schema[0])


def parse_header(entries: list[tuple[str, list]]) -> Header:
    description = name = schema = None
    other = []
    for entry_name, args in entries:
        if entry_name == "FILE_DESCRIPTION":
            description = args
        elif entry_name == "FILE_NAME":
            name = args
        elif entry_name == "FILE_SCHEMA":
            schema = args
        else:
            other.append((entry_name, args))
    return Header(description or [], name or [], schema or [], other)


# --------------------------------------------------------------------------
# The bounded geometry subset
# --------------------------------------------------------------------------

#: Positional EXPRESS argument names for every entity this module actually
#: interprets. This *is* the subset: an entity type not listed here is
#: preserved as a generic record (see :class:`StepModel`) and reported, never
#: silently coerced or dropped. Names come from the AP214/AP242 integrated
#: resource schemas (ISO 10303-42), trimmed to what a single-face manifold
#: solid needs.
SUPPORTED_ENTITIES: dict[str, tuple[str, ...]] = {
    # Geometry
    "CARTESIAN_POINT": ("name", "coordinates"),
    "DIRECTION": ("name", "direction_ratios"),
    "VECTOR": ("name", "orientation", "magnitude"),
    "AXIS2_PLACEMENT_3D": ("name", "location", "axis", "ref_direction"),
    "LINE": ("name", "pnt", "dir"),
    "CIRCLE": ("name", "position", "radius"),
    "PLANE": ("name", "position"),
    "CYLINDRICAL_SURFACE": ("name", "position", "radius"),
    # Topology
    "EDGE_CURVE": ("name", "edge_start", "edge_end", "edge_geometry", "same_sense"),
    "ORIENTED_EDGE": ("name", "edge_start", "edge_end", "edge_element", "orientation"),
    "EDGE_LOOP": ("name", "edge_list"),
    "FACE_BOUND": ("name", "bound", "orientation"),
    "FACE_OUTER_BOUND": ("name", "bound", "orientation"),
    "ADVANCED_FACE": ("name", "bounds", "face_geometry", "same_sense"),
    "CLOSED_SHELL": ("name", "cfs_faces"),
    "MANIFOLD_SOLID_BREP": ("name", "outer"),
    # Product/context — not geometry, but the chain a root is found through.
    "PRODUCT": ("name", "description", "id_2", "frame_of_reference"),
    "PRODUCT_DEFINITION_FORMATION": ("id", "description", "of_product"),
    "PRODUCT_DEFINITION": ("id", "description", "formation", "frame_of_reference"),
    "PRODUCT_DEFINITION_SHAPE": ("name", "description", "definition"),
    "SHAPE_DEFINITION_REPRESENTATION": ("definition", "used_representation"),
    "SHAPE_REPRESENTATION": ("name", "items", "context_of_items"),
    "ADVANCED_BREP_SHAPE_REPRESENTATION": ("name", "items", "context_of_items"),
    "APPLICATION_CONTEXT": ("application",),
    "APPLICATION_PROTOCOL_DEFINITION": ("status", "application_interpreted_model_schema_name",
                                         "application_protocol_year", "application"),
    "PRODUCT_CONTEXT": ("name", "frame_of_reference", "discipline_type"),
    "PRODUCT_DEFINITION_CONTEXT": ("name", "frame_of_reference", "life_cycle_stage"),
}

#: Entity types that mark a *shape root* — a MANIFOLD_SOLID_BREP is the
#: geometric root of one solid; a SHAPE_REPRESENTATION is the representation
#: root a PRODUCT hangs its shape off of. :func:`find_roots` starts from
#: these when present rather than relying purely on "nothing references it",
#: because a well-formed file often has *everything* transitively referenced
#: from the header/product chain, which would otherwise report zero roots.
ROOT_ENTITY_TYPES = ("MANIFOLD_SOLID_BREP",)


@dataclass
class StepModel:
    """A parsed, reference-resolved STEP file.

    ``supported`` holds one :class:`Entity` per instance whose type is in
    ``SUPPORTED_ENTITIES``, with positional args resolved to a name->value
    dict and every :class:`Ref` left as a ``Ref`` (resolution to *another*
    ``Entity`` is a lookup through ``by_id``, not inlined, so the structure
    stays a graph rather than an unboundedly-deep tree — a cyclic reference,
    legal in STEP for e.g. mutually-referencing loops, would otherwise
    recurse forever).

    ``unsupported`` holds a :class:`RawInstance` per instance whose type (or
    "COMPLEX" for a multiple-inheritance instance) is not in
    ``SUPPORTED_ENTITIES``. Nothing here is dropped: ``by_id`` also contains
    these, so a supported entity that references an unsupported one still
    resolves, and :func:`write_step_file` re-emits them byte-faithfully.
    """

    header: Header
    by_id: dict[int, RawInstance]
    supported: dict[int, "Entity"]
    unsupported: dict[int, RawInstance]

    def unsupported_report(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for inst in self.unsupported.values():
            key = "COMPLEX" if inst.kind == "complex" else inst.type
            counts[key] = counts.get(key, 0) + 1
        return counts


@dataclass
class Entity:
    id: int
    type: str
    fields: dict[str, Any]


def _resolve(value: Any) -> Any:
    """Refs stay as :class:`Ref`; only the container shape is normalised
    (nested lists resolved recursively so a face's ``bounds`` list is a
    Python list of ``Ref``, not a raw parser list)."""
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    return value


def build_model(text: str) -> StepModel:
    """Parse a full physical file into a :class:`StepModel`."""
    header_text, data_text = _split_sections(text)
    header = parse_header(parse_statements(_lex(header_text)))
    instances = parse_data(_lex(data_text))

    by_id = {inst.id: inst for inst in instances}
    seen_ids = set()
    for inst in instances:
        if inst.id in seen_ids:
            raise StepSyntaxError(f"duplicate instance id #{inst.id}")
        seen_ids.add(inst.id)

    supported: dict[int, Entity] = {}
    unsupported: dict[int, RawInstance] = {}
    for inst in instances:
        if inst.kind == "simple" and inst.type in SUPPORTED_ENTITIES:
            names = SUPPORTED_ENTITIES[inst.type]
            args = inst.args
            if len(args) > len(names):
                # A newer schema revision with extra trailing optional
                # attributes this module does not model: keep the instance,
                # but as unsupported rather than truncating its data.
                unsupported[inst.id] = inst
                continue
            fields = {name: _resolve(v) for name, v in zip(names, args)}
            for name in names[len(args):]:
                fields[name] = NULL
            supported[inst.id] = Entity(inst.id, inst.type, fields)
        else:
            unsupported[inst.id] = inst

    return StepModel(header=header, by_id=by_id, supported=supported, unsupported=unsupported)


def _referenced_ids(model: StepModel) -> set[int]:
    seen: set[int] = set()

    def walk(value: Any) -> None:
        if isinstance(value, Ref):
            seen.add(value.id)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    for inst in model.by_id.values():
        if inst.kind == "simple":
            walk(inst.args)
        else:
            for _, args in inst.args:
                walk(args)
    return seen


def find_roots(model: StepModel) -> list[int]:
    """Instance ids that are shape roots: an explicit
    :data:`ROOT_ENTITY_TYPES` match takes priority; failing that, any
    instance nothing else in the file references (a graph source) is
    reported as a root, which is the right fallback for a file that only
    contains bare geometry with no product wrapper."""
    explicit = [i for i, e in model.supported.items() if e.type in ROOT_ENTITY_TYPES]
    if explicit:
        return sorted(explicit)
    referenced = _referenced_ids(model)
    return sorted(i for i in model.by_id if i not in referenced)


# --------------------------------------------------------------------------
# Token grammar
# --------------------------------------------------------------------------
#
# A flat sequence of small, typed tokens: this is the "typed, canonically
# ordered sequence a sequence model can learn to emit" the task asks for.
# Integer/subword vocabulary assignment (turning e.g. ("NUM", 3.5) into a
# vocabulary id or a continuous slot for a flow head) is left to
# ``iridium/codecs/bank.py`` — the same boundary ``geometry3d.py`` draws
# around ``tokenize()``: this module stops at the grammar, not a trained
# embedding table.
#
# Canonical order: instances are emitted by *ascending original file id*.
# This is a real, documented limitation, not full canonicalisation: unlike
# ``geometry3d.canonicalize``, this order still depends on how the source
# file happened to number its instances, so two STEP files describing the
# identical solid with different numbering yield different token sequences.
# True numbering-independent canonicalisation would need a topological
# order derived from the reference *graph* (e.g. a deterministic DFS from
# each root) plus a tie-break for siblings with no ordering relation, which
# is exactly the kind of combinatorial-canonicalisation problem
# ``geometry3d.py`` solves for meshes; scoping it in here as well was cut to
# keep this module's claims checkable by the round-trip test below rather
# than asserted without one.


def _local_index_map(model: StepModel) -> dict[int, int]:
    ordered = sorted(model.by_id)
    return {orig: i for i, orig in enumerate(ordered)}


def _encode_value(value: Any, index_of: dict[int, int]) -> list:
    """Flatten one value into a flat *list* of primitive tokens. Must return
    a genuinely flat sequence, not a tuple-of-tuples: :func:`_decode_value`
    consumes tokens one at a time off a shared cursor, so a nested-object
    token in the "flat" stream would desync every read after it.
    """
    out: list = []
    _encode_into(value, index_of, out)
    return out


def _encode_into(value: Any, index_of: dict[int, int], out: list) -> None:
    if isinstance(value, Ref):
        out.extend(("REF", index_of[value.id]))
    elif value is NULL:
        out.append("NULL")
    elif value is DERIVED:
        out.append("DERIVED")
    elif isinstance(value, Enum):
        out.extend(("ENUM", value.name))
    elif isinstance(value, bool):  # bool is an int subclass; check first
        out.extend(("BOOL", value))
    elif isinstance(value, int):
        out.extend(("INT", value))
    elif isinstance(value, float):
        out.extend(("FLOAT", value))
    elif isinstance(value, str):
        out.extend(("STR", value))
    elif isinstance(value, list):
        out.extend(("LIST", len(value)))
        for v in value:
            _encode_into(v, index_of, out)
    elif isinstance(value, TypedValue):
        out.extend(("TYPED", value.name, len(value.args)))
        for v in value.args:
            _encode_into(v, index_of, out)
    else:
        raise TypeError(f"cannot tokenize value {value!r}")


def _decode_value(tokens: list, pos: int, index_id: dict[int, int]) -> tuple[Any, int]:
    tag = tokens[pos]
    if tag == "REF":
        return Ref(index_id[tokens[pos + 1]]), pos + 2
    if tag == "NULL":
        return NULL, pos + 1
    if tag == "DERIVED":
        return DERIVED, pos + 1
    if tag == "ENUM":
        return Enum(tokens[pos + 1]), pos + 2
    if tag in ("BOOL", "INT", "FLOAT", "STR"):
        return tokens[pos + 1], pos + 2
    if tag == "LIST":
        n = tokens[pos + 1]
        pos += 2
        items = []
        for _ in range(n):
            v, pos = _decode_value(tokens, pos, index_id)
            items.append(v)
        return items, pos
    if tag == "TYPED":
        name = tokens[pos + 1]
        n = tokens[pos + 2]
        pos += 3
        items = []
        for _ in range(n):
            v, pos = _decode_value(tokens, pos, index_id)
            items.append(v)
        return TypedValue(name, items), pos
    raise StepSyntaxError(f"bad token tag {tag!r} at {pos}")


def tokenize_step(model: StepModel) -> list:
    """Flatten every instance (supported and unsupported alike) into one
    token list, ascending-local-id order (see the note above the
    canonical-order discussion).

    Layout, per instance::

        ENTITY_START, kind_tag, type_name_or_subtype_list, <encoded fields...>, ENTITY_END

    where ``kind_tag`` is ``"SIMPLE"`` or ``"COMPLEX"``. A simple instance's
    fields are its resolved argument list (positional, matching
    ``SUPPORTED_ENTITIES[type]`` order when supported, or the raw parsed
    order when not); a complex instance's fields are
    ``(subtype_name, n_args, encoded_args...)`` repeated per subtype.
    """
    index_of = _local_index_map(model)
    out: list = [len(model.by_id)]
    for orig_id in sorted(model.by_id):
        inst = model.by_id[orig_id]
        out.append("ENTITY_START")
        if inst.kind == "simple":
            out.append("SIMPLE")
            out.append(inst.type)
            out.append(len(inst.args))
            for v in inst.args:
                out.extend(_encode_value(v, index_of))
        else:
            out.append("COMPLEX")
            out.append(len(inst.args))
            for subtype, args in inst.args:
                out.append(subtype)
                out.append(len(args))
                for v in args:
                    out.extend(_encode_value(v, index_of))
        out.append("ENTITY_END")
    return out


def detokenize_step(tokens: list, header: Optional[Header] = None) -> StepModel:
    """Exact inverse of :func:`tokenize_step`. Local ids are renumbered back
    to a contiguous ``#1..#N`` file (original numbering is not recoverable —
    it was discarded on purpose; see the canonical-order note above)."""
    n = tokens[0]
    pos = 1
    index_id = {i: i + 1 for i in range(n)}   # local index -> new file id (1-based)
    raws: list[RawInstance] = []
    for local_id in range(n):
        assert tokens[pos] == "ENTITY_START"
        pos += 1
        kind_tag = tokens[pos]
        pos += 1
        new_id = index_id[local_id]
        if kind_tag == "SIMPLE":
            type_name = tokens[pos]
            n_args = tokens[pos + 1]
            pos += 2
            args = []
            for _ in range(n_args):
                v, pos = _decode_value(tokens, pos, index_id)
                args.append(v)
            raws.append(RawInstance(new_id, "simple", type_name, args))
        else:
            n_sub = tokens[pos]
            pos += 1
            parts = []
            for _ in range(n_sub):
                subtype = tokens[pos]
                n_args = tokens[pos + 1]
                pos += 2
                args = []
                for _ in range(n_args):
                    v, pos = _decode_value(tokens, pos, index_id)
                    args.append(v)
                parts.append((subtype, args))
            raws.append(RawInstance(new_id, "complex", None, parts))
        assert tokens[pos] == "ENTITY_END"
        pos += 1

    by_id = {r.id: r for r in raws}
    supported: dict[int, Entity] = {}
    unsupported: dict[int, RawInstance] = {}
    for r in raws:
        if r.kind == "simple" and r.type in SUPPORTED_ENTITIES:
            names = SUPPORTED_ENTITIES[r.type]
            fields = {name: v for name, v in zip(names, r.args)}
            for name in names[len(r.args):]:
                fields[name] = NULL
            supported[r.id] = Entity(r.id, r.type, fields)
        else:
            unsupported[r.id] = r
    return StepModel(
        header=header or Header([""], ["", "", [], [], "", "", ""], [["UNKNOWN"]], []),
        by_id=by_id, supported=supported, unsupported=unsupported,
    )


# --------------------------------------------------------------------------
# Writer: StepModel -> Part 21 text
# --------------------------------------------------------------------------


def _format_value(value: Any) -> str:
    if isinstance(value, Ref):
        return f"#{value.id}"
    if value is NULL:
        return "$"
    if value is DERIVED:
        return "*"
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, bool):
        return ".T." if value else ".F."
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        s = repr(value)
        return s if ("." in s or "e" in s or "E" in s) else s + "."
    if isinstance(value, str):
        return encode_string(value)
    if isinstance(value, list):
        return "(" + ",".join(_format_value(v) for v in value) + ")"
    if isinstance(value, TypedValue):
        return value.name + "(" + ",".join(_format_value(v) for v in value.args) + ")"
    raise TypeError(f"cannot format value {value!r}")


def write_step_file(model: StepModel) -> str:
    """Render a :class:`StepModel` back to a valid ISO 10303-21 physical
    file. Instances are written in ``by_id`` order (ascending id): if the
    model came from :func:`detokenize_step` that is already the canonical
    local order, and if it came straight from :func:`build_model` it is
    simply the original file's own order.
    """
    lines = ["ISO-10303-21;", "HEADER;"]
    h = model.header
    lines.append(f"FILE_DESCRIPTION({_format_value(h.description)[1:-1]});" if h.description
                 else "FILE_DESCRIPTION();")
    lines.append(f"FILE_NAME({_format_value(h.name)[1:-1]});" if h.name else "FILE_NAME();")
    lines.append(f"FILE_SCHEMA({_format_value(h.schema)[1:-1]});" if h.schema else "FILE_SCHEMA();")
    for name, args in h.other:
        lines.append(f"{name}({_format_value(args)[1:-1]});")
    lines.append("ENDSEC;")
    lines.append("DATA;")
    for inst_id in sorted(model.by_id):
        inst = model.by_id[inst_id]
        if inst.kind == "simple":
            body = ",".join(_format_value(v) for v in inst.args)
            lines.append(f"#{inst_id}={inst.type}({body});")
        else:
            parts = "".join(f"{name}({','.join(_format_value(v) for v in args)})" for name, args in inst.args)
            lines.append(f"#{inst_id}=({parts});")
    lines.append("ENDSEC;")
    lines.append("END-ISO-10303-21;")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------


def parse_step_file(text: str) -> StepModel:
    return build_model(text)


def graph_signature(model: StepModel) -> set:
    """A structural fingerprint used to check "equivalent graph" after a
    round trip: for every instance, its type and its resolved argument
    structure with references replaced by *relative position in sorted id
    order* rather than raw id (so renumbering during
    tokenize/detokenize does not register as a difference), as a hashable
    tuple. Two models with the same signature agree on every instance's
    type, argument values, and reference topology.
    """
    index_of = _local_index_map(model)

    def freeze(value: Any) -> Any:
        if isinstance(value, Ref):
            return ("REF", index_of[value.id])
        if isinstance(value, list):
            return ("LIST", tuple(freeze(v) for v in value))
        if isinstance(value, TypedValue):
            return ("TYPED", value.name, tuple(freeze(v) for v in value.args))
        if isinstance(value, Enum):
            return ("ENUM", value.name)
        if value is NULL:
            return ("NULL",)
        if value is DERIVED:
            return ("DERIVED",)
        return value

    sig = set()
    for local_id, orig_id in enumerate(sorted(model.by_id)):
        inst = model.by_id[orig_id]
        if inst.kind == "simple":
            sig.add((local_id, "simple", inst.type, freeze(inst.args)))
        else:
            sig.add((local_id, "complex", tuple((name, freeze(args)) for name, args in inst.args)))
    return sig
