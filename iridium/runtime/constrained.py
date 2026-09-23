"""Schema-constrained decoding: make an invalid tool call unrepresentable.

An unconstrained model can emit any byte at any position, and everything this
project's own :class:`~iridium.model.typed_head.TypedHead` says about masking
applies here too: a mask that makes an invalid output structurally impossible
is a different and stronger guarantee than one that merely down-weights it,
because it holds *regardless of how well trained the model is*. That is the
whole value proposition of constraining a tool call's JSON at decode time
rather than only validating it after the fact (as
``iridium.runtime.tools.ToolRegistry.validate_arguments`` still does, as a
second, independent line of defense): a garbled call from a barely-trained
model is not a training problem this module claims to fix, it is a decoding
problem this module actually does fix — see
``tests/unit/test_constrained.py``'s untrained-tiny-model test, which is the
one claim this module stands behind.

**What is checked, byte by byte, as generation proceeds:** whether the bytes
emitted *so far* are a valid prefix of *some* complete JSON value conforming
to a schema. "Prefix" is the operative word — ``{"a"`` is not valid JSON, but
it is the unique valid prefix of every schema-conformant object whose first
key is ``"a"``, and the whole point of checking prefixes rather than only
complete values is that the check has to run *before* the value is finished,
while there is still a byte left to mask.

**Supported JSON-Schema subset — read this before calling anything else in
this module, because unsupported input raises at construction, not partway
through a decode:**

* ``{"type": "object", "properties": {...}, "required": [...]}`` — every
  property's schema must itself be in this subset (arbitrary nesting depth is
  fine; ``TypedHead``'s flat-struct-only restriction does not apply here).
  **Unknown keys are always rejected** — this engine behaves as
  ``"additionalProperties": false`` unconditionally; a schema that explicitly
  asks for ``"additionalProperties": true`` is rejected at construction
  rather than silently honored or silently ignored.
* ``{"type": "string"[, "enum": [...]][, "maxLength": N]}`` — with an
  ``enum``, a partial string is rejected the moment no enum member could
  still complete it (the mechanism that makes an enum-constrained field
  *forceable*, not just checkable after the close quote). ``maxLength`` is
  enforced the same way, byte-count on the (unescaped) content, and is the
  one keyword outside plain JSON Schema's string type this engine adds
  meaning to at decode time rather than only checking after the close quote
  — see the untrained-model termination note on :func:`generate_constrained_json`
  for why an *unbounded* string field is a real practical hazard this
  keyword exists to let a schema author opt out of.
* ``{"type": "number"}`` / ``{"type": "integer"}`` — JSON number grammar;
  ``minimum``/``maximum``/``multipleOf`` are **not** enforced (documented, not
  silently dropped: a schema using them still validates, just not on range).
* ``{"type": "boolean"}``, ``{"type": "null"}``.
* ``{"type": "array", "items": <schema>[, "minItems", "maxItems"]}`` — one
  homogeneous item schema; a *tuple*-typed ``items`` (a list of per-position
  schemas) is not supported.
* ``{"type": "any"}`` — this engine's own extension, not standard JSON Schema:
  accepts any JSON value, dispatching on the next non-whitespace byte. Used
  internally wherever a sub-schema is genuinely unconstrained; also usable
  directly by a caller that wants "some JSON here, shape unknown."

**Rejected outright, at construction, every time:** ``oneOf``/``anyOf``/
``allOf``/``not``/``if``/``then``/``else``, ``$ref``, a list-of-types
(``["string", "null"]``), ``pattern``, ``patternProperties``, ``multipleOf``,
``uniqueItems``, ``const``, ``dependentRequired``, ``propertyNames``, and any
schema without exactly one ``"type"``. This is "fail closed" in the literal
sense the brief asks for: a schema this engine cannot enforce raises
immediately rather than being partially enforced and silently under-checked.

**Two decoders, two vocabularies:**

* :func:`generate_constrained_json` — for this project's actual text codec
  (``iridium.codecs.spans.text_span``, byte-level): checks all 256 possible
  next bytes every step. Cheap, because 256 is small, and exact — the
  validator itself is byte-oriented (see below), not a string one, so there
  is no encode/decode round-trip to reason about.
* :func:`bpe_candidate_mask` — for a subword vocabulary
  (``iridium.data.tokenizer.BytePairTokenizer``), which can have tens of
  thousands of ids: checking every one of them per step the way the byte path
  checks all 256 is a cost this module declines to pay, so only a caller-
  supplied top-``k`` of already-ranked candidates is checked, with the byte
  path itself as the honest fallback for the (documented, real) case where
  none of the top-``k`` survives. See that function's docstring for exactly
  what this does and does not guarantee, and why it is exercised directly
  against a tokenizer rather than through a live model in this codebase
  today — this project's text head is byte-level end to end
  (``iridium.runtime.generate``), so there is no BPE-native decode loop yet
  for this to plug into.

**Why the validator works on ``bytes``, not ``str``.** JSON's own grammar
(braces, colons, digits, ``true``/``false``/``null``) is pure ASCII, and RFC
8259 allows any raw UTF-8 byte $\\ge$ 0x20 to sit unescaped inside a string.
Operating on raw bytes means a multi-byte UTF-8 character never needs to be
*decoded* to be validated — every byte of it is simply ordinary string
content, checked the same way an ASCII byte is. The one thing this does not
check is that those content bytes form well-formed UTF-8 at all; a model that
emits a broken UTF-8 sequence inside a string value produces a call this
validator still calls valid and ``json.loads`` would decode with byte loss
under a permissive error handler. Stated here rather than assumed away: it is
irrelevant for the ASCII tool names, enum members, numbers and punctuation
this module is built to constrain, and checking full UTF-8 well-formedness
byte-by-byte would cost real complexity for a case this project's tool-call
data does not exercise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

import torch

__all__ = [
    "SchemaPrefixValidator", "ToolCallValidator",
    "generate_constrained_json", "generate_constrained_call",
    "bpe_candidate_mask",
]

_ANY_SCHEMA: dict = {"type": "any"}
_SUPPORTED_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null", "any"}
_UNSUPPORTED_KEYWORDS = (
    "oneOf", "anyOf", "allOf", "not", "$ref", "if", "then", "else", "pattern",
    "patternProperties", "multipleOf", "uniqueItems", "const",
    "dependentRequired", "propertyNames",
)

Schema = dict  # a supported-subset JSON-Schema dict; see the module docstring
PropertySchema = Union[Schema, Callable[[dict], Optional[Schema]]]


def _validate_supported(schema: Any, path: str = "<root>") -> None:
    """Raise :class:`ValueError` if ``schema`` uses anything outside the
    documented subset. Called eagerly — at :class:`SchemaPrefixValidator`
    construction, never mid-decode — so an unsupported schema is refused
    before a single byte is generated against it, not discovered three tool
    calls into a run."""
    if callable(schema):
        return  # a dynamic property resolver (ToolCallValidator); checked
                 # when it is actually invoked, since its return value depends
                 # on data not known until decode time.
    if not isinstance(schema, dict):
        raise ValueError(f"{path}: schema must be a dict, got {type(schema).__name__}")
    t = schema.get("type")
    if isinstance(t, (list, tuple)):
        raise ValueError(f"{path}: a list of types (e.g. [\"string\",\"null\"]) is not "
                         "supported; every schema must declare exactly one type")
    if t is None or t not in _SUPPORTED_TYPES:
        raise ValueError(
            f"{path}: unsupported or missing 'type' ({t!r}); supported: "
            f"{sorted(_SUPPORTED_TYPES - {'any'})}"
        )
    for bad in _UNSUPPORTED_KEYWORDS:
        if bad in schema:
            raise ValueError(f"{path}: unsupported JSON-Schema keyword {bad!r}")
    if schema.get("additionalProperties") is True:
        raise ValueError(
            f"{path}: additionalProperties: true is not supported — this engine "
            "always behaves as additionalProperties: false"
        )
    if t == "object":
        for name, sub in schema.get("properties", {}).items():
            _validate_supported(sub, f"{path}.{name}")
    if t == "array":
        items = schema.get("items")
        if items is not None:
            if isinstance(items, (list, tuple)):
                raise ValueError(
                    f"{path}: tuple-typed 'items' (a list of per-position schemas) "
                    "is not supported; use one homogeneous item schema"
                )
            _validate_supported(items, f"{path}[]")
    if t == "string" and "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, (list, tuple)) or not all(isinstance(v, str) for v in enum):
            raise ValueError(f"{path}: enum must be a list of strings")


# --------------------------------------------------------------------------
# the byte-level incremental parser
#
# Every _parse_* function takes (buf: bytes, i: int, ...) and returns
# (status, pos, value) where status is one of:
#   "invalid"   -- buf[i:] cannot be extended to match this schema
#   "incomplete"-- buf[i:] is a valid-so-far prefix; more bytes could complete it
#   "complete"  -- buf[i:pos] is a complete, schema-valid value
# `pos`/`value` are meaningful only when status == "complete". This mirrors
# exactly the three-way answer schema-constrained decoding needs at every
# step ("still fine", "definitely broken", "done") and nothing else.
# --------------------------------------------------------------------------

_WS = frozenset(b" \t\n\r")

#: JSON's grammar permits *unlimited* insignificant whitespace between any two
#: tokens, and every whitespace byte is, by that grammar, a valid extension of
#: any prefix at all -- which makes "pad forever with whitespace" a
#: perfectly JSON-legal escape hatch from ever having to commit to real
#: content. A greedy decoder over an undertrained model's arbitrary logits can
#: and (see ``tests/unit/test_constrained.py``'s regression) does find that
#: escape hatch: nothing about validity favours the closing token over one
#: more space. Capping consecutive whitespace is a deliberate, documented
#: deviation from strict JSON Schema/JSON grammar rather than a bug -- past
#: this many consecutive whitespace bytes, this engine calls the prefix
#: invalid, which is what forces the decoder to commit to a real token
#: instead. Four bytes tolerates ordinary pretty-printed indentation; the
#: wire format this project actually trains and decodes
#: (``iridium.runtime.tools.format_tool_call``) has no whitespace at all.
_MAX_WS_RUN = 4


def _skip_ws(buf: bytes, i: int) -> int:
    n = len(buf)
    stop = min(n, i + _MAX_WS_RUN)
    while i < stop and buf[i] in _WS:
        i += 1
    return i


def _parse_literal(buf: bytes, i: int, word: bytes, value: Any):
    n = len(buf)
    avail = n - i
    if avail <= 0:
        return "incomplete", i, None
    take = min(avail, len(word))
    if buf[i:i + take] != word[:take]:
        return "invalid", i, None
    if take < len(word):
        return "incomplete", i, None
    return "complete", i + len(word), value


def _parse_boolean(buf: bytes, i: int):
    if i >= len(buf):
        return "incomplete", i, None
    c = buf[i]
    if c == 0x74:  # 't'
        return _parse_literal(buf, i, b"true", True)
    if c == 0x66:  # 'f'
        return _parse_literal(buf, i, b"false", False)
    return "invalid", i, None


def _parse_null(buf: bytes, i: int):
    if i >= len(buf):
        return "incomplete", i, None
    if buf[i] != 0x6E:  # 'n'
        return "invalid", i, None
    return _parse_literal(buf, i, b"null", None)


def _parse_number(buf: bytes, i: int, integer_only: bool = False):
    """JSON number grammar. A number has no explicit terminator in JSON — its
    own length is genuinely ambiguous until a non-digit byte is seen or input
    ends — so this reports "incomplete" whenever the digits-so-far reach the
    end of ``buf`` (it might still grow) and "complete" the moment a
    structurally valid number is followed by *any* other byte, leaving it to
    the caller (which always knows what must follow a value: a comma, a
    closing bracket, or end of input) to decide whether that next byte is
    itself acceptable."""
    n = len(buf)
    if i >= n:
        return "incomplete", i, None
    j = i
    if buf[j] == 0x2D:  # '-'
        j += 1
    if j >= n:
        return "incomplete", i, None
    if buf[j] == 0x30:  # '0'
        j += 1
    elif 0x31 <= buf[j] <= 0x39:
        j += 1
        while j < n and 0x30 <= buf[j] <= 0x39:
            j += 1
    else:
        return "invalid", i, None
    if not integer_only:
        if j < n and buf[j] == 0x2E:  # '.'
            k = j + 1
            if k >= n:
                return "incomplete", i, None
            if not (0x30 <= buf[k] <= 0x39):
                return "invalid", i, None
            k += 1
            while k < n and 0x30 <= buf[k] <= 0x39:
                k += 1
            j = k
        if j < n and buf[j] in (0x65, 0x45):  # 'e'/'E'
            k = j + 1
            if k < n and buf[k] in (0x2B, 0x2D):
                k += 1
            if k >= n:
                return "incomplete", i, None
            if not (0x30 <= buf[k] <= 0x39):
                return "invalid", i, None
            k += 1
            while k < n and 0x30 <= buf[k] <= 0x39:
                k += 1
            j = k
    if j >= n:
        return "incomplete", i, None
    text = buf[i:j].decode("ascii")
    value = int(text) if integer_only or ("." not in text and "e" not in text and "E" not in text) else float(text)
    return "complete", j, value


_ESCAPES = {0x22: 0x22, 0x5C: 0x5C, 0x2F: 0x2F, 0x62: 0x08, 0x66: 0x0C,
            0x6E: 0x0A, 0x72: 0x0D, 0x74: 0x09}


def _parse_string(buf: bytes, i: int, enum: Optional[Sequence[str]] = None,
                   max_length: Optional[int] = None):
    """The six single-character JSON escapes (``\\" \\\\ \\/ \\b \\f \\n \\r
    \\t``) are supported; ``\\uXXXX`` is not (see the inline comment on that
    branch for why) and makes a string invalid the moment it appears — not a
    silent gap, since any character it could represent can already be
    written as a raw UTF-8 byte inside a JSON string unescaped."""
    n = len(buf)
    if i >= n:
        return "incomplete", i, None
    if buf[i] != 0x22:
        return "invalid", i, None
    i += 1
    content = bytearray()
    enc = [e.encode("utf-8") for e in enum] if enum is not None else None
    while True:
        if i >= n:
            return "incomplete", i, None
        c = buf[i]
        if max_length is not None and len(content) >= max_length and c != 0x22:
            # Already at the length cap: the *only* byte that can still be
            # valid here is the closing quote, checked next. A backslash
            # would otherwise be accepted as "incomplete" (a bare backslash
            # always is, pending its escape partner) without regard to the
            # cap already being full, which is exactly the gap this module's
            # own regression test found: a maxLength-bounded field that had
            # already reached its cap still accepted one more escape
            # sequence because the cap was only ever checked *after* a
            # content byte was appended, never before deciding to start one.
            return "invalid", i, None
        if c == 0x22:
            i += 1
            if enc is not None:
                if bytes(content) not in enc:
                    return "invalid", i, None
                return "complete", i, bytes(content).decode("utf-8")
            try:
                text_value = bytes(content).decode("utf-8")
            except UnicodeDecodeError:
                text_value = None  # opaque but structurally valid; see docstring
            return "complete", i, text_value
        if c == 0x5C:
            if enc is not None:
                # An enum-constrained string (this includes an object key
                # checked against known property names -- see
                # ``_parse_object``) never needs an escape: every enum member
                # and every property name this engine's own callers pass it
                # is a plain identifier-like string with no character that
                # requires JSON escaping. Refusing a backslash outright here,
                # rather than resolving it and *then* pruning, is what keeps
                # the enum-prefix pruning exact: pruning after the fact would
                # have to reason about which of up to eight possible escape
                # resolutions could still match a remaining candidate, and a
                # bare backslash is a dead end for every escape this engine
                # supports in practice anyway (none of them resolve to a
                # letter), so a model that starts one here has already
                # committed to a string no enum member can match.
                return "invalid", i, None
            if i + 1 >= n:
                return "incomplete", i, None
            e = buf[i + 1]
            if e in _ESCAPES:
                content.append(_ESCAPES[e])
                i += 2
            else:
                # ``\uXXXX`` is valid JSON but is deliberately not supported
                # here: checking it correctly needs to reject a bad hex digit
                # the moment it appears rather than buffering all four and
                # checking once (an earlier version buffered, and a greedy
                # decoder over an untrained model happily typed non-hex
                # filler for three bytes before the check ever ran, since
                # "incomplete" is genuinely the honest answer for a partial
                # buffer under that design). Any raw non-ASCII character can
                # already be written unescaped inside a JSON string (RFC 8259
                # permits any UTF-8 byte >= 0x20), so \u is not needed to
                # represent one and this engine fails closed on it rather
                # than re-solving the buffering problem for a rarely-needed
                # escape.
                return "invalid", i, None
        elif c < 0x20:
            return "invalid", i, None  # raw control bytes must be escaped
        else:
            content.append(c)
            i += 1
        # The bound checks below apply after *every* content-extending path
        # above, escaped or raw alike -- an earlier version of this function
        # only ran them after a raw byte, which let an escape sequence (e.g.
        # an endless run of ``\\``, each one a legal two-byte escape for a
        # single backslash) dodge both maxLength and enum pruning entirely.
        # ``tests/unit/test_constrained.py``'s untrained-model regression
        # test is what caught that: a greedy decoder over arbitrary logits
        # found exactly this escape hatch before this fix.
        if max_length is not None and len(content) > max_length:
            # Fail closed the instant the bound is crossed, not just at the
            # close quote: this is what makes maxLength an actual decode-time
            # bound (forcing termination) rather than a check that only ever
            # fires after the fact.
            return "invalid", i, None
        if enc is not None and not any(cand.startswith(bytes(content)) for cand in enc):
            # No enum member can still complete this prefix: fail closed
            # right here rather than waiting for the closing quote. This is
            # what makes an enum genuinely *forceable* during decoding.
            return "invalid", i, None


def _resolve_property_schema(entry: PropertySchema, obj_so_far: dict) -> Schema:
    if callable(entry):
        resolved = entry(dict(obj_so_far))
        return resolved if resolved is not None else _ANY_SCHEMA
    return entry if entry is not None else _ANY_SCHEMA


#: Real JSON nesting has no grammar-level bound, and neither did an earlier
#: version of this parser: ``{"type": "any"}`` (reachable whenever a
#: ``ToolCallValidator`` has not yet resolved which tool's schema governs
#: ``"arguments"`` -- see that class's docstring) accepts an array of arrays
#: of arrays without limit, and a greedy decoder over an untrained model can
#: and did (this module's own regression test caught it) pick ``[`` forever,
#: turning each extra nesting level into one more Python call frame and
#: eventually raising ``RecursionError`` -- a crash, not a clean rejection.
#: Depth is threaded through every recursive parse function and checked
#: before it grows any further, so a schema (or an ``"any"`` fallback) that
#: nests this deep is answered with an ordinary ``"invalid"`` instead.
_MAX_DEPTH = 24


def _parse_object(schema: Schema, buf: bytes, i: int, depth: int = 0):
    if depth > _MAX_DEPTH:
        return "invalid", i, None
    n = len(buf)
    i = _skip_ws(buf, i)
    if i >= n:
        return "incomplete", i, None
    if buf[i] != 0x7B:  # '{'
        return "invalid", i, None
    i = _skip_ws(buf, i + 1)
    props: dict = schema.get("properties", {})
    required = set(schema.get("required", ()))
    open_ = bool(schema.get("_open"))
    obj: dict = {}

    if i < n and buf[i] == 0x7D:  # '}' -- empty object
        return ("complete", i + 1, obj) if required <= set(obj) else ("invalid", i, None)
    if i >= n:
        return "incomplete", i, None

    while True:
        # Keys are constrained to the known property names *not already
        # seen*, exactly like an ``enum``-typed string value -- not merely
        # checked once the closing quote arrives. Two things this fixes at
        # once, both caught by this module's own untrained-model regression
        # test before the fix: without any pruning, a key string has no
        # bound at all (it is not declared as an enum, it *is* the schema's
        # property names) and a greedy decoder can spell an endless key that
        # never matches anything and never closes; and without *excluding
        # already-seen keys*, JSON's tolerance of syntactically-duplicate
        # keys lets the same one property be re-typed forever, which is a
        # separate infinite loop with an otherwise-valid-looking key every
        # time.
        key_order = schema.get("_key_order")
        if key_order is not None:
            # A strict-order object (only ``ToolCallValidator`` sets this):
            # the *only* legal next key is the one at this position in
            # ``key_order`` -- see that class's docstring on why "name" must
            # come first for its ``"arguments"`` schema-switch to work at
            # all. Enforcing it here, as a one-candidate "enum", is what
            # turns "the encoder always does this" into "the decoder cannot
            # produce anything else", closing the gap an earlier version
            # left open (arguments-first was merely *under*-validated, not
            # rejected, which let an untrained model wander into an
            # effectively unbounded ``"any"``-typed arguments placeholder).
            key_enum = [key_order[len(obj)]] if len(obj) < len(key_order) else []
        else:
            key_enum = [p for p in props if p not in obj] if not open_ else None
        kstatus, kpos, kval = _parse_string(buf, i, key_enum)
        if kstatus != "complete":
            return kstatus, i, None
        if kval is None or (kval not in props and not open_):
            return "invalid", i, None
        j = _skip_ws(buf, kpos)
        if j >= n:
            return "incomplete", i, None
        if buf[j] != 0x3A:  # ':'
            return "invalid", i, None
        j = _skip_ws(buf, j + 1)
        if j >= n:
            return "incomplete", i, None
        value_schema = _resolve_property_schema(props.get(kval), obj)
        vstatus, vpos, vval = _parse_value(value_schema, buf, j, depth + 1)
        if vstatus != "complete":
            return vstatus, i, None
        obj[kval] = vval
        k = _skip_ws(buf, vpos)
        if k >= n:
            return "incomplete", i, None
        if buf[k] == 0x7D:
            return ("complete", k + 1, obj) if required <= set(obj) else ("invalid", i, None)
        if buf[k] != 0x2C:  # ','
            return "invalid", i, None
        if not open_ and set(props) <= set(obj):
            # A comma promises another key follows, and there is no closed-
            # world key left to supply one (every declared property is
            # already in ``obj`` and unknown keys are always rejected) -- so
            # the comma itself is the invalid byte, not (only, eventually)
            # whatever unmatchable key string would have to follow it. Catch
            # it exactly here rather than one key-parse later: the earlier,
            # weaker form of this ("prune impossible keys") still let a
            # decoder open a string it could never legally close, and only
            # discovered that a byte later with nothing left to fall back on
            # -- see this module's own untrained-model regression test.
            return "invalid", i, None
        i = _skip_ws(buf, k + 1)
        if i >= n:
            return "incomplete", i, None
        # loop again straight into _parse_string: a bare '}' is never
        # accepted here, which is what rejects a trailing comma.


def _parse_array(schema: Schema, buf: bytes, i: int, depth: int = 0):
    if depth > _MAX_DEPTH:
        return "invalid", i, None
    n = len(buf)
    i = _skip_ws(buf, i)
    if i >= n:
        return "incomplete", i, None
    if buf[i] != 0x5B:  # '['
        return "invalid", i, None
    i = _skip_ws(buf, i + 1)
    items_schema = schema.get("items", _ANY_SCHEMA)
    arr: list = []

    def finish(pos: int):
        lo, hi = schema.get("minItems"), schema.get("maxItems")
        if lo is not None and len(arr) < lo:
            return "invalid", pos, None
        if hi is not None and len(arr) > hi:
            return "invalid", pos, None
        return "complete", pos, arr

    if i < n and buf[i] == 0x5D:  # ']'
        return finish(i + 1)
    if i >= n:
        return "incomplete", i, None

    while True:
        vstatus, vpos, vval = _parse_value(items_schema, buf, i, depth + 1)
        if vstatus != "complete":
            return vstatus, i, None
        arr.append(vval)
        k = _skip_ws(buf, vpos)
        if k >= n:
            return "incomplete", i, None
        if buf[k] == 0x5D:
            return finish(k + 1)
        if buf[k] != 0x2C:
            return "invalid", i, None
        i = _skip_ws(buf, k + 1)
        if i >= n:
            return "incomplete", i, None


def _parse_any(buf: bytes, i: int, depth: int = 0):
    if depth > _MAX_DEPTH:
        return "invalid", i, None
    n = len(buf)
    i = _skip_ws(buf, i)
    if i >= n:
        return "incomplete", i, None
    c = buf[i]
    if c == 0x22:
        return _parse_string(buf, i)
    if c == 0x7B:
        return _parse_object({"type": "object", "properties": {}, "required": (), "_open": True}, buf, i, depth)
    if c == 0x5B:
        return _parse_array({"type": "array", "items": _ANY_SCHEMA}, buf, i, depth)
    if c in (0x74, 0x66):
        return _parse_boolean(buf, i)
    if c == 0x6E:
        return _parse_null(buf, i)
    if c == 0x2D or 0x30 <= c <= 0x39:
        return _parse_number(buf, i)
    return "invalid", i, None


def _parse_value(schema: Schema, buf: bytes, i: int, depth: int = 0):
    t = schema["type"]
    if t == "object":
        return _parse_object(schema, buf, i, depth)
    if t == "array":
        return _parse_array(schema, buf, i, depth)
    if t == "string":
        return _parse_string(buf, i, schema.get("enum"), schema.get("maxLength"))
    if t in ("number", "integer"):
        return _parse_number(buf, i, integer_only=(t == "integer"))
    if t == "boolean":
        return _parse_boolean(buf, i)
    if t == "null":
        return _parse_null(buf, i)
    if t == "any":
        return _parse_any(buf, i, depth)
    raise AssertionError(f"unsupported type {t!r} reached the parser; "
                         "_validate_supported should have rejected it earlier")


# --------------------------------------------------------------------------
# public validators
# --------------------------------------------------------------------------


class SchemaPrefixValidator:
    """Prefix- and instance-validity against one supported-subset schema.

    Construction validates the schema itself (see the module docstring);
    everything after that is pure functions of ``bytes`` in, no state kept
    between calls — a caller decoding token by token calls
    :meth:`is_valid_prefix` on the *whole* accumulated buffer each time
    (reparsing from the start), which is $O(n)$ per byte and $O(n^2)$ over an
    $n$-byte call. Stated plainly: this is the simple, obviously-correct
    choice, not the fast one — a production decoder emitting long JSON values
    under this constraint would want to carry parser state across steps
    instead of reparsing; for the short tool-call payloads this module
    targets (a name and a handful of arguments, generally well under a
    kilobyte) the quadratic cost is not worth the extra state-machine
    bookkeeping it would take to avoid.
    """

    def __init__(self, schema: Schema) -> None:
        _validate_supported(schema)
        self.schema = schema

    def is_valid_prefix(self, data: bytes | str) -> bool:
        buf = data if isinstance(data, (bytes, bytearray)) else data.encode("utf-8")
        status, pos, _ = _parse_value(self.schema, buf, 0)
        if status == "invalid":
            return False
        if status == "incomplete":
            return True
        return bytes(buf[pos:]).strip() == b""

    def is_valid_instance(self, data: bytes | str) -> bool:
        buf = data if isinstance(data, (bytes, bytearray)) else data.encode("utf-8")
        status, pos, _ = _parse_value(self.schema, buf, 0)
        return status == "complete" and bytes(buf[pos:]).strip() == b""

    def parse(self, data: bytes | str) -> Any:
        buf = data if isinstance(data, (bytes, bytearray)) else data.encode("utf-8")
        status, pos, value = _parse_value(self.schema, buf, 0)
        if status != "complete" or bytes(buf[pos:]).strip() != b"":
            raise ValueError("not a complete, schema-valid instance")
        return value


class ToolCallValidator(SchemaPrefixValidator):
    """Constrains a whole ``{"name": ..., "arguments": {...}}`` call against
    a :class:`~iridium.runtime.tools.ToolRegistry`, switching to the *named*
    tool's own parameter schema for ``"arguments"`` as soon as ``"name"``'s
    closing quote is seen.

    That switch is a genuine departure from the generic engine above: which
    schema governs ``"arguments"`` depends on a sibling field's *value*, which
    is exactly the kind of cross-field conditional the module docstring lists
    as unsupported for a general schema (it is the shape of ``if``/``then``,
    just spelled with a JSON key instead of a schema keyword). It is
    supported here, specifically and only for this fixed two-key wrapper
    shape, via ``_resolve_property_schema``'s support for a callable property
    entry — not exposed as a general schema feature.

    **This requires (and, unlike the generic engine, actually enforces)
    ``"name"`` before ``"arguments"``.** JSON key order is not normally
    meaningful, but :func:`iridium.runtime.tools.format_tool_call` always
    emits ``name`` first, and a decoder built against its own encoder's
    convention does not need to solve the harder, order-independent version
    of this problem. This schema sets the engine's ``"_key_order"`` extension
    (see ``_parse_object``) to make ``"arguments"`` a syntactically invalid
    first key, not merely one that parses against an unconstrained
    placeholder until ``"name"`` arrives — an earlier version of this class
    did the latter, and paired with an untrained model's arbitrary logits it
    is exactly how :func:`generate_constrained_json` found its way into
    an unbounded, un-schema'd ``"any"``-typed structure with no natural
    stopping point (this module's own regression test caught it). Rejecting
    the wrong order outright, rather than tolerating and under-validating it,
    is what keeps every reachable state of a call decode bounded by some
    tool's actual argument schema.
    """

    def __init__(self, registry) -> None:
        names = sorted(registry.names())
        if not names:
            raise ValueError("registry has no tools to constrain a call against")
        for name in names:
            _validate_supported(registry.get(name).parameters, f"tool[{name}].parameters")

        def arguments_schema(obj_so_far: dict) -> Schema:
            name = obj_so_far.get("name")
            tool = registry.get(name) if isinstance(name, str) else None
            return dict(tool.parameters) if tool is not None else None

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": names},
                "arguments": arguments_schema,
            },
            "required": ["name", "arguments"],
            "_key_order": ["name", "arguments"],
        }
        self.registry = registry
        super().__init__(schema)


# --------------------------------------------------------------------------
# decode loops
# --------------------------------------------------------------------------


@dataclass
class ConstrainedResult:
    text: str
    ids: list[int]


@torch.no_grad()
def generate_constrained_json(
    model,
    sample,
    schema: Schema | SchemaPrefixValidator,
    max_new_tokens: int = 256,
    seed: int = 0,
    text_offset: int = 16,
    n_loops: Optional[int] = None,
) -> ConstrainedResult:
    """Byte-level constrained decode: every emitted byte keeps the buffer a
    valid JSON prefix of ``schema``; generation stops the instant the buffer
    becomes a complete, schema-valid instance (no reliance on the model
    emitting its own stop token — the schema itself is the stop condition).

    Reuses the same cached-prefill machinery
    ``iridium.runtime.generate.generate`` uses (``atomic_chunks``/
    ``slice_batch`` from ``iridium.runtime.decode``, and that module's own
    private ``_single_token_batch`` for feeding one chosen byte back through
    the cache) rather than a second implementation of incremental decoding —
    see ``iridium.runtime.decode``'s module docstring on why cached and
    uncached decoding computing different functions is a whole category of
    silent bug this project already guards against elsewhere, and this loop
    does not get to be the one exception.

    Greedy in the sense that matters for this guarantee: among the bytes that
    keep the prefix valid, the one with the highest logit is chosen, every
    time, with no temperature/top-p sampling knob exposed. Validity does not
    need help from randomness, and offering a temperature here would invite
    "which knob made my constrained decode invalid" — the answer, with this
    design, is always none of them.

    **What this does not guarantee: termination within ``max_new_tokens``.**
    JSON string values have no length bound of their own, and neither does
    this decoder unless the schema supplies one (``"maxLength"``, or an
    ``enum``, both bound a string's length by construction). A schema whose
    every open-ended field is unconstrained handed to a genuinely untrained
    model — greedy over arbitrary logits — can in principle keep picking
    valid *content* bytes for an unconstrained string field forever without
    ever picking the closing quote, because nothing about "closing quote has
    higher logit than more content" follows from validity alone. This raises
    a clear ``RuntimeError`` rather than truncating a value silently or
    looping past the caller's budget, and it is *why*
    ``tests/unit/test_constrained.py``'s untrained-model regression test uses
    tool schemas built from enums, numbers, booleans and bounded arrays —
    fields whose length is bounded by the schema itself — for the "always
    terminates with a valid call" claim this module actually stands behind.
    A schema with a genuinely open-ended string field should set
    ``"maxLength"`` if that guarantee matters for it too.
    """
    from ..codecs.bank import TensorBatch, continuous_dims
    from ..codecs.spans import MODALITY_INDEX, collate
    from .decode import atomic_chunks, slice_batch
    from .device import device_of
    from .generate import _single_token_batch

    validator = schema if isinstance(schema, SchemaPrefixValidator) else SchemaPrefixValidator(schema)
    if n_loops is None:
        n_loops = min(3, model.cfg.router.max_loops) if model.cfg.controller_mode else 1
    model.eval()
    dims = continuous_dims(model.cfg.codecs)
    device = device_of(model)
    batch = TensorBatch(collate([sample], dims), device=device)
    cache: dict = {}

    hidden = None
    for start, end in atomic_chunks(batch, int(batch.modality.shape[1])):
        out = model(slice_batch(batch, start, end), n_loops=n_loops, cache=cache)
        hidden = out.hidden
    assert hidden is not None
    position = int(batch.positions[0, -1]) + 1

    slot = MODALITY_INDEX["text"]
    buf = bytearray()
    ids: list[int] = []
    for _ in range(max_new_tokens):
        logits = model.codecs.text_head(hidden[:, -1:])[0, -1].float()
        best_byte, best_score = None, float("-inf")
        for b in range(256):
            if not validator.is_valid_prefix(bytes(buf) + bytes([b])):
                continue
            score = float(logits[b + text_offset])
            if score > best_score:
                best_score, best_byte = score, b
        if best_byte is None:
            raise RuntimeError(
                "no next byte keeps the buffer a schema-valid JSON prefix -- "
                "the schema is unsatisfiable, or (for a ToolCallValidator) "
                "the accumulated bytes violated its name-first requirement"
            )
        buf.append(best_byte)
        token_id = best_byte + text_offset
        ids.append(token_id)
        if validator.is_valid_instance(bytes(buf)):
            # errors="replace", not strict: this engine's own string parsing
            # (see _parse_string) already tolerates a content byte sequence
            # that is not well-formed UTF-8 -- see the module docstring's
            # caveat on why -- so the final decode must match that tolerance
            # rather than raising here, which would turn a documented,
            # deliberate leniency into an undocumented crash.
            return ConstrainedResult(text=bytes(buf).decode("utf-8", errors="replace"), ids=ids)
        step = _single_token_batch(batch, slot, token_id, position)
        out = model(step, n_loops=n_loops, cache=cache)
        hidden = out.hidden
        position += 1

    raise RuntimeError(
        f"schema-constrained decode did not close within {max_new_tokens} bytes; "
        "raise max_new_tokens or check the schema is actually satisfiable that short"
    )


def generate_constrained_call(
    model, sample, registry, max_new_tokens: int = 256, seed: int = 0,
    text_offset: int = 16, n_loops: Optional[int] = None,
) -> ConstrainedResult:
    """:func:`generate_constrained_json` against a
    :class:`ToolCallValidator` built from ``registry`` — the call this
    project's tool loop actually wants: "decode a call to one of these
    tools, with syntactically-guaranteed-valid arguments for whichever tool
    the model names.\""""
    validator = ToolCallValidator(registry)
    return generate_constrained_json(model, sample, validator, max_new_tokens=max_new_tokens,
                                      seed=seed, text_offset=text_offset, n_loops=n_loops)


def bpe_candidate_mask(
    tokenizer, buf: bytes, candidate_ids: Sequence[int], validator: SchemaPrefixValidator,
) -> list[int]:
    """Filter a caller-ranked top-``k`` of BPE candidate ids down to those
    that keep ``buf`` a schema-valid prefix once decoded and appended.

    This is the piece a BPE-integrated decode loop would call in place of
    :func:`generate_constrained_json`'s per-byte scan of all 256 raw bytes:
    with a vocabulary of thousands to tens of thousands of ids, scanning
    every one of them the way the byte path scans 256 is a cost this module
    does not accept paying every generated step, so only the caller's
    already-ranked top-``k`` logits are checked here.

    **The honest limitation, stated plainly rather than left implicit:** a
    schema-valid continuation whose token did not make the top-``k`` cut is
    invisible to this function and will never be offered. A caller using
    this in a real decode loop must keep a byte-level fallback for the step
    where the filtered list comes back empty — this tokenizer's own
    guarantee that raw-byte ids ``0..255`` always exist in its vocabulary
    (``iridium.data.tokenizer.BytePairTokenizer``'s module docstring) is
    exactly what makes such a fallback always available, by falling back to
    :func:`generate_constrained_json`'s per-byte check for that one step.
    The guarantee this whole module makes — every emitted byte keeps the
    prefix schema-valid — still holds exactly under that fallback; only the
    *speed* of getting there degrades, never the correctness.

    **Why this is tested against a tokenizer directly rather than through a
    live decode loop:** this project's text codec
    (``iridium.codecs.spans.text_span``) and text head
    (``iridium.runtime.generate``) are byte-level throughout — nothing in
    this codebase today produces BPE token *logits* for a next-token
    distribution to rank in the first place. Wiring a BPE-native text head is
    out of this module's scope; what is in scope, and what is implemented and
    tested here, is the piece of that future decoder that is specific to
    schema-constrained decoding.
    """
    kept: list[int] = []
    for tid in candidate_ids:
        piece = tokenizer.id_to_bytes.get(int(tid))
        if piece is None:
            continue
        if validator.is_valid_prefix(bytes(buf) + piece):
            kept.append(int(tid))
    return kept
