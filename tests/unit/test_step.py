"""STEP (ISO 10303-21) parsing, the bounded geometry subset, and round-trip
token grammar. Gate: an unsupported entity must never disappear, and a
tokenize/detokenize/write cycle must reproduce an equivalent graph.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iridium.codecs.step import (
    DERIVED,
    NULL,
    Enum,
    Ref,
    SUPPORTED_ENTITIES,
    StepSyntaxError,
    build_model,
    decode_string,
    detokenize_step,
    encode_string,
    find_roots,
    graph_signature,
    parse_data,
    tokenize_step,
    write_step_file,
)
from iridium.codecs.step import _lex  # internal, used to test the tokenizer directly

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
PLANE_FACE = (FIXTURES / "plane_face.step").read_text()
ESCAPES = (FIXTURES / "escapes_forward_unsupported.step").read_text()


# -- string escapes -----------------------------------------------------------


def test_decode_doubled_quote():
    assert decode_string("'it''s'") == "it's"


def test_decode_x1_single_byte_escape():
    # \X\E9 is Latin-1 0xE9 = 'é'
    assert decode_string("'caf\\X\\E9'") == "café"


def test_decode_x2_run_of_code_points():
    # \X2\0042 0043\X0\ = U+0042 U+0043 = "BC"
    assert decode_string("'A\\X2\\00420043\\X0\\D'") == "ABCD"


@pytest.mark.parametrize("s", ["plain", "it's a test", "café", "日本語", "a\tb"])
def test_encode_decode_round_trip(s):
    assert decode_string(encode_string(s)) == s


# -- tokenizer / value parser --------------------------------------------------


def test_lex_rejects_garbage():
    with pytest.raises(StepSyntaxError):
        _lex("#1=FOO(@@@);")


def test_parse_data_resolves_refs_lists_enums_null_derived():
    text = "#1=FOO('',#2,(1,2,3),.T.,$,*);"
    instances = parse_data(_lex(text))
    assert len(instances) == 1
    inst = instances[0]
    assert inst.id == 1 and inst.type == "FOO"
    assert inst.args[1] == Ref(2)
    assert inst.args[2] == [1, 2, 3]
    assert inst.args[3] == Enum(".T.")
    assert inst.args[4] is NULL
    assert inst.args[5] is DERIVED


def test_parse_data_forward_reference_does_not_require_two_passes():
    text = "#1=FOO(#2); #2=BAR(1);"
    instances = {i.id: i for i in parse_data(_lex(text))}
    assert instances[1].args[0] == Ref(2)
    assert instances[2].type == "BAR"


# -- full-file parsing ----------------------------------------------------


def test_build_model_plane_face_finds_full_supported_chain():
    model = build_model(PLANE_FACE)
    # Every entity in this fixture except VERTEX_POINT is in the subset.
    assert model.unsupported_report() == {"VERTEX_POINT": 2}
    assert len(model.supported) == 16
    brep = model.supported[18]
    assert brep.type == "MANIFOLD_SOLID_BREP"
    assert brep.fields["name"] == "block"
    assert brep.fields["outer"] == Ref(17)


def test_build_model_rejects_non_step_text():
    with pytest.raises(StepSyntaxError):
        build_model("not a step file at all")


def test_find_roots_prefers_manifold_solid_brep():
    model = build_model(PLANE_FACE)
    assert find_roots(model) == [18]


def test_find_roots_falls_back_to_unreferenced_when_no_brep():
    text = (
        "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\n"
        "FILE_NAME('','',(''),(''),'','','');\nFILE_SCHEMA(('X'));\nENDSEC;\n"
        "DATA;\n#1=CARTESIAN_POINT('',(0.,0.,0.));\n#2=DIRECTION('',(0.,0.,1.));\n"
        "ENDSEC;\nEND-ISO-10303-21;\n"
    )
    model = build_model(text)
    assert find_roots(model) == [1, 2]  # neither references the other


def test_unsupported_entities_preserved_structurally_not_dropped():
    model = build_model(ESCAPES)
    report = model.unsupported_report()
    assert report["VERTEX_POINT"] == 2
    assert report["COMPLEX"] == 1
    assert report["UNSUPPORTED_UNIT_THING"] == 1
    # Preserved means present in the graph and still resolvable: the
    # supported EDGE_CURVE at #120 references VERTEX_POINT #130/#140, which
    # must still exist in by_id even though they're not "interpreted".
    edge = model.supported[120]
    assert edge.fields["edge_start"] == Ref(130)
    assert model.by_id[130].type == "VERTEX_POINT"


def test_string_with_doubled_quote_and_escapes_parses_in_context():
    model = build_model(ESCAPES)
    point = model.supported[90]
    assert point.fields["name"] == "it's"
    assert model.header.name[0] == "escé.stp"
    assert model.header.name[2] == ["ABCuth"]


def test_derived_attributes_on_oriented_edge_preserved():
    model = build_model(ESCAPES)
    oriented_edge = model.supported[80]
    assert oriented_edge.fields["edge_start"] is DERIVED
    assert oriented_edge.fields["edge_end"] is DERIVED


def test_extra_trailing_arguments_marked_unsupported_not_truncated():
    """A newer schema revision might add a trailing optional attribute this
    module does not model; that instance must be preserved wholesale
    (unsupported), never silently truncated to the known arg count."""
    text = (
        "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\n"
        "FILE_NAME('','',(''),(''),'','','');\nFILE_SCHEMA(('X'));\nENDSEC;\n"
        "DATA;\n#1=PLANE('',#2,'unexpected_extra_arg');\n"
        "#2=AXIS2_PLACEMENT_3D('',#3,$,$);\n#3=CARTESIAN_POINT('',(0.,0.,0.));\n"
        "ENDSEC;\nEND-ISO-10303-21;\n"
    )
    model = build_model(text)
    assert 1 in model.unsupported
    assert model.by_id[1].args[-1] == "unexpected_extra_arg"


# -- token grammar round trip -----------------------------------------------


@pytest.mark.parametrize("text", [PLANE_FACE, ESCAPES])
def test_tokenize_detokenize_preserves_graph_signature(text):
    model = build_model(text)
    tokens = tokenize_step(model)
    restored = detokenize_step(tokens, header=model.header)
    assert graph_signature(model) == graph_signature(restored)
    assert model.unsupported_report() == restored.unsupported_report()


@pytest.mark.parametrize("text", [PLANE_FACE, ESCAPES])
def test_full_round_trip_reparses_to_equivalent_graph(text):
    """parse -> tokenize -> detokenize -> write -> reparse must land on a
    structurally equivalent graph: the single strongest claim this module
    makes about the token grammar being lossless for the supported subset
    (and non-lossy — merely un-interpreted — for everything else)."""
    model = build_model(text)
    tokens = tokenize_step(model)
    restored = detokenize_step(tokens, header=model.header)
    written = write_step_file(restored)
    reparsed = build_model(written)
    assert graph_signature(model) == graph_signature(reparsed)
    assert model.unsupported_report() == reparsed.unsupported_report()
    assert {e.type for e in model.supported.values()} == {
        e.type for e in reparsed.supported.values()
    }


def test_written_file_is_syntactically_valid_step():
    model = build_model(PLANE_FACE)
    text = write_step_file(model)
    assert text.startswith("ISO-10303-21;\n")
    assert text.rstrip().endswith("END-ISO-10303-21;")
    assert "HEADER;" in text and "DATA;" in text
    # And, of course, it must parse.
    build_model(text)


def test_tokenize_vocabulary_uses_local_contiguous_ids():
    """References in the token stream are local indices in [0, n), not raw
    file ids — this is what makes the grammar learnable: a model should
    never need file-numbering-scale embeddings for a mesh with 40 entities
    numbered up into the thousands by some exporter."""
    model = build_model(ESCAPES)
    tokens = tokenize_step(model)
    n = tokens[0]
    ref_values = [tokens[i + 1] for i, t in enumerate(tokens) if t == "REF"]
    assert all(0 <= v < n for v in ref_values)


def test_supported_entities_table_matches_docstring_subset():
    expected = {
        "CARTESIAN_POINT", "DIRECTION", "VECTOR", "AXIS2_PLACEMENT_3D", "LINE",
        "CIRCLE", "PLANE", "CYLINDRICAL_SURFACE", "EDGE_CURVE", "ORIENTED_EDGE",
        "EDGE_LOOP", "FACE_BOUND", "FACE_OUTER_BOUND", "ADVANCED_FACE",
        "CLOSED_SHELL", "MANIFOLD_SOLID_BREP",
    }
    assert expected <= set(SUPPORTED_ENTITIES)
