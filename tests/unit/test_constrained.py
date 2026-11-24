"""Offline checks for the schema-prefix validator and constrained decoding.

The one claim this suite exists to hold this module to: an untrained tiny
model, decoding under the constraint, always produces a parseable,
schema-valid tool call. Everything else here is the validator's own
correctness — every prefix of a valid instance accepted, every genuinely
invalid extension rejected — which is what that claim rests on.
"""

import json

import pytest
import torch

from iridium.config import get_config
from iridium.model.iridium1 import Iridium1
from iridium.runtime.chat import Turn, conversation_sample
from iridium.runtime.constrained import (
    SchemaPrefixValidator, ToolCallValidator, generate_constrained_call,
)
from iridium.runtime.tools import Tool, ToolRegistry


# -- SchemaPrefixValidator: accepts every prefix of every valid instance -----


def _assert_all_prefixes_valid(validator, instance_json: str):
    for i in range(len(instance_json) + 1):
        assert validator.is_valid_prefix(instance_json[:i]), (i, instance_json[:i])
    assert validator.is_valid_instance(instance_json)


def test_scalar_types():
    for schema, value in [
        ({"type": "boolean"}, True),
        ({"type": "boolean"}, False),
        ({"type": "null"}, None),
        ({"type": "integer"}, 42),
        ({"type": "number"}, -3.5),
    ]:
        _assert_all_prefixes_valid(SchemaPrefixValidator(schema), json.dumps(value))


def test_enum_string_prunes_impossible_prefixes():
    v = SchemaPrefixValidator({"type": "string", "enum": ["add", "sub"]})
    _assert_all_prefixes_valid(v, '"add"')
    _assert_all_prefixes_valid(v, '"sub"')
    assert not v.is_valid_prefix('"m')      # no member starts with 'm'
    assert not v.is_valid_prefix('"addd"')  # too long, not a real member
    assert not v.is_valid_instance('"a"')   # a real prefix, but not a full member


def test_nested_object_and_array():
    schema = {
        "type": "object",
        "properties": {
            "x": {"type": "object", "properties": {"y": {"type": "boolean"}}, "required": ["y"]},
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "required": ["x"],
    }
    v = SchemaPrefixValidator(schema)
    _assert_all_prefixes_valid(v, json.dumps({"x": {"y": True}, "items": [1, 2, 3.5]}))
    _assert_all_prefixes_valid(v, json.dumps({"x": {"y": False}}))


def test_required_and_unknown_keys():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    v = SchemaPrefixValidator(schema)
    assert not v.is_valid_instance("{}")             # missing required
    assert not v.is_valid_prefix('{"b":1}')           # unknown key -- always closed-world
    assert not v.is_valid_prefix('{"a":1,"a":2}')     # duplicate key rejected mid-stream


def test_trailing_comma_and_garbage_rejected():
    v = SchemaPrefixValidator({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]})
    assert not v.is_valid_prefix('{"a":1,}')
    assert not v.is_valid_instance('{"a":1}  extra')


def test_maxlength_and_maxitems_bound_growth():
    v = SchemaPrefixValidator({"type": "string", "maxLength": 3})
    assert v.is_valid_prefix('"ab')
    assert v.is_valid_instance('"abc"')
    assert not v.is_valid_prefix('"abcd')

    av = SchemaPrefixValidator({"type": "array", "items": {"type": "integer"}, "maxItems": 2})
    assert av.is_valid_instance("[1,2]")
    assert not av.is_valid_prefix("[1,2,")


@pytest.mark.parametrize("bad_schema", [
    {"oneOf": [{"type": "string"}]},
    {"type": ["string", "null"]},
    {"type": "string", "pattern": "^a"},
    {"type": "object", "properties": {}, "additionalProperties": True},
    {"type": "array", "items": [{"type": "string"}, {"type": "integer"}]},
    {"not_a_type_key": 1},
])
def test_unsupported_schema_fails_closed_at_construction(bad_schema):
    with pytest.raises(ValueError):
        SchemaPrefixValidator(bad_schema)


# -- ToolCallValidator ---------------------------------------------------------


def _registry():
    return ToolRegistry([
        Tool("add", "add two numbers", {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        }, fn=lambda a, b: a + b),
        Tool("convert_units", "convert between units", {
            "type": "object",
            "properties": {"value": {"type": "number"},
                           "unit": {"type": "string", "enum": ["m", "ft", "km"]}},
            "required": ["value", "unit"],
        }),
        Tool("lookup", "look up a key", {
            "type": "object",
            "properties": {"key": {"type": "string", "maxLength": 8}},
            "required": ["key"],
        }),
    ])


def test_tool_call_validator_accepts_every_registered_tool():
    reg = _registry()
    v = ToolCallValidator(reg)
    for name, args in [("add", {"a": 1, "b": 2}), ("convert_units", {"value": 3.5, "unit": "ft"}),
                       ("lookup", {"key": "x"})]:
        _assert_all_prefixes_valid(v, json.dumps({"name": name, "arguments": args}, separators=(",", ":")))


def test_tool_call_validator_rejects_unknown_tool_name():
    v = ToolCallValidator(_registry())
    assert not v.is_valid_prefix('{"name":"z')


def test_tool_call_validator_requires_name_first():
    v = ToolCallValidator(_registry())
    # "arguments" as the first key is rejected outright -- see the class's
    # own docstring on why this is enforced rather than merely conventional.
    assert not v.is_valid_prefix('{"arguments"')


def test_tool_call_validator_needs_a_nonempty_registry():
    with pytest.raises(ValueError):
        ToolCallValidator(ToolRegistry([]))


# -- the untrained-model regression: the key claim ----------------------------


def test_untrained_tiny_model_always_yields_a_valid_tool_call():
    reg = ToolRegistry([
        Tool("add", "add two numbers", {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        }, fn=lambda a, b: a + b),
        Tool("convert_units", "convert between units", {
            "type": "object",
            "properties": {"value": {"type": "number"},
                           "unit": {"type": "string", "enum": ["m", "ft", "km"]}},
            "required": ["value", "unit"],
        }),
        Tool("lookup", "look up a key", {
            "type": "object",
            "properties": {"key": {"type": "string", "maxLength": 8}},
            "required": ["key"],
        }),
        Tool("flag", "set a boolean flag with optional tags", {
            "type": "object",
            "properties": {
                "on": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string", "enum": ["x", "y", "z"]},
                        "maxItems": 3},
            },
            "required": ["on"],
        }),
    ])
    turns = [Turn("system", reg.system_prompt_block()), Turn("user", "please add 2 and 3")]
    sample = conversation_sample(turns, supervise_assistant=False, open_for_reply=True)
    cfg = get_config("tiny")

    for model_seed in range(8):
        torch.manual_seed(model_seed)
        model = Iridium1(cfg)
        model.eval()
        result = generate_constrained_call(model, sample, reg, max_new_tokens=600)
        obj = json.loads(result.text)  # must parse
        assert set(obj) == {"name", "arguments"}
        assert obj["name"] in reg.names()
        reg.validate_arguments(obj["name"], obj["arguments"])  # must be schema-valid


def test_constrained_decode_raises_rather_than_loops_forever_when_impossible():
    reg = ToolRegistry([Tool("f", "d", {
        "type": "object",
        "properties": {"x": {"type": "string", "enum": ["only_this_one_value"]}},
        "required": ["x"],
    })])
    turns = [Turn("system", reg.system_prompt_block()), Turn("user", "call f")]
    sample = conversation_sample(turns, supervise_assistant=False, open_for_reply=True)
    cfg = get_config("tiny")
    torch.manual_seed(0)
    model = Iridium1(cfg)
    model.eval()
    # A tiny max_new_tokens budget cannot fit even the shortest valid call
    # ({"name":"f","arguments":{"x":"only_this_one_value"}}); this must raise
    # a clean RuntimeError, never hang or silently return a truncated call.
    with pytest.raises(RuntimeError):
        generate_constrained_call(model, sample, reg, max_new_tokens=5)
