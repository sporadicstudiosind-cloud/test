"""Offline checks for the tool-call format, the registry and the loop."""

from types import SimpleNamespace

import jsonschema
import pytest

from iridium.runtime import chat
from iridium.runtime.tools import (
    Tool, ToolCallError, ToolRegistry, extract_tool_call, format_tool_call,
    parse_tool_call, run_tool_loop,
)


def _controls(sample):
    return [int(span.payload[0]) for span in sample.spans if span.modality == "control"]


# -- Tool / ToolRegistry -----------------------------------------------------


def test_tool_rejects_non_identifier_name():
    with pytest.raises(ValueError):
        Tool("not a name", "d", {"type": "object", "properties": {}})


def test_tool_rejects_non_object_parameters():
    with pytest.raises(ValueError):
        Tool("f", "d", {"type": "string"})


def test_tool_rejects_invalid_schema():
    with pytest.raises(jsonschema.SchemaError):
        Tool("f", "d", {"type": "object", "properties": {"x": {"type": "not-a-type"}}})


def test_registry_call_validates_and_executes():
    reg = ToolRegistry([Tool("add", "add", {
        "type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }, fn=lambda a, b: a + b)])
    assert reg.call("add", {"a": 1, "b": 2}) == 3
    with pytest.raises(jsonschema.ValidationError):
        reg.call("add", {"a": 1})
    with pytest.raises(KeyError):
        reg.call("nope", {})


def test_registry_call_with_no_fn_raises():
    reg = ToolRegistry([Tool("f", "d", {"type": "object", "properties": {}})])
    with pytest.raises(RuntimeError):
        reg.call("f", {})


def test_system_prompt_block_is_order_independent():
    a = Tool("a_tool", "does a", {"type": "object", "properties": {}})
    b = Tool("b_tool", "does b", {"type": "object", "properties": {}})
    assert ToolRegistry([a, b]).system_prompt_block() == ToolRegistry([b, a]).system_prompt_block()


def test_tool_from_mcp():
    spec = {"name": "search", "description": "search the web",
            "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}
    tool = Tool.from_mcp(spec, fn=lambda q: f"results for {q}")
    assert tool.name == "search"
    assert tool.parameters["properties"]["q"]["type"] == "string"
    assert tool.fn("x") == "results for x"


# -- wire format --------------------------------------------------------------


def test_format_parse_round_trip():
    text = format_tool_call("add", {"a": 1, "b": 2})
    assert text.startswith('{"name":"add"')  # name first -- constrained.py depends on this
    name, args = parse_tool_call(text)
    assert name == "add" and args == {"a": 1, "b": 2}


@pytest.mark.parametrize("bad", [
    "not json", "42", '{"name": "a"}', '{"name": 1, "arguments": {}}',
    '{"name": "a", "arguments": [1]}', '{"name": "a", "arguments": {}, "extra": 1}',
])
def test_parse_tool_call_rejects_malformed(bad):
    with pytest.raises(ToolCallError):
        parse_tool_call(bad)


def test_extract_tool_call_distinguishes_call_from_text():
    call_json = format_tool_call("f", {"x": 1})
    ids = [chat.TOOL_CALL] + [b + chat.TEXT_OFFSET for b in call_json.encode("utf-8")]
    assert extract_tool_call(ids) == call_json
    text_ids = [b + chat.TEXT_OFFSET for b in b"hello"]
    assert extract_tool_call(text_ids) is None
    assert extract_tool_call([]) is None


class _FakeTokenizer:
    """Minimal encode/decode stand-in, enough to exercise the tokenizer path."""

    def encode(self, s):
        return list(s.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


def test_extract_tool_call_with_tokenizer():
    call_json = format_tool_call("f", {"x": 1})
    tok = _FakeTokenizer()
    ids = [chat.TOOL_CALL] + [b + chat.TEXT_OFFSET for b in tok.encode(call_json)]
    assert extract_tool_call(ids, tokenizer=tok) == call_json


# -- chat.py's supervision wiring for the two new roles -----------------------


def test_conversation_spans_supervises_tool_call_not_tool_result():
    turns = [
        chat.Turn("system", "sys"),
        chat.Turn("user", "hi"),
        chat.Turn("tool_call", '{"name":"f","arguments":{}}'),
        chat.Turn("tool_result", '{"ok":true}'),
        chat.Turn("assistant", "done"),
    ]
    sample = chat.conversation_sample(turns, supervise_assistant=True)
    by_role = {}
    cursor_role = None
    for span in sample.spans:
        if span.modality == "control" and int(span.payload[0]) in chat.TOKEN_ROLE:
            cursor_role = chat.TOKEN_ROLE[int(span.payload[0])]
        elif span.modality == "text":
            by_role.setdefault(cursor_role, []).append(span.supervised)
    assert by_role["tool_call"] == [True]
    assert by_role["tool_result"] == [False]
    assert by_role["assistant"] == [True]
    assert by_role["user"] == [False]

    unsupervised = chat.conversation_sample(turns, supervise_assistant=False)
    for span in unsupervised.spans:
        if span.modality == "text":
            assert span.supervised is False


def test_roles_10_and_11_and_12_are_distinct_and_stable():
    # id 10 stays reserved for iridium.agency.media_agent's own "tool" role
    # (see chat.py's marker-block docstring); this project's own tool-call
    # boundary tokens must not collide with it.
    assert chat.ROLE_TOKEN["tool"] == 10
    assert chat.TOOL_CALL == 11
    assert chat.TOOL_RESULT == 12
    assert chat.ROLE_TOKEN["tool_call"] == chat.TOOL_CALL
    assert chat.ROLE_TOKEN["tool_result"] == chat.TOOL_RESULT


# -- run_tool_loop -------------------------------------------------------------


def _registry():
    return ToolRegistry([Tool("add", "add two numbers", {
        "type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }, fn=lambda a, b: a + b)])


def test_run_tool_loop_executes_a_call_then_answers():
    reg = _registry()
    call_json = format_tool_call("add", {"a": 1, "b": 2})
    responses = iter([
        SimpleNamespace(ids=[chat.TOOL_CALL] + [b + chat.TEXT_OFFSET for b in call_json.encode()],
                        text=""),
        SimpleNamespace(ids=[b + chat.TEXT_OFFSET for b in b"the answer is 3"],
                        text="the answer is 3"),
    ])

    def fake_generate(_model, _sample, **_kwargs):
        return next(responses)

    model = SimpleNamespace(cfg=SimpleNamespace(max_seq_len=4096))
    messages = [chat.Turn("system", reg.system_prompt_block()), chat.Turn("user", "add 1 and 2")]
    result = run_tool_loop(model, messages, reg, generate_fn=fake_generate)

    assert result.stopped == "answer"
    assert result.answer == "the answer is 3"
    assert len(result.calls) == 1
    assert result.calls[0] == {"name": "add", "arguments": {"a": 1, "b": 2}, "result": 3}
    roles = [t.role for t in result.turns]
    assert roles == ["system", "user", "tool_call", "tool_result", "assistant"]


def test_run_tool_loop_turns_a_bad_call_into_a_tool_result_not_an_exception():
    reg = _registry()
    bad_call = '{"name":"add","arguments":{"a":1}}'  # missing required "b"
    good_call = format_tool_call("add", {"a": 1, "b": 2})
    responses = iter([
        SimpleNamespace(ids=[chat.TOOL_CALL] + [b + chat.TEXT_OFFSET for b in bad_call.encode()], text=""),
        SimpleNamespace(ids=[chat.TOOL_CALL] + [b + chat.TEXT_OFFSET for b in good_call.encode()], text=""),
        SimpleNamespace(ids=[b + chat.TEXT_OFFSET for b in b"3"], text="3"),
    ])

    def fake_generate(_model, _sample, **_kwargs):
        return next(responses)

    model = SimpleNamespace(cfg=SimpleNamespace(max_seq_len=4096))
    messages = [chat.Turn("system", reg.system_prompt_block()), chat.Turn("user", "add 1 and 2")]
    result = run_tool_loop(model, messages, reg, max_calls=3, generate_fn=fake_generate)

    assert result.stopped == "answer"
    assert result.calls[0]["error"]  # the malformed call was recorded as an error, not raised
    assert "error" not in result.calls[1]


def test_run_tool_loop_reports_max_calls_without_claiming_an_answer():
    reg = _registry()
    call_json = format_tool_call("add", {"a": 1, "b": 2})
    ids = [chat.TOOL_CALL] + [b + chat.TEXT_OFFSET for b in call_json.encode()]

    def fake_generate(_model, _sample, **_kwargs):
        return SimpleNamespace(ids=ids, text="")  # always calls, never answers

    model = SimpleNamespace(cfg=SimpleNamespace(max_seq_len=4096))
    messages = [chat.Turn("system", reg.system_prompt_block()), chat.Turn("user", "go")]
    result = run_tool_loop(model, messages, reg, max_calls=2, generate_fn=fake_generate)

    assert result.stopped == "max_calls"
    assert result.answer == ""
    assert len(result.calls) == 2


def test_run_tool_loop_rejects_bad_max_calls():
    with pytest.raises(ValueError):
        run_tool_loop(SimpleNamespace(cfg=SimpleNamespace(max_seq_len=64)), [], _registry(), max_calls=0)
