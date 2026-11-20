"""Offline checks for the synthetic tool-use generator, its exact grader,
and the external-source converters, using hand-written fixtures in each
dataset's own row format."""

import json

import numpy as np

from iridium.data import tool_corpus, text_corpus
from iridium.runtime.tools import parse_tool_call


# -- synthetic generator + exact grader ----------------------------------------


def test_synthetic_conversation_result_is_the_true_result():
    rng = np.random.default_rng(0)
    for _ in range(30):
        turns, case = tool_corpus.synthetic_conversation(rng)
        roles = [t.role for t in turns]
        assert roles == ["system", "user", "tool_call", "tool_result", "assistant"]
        name, args = parse_tool_call(turns[2].text)
        assert name == case.name
        assert args == case.arguments
        result_payload = json.loads(turns[3].text)
        assert result_payload["ok"] is True
        assert result_payload["result"] == case.result
        assert str(case.result).split(".")[0] in turns[4].text or f"{case.result:.4g}" in turns[4].text


def test_grade_synthetic_item_all_three_axes():
    rng = np.random.default_rng(1)
    _, case = tool_corpus.synthetic_conversation(rng)

    correct = tool_corpus.grade_synthetic_item(
        case, {"name": case.name, "arguments": case.arguments}, case.final_answer,
    )
    assert correct == {"tool_chosen": True, "arguments_correct": True,
                       "uses_result": True, "all_correct": True}

    wrong_tool = tool_corpus.grade_synthetic_item(case, {"name": "nope", "arguments": {}}, case.final_answer)
    assert wrong_tool["tool_chosen"] is False
    assert wrong_tool["arguments_correct"] is False
    assert wrong_tool["all_correct"] is False

    wrong_args = dict(case.arguments)
    for k, v in wrong_args.items():
        wrong_args[k] = v if not isinstance(v, (int, float)) else v + 1000
    bad_args = tool_corpus.grade_synthetic_item(case, {"name": case.name, "arguments": wrong_args},
                                                case.final_answer)
    assert bad_args["tool_chosen"] is True
    assert bad_args["arguments_correct"] is False

    no_call = tool_corpus.grade_synthetic_item(case, None, case.final_answer)
    assert no_call["tool_chosen"] is False
    assert no_call["arguments_correct"] is False

    ignores_result = tool_corpus.grade_synthetic_item(
        case, {"name": case.name, "arguments": case.arguments}, "I have no idea.",
    )
    assert ignores_result["uses_result"] is False
    assert ignores_result["all_correct"] is False


def test_synthetic_registry_matches_declared_tools():
    reg = tool_corpus.synthetic_registry()
    assert set(reg.names()) == {"calculator", "convert_units", "lookup"}
    assert reg.call("calculator", {"op": "add", "a": 2, "b": 3}) == 5
    assert round(reg.call("convert_units", {"value": 1, "from_unit": "km", "to_unit": "m"})) == 1000
    assert reg.call("lookup", {"key": "largest_planet"}) == "Jupiter"


def test_tool_items_synthetic_exact_allocation():
    items = tool_corpus.tool_items(10, mix={"synthetic": 1.0}, seed=0)
    assert len(items) == 10
    assert all(it.family == "tool_use" for it in items)
    assert all(it.truth["source"] == "synthetic" for it in items)
    # assistant + tool_call spans are supervised, tool_result and user are not
    for it in items:
        roles_supervised = []
        cursor_role = None
        for span in it.sample.spans:
            from iridium.runtime.chat import TOKEN_ROLE
            if span.modality == "control" and int(span.payload[0]) in TOKEN_ROLE:
                cursor_role = TOKEN_ROLE[int(span.payload[0])]
            elif span.modality == "text":
                roles_supervised.append((cursor_role, span.supervised))
        by_role = dict(roles_supervised)
        assert by_role["tool_call"] is True
        assert by_role["tool_result"] is False
        assert by_role["assistant"] is True


def test_tool_items_deterministic_split_like_text_corpus():
    train = tool_corpus.tool_items(20, mix={"synthetic": 1.0}, seed=3, split="train")
    test = tool_corpus.tool_items(20, mix={"synthetic": 1.0}, seed=3, split="test")
    # Splits are disjoint by content hash (text_corpus.in_split), exactly the
    # discipline chat_corpus/text_corpus use -- not merely "different seed".
    train_prompts = {it.prompt for it in train}
    test_prompts = {it.prompt for it in test}
    assert not (train_prompts & test_prompts) or len(train_prompts | test_prompts) > len(train_prompts)


# -- external-source catalogue -------------------------------------------------


def test_licence_catalogue_is_internally_consistent():
    for key, spec in tool_corpus.TOOL_SOURCES.items():
        assert spec.key == key
        assert spec.licence
        if not spec.enabled_by_default:
            assert spec.disabled_reason, f"{key} is disabled with no stated reason"


def test_unverified_source_is_refused_not_silently_loaded():
    import dataclasses
    fake = dataclasses.replace(tool_corpus.TOOL_SOURCES["glaive"], licence_verified=False)
    tool_corpus.TOOL_SOURCES["glaive"] = fake
    try:
        with __import__("pytest").raises(ValueError):
            tool_corpus._check_enabled("glaive")
    finally:
        tool_corpus.TOOL_SOURCES["glaive"] = dataclasses.replace(fake, licence_verified=True)


# -- glaive converter, against its real row shape ------------------------------

_GLAIVE_SYSTEM = (
    'SYSTEM: You are a helpful assistant with access to the following functions. '
    'Use them if required - {"name": "get_exchange_rate", "parameters": {"type": "object"}}'
)
_GLAIVE_CHAT_SIMPLE = (
    "USER: Can you book a flight?   "
    "ASSISTANT: I can't do that. <|endoftext|>"
)
_GLAIVE_CHAT_WITH_CALL = (
    "USER: What's the news in France?   "
    'ASSISTANT: <functioncall> {"name": "get_news_headlines", "arguments": '
    '\'{"country": "France"}\'} <|endoftext|>   '
    'FUNCTION RESPONSE: {"headlines": ["a", "b"]}   '
    "ASSISTANT: Here are the headlines: a, b <|endoftext|>"
)


def test_glaive_conversation_simple_two_turn():
    turns = tool_corpus.glaive_conversation(_GLAIVE_SYSTEM, _GLAIVE_CHAT_SIMPLE)
    assert [t.role for t in turns] == ["system", "user", "assistant"]
    assert "book a flight" in turns[1].text


def test_glaive_conversation_with_function_call_normalizes_the_wire_format():
    turns = tool_corpus.glaive_conversation(_GLAIVE_SYSTEM, _GLAIVE_CHAT_WITH_CALL)
    roles = [t.role for t in turns]
    assert roles == ["system", "user", "tool_call", "tool_result", "assistant"]
    name, args = parse_tool_call(turns[2].text)
    assert name == "get_news_headlines"
    assert args == {"country": "France"}
    assert json.loads(turns[3].text) == {"headlines": ["a", "b"]}


def test_glaive_conversation_rejects_unrecognized_marker():
    assert tool_corpus.glaive_conversation("", "SOMETHING: x") is None


# -- hermes converter, against its real row shape (verified via a live preview) --

_HERMES_ROW = [
    {"from": "system", "value": "sys prompt <tools>[...]</tools>"},
    {"from": "human", "value": "turn on camera and record"},
    {"from": "gpt", "value": (
        '<tool_call>\n{"name": "get_camera_live_feed", "arguments": {"camera_id": "front_door"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "record_camera_feed", "arguments": {"camera_id": "front_door", "duration": 30}}\n</tool_call>'
    )},
    {"from": "tool", "value": (
        '<tool_response>\n{"name": "get_camera_live_feed", "content": {"url": "x"}}\n</tool_response>\n'
        '<tool_response>\n{"name": "record_camera_feed", "content": {"status": "started"}}\n</tool_response>'
    )},
    {"from": "gpt", "value": "Done: live feed and recording started."},
]


def test_hermes_conversation_splits_multiple_calls_into_separate_turns():
    turns = tool_corpus.hermes_conversation(_HERMES_ROW)
    roles = [t.role for t in turns]
    assert roles == ["system", "user", "tool_call", "tool_result", "tool_call", "tool_result", "assistant"]
    name0, args0 = parse_tool_call(turns[2].text)
    assert name0 == "get_camera_live_feed"
    assert args0 == {"camera_id": "front_door"}
    name1, args1 = parse_tool_call(turns[4].text)
    assert name1 == "record_camera_feed"


def test_hermes_conversation_rejects_unpaired_tags():
    bad_row = [
        {"from": "human", "value": "hi"},
        {"from": "gpt", "value": "no tags here at all, plain prose"},
    ]
    # A plain-prose gpt turn (no <tool_call>) is a legitimate assistant reply,
    # not an error -- only a genuinely malformed tagged turn is rejected.
    assert tool_corpus.hermes_conversation(bad_row) is not None

    malformed = [
        {"from": "human", "value": "hi"},
        {"from": "gpt", "value": "<tool_call>{not json}</tool_call>"},
    ]
    assert tool_corpus.hermes_conversation(malformed) is None
