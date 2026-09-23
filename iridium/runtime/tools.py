"""Tool use: one call format, one loop, used by training data and inference alike.

The same principle ``chat.py`` states for turns applies one level down, inside
a turn: a tool call written into training data and a tool call parsed out of
generation have to be *the same bytes for the same reasons*, or a model that
scores perfectly on the training format emits something the runtime cannot
parse. So this module is the only place a tool call is serialized or parsed,
and both :mod:`iridium.data.tool_corpus` (training data) and this module's own
:func:`run_tool_loop` (inference) import it rather than restating the shape.

**The call itself is JSON, but its boundaries are control tokens, not text.**
``iridium.runtime.chat`` reserves :data:`~iridium.runtime.chat.TOOL_CALL` and
:data:`~iridium.runtime.chat.TOOL_RESULT` for exactly this — see that module's
docstring for why a reserved id and not a text tag like ``<tool_call>``
(the tag two of the three licensed corpora this project draws on actually use,
and what ``iridium.agency.media_agent``'s own ad hoc loop also falls back on):
a tag can be split across an arbitrary number of byte-level tokens depending
on what precedes it, a user's own message can contain the literal string, and
every occurrence spends real sequence budget a reserved id does not. What sits
*between* the markers is still JSON — reinventing a slot-typed wire format for
"a name and a dict of arguments" would be solving a problem JSON already
solves, and every tool author already thinks in JSON Schema for the
parameters, which is exactly what :mod:`iridium.runtime.constrained` decodes
against.

**Supervision is asymmetric, and it is the single fact this module has to get
right.** A tool call is the model's own output — trained on, exactly like a
prose reply. A tool result is the *environment's* output — read as context,
never supervised. Training on tool results teaches the model to predict what
a tool will say instead of calling it, which is indistinguishable during
training (the loss goes down either way) and is a model that hallucinates API
responses in production. ``iridium.runtime.chat.conversation_spans`` enforces
this at the span level for the ``"tool_call"``/``"tool_result"`` turn roles it
now knows about; this module never constructs an unsupervised call or a
supervised result by hand.

**Execution safety.** ``ToolRegistry`` is a mapping from name to a *caller-
supplied* Python callable. Nothing here ever ``exec``s or ``eval``s
model-produced code — the model chooses a name and a dict of arguments, both
inert data, and the registry is what turns a name into a function to call.
That makes the registry itself the trust boundary: a program that registers
``os.system`` as a tool has handed the model a shell, and no amount of JSON
Schema validation on the arguments changes that. Argument *values* are still
untrusted (they came from a model, however constrained the JSON shape is) —
sanitizing a shell-injection payload string, or bounding an infinite-loop
``n``, is the tool function's own job, not this module's.
``iridium.runtime.sandbox`` exists for a tool author who wants to run
untrusted model-*generated code* (a different thing than a tool call) with
resource limits; nothing here changes that boundary or duplicates it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import jsonschema

from .chat import (
    ASSISTANT, EOS, STOP_IDS, TEXT_OFFSET, TOOL_CALL, TOOL_RESULT,
    Turn, conversation_sample, fit_to_budget,
)

__all__ = [
    "Tool", "ToolRegistry", "ToolCallError",
    "format_tool_call", "parse_tool_call", "extract_tool_call",
    "system_prompt_block", "ToolLoopResult", "run_tool_loop",
]


class ToolCallError(Exception):
    """The model's own output was not a usable tool call.

    Raised only *within* this module, to be caught immediately by
    :func:`run_tool_loop` and turned into a tool-result turn reporting the
    error — never propagated out of the loop. A malformed call is not a bug
    in the harness, it is the single most informative signal a small or
    undertrained model can produce about what it still needs to learn, and
    the loop's whole job is to keep going and hand that signal back as
    context rather than crash the caller's program.
    """


@dataclass(frozen=True)
class Tool:
    """One callable the model may invoke.

    ``parameters`` is a JSON-Schema object describing the arguments dict —
    the same schema :mod:`iridium.runtime.constrained` decodes against and
    :mod:`jsonschema` validates a completed call against, so there is exactly
    one schema per tool, read by both the constrained decoder and the
    post-hoc validator, rather than two descriptions that can drift apart.
    ``fn`` is optional so a ``Tool`` can describe a call the caller intends to
    execute out-of-band (e.g. a human-in-the-loop action) — :meth:`ToolRegistry.call`
    raises plainly if asked to run one that has no ``fn``.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    fn: Optional[Callable[..., Any]] = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError(
                f"tool name {self.name!r} must be a valid identifier — it is "
                "serialized verbatim into JSON and into the system prompt, "
                "and an identifier is the one shape guaranteed not to need "
                "escaping in either place"
            )
        if not isinstance(self.parameters, Mapping) or self.parameters.get("type") != "object":
            raise ValueError(
                f"tool {self.name!r}: parameters must be a JSON-Schema object "
                "schema (top-level {'type': 'object', 'properties': {...}}) — "
                "a tool call's arguments are always a dict, never a bare "
                "scalar or array, so anything else cannot describe one"
            )
        jsonschema.validators.Draft202012Validator.check_schema(dict(self.parameters))

    def signature(self) -> str:
        """One compact, deterministic line for the system prompt.

        Deterministic matters as much as compact: two equal ``Tool`` objects
        must render identically, because this string is part of the training
        distribution (see ``tool_corpus.system_prompt_for``) — a renderer
        that iterates a ``dict`` in whatever order Python happens to give it
        would make the same tool set describe itself differently run to run.
        """
        props = self.parameters.get("properties", {}) or {}
        required = set(self.parameters.get("required", ()))
        args = ", ".join(
            f"{name}{'' if name in required else '?'}: {_type_hint(schema)}"
            for name, schema in sorted(props.items())
        )
        return f"{self.name}({args}) — {self.description}"


def _type_hint(schema: Mapping[str, Any]) -> str:
    t = schema.get("type", "any")
    if t == "string" and "enum" in schema:
        return "enum[" + ",".join(schema["enum"]) + "]"
    if t == "array":
        return f"array[{_type_hint(schema.get('items', {}))}]"
    return str(t)


class ToolRegistry:
    """Name -> :class:`Tool`, and the one place a call is actually executed.

    This *is* the trust boundary (see the module docstring): register only
    functions the caller is willing to have invoked with model-chosen,
    schema-shaped-but-otherwise-untrusted arguments.
    """

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def validate_arguments(self, name: str, arguments: Mapping[str, Any]) -> None:
        """Raise :class:`jsonschema.ValidationError` if ``arguments`` do not
        match the named tool's schema. Split out from :meth:`call` so
        constrained decoding (which already guarantees schema validity by
        construction) and unconstrained decoding (which does not, and needs
        this checked before ``fn`` ever sees the arguments) can share one
        validation path rather than each rolling its own."""
        tool = self._tools[name]
        jsonschema.validate(dict(arguments), dict(tool.parameters))

    def call(self, name: str, arguments: Mapping[str, Any]) -> Any:
        """Validate then execute. Raises on a bad name/arguments/absent
        ``fn`` — callers in a tool loop are expected to catch this and turn
        it into a tool-result error, exactly as they do for the underlying
        tool raising; this method does not swallow anything itself, because
        a registry used directly (outside :func:`run_tool_loop`, e.g. from a
        notebook) should fail loudly rather than silently."""
        if name not in self._tools:
            raise KeyError(f"no tool registered under {name!r}; known: {self.names()}")
        self.validate_arguments(name, arguments)
        tool = self._tools[name]
        if tool.fn is None:
            raise RuntimeError(f"tool {name!r} has no callable attached; it is a spec only")
        return tool.fn(**dict(arguments))

    def system_prompt_block(self) -> str:
        return system_prompt_block(self._tools[n] for n in self.names())


def system_prompt_block(tools: Sequence[Tool]) -> str:
    """Render tool specs into the compact, stable text a system turn carries.

    Sorted by name and joined with a fixed separator: the block is part of
    what the model conditions on, so two calls with the same tool *set* must
    produce byte-identical text regardless of registration order, or the same
    logical prompt trains/serves as two different prompts.
    """
    lines = ["TOOLS:"] + [f"- {t.signature()}" for t in sorted(tools, key=lambda t: t.name)]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# wire format: {"name": ..., "arguments": {...}}
# --------------------------------------------------------------------------


def format_tool_call(name: str, arguments: Mapping[str, Any]) -> str:
    """The one serialization every call in this codebase uses.

    ``name`` first, always — :mod:`iridium.runtime.constrained`'s
    ``ToolCallValidator`` depends on this exact key order to resolve which
    tool's argument schema applies as soon as the name string closes, rather
    than buffering unconstrained content in case ``arguments`` came first
    (see that module's docstring). ``json.dumps`` on a plain ``dict`` in
    Python 3.7+ preserves insertion order, so building the dict
    ``{"name": ..., "arguments": ...}`` in this order is sufficient — no
    separate key-ordering step is needed, but it is exactly why this
    function exists rather than every caller doing its own ``json.dumps``.
    """
    return json.dumps({"name": name, "arguments": dict(arguments)}, separators=(",", ":"))


def parse_tool_call(text: str) -> tuple[str, dict]:
    """Inverse of :func:`format_tool_call`. Raises :class:`ToolCallError` —
    never a bare ``json.JSONDecodeError``/``KeyError`` — so every caller in
    this module can catch one exception type regardless of which way the
    text was malformed."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolCallError(f"tool call is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict) or set(obj) != {"name", "arguments"}:
        raise ToolCallError(
            'tool call must be exactly {"name": ..., "arguments": {...}}, got '
            f"{obj!r}"
        )
    name, arguments = obj["name"], obj["arguments"]
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise ToolCallError('"name" must be a string and "arguments" an object')
    return name, arguments


def extract_tool_call(ids: Sequence[int], text_offset: int = TEXT_OFFSET) -> Optional[str]:
    """Given one turn's raw generated ids, the JSON text if it is a tool call,
    else ``None`` if it is an ordinary text reply.

    The distinguishing fact is purely positional: :func:`run_tool_loop` opens
    a turn the same way for either outcome (see ``chat.conversation_spans``'s
    ``open_for_reply``) and lets the model's very first emitted id decide
    which kind of turn this is — :data:`~iridium.runtime.chat.TOOL_CALL` if
    it chose to call a tool, an ordinary text byte otherwise. Bytes after
    that are decoded exactly the way ``generate.Generated.text`` already
    does (subtract ``text_offset``, drop anything outside ``0..255``, ignore
    any other control id that sneaks in) so the JSON text handed to
    :func:`parse_tool_call` matches what training actually supervised.
    """
    if not ids or int(ids[0]) != TOOL_CALL:
        return None
    body = bytearray()
    for token in ids[1:]:
        v = int(token) - text_offset
        if 0 <= v < 256:
            body.append(v)
    return body.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


@dataclass
class ToolLoopResult:
    answer: str
    turns: list[Turn]
    calls: list[dict] = field(default_factory=list)   # [{"name", "arguments", "result"|"error"}]
    stopped: str = "answer"                            # "answer" | "max_calls"


def run_tool_loop(
    model,
    messages: Sequence[Turn],
    registry: ToolRegistry,
    max_calls: int = 4,
    max_new_tokens: int = 256,
    seed: int = 0,
    max_bytes: int = 4096,
    generate_fn: Optional[Callable[..., Any]] = None,
    **generate_kwargs,
) -> ToolLoopResult:
    """Generate -> execute -> feed the result back -> repeat, until a final
    answer or ``max_calls`` is spent.

    ``messages`` is the conversation so far (typically a system turn carrying
    :meth:`ToolRegistry.system_prompt_block`, then user/assistant turns);
    this function only ever *appends* ``"tool_call"``/``"tool_result"`` turns,
    matching exactly what :mod:`iridium.data.tool_corpus` trains on, so a
    session built by this loop is legal input to itself on the next call.

    A malformed call — bad JSON, wrong top-level shape, unknown tool name,
    arguments that fail the schema — is **not** raised out of this loop: it
    is turned into a tool-result turn describing the error (see the module
    docstring on why raising here would be the wrong response to exactly the
    situation this loop exists to handle) and generation continues, spending
    one of the ``max_calls`` budget the same as a real call would. Running
    out of budget without a final answer is reported via
    ``ToolLoopResult.stopped == "max_calls"`` rather than an exception,
    mirroring ``iridium.agency.media_agent``'s own "no completion claimed"
    discipline: a caller must check ``stopped`` rather than assume ``answer``
    is a real answer.
    """
    if max_calls < 1:
        raise ValueError("max_calls must be at least 1")
    if generate_fn is None:
        from .generate import generate as generate_fn

    params = dict(
        max_new_tokens=max_new_tokens, stop_ids=STOP_IDS, text_offset=TEXT_OFFSET,
        text_only=True, seed=seed,
    )
    params.update(generate_kwargs)

    turns = list(messages)
    calls: list[dict] = []
    max_seq_len = model.cfg.max_seq_len

    for _ in range(max_calls):
        context = fit_to_budget(turns, max_bytes)
        while context and context[0].role == "assistant":
            context = context[1:]
        sample = conversation_sample(context, supervise_assistant=False, open_for_reply=True)
        budget = params["max_new_tokens"]
        if len(sample) + budget > max_seq_len:
            budget = max(1, max_seq_len - len(sample))
        step_params = dict(params, max_new_tokens=budget)
        out = generate_fn(model, sample, **step_params)
        params["seed"] = params.get("seed", seed) + 1  # vary sampling call to call

        call_text = extract_tool_call(out.ids, TEXT_OFFSET)
        if call_text is None:
            answer = out.text.strip()
            turns = [*context, Turn("assistant", answer)]
            return ToolLoopResult(answer=answer, turns=turns, calls=calls, stopped="answer")

        turns = [*context, Turn("tool_call", call_text)]
        try:
            name, arguments = parse_tool_call(call_text)
            if name not in registry:
                raise ToolCallError(f"no such tool {name!r}; known: {registry.names()}")
            registry.validate_arguments(name, arguments)
            result = registry.call(name, arguments)
            record = {"name": name, "arguments": arguments, "result": result}
            result_text = json.dumps({"ok": True, "result": result}, separators=(",", ":"))
        except (ToolCallError, jsonschema.ValidationError, KeyError, RuntimeError) as exc:
            record = {"name": None, "arguments": None, "error": str(exc)}
            result_text = json.dumps({"ok": False, "error": str(exc)}, separators=(",", ":"))
        except Exception as exc:  # the tool's own fn raised something else
            record = {"name": None, "arguments": None, "error": f"{type(exc).__name__}: {exc}"}
            result_text = json.dumps({"ok": False, "error": record["error"]}, separators=(",", ":"))
        calls.append(record)
        turns = [*turns, Turn("tool_result", result_text)]

    return ToolLoopResult(answer="", turns=turns, calls=calls, stopped="max_calls")
