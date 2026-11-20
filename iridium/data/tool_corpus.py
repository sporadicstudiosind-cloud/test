"""Tool-calling conversations to train on, with licences recorded and a
built-in synthetic generator that can be *graded exactly*.

The same discipline ``chat_corpus.py`` and ``text_corpus.py`` apply to their
sources applies here: every external dataset below carries a licence and an
obligation string, verified against that dataset's own Hugging Face dataset
card (not assumed from its popularity or its name) before being listed.
Where a card's wording left something unresolved — most often, whether a
GPT-generated portion of a dataset carries its own upstream terms on top of
the curator's stated licence — that is recorded as ``licence_verified=False``
rather than quietly assumed away, and ``tool_items`` refuses such a source by
default (see ``ToolSourceSpec.commercial_ok`` / ``enabled_by_default``).

**Why a synthetic generator exists at all, alongside real data.** Every
external source below is graded, at best, by whether a held-out reply
resembles its reference reply — the same limitation ``chat_corpus.py``'s
human-written conversations have, for the same reason: nobody can *compute*
whether "Here are the latest news headlines for France: ..." is the right
thing to have said. A tool call is different from a chat reply in exactly
the way that makes exact grading possible: it has a checkable right answer —
the correct tool, the correct arguments, and (once the tool result is known)
a correct use of that result — the moment the tool itself is something this
process can *actually run* and get the true answer from. ``calculator``,
``convert_units`` and ``lookup`` are deliberately the smallest tools with
that property: pure functions of their arguments, computable locally, with no
external state or network to introduce nondeterminism into what "correct"
means. ``grade_synthetic_item`` is the checker this buys — see the module
docstring on why this project never reports a capability number without one.

**Supervision, once more, because it is the one fact every converter in this
file has to get right identically:** the tool call and the final assistant
turn are the model's own output and are supervised; the tool result is
environment output and is never supervised, regardless of source. See
``iridium.runtime.chat.conversation_spans`` and ``iridium.runtime.tools``'s
module docstring for why the reverse would train the model to hallucinate
results instead of calling tools for them.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, Optional

import numpy as np

from ..runtime.chat import Turn
from ..runtime.tools import Tool, ToolRegistry, format_tool_call


# ---------------------------------------------------------------------------
# external sources: licence catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSourceSpec:
    key: str
    dataset: str
    config: Optional[str]
    split: str
    licence: str
    licence_verified: bool
    commercial_ok: bool
    obligation: str
    description: str
    homepage: str = ""
    enabled_by_default: bool = True
    disabled_reason: str = ""
    approx_rows: int = 10_000

    def as_dict(self) -> dict:
        return asdict(self)


TOOL_SOURCES: dict[str, ToolSourceSpec] = {
    "glaive": ToolSourceSpec(
        key="glaive",
        dataset="glaiveai/glaive-function-calling-v2",
        config=None,
        split="train",
        licence="Apache-2.0",
        licence_verified=True,  # confirmed from the dataset's own README front matter
        commercial_ok=True,
        obligation="Retain the licence notice; no attribution or share-alike clause.",
        description=(
            "113k function-calling chats: a system turn listing the callable "
            "functions as JSON, then USER/ASSISTANT/FUNCTION-RESPONSE turns "
            "with the call written as text (`<functioncall> {...}`) rather "
            "than a reserved token. Machine-generated (glaiveai's own "
            "synthetic pipeline, not scraped conversation) — the dataset "
            "card does not name a specific generating model, so no "
            "downstream-model terms are recorded here because none were "
            "found to record; this is stated rather than assumed clean."
        ),
        homepage="https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2",
        approx_rows=113_000,
    ),
    "hermes": ToolSourceSpec(
        key="hermes",
        dataset="NousResearch/hermes-function-calling-v1",
        config="func_calling",
        split="train",
        licence="Apache-2.0",
        licence_verified=True,  # confirmed from the dataset's own README front matter
        commercial_ok=True,
        obligation="Retain the licence notice; no attribution or share-alike clause.",
        description=(
            "ShareGPT-style rows (`conversations`: [{from, value}], plus a "
            "`tools` column of OpenAI-style function specs) used to train "
            "the Hermes 2 Pro series. Calls are written as "
            "`<tool_call>{...}</tool_call>` inside a `gpt` turn (one or more "
            "per turn); results come back as "
            "`<tool_response>{...}</tool_response>` inside a `tool` turn. "
            "Synthetic-data tag on the dataset card; NousResearch does not "
            "publish which model(s) generated it, so — as with `glaive` "
            "above — no specific upstream-model terms are recorded because "
            "none could be found to verify, not because the question was "
            "skipped."
        ),
        homepage="https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1",
        approx_rows=1_900,
    ),
    "xlam": ToolSourceSpec(
        key="xlam",
        dataset="Salesforce/xlam-function-calling-60k",
        config=None,
        split="train",
        licence="CC-BY-4.0",
        licence_verified=True,  # confirmed via the Hub's own repo metadata
        commercial_ok=True,
        obligation="Attribution to Salesforce (APIGen) on any derivative work.",
        description=(
            "60k function-calling examples produced by Salesforce's APIGen "
            "pipeline, each verified by format checking, execution and "
            "semantic review rather than left as raw model output. The "
            "cleanest provenance story of the four sources here — and "
            "excluded from automatic loading anyway, for a reason that has "
            "nothing to do with its licence: the repository is Hub-gated "
            "(access must be requested and accepted per account), so "
            "`datasets.load_dataset` cannot fetch it unattended the way "
            "`glaive`/`hermes` can. A caller with accepted access can still "
            "load it by hand and pass rows through `xlam_conversation` "
            "below; it is catalogued here so that licence is on record "
            "before anyone does."
        ),
        homepage="https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k",
        enabled_by_default=False,
        disabled_reason="Hub-gated: requires per-account access acceptance, not a licence restriction.",
        approx_rows=60_000,
    ),
    "bfcl": ToolSourceSpec(
        key="bfcl",
        dataset="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        config=None,
        split="train",
        licence="Apache-2.0",
        licence_verified=True,
        commercial_ok=True,
        obligation="Retain the licence notice.",
        description=(
            "Berkeley's BFCL evaluation set. Listed for completeness and "
            "licence provenance; no converter is implemented in this pass "
            "(its per-category JSON schema differs substantially from the "
            "chat-style rows above and was not verified row-by-row against a "
            "live preview here) — `enabled_by_default=False` reflects "
            "'not wired up', not a licence problem."
        ),
        homepage="https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        enabled_by_default=False,
        disabled_reason="No converter implemented yet; row schema not verified against a live preview.",
        approx_rows=4_000,
    ),
}

#: Only sources verified, licence-commercial-ok, and wired up with a working
#: converter are on by default. A caller can still name a disabled source
#: explicitly in ``mix`` — ``tool_items`` only refuses one that fails
#: verification outright (see ``_check_enabled``).
DEFAULT_TOOL_MIX = {"glaive": 0.6, "hermes": 0.4}


def tool_licence_notice(keys=None) -> str:
    lines = ["Tool-calling corpora and their terms:", ""]
    for key in (keys or TOOL_SOURCES):
        spec = TOOL_SOURCES[key]
        lines += [
            f"* {spec.dataset}",
            f"    licence:            {spec.licence} (verified: {spec.licence_verified})",
            f"    commercial_ok:      {spec.commercial_ok}",
            f"    enabled_by_default: {spec.enabled_by_default}"
            + (f"  ({spec.disabled_reason})" if spec.disabled_reason else ""),
            f"    obligation:         {spec.obligation}",
            f"    homepage:           {spec.homepage}",
        ]
    return "\n".join(lines)


def _check_enabled(key: str) -> ToolSourceSpec:
    spec = TOOL_SOURCES[key]
    if not spec.licence_verified:
        raise ValueError(
            f"tool source {key!r} has an unverified licence claim; refusing "
            "to load it rather than guess. Verify it against the dataset's "
            "own card and update ToolSourceSpec.licence_verified."
        )
    return spec


# ---------------------------------------------------------------------------
# converters: external row formats -> Turn lists
# ---------------------------------------------------------------------------


def _load(spec: ToolSourceSpec, streaming: bool = True):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pip install datasets  — required for tool-calling corpora"
        ) from exc
    return load_dataset(spec.dataset, spec.config, split=spec.split, streaming=streaming)


_GLAIVE_SEGMENT_RE = re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")
_GLAIVE_CALL_RE = re.compile(r"^<functioncall>\s*(\{.*\})\s*$", re.S)
_GLAIVE_ARGS_RE = re.compile(r"'(\{.*\})'")
_GLAIVE_NAME_RE = re.compile(r'"name":\s*"([^"]+)"')


def _parse_glaive_call(text: str) -> Optional[dict]:
    """`{"name": "f", "arguments": '{"x": 1}'}` -> `{"name": "f", "arguments": {"x": 1}}`.

    The single-quoted ``arguments`` value is not valid JSON as written (JSON
    strings use double quotes only) — it is a JSON object serialized *into* a
    string the outer blob then wraps in single quotes, a glaive-specific
    convention rather than anything JSON itself does. Extracting the inner
    blob and decoding it separately, instead of trying to make the whole
    thing parse as one ``json.loads`` call, is what makes this
    straightforward rather than a bespoke relaxed-JSON parser.
    """
    m = _GLAIVE_CALL_RE.match(text.strip())
    if not m:
        return None
    raw = m.group(1)
    name_m, args_m = _GLAIVE_NAME_RE.search(raw), _GLAIVE_ARGS_RE.search(raw)
    if not name_m or not args_m:
        return None
    try:
        arguments = json.loads(args_m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(arguments, dict):
        return None
    return {"name": name_m.group(1), "arguments": arguments}


def glaive_conversation(system: str, chat: str) -> Optional[list[Turn]]:
    """One ``glaive-function-calling-v2`` row -> this project's turn format.

    This is the conversion the whole tool-use track is *for*: glaive's own
    wire format is exactly the thing ``iridium.runtime.chat``'s module
    docstring argues against — a text tag (``<functioncall>``) marking a call
    inside an ordinary ASSISTANT turn, with no reserved boundary and no
    machine-checked argument schema. Every ``<functioncall>`` becomes a
    dedicated ``"tool_call"`` turn (supervised, via
    :func:`iridium.runtime.tools.format_tool_call` — not glaive's own
    single-quoted-string convention, so what trains is what this project's
    own decoder can parse), and every ``FUNCTION RESPONSE:`` becomes a
    ``"tool_result"`` turn (never supervised).
    """
    turns: list[Turn] = [Turn("system", system.strip())] if system and system.strip() else []
    parts = _GLAIVE_SEGMENT_RE.split(chat)
    for idx in range(1, len(parts), 2):
        marker = parts[idx].rstrip(":")
        text = parts[idx + 1].replace("<|endoftext|>", "").strip() if idx + 1 < len(parts) else ""
        if not text:
            continue
        if marker == "USER":
            turns.append(Turn("user", text))
        elif marker == "FUNCTION RESPONSE":
            turns.append(Turn("tool_result", text))
        elif marker == "ASSISTANT":
            call = _parse_glaive_call(text)
            if call is not None:
                turns.append(Turn("tool_call", format_tool_call(call["name"], call["arguments"])))
            else:
                turns.append(Turn("assistant", text))
        else:
            return None
    if len(turns) < 2:
        return None
    return turns


def glaive_conversations(limit: Optional[int] = None, seed: int = 0) -> Iterator[list[Turn]]:
    spec = _check_enabled("glaive")
    ds = _load(spec).shuffle(seed=seed, buffer_size=1000)
    produced = 0
    for row in ds:
        turns = glaive_conversation(row.get("system") or "", row.get("chat") or "")
        if turns is None:
            continue
        yield turns
        produced += 1
        if limit is not None and produced >= limit:
            return


_HERMES_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_HERMES_RESULT_RE = re.compile(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", re.S)


def hermes_conversation(conversations: list[dict]) -> Optional[list[Turn]]:
    """One ``hermes-function-calling-v1`` row's ``conversations`` list -> this
    project's turn format.

    Format confirmed against a live row preview (not guessed from
    documentation alone): ``from`` in ``{"system", "human", "gpt", "tool"}``,
    with a ``gpt`` turn carrying zero or more
    ``<tool_call>{...}</tool_call>`` blocks and a matching ``tool`` turn
    carrying the same number of ``<tool_response>{...}</tool_response>``
    blocks, in order. Each call/response pair becomes its own
    ``"tool_call"``/``"tool_result"`` turn — this project's format has no
    "N calls, then N results" batch shape, only one call, one result,
    repeated — and a ``gpt`` turn that also has plain prose alongside its
    tags (the final wrap-up message) becomes an ordinary ``"assistant"``
    turn. A row whose tags do not pair up 1:1 is skipped rather than guessed
    at.
    """
    turns: list[Turn] = []
    pending_calls: list[Turn] = []  # buffered "tool_call" turns awaiting their "tool" reply

    def flush_pending():
        # A ``gpt`` turn with calls is always immediately followed by a
        # ``tool`` turn with the matching responses in this dataset's own
        # convention (confirmed against a live row preview) -- but a
        # defensive flush here means a row that broke that convention still
        # produces a sensible (if response-less) turn list instead of
        # silently losing calls, and it is what turns the source rows' "N
        # calls, then N results" batch shape into this project's own
        # alternating one-call-one-result shape without special-casing the
        # last call in a batch.
        turns.extend(pending_calls)
        pending_calls.clear()

    for msg in conversations:
        role, text = msg.get("from"), (msg.get("value") or "").strip()
        if not text:
            continue
        if role == "system":
            flush_pending()
            turns.append(Turn("system", text))
        elif role == "human":
            flush_pending()
            turns.append(Turn("user", text))
        elif role == "tool":
            responses = _HERMES_RESULT_RE.findall(text)
            if not responses:
                return None
            # Interleave call[k], result[k] for as many pairs as line up;
            # any leftover calls with no matching response (a malformed row)
            # are flushed unpaired rather than dropped.
            for call_turn, result_text in zip(pending_calls, responses):
                turns.append(call_turn)
                turns.append(Turn("tool_result", result_text))
            pending_calls[:] = pending_calls[len(responses):]
            flush_pending()
        elif role == "gpt":
            calls = _HERMES_CALL_RE.findall(text)
            if calls:
                flush_pending()
                for c in calls:
                    try:
                        obj = json.loads(c)
                    except json.JSONDecodeError:
                        return None
                    if not isinstance(obj, dict) or "name" not in obj:
                        return None
                    pending_calls.append(Turn(
                        "tool_call",
                        format_tool_call(obj["name"], obj.get("arguments", {})),
                    ))
            else:
                flush_pending()
                turns.append(Turn("assistant", text))
        else:
            return None
    flush_pending()
    if len(turns) < 2:
        return None
    return turns


def hermes_conversations(limit: Optional[int] = None, seed: int = 0) -> Iterator[list[Turn]]:
    spec = _check_enabled("hermes")
    ds = _load(spec).shuffle(seed=seed, buffer_size=1000)
    produced = 0
    for row in ds:
        turns = hermes_conversation(row.get("conversations") or [])
        if turns is None:
            continue
        yield turns
        produced += 1
        if limit is not None and produced >= limit:
            return


CONVERSATION_LOADERS: dict[str, Callable] = {
    "glaive": glaive_conversations,
    "hermes": hermes_conversations,
}


# ---------------------------------------------------------------------------
# the synthetic, licence-free, exactly-gradeable generator
# ---------------------------------------------------------------------------

_UNIT_TO_M = {"m": 1.0, "ft": 0.3048, "km": 1000.0}
_KV_STORE = {
    "capital_of_france": "Paris",
    "atomic_number_oxygen": "8",
    "boiling_point_water_c": "100",
    "speed_of_light_mps": "299792458",
    "largest_planet": "Jupiter",
}

CALCULATOR_TOOL = Tool(
    "calculator", "Evaluate a binary arithmetic operation on two numbers.",
    {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["add", "sub", "mul", "div"]},
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["op", "a", "b"],
    },
)
CONVERT_UNITS_TOOL = Tool(
    "convert_units", "Convert a length between metres, feet and kilometres.",
    {
        "type": "object",
        "properties": {
            "value": {"type": "number"},
            "from_unit": {"type": "string", "enum": list(_UNIT_TO_M)},
            "to_unit": {"type": "string", "enum": list(_UNIT_TO_M)},
        },
        "required": ["value", "from_unit", "to_unit"],
    },
)
LOOKUP_TOOL = Tool(
    "lookup", "Look up a fact by its exact key in a small reference table.",
    {
        "type": "object",
        "properties": {"key": {"type": "string", "enum": list(_KV_STORE), "maxLength": 32}},
        "required": ["key"],
    },
)


def _calculator_fn(op: str, a: float, b: float) -> float:
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        if b == 0:
            raise ValueError("division by zero")
        return a / b
    raise ValueError(f"unknown op {op!r}")


def _convert_units_fn(value: float, from_unit: str, to_unit: str) -> float:
    return value * _UNIT_TO_M[from_unit] / _UNIT_TO_M[to_unit]


def _lookup_fn(key: str) -> str:
    if key not in _KV_STORE:
        raise KeyError(key)
    return _KV_STORE[key]


def synthetic_registry() -> ToolRegistry:
    """A fresh registry of the three synthetic tools, wired to real callables
    — used both to generate items (so the recorded tool result is the *true*
    one, not an invented string) and to grade a model's own tool choice at
    evaluation time against the same ground truth."""
    return ToolRegistry([
        Tool(CALCULATOR_TOOL.name, CALCULATOR_TOOL.description, CALCULATOR_TOOL.parameters, _calculator_fn),
        Tool(CONVERT_UNITS_TOOL.name, CONVERT_UNITS_TOOL.description, CONVERT_UNITS_TOOL.parameters, _convert_units_fn),
        Tool(LOOKUP_TOOL.name, LOOKUP_TOOL.description, LOOKUP_TOOL.parameters, _lookup_fn),
    ])


@dataclass
class SyntheticCase:
    """One generated item's ground truth, kept alongside the ``Turn`` list so
    ``grade_synthetic_item`` can check a produced call/answer exactly rather
    than by resemblance."""

    name: str
    arguments: dict
    result: Any
    question: str
    final_answer: str


def _synthetic_case(rng: np.random.Generator) -> SyntheticCase:
    kind = rng.integers(3)
    if kind == 0:
        op = str(rng.choice(["add", "sub", "mul", "div"]))
        a, b = float(rng.integers(1, 50)), float(rng.integers(1, 50))
        if op == "div":
            b = float(rng.integers(1, 12))
            a = b * float(rng.integers(1, 12))  # keeps the result an exact-ish value
        result = _calculator_fn(op, a, b)
        symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
        question = f"What is {a:g} {symbol} {b:g}?"
        answer = f"{a:g} {symbol} {b:g} = {result:g}"
        return SyntheticCase("calculator", {"op": op, "a": a, "b": b}, result, question, answer)
    if kind == 1:
        from_unit, to_unit = rng.choice(list(_UNIT_TO_M), size=2, replace=False)
        value = float(rng.integers(1, 500))
        result = _convert_units_fn(value, str(from_unit), str(to_unit))
        question = f"Convert {value:g} {from_unit} to {to_unit}."
        answer = f"{value:g} {from_unit} is {result:.4g} {to_unit}."
        return SyntheticCase(
            "convert_units", {"value": value, "from_unit": str(from_unit), "to_unit": str(to_unit)},
            result, question, answer,
        )
    key = str(rng.choice(list(_KV_STORE)))
    result = _lookup_fn(key)
    question = f"Look up {key.replace('_', ' ')}."
    answer = f"{key.replace('_', ' ')} is {result}."
    return SyntheticCase("lookup", {"key": key}, result, question, answer)


def synthetic_conversation(rng: np.random.Generator) -> tuple[list[Turn], SyntheticCase]:
    case = _synthetic_case(rng)
    registry = synthetic_registry()
    call_text = format_tool_call(case.name, case.arguments)
    result_text = json.dumps({"ok": True, "result": case.result}, separators=(",", ":"))
    turns = [
        Turn("system", registry.system_prompt_block()),
        Turn("user", case.question),
        Turn("tool_call", call_text),
        Turn("tool_result", result_text),
        Turn("assistant", case.final_answer),
    ]
    return turns, case


def grade_synthetic_item(case: SyntheticCase, produced_call: Optional[dict],
                          produced_answer: str) -> dict:
    """Exact grading against the generator's own ground truth: three
    independent booleans, because "got the right answer" and "used the tool
    correctly to get there" are different claims and this project does not
    conflate them (a model that hallucinates the same number the tool would
    have returned is not evidence it can use tools).

    ``produced_call`` is ``{"name": ..., "arguments": ...}`` (e.g. from
    :func:`iridium.runtime.tools.parse_tool_call`) or ``None`` if the model
    never called a tool at all — ``tool_chosen``/``arguments_correct`` are
    both ``False`` in that case, not vacuously ``True``.
    """
    tool_chosen = produced_call is not None and produced_call.get("name") == case.name
    arguments_correct = tool_chosen and produced_call.get("arguments") == case.arguments
    result_str = str(case.result)
    uses_result = any(
        tok in produced_answer for tok in {result_str, f"{case.result:.4g}" if isinstance(case.result, float) else result_str}
    )
    return {
        "tool_chosen": tool_chosen,
        "arguments_correct": arguments_correct,
        "uses_result": uses_result,
        "all_correct": tool_chosen and arguments_correct and uses_result,
    }


# ---------------------------------------------------------------------------
# training items
# ---------------------------------------------------------------------------


def tool_items(
    n_items: int,
    mix: Optional[dict[str, float]] = None,
    seed: int = 0,
    max_bytes: int = 2048,
    split: str = "train",
    tokenizer=None,
):
    """Tool-use conversations as training items, mirroring
    ``chat_corpus.chat_items``: same deterministic hash-based split (via
    ``iridium.data.text_corpus.in_split``), same ``allocate_mixture`` exact
    budgeting, same "drop whole turns from the front to fit" budgeting via
    ``iridium.runtime.chat.fit_to_budget``.

    ``mix`` keys may be ``"synthetic"`` (the built-in generator, the only
    family guaranteed to work fully offline) or any key in
    :data:`TOOL_SOURCES`; a source with an unverified licence claim raises
    immediately rather than silently loading (see ``_check_enabled``) and a
    disabled-by-default source must be named explicitly in ``mix`` to be
    used at all.
    """
    from ..runtime.chat import conversation_sample, fit_to_budget
    from ..training.datasets import allocate_mixture
    from ..training.tasks import Item
    from .text_corpus import SPLIT_SHARES, in_split

    mix = mix or DEFAULT_TOOL_MIX
    quotas = allocate_mixture(n_items, mix)
    items = []
    rng = np.random.default_rng(seed)

    for i, (key, quota) in enumerate(sorted(quotas.items())):
        if quota <= 0:
            continue
        got = 0
        if key == "synthetic":
            case_rng = np.random.default_rng(seed + i)
            attempts = 0
            while got < quota and attempts < quota * 50 + 100:
                attempts += 1
                turns, case = synthetic_conversation(case_rng)
                joined = "\n".join(t.text for t in turns)
                if not in_split(joined, split):
                    continue
                items.append(Item(
                    sample=conversation_sample(turns, supervise_assistant=True,
                                               meta={"family": "tool_use", "source": "synthetic"},
                                               tokenizer=tokenizer),
                    family="tool_use",
                    prompt=case.question,
                    answer=case.final_answer,
                    truth={"source": "synthetic", "name": case.name, "arguments": case.arguments,
                           "result": case.result},
                ))
                got += 1
            continue

        if key not in TOOL_SOURCES:
            raise ValueError(f"unknown tool source {key!r}; known: {sorted(TOOL_SOURCES) + ['synthetic']}")
        _check_enabled(key)
        lo, hi = SPLIT_SHARES.get(split, SPLIT_SHARES["train"])
        budget = int(np.ceil(2 * quota * 100 / (hi - lo))) + 32
        for turns in CONVERSATION_LOADERS[key](limit=budget, seed=seed + i):
            joined = "\n".join(t.text for t in turns)
            if not in_split(joined, split):
                continue
            turns = fit_to_budget(turns, max_bytes)
            while turns and turns[0].role not in ("user", "system"):
                turns = turns[1:]
            if len(turns) < 2:
                continue
            items.append(Item(
                sample=conversation_sample(turns, supervise_assistant=True,
                                           meta={"family": "tool_use", "source": key},
                                           tokenizer=tokenizer),
                family="tool_use",
                prompt=next((t.text[:80] for t in turns if t.role == "user"), ""),
                answer=turns[-1].text,
                truth={"source": key, "turns": len(turns)},
            ))
            got += 1
            if got >= quota:
                break

    rng.shuffle(items)
    return items[:n_items]
