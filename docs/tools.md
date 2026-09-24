# Tool use

Nothing in this document is trained. It describes the format, the
constrained-decoding guarantees (and their limits), the data, and how to
grade — the same measured-not-claimed discipline as `docs/results.md`.

## 1. Format

Two new control ids, added to `iridium/runtime/chat.py`'s marker block:
`TOOL_CALL = 11`, `TOOL_RESULT = 12`. (Id `10` was already spent, before
this track existed, on `iridium.agency.media_agent`'s generic `"tool"` role
— left untouched; see `chat.py`'s docstring.) They are reserved ids, not a
text tag like `<tool_call>` — two of the three licensed corpora below use
exactly that tag, and so does `media_agent`'s own call-side convention — for
three concrete reasons: a byte-level tokenizer can split a text tag across
an arbitrary number of tokens depending on context, a user's own message can
contain the literal string, and every occurrence spends sequence budget a
reserved id does not.

A conversation with a tool call is a `Turn` sequence:

```
system("TOOLS:\n- add(a, b) - ...")
user("what is 2 + 3?")
tool_call('{"name":"add","arguments":{"a":2,"b":3}}')   # TOOL_CALL ... EOS
tool_result('{"ok":true,"result":5}')                    # TOOL_RESULT ...
assistant("2 + 3 = 5")                                    # ASSISTANT ... EOS
```

`iridium.runtime.chat.conversation_spans` handles `"tool_call"`/`"tool_result"`
exactly like it already handles `"assistant"`/`"user"`: a `TOOL_CALL`/
`TOOL_RESULT` turn gets its role marker, its text, and (for `tool_call`) a
trailing EOS. This is additive — existing roles, existing tests, and
`ChatSession` are unchanged.

Wire format: `{"name": ..., "arguments": {...}}`, compact JSON,
**`"name"` always first** — `iridium.runtime.tools.format_tool_call` is the
one place this is serialized, and `iridium.runtime.constrained.ToolCallValidator`
depends on that exact order to know which tool's argument schema applies as
soon as the name closes.

Tool specs render into the system turn via `ToolRegistry.system_prompt_block()`:
one line per tool, sorted by name, so the same tool set always produces the
same prompt text regardless of registration order.

MCP-style definitions (`{"name", "description", "inputSchema"}`) register
directly — `Tool.from_mcp(spec, fn=...)` — since `inputSchema` already *is*
a JSON-Schema object schema, the same thing `Tool.parameters` is.

## 2. Supervision

**Supervised:** the tool call, the same as a prose reply — it is the model's
own output.
**Never supervised:** the tool result — it is the environment's output.
Training on it teaches the model to predict what a tool will say instead of
calling it, which looks identical to real learning on a loss curve and is a
model that hallucinates API responses in production.

This is enforced once, in `chat.conversation_spans`, not re-implemented by
every data source or the inference loop.

## 3. The loop (`iridium.runtime.tools.run_tool_loop`)

Generate → if the first emitted id is `TOOL_CALL`, parse the JSON, validate
arguments against the named tool's schema (`jsonschema`), execute via the
registry, append the result as a `tool_result` turn, repeat (up to
`max_calls`) → otherwise the turn is a final answer, return it.

A malformed call (bad JSON, unknown tool, schema-invalid arguments) is
**never** raised out of the loop — it becomes a tool-result turn describing
the error, spending one call of budget, so the model gets a chance to
correct itself. Running out of `max_calls` without an answer is reported as
`ToolLoopResult.stopped == "max_calls"`, not an exception — a caller must
check `stopped`, mirroring `media_agent`'s "no completion claimed" discipline.

**Execution safety.** `ToolRegistry` maps a name to a caller-supplied Python
callable. Nothing in this track ever `exec`s or `eval`s model-produced code
— the model chooses a name and a dict of arguments, both inert data. The
registry *is* the trust boundary: registering `os.system` as a tool hands
the model a shell regardless of how tightly its arguments are schema-checked.
Sanitizing an argument value (a shell-injection string, an unbounded `n`) is
the tool function's own job. `iridium.runtime.sandbox` is a different,
unrelated boundary — for running untrusted model-*generated code*, not for
tool calls.

**Target tool families** (all four weighted, per the current direction):
calculator / code execution (exactly graded, sandboxed — this track's own
synthetic tools); search / retrieval (a tool returns documents, the model
answers from them — graded by whether the final answer cites/uses the
returned passage; not yet built as a synthetic family, only the mechanism);
creative / 3D / media (via `iridium.agency`'s existing scene-editing,
Blender-emission and media-agent layers — this track's `ToolRegistry`/
`run_tool_loop` can wrap those tools directly, since a `Tool.fn` is any
Python callable); general MCP tools (via `Tool.from_mcp`, above, and
`constrained.py` decoding against `inputSchema` exactly like any other
schema).

## 4. Constrained decoding (`iridium.runtime.constrained`)

A byte-level incremental parser that answers, for a supported JSON-Schema
subset: is this string a valid prefix of *some* value the schema allows?

**Supported subset:** `object` (closed-world `properties`/`required` — an
unknown key is always rejected, unconditionally), `string` (optional `enum`,
optional `maxLength`), `number`/`integer`, `boolean`, `null`, `array` (one
homogeneous `items` schema, optional `minItems`/`maxItems`), and this
engine's own `any` extension (accepts any value, dispatching on the next
byte). **Rejected outright, at construction:** `oneOf`/`anyOf`/`allOf`/`not`/
`if`/`then`/`else`, `$ref`, a list of types, `pattern`, `patternProperties`,
`multipleOf`, `uniqueItems`, `const`, a schema without exactly one `type`,
and `additionalProperties: true`. This is fail-closed: an unsupported schema
raises before generation starts, never silently under-enforced mid-decode.

**Practical caps, documented rather than silently assumed away:** JSON's
grammar puts no bound on whitespace runs or number-literal length, and every
one more whitespace/digit byte is a genuinely valid extension of a valid
prefix — an untrained model's greedy decode found and exploited exactly
this before these caps existed (`_MAX_WS_RUN = 4`, `_MAX_NUMBER_LEN = 32`,
`_MAX_DEPTH = 24` for nesting). A schema whose own fields are unbounded
(a plain `string` with no `enum`/`maxLength`) should set `maxLength` (or
`maxItems`, for arrays) if a hard termination guarantee matters for it too.

**The tested claim:** `tests/unit/test_constrained.py::test_untrained_tiny_model_always_yields_a_valid_tool_call`
constructs a fresh `Iridium1(get_config("tiny"))` from 8 different random
seeds and decodes a tool call under the constraint against a 4-tool
registry (numbers, an enum, a bounded string, a bounded array). Every one
parses as JSON and validates against its tool's schema. Verbatim result:

```
$ OMP_NUM_THREADS=1 python -m pytest tests/unit/test_constrained.py -q
..................                                                       [100%]
```

**Two decode paths.** `generate_constrained_json` — for this project's
byte-level text codec — checks all 256 next bytes every step (cheap; exact).
`bpe_candidate_mask` — for a subword vocabulary — filters a caller-ranked
top-`k` of BPE token ids instead of scanning the whole vocabulary, with the
documented limitation that a valid continuation outside the top-`k` is
invisible to it. It is tested directly against `BytePairTokenizer`, not
through a live decode loop: this project's text head
(`iridium.runtime.generate`) is byte-level end to end today, so there is no
BPE-native next-token distribution yet to rank in the first place.

## 5. Data (`iridium.data.tool_corpus`)

| source | licence | verified | commercial_ok | default | why |
|---|---|---|---|---|---|
| `synthetic` (built in) | none needed | — | yes | **on** | calculator / unit conversion / key-value lookup — pure local functions, exactly gradeable |
| `glaiveai/glaive-function-calling-v2` | Apache-2.0 | yes (dataset card) | yes | **on** | 113k rows, converter implemented and tested against its real row format |
| `NousResearch/hermes-function-calling-v1` | Apache-2.0 | yes (dataset card) | yes | **on** | converter verified against a live row preview (ShareGPT + `<tool_call>`/`<tool_response>` tags), multi-call turns split into alternating call/result pairs |
| `Salesforce/xlam-function-calling-60k` | CC-BY-4.0 | yes (Hub metadata) | yes | off | Hub-gated (needs per-account access acceptance) — a loading obstacle, not a licence problem; catalogued so the licence is on record |
| `gorilla-llm/Berkeley-Function-Calling-Leaderboard` (BFCL) | Apache-2.0 | yes | yes | off | no converter implemented this pass; row schema not verified against a live preview |

No non-commercial or restricted-terms source is currently catalogued —
`ToolSourceSpec.commercial_ok` and `licence_verified` exist and are enforced
(`tool_corpus._check_enabled` raises on an unverified source) so a future
one can be added safely, disabled by default, rather than because one needed
disabling today.

Neither `glaive` nor `hermes`'s dataset card names the specific model that
generated its synthetic conversations — recorded as such rather than
assumed clean of any upstream-model terms.

`tool_items(n, mix, seed, split, tokenizer=None)` mirrors
`chat_corpus.chat_items`: `allocate_mixture`'s exact largest-remainder
budgeting, `text_corpus.in_split`'s deterministic content-hash train/test
split, `chat.fit_to_budget`'s whole-turn budget trimming.

## 6. Training

Mix `"tool_use"` items alongside `"chat"`/`"text_lm"`/the synthetic physics
families in `iridium.training.datasets.build_corpus`'s mixture dict — the
same pattern `"chat"` already uses there. `tool_corpus.tool_items` is the
function to call for a mixture key.

## 7. Grading

`tool_corpus.grade_synthetic_item(case, produced_call, produced_answer)`
returns three independent booleans plus their conjunction:
`tool_chosen`, `arguments_correct`, `uses_result`, `all_correct` — kept
separate deliberately, because a model that happens to state the same
number the tool would have returned is not evidence it can use tools.

Nothing here has been trained; no accuracy number is claimed.
