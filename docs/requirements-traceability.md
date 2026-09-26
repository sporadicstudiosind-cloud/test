# Requirements Traceability

The original request, in the requester's own words, mapped to the requirement it became,
where it is specified, and what checks it. Nothing in the request was dropped; two items
were narrowed and the narrowing is stated.

| # | Original wording | Requirement | Specified in | Check |
|---|---|---|---|---|
| 1 | "input simulation data, text, images, video, audio, physics data, etc" | IR 03 | [arch §3](architecture.md#3-representation) | `test_events.py` (typed payloads, per-channel units) |
| 2 | "output any type of output like the inputs … theoretically all together" | IR 03, IR 05 | [arch §5](architecture.md#5-native-generation-and-joint-outputs) | `first_slice.py` trains symbolic + field + image heads on one trunk |
| 3 | "deep complex math" | IR 04 | [arch §10](architecture.md#10-mathematics-and-exact-computation) | `test_inlet_arithmetic.py` (unit algebra); proof checker is backlog M9 |
| 4 | "full access to a simulation" | IR 06 | [arch §6.2](architecture.md#62-three-execution-modes) | three execution modes; reference mode is M5 |
| 5 | "calculate that fluid flow if I double the input water … accurate physics based video and short report" | IR 04, IR 05 | [arch §11.1](architecture.md#111-waterfall-input-to-deliverables) | `test_inlet_arithmetic.py`; end-to-end package is M9 |
| 6 | "never need to call other models, at most APIs and code" | IR 01, IR 06 | [arch §1.2](architecture.md#12-what-one-model-means) | `test_no_expert_router_exists`; Invariant 1 |
| 7 | "full native physics understanding" | IR 04 | [arch §6](architecture.md#6-physics-as-prediction-and-computation) | `test_conservation.py`; measured natively in [first-slice.md](first-slice.md) |
| 8 | "video gen, image gen, text, audio" | IR 05 | [arch §5.2](architecture.md#52-general-audiovisual-output) | image head trained in the slice; video/audio are M4 |
| 9 | "output motion vectors and things" | IR 05 | [arch §5.5](architecture.md#55-motion-and-geometry-outputs) | `test_frames.py` (world velocity ≠ screen motion) |
| 10 | "agentic control (exactly like a human) to edit and do things in softwares like blender" | IR 10 | [arch §9](architecture.md#9-agentic-software-control) | `action.v1.json` schema; bridge is M8 |
| 11 | "recursive self improvement but like it doesn't need to" | IR 11 | [arch §15](architecture.md#15-optional-self-improvement) | Invariant 10 (service works with it disabled) |
| 12 | "one massive instance … instead of spinning up a new model each time" | IR 07 | [arch §8.1](architecture.md#81-resident-deployment) | resident checkpoint, continuous batching |
| 13 | "always running, always streaming … computed in parallel" | IR 08 | [arch §8.5](architecture.md#85-streaming-and-clocks) | `test_causality_rejects_events_that_had_not_arrived` |
| 14 | "the model should decide how much focus … should be diverted toward each chat" | IR 09 | [arch §4.4](architecture.md#44-halting-with-precise-semantics), [§8.3](architecture.md#83-compute-requests) | `test_halting.py`; broker arbitration is M7 |
| 15 | "Not MoE" | IR 02 | [arch §1.3](architecture.md#13-what-dense-means) | `test_no_expert_router_exists` |
| 16 | "ignoring compute costs" | IR 12 | [arch §16](architecture.md#16-scale-and-resource-accounting) | `test_inventory.py` — cost ignored, *finite memory* still accounted |

## The two narrowings, stated plainly

**Item 13 — "any request routed to it can be computed in parallel."** Parallel service across
concurrent workspaces is kept and is the design. What is rejected is attention flowing
*between* concurrent chats: if one stream's query can attend to another's KV cache, that is a
cross-tenant read. Shared weights, shared capacity, shared scheduler and continuity across
time are all retained. See [arch §8.2](architecture.md#82-isolation-and-the-one-requirement-this-specification-rejects)
and D09.

**Item 16 — "ignoring compute costs."** Monetary cost is ignored when defining the endpoint,
exactly as asked. Finite memory, bandwidth, latency and numerical precision are not ignored,
because they do not go away when the budget does: [arch §16.2](architecture.md#162-cache-arithmetic)
shows a fully retained million-token stream at the flagship scale needs 1.375 TiB of KV at
depth 1 and 5.125 TiB at depth 4. That is a design constraint, not a budget line.

## Where the ambition currently outruns the evidence

Honest placement of the request against what exists. Detail in
[capability-register.md](capability-register.md).

- **Trained and measured:** one shared trunk over instruction + field + image, a conservative
  flux head, a native flow-matching image head, at ~0.8 M parameters.
- **Implemented and tested, not trained:** event/unit/frame contracts, cache parity, mask
  isolation, stopping-time semantics, parameter and cache inventory, conservation mechanics.
- **Specified only:** video, audio, speech, Blender agency, coupled multiphysics, inverse
  problems from observation, the compute broker, self-improvement.
- **Not established by anything here:** that this scales to the flagship, that a single dense
  model reaches "understands everything", or that agentic control approaches a human operator.
  "Understands everything" has no finite acceptance test and is decomposed into the expanding,
  measured competence in [backlog.md](backlog.md) instead.
