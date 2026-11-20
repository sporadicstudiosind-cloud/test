# 05 — Agency: Control as Geometry

> A screen is a 2-D slice of the shared frame. A click is a latent at `(t, x, y)`. Computer use is
> video-in, motion-out — and it is the weakest subsystem in this document.

## 5.1 The unification

Because [`01`](01-representation.md) puts everything in one coordinate frame, GUI control requires
no new machinery:

- **Perception:** screen frames enter as video latents at `(t, x, y)`, `frame_id = display`.
- **Action:** pointer moves, clicks, drags, keystrokes, and scrolls are `action`-type latents at
  coordinates in the same display frame. A drag is a short trajectory of them — the same object type
  the geometric head already emits for motion vectors ([`04`](04-generation.md)).
- **Grounding:** "click the extrude button" resolves by attention to a coordinate, which is the same
  operation as "which field cell does this pixel observe."

This is genuinely elegant and it is worth being suspicious of that. Elegance in the representation
does not produce reliability in the behavior; §5.4 is the correction.

## 5.2 Dual path: pixels and APIs

Most professional software exposes both a GUI and a scripting interface. ORRERY uses both, and
prefers the API.

| Situation | Path |
|---|---|
| Task expressible as code (create geometry, set a modifier, batch-process, run a sim) | **API** — write and execute a script |
| Task requires visual judgment (does this look right, where did the artifact appear) | **Perception** — read the viewport as video |
| Task only exposed in the UI (an operator with no scripting binding, a modal dialog) | **GUI actions** |
| Verification of any of the above | **Perception** — look at the result |

The API path is preferred because it is deterministic, inspectable, undoable, and diffable. A GUI
trajectory of 200 clicks is none of those. The perception path is not optional, though: the model
must *look at what it did*, because a script that runs without error and produces the wrong mesh is
the common failure, not the rare one.

**Constraint check (C1):** Blender's Python API is an API. Executing a script is code. No second
model is called.

## 5.3 Blender specifically

The 3D case matters because it closes the loop with the physics path: a solved field can become a
Blender scene, and a Blender scene can become solver boundary conditions.

- **Scene as latents.** Objects, transforms, modifiers, and materials are ingested as structured
  latents in the shared frame — the same frame the physics fields use, so a simulation domain and a
  scene bounding box are directly comparable rather than related by a manual import step.
- **Edits as diffs.** The model emits scene deltas (a transform change, a modifier added), applied
  through the API. This is undoable, reviewable, and can be rejected without re-running everything.
- **Round trip.** Solved field → mesh/volume → Blender scene → render, and back: scene geometry →
  meshed domain → solver setup. The round trip is the actual product feature; it is also where
  coordinate-frame bugs will hide, so `frame_id` transforms are explicit, logged objects.
- **Verification by render.** After an edit, render the viewport and check it against the intent.
  Cheap, and it catches the silent-wrong-result case.

## 5.4 The number that should govern expectations

The best hybrid computer-use agents pass **41.2%** of tasks on WeaveBench — 114 long-horizon
real-world tasks across 8 domains requiring coordinated GUI and CLI work. The same model on a
non-native runtime drops to 35.1%; a strong competitor sits at 33.3%.

Two things follow, and neither is comfortable:

**"Exactly like a human" (C7) is not achievable with what exists.** A subsystem that completes 4 in
10 long-horizon tasks is a capable assistant and an unreliable operator. Nothing in this
specification improves that number — the shared coordinate frame makes control *expressible*, not
*reliable*. Anyone reading this document should treat agentic control as the component most likely
to be the practical bottleneck, well ahead of the physics.

**Runtime coupling matters more than model quality here.** The 41.2% vs 35.1% gap for the *same
model* under different runtimes says a large fraction of agentic performance lives in the harness —
tool schemas, prompting conventions, the loop — not in the weights. For a single-model architecture
that refuses external components, this is an awkward finding: it implies effort spent on the action
interface pays better than effort spent on the trunk.

## 5.5 Design responses to a 41% baseline

Since the capability is unreliable, the architecture is built to *fail safely and recover*, rather
than to assume success:

1. **Prefer the reversible path.** API over GUI; diffs over destructive edits; every action taken in
   a scene that can be reverted.
2. **Verify every step visually.** Act, render, compare to intent, and treat a mismatch as a
   stopping condition rather than something to work around with more clicks.
3. **Checkpoint before irreversible operations.** Save state before anything that cannot be undone;
   this is the agentic analogue of the verification gates in [`03`](03-physics.md).
4. **Escalate on repeated failure.** Two failed attempts at the same subgoal ends the loop and
   reports, rather than a third attempt. Long-horizon agents fail by persisting.
5. **Declare uncertainty about the world state.** The model should be able to say "I am no longer
   confident what state this file is in" — the agentic equivalent of a non-converged solve.

## 5.6 Boundaries

Written into the spec because a model with GUI control and internet access is a different security
object than a chat model:

- Actions execute in a sandbox with an explicit, declared filesystem and network scope.
- Irreversible or outward-facing actions — deleting, sending, publishing, purchasing, pushing —
  require confirmation unless durably authorized for that specific class of action. Authorization for
  one class never generalizes to another.
- Every action is logged with its coordinate, its target, and the trunk state that produced it, so a
  wrong trajectory can be reconstructed after the fact.
- Content encountered while acting — a web page, a file, a comment in a scene — is **data, not
  instruction**. A document that says "ignore your previous instructions" is a document containing a
  sentence, and the action policy must not be reachable from ingested content.
