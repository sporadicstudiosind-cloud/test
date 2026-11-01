# World model: explicit splat memory + action-conditioned generation

`iridium/world/` gives Iridium-1 a persistent, revisitable notion of *place*.
It hybridises two designs currently in production elsewhere rather than
picking one:

* **Explicit 3D state** — World Labs' [Marble](https://www.worldlabs.ai/blog/marble-world-model)
  and [Atlas](https://www.worldlabs.ai/blog/atlas). Atlas was announced
  September 1, 2026 as an "omni world model" pretrained from scratch,
  natively operating on text, image, video, camera poses, depth maps and 3D
  Gaussian splats from one multimodal autoregressive-diffusion transformer;
  it generates a minute of 1440p camera-controlled video, reconstructs
  scenes, and outputs point clouds and splats that "fill remaining gaps" in
  a scene. It is entering early access with select partners (no public API
  or pricing yet). Marble is World Labs' existing product for building
  explorable 3D worlds as Gaussian-splat scenes; Atlas is stated to power
  future Marble versions. The property this design guarantees "by
  construction" is the one `world_state.py`'s docstring leads with: a wall
  you looked at a minute ago is still there because it is *stored*, not
  because a network remembered it.
* **Autoregressive, action-conditioned frame generation** — DeepMind's
  [Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/),
  released August 2025: an autoregressive latent-diffusion model that
  generates the next frame from the frame history and the user's action, at
  24 fps and 720p, holding "a few minutes" of visual consistency before it
  degrades. Genie 3 needs no explicit 3D representation and no labelled
  action data — a latent action model learns the action space directly from
  video — and this is exactly what lets it depict content no fixed scene
  representation anticipated (motion, lighting change, something entering
  frame). It is also exactly why its consistency is *emergent*: nothing
  stops the frame history from drifting, and revisiting a spot after
  wandering away is the standard way that drift is exposed.

The hybrid: keep a splat scene as persistent memory, and on every step
**render** the stored scene from the new camera, **condition** the frame
model on that render plus retrieved memory keyframes plus the camera's own
ray geometry, let the model **generate** the frame, then **fuse** what it
produced back into the scene. The render anchors generation where the scene
already has evidence; generation fills in where it does not. This is not a
novel synthesis claim beyond what it looks like: it is Atlas's
render-and-complete loop with Genie 3's frame model doing the completion,
built here without either external product's code or weights (no learned
frame model exists in this repository yet — see "what is built vs what
needs training" below).

## Module map

| Module | Owns |
|---|---|
| `camera.py` (existing, pre-dates this build) | Pinhole cameras, rays, Plücker embeddings, motion primitives (`move`, `orbit`, `interpolate`, `trajectory`). |
| `splats.py` (built concurrently by another agent; not covered by this doc's testing) | `GaussianScene`, the differentiable renderer `render`, and `fit`. |
| `world_state.py` | `Keyframe`, `WorldState`: persistent scene + keyframe bank, frustum-overlap `retrieve`, `coverage`, coverage-preserving eviction, `fuse`, `save`/`load`. |
| `rollout.py` | `ActionMap`, `FrameContext`, `Frame`, `RenderOnlyFrameModel`, `WorldRollout` (`step`/`play`/`fly`), `revisit_consistency`. |
| `tokens.py` | `camera_span`, `anchor_coordinates`, `scene_span` — cameras and scenes as spans the model consumes. |

`world_state.py` and `rollout.py` never import `splats` at module scope —
only lazily, inside the one or two functions that need a default
`GaussianScene`/`from_points`, and every other splat-shaped operation
(`renderer`, `from_points_fn`, `fit_fn`) is accepted as an injected callable.
This is what let this module's own tests run and pass without `splats.py`
existing yet, and it is deliberate beyond the immediate scheduling
constraint: a scene-memory module should not need a specific renderer
implementation to define what memory *is*.

## API reference

### `world_state.py`

```python
@dataclass
class Keyframe:
    camera: Camera
    rgb: torch.Tensor                     # [H, W, 3]
    depth: Optional[torch.Tensor]         # [H, W] or None
    step: int
    source: Literal["observed", "generated"] = "observed"

@dataclass
class WorldState:
    scene: object = None                  # a GaussianScene, or any duck-typed equivalent
    max_keyframes: int = 32
    keyframes: list[Keyframe] = field(default_factory=list)

    def retrieve(self, camera: Camera, k: int = 4, grid: int = 8,
                 default_depth: float = 5.0) -> list[Keyframe]: ...
    def coverage(self, camera: Camera, renderer: Optional[Callable] = None,
                 background=(0.0, 0.0, 0.0)) -> float: ...
    def add_keyframe(self, keyframe: Keyframe) -> None: ...
    def fuse(self, keyframe: Keyframe, renderer: Optional[Callable] = None,
             from_points_fn: Optional[Callable] = None, fit_fn: Optional[Callable] = None,
             coverage_threshold: float = 0.35, grid: int = 48, point_scale: float = 0.02,
             fit_steps: int = 0, fit_lr: float = 1e-2) -> None: ...
    def save(self, directory: str | Path) -> None: ...
    @classmethod
    def load(cls, directory: str | Path, max_keyframes: int = 32) -> "WorldState": ...
```

**Retrieval is by frustum overlap, not camera-centre distance.** Two cameras
at the same point facing opposite directions share nothing; two cameras far
apart can stare at the same wall. `retrieve` scores each stored keyframe by
unprojecting a coarse pixel grid (from the keyframe's own depth when it has
one, otherwise a depth guess cast from the query camera) and measuring what
fraction of the resulting points the other camera's frustum contains
(`Camera.sees`). `tests/unit/test_world_state.py::test_retrieve_prefers_co_oriented_camera_over_co_located_opposite_facing`
pins this down directly.

**Eviction keeps spatial coverage, not recency.** When `add_keyframe` pushes
the bank over `max_keyframes`, the keyframe with the *highest average
overlap against the rest of the bank* is dropped — the one whose content the
remaining keyframes already explain. FIFO was rejected because a rollout
that lingers in one room accumulates many near-duplicate views of it and
would age out the *one* keyframe of a newly-discovered room before any of
those duplicates, destroying unique coverage while keeping redundant recent
frames.

**`fuse` only adds splats where coverage is currently low.** Given a
renderer, it renders the existing scene from the keyframe's camera and masks
to pixels below `coverage_threshold`; only those are unprojected into new
splats via `from_points_fn` (default: lazy-imported
`splats.GaussianScene.from_points`). This is the concrete, minimal
implementation of "anchor to what is known, extend where it is not" —
re-splatting already-covered pixels every step would both waste splats and
let per-step model noise slowly drift geometry the render is supposed to be
anchoring against. `fit_fn` (shaped like `splats.fit`) is optional
least-effort refinement, off by default (`fit_steps=0`).

### `rollout.py`

```python
@dataclass
class ActionMap:
    move_speed: float = 0.5
    turn_speed_degrees: float = 15.0
    keymap: dict[str, tuple[float, float, float, float, float]] = ...  # WASD + aliases

    def resolve(self, action: str | dict | Sequence[float]) -> dict: ...
        # str: "w" / "forward 0.5" -> keymap lookup * (move_speed | turn_speed_degrees)
        # dict: literal move() kwargs, unscaled (the "already resolved" escape hatch)
        # sequence of 5 or 6 floats: [forward, right, down, yaw, pitch, (roll, dropped)]

@dataclass
class FrameContext:
    camera: Camera
    plucker: torch.Tensor                 # [H', W', 6]
    anchor_rgb: Optional[torch.Tensor]    # [H, W, 3] or None
    anchor_alpha: Optional[torch.Tensor]  # [H, W] or None
    anchor_depth: Optional[torch.Tensor]  # [H, W] or None
    memory: list[Keyframe]
    previous_frame: Optional[Frame]
    action: Any

@dataclass
class Frame:
    camera: Camera
    rgb: torch.Tensor
    depth: Optional[torch.Tensor]
    step: int
    context: FrameContext

class RenderOnlyFrameModel:
    def __call__(self, context: FrameContext) -> tuple[torch.Tensor, Optional[torch.Tensor]]: ...

@dataclass
class WorldRollout:
    world_state: WorldState
    frame_model: Callable[[FrameContext], torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]]
    camera: Camera
    renderer: Optional[Callable] = None
    action_map: ActionMap = field(default_factory=ActionMap)
    memory_k: int = 4
    plucker_stride: int = 8
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def step(self, action) -> Frame: ...
    def play(self, actions: Sequence) -> list[Frame]: ...
    def fly(self, cameras: Sequence[Camera]) -> list[Frame]: ...

def revisit_consistency(frames: Sequence[Frame], position_threshold: float = 0.2,
                        direction_threshold: float = 0.98) -> list[RevisitPair]: ...
```

**The `frame_model` contract** (the one integration point the owning agent
wires a trained Iridium-1 adapter into): a callable taking one
`FrameContext` and returning either an `rgb [H, W, 3]` tensor matching
`context.camera`'s resolution, or an `(rgb, depth)` tuple where `depth` is
`[H, W]` or `None`. Returning depth is what lets the step's output be folded
back into the splat scene (`WorldState.fuse`); an RGB-only model still runs
the loop, it just never grows persistent geometry. `RenderOnlyFrameModel`
satisfies this contract trivially — it returns the anchor render unchanged,
or a mid-grey frame when there is no scene yet — and is shipped both as the
system's zero-learned-capability lower bound and as the fixture that makes
the rest of the loop testable without a trained model.

**`revisit_consistency`** operationalises the Genie 3 "look away and look
back" test: for every pair of frames whose cameras are within
`position_threshold` and whose forward vectors' cosine similarity exceeds
`direction_threshold` (a *revisit*, stricter than `retrieve`'s general
overlap test — this wants near-identical pose, not just shared content), it
reports the mean absolute pixel difference between the two frames. On a
static scene driven by `RenderOnlyFrameModel`, this is measured at machine
precision (`< 1e-6` in `tests/unit/test_rollout.py`) because the anchor
render is a deterministic function of camera pose and an unchanged scene;
injecting per-step noise into the frame model reliably produces a positive
error on the same revisit pairs (also tested).

### `tokens.py`

```python
def camera_span(camera: Camera, patch: int = 8, modality: str = "field",
                supervised: bool = False, meta: Optional[dict] = None) -> Span: ...

def anchor_coordinates(camera: Camera, depth: torch.Tensor, patch: int = 8) -> torch.Tensor:
    # -> [n_patches, 3] world-space (x, y, z) per patch centre

def scene_span(scene, max_splats: Optional[int] = None, supervised: bool = False) -> Span:
    # -> "geometry" span, [n, 14]: means(3) + log_scales(3) + quats(4) + opacity_logit(1) + sh0(3)
```

`camera_span` produces per-patch Plücker rays with the same grid an
`image`/`video` span of the same view and patch size would have, so the two
can be interleaved or their payloads combined without resampling.

`anchor_coordinates` unprojects each patch's centre pixel (matching the
convention `Camera.rays`/`Camera.plucker` already use) at its depth, giving
every visual token a world-space anchor — the concrete form of Atlas's
"every token is anchored to a position in 3D space." These are **not** the
same numbers as `Batch.media_coordinates` (pixel-frame `(t, y, x)`,
image-local); conflating world coordinates with pixel coordinates under one
rotary table is the same failure mode `codecs/spatial.py`'s own docstring
warns about for normalized-vs-pixel coordinates, one level up. See
"config/integration needed" below for the recommended wiring.

`scene_span` orders splats by Morton (Z-order) code on quantised `means`
before emitting them, for the same reason `codecs/geometry3d.canonicalize`
sorts mesh vertices: a splat scene is a *set* (`GaussianScene.concat` and any
renderer treat order as irrelevant), so without a canonical order N splats
have N! equally valid token sequences for one scene and a sequence model
would spend capacity learning that the permutation is meaningless instead of
never seeing it. When `max_splats` truncates, splats are *selected* by
highest `opacity_logits` first (a value-based, therefore
permutation-invariant, criterion) before Morton-ordering the survivors —
verified directly in `tests/unit/test_world_tokens.py` by permuting the
input scene and checking the emitted span is bit-identical.

## Config / integration this module needs from the owner of `config.py`, `codecs/*`, `model/*`

Nothing in `spans.py` was edited, per the build constraints; the two spans
above use the closest existing modality as a stand-in and need one of the
following before they can round-trip through `collate()` without a width
mismatch:

1. **`scene_span`** uses `"geometry"` — the correct modality by name and
   description (`spans.py`: "point / splat features") but the wrong width:
   `CodecConfig.point_features = 10` today, while one splat needs 14 floats
   (`means[3] + log_scales[3] + quats[4] + opacity[1] + sh0[3]`). Either
   raise `point_features` to 14, or add a separate field (e.g.
   `splat_features: int = 14`) if some other caller depends on 10-wide
   "geometry" spans staying 10-wide.
2. **`camera_span`** defaults to `"field"` purely as a structural
   placeholder (a per-patch multi-channel signal on a grid, like a physical
   field) — `CodecConfig.field_channels = 4` is unrelated content
   (physical-field channels) at the wrong width (Plücker rays are 6-wide).
   The clean fix is a new `"camera"` modality, width 6/patch, added to
   `spans.MODALITIES`/`CONTINUOUS`; `camera_span` takes `modality` as a
   parameter for exactly this reason, so adopting it is a one-line
   call-site change (`camera_span(cam, modality="camera")`), not a rewrite.
3. **`anchor_coordinates`** output (`[n, 3]` world `(x, y, z)`) needs a home
   in `codecs/spatial.py`'s M-RoPE distinct from the existing pixel-frame
   `(t, y, x)` axes `Batch.media_coordinates` carries. Recommended: a second
   `AxialRotaryEmbedding(axes=("x", "y", "z"))` applied only to tokens
   carrying anchor coordinates, combined into the token the way
   `continuous_conditioning="add"` already combines the flow head's
   timestep embedding (project, then add) — not merged into the existing
   table, since pixels and world length are different units under one
   `theta` schedule.

## What is built vs. what needs training

**Built and tested** (33 unit tests, `tests/unit/test_world_state.py`,
`test_rollout.py`, `test_world_tokens.py`, all passing against a fake
splat scene + fake renderer, no dependency on `splats.py` existing):
persistent keyframe memory with overlap-based retrieval and
coverage-preserving eviction; the render → retrieve → generate → fuse loop
with a WASD/continuous/dict action map; a measurable revisit-consistency
metric; camera-ray and canonically-ordered splat tokenization; save/load
round-trip.

**Not built here, and not claimed to be:**

* **A learned `frame_model`.** `RenderOnlyFrameModel` is a correct lower
  bound and test fixture, not a generative model — it never produces
  content the stored scene does not already have. Wiring an Iridium-1
  adapter that satisfies the `frame_model` contract (see above) is
  integration work for whoever owns `iridium/model/*`.
* **Camera-conditioned video training data.** Nothing here trains or
  fine-tunes on paired (camera, action, next-frame) sequences; that is what
  a trained `frame_model` would need, in the CameraCtrl/CAT3D/Genie-3 style
  already cited in `camera.py`'s own docstring.
* **Splat supervision / real photometric fitting.** `fuse`'s point-cloud
  splatting is unproject-and-place, not gradient-fit; `fit_fn` is accepted
  and wired through but is a no-op unless `fit_steps > 0` and a real
  `splats.fit` is supplied. Whether unrefined point splats are good enough
  anchors for a trained frame model, versus needing per-fuse photometric
  refinement, is unmeasured.
* **The `spans.py`/`config.py`/`spatial.py` integration** listed above —
  reported to their owner, not made here, per the file-ownership rule for
  this build.
