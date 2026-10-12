"""Spatial world modelling: cameras, persistent 3D scenes, and camera-driven rollout.

The package exists to give the model the one thing a flat token sequence cannot
supply on its own: a *place* for everything it has seen. Two current systems
bracket the design space, and this package deliberately takes from both:

* **Explicit 3D state** (World Labs' Marble and Atlas): the world is a set of 3D
  Gaussian splats. Consistency is guaranteed by construction -- a wall you
  looked at a minute ago is still there because it is stored, not because the
  model remembered it -- and the result exports to standard tools.
* **Autoregressive frame generation conditioned on actions** (DeepMind's Genie
  3): each new frame is generated from the history of frames and the user's
  latest input. Consistency is emergent and degrades with horizon, but the
  model can depict things no fixed representation anticipated -- motion,
  lighting change, objects entering.

The hybrid here keeps a splat scene as the persistent memory *and* conditions
frame generation on it: render the stored scene from the new camera, hand that
render and the camera's rays to the model as context, let the model produce the
frame, and fold what it produced back into the scene. The render anchors the
generation where the scene is known; the generation fills in where it is not.

Modules:

* :mod:`camera` -- pinhole cameras, rays, Plücker embeddings, trajectories,
  and camera motion primitives.
* :mod:`splats` -- 3D Gaussian scenes, a differentiable pure-torch renderer,
  and standard 3DGS ``.ply`` import/export.
* :mod:`world_state` -- persistent scene memory with keyframes and
  frustum-overlap retrieval for revisits.
* :mod:`rollout` -- action-to-camera mapping and the autoregressive
  render-condition-generate-fuse loop.
* :mod:`tokens` -- turning cameras and scenes into spans the model consumes.
"""
