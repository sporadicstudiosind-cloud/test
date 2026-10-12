# Archived: the ORRERY specification

These fourteen files are the superseded ORRERY specification, preserved **byte-identical**
as historical reference. They are not normative and must not be cited as current design.

The successor is [`../../architecture.md`](../../architecture.md). Every normative change,
with its reason and its test, is in [`../../decisions.md`](../../decisions.md).

Known errors in this material, corrected in the successor:

- One universal coordinate frame (D01), one spectral axis (D02), an over-broad Clifford
  equivariance claim (D03), shear rate as a bivector (D04).
- Conflating non-numeric with dimensionless (D05); treating matched dimensionless groups as
  dynamic similarity (D06); treating continuous latents as lossless (D07).
- Combining ACT averaging with PonderNet stopping (D08); shared mutable attention sinks (D09).
- The claim that a conservative integrator plus *any* learned correction cannot invent mass
  (D10) — false for per-cell corrections, which is what the text permitted.
- Integrator consistency bounding learned error (D11); Π-distance alone governing escalation
  (D12); an uncounted density model (D13).
- An unsupported claim that doubling inflow approaches Fr = 1 (D14); backflow always
  unphysical (D15); constant total mass on an open domain (D16).
- Rendering treated as establishing physical accuracy (D17); screen motion equated with fluid
  velocity (D18); a promised 4–10 flow steps (D19); a shared clock treated as synchronization
  (D20).
- A blanket prohibition on learning physics from video (D21).
- Universal claims about self-improvement (D22); an external benchmark score used as a target
  (D23); mass conservation measured from arbitrary RGB (D24).

`12-blueprint-reconciliation.md` additionally contains three errors of its own, recorded as
D25–D27: a 3-D rfft corner-block count (four, not eight), a KV total that should read 24 TiB
rather than ~25 TiB, and the Fr = 1 overclaim. The second is the sharpest lesson in this
repository: it is an arithmetic slip made inside a document criticizing arithmetic slips, and
it is why numeric reconciliation is automated in the successor rather than left to review.
