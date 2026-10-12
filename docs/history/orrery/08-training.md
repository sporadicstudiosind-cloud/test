# 08 — Training

> Seven stages, one loss family, and a first stage that is more likely to sink the project than
> anything downstream of it.

## 8.1 Loss composition

```
L_total =  L_symbolic                  # cross-entropy on text, code, action tokens
        + λ_flow  · L_flow             # OT conditional flow matching on continuous latents
        + λ_recon · L_recon            # codec reconstruction (Stage 0 dominant)
        + λ_phys  · L_physics          # PDE residuals on generated/corrected fields
        + λ_cons  · L_conserve         # global mass / momentum / energy drift
        + λ_rend  · L_render           # rendered field vs. ground-truth image
        + λ_pond  · L_ponder           # KL(halting ‖ geometric prior)
        + λ_geo   · L_geometric        # weak: grade-2 channels ↔ vorticity, physics tokens only
```

Two notes on terms that are easy to get wrong:

**`L_physics` and `L_conserve` are different objectives.** Residual penalties are local and
differentiable and pull the solution toward satisfying the PDE pointwise. Global conservation is an
integral property, and a field can have small pointwise residuals everywhere while total mass drifts
steadily. Both are needed; only one of them is what a user notices.

**`L_geometric` is deliberately weak and masked.** Forcing internal grade-2 channels to equal the
true vorticity field is an appealing idea — it makes the representation interpretable — but
supervising intermediate representations to equal specific physical quantities over-constrains the
network and is a known way to lose performance. It is also undefined for text: a sentence has no
vorticity. Applied at low weight, to physics tokens only, as an auxiliary that can be annealed to
zero.

## 8.2 The curriculum

| Stage | What it does | Why here |
|---|---|---|
| **0 · Codec grounding** | Train perceptual codecs and the coordinate frame, reconstruction objectives only. Conservation-penalized field reconstruction. | Everything downstream reads through these. Also the largest risk — see §8.3. |
| **1 · Dimensionless physics** | Pretrain on solver-generated fields in Π-normalized form. Curated sets plus farm output ([`07`](07-data.md)). | Establish the physics representation *before* it competes with web-scale text for capacity. |
| **2 · Omnimodal joint pretraining** | Early-fused training across the full corpus. | The bulk stage; where cross-modal transfer is either won or lost. |
| **3 · Solver-in-the-loop correction** | Differentiate through the Tier A integrator; the trunk learns to predict corrections and closure parameters, not states. | This is where physical understanding actually enters the weights ([`03`](03-physics.md)). |
| **4 · Rendering alignment** | Train the differentiable renderer on synthesized `(field ↔ image)` pairs; align real footage against solved fields where paired data exists. | Makes render-the-field ([`04`](04-generation.md)) produce images rather than diagrams. |
| **5 · Agentic RL** | Sandboxed software environments; reward on verified task completion, penalize irreversible-action errors. | Weakest capability ([`05`](05-agency.md)), needs the most environment interaction. |
| **6 · Duplex and depth-budget RL** | Full-duplex streaming behavior; learn to request depth budgets that correlate with actual difficulty. | Cannot be learned offline — depth allocation is only meaningful against a live scheduler. |

Stages overlap rather than gate cleanly; earlier objectives are replayed at low weight to prevent
forgetting. The ordering constraints that genuinely matter are: 0 before everything, 1 before 2 (so
physics is not a late-arriving minority modality), 3 after 2 (the correction task needs a competent
trunk), 4 after 3 (you cannot render fields you cannot solve).

## 8.3 Stage 0 is the schedule risk

Constraint C1 forbids frozen third-party codecs, which is the standard stabilizer. Joint codec
training invites, in rough order of likelihood: posterior collapse in the continuous branch, codebook
death in the discrete branch, a reconstruction-versus-likelihood tug-of-war that quietly degrades
both, and — specific to this project — a field codec that achieves excellent PSNR while losing mass.

Mitigations, none of them clever:

- **Long reconstruction-only Stage 0** before any generative objective is switched on.
- **Conservation as a codec gate.** A field codec that fails to preserve total mass and momentum
  within tolerance is disqualified regardless of its reconstruction metrics. A codec that loses mass
  makes every downstream conservation claim unrecoverable.
- **Frozen-codec ablations run in parallel** so the cost of C1 is *measured*. If the frozen-codec
  control is materially better and stays better, that is evidence about C1 the operator should have.
- **Spectral reconstruction metrics, not just L2.** L2 is dominated by large scales; a codec can look
  excellent while erasing exactly the high-wavenumber content [`03`](03-physics.md) warns about.

## 8.4 Multi-physics co-training and the tripwire

Stages 1–3 are where the negative-transfer risk from [`00`](00-premise.md) materializes. The
instrumentation and kill thresholds in [`02`](02-trunk.md) §2.4 run continuously through these
stages, not as a post-hoc analysis.

The escalation ladder if the tripwire fires — gradient surgery, uncertainty-weighted loss balancing,
longer-period curriculum interleaving, widening the core, and finally reporting that C2 is not
survivable — is specified there. What matters at training time is that someone is *watching* the
gradient-conflict metrics from the first multi-physics step, rather than diagnosing "the model is
oddly bad at porous media" six months later.

## 8.5 Stability

At the scale C9 permits, the failure modes are mundane and expensive:

- **Loss spikes and divergence.** Checkpoint frequently enough that a spike costs hours, not weeks.
  Skip-and-resume on anomalous batches, with the skipped batches logged rather than discarded.
- **Modality imbalance.** The dominant modality absorbs capacity. Monitor per-modality loss against
  single-modality controls and rebalance mixture weights on the evidence — this is the same
  measurement as the gradient tripwire, applied across modalities rather than physics domains.
- **The halting collapse.** Watch the depth distribution directly. Collapse to always-shallow or
  always-deep means `λ_pond` or the geometric prior's expected depth is mistuned, and it silently
  removes the mechanism C3 depends on.
- **Renderer/solver co-adaptation.** In Stage 4 the renderer can learn to compensate for solver
  error, producing images that look right from fields that are wrong. Freeze the solver during
  renderer training and validate the renderer against *ground-truth* fields, not corrected ones.
- **Equivariance ablation.** The control model from [`01`](01-representation.md) §1.2 trains
  alongside. If the constrained-equivariant trunk is not measurably better, that is a finding worth
  having early, while the architecture can still change.
