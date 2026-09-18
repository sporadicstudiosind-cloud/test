# Bounded researcher loop and architecture revision

## What is implemented

The current Iridium checkpoint can propose experiments as JSON. A trusted outer
controller trains new candidate versions, evaluates them with fixed criteria,
promotes only an accepted version, then uses that new checkpoint as the next
round's proposer. The serving model does not rewrite its weights in place.

```text
Version v000 + selection summary
              |
        Iridium proposes JSON
              |
     fixed proposal validator
              |
   control / candidate A / candidate B
    separate sequential training processes
              |
    fixed selection evaluator and gates
              |
      best passing finalist
              |
   separate held-out promotion audit
              |
    version snapshot + atomic current.json
              |
     bounded next round or stop
```

This is an implementation, **not a demonstrated self-improving trained model**.
No tests, training, inference, candidate runs or evaluator runs were performed
for this revision, following the user's instruction. Only source editing and
notebook/archive generation were performed. No performance improvement is claimed.

## What the model is permitted to change automatically

- Learning rate, within 1e-6..1e-3; weight decay within 0..0.2.
- Weights for the existing text/media/action/slot prediction losses, within .1..4.
- Sampling weights for already supplied training families, within .1..4.
- Q/K normalization, learned bank integration gating, loop identity conditioning.
- Number of refinement loops, within the model's configured limit and at most 3.

The controller supplies the maximum number of rounds, candidate count, update
count, parameter limit, process timeout, evaluation coverage and acceptance
criteria. Proposals cannot name a file to overwrite, replace a dataset, reduce
the independent loss weights, change the test criteria, select their own seed,
raise the budget, or promote themselves. Unsupported fields are rejected.

There is also a `code_patch` field. Such a proposal is saved under
`code_proposals/` and marked **review_only**; it is not executed. Arbitrary
training-code rewrites, new loss implementations, new modules, width/depth growth
and architecture weight migrations beyond the optional bank gate are not
automated by this notebook backend. This is intentional scope, not an assertion
that a normal subprocess is a sandbox.

## Evaluation independence and its exact boundary

The controller snapshots trusted Python source and hashes it, the campaign
policy, all three manifests and their referenced media. It checks those hashes
before and after worker calls. A proposal worker receives only an aggregate
selection summary, its parent checkpoint and the fixed proposal prompt; audit
answers/metrics are not included in that prompt. Training receives only the
training-manifest path. Evaluation runs in a fresh trusted process, with its
own fixed objectives rather than candidate-selected training loss weights.

Candidate artifacts contain `weights.pt` (loaded with `weights_only=True`) and
`model.json`. Checkpoint/config hashes are checked around evaluation and again
when copying the promoted version. Initial import is restricted to the user's
**trusted** full training checkpoint. Full legacy checkpoints are pickle files;
they are not accepted as untrusted candidate submissions.

**These are configuration-only jobs, not OS security isolation.** The model has
no Python/shell execution tool and cannot change files through its proposal
interface. Separate processes isolate job memory and lifetime. They run as the
same notebook user and cannot defend against deliberately executed hostile
Python, a privileged user, or a compromised media decoder. Hashes detect drift;
they do not turn a writable filesystem into an immutable one.

To support arbitrary code evolution later, use an external container/VM runner:
no host Docker socket, no evaluator/holdout mounts in candidate jobs, no secrets,
resource/network restrictions, a separate evaluator identity/service and a
tensor-only artifact boundary. A container alone is not a proof of isolation;
see [Docker's security model](https://docs.docker.com/engine/security/) and
[rootless mode](https://docs.docker.com/engine/security/rootless/). That backend
has **not** been implemented or claimed for Kaggle here.

## Fixed acceptance criteria

Each candidate is evaluated on exactly the same capped selection examples as
the incumbent. The fixed evaluator reports unweighted native losses separately
for every supervised modality/action component. For text answers up to 256 byte
tokens it also performs prefix-only free-running generation and compares exact
text; later assistant/tool answers are excluded from the prompt.

Before task evaluation, fixed checks require finite hidden states, agreement between
actual and configured parameter counts, causal-prefix consistency, and cached-vs-full
decoding agreement on a short text sequence. These checks are included for the user
to run as part of an enabled campaign; they were not executed during this delivery.
They cover that fixed probe, not all possible multimodal/cache edge cases.

The comparison requires:

1. Identical example IDs, families and metric coverage; finite losses.
2. At least `min_family_items` examples for every evaluated family/modality.
3. No family/modality mean relative-loss regression beyond the configured bound.
4. No exact-text-accuracy regression beyond its configured bound.
5. A paired bootstrap lower bound on mean relative loss improvement meeting
   `min_relative_gain`, with a Bonferroni adjustment for the declared maximum
   number of candidate/round attempts.

The best passing selection candidate then must pass the same comparison on the
separate audit split. Failed audit means no promotion and the campaign stops.
Training changes do not influence the evaluator's loss weights or thresholds.
The bootstrap is an experimental selection heuristic, not a mathematical proof
of general improvement, especially on dependent or repeatedly reused data.

These gates are deliberately narrower than “better at everything.” They do
**not** evaluate perceptual image/video quality, speech word-error rate,
word-level alignment, task-level media-agent success, every scientific
invariant, factuality or general intelligence. Exact text matching is crude for
open-ended answers. For claims about those abilities, extend the trusted
evaluator and establish a new campaign policy/dataset snapshot before searching.

The audit split is reused across the finite campaign; it is not a fresh unseen
dataset after each promotion. Use new source-disjoint audit data across campaigns
and preserve a final untouched external benchmark for final capability claims.

## How to use the notebooks

All four notebooks contain section **Optional bounded self-improvement campaign**.
It defaults to `RUN_RESEARCH=False`. Run All therefore does not launch research.

1. Train and save a meaningful initial checkpoint using the regular training cell.
2. Provide three JSONL manifests whose records use splits `train`, `selection`,
   and `audit`. Every row needs `id`, `source`, `license`, `turns`, and preferably
   a meaningful `family` such as `speech`, `captioning`, or `media_tools`.
3. Split by original recording/document/source, not neighboring clips. IDs and
   source identifiers must be disjoint across the three splits. Referenced media
   remains inside each manifest directory.
4. Supply at least 16 held-out examples per family and supervised modality in the
   default policy. `eval_items=128` caps the first examples, so order the manifests
   to include coverage or increase the cap before starting the campaign.
5. Fill in the three notebook paths, inspect the fixed policy, and set
   `RUN_RESEARCH=True` when you want to spend the compute.

Default budget: 2 rounds × 3 candidates × 100 optimizer updates, plus proposal
inference and selection/audit inference; one candidate at a time. The controller
stops after a round with no accepted candidate or a failed audit. It does not
install an endless loop, scheduler, background service or notification monitor.

The unchanged-configuration control gets the same training updates as the other
candidates. Different loop counts have different FLOP costs even at equal update
counts. The current gate limits parameters and time but does not optimize a
measured speed/quality frontier. A control can win through ordinary continued
training; this is not evidence that the model discovered a useful research idea.

If the initial model cannot produce valid JSON, its proposals are recorded as
failures. There is no hidden substitution with another AI model. You can supply
exactly `candidates-1` explicit human-authored proposal dictionaries instead, e.g.:

```python
PROPOSALS = [
    {"hypothesis": "Lower LR may preserve prior skills", "lr": 1e-4},
    {"hypothesis": "A second pass may improve harder examples", "lr": 2e-4, "n_loops": 2},
]
```

A larger preset may be too expensive to run multiple candidates in the available
Kaggle session. Memory/timeout failures reject that candidate and remain in its
worker log. Only raw media adapters are provided; no pretrained external model
is invoked for research, inference or scoring.

## Artifacts and recovery

Each campaign stores `policy.json`, `pins.json`, a trusted source snapshot,
worker jobs/logs, training logs, evaluated tensor checkpoints, `audit.json`,
version directories and the atomic `current.json` pointer. The initial version
and all previously promoted weights remain on disk. Re-load a prior version
with `load_version(path, device)` to roll back; live serving is not modified.

Campaign reuse/resume is intentionally rejected; start a new directory from a
saved accepted version. On a timeout or interruption inspect `current.json` and
the logs. Only versions with a completed pointer update count as promoted.
Retain these files outside an ephemeral notebook session.

## Architecture changes in this revision

**Token-specific pondering loss.** Previously the implementation first averaged
stopping probabilities over the batch/sequence, then weighted scalar task losses.
That incorrectly assigns a shared stopping responsibility to unrelated tokens.
Now each prediction error is weighted by the stopping probability at its own
prediction position, normalized by the original modality target count, and
summed across loops. Unreduced continuous-head errors make this possible for
audio, image, video, fields, geometry and quantities, as well as text/actions.
Padding remains excluded. With one loop the stopping weight is one.

**Learned bank integration gate.** Optional `gated_bank=True` adds one trainable
scalar per hidden channel. A sigmoid gate (initialized to 0.5) scales specialist
output before it joins the core residual. The model can learn which channels
benefit from expert output rather than always injecting it at full strength.
Parameter accounting and serialization include the gate. Its benefit is unmeasured.

**Loop identity conditioning.** Optional `loop_identity=True` gives the recurrent
core an explicit refinement-pass signal by reusing the router's existing loop
embedding at re-entry. This adds no parameters and keeps loop-specific cache
keys. It allows the shared core to distinguish repeated passes more directly.

New notebooks enable both options; old configs default them off for checkpoint
compatibility. Research candidates can compare them under fixed evaluation.
New loop-loss behavior changes multi-loop training semantics; do not interpret
historical multi-loop numbers as validation. All earlier multimodal, agent,
Kaggle and optimization changes remain included.
