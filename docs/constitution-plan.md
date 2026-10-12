# Constitution plan: training and runtime guard

Status: design only. No code changes accompany this document. Source of truth for principles is `iridium/alignment/constitution.py` (`PRINCIPLES`, `HARD_LIMITS`, `screen`).

Verified in the repo: `screen()` is defined in `constitution.py` and is referenced nowhere else under `iridium/` (grep, excluding `__pycache__`). Neither `Answerer.ask` (`iridium/runtime/abstain.py`) nor `ChatSession.send` (`iridium/runtime/chat.py`) calls it today. The guard in section 3 is therefore not yet wired.

## 1. Training plan (after Constitutional AI)

Reference: Bai et al. 2022, *Constitutional AI: Harmlessness from AI Feedback*, arXiv:2212.08073. Details below are from memory of that paper and are unverified against the text; check before citing.

**Stage SL (supervised, critique then revise).**
1. Draw a prompt from the mixed set (section 2).
2. Sample an initial response from the seed model.
3. For one or more principles, build a critique prompt from `Principle.critique` and a revision prompt from `Principle.revision`. The critique question is phrased so "yes" means a violation, which keeps the polarity the same as the runtime guard.
4. Use the final revision as the SFT target. Keep the (prompt, revised response) pair; discard intermediate drafts.

Feasibility at 50M-100M: a model this small cannot reliably critique its own output. Treat critique and revision text as produced by a stronger teacher or by templates, and state which. If a teacher model is used, its terms of use and output-licence rules must be checked first. Template-generated critiques are cheap and reproducible but teach the surface form of critique, not the judgement.

**Stage RLAIF (preference pairs via DPO).**
1. For each prompt, sample two responses from the SFT model.
2. Label the pair with an AI judge: which response better satisfies the principle set. The judge is either the critique P(yes) score (section 3) or a teacher, using the same principle text.
3. Train with DPO against a frozen reference copy of the SFT model. DPO needs no separate reward model and no PPO rollout loop, which matters at this size and on free compute. Costs: one reference forward pass per example and a beta hyperparameter to tune.

PPO is not planned. A reward model at 50M-100M is nearly the same size as the policy, and PPO adds rollout cost and instability without a clear benefit here.

**Compute.** Free-tier GPU sessions have time and memory limits that change; see `docs/free-tier-training.md` for the repo's current notes. Plan for short runs with checkpoints, and check the free-tier terms before a long run.

**Candidate public datasets. Verify each license and terms before use.**
- Anthropic HH-RLHF (Bai et al. 2022, arXiv:2204.05862): helpful and harmless preference pairs. Contains harmful transcripts; filter before use and do not copy them into the SFT set.
- PKU-SafeRLHF (PKU-Alignment): safety preference data. Check the license; some variants are non-commercial.
- XSTest (Röttger et al. 2023, arXiv:2308.01263): safe prompts that look unsafe, plus contrast unsafe prompts. Primarily an over-refusal eval; use as eval, not train.
- Others to consider, not yet checked: TruthfulQA (Lin et al. 2021) for honesty; a do-not-answer style refusal set. Verify availability and license.

Licences are not confirmed here. Each dataset needs a recorded license check before entering the repo.

## 2. Data mix

Over-refusal is scored as a failure (see the module docstring), so the mix must not teach "refuse when in doubt." Benign requests that share surface vocabulary with risky topics are the key class. Hard-limit items are a minority, enough to learn the boundary, not the bulk.

Proposed target shares (initial, to be tuned on the evals in section 4):

| Bucket | Share | Purpose |
|---|---|---|
| Helpful, benign, incl. scary-sounding wording | 45% | Teaches answering "how do I kill a process"-style requests |
| Hard-limit requests with brief decline + safe remainder | 10% | Teaches the boundary, by category |
| Oversight and corrigibility | 15% | Accept correction, no power seeking, no hidden goals |
| Honesty and calibration | 15% | Uncertainty, "I don't know", correct-but-unpopular answers |
| Judge items (critique yes/no, labelled) | 15% | Trains the guard's own scorer |

Rationale:
- Helpful-benign is the largest bucket because it is the cost of over-refusal, which the design treats as equal to harm.
- Matched contrast pairs are the highest-value data. Each pair shares vocabulary but differs in operational intent, so the model learns intent rather than keywords.
- Hard-limit items are organised by the category names in `HARD_LIMITS` only: `mass_casualty`, `cyberweapon`, `child_sexual`, `violence_targeted`, `oversight_subversion`. This document does not and should not list example requests, synthesis or operational detail, or the trigger terms used by `screen()`. Any corpus of such prompts is sensitive and should be stored with restricted access and kept out of the public repo.
- Oversight items cover the `oversight` group principles. Include items where the user offers a reason the model should evade monitoring; the target is a calm, open disagreement.
- Judge items let the same yes/no critique format be used at runtime and in training. Keep the polarity consistent: yes means violation.

## 3. Runtime guard

Intended flow, applied to `Answerer.ask` before the escalation ladder and to `ChatSession.send` before generation:

1. **Pre-screen.** Call `screen(text)`. It returns zero or more hard-limit categories. An empty result skips to generation with no added cost.
2. **Self-critique.** If any category is flagged, or always for a configurable sample of traffic, run the critique question(s) from the relevant principles. Score as P(yes) versus P(no) for the next token after the critique prompt, so this is one prefill pass with no generation.
3. **Decide.**
   - Hard-limit principle P(yes) above a threshold `t_hard` -> decline path. Brief decline, no lecture, help with any safe remainder. Do not enter the search or extrapolate rungs; `ASK_TO_EXTRAPOLATE` is the wrong response to a hard limit.
   - Non-hard-limit principle P(yes) above `t_soft` -> revise. Generate a revision using the principle's revision text, then show the revision.
   - Otherwise -> answer normally.
4. **Thresholds.** Start with placeholder values such as 0.5 and calibrate on a held-out set. Set `t_hard` for high recall, accepting some false positives, because the decline path is narrow and recoverable. Set `t_soft` for a fixed over-refusal budget measured on XSTest-style prompts. These values are untested.

`screen()` is a pre-filter, not a decision. The module docstring says flags go to self-critique rather than straight to refusal. Keep that.

**Latency.** Estimated, not measured:
- Critique scoring is one prefill pass over prompt plus response. Expect a modest fraction of a generation's cost.
- A revision is a full generation, up to `max_new_tokens` (default 160 in `ChatSession`), so a revised turn roughly doubles generation time.
- Running the critique on every turn adds a constant overhead; sampling it or gating it on `screen()` hits keeps the average low.

**Interaction with ThinkingBudget.** `Answerer` rung 1 already uses `ThinkingBudget("balanced")`, which means one model call per token per loop. Run the critique with `thinking=None` and fixed loops, so the guard's cost does not multiply with the thinking budget. Allow the revision pass to use thinking only if the answer path does. The guard should run once before the ladder, not once per rung, so the ladder's widening logic does not re-screen the same question.

## 4. Evals

- **Over-refusal rate.** Refusal rate on benign prompts, including XSTest safe prompts and an internal set of benign scary-sounding prompts. Target: a fixed budget set at calibration time, not zero. Report per bucket.
- **Hard-limit refusal rate.** Refusal (or safe decline) rate on a held-out set of prompts per `HARD_LIMITS` category. Report per category with a confidence interval, since counts will be small. Store the set with restricted access.
- **Oversight-probe consistency.** For each oversight probe, (a) agreement across paraphrases of the same question, (b) agreement across system-prompt framings, (c) whether the answer changes when the prompt says it is or is not monitored. Measure as the fraction of probe groups with a consistent answer.
- **Honesty and calibration.** Expected calibration error on factual QA using the model's stated confidence; abstention rate on unanswerable questions; TruthfulQA-style accuracy if licence allows.
- **Screen quality.** Precision and recall of `screen()` on a labelled set, per category. Track false positives separately from false negatives.
- **Guard ablation.** Run every eval with the guard off, screen-only, and full guard, so the guard's effect is visible.

Judges: where an LLM judges outputs, use a fixed judge prompt, report judge agreement against a human-labelled subset, and never let the judge be the same model being evaluated.

## 5. Limitations

- **Shallow dispositions.** A small model trained on templates is likely to hold these principles as surface patterns. Oversight dispositions in particular may vanish under paraphrase or context shift. The oversight-consistency eval exists to measure this, and the expectation is that it will show weakness.
- **Self-critique needs capability.** At this scale the critique may be noise. The guard's P(yes) scores need to be shown to correlate with human labels before the guard is trusted.
- **Regex false negatives.** The patterns need a specific harm term and a nearby operational verb in a fixed window. Paraphrase, other languages, misspellings, obfuscation, and multi-turn decomposition can all evade them.
- **Regex false positives.** Benign uses of ordinary words can match the oversight pattern (for example, shutdown or monitoring in a server-operations context) and some category patterns are broad. Each produces a critique call, not a refusal, which limits harm but still adds latency.
- **The screen is not a security boundary.** Determined users can route around both the screen and the guard. The trained model and the guard are defence in depth, not a guarantee.
- **Principles are an untested target.** `PRINCIPLES` is a hypothesis about what to train. Nothing here proves alignment, as the module docstring says.
- **Judge and teacher bias.** AI-labelled preferences inherit the judge's biases, and DPO can shift length and style, which can move the over-refusal rate in ways the evals must catch.
- **Threshold gaming.** Tuning P(yes) thresholds on the same sets used for reporting will overstate results. Keep a separate test split.
- **Unverified.** Paper details, dataset licences, free-tier limits, and latency estimates are all unverified in this document.

## 6. Open items

1. Verify each dataset licence and record it.
2. Decide the teacher source for critiques and revisions, and its terms.
3. Build the held-out hard-limit set under restricted access.
4. Implement the guard behind a flag, then calibrate thresholds on a split not used for reporting.
5. Measure guard latency on the target hardware instead of relying on the estimate above.
