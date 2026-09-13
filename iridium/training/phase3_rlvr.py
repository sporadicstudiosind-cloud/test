"""Phase 3 — reinforcement learning from *verifiable* rewards.

No reward model. Every reward in this phase comes from something that can be
recomputed by a third party: an analytic law, the deterministic solver, the
sandbox's exit status, or the scene environment's goal predicate. A learned
reward model would reintroduce exactly the failure this architecture is
supposed to remove — a scorer that can be satisfied without being right.

Algorithm: group-relative REINFORCE. For each prompt, sample ``G`` completions,
score them, and use the group mean as the baseline:

    A_i = (r_i - mean(r)) / (std(r) + eps)
    L   = -mean_i( A_i * sum_t log pi(y_t^i | y_<t^i) ) + beta * KL(pi || pi_ref)

The group baseline needs no value network, which matters here because a value
head trained on the same tiny corpus would be the least reliable component in
the loop. The KL term is against a frozen copy of the phase-2 policy and is
what stops the model from finding a degenerate string that happens to score.

**Anti-sycophancy is trained here**, not bolted on. The false-premise family
rewards refuting a false claim and rewards confirming a true one, with the
*same* magnitude. Rewarding only refusal produces a model that refuses, which
is a different failure with better manners. ``reward_shape`` records both.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..codecs.bank import TensorBatch, continuous_dims
from ..codecs.spans import MODALITY_INDEX, Sample, Span, collate
from ..evaluation.harness import evaluate, prompt_only
from ..runtime.generate import generate
from .datasets import build_corpus
from .tasks import Item, TEXT_OFFSET, decode_text
from .trainer import load_checkpoint


@dataclass
class RewardShape:
    correct: float = 1.0
    incorrect: float = 0.0
    malformed: float = -0.5
    refuted_false_premise: float = 1.0
    accepted_false_premise: float = -1.0
    confirmed_true_premise: float = 1.0
    denied_true_premise: float = -1.0
    length_penalty: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


def verifiable_reward(item: Item, produced: str, shape: RewardShape) -> tuple[float, str]:
    """Reward, plus the reason, so the log says *why* a sample scored."""
    text = produced.strip()
    if not text:
        return shape.malformed, "empty"
    if item.family == "false_premise":
        truthful = bool(item.truth["truthful"])
        said_yes = text.upper().startswith("YES")
        said_no = text.upper().startswith("NO")
        if not (said_yes or said_no):
            return shape.malformed, "no verdict token"
        if truthful:
            return (
                (shape.confirmed_true_premise, "confirmed a true premise")
                if said_yes
                else (shape.denied_true_premise, "denied a true premise")
            )
        return (
            (shape.refuted_false_premise, "refuted a false premise")
            if said_no
            else (shape.accepted_false_premise, "accepted a false premise")
        )
    ok = item.grade(text)
    penalty = shape.length_penalty * len(text)
    return (
        (shape.correct - penalty, "correct") if ok
        else (shape.incorrect - penalty, "incorrect")
    )


def _answer_batch(item: Item, answer: str, dims: dict[str, int]) -> TensorBatch:
    """Prompt + a candidate answer, with only the answer supervised."""
    from .tasks import control_span, encode_text

    spans = [s for s in item.sample.spans if not s.supervised]
    spans = spans + [encode_text(answer, supervised=True), control_span(2)]
    return TensorBatch(collate([Sample(spans)], dims))


def sequence_logprob(model, batch: TensorBatch, n_loops: int = 1) -> torch.Tensor:
    """Sum of log pi over the supervised text positions of one sample."""
    out = model(batch, n_loops=n_loops)
    h = out.hidden[:, :-1]
    target = batch.discrete[:, 1:]
    mask = batch.supervised[:, 1:] & batch.valid[:, 1:] & (
        (batch.modality[:, 1:] == MODALITY_INDEX["text"])
        | (batch.modality[:, 1:] == MODALITY_INDEX["control"])
    )
    logits = model.codecs.text_head(h)
    logp = -F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1).clamp(0, logits.shape[-1] - 1),
        reduction="none",
    ).view_as(mask)
    return (logp * mask).sum()


@dataclass
class RLVRStats:
    step: int
    mean_reward: float
    accuracy: float
    kl: float
    reasons: dict[str, int] = field(default_factory=dict)

    def render(self) -> str:
        top = ", ".join(f"{k}:{v}" for k, v in sorted(self.reasons.items()))
        return (
            f"rlvr step {self.step:4d}  reward={self.mean_reward:+.3f}  "
            f"acc={self.accuracy:.3f}  kl={self.kl:.4f}  [{top}]"
        )


def train_rlvr(
    model,
    items: Sequence[Item],
    steps: int = 80,
    group: int = 4,
    lr: float = 5e-5,
    temperature: float = 0.9,
    kl_beta: float = 0.02,
    max_new_tokens: int = 12,
    shape: Optional[RewardShape] = None,
    seed: int = 0,
    log_every: int = 10,
) -> list[RLVRStats]:
    shape = shape or RewardShape()
    dims = continuous_dims(model.cfg.codecs)
    reference = copy.deepcopy(model).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history: list[RLVRStats] = []

    for step in range(steps):
        item = items[int(rng.integers(len(items)))]
        prompt = prompt_only(item)

        model.eval()
        samples: list[str] = []
        rewards: list[float] = []
        reasons: dict[str, int] = {}
        with torch.no_grad():
            for g in range(group):
                out = generate(
                    model, prompt, max_new_tokens=max_new_tokens,
                    temperature=temperature, seed=int(rng.integers(1 << 30)),
                )
                reward, why = verifiable_reward(item, out.text, shape)
                samples.append(out.text)
                rewards.append(reward)
                reasons[why] = reasons.get(why, 0) + 1

        r = np.array(rewards, dtype=np.float64)
        if float(r.std()) < 1e-8:
            # Every completion scored the same: the group baseline cancels the
            # gradient exactly. Skipping is correct; taking the step would just
            # add optimizer noise.
            continue
        advantage = (r - r.mean()) / (r.std() + 1e-8)

        model.train()
        loss = None
        kl_total = 0.0
        for adv, text in zip(advantage, samples):
            batch = _answer_batch(item, text, dims)
            logp = sequence_logprob(model, batch)
            with torch.no_grad():
                ref_logp = sequence_logprob(reference, batch)
            kl = (logp - ref_logp)
            kl_total += float(kl.detach())
            term = -float(adv) * logp + kl_beta * kl.abs()
            loss = term if loss is None else loss + term
        loss = loss / len(samples)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        stats = RLVRStats(
            step=step,
            mean_reward=float(r.mean()),
            accuracy=float(np.mean(r > 0)),
            kl=kl_total / max(len(samples), 1),
            reasons=reasons,
        )
        history.append(stats)
        if step % log_every == 0:
            print(stats.render(), flush=True)
    return history


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--items", type=int, default=400)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="runs/phase3")
    ap.add_argument("--families", default="false_premise,channel_depth")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    model, manifest = load_checkpoint(args.init)
    wanted = set(args.families.split(","))
    corpus = build_corpus(args.items, seed=11, split="train")
    items = [it for it in corpus.items if it.family in wanted]
    print(f"{len(items)} verifiable items across {sorted(wanted)}")

    test = build_corpus(300, seed=12, split="test")
    before = evaluate(model, test, max_per_family=20)
    history = train_rlvr(model, items, steps=args.steps, group=args.group, lr=args.lr)
    after = evaluate(model, test, max_per_family=20)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "before": before, "after": after,
        "reward_shape": RewardShape().as_dict(),
        "history": [h.__dict__ for h in history],
    }
    (out_dir / "phase3.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "manifest": manifest},
               out_dir / "phase3-final.pt")
    print(json.dumps({"before": before, "after": after}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
