"""Phase 2 — let the bank differentiate.

Phase 1 needs the balance term to stop the router collapsing before any stack
has a reason to prefer anything. Phase 2 needs it *out of the way*, because a
balance objective computed over a mixed corpus and a specialisation objective
are the same pressure pointed in opposite directions once the stacks start to
differ.

The schedule: anneal the balance coefficient toward a floor, freeze the core's
attention so the shared representation stops moving underneath the router, and
let the stacks fit their traffic. The measurement that matters is
``I(family; stack)`` before and after — reported by
``iridium.evaluation.routing``. Balance alone proves nothing.

    python -m iridium.training.phase2_specialize --init runs/phase1/phase1-final.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..config import get_config
from ..evaluation.harness import evaluate
from ..evaluation.routing import analyse
from ..model.iridium1 import Iridium1
from .datasets import BatchLoader, build_corpus, describe
from .losses import LossWeights
from .trainer import TrainConfig, Trainer, load_checkpoint


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="", help="phase 1 checkpoint")
    ap.add_argument("--rung", default="nano")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--balance", type=float, default=0.05,
                    help="balance weight floor; phase 1 uses 1.0")
    ap.add_argument("--train-items", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="runs/phase2")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    if args.init:
        model, manifest = load_checkpoint(args.init)
        print(f"initialised from {args.init}")
    else:
        model = Iridium1(get_config(args.rung))
        print("initialised from scratch (no phase-1 checkpoint given)")

    train = build_corpus(args.train_items, seed=args.seed, split="train")
    test = build_corpus(400, seed=args.seed + 1000, split="test")
    print(describe(train))

    before = analyse(
        model, BatchLoader(test, model.cfg.codecs, args.batch_size, args.seed)
    )
    print("BEFORE specialisation:")
    print(before.render())

    # Freeze the core's attention: the router is learning to read a
    # representation, and moving both at once makes the measurement
    # uninterpretable.
    tcfg = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, label="phase2", log_every=max(args.steps // 30, 1),
        freeze=("core.layers",),
    )
    weights = LossWeights(router_balance=args.balance)
    trainer = Trainer(model, train, tcfg, weights, out_dir=Path(args.out))
    print(f"frozen parameters: {getattr(trainer, 'frozen_params', 0):,}")
    trainer.train()

    after = analyse(
        model, BatchLoader(test, model.cfg.codecs, args.batch_size, args.seed)
    )
    print("AFTER specialisation:")
    print(after.render())

    results = {
        "routing_before": before.as_dict(),
        "routing_after": after.as_dict(),
        "task_accuracy": evaluate(model, test, max_per_family=24),
    }
    print(json.dumps(results, indent=2))
    print("saved", trainer.save("final", extra=results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
