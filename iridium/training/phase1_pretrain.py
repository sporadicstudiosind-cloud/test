"""Phase 1 — omnimodal foundation training.

Every modality, one next-slot objective, one dense pass through the whole
model. Routing is learned here but not yet specialised: the balance term keeps
the bank from collapsing onto one stack before any stack has a reason to
prefer anything.

    python -m iridium.training.phase1_pretrain --rung nano --steps 1800
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..config import get_config
from ..evaluation.harness import evaluate
from ..model.iridium1 import Iridium1
from .datasets import build_corpus, describe
from .losses import LossWeights
from .trainer import TrainConfig, Trainer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung", default="nano")
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-loops", type=int, default=1)
    ap.add_argument("--train-items", type=int, default=12000)
    ap.add_argument("--eval-items", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="runs/phase1")
    ap.add_argument("--eval-per-family", type=int, default=24)
    ap.add_argument("--mixture", default="",
                    help="family=weight,... ; default is DEFAULT_MIXTURE")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="periodic checkpoints; a long run that is killed "
                         "without one loses everything, which is how the "
                         "first attempt at this run ended")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    cfg = get_config(args.rung)
    model = Iridium1(cfg)
    print(f"{cfg.name}: {sum(p.numel() for p in model.parameters()):,} parameters")

    mixture = None
    if args.mixture:
        mixture = {}
        for part in args.mixture.split(","):
            name, _, weight = part.partition("=")
            mixture[name.strip()] = float(weight or 1.0)
        print("mixture:", mixture, flush=True)
    train = build_corpus(args.train_items, seed=args.seed, split="train",
                         mixture=mixture)
    test = build_corpus(args.eval_items, seed=args.seed + 1000, split="test",
                        mixture=mixture)
    extra = build_corpus(args.eval_items // 2, seed=args.seed + 2000,
                         split="extrapolation", mixture=mixture)
    print(describe(train))
    print(describe(test))

    tcfg = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        n_loops=args.n_loops, seed=args.seed, label="phase1",
        log_every=max(args.steps // 60, 1),
        checkpoint_every=args.checkpoint_every,
    )
    trainer = Trainer(model, train, tcfg, LossWeights(), out_dir=Path(args.out))

    baseline = evaluate(model, test, max_per_family=8, n_loops=args.n_loops)
    print("untrained baseline:", json.dumps(baseline))

    trainer.train()

    model.eval()
    results = {
        "interpolation": evaluate(model, test, args.eval_per_family, args.n_loops),
        "extrapolation": evaluate(model, extra, args.eval_per_family, args.n_loops),
        "untrained": baseline,
    }
    print(json.dumps(results, indent=2))
    path = trainer.save("final", extra={"evaluation": results})
    print("saved", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
