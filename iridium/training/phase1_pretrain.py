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
    ap.add_argument("--recipe", choices=("default", "omni"), default="default",
                    help="omni = OMNI_MIXTURE (half natural language), WSD schedule, "
                         "EMA weights and the rung's own tokenizer; needs network")
    ap.add_argument("--optimizer", default="eager_adamw",
                    help="eager_adamw | adamw | muon | adafactor | ...")
    ap.add_argument("--schedule", choices=("cosine", "wsd"), default=None)
    ap.add_argument("--ema-decay", type=float, default=None)
    ap.add_argument("--loss-balance", choices=("none", "ema"), default="none")
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
    from .datasets import OMNI_MIXTURE
    if args.recipe == "omni" and not args.mixture:
        mixture = dict(OMNI_MIXTURE)
    if args.mixture:
        mixture = {}
        for part in args.mixture.split(","):
            name, _, weight = part.partition("=")
            mixture[name.strip()] = float(weight or 1.0)
        print("mixture:", mixture, flush=True)
    from .tokenizer_bridge import tokenizer_for_config, tokenizer_manifest
    tokenizer = tokenizer_for_config(cfg) if args.recipe == "omni" else None
    print("tokenizer:", tokenizer_manifest(tokenizer, cfg))
    from .budget import audit
    from .datasets import DEFAULT_MIXTURE
    print(audit(cfg, args.train_items, args.steps, args.batch_size,
                mixture=mixture or DEFAULT_MIXTURE).describe(), flush=True)
    train = build_corpus(args.train_items, seed=args.seed, split="train",
                         mixture=mixture, tokenizer=tokenizer)
    # Same vocabulary for evaluation as for training; a mismatch does not
    # raise, it just scores the model on text it reads differently.
    test = build_corpus(args.eval_items, seed=args.seed + 1000, split="test",
                        mixture=mixture, tokenizer=tokenizer)
    extra = build_corpus(args.eval_items // 2, seed=args.seed + 2000,
                         split="extrapolation", mixture=mixture, tokenizer=tokenizer)
    print(describe(train))
    print(describe(test))

    tcfg = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        n_loops=args.n_loops, seed=args.seed, label="phase1",
        log_every=max(args.steps // 60, 1),
        checkpoint_every=args.checkpoint_every,
        optimizer=args.optimizer,
        schedule=args.schedule or ("wsd" if args.recipe == "omni" else "cosine"),
        ema_decay=(args.ema_decay if args.ema_decay is not None
                   else (0.999 if args.recipe == "omni" else 0.0)),
        loss_balance=args.loss_balance,
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
