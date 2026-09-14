"""The training loop, its manifest, and its checkpoints.

Nothing exotic. What it does insist on:

* every run writes a **manifest** — config, mixture, seeds, loss weights, git
  revision — so a number can be traced to the run that produced it;
* evaluation reports **graded accuracy** from free-running generation, not only
  teacher-forced loss, because the two come apart and only one of them is the
  claim anyone cares about;
* router diagnostics are logged every step, because a collapsed router shows up
  in the entropy long before it shows up in the loss.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch

from ..config import IridiumConfig
from ..model.iridium1 import Iridium1
from .datasets import BatchLoader, Corpus
from .losses import LossWeights, combine, grad_global_norm


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 8
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup: int = 50
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    n_loops: int = 1
    seed: int = 0
    log_every: int = 25
    eval_every: int = 0
    checkpoint_every: int = 0
    freeze: tuple[str, ...] = ()
    label: str = "phase1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except Exception:                                    # pragma: no cover
        return "unknown"


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / max(cfg.warmup, 1)
    t = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
    floor = cfg.lr * cfg.min_lr_ratio
    return floor + 0.5 * (cfg.lr - floor) * (1.0 + math.cos(math.pi * min(t, 1.0)))


class Trainer:
    def __init__(
        self,
        model: Iridium1,
        train_corpus: Corpus,
        cfg: TrainConfig,
        weights: Optional[LossWeights] = None,
        out_dir: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.weights = weights or LossWeights()
        self.device = device
        self.out_dir = Path(out_dir) if out_dir else None
        self.loader = BatchLoader(
            train_corpus, model.cfg.codecs, cfg.batch_size, cfg.seed, device=device
        )
        self.history: list[dict[str, Any]] = []
        self._apply_freeze()
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
        )
        from ..runtime.device import generator_for
        self.generator = generator_for(device, cfg.seed)

    def _apply_freeze(self) -> None:
        if not self.cfg.freeze:
            return
        frozen = 0
        for name, param in self.model.named_parameters():
            if any(name.startswith(prefix) for prefix in self.cfg.freeze):
                param.requires_grad_(False)
                frozen += param.numel()
        self.frozen_params = frozen

    # -- loop -------------------------------------------------------------

    def train(
        self, on_eval: Optional[Callable[[int], dict[str, Any]]] = None
    ) -> list[dict[str, Any]]:
        torch.manual_seed(self.cfg.seed)
        self.model.train()
        step = 0
        started = time.time()
        stream = self._infinite_batches()
        while step < self.cfg.steps:
            batch, items = next(stream)
            lr = cosine_lr(step, self.cfg)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            losses, out = self.model.losses(
                batch, n_loops=self.cfg.n_loops, generator=self.generator
            )
            total, report = combine(losses, self.weights)
            total.backward()
            norm = grad_global_norm(self.model.parameters())
            if self.cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if step % self.cfg.log_every == 0 or step == self.cfg.steps - 1:
                record = {
                    "step": step,
                    "lr": lr,
                    "grad_norm": norm,
                    "elapsed": time.time() - started,
                    **report,
                    **self._router_diagnostics(out),
                }
                self.history.append(record)
                print(self._format(record), flush=True)

            if (
                self.cfg.eval_every
                and step
                and step % self.cfg.eval_every == 0
                and on_eval is not None
            ):
                self.model.eval()
                metrics = on_eval(step)
                self.model.train()
                self.history.append({"step": step, "eval": metrics})
                print(f"  eval@{step}: {metrics}", flush=True)

            if (
                self.cfg.checkpoint_every
                and step
                and step % self.cfg.checkpoint_every == 0
            ):
                self.save(f"step{step}")
            step += 1
        return self.history

    def _infinite_batches(self):
        while True:
            for batch, items in self.loader.batches():
                yield batch, items

    @staticmethod
    def _router_diagnostics(out) -> dict[str, float]:
        decision = out.decisions[0]
        stats = out.stats["stack_stats"][0]
        tokens = np.array(stats["per_stack_tokens"], dtype=float)
        share = tokens / max(tokens.sum(), 1.0)
        nonzero = share[share > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum()) if nonzero.size else 0.0
        depths = [d for d in stats["per_stack_expected_depth"] if d > 0]
        return {
            "router_entropy": float(decision.entropy()),
            "stack_usage_entropy": entropy,
            "stack_usage_max": float(share.max()) if share.size else 0.0,
            "mean_focus": float(decision.focus.mean()),
            "mean_depth": float(np.mean(depths)) if depths else 0.0,
            "expected_loops": float(out.expected_loops.mean()),
            "grid_intact": float(stats.get("grid_intact_fraction", 1.0)),
        }

    @staticmethod
    def _format(record: dict[str, Any]) -> str:
        keys = ("total", "quantity", "text", "field", "slot_type")
        parts = [f"step {record['step']:5d}"]
        parts += [f"{k}={record[k]:.4f}" for k in keys if k in record]
        parts.append(f"H(stack)={record.get('stack_usage_entropy', 0):.3f}")
        parts.append(f"focus={record.get('mean_focus', 0):.3f}")
        parts.append(f"depth={record.get('mean_depth', 0):.2f}")
        parts.append(f"|g|={record.get('grad_norm', 0):.2f}")
        parts.append(f"{record.get('elapsed', 0):.0f}s")
        return "  ".join(parts)

    # -- persistence ------------------------------------------------------

    def manifest(self, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        inventory = self.model.parameter_inventory()
        data = {
            "label": self.cfg.label,
            "git_revision": git_revision(),
            "model_config": self.model.cfg.to_dict(),
            "train_config": self.cfg.as_dict(),
            "loss_weights": self.weights.as_dict(),
            "parameters": inventory,
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "torch_version": torch.__version__,
        }
        if extra:
            data.update(extra)
        return data

    def save(self, tag: str = "final", extra: Optional[dict[str, Any]] = None) -> Optional[Path]:
        if self.out_dir is None:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.cfg.label}-{tag}.pt"
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "manifest": self.manifest(extra),
                "history": self.history,
            },
            path,
        )
        (self.out_dir / f"{self.cfg.label}-{tag}.json").write_text(
            json.dumps(
                {"manifest": self.manifest(extra), "history": self.history},
                indent=2, default=str,
            ),
            encoding="utf-8",
        )
        return path


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[Iridium1, dict]:
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = IridiumConfig.from_dict(blob["manifest"]["model_config"])
    model = Iridium1(cfg)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob["manifest"]
