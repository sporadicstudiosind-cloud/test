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
import os
import math
import subprocess
import time
from dataclasses import asdict, dataclass, replace, field
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
    #: Micro-batches summed before each optimizer step. ``batch_size`` is the
    #: micro-batch, so the effective batch is ``batch_size * accumulate``. This
    #: is the cheapest way to train at a large batch on a small card: the
    #: gradient is identical, only the peak activation memory differs.
    accumulate: int = 1
    #: "adamw" | "adamw_8bit" | "paged_adamw" | "adafactor" | "sgd".
    #: ``paged_adamw`` keeps the moments in host RAM.
    optimizer: str = "eager_adamw"
    devices: tuple[str, ...] = ()  # empty: use the supplied primary device
    precision: str = "auto"  # auto | fp32 | fp16 | bf16
    max_length: int | None = None
    #: On CUDA OOM, halve the micro-batch and retry rather than losing the run.
    oom_retry: bool = True
    #: Split parameters so that weight decay reaches only rank->=2 weights. See
    #: :func:`iridium.runtime.memory.decay_groups` for what each excluded kind of
    #: tensor loses when it is decayed; the short version is that decaying an
    #: RMSNorm gain attenuates the whole residual stream and decaying a halting
    #: bias collapses the ponder loop. Set False only to reproduce a run made
    #: before this existed.
    decay_groups: bool = True
    #: Fuse per-tensor optimizer and gradient-clipping work into list operations.
    #: ``None`` decides by device: on a GPU this model's thousands of small
    #: tensors make the update launch-bound, and on a CPU there is no launch
    #: overhead to remove. Set explicitly only to pin behaviour for a comparison.
    foreach: bool | None = None
    #: Warmup as a fraction of ``steps``, used when it exceeds ``warmup``. A
    #: fixed 50-step warmup is far too short for a routed model: the macro
    #: router's gate is what decides which superstack sees which token, and at a
    #: high learning rate in the first hundred steps it will commit to a
    #: partition before any stack has learned anything to justify it. That
    #: commitment does not come back — the unvisited stacks receive no gradient
    #: and stay untrained, and the loss curve looks merely mediocre rather than
    #: broken. The larger of the two is used so an explicit ``warmup`` still
    #: wins on a short debug run.
    warmup_ratio: float = 0.02
    #: ``"cosine"`` or ``"wsd"`` (warmup, stable, decay). WSD holds the peak
    #: rate and decays only over the final ``decay_ratio`` of the run, with a
    #: ``1 - sqrt`` cooldown -- Hagele et al. (2024, arXiv 2405.18392) found it
    #: matches cosine at equal compute. Its practical advantage is that the
    #: stable phase has no end date: a run can be extended, or branched into
    #: several cooldowns, without restarting a schedule whose shape was fixed
    #: by a step count chosen before anyone knew how long training should be.
    schedule: str = "cosine"
    decay_ratio: float = 0.2
    #: Per-modality task-loss balancing; see :class:`~.losses.LossBalancer`.
    #: Off by default because it changes the objective and is unvalidated here.
    loss_balance: str = "none"
    balance_momentum: float = 0.99
    #: Exponential moving average of the weights, kept alongside them (0 off).
    #: Generators -- the flow heads here -- are evaluated on the averaged
    #: weights as standard practice (DDPM, EDM; Karras et al. 2024 on
    #: post-hoc EMA), because the raw weights at any single step carry the
    #: last few updates' noise straight into the samples. 0.999 is typical.
    ema_decay: float = 0.0

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


def warmup_steps(cfg: TrainConfig) -> int:
    """The effective warmup: the larger of the explicit count and the ratio.

    Capped below the total step count, because a warmup longer than the run is
    a linear ramp that never reaches the configured learning rate, and the run
    then reports a schedule it did not follow.
    """
    ratio = int(cfg.steps * max(cfg.warmup_ratio, 0.0))
    return min(max(cfg.warmup, ratio), max(cfg.steps - 1, 0))


def learning_rate(step: int, cfg: TrainConfig) -> float:
    """The configured schedule's rate at ``step``."""
    if cfg.schedule == "cosine":
        return cosine_lr(step, cfg)
    if cfg.schedule == "wsd":
        return wsd_lr(step, cfg)
    raise ValueError(f"unknown schedule {cfg.schedule!r}")


def wsd_lr(step: int, cfg: TrainConfig) -> float:
    warm = warmup_steps(cfg)
    if step < warm:
        return cfg.lr * (step + 1) / max(warm, 1)
    decay_start = max(warm, int(round(cfg.steps * (1.0 - cfg.decay_ratio))))
    if step < decay_start:
        return cfg.lr
    t = min((step - decay_start) / max(cfg.steps - decay_start, 1), 1.0)
    floor = cfg.lr * cfg.min_lr_ratio
    return floor + (cfg.lr - floor) * (1.0 - math.sqrt(t))


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    warm = warmup_steps(cfg)
    if step < warm:
        return cfg.lr * (step + 1) / max(warm, 1)
    t = (step - warm) / max(cfg.steps - warm, 1)
    floor = cfg.lr * cfg.min_lr_ratio
    return floor + 0.5 * (cfg.lr - floor) * (1.0 + math.cos(math.pi * min(t, 1.0)))


def _fsync(path: Path) -> None:
    """Force a file (or directory entry) to stable storage.

    Colab's Drive mount uploads lazily: a checkpoint that is only in the page
    cache when the runtime disconnects never reaches Drive, which is how a run
    that logged three rounds can leave only ``round0`` behind. fsync makes the
    FUSE layer flush before training continues.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass                       # some filesystems refuse fsync on directories
    finally:
        os.close(fd)


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
        if min(cfg.steps, cfg.batch_size, cfg.accumulate, cfg.log_every) < 1:
            raise ValueError("steps, batch size, accumulation and log interval must be positive")
        if not train_corpus.items:
            raise ValueError("training corpus is empty")
        if not 1 <= cfg.n_loops <= model.cfg.router.max_loops:
            raise ValueError("n_loops exceeds the configured loop budget")
        from ..runtime.placement import place_model, native_bf16
        self.devices = tuple(cfg.devices) or (device,)
        if str(self.devices[0]) != str(device):
            raise ValueError('primary device must equal devices[0]')
        if len(self.devices) > 1 and cfg.optimizer not in ('eager_adamw', 'adamw8'):
            raise ValueError('multi-device training currently requires eager_adamw')
        place_model(model, self.devices)
        self.model = model
        self.cfg = cfg
        self.weights = weights or LossWeights()
        self.device = device
        self.out_dir = Path(out_dir) if out_dir else None
        if cfg.max_length is not None and not 2 <= cfg.max_length <= model.cfg.max_seq_len:
            raise ValueError("max_length must lie within model context and be >= 2")
        if not hasattr(torch.amp, "GradScaler"):
            raise RuntimeError("This training path requires PyTorch 2.3+ with torch.amp.GradScaler")
        self.loader = BatchLoader(
            train_corpus, model.cfg.codecs, cfg.batch_size, cfg.seed, device=device,
            max_length=cfg.max_length or model.cfg.max_seq_len
        )
        self.history: list[dict[str, Any]] = []
        self._apply_freeze()
        from ..runtime.memory import build_optimizer, decay_groups
        if cfg.schedule not in ("cosine", "wsd") or not 0.0 < cfg.decay_ratio <= 1.0:
            raise ValueError("schedule must be cosine or wsd, decay_ratio in (0, 1]")
        if not 0.0 <= cfg.ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")
        if cfg.optimizer == "muon":
            # Muon is defined for hidden matrices only; embeddings, the output
            # head, norms and biases go to its internal AdamW path. The split
            # is by module type, so it needs the model, not a parameter list.
            from .muon import muon_param_groups
            optimizer_params = muon_param_groups(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
            self.decayed_params = sum(p.numel() for g in optimizer_params for p in g["params"]
                                      if g.get("weight_decay", 0.0))
            self.undecayed_params = sum(p.numel() for g in optimizer_params
                                        for p in g["params"]) - self.decayed_params
        elif cfg.decay_groups:
            groups = decay_groups(model, cfg.weight_decay)
            self.decayed_params = sum(p.numel() for p in groups[0]["params"])
            self.undecayed_params = sum(p.numel() for p in groups[1]["params"])
            optimizer_params = groups
        else:
            optimizer_params = [p for p in model.parameters() if p.requires_grad]
            self.decayed_params = sum(p.numel() for p in optimizer_params)
            self.undecayed_params = 0
        if cfg.optimizer == "muon":
            from .muon import Muon
            self.optimizer = Muon(optimizer_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            self.optimizer = build_optimizer(
                optimizer_params, kind=cfg.optimizer, lr=cfg.lr,
                weight_decay=cfg.weight_decay, foreach=cfg.foreach,
            )
        from .losses import LossBalancer
        self.balancer = LossBalancer(cfg.loss_balance, cfg.balance_momentum)
        self.ema: Optional[dict[str, torch.Tensor]] = (
            {k: v.detach().clone().float() for k, v in model.state_dict().items()
             if v.is_floating_point()}
            if cfg.ema_decay else None)
        from ..runtime.device import generator_for
        self.generator = generator_for(device, cfg.seed)
        cuda = str(device).startswith("cuda")
        precision = cfg.precision
        if precision == "auto":
            precision = ("bf16" if native_bf16(self.devices) else "fp16") if cuda else "fp32"
        if precision not in ("fp32", "fp16", "bf16") or (not cuda and precision != "fp32"):
            raise ValueError("use fp32 on CPU; CUDA supports fp32/fp16/bf16")
        if precision == "bf16" and not native_bf16(self.devices):
            raise ValueError("this GPU does not support bf16; use fp16")
        self.precision = precision
        self.amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        self.completed_steps = 0

    def set_corpus(self, corpus: Corpus) -> None:
        """Swap in a fresh corpus between rounds without touching model,
        optimizer, schedule or step count -- how a run streams more data than
        fits in memory as one list of items."""
        if not corpus.items:
            raise ValueError("training corpus is empty")
        self.loader = BatchLoader(
            corpus, self.model.cfg.codecs, self.loader.batch_size,
            self.cfg.seed + self.completed_steps, device=self.device,
            max_length=self.cfg.max_length or self.model.cfg.max_seq_len,
        )

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
        self, on_eval: Optional[Callable[[int], dict[str, Any]]] = None,
        until: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Train to ``cfg.steps``, or stop early at step ``until``.

        ``until`` is how a run proceeds in rounds of fresh data
        (:meth:`set_corpus` between calls) while the learning-rate schedule
        still spans the whole run: the schedule reads ``cfg.steps``, which
        never changes, so round boundaries are invisible to it.
        """
        if not self.completed_steps:
            torch.manual_seed(self.cfg.seed)
        self.model.train()
        step = self.completed_steps
        started = time.time()
        stream = self._infinite_batches()
        overflow_retries = 0
        stop = self.cfg.steps if until is None else min(until, self.cfg.steps)
        while step < stop:
            lr = learning_rate(step, self.cfg)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            # An OOM restarts the entire update, never an individual forward.
            # Catch outside the helper frame so failed graphs can be released.
            try:
                report, diagnostics, norm, updated = self._update(stream)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                # CPU and ROCm allocators raise a plain RuntimeError; anything
                # else that is a RuntimeError is a real bug and re-raised.
                if not isinstance(err, torch.cuda.OutOfMemoryError) \
                        and "allocate memory" not in str(err):
                    raise
                self.optimizer.zero_grad(set_to_none=True)
                if not self.cfg.oom_retry or self.loader.batch_size <= 1:
                    raise
                import sys
                exc = sys.exc_info()[1]
                exc.__traceback__ = None
                old = self.loader.batch_size
                effective = old * self.cfg.accumulate
                new = max(1, old // 2)
                self.loader.batch_size = new
                self.cfg = replace(self.cfg, batch_size=new,
                                   accumulate=math.ceil(effective / new))
                scale_state = self.scaler.state_dict()
                self.scaler = torch.amp.GradScaler("cuda", enabled=self.precision == "fp16")
                self.scaler.load_state_dict(scale_state)
                stream = self._infinite_batches()  # discard old prebuilt groups
                import gc
                gc.collect()
                for device in self.devices:
                    if str(device).startswith('cuda'):
                        with torch.cuda.device(device):
                            torch.cuda.empty_cache()
                print(f"[trainer] retry whole update: micro-batch {old}->{new}; "
                      f"accumulate={self.cfg.accumulate}", flush=True)
                continue
            if not updated:
                overflow_retries += 1
                if overflow_retries >= 16:
                    raise FloatingPointError("16 consecutive fp16 overflows; restart in fp32")
                print("[trainer] fp16 overflow: reduced loss scale; retry update", flush=True)
                continue
            overflow_retries = 0
            self.completed_steps = step + 1
            if step % self.cfg.log_every == 0 or step == self.cfg.steps - 1:
                from ..runtime.placement import memory_snapshot
                record = {
                    "gpu_memory": memory_snapshot(self.devices),
                    "step": step,
                    "lr": lr,
                    "grad_norm": norm,
                    "elapsed": time.time() - started,
                    **report,
                    **diagnostics,
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

    def _update(self, stream):
        self.optimizer.zero_grad(set_to_none=True)
        report = {}
        diagnostics = {}
        for _ in range(self.cfg.accumulate):
            batch, _items = next(stream)
            with torch.autocast(device_type="cuda" if str(self.device).startswith("cuda") else "cpu",
                                dtype=self.amp_dtype, enabled=self.precision != "fp32"):
                losses, out = self.model.losses(batch, n_loops=self.cfg.n_loops,
                                                generator=self.generator)
                total, micro_report = combine(self.balancer(losses), self.weights)
                if self.balancer.mode != "none":
                    # Balanced terms sit near 1 by construction; log the raw
                    # values too, or the log hides whether anything is learning.
                    micro_report.update({f"raw.{k}": float(v.detach()) for k, v in losses.items()})
            if not bool(torch.isfinite(total)):
                raise FloatingPointError("non-finite loss; optimizer update cancelled")
            self.scaler.scale(total / self.cfg.accumulate).backward()
            for key, value in micro_report.items():
                report[key] = report.get(key, 0.0) + value / self.cfg.accumulate
            diagnostics = self._router_diagnostics(out)
            del losses, out, total, batch
        self.scaler.unscale_(self.optimizer)
        params = [p for p in self.model.parameters() if p.grad is not None]
        # foreach=None lets torch fuse the norm reduction and the rescale across
        # the whole parameter list. Pinning it False issued two kernels per
        # tensor every single step, on a model that has thousands of them.
        norm = torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip or float("inf"),
                                             error_if_nonfinite=False,
                                             foreach=self.cfg.foreach)
        if not bool(torch.isfinite(norm)) and self.precision != "fp16":
            raise FloatingPointError("non-finite gradient; optimizer update cancelled")
        previous_scale = self.scaler.get_scale()
        try:
            if torch.device(self.device).type == "xla":
                # XLA tensors are lazy: optimizer_step applies the update *and*
                # marks the graph boundary that executes the queued work. A plain
                # optimizer.step() would keep growing one enormous graph. The
                # GradScaler is disabled off fp16, so there is nothing to unscale.
                import torch_xla.core.xla_model as xm
                xm.optimizer_step(self.optimizer, barrier=True)
            else:
                self.scaler.step(self.optimizer)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("OOM during optimizer update; restart from last checkpoint with "
                               "a smaller model/optimizer. Retrying could double-update weights.") from exc
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        updated = self.scaler.get_scale() >= previous_scale
        if self.ema is not None and updated:
            self._update_ema()
        return report, diagnostics, float(norm), updated

    @torch.no_grad()
    def _update_ema(self) -> None:
        d = self.cfg.ema_decay
        for name, value in self.model.state_dict().items():
            shadow = self.ema.get(name)
            if shadow is not None:
                shadow.lerp_(value.detach().to(shadow.device, shadow.dtype), 1.0 - d)

    def ema_state_dict(self) -> Optional[dict[str, torch.Tensor]]:
        """The averaged weights, shaped like ``model.state_dict()`` (buffers
        and non-float entries taken from the live model). ``None`` when off."""
        if self.ema is None:
            return None
        live = self.model.state_dict()
        return {k: (self.ema[k].to(v.dtype) if k in self.ema else v) for k, v in live.items()}

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
            "router_entropy": float(decision.entropy().detach()),
            "stack_usage_entropy": entropy,
            "stack_usage_max": float(share.max()) if share.size else 0.0,
            "mean_focus": float(decision.focus.detach().mean()),
            "mean_depth": float(np.mean(depths)) if depths else 0.0,
            "expected_loops": float(out.expected_loops.detach().mean()),
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
        for device, memory in record.get('gpu_memory', {}).items():
            parts.append(f"{device}={memory['allocated_gb']:.2f}GB")
        return "  ".join(parts)

    # -- persistence ------------------------------------------------------

    def resume(self, path):
        # Trusted local training checkpoints only: pickle contains optimizer/RNG state.
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob["manifest"]["model_config"] != self.model.cfg.to_dict():
            raise ValueError("resume model config differs")
        if blob["manifest"]["train_config"]["optimizer"] != self.cfg.optimizer:
            raise ValueError("resume optimizer differs")
        if blob["manifest"].get("precision") != self.precision:
            raise ValueError("resume precision differs")
        self.model.load_state_dict(blob["state_dict"])
        self.optimizer.load_state_dict(blob["optimizer"])
        self.scaler.load_state_dict(blob["scaler"])
        if self.ema is not None and blob.get("ema_state_dict") is not None:
            self.ema = {k: v.detach().clone().float() for k, v in blob["ema_state_dict"].items()
                        if k in self.ema}
        if blob.get("loss_balancer") is not None:
            self.balancer.load_state_dict(blob["loss_balancer"])
        self.completed_steps = blob["completed_steps"]
        self.history = blob.get("history", [])
        torch.set_rng_state(blob["torch_rng"].cpu())
        if self.generator is not None and blob.get("generator_rng") is not None:
            self.generator.set_state(blob["generator_rng"].cpu())
        self.loader.rng.bit_generator.state = blob["loader_rng"]
        if blob.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([v.cpu() for v in blob["cuda_rng"]])
        # A resumed loader starts a new shuffle; this is not bit-exact replay.

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
            "precision": self.precision,
            "device_plan": self.model.device_plan,
            "completed_steps": self.completed_steps,
            "data": getattr(self, "data_info", {}),
        }
        if extra:
            data.update(extra)
        return data

    def save(self, tag: str = "final", extra: Optional[dict[str, Any]] = None) -> Optional[Path]:
        if self.out_dir is None:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.cfg.label}-{tag}.pt"
        save = torch.save
        if torch.device(self.device).type == "xla":
            # xm.save moves lazy XLA tensors to the host before writing;
            # torch.save cannot serialise XLA device storage.
            import torch_xla.core.xla_model as xm
            save = lambda obj, f: xm.save(obj, f, master_only=True)  # noqa: E731
        save(
            {
                "state_dict": self.model.state_dict(),
                "manifest": self.manifest(extra),
                "history": self.history,
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "ema_state_dict": self.ema_state_dict(),
                "loss_balancer": self.balancer.state_dict(),
                "completed_steps": self.completed_steps,
                "torch_rng": torch.get_rng_state(),
                "generator_rng": self.generator.get_state() if self.generator is not None else None,
                "loader_rng": self.loader.rng.bit_generator.state,
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            path.with_suffix(".pt.tmp"),
        )
        tmp = path.with_suffix(".pt.tmp")
        _fsync(tmp)
        tmp.replace(path)
        _fsync(self.out_dir)
        if path.stat().st_size == 0:
            raise RuntimeError(f"checkpoint {path} was written empty")
        (self.out_dir / f"{self.cfg.label}-{tag}.json").write_text(
            json.dumps(
                {"manifest": self.manifest(extra), "history": self.history},
                indent=2, default=str,
            ),
            encoding="utf-8",
        )
        # Pointer to the newest checkpoint, so resume="auto" needs no filename.
        (self.out_dir / "latest.json").write_text(json.dumps(
            {"path": path.name, "tag": tag, "completed_steps": self.completed_steps}),
            encoding="utf-8")
        _fsync(self.out_dir / "latest.json")
        return path


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[Iridium1, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = IridiumConfig.from_dict(blob["manifest"]["model_config"])
    model = Iridium1(cfg).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob["manifest"]
