"""Data-parallel training across GPUs: one process per device, gradients averaged.

Why not ``torch.nn.parallel.DistributedDataParallel``: it wraps ``forward``,
and this model trains through ``Iridium1.losses`` with data-dependent paths (a
superstack that receives no tokens in a batch has no gradient at all), which
DDP only tolerates with ``find_unused_parameters`` and a graph walk per step.
The explicit version is smaller and has no such cases: every rank runs its own
forward and backward on its own shard of the data, the flattened gradients are
summed with one ``all_reduce`` per optimizer step, and every rank then takes
the same optimizer step from the same averaged gradient. The global batch and
the schedule are unchanged, so a two-GPU run is the one-GPU run in about half
the time, not a different experiment.

Kaggle's "GPU T4 x2" is the case this exists for. The older multi-device path
(:mod:`iridium.runtime.placement`) spreads one model's stacks over two cards,
which buys memory, not speed.

``launch`` is the process launcher (what ``torchrun`` does, without needing it
on PATH): it re-runs the same command once per GPU with ``RANK`` and
``WORLD_SIZE`` set, and brings the whole group down if any rank fails.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch

__all__ = ["Group", "world_from_env", "launch", "gpu_count"]


def world_from_env() -> tuple[int, int]:
    """``(rank, world_size)`` for this process; ``(0, 1)`` when not launched."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world < 1 or not 0 <= rank < world:
        raise ValueError(f"bad RANK/WORLD_SIZE: {rank}/{world}")
    return rank, world


def gpu_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


class Group:
    """The set of training processes, seen from one of them."""

    def __init__(self, rank: int, world: int, device) -> None:
        import torch.distributed as dist

        self.rank, self.world, self.device = rank, world, torch.device(device)
        self.dist = dist
        if not dist.is_initialized():
            backend = "nccl" if self.device.type == "cuda" else "gloo"
            dist.init_process_group(backend=backend, rank=rank, world_size=world)
        self._buf: Optional[torch.Tensor] = None

    @property
    def primary(self) -> bool:
        return self.rank == 0

    # -- gradients ---------------------------------------------------------

    def average_gradients(self, params: Sequence[torch.nn.Parameter]) -> None:
        """Replace every rank's gradients with their mean over the group.

        A parameter no rank produced a gradient for stays ``None``, so the
        optimizer skips it exactly as it would in a single-process run (decay
        and momentum on an untouched superstack are not a no-op).
        """
        total = sum(p.numel() for p in params)
        if self._buf is None or self._buf.numel() != total:
            self._buf = torch.zeros(total, device=self.device, dtype=torch.float32)
        buf = self._buf
        has = torch.zeros(len(params), device=self.device, dtype=torch.int32)
        offset = 0
        for i, p in enumerate(params):
            n = p.numel()
            if p.grad is None:
                buf[offset:offset + n].zero_()
            else:
                buf[offset:offset + n].copy_(p.grad.reshape(-1))
                has[i] = 1
            offset += n
        self.dist.all_reduce(buf)
        self.dist.all_reduce(has)
        buf.div_(self.world)
        offset = 0
        present = has.tolist()
        for p, any_rank in zip(params, present):
            n = p.numel()
            if any_rank:
                # A copy, not a view: the buffer is reused next step, and the
                # optimizer may keep a reference to the gradient it was given.
                p.grad = buf[offset:offset + n].view_as(p).to(p.dtype, copy=True)
            else:
                p.grad = None
            offset += n

    # -- state -------------------------------------------------------------

    @torch.no_grad()
    def sync_from_primary(self, tensors: Iterable[torch.Tensor]) -> float:
        """Overwrite every rank's ``tensors`` with rank 0's; returns the
        largest difference this rank had (0.0 on rank 0), as a drift check."""
        worst = 0.0
        for t in tensors:
            if not t.is_floating_point():
                continue
            ref = t.detach().clone()
            self.dist.broadcast(ref, src=0)
            if not self.primary:
                worst = max(worst, float((t - ref).abs().max()) if t.numel() else 0.0)
                t.copy_(ref)
        return worst

    def decide(self, flag: bool) -> bool:
        """Rank 0's ``flag``, on every rank (e.g. "the time budget is spent")."""
        t = torch.tensor([1 if flag else 0], device=self.device, dtype=torch.int32)
        self.dist.broadcast(t, src=0)
        return bool(t.item())

    def total(self, value: float) -> float:
        t = torch.tensor([float(value)], device=self.device, dtype=torch.float64
                         if self.device.type != "mps" else torch.float32)
        self.dist.all_reduce(t)
        return float(t.item())

    def barrier(self) -> None:
        self.dist.barrier()

    def close(self) -> None:
        if self.dist.is_initialized():
            self.dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch(argv: Sequence[str], world: int, log_dir: Optional[str | Path] = None,
           python: Optional[str] = None, command: Optional[Sequence[str]] = None) -> int:
    """Run ``python -m iridium <argv>`` once per rank; returns the exit code.

    Rank 0 keeps this process's stdout, so its log is the notebook's log;
    other ranks write to ``<log_dir>/rank<N>.log`` (stderr is always shown).
    If any rank exits non-zero, the rest are terminated: a rank that died
    would otherwise leave its peers waiting in ``all_reduce`` forever.
    """
    import time

    if world < 2:
        raise ValueError("launch needs at least two ranks")
    env = dict(os.environ)
    env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(_free_port()), WORLD_SIZE=str(world))
    env.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or world) // world)))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = list(command) if command else [python or sys.executable, "-m", "iridium", *argv]
    procs, logs = [], []
    for rank in range(world):
        out = None
        if rank and log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            out = open(Path(log_dir) / f"rank{rank}.log", "a", encoding="utf-8")
            logs.append(out)
        procs.append(subprocess.Popen(cmd, env={**env, "RANK": str(rank), "LOCAL_RANK": str(rank)},
                                      stdout=out))
    code = 0
    try:
        while procs:
            for p in list(procs):
                rc = p.poll()
                if rc is None:
                    continue
                procs.remove(p)
                if rc != 0:
                    code = code or rc
                    for other in procs:
                        other.terminate()
            time.sleep(0.5)
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        code = 130
    finally:
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
        for f in logs:
            f.close()
    return code
