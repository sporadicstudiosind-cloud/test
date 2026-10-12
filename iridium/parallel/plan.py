"""4-D parallelism planner and communication cost model.

Takes a rung of the configuration ladder and a cluster description, returns a
partitioning, a per-GPU memory budget, a per-token communication bill, and a
verdict. It imports nothing from torch, so a 39-trillion-parameter layout can
be costed on a laptop.

What it is for
--------------
The source plan asserted a topology (TP=8, PP=4, EP=32, CP=8 on 1,024 B200s)
and asserted that it clears the "all-to-all latency wall". Neither claim was
computed. This module computes both, and two of the numbers come out badly:

**F-04 · Training memory.** The plan put the 8.8 T configuration's BF16
training on 1,024 H200s. Weights alone are 18.1 TB; mixed-precision Adam needs
about 16 bytes per parameter — BF16 weights and gradients plus an FP32 master
copy and two FP32 moments — which is 145 TB. 1,024 H200s hold 144 TB *in
total*, leaving nothing for activations, and the plan's own table lists the
same 1,024 GPUs as holding only the 17.58 TB of weights. The optimizer state
is missing from the accounting.

**F-05 · The bridge broadcast, not the dispatch, is the wall.** Dispatching a
token's hidden state to ``top_k`` stacks is small — a few tens of kilobytes.
Broadcasting the core's stage-I states so that every active stack can
cross-attend to the stream is ``T * d`` bytes *per active stack*, which at a
long context dominates the dispatch by three orders of magnitude. Planning
around the dispatch cost and ignoring the broadcast gets the bottleneck wrong.
``bridge_strategy`` costs the three ways out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from ..config import IridiumConfig, get_config

TB = 1e12
GB = 1e9


@dataclass(frozen=True)
class Accelerator:
    name: str
    memory_bytes: float
    dense_flops: float          # achievable, at the serving precision
    nvlink_bytes_per_s: float   # intra-node, per GPU, bidirectional
    fabric_bytes_per_s: float   # inter-node, per GPU

    @classmethod
    def b200(cls) -> "Accelerator":
        # 192 GB HBM3e; NVLink 5 at 1.8 TB/s per GPU. Dense FP8 throughput is
        # quoted per-vendor with and without structured sparsity; the sparse
        # figure is not achievable on dense GEMMs, so the dense number is used
        # and further derated by the MFU passed to the planner.
        return cls("B200", 192 * GB, 4.5e15, 1.8e12, 100e9)

    @classmethod
    def h200(cls) -> "Accelerator":
        return cls("H200", 141 * GB, 2.0e15, 900e9, 50e9)


@dataclass(frozen=True)
class Cluster:
    accelerator: Accelerator
    n_gpus: int
    gpus_per_node: int = 8
    mfu: float = 0.35

    @property
    def n_nodes(self) -> int:
        return math.ceil(self.n_gpus / self.gpus_per_node)

    @property
    def total_memory(self) -> float:
        return self.n_gpus * self.accelerator.memory_bytes

    @property
    def usable_memory(self) -> float:
        """Framework overhead, fragmentation and workspace are not free."""
        return 0.85 * self.total_memory


@dataclass(frozen=True)
class Partition:
    tensor: int
    pipeline: int
    stack: int
    data: int
    context: int

    @property
    def product(self) -> int:
        return self.tensor * self.pipeline * self.stack * self.data * self.context

    def describe(self) -> str:
        return (
            f"TP={self.tensor} PP={self.pipeline} SP={self.stack} "
            f"DP={self.data} CP={self.context} -> {self.product} GPUs"
        )


@dataclass
class Plan:
    config_name: str
    cluster: Cluster
    partition: Partition
    memory: dict[str, float]
    communication: dict[str, float]
    compute: dict[str, float]
    verdict: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"{self.config_name} on {self.cluster.n_gpus} x {self.cluster.accelerator.name}",
            f"  {self.partition.describe()}",
            "  memory per GPU",
        ]
        for key, value in self.memory.items():
            lines.append(f"    {key:<26} {value / GB:10.2f} GB")
        lines.append("  communication")
        for key, value in self.communication.items():
            unit = "MB/token" if "per_token" in key else ""
            scale = 1e6 if "per_token" in key else 1.0
            lines.append(f"    {key:<26} {value / scale:10.3f} {unit}")
        lines.append("  compute")
        for key, value in self.compute.items():
            lines.append(f"    {key:<26} {value:10.3f}")
        lines.append(f"  verdict: {self.verdict['status']} - {self.verdict['reason']}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def suggest_partition(cfg: IridiumConfig, cluster: Cluster) -> Partition:
    """A defensible default: fill the node with TP, give each stack a shard."""
    tp = min(cluster.gpus_per_node, max(1, cfg.core.n_kv_heads))
    remaining = max(cluster.n_gpus // tp, 1)
    sp = min(cfg.stacks.n_stacks, remaining)
    remaining = max(remaining // sp, 1)
    pp = min(4, remaining)
    remaining = max(remaining // pp, 1)
    cp = min(8, remaining)
    dp = max(remaining // cp, 1)
    return Partition(tensor=tp, pipeline=pp, stack=sp, data=dp, context=cp)


def plan(
    rung: str | IridiumConfig,
    cluster: Cluster,
    partition: Optional[Partition] = None,
    *,
    training: bool = False,
    bits_per_param: float = 4.25,
    context_tokens: int = 8192,
    batch_tokens: int = 2048,
    bridge_strategy: Literal["broadcast", "cache_kv", "colocate"] = "cache_kv",
) -> Plan:
    cfg = get_config(rung) if isinstance(rung, str) else rung
    partition = partition or suggest_partition(cfg, cluster)
    notes: list[str] = []

    # -- memory ----------------------------------------------------------
    core_bytes = cfg.core.params * (2.0 if training else bits_per_param / 8.0)
    stack_bytes = cfg.stacks.params * (2.0 if training else bits_per_param / 8.0)
    codec_bytes = sum(cfg.codecs.params(cfg.core.d_model).values()) * 2.0

    core_gpus = partition.tensor * partition.pipeline
    stack_gpus_each = max(cluster.n_gpus // max(cfg.stacks.n_stacks, 1), 1)
    per_gpu_core = core_bytes / max(core_gpus, 1)
    per_gpu_stack = (stack_bytes / max(cfg.stacks.n_stacks, 1)) / stack_gpus_each

    optimizer = 0.0
    if training:
        # BF16 weights + BF16 grads + FP32 master + FP32 m + FP32 v = 16 B/param.
        optimizer = (cfg.n_params * 16.0) / cluster.n_gpus
        notes.append(
            "training memory includes 16 B/param of mixed-precision Adam state; "
            "omitting it understates the requirement by 8x"
        )

    kv_per_token = cfg.kv_bytes_per_token(loops=cfg.router.max_loops, bytes_per_element=1)
    kv_bytes = kv_per_token * context_tokens / max(partition.context, 1)
    bridge_state_bytes = cfg.core.d_model * context_tokens  # FP8 stage-I states
    activations = (
        batch_tokens * cfg.core.d_model * 2 * cfg.core.n_layers / max(partition.tensor, 1)
    )

    memory = {
        "core_weights": per_gpu_core,
        "stack_weights": per_gpu_stack,
        "codecs": codec_bytes / max(core_gpus, 1),
        "optimizer_state": optimizer,
        "kv_cache": kv_bytes,
        "bridge_states": bridge_state_bytes,
        "activations": activations,
    }
    memory["total"] = sum(memory.values())

    # -- communication ---------------------------------------------------
    d = cfg.core.d_model
    k = cfg.router.top_k
    loops = cfg.router.max_loops
    dispatch = 2 * k * d * loops                       # out and back, 1 B/elem
    tp_allreduce = 2 * d * cfg.core.n_layers * loops * (partition.tensor - 1) / max(partition.tensor, 1)

    if bridge_strategy == "broadcast":
        active_stacks = min(cfg.stacks.n_stacks, k * batch_tokens)
        bridge = active_stacks * context_tokens * d / max(batch_tokens, 1)
        notes.append(
            "broadcast: every active stack receives the whole stage-I history; "
            "this is the term that dominates, not the dispatch"
        )
    elif bridge_strategy == "cache_kv":
        # Each stack keeps its own projected K/V for the stream: paid once per
        # token per stack it visits, then read locally forever after.
        bridge = k * 2 * cfg.stacks.d_kv * cfg.stacks.n_cross_layers
        notes.append(
            "cache_kv: bridge K/V cached at the stack; cost is per new token, "
            "not per chunk, at the price of the residency in the memory table"
        )
    else:
        bridge = 0.0
        notes.append(
            "colocate: stacks share a node with a core shard; no fabric cost "
            "and no stack-level elasticity"
        )

    per_token_bytes = dispatch + bridge + tp_allreduce
    link = cluster.accelerator.fabric_bytes_per_s
    # Throughput basis, not latency: every GPU has its own fabric port, so the
    # cluster moves n_gpus * link bytes per second in aggregate, just as it
    # executes n_gpus * achievable FLOPs per second. Comparing a single link's
    # latency against the cluster's aggregate compute is the arithmetic that
    # makes any routed design look communication-bound by two orders of
    # magnitude, and it is not a like-for-like comparison.
    aggregate_link = link * cluster.n_gpus
    comm_seconds = per_token_bytes / aggregate_link

    # -- compute ---------------------------------------------------------
    flops_lo, flops_hi = cfg.flops_per_token()
    attention_flops = (
        2 * 2 * cfg.core.d_model * context_tokens * cfg.core.n_layers * loops
    )
    achievable = cluster.accelerator.dense_flops * cluster.mfu
    compute_seconds = (flops_hi + attention_flops) / (achievable * cluster.n_gpus)
    ratio = comm_seconds / max(compute_seconds, 1e-30)

    compute = {
        "gflop_per_token_min": flops_lo / 1e9,
        "gflop_per_token_max": flops_hi / 1e9,
        "attention_gflop_per_token": attention_flops / 1e9,
        "compute_us_per_token_aggregate": compute_seconds * 1e6,
        "comm_us_per_token_aggregate": comm_seconds * 1e6,
        "comm_to_compute_ratio": ratio,
        "tokens_per_second_overlapped": 1.0 / max(compute_seconds, comm_seconds, 1e-30),
        "tokens_per_second_serial": 1.0 / max(compute_seconds + comm_seconds, 1e-30),
    }
    communication = {
        "dispatch_bytes_per_token": dispatch,
        "bridge_bytes_per_token": bridge,
        "tensor_parallel_bytes_per_token": tp_allreduce,
        "total_bytes_per_token": per_token_bytes,
        "fabric_bytes_per_s": link,
    }

    fits = memory["total"] <= cluster.accelerator.memory_bytes
    if not fits:
        status, reason = "infeasible", (
            f"{memory['total'] / GB:.1f} GB per GPU exceeds "
            f"{cluster.accelerator.memory_bytes / GB:.0f} GB"
        )
    elif ratio > 1.0:
        status, reason = "communication-bound", (
            f"communication is {ratio:.1f}x compute; the fabric decides the "
            f"throughput, not the accelerators"
        )
    elif ratio > 0.25:
        status, reason = "feasible", (
            f"communication is {ratio:.2f}x compute; overlap is required"
        )
    else:
        status, reason = "feasible", f"compute-bound ({ratio:.3f}x)"

    return Plan(
        config_name=cfg.name, cluster=cluster, partition=partition,
        memory=memory, communication=communication, compute=compute,
        verdict={"status": status, "reason": reason, "fits": fits,
                 "comm_to_compute": ratio},
        notes=notes,
    )


def minimum_gpus_for_training(cfg: IridiumConfig, accelerator: Accelerator,
                              bytes_per_param: float = 16.0) -> int:
    """GPUs needed to hold weights + gradients + optimizer state, nothing else."""
    need = cfg.n_params * bytes_per_param
    return math.ceil(need / (0.85 * accelerator.memory_bytes))


if __name__ == "__main__":                               # pragma: no cover
    for rung in ("small", "base"):
        cfg = get_config(rung)
        cl = Cluster(Accelerator.b200(), 1024)
        for strategy in ("broadcast", "cache_kv"):
            print(plan(rung, cl, bridge_strategy=strategy).render())
            print()
