"""Smoke test for the 1.00 B `test1b` rung.

Answers, with measurements rather than assertions:

* does a billion-parameter instantiation of this architecture run at all;
* does the parity gate still hold at this size (it is a property of the
  mechanism, not the scale, so it had better);
* what does it actually cost per token on a CPU;
* does the router spread traffic over five general superstacks;
* does it train — checked by LoRA adapters, because fp32 Adam over 1.00 B
  parameters needs ~17 GB and this machine has 15 GB. That is a real
  constraint, not a shortcut: the base weights stay frozen and only the
  adapters move, which is exactly the continual-learning path in
  `iridium/training/continual.py`.

    python experiments/run_test1b.py --out experiments/results/test1b.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from iridium.codecs.bank import TensorBatch, continuous_dims
from iridium.codecs.spans import Sample, Span, collate, text_span
from iridium.config import get_config
from iridium.model.iridium1 import Iridium1
from iridium.runtime.decode import run_atomic_chunked
from iridium.runtime.generate import generate
from iridium.training.continual import attach_lora, lora_parameters
from iridium.training.datasets import build_corpus
from iridium.training.losses import LossWeights, combine


def measure(model, cfg, threads: int) -> dict:
    torch.set_num_threads(threads)
    dims = continuous_dims(cfg.codecs)
    out: dict = {}

    # -- forward throughput ------------------------------------------------
    rng = np.random.default_rng(0)
    sample = Sample([
        text_span("simulate the channel and report the depth", offset=16),
        Span("field", rng.normal(size=(4, dims["field"])).astype(np.float32),
             grid=(2, 2)),
        text_span("h=", offset=16),
    ])
    batch = TensorBatch(collate([sample, sample], dims))
    n_tokens = int(batch.modality.numel())

    with torch.no_grad():
        model(batch, n_loops=1)                       # warm
        timings = {}
        for loops in (1, 2, 3):
            t0 = time.time()
            result = model(batch, n_loops=loops)
            dt = time.time() - t0
            timings[f"loops_{loops}"] = {
                "seconds": dt,
                "tokens_per_second": n_tokens / dt,
                "expected_loops": float(result.expected_loops.mean()),
                "mean_focus": float(result.decisions[0].focus.mean()),
            }
        out["forward"] = {"tokens": n_tokens, **timings}

        stats = result.stats["stack_stats"][0]
        out["routing"] = {
            "tokens_per_stack": stats["per_stack_tokens"],
            "expected_depth_per_stack": [
                round(d, 3) for d in stats["per_stack_expected_depth"]
            ],
            "grid_intact_fraction": stats["grid_intact_fraction"],
            "specializations": list(cfg.stacks.specializations),
        }

    # -- the parity gate, at 1 B ------------------------------------------
    with torch.no_grad():
        reference = model(batch, n_loops=1).hidden
        chunked = run_atomic_chunked(model, batch, chunk=1, n_loops=1)
        delta = float((reference - chunked).abs().max())
    out["cache_parity"] = {
        "max_abs_delta_fp32": delta,
        "bit_exact": bool(torch.equal(reference, chunked)),
        "note": "fp32 here; the float64 gate in tests/integration runs at `tiny`",
    }

    # -- generation --------------------------------------------------------
    with torch.no_grad():
        t0 = time.time()
        gen = generate(model, Sample([text_span("CHAN|S=0.0020|n=0.030|q=3.00|h=",
                                                offset=16)]),
                       max_new_tokens=8, n_loops=1)
        dt = time.time() - t0
    out["generation"] = {
        "seconds": dt,
        "tokens": len(gen.ids),
        "seconds_per_token": dt / max(len(gen.ids), 1),
        "text": gen.text,
        "mean_focus": gen.mean_focus,
    }
    return out


@torch.no_grad()
def routing_census(model, cfg, n_items: int = 24, seed: int = 5) -> dict:
    """How traffic actually spreads over the five stacks, over real items."""
    from iridium.training.datasets import BatchLoader

    corpus = build_corpus(n_items, seed=seed, split="train")
    loader = BatchLoader(corpus, cfg.codecs, batch_size=2, seed=seed)
    counts = np.zeros(cfg.stacks.n_stacks)
    was_training = model.training
    model.eval()
    for batch, _ in loader.batches():
        out = model(batch, n_loops=1)
        index = out.decisions[0].stack_index[batch.valid]
        counts += np.bincount(index.reshape(-1).cpu().numpy(),
                              minlength=cfg.stacks.n_stacks)
    model.train(was_training)
    share = counts / max(counts.sum(), 1)
    nz = share[share > 0]
    return {
        "tokens_per_stack": counts.astype(int).tolist(),
        "share": [round(float(x), 4) for x in share],
        "stacks_used": int((counts > 0).sum()),
        "entropy_nats": float(-(nz * np.log(nz)).sum()) if nz.size else 0.0,
        "max_entropy_nats": float(np.log(cfg.stacks.n_stacks)),
    }


def lora_train(model, cfg, steps: int, threads: int, seed: int = 0) -> dict:
    """Train adapters and the router only. Reports why it must be adapters.

    The router gate is 3,840 parameters out of a billion and is the one part
    that *must* move: at initialization a fixed random gate gives every token
    the same top-2, so two stacks take all the traffic and three are dead
    weight. Whether the balance objective actually recovers from that at this
    size is a question, not an assumption - hence the census either side.
    """
    torch.set_num_threads(threads)
    total = sum(p.numel() for p in model.parameters())
    info = attach_lora(model, rank=8, alpha=16.0)
    for p in model.parameters():
        p.requires_grad_(False)
    adapters = lora_parameters(model)
    for p in adapters:
        p.requires_grad_(True)
    n_adapter = sum(p.numel() for p in adapters)

    router_params = [p for n, p in model.named_parameters() if n.startswith("router.")]
    for p in router_params:
        p.requires_grad_(True)
    trainable = adapters + router_params
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    corpus = build_corpus(max(steps * 4, 64), seed=seed, split="train")
    from iridium.training.datasets import BatchLoader

    loader = BatchLoader(corpus, cfg.codecs, batch_size=2, seed=seed)
    weights = LossWeights()
    history = []
    t0 = time.time()
    stream = iter(loader.batches())
    for step in range(steps):
        try:
            batch, _ = next(stream)
        except StopIteration:
            stream = iter(loader.batches())
            batch, _ = next(stream)
        losses, _ = model.losses(batch, n_loops=1)
        loss, report = combine(losses, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        history.append({"step": step, "total": report["total"],
                        "text": report.get("text", 0.0)})
        print(f"  lora step {step:3d}  total={report['total']:.4f} "
              f"text={report.get('text', 0):.4f}  {time.time() - t0:.0f}s",
              flush=True)

    after = routing_census(model, cfg)
    first = np.mean([h["total"] for h in history[: max(len(history) // 4, 1)]])
    last = np.mean([h["total"] for h in history[-max(len(history) // 4, 1):]])
    return {
        "steps": steps,
        "seconds": time.time() - t0,
        "adapted_modules": info["adapted_modules"],
        "adapter_parameters": n_adapter,
        "adapter_fraction": n_adapter / total,
        "frozen_parameters": total,
        "loss_first_quarter": float(first),
        "loss_last_quarter": float(last),
        "loss_fell": bool(last < first),
        "routing_after": after,
        "router_parameters": sum(p.numel() for p in router_params),
        "history": history,
        "memory_argument": {
            "fp32_weights_gb": total * 4 / 1e9,
            "full_adam_state_gb": total * 16 / 1e9,
            "lora_adam_state_gb": n_adapter * 16 / 1e9,
            "machine_ram_gb": 15.0,
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung", default="test1b")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--lora-steps", type=int, default=12)
    ap.add_argument("--out", default="experiments/results/test1b.json")
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    cfg = get_config(args.rung)
    print(f"building {cfg.name} ...", flush=True)
    t0 = time.time()
    model = Iridium1(cfg).eval()
    build_seconds = time.time() - t0
    actual = sum(p.numel() for p in model.parameters())
    print(f"  {actual:,} parameters in {build_seconds:.1f}s", flush=True)

    payload = {
        "config": cfg.name,
        "parameters": {
            "total": actual,
            "formula": cfg.n_params,
            "delta": actual - cfg.n_params,
            "parts": cfg.parameter_parts(),
            "active_min": cfg.active_parameters()[0],
            "active_max": cfg.active_parameters()[1],
            "gflops_per_token_min": cfg.flops_per_token()[0] / 1e9,
            "gflops_per_token_max": cfg.flops_per_token()[1] / 1e9,
        },
        "geometry": {
            "core_layers": cfg.core.n_layers,
            "core_d_model": cfg.core.d_model,
            "n_superstacks": cfg.stacks.n_stacks,
            "superstack_layers": cfg.stacks.n_layers,
            "top_k": cfg.router.top_k,
            "max_loops": cfg.router.max_loops,
            "specializations": list(cfg.stacks.specializations),
        },
        "build_seconds": build_seconds,
        "torch": torch.__version__,
        "threads": args.threads,
    }
    print("measuring ...", flush=True)
    payload.update(measure(model, cfg, args.threads))
    if args.lora_steps:
        print("routing census before ...", flush=True)
        before = routing_census(model, cfg)
        print("  ", before["tokens_per_stack"], flush=True)
        print("lora + router training ...", flush=True)
        payload["lora"] = lora_train(model, cfg, args.lora_steps, args.threads)
        payload["lora"]["routing_before"] = before

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "lora"},
                     indent=2, default=float)[:2500])
    print("written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
