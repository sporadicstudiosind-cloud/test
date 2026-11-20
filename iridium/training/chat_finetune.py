"""Fine-tune the bundled nano checkpoint on human-written conversations.

This is a bounded, reproducible training entry point, not a pretrained chat
checkpoint. The source conversation split is content-hashed before sampling;
held-out loss and generated replies are saved for review after training.

    python -m iridium.training.chat_finetune --device cuda --steps 200

Install Hugging Face ``datasets`` before using the default Dolly/OASST data.
The output contains a resumable trainer checkpoint and a smaller inference-only
checkpoint that ``python -m iridium chat`` can load safely.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from ..codecs.spans import MODALITY_INDEX, Sample
from ..config import IridiumConfig
from ..data.chat_corpus import CHAT_SOURCES, chat_items, chat_licence_notice
from ..model.iridium1 import Iridium1
from ..runtime.chat import ASSISTANT, STOP_IDS
from ..runtime.checkpoint_compat import load_compatible
from ..runtime.device import detect, generator_for, inference_autocast
from ..runtime.generate import generate
from .datasets import BatchLoader, Corpus
from .losses import LossWeights
from .trainer import TrainConfig, Trainer


DEFAULT_INIT = Path(__file__).resolve().parents[2] / "serve" / "weights" / "nano-phase1-fp16.pt"


def load_initial_model(path: Path) -> tuple[Iridium1, dict]:
    """Load state and manifest without unpickling arbitrary checkpoint classes."""
    from torch.serialization import safe_globals
    from torch.torch_version import TorchVersion

    with safe_globals([TorchVersion]):
        blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or not isinstance(blob.get("state_dict"), dict):
        raise ValueError("checkpoint must contain a state_dict and manifest")
    manifest = blob.get("manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("model_config"), dict):
        raise ValueError("checkpoint manifest has no model_config")
    model = Iridium1(IridiumConfig.from_dict(manifest["model_config"]))
    load_compatible(model, blob["state_dict"])
    return model, manifest


def heldout_chat_loss(model: Iridium1, corpus: Corpus, *, device: str,
                      batch_size: int, n_loops: int, info) -> dict[str, float | int]:
    """Assistant target NLL, weighted by target count across held-out batches."""
    loader = BatchLoader(corpus, model.cfg.codecs, batch_size=batch_size,
                         device=device, max_length=model.cfg.max_seq_len)
    text_id, control_id = MODALITY_INDEX["text"], MODALITY_INDEX["control"]
    n_targets = 0
    total_nll = 0.0
    model.eval()
    with torch.inference_mode(), inference_autocast(info):
        for batch, _ in loader.batches():
            target = batch.valid[:, 1:] & batch.supervised[:, 1:]
            target &= (batch.modality[:, 1:] == text_id) | (batch.modality[:, 1:] == control_id)
            count = int(target.sum().item())
            losses, _ = model.losses(batch, n_loops=n_loops,
                                     generator=generator_for(device, 0))
            total_nll += float(losses["text"].detach()) * count
            n_targets += count
    if n_targets == 0:
        raise ValueError("held-out conversations have no assistant targets")
    nats = total_nll / n_targets
    return {"assistant_targets": n_targets, "nats_per_target": nats,
            "bits_per_target": nats / math.log(2)}


def assistant_prompt(item) -> Sample:
    """Keep every known turn through the final assistant role marker."""
    markers = [index for index, span in enumerate(item.sample.spans)
               if span.modality == "control" and len(span) == 1
               and int(span.payload[0]) == ASSISTANT]
    if not markers:
        raise ValueError("chat item has no assistant role marker")
    # These spans are all observed context for generation, including earlier
    # assistant replies in a multi-turn conversation.
    spans = [replace(span, supervised=False) for span in item.sample.spans[:markers[-1] + 1]]
    return Sample(spans, meta=dict(item.sample.meta))


def review_responses(model: Iridium1, corpus: Corpus, *, n: int, n_loops: int,
                     max_new_tokens: int, info) -> list[dict]:
    """Save free-running answers beside references; no exact-match chat grade."""
    reviews = []
    model.eval()
    with torch.inference_mode(), inference_autocast(info):
        for index, item in enumerate(corpus.items[:n]):
            prompt = assistant_prompt(item)
            available = model.cfg.max_seq_len - len(prompt)
            if available < 1:
                continue
            output = generate(model, prompt,
                              max_new_tokens=min(max_new_tokens, available),
                              temperature=0.0, stop_ids=STOP_IDS,
                              n_loops=n_loops, text_only=True)
            reviews.append({"source": item.truth["source"],
                            "prompt": item.prompt, "reference": item.answer,
                            "generated": output.text, "stopped": output.stopped})
    return reviews


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", type=Path, default=DEFAULT_INIT)
    ap.add_argument("--out", type=Path, default=Path("runs/chat-nano"))
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda[:N], or rocm")
    ap.add_argument("--precision", default="auto", help="auto, fp32, fp16, or bf16")
    ap.add_argument("--sources", choices=("both", "dolly", "oasst"), default="both")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accumulate", type=int, default=4)
    ap.add_argument("--train-items", type=int, default=2000)
    ap.add_argument("--eval-items", type=int, default=40)
    ap.add_argument("--eval-responses", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--max-bytes", type=int, default=768)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if min(args.steps, args.batch_size, args.accumulate,
           args.train_items, args.eval_items, args.max_bytes) < 1:
        ap.error("training, evaluation, and byte budgets must be positive")
    if args.eval_responses < 0 or args.max_new_tokens < 1:
        ap.error("eval-responses must be nonnegative and max-new-tokens positive")

    info = detect(args.device, args.precision)
    model, initial_manifest = load_initial_model(args.init)
    mix = ({"dolly": 0.5, "oasst": 0.5} if args.sources == "both"
           else {args.sources: 1.0})
    train = Corpus(chat_items(args.train_items, mix=mix, seed=args.seed,
                              max_bytes=args.max_bytes, split="train"), "train")
    heldout = Corpus(chat_items(args.eval_items, mix=mix, seed=args.seed + 1,
                                max_bytes=args.max_bytes, split="test"), "test")
    if len(train) != args.train_items or len(heldout) != args.eval_items:
        raise RuntimeError(f"conversation source underfilled quota: "
                           f"train={len(train)}/{args.train_items}, "
                           f"test={len(heldout)}/{args.eval_items}")

    n_loops = int(initial_manifest.get("train_config", {}).get("n_loops", 1))
    cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size,
                      accumulate=args.accumulate, lr=args.lr,
                      warmup=min(10, max(0, args.steps // 10)),
                      n_loops=n_loops, seed=args.seed, label="chat-nano",
                      log_every=max(1, args.steps // 20),
                      precision=info.precision,
                      checkpoint_every=max(1, args.steps // 4))
    trainer = Trainer(model, train, cfg, LossWeights(router_balance=0.05),
                      out_dir=args.out, device=info.device)
    trainer.data_info = {"source_specs": [CHAT_SOURCES[key].as_dict() for key in mix],
                         "source_mix": mix, "train_items": len(train),
                         "heldout_items": len(heldout),
                         "split_method": "content hash", "base_checkpoint": str(args.init)}
    print(info.describe(), flush=True)
    before = heldout_chat_loss(model, heldout, device=info.device,
                               batch_size=args.batch_size, n_loops=n_loops, info=info)
    print(f"held-out before: {before}", flush=True)
    trainer.train()
    after = heldout_chat_loss(model, heldout, device=info.device,
                              batch_size=args.batch_size, n_loops=n_loops, info=info)
    reviews = review_responses(model, heldout, n=args.eval_responses,
                               n_loops=n_loops, max_new_tokens=args.max_new_tokens,
                               info=info)
    evaluation = {"before": before, "after": after,
                  "response_review": reviews, "licence_notice": chat_licence_notice(mix)}
    args.out.mkdir(parents=True, exist_ok=True)
    trainer.save("final", extra={"chat_evaluation": evaluation})
    (args.out / "chat-evaluation.json").write_text(
        json.dumps(evaluation, indent=2), encoding="utf-8")

    # A portable state-only checkpoint for inference. Convert TorchVersion
    # and any other scalar subclasses in the manifest to plain JSON types.
    manifest = json.loads(json.dumps(trainer.manifest(
        {"chat_evaluation": evaluation}), default=str))
    torch.save({"state_dict": {name: value.detach().cpu()
                               for name, value in model.state_dict().items()},
                "manifest": manifest}, args.out / "chat-inference.pt")
    print(f"saved {args.out / 'chat-inference.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
