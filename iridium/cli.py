"""``python -m iridium <command>`` — one entry point for the whole system."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_DEFAULT_CHAT_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "serve" / "weights" / "nano-phase1-fp16.pt"
)


def cmd_ladder(args) -> int:
    from .config import ladder_table
    print(ladder_table())
    return 0


def cmd_report(args) -> int:
    from .config import get_config
    cfg = get_config(args.rung)
    print(cfg.report().render())
    print()
    lo, hi = cfg.flops_per_token()
    print(f"  forward FLOPs/token   {lo / 1e9:,.2f} - {hi / 1e9:,.2f} GFLOP")
    print(f"  KV bytes/token        {cfg.kv_bytes_per_token(cfg.router.max_loops):,}")
    for bits, label in ((16.0, "BF16"), (8.0, "FP8"), (4.25, "MXFP4")):
        print(f"  weights @ {label:<6}      {cfg.weight_bytes(bits) / 1e12:,.4f} TB")
    if args.verify:
        import torch
        from .model.iridium1 import Iridium1
        actual = sum(p.numel() for p in Iridium1(cfg).parameters())
        print(f"  instantiated          {actual:,} (delta {actual - cfg.n_params:+,})")
    return 0


def cmd_plan(args) -> int:
    from .parallel.plan import Accelerator, Cluster, minimum_gpus_for_training, plan
    from .config import get_config
    accel = {"b200": Accelerator.b200, "h200": Accelerator.h200}[args.accelerator]()
    cluster = Cluster(accel, args.gpus, mfu=args.mfu)
    result = plan(
        args.rung, cluster, training=args.training,
        context_tokens=args.context, bridge_strategy=args.bridge,
    )
    print(result.render())
    if args.training:
        cfg = get_config(args.rung)
        print(f"  minimum {accel.name}s for weights+grads+Adam: "
              f"{minimum_gpus_for_training(cfg, accel):,}")
    return 0


def cmd_quant(args) -> int:
    import torch
    from .quant.mxfp4 import bits_per_param, measure
    from .config import get_config
    cfg = get_config(args.rung)
    print(f"{cfg.name}: {cfg.n_params:,} parameters")
    print(f"  MXFP4 block {args.block}: {bits_per_param(args.block)} bits/param")
    for bits, label in ((16.0, "BF16"), (8.0, "FP8 E4M3"),
                        (bits_per_param(args.block), "MXFP4")):
        print(f"  {label:<12} {cfg.weight_bytes(bits) / 1e12:10.4f} TB")
    torch.manual_seed(0)
    sample = torch.randn(2048, 512) * 0.02
    for mode in ("mxfp4", "fp8", "bf16"):
        s = measure(sample, args.block, mode)
        print(f"  {mode:<6} SQNR {s.sqnr_db:6.2f} dB   relative L2 {s.relative_l2:.5f}")
    return 0


def cmd_waterfall(args) -> int:
    from .physics.shallow_water import Channel, intervention
    ch = Channel(length=args.length, n_cells=args.cells, slope=args.slope,
                 manning=args.manning)
    result = intervention(
        ch, args.q, args.q * args.factor, max_time=args.max_time, tol=1e-8,
        initial_depth=args.initial_depth,
    )
    print(json.dumps(result, indent=2, default=float))
    return 0


def cmd_fluid(args) -> int:
    import numpy as np
    from .physics.fluid2d import NavierStokes2D, taylor_green
    solver = NavierStokes2D(args.n, nu=args.nu)
    state = taylor_green(args.n, 0.0, args.nu)
    dt = args.dt
    for _ in range(int(args.time / dt)):
        state = solver.step(state, dt)
    exact = taylor_green(args.n, state.t, args.nu)
    err = float(np.sqrt(np.mean((state.u - exact.u) ** 2 + (state.v - exact.v) ** 2)))
    ref = float(np.sqrt(np.mean(exact.u ** 2 + exact.v ** 2)))
    print(json.dumps({
        "t": state.t, "relative_l2_vs_exact": err / ref,
        **solver.diagnostics(state)
    }, indent=2))
    return 0


def cmd_serve(args) -> int:
    import torch
    from .codecs.spans import Sample, text_span
    from .config import get_config
    from .model.iridium1 import Iridium1
    from .runtime.scheduler import SchedulerPolicy
    from .runtime.service import IridiumService
    from .training.trainer import load_checkpoint

    if args.checkpoint:
        model, _ = load_checkpoint(args.checkpoint)
    else:
        torch.manual_seed(0)
        model = Iridium1(get_config(args.rung)).eval()
    service = IridiumService(model, SchedulerPolicy(token_budget=args.budget))
    for i, prompt in enumerate(args.prompt or ["hello from stream one"]):
        service.admit(f"stream-{i}", f"owner-{i % 2}", Sample([text_span(prompt)]))
    for report in service.run(max_ticks=args.ticks):
        print(report.summary())
    print(json.dumps(service.status(), indent=2))
    return 0


def cmd_generate(args) -> int:
    from .codecs.spans import Sample, text_span
    from .runtime.generate import generate
    from .training.tokenizer_bridge import tokenizer_from_manifest
    from .training.trainer import load_checkpoint
    model, manifest = load_checkpoint(args.checkpoint)
    tokenizer = tokenizer_from_manifest(manifest)
    sample = Sample([text_span(args.prompt, offset=16, tokenizer=tokenizer)])
    out = generate(model, sample, max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature, tokenizer=tokenizer)
    print(json.dumps({
        "prompt": args.prompt, "output": out.text, "stopped": out.stopped,
        "mean_focus": out.mean_focus,
    }, indent=2))
    return 0


def _load_chat_checkpoint(path: Path, device: str):
    """Load inference weights without executing arbitrary checkpoint pickles.

    The bundled phase-1 checkpoint serialised ``torch.__version__`` as a
    ``TorchVersion`` value. It is the only extra type accepted here. Training
    resume checkpoints with optimizer/RNG objects should be exported as an
    inference-only state_dict + plain manifest before being used for chat.
    """
    import torch
    from torch.serialization import safe_globals
    from torch.torch_version import TorchVersion

    from .config import IridiumConfig
    from .model.iridium1 import Iridium1
    from .runtime.checkpoint_compat import load_compatible

    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {path}. Train/export a chat checkpoint or "
            "use the bundled serve/weights/nano-phase1-fp16.pt"
        )
    try:
        with safe_globals([TorchVersion]):
            blob = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"cannot safely load {path}; chat requires an inference checkpoint "
            "containing only a state_dict and plain manifest"
        ) from exc
    if not isinstance(blob, dict) or not isinstance(blob.get("manifest"), dict) \
            or not isinstance(blob.get("state_dict"), dict):
        raise ValueError("checkpoint must contain a state_dict and manifest dict")
    manifest = blob["manifest"]
    if not isinstance(manifest.get("model_config"), dict):
        raise ValueError("checkpoint manifest is missing model_config")
    cfg = IridiumConfig.from_dict(manifest["model_config"])
    model = Iridium1(cfg)
    report = load_compatible(model, blob["state_dict"])
    if not report.exact:
        print(f"Checkpoint compatibility: {report.summary()}", file=sys.stderr)
    model.to(device).eval()
    return model, manifest


def _chat_status(model, manifest: dict, path: Path, device_info) -> None:
    current_params = sum(p.numel() for p in model.parameters())
    trained_params = (manifest.get("parameters") or {}).get("total")
    print(f"Model: {model.cfg.name} ({current_params:,} runtime parameters)")
    if isinstance(trained_params, int) and trained_params != current_params:
        print(f"Checkpoint recorded {trained_params:,} trained parameters")
    print(f"Checkpoint: {path}")
    print(f"Device: {device_info.describe()}")
    completed = manifest.get("completed_steps")
    configured = (manifest.get("train_config") or {}).get("steps")
    if isinstance(completed, int) and completed > 0:
        print(f"Training recorded: {completed:,} completed steps")
    elif isinstance(configured, int) and configured > 0:
        print(f"Training configured: {configured:,} steps; completion not recorded")
    else:
        print("Training steps: not recorded")
    scores = ((manifest.get("evaluation") or {}).get("interpolation") or {})
    graded = [r for r in scores.values() if isinstance(r, dict)
              and isinstance(r.get("accuracy"), (int, float))
              and isinstance(r.get("baseline"), (int, float))]
    if graded:
        above = sum(r["accuracy"] > r["baseline"] for r in graded)
        print(f"Recorded tasks above baseline: {above}/{len(graded)}; these are not chat benchmarks")
    print("Chat replies are experimental; conversational quality is not established.")


def _terminal_text(value: str) -> str:
    """Make arbitrary byte-model output safe for a terminal's text encoding."""
    visible = "".join(
        ch if ch in "\n\t" or ch.isprintable() else f"\\u{ord(ch):04x}"
        for ch in value
    )
    encoding = sys.stdout.encoding or "utf-8"
    return visible.encode(encoding, errors="backslashreplace").decode(encoding)


def cmd_chat(args) -> int:
    """Chat with an actual small checkpoint, with honest provenance and limits."""
    from .runtime.chat import ChatSession
    from .runtime.device import detect, inference_autocast

    if args.max_new_tokens < 1 or args.n_loops is not None and args.n_loops < 1:
        print("reply and loop budgets must be positive", file=sys.stderr)
        return 2
    if args.temperature < 0 or not 0 <= args.top_p <= 1 or args.top_k < 0:
        print("temperature/top-k must be nonnegative and top-p must be in [0, 1]",
              file=sys.stderr)
        return 2
    path = Path(args.checkpoint).expanduser().resolve()
    try:
        info = detect(args.device, args.precision)
        model, manifest = _load_chat_checkpoint(path, info.device)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"chat: {exc}", file=sys.stderr)
        return 2

    trained_loops = (manifest.get("train_config") or {}).get("n_loops")
    loops = args.n_loops if args.n_loops is not None else trained_loops
    from .training.tokenizer_bridge import tokenizer_from_manifest
    try:
        tokenizer = tokenizer_from_manifest(manifest)
    except ValueError as exc:
        print(f"chat: {exc}", file=sys.stderr)
        return 2
    session = ChatSession(
        model, tokenizer=tokenizer, system=args.system, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, max_new_tokens=args.max_new_tokens, n_loops=loops,
        seed=args.seed,
    )
    _chat_status(model, manifest, path, info)

    def reply_to(message: str) -> bool:
        try:
            with inference_autocast(info):
                reply = session.send(message)
        except ValueError as exc:
            print(f"chat: {exc}", file=sys.stderr)
            return False
        print(f"iridium> {_terminal_text(reply)}")
        return True

    if args.prompt is not None:
        return 0 if reply_to(args.prompt) else 2

    print("Type /help for commands; /exit or Ctrl-D to leave.")
    while True:
        try:
            message = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message.strip():
            continue
        if message.strip() in ("/exit", "/quit"):
            return 0
        if message.strip() == "/reset":
            session.reset()
            print("Conversation reset.")
            continue
        if message.strip() == "/help":
            print("/reset clears the conversation; /exit quits.")
            continue
        reply_to(message)


def cmd_evaluate(args) -> int:
    from .evaluation.harness import evaluate
    from .training.datasets import build_corpus
    from .training.trainer import load_checkpoint
    model, manifest = load_checkpoint(args.checkpoint)
    results = {}
    for split in ("test", "extrapolation"):
        corpus = build_corpus(args.items, seed=1234, split=split)
        results[split] = evaluate(model, corpus, args.per_family)
    print(json.dumps(results, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="iridium", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ladder", help="the whole configuration ladder")
    p.set_defaults(func=cmd_ladder)

    p = sub.add_parser("report", help="parameter and memory report for a rung")
    p.add_argument("rung", default="nano", nargs="?")
    p.add_argument("--verify", action="store_true",
                   help="instantiate the model and compare the counts")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("plan", help="4-D parallelism plan and cost model")
    p.add_argument("rung", default="base", nargs="?")
    p.add_argument("--gpus", type=int, default=1024)
    p.add_argument("--accelerator", choices=("b200", "h200"), default="b200")
    p.add_argument("--mfu", type=float, default=0.35)
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--training", action="store_true")
    p.add_argument("--bridge", choices=("broadcast", "cache_kv", "colocate"),
                   default="cache_kv")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("quant", help="quantization memory and error")
    p.add_argument("rung", default="base", nargs="?")
    p.add_argument("--block", type=int, default=32)
    p.set_defaults(func=cmd_quant)

    p = sub.add_parser("waterfall", help="the open-channel intervention")
    p.add_argument("--q", type=float, default=3.0)
    p.add_argument("--factor", type=float, default=2.0)
    p.add_argument("--slope", type=float, default=0.002)
    p.add_argument("--manning", type=float, default=0.030)
    p.add_argument("--length", type=float, default=100.0)
    p.add_argument("--cells", type=int, default=200)
    p.add_argument("--max-time", type=float, default=20000.0)
    p.add_argument("--initial-depth", type=float, default=0.5)
    p.set_defaults(func=cmd_waterfall)

    p = sub.add_parser("fluid", help="Taylor-Green validation of the NS solver")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--nu", type=float, default=0.05)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--time", type=float, default=0.5)
    p.set_defaults(func=cmd_fluid)

    p = sub.add_parser("serve", help="run the persistent instance on some streams")
    p.add_argument("--rung", default="nano")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--prompt", action="append")
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--ticks", type=int, default=8)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("generate", help="continue a prompt from a checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("prompt")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.0)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("chat", help="chat with the bundled small checkpoint or your own")
    p.add_argument("--checkpoint", default=str(_DEFAULT_CHAT_CHECKPOINT))
    p.add_argument("--prompt", help="send one prompt and exit; omit for interactive chat")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda[:N], rocm, or mps")
    p.add_argument("--precision", default="auto", choices=("auto", "fp32", "bf16", "fp16"))
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--n-loops", type=int, help="default: loop count recorded by the checkpoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--system", default=None, help="optional system message")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("presets", help="the ready-to-train presets and free-tier estimates")
    p.add_argument("name", nargs="?", help="show one preset in detail")
    p.set_defaults(func=cmd_presets)

    p = sub.add_parser("train", help="train a preset end to end, in rounds of fresh data")
    p.add_argument("--preset", required=True, help="see `iridium presets`")
    p.add_argument("--steps", type=int, default=None, help="override the preset's step budget")
    p.add_argument("--rounds", type=int, default=None, help="fresh-data rounds (bounds memory)")
    p.add_argument("--device", default=None, help="cpu | cuda | cuda:N (default: detect)")
    p.add_argument("--out", default="runs")
    p.add_argument("--init", default=None, help="start from a checkpoint (e.g. chat -> tools)")
    p.add_argument("--resume", default=None, help="continue from a round checkpoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="build on the meta device, audit and estimate; no network, no training")
    p.add_argument("--max-data", action="store_true",
                   help="also use opt-in chat sources (non-commercial or unclear terms); "
                        "the manifest records which")
    p.add_argument("--data", default=None,
                   help="train from shards written by `iridium data prepare` (memory-mapped)")
    _budget_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("data", help="tokenize a preset's language data to disk shards")
    p.add_argument("action", choices=["prepare"])
    p.add_argument("--preset", required=True)
    p.add_argument("--out", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-data", action="store_true")
    p.add_argument("--plan", action="store_true", help="print item counts and exit")
    _budget_args(p)
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("evaluate", help="graded accuracy on held-out splits")
    p.add_argument("checkpoint")
    p.add_argument("--items", type=int, default=300)
    p.add_argument("--per-family", type=int, default=24)
    p.set_defaults(func=cmd_evaluate)
    return ap


def cmd_presets(args) -> int:
    from .presets import get_preset, preset_table
    from .training.run_preset import describe
    if args.name:
        print(describe(get_preset(args.name)))
    else:
        print(preset_table())
    return 0


def _budgeted_preset(args):
    """The named preset, with ``--tokens`` / ``--tokens-per-param`` applied."""
    from .presets import get_preset, with_tokens
    preset = get_preset(args.preset)
    if getattr(args, "tokens_per_param", None):
        preset = with_tokens(preset, int(args.tokens_per_param * preset.config.n_params))
    elif getattr(args, "tokens", None):
        preset = with_tokens(preset, int(float(args.tokens)))
    return preset


def cmd_data(args) -> int:
    from .training.prepare import plan, prepare
    try:
        preset = _budgeted_preset(args)
    except KeyError as exc:
        print(f"data: {exc.args[0]}", file=sys.stderr)
        return 2
    if args.plan:
        for family, n in plan(preset).items():
            print(f"{family:<8} {n:>12,} items of up to {preset.window} tokens")
        return 0
    out = prepare(preset, args.out, seed=args.seed, max_data=args.max_data)
    print(f"shards written to {out}")
    return 0


def cmd_train(args) -> int:
    from .training import run_preset
    try:
        preset = _budgeted_preset(args)
    except KeyError as exc:
        print(f"train: {exc.args[0]}", file=sys.stderr)
        return 2
    if args.dry_run:
        result = run_preset.dry_run(preset)
        return 0 if result["match"] else 1
    run_preset.train_preset(preset, steps=args.steps, rounds=args.rounds, device=args.device,
                            out=args.out, init=args.init, resume=args.resume, seed=args.seed,
                            max_data=args.max_data, data=args.data)
    return 0


def _budget_args(p) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--tokens", default=None,
                   help="override the preset's token budget, e.g. 2e9")
    g.add_argument("--tokens-per-param", type=float, default=None,
                   help="budget as a multiple of parameters (Chinchilla-optimal is ~20)")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
