"""Inference server for Iridium-1. Standard library only, plus torch.

Serves two things over one port:

* ``/`` — a chat page that is honest about what it is. The trained rung is 34 M
  parameters fitted for 800 steps on a synthetic corpus; its *text* is not
  meaningful and the page says so in the first line. What is meaningful is the
  telemetry beside every response: which superstacks the router chose, how deep
  the focus ladder went into each, how many ponder loops the halting head
  spent, and what that cost. Those are the mechanisms this architecture exists
  to demonstrate, and they are real on every request.
* ``/api/*`` — JSON endpoints for the same.

No FastAPI, no uvicorn: ``ThreadingHTTPServer`` keeps the image small and the
dependency surface at one package. Concurrency is bounded by a lock around the
model, because a single CPU process running a 1 B forward pass does not benefit
from interleaving and an unbounded queue would just OOM.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from iridium.codecs.spans import Sample, quantity_span, text_span
from iridium.config import IridiumConfig, get_config
from iridium.model.iridium1 import Iridium1
from iridium.runtime import backend as backend_module
from iridium.runtime.device import detect as detect_device, device_of
from iridium.runtime.generate import generate

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
REPO = ROOT.parent

MAX_PROMPT = 2000
MAX_NEW_TOKENS = 64

_lock = threading.Lock()
_models: dict[str, dict] = {}


# --------------------------------------------------------------------------
# model registry
# --------------------------------------------------------------------------

# The shipped checkpoint is stored fp16 (66 MB) so it fits under GitHub's
# 100 MB per-file limit; it is cast back to fp32 on load. A local full-precision
# run is preferred when present.
CHECKPOINT_CANDIDATES = [
    os.environ.get("IRIDIUM_CHECKPOINT", ""),
    "runs/phase1/phase1-final.pt",
    "serve/weights/nano-phase1-fp16.pt",
]
CHECKPOINT = next(
    (c for c in CHECKPOINT_CANDIDATES if c and (REPO / c).exists()),
    "serve/weights/nano-phase1-fp16.pt",
)
ENABLE_1B = os.environ.get("IRIDIUM_ENABLE_1B", "1") not in ("0", "false", "")

CATALOG = {
    "nano-trained": {
        "label": "nano · 34 M · trained",
        "rung": "nano",
        "checkpoint": CHECKPOINT,
        "trained": True,
        "caveat": (
            "800 steps on a synthetic corpus. Graded accuracy is at or below a "
            "prompt-ignoring baseline on 4 of 5 task families. Treat the text as "
            "noise; the telemetry is the real output."
        ),
    },
    "test1b-untrained": {
        "label": "test1b · 1.00 B · untrained",
        "rung": "test1b",
        "checkpoint": None,
        "trained": False,
        "caveat": (
            "Randomly initialised. It exists to show the 1 B architecture "
            "executing — routing, focus, ponder, per-stack dispatch — not to say "
            "anything. Its text is definitionally meaningless."
        ),
    },
}


def _si(n: int) -> str:
    for unit, scale in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f} {unit}"
    return str(n)


def describe_label(cfg, spec: dict, manifest: dict) -> str:
    """Name the model that is actually loaded.

    The catalogue entry is a *default*, written when the bundled checkpoint was
    the only one. Point ``IRIDIUM_CHECKPOINT`` at a notebook's output — which is
    exactly what the studio notebooks do — and a hardcoded label goes on
    announcing 34 M for a model of some other size. A serving layer that
    misreports which weights it loaded undermines every number beside it.
    """
    if not manifest:
        return spec["label"]
    trained = bool((manifest.get("train_config") or {}).get("steps"))
    return f"{cfg.name} · {_si(cfg.n_params)} · {'trained' if trained else 'untrained'}"


def describe_caveat(spec: dict, manifest: dict) -> str:
    """Say what this checkpoint's own manifest says, not what a past run's did.

    The caveat is the most load-bearing string the server returns: it is what
    stops somebody reading the generated text as an answer. Reciting a stored
    sentence about a different run is worse than saying nothing, so an unknown
    provenance is reported as unknown.
    """
    if not manifest:
        return spec["caveat"]
    train = manifest.get("train_config", {}) or {}
    steps = train.get("steps")
    licences = manifest.get("data_licences", "unrecorded")
    parts = []
    if steps:
        parts.append(f"{steps:,} training steps (batch {train.get('batch_size', '?')})")
    parts.append(f"data: {licences}")

    ev = (manifest.get("evaluation") or {}).get("interpolation") or {}
    scored = [
        (fam, r) for fam, r in ev.items()
        if isinstance(r, dict) and "accuracy" in r and "baseline" in r
    ]
    if scored:
        beat = sum(1 for _, r in scored if r["accuracy"] > r["baseline"] + 0.1)
        verdict = (
            f"graded above a prompt-ignoring baseline on {beat} of {len(scored)} "
            f"task families"
        )
        if beat == 0:
            verdict += " — so the text is noise, and the telemetry is the real output"
        parts.append(verdict)
    else:
        parts.append("no grading recorded in this checkpoint, so treat it as unverified")
    return ". ".join(p[0].upper() + p[1:] for p in parts) + "."


def load(name: str) -> dict:
    if name in _models:
        return _models[name]
    spec = CATALOG[name]
    started = time.time()
    ckpt = spec["checkpoint"]
    if ckpt and (REPO / ckpt).exists():
        blob = torch.load(REPO / ckpt, map_location="cpu", weights_only=False)
        manifest = blob["manifest"]
        model = Iridium1(IridiumConfig.from_dict(manifest["model_config"]))
        # Named, per-key compatibility rather than strict=True (which has refused
        # this file since the quantity modality was added) or strict=False
        # (which would also accept a missing attention projection). See
        # iridium/runtime/checkpoint_compat.py.
        from iridium.runtime.checkpoint_compat import load_compatible
        compat = load_compatible(model, {k: v.float() for k, v in blob["state_dict"].items()})
        source = f"checkpoint {ckpt} ({compat.summary()})"
    else:
        torch.manual_seed(0)
        model = Iridium1(get_config(spec["rung"]))
        manifest = {}
        source = "random initialisation"
    model.eval()
    info = detect_device(os.environ.get("IRIDIUM_DEVICE"))
    model = model.to(info.device)
    cfg = model.cfg
    entry = {
        "name": name,
        "model": model,
        "cfg": cfg,
        "source": source,
        "label": describe_label(cfg, spec, manifest),
        "caveat": describe_caveat(spec, manifest),
        "load_seconds": time.time() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "specializations": list(cfg.stacks.specializations),
        "device": info.describe(),
        **{k: v for k, v in spec.items()
           if k not in ("checkpoint", "label", "caveat")},
    }
    _models[name] = entry
    print(f"[iridium] loaded {name}: {entry['parameters']:,} params from {source} "
          f"in {entry['load_seconds']:.1f}s", flush=True)
    return entry


def available() -> list[dict]:
    out = []
    for name, spec in CATALOG.items():
        if name == "test1b-untrained" and not ENABLE_1B:
            continue
        # A loaded model describes itself; an unloaded one can only be described
        # from its catalogue rung, which is a guess until the weights arrive.
        loaded = _models.get(name)
        cfg = loaded["cfg"] if loaded else get_config(spec["rung"])
        out.append({
            "name": name,
            "label": loaded["label"] if loaded else spec["label"],
            "trained": spec["trained"],
            "caveat": loaded["caveat"] if loaded else spec["caveat"],
            "parameters": cfg.n_params,
            "loaded": name in _models,
            "device": loaded["device"] if loaded else None,
            "geometry": {
                "core_layers": cfg.core.n_layers,
                "d_model": cfg.core.d_model,
                "n_stacks": cfg.stacks.n_stacks,
                "stack_layers": cfg.stacks.n_layers,
                "top_k": cfg.router.top_k,
                "max_loops": cfg.router.max_loops,
                "specializations": list(cfg.stacks.specializations),
            },
        })
    return out


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# typed physics queries
# --------------------------------------------------------------------------

import re as _re

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_PARAM = _re.compile(rf"\b(S|n|q|x|factor)\s*=?\s*({_NUM})", _re.I)

QUERY_HELP = (
    "Ask for a number and it answers with one. Two forms:\n"
    "  normal depth | S=0.0020 n=0.030 q=3.0\n"
    "  depth ratio  | S=0.0020 n=0.030 q=3.0 x2.0\n"
    "Anything else is read as a claim and gets a TRUE/FALSE verdict."
)


def parse_query(text: str) -> dict:
    """Read a physics query into typed quantities, or fall back to a claim."""
    params: dict[str, float] = {}
    for key, value in _PARAM.findall(text):
        key = key.lower()
        key = "x" if key == "factor" else key
        try:
            params[key] = float(value)
        except ValueError:
            continue
    lowered = text.lower()
    wants_ratio = ("ratio" in lowered or "double" in lowered or "x" in params)
    if {"s", "n", "q"} <= set(params):
        if wants_ratio:
            factor = params.get("x", 2.0)
            return {"kind": "ratio", "slope": params["s"], "manning": params["n"],
                    "discharge": params["q"], "factor": factor}
        return {"kind": "depth", "slope": params["s"], "manning": params["n"],
                "discharge": params["q"]}
    return {"kind": "verdict", "claim": text}


def analytic(query: dict) -> dict:
    """The exact answer, so the model's number can be checked, not believed."""
    from iridium.physics.shallow_water import critical_depth, normal_depth

    if query["kind"] == "depth":
        value = normal_depth(query["discharge"], query["slope"], query["manning"])
        return {"value": value, "law": "Manning normal depth  h = (q n / sqrt(S))^(3/5)",
                "unit": "m",
                "also": {"critical_depth_m": critical_depth(query["discharge"])}}
    if query["kind"] == "ratio":
        f = query["factor"]
        return {"value": f ** 0.6, "unit": "1",
                "law": "normal-depth ratio  = factor^(3/5)",
                "also": {"critical_depth_ratio": f ** (2.0 / 3.0),
                         "velocity_ratio": f ** 0.4,
                         "naive_answer": f}}
    return {}


def build_sample(query: dict) -> Sample:
    from iridium.training.tasks import BOS, SEP, control_span, encode_text

    if query["kind"] == "depth":
        return Sample([
            control_span(BOS),
            encode_text("normal depth", supervised=False),
            quantity_span([("slope", query["slope"]), ("manning", query["manning"]),
                           ("discharge", query["discharge"])], supervised=False),
            control_span(SEP),
        ])
    if query["kind"] == "ratio":
        return Sample([
            control_span(BOS),
            encode_text("depth ratio", supervised=False),
            quantity_span([("slope", query["slope"]), ("manning", query["manning"]),
                           ("discharge", query["discharge"]),
                           ("factor", query["factor"])], supervised=False),
            control_span(SEP),
        ])
    return Sample([
        control_span(BOS),
        encode_text(query["claim"][:MAX_PROMPT], supervised=False),
        control_span(SEP),
    ])


@torch.no_grad()
def run_query(prompt: str, model_name: str, loops: int) -> dict:
    """Answer a typed query with a number (or a verdict) plus the telemetry."""
    from iridium.codecs.bank import TensorBatch, continuous_dims
    from iridium.codecs.spans import collate
    from iridium.runtime.decode import atomic_chunks, slice_batch
    from iridium.training.tasks import VERDICT_FALSE, VERDICT_TRUE

    entry = load(model_name)
    model, cfg = entry["model"], entry["cfg"]
    loops = max(1, min(int(loops), cfg.router.max_loops))
    query = parse_query(prompt)
    sample = build_sample(query)

    dims = continuous_dims(cfg.codecs)
    batch = TensorBatch(collate([sample], dims), device=device_of(model))
    cache: dict = {}
    started = time.time()
    result = None
    for lo, hi in atomic_chunks(batch, int(batch.modality.shape[1])):
        result = model(slice_batch(batch, lo, hi), n_loops=loops, cache=cache)
    elapsed = time.time() - started
    hidden = result.hidden[:, -1:]

    truth = analytic(query)
    answer: dict = {"kind": query["kind"]}
    if query["kind"] in ("depth", "ratio"):
        log10_value = float(model.codecs.decode_continuous(hidden, "quantity")[0, 0, 0])
        predicted = 10.0 ** log10_value
        answer["predicted"] = predicted
        answer["analytic"] = truth["value"]
        answer["relative_error"] = abs(predicted - truth["value"]) / max(abs(truth["value"]), 1e-12)
        answer["within_2pct"] = answer["relative_error"] <= 0.02
        answer["law"] = truth["law"]
        answer["unit"] = truth["unit"]
        answer["also"] = truth["also"]
    else:
        logits = model.codecs.text_head(hidden)[0, 0]
        answer["verdict"] = ("TRUE" if float(logits[VERDICT_TRUE]) > float(logits[VERDICT_FALSE])
                             else "FALSE")
        margin = float(logits[VERDICT_TRUE]) - float(logits[VERDICT_FALSE])
        answer["margin"] = margin
        answer["help"] = QUERY_HELP

    stats = result.stats["stack_stats"][0]
    tokens = stats["per_stack_tokens"]
    total = max(sum(tokens), 1)
    names = entry["specializations"] or [f"stack {i}" for i in range(cfg.stacks.n_stacks)]
    return {
        "query": query,
        "answer": answer,
        "seconds": elapsed,
        "model": {"name": entry["name"], "label": entry["label"],
                  "parameters": entry["parameters"], "trained": entry["trained"],
                  "source": entry["source"], "caveat": entry["caveat"]},
        "telemetry": {
            "prompt_tokens": int(batch.valid.sum()),
            "loops_requested": loops,
            "expected_loops": float(result.expected_loops.mean()),
            "mean_focus": float(result.decisions[0].focus.mean()),
            "router_entropy": float(result.decisions[0].entropy()),
            "stacks_used": int(sum(1 for t in tokens if t > 0)),
            "max_depth": cfg.stacks.n_layers,
            "routing": [
                {"stack": i, "name": names[i].replace("_", " "),
                 "tokens": int(tokens[i]), "share": tokens[i] / total,
                 "expected_depth": round(stats["per_stack_expected_depth"][i], 3)}
                for i in range(cfg.stacks.n_stacks)
            ],
        },
    }


@torch.no_grad()
def run_chat(prompt: str, model_name: str, max_new_tokens: int,
             temperature: float, loops: int) -> dict:
    entry = load(model_name)
    model, cfg = entry["model"], entry["cfg"]
    loops = max(1, min(int(loops), cfg.router.max_loops))
    max_new_tokens = max(1, min(int(max_new_tokens), MAX_NEW_TOKENS))

    sample = Sample([text_span(prompt[:MAX_PROMPT], offset=16)])
    t0 = time.time()
    out = generate(model, sample, max_new_tokens=max_new_tokens,
                   temperature=float(temperature), n_loops=loops,
                   seed=int(time.time() * 1000) % (1 << 30))
    gen_seconds = time.time() - t0

    # A second, non-sampling pass over the prompt to read the routing telemetry.
    from iridium.codecs.bank import TensorBatch, continuous_dims
    from iridium.codecs.spans import collate
    from iridium.runtime.decode import atomic_chunks, slice_batch

    dims = continuous_dims(cfg.codecs)
    batch = TensorBatch(collate([sample], dims), device=device_of(model))
    cache: dict = {}
    result = None
    for lo, hi in atomic_chunks(batch, int(batch.modality.shape[1])):
        result = model(slice_batch(batch, lo, hi), n_loops=loops, cache=cache)
    stats = result.stats["stack_stats"][0]
    tokens = stats["per_stack_tokens"]
    total = max(sum(tokens), 1)
    names = entry["specializations"] or [f"stack {i}" for i in range(cfg.stacks.n_stacks)]

    return {
        "text": out.text,
        "stopped": out.stopped,
        "tokens_generated": len(out.ids),
        "seconds": gen_seconds,
        "tokens_per_second": len(out.ids) / max(gen_seconds, 1e-6),
        "model": {
            "name": entry["name"], "label": entry["label"],
            "parameters": entry["parameters"], "trained": entry["trained"],
            "source": entry["source"], "caveat": entry["caveat"],
        },
        "telemetry": {
            "prompt_tokens": int(batch.valid.sum()),
            "loops_requested": loops,
            "expected_loops": float(result.expected_loops.mean()),
            "mean_focus": float(result.decisions[0].focus.mean()),
            "generation_mean_focus": out.mean_focus,
            "routing": [
                {"stack": i, "name": names[i].replace("_", " "),
                 "tokens": int(tokens[i]), "share": tokens[i] / total,
                 "expected_depth": round(stats["per_stack_expected_depth"][i], 3)}
                for i in range(cfg.stacks.n_stacks)
            ],
            "stacks_used": int(sum(1 for t in tokens if t > 0)),
            "max_depth": cfg.stacks.n_layers,
            "router_entropy": float(result.decisions[0].entropy()),
            "balance_loss": float(result.stats["balance_loss"]),
        },
    }


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "iridium/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def _send(self, code: int, body: bytes, ctype: str, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict):
        self._send(code, json.dumps(payload, default=float).encode(),
                   "application/json; charset=utf-8",
                   {"Cache-Control": "no-store"})

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/api/health":
            return self._json(200, {
                "ok": True, "loaded": sorted(_models), "help": QUERY_HELP,
                "torch": torch.__version__,
                "threads": torch.get_num_threads(),
                "device": detect_device(os.environ.get("IRIDIUM_DEVICE")).describe(),
                # Full backend report (HIP vs CUDA, arch, bf16 basis, SDPA
                # kernels, torch.compile, matmul precision, env vars) so a
                # user pointing this at an AMD card can see what they actually
                # got instead of inferring it from a one-line device string.
                "backend": backend_module.capabilities(os.environ.get("IRIDIUM_DEVICE")),
            })
        if route == "/api/models":
            return self._json(200, {"models": available()})
        if route in ("/", "/chat"):
            return self._file(STATIC / "chat.html")
        if route == "/explore":
            return self._file(STATIC / "index.html")
        name = route.lstrip("/")
        if name and "/" not in name and ".." not in name:
            candidate = STATIC / name
            if candidate.is_file():
                return self._file(candidate)
        self._json(404, {"error": "not found", "path": route})

    def _file(self, path: Path):
        if not path.is_file():
            return self._json(404, {"error": "not found"})
        ctype = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        cache = "no-store" if path.suffix == ".json" else "public, max-age=300"
        self._send(200, path.read_bytes(), ctype, {"Cache-Control": cache})

    def do_POST(self):
        route = urlparse(self.path).path
        if route not in ("/api/chat", "/api/ask"):
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 64_000:
                return self._json(413, {"error": "request too large"})
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": f"bad JSON: {exc}"})

        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            return self._json(400, {"error": "prompt is required"})
        name = payload.get("model", "nano-trained")
        if name not in CATALOG or (name == "test1b-untrained" and not ENABLE_1B):
            return self._json(400, {"error": f"unknown model {name!r}"})
        try:
            with _lock:
                if route == "/api/ask":
                    return self._json(200, run_query(
                        prompt, name, payload.get("loops", 1)
                    ))
                result = run_chat(
                    prompt, name,
                    payload.get("max_new_tokens", 24),
                    payload.get("temperature", 0.8),
                    payload.get("loops", 1),
                )
            return self._json(200, result)
        except Exception as exc:                       # pragma: no cover
            traceback.print_exc()
            return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    threads = int(os.environ.get("IRIDIUM_THREADS", "0")) or (os.cpu_count() or 2)
    torch.set_num_threads(threads)
    port = int(os.environ.get("PORT", "8080"))
    info = detect_device(os.environ.get("IRIDIUM_DEVICE"))
    print(f"[iridium] torch {torch.__version__}, {info.describe()}, "
          f"{threads} threads, port {port}", flush=True)
    if info.backend == "rocm":
        caps = backend_module.capabilities(os.environ.get("IRIDIUM_DEVICE"))
        compiled = caps["torch_compile"]
        print(f"[iridium] ROCm: torch.compile {'ok' if compiled['available'] else 'unavailable'} "
              f"({compiled['detail']}), matmul precision: {caps['matmul_precision']}", flush=True)
    if os.environ.get("IRIDIUM_PRELOAD", "1") not in ("0", "false", ""):
        try:
            load("nano-trained")
        except Exception:                              # pragma: no cover
            traceback.print_exc()
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[iridium] listening on 0.0.0.0:{port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
