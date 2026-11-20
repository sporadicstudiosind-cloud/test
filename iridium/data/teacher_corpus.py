"""Sequence-level distillation: a hosted teacher model writes text, a subject stack trains on it.

Each subject (maths, science, code, chat) gets one teacher family. A teacher
writes short, simple texts from seed prompts; the rows are kept as JSON lines
so a run can be resumed, audited and re-sharded without calling the API again.
:func:`to_shards` then packs them into the same on-disk shards the rest of the
data pipeline reads (see :mod:`iridium.data.shards`).

Seed prompts are Cosmopedia-style: a topic, an audience and a format are
combined, and each seed index maps to one fixed combination, so the same
``(seed, subject, index)`` always gives the same prompt. Topics cycle through
their list so every topic is covered before any repeats.

Calls use the OpenAI-compatible ``POST {base_url}/chat/completions`` shape that
Groq, OpenRouter and NVIDIA NIM all accept. The key is read from an environment
variable at call time and is never printed, logged or written to disk.

**Check your provider's terms before training on these outputs.** Many hosted
APIs restrict using their outputs to train competing models, and some
licences attach conditions to derived data. ``teacher_manifest.json`` records
which models and hosts produced each shard so that provenance can be audited;
it does not settle whether the terms allow the use. The text is also only as
good as the teacher: it teaches that model's habits and mistakes too.

Module-level imports are standard library only, so :mod:`iridium.data.domains`
can import :data:`TEACHER_FAMILIES` without pulling in numpy or torch.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

__all__ = ["TEACHER_FAMILIES", "generate", "to_shards", "build_prompt"]

#: Family name -> subject. Families are the keys; subjects match domains.SUBJECTS.
TEACHER_FAMILIES = {
    "math_teacher": "math",
    "science_teacher": "science",
    "code_teacher": "code",
    "chat_teacher": "chat",
}

MAX_TRIES = 5          # HTTP attempts per sample before giving up on rate limits
BACKOFF_BASE = 2.0     # seconds; attempt k waits BACKOFF_BASE * 2**k
TIMEOUT = 120.0
_USER_AGENT = "iridium-teacher-corpus/1.0"

AUDIENCES = (
    "a young student aged 8 to 10",
    "a high-school student",
    "an undergraduate student",
)

TOPICS = {
    "math": [
        "adding fractions with different denominators", "the area of a triangle",
        "order of operations", "solving one-step equations", "ratios and rates",
        "percentages when shopping", "the Pythagorean theorem", "mean, median and mode",
        "place value in whole numbers", "negative numbers on a number line",
        "perimeter of a rectangle", "linear equations in one variable", "the slope of a line",
        "probability of simple events", "prime and composite numbers",
        "greatest common divisor", "exponents and powers", "estimating square roots",
        "the distributive property", "the area of a circle", "volume of a rectangular box",
        "simple interest", "converting metres to centimetres", "reading bar graphs",
        "combining like terms", "factoring quadratic expressions",
        "systems of two linear equations", "functions as input and output machines",
        "the derivative as a rate of change", "limits of number sequences",
        "adding matrices", "the dot product of two vectors", "clock arithmetic",
        "mathematical induction", "sums of geometric series",
    ],
    "science": [
        "photosynthesis", "the three states of matter", "Newton's first law of motion",
        "the water cycle", "cells as the basic unit of life", "Ohm's law",
        "density and why things float", "plate tectonics", "groups in the periodic table",
        "conservation of mass in chemical reactions", "food chains and food webs",
        "gravity and orbits", "reflection of light in a mirror", "sound waves and pitch",
        "magnets and electromagnetism", "the planets of the solar system",
        "the human digestive system", "acids and bases", "energy transfer and conservation",
        "heat conduction in metals", "natural selection", "the parts of an atom",
        "air pressure and weather", "igneous, sedimentary and metamorphic rocks",
        "series and parallel electric circuits", "DNA and heredity",
        "the immune system", "moles and molar mass", "Hooke's law and springs",
        "radioactive decay and half-life", "ecosystems and biodiversity",
        "the greenhouse effect", "enzymes and temperature", "the kinetic theory of gases",
        "forces as vectors", "the lifecycle of a star", "osmosis across a membrane",
    ],
    "code": [
        "variables and assignment in Python", "for loops over a list",
        "while loops and counters", "defining functions with parameters and return values",
        "string slicing", "dictionaries as key-value stores", "reading and writing text files",
        "recursion with a factorial function", "binary search on a sorted list",
        "bubble sort", "list comprehensions", "catching errors with try and except",
        "classes and objects", "unit tests with assert", "Big-O cost of nested loops",
        "stacks and queues", "parsing a line of CSV", "string formatting",
        "sets and membership tests", "sorting tuples by a key", "finding an off-by-one error",
        "creating a git branch and committing", "regular expressions for dates",
        "encoding and decoding JSON", "command-line arguments with argparse",
        "a simple HTTP GET request", "generators and yield", "map, filter and reduce",
        "two-dimensional lists as grids", "breadth-first search on a graph",
        "memoizing Fibonacci numbers", "bit operations with AND, OR and shifts",
        "writing a docstring", "the difference between == and is",
        "reading a value from a dictionary safely with get",
    ],
    "chat": [
        "planning a weekend hike", "choosing a first houseplant", "making a simple budget",
        "writing a thank-you note", "preparing for a job interview",
        "what to pack for a camping trip", "learning a second language",
        "fixing a slow laptop", "tips for better sleep", "cooking rice on the stove",
        "returning an item to a shop", "starting to run as a beginner",
        "explaining a rainbow to a child", "what a weir is", "a question the assistant cannot answer",
        "recommending a book for a rainy day", "making small talk at a party",
        "cleaning a cast-iron pan", "a small disagreement with a roommate",
        "how bees make honey", "a birthday party on a small budget",
        "reading a nutrition label", "caring for a new puppy",
        "how elections work in simple terms", "why the sky is blue",
        "choosing a bicycle for commuting", "a greeting and talk about the weather",
        "writing a polite email to a teacher", "spotting a phishing email",
        "feeling stressed before an exam", "folding a paper airplane",
        "the history of the bicycle", "tying a bowline knot", "composting kitchen scraps",
        "a question about the plot of a story", "whether the assistant knows today's news",
    ],
}

#: Formats for the three text subjects. Chat has one format: a conversation.
FORMATS = (
    "a short textbook section with a heading and three or four short paragraphs",
    "a worked example that states a problem and then solves it step by step",
    "a question-and-answer exchange with three questions and clear answers",
    "a short lesson with a learning goal, a simple explanation, and one check question with its answer",
)

_SUBJECT_NOTE = {
    "math": "Show every calculation line by line.",
    "science": "Explain the cause of each idea, and say what to observe.",
    "code": "Include short Python code where it helps, and explain each line.",
}


def _family_for(subject: str) -> str:
    """Accepts a subject ("math") or its family ("math_teacher")."""
    if subject in TEACHER_FAMILIES:
        return subject
    for family, name in TEACHER_FAMILIES.items():
        if name == subject:
            return family
    raise ValueError(f"unknown teacher subject {subject!r}; expected one of "
                     f"{sorted(set(TEACHER_FAMILIES.values()))}")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_prompt(subject: str, index: int, seed: int = 0) -> str:
    """The seed prompt for one sample. Pure function of its arguments."""
    family = _family_for(subject)
    subj = TEACHER_FAMILIES[family]
    topics = TOPICS[subj]
    # One offset per (seed, subject), so consecutive indices walk the topic list
    # in a strict cycle: every topic appears before any repeats.
    offset = int(_digest(f"{seed}|{subj}")[:8], 16)
    topic = topics[(index + offset) % len(topics)]
    rng = random.Random(int(_digest(f"{seed}|{subj}|{index}")[:16], 16))
    audience = rng.choice(AUDIENCES)
    if subj == "chat":
        return (f"Write a short, realistic conversation of three or four exchanges between a user "
                f"and a helpful AI assistant about {topic}. The user is {audience}. Label every "
                "line 'User:' or 'Assistant:'. The assistant is accurate, warm and concise, and "
                "says so when it is unsure. Use simple words and short sentences. Keep it under "
                "250 words. Output only the conversation.")
    fmt = rng.choice(FORMATS)
    note = _SUBJECT_NOTE[subj]
    return (f"Write {fmt} about {topic} for {audience}. {note} Use short sentences and simple "
            "words. Keep it under 250 words. Output only the text itself, with no preamble.")


def _read_existing(out: Path) -> tuple[set[tuple[str, int]], set[str]]:
    """Seed indices and text digests already in the output file (partial lines skipped)."""
    indices: set[tuple[str, int]] = set()
    digests: set[str] = set()
    if not out.exists():
        return indices, digests
    with out.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue                      # a line cut short by an earlier crash
            indices.add((row.get("family"), row.get("seed_index")))
            if isinstance(row.get("text"), str):
                digests.add(_digest(row["text"].strip()))
    return indices, digests


def _ends_with_newline(path: Path) -> bool:
    with path.open("rb") as fh:
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) == b"\n"


def _chat_once(url: str, key: str, model: str, prompt: str, max_tokens: int) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.8,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "User-Agent": _USER_AGENT,
    })
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return (payload["choices"][0]["message"].get("content") or "")


def generate(subject: str, n: int, out_path, base_url: str, model: str, *,
             api_key_env: str = "TEACHER_API_KEY", seed: int = 0, max_tokens: int = 700,
             sleep: float = 0.0) -> int:
    """Append up to ``n`` teacher texts for ``subject`` to ``out_path``.

    Resumable: seed indices already in the file are skipped. Exact duplicates
    and empty outputs are dropped. On repeated HTTP 429 the call stops cleanly
    and returns how many rows it wrote. Returns the number written by this call.
    """
    family = _family_for(subject)
    subj = TEACHER_FAMILIES[family]
    out = Path(out_path)
    done, digests = _read_existing(out)
    pending = [i for i in range(n) if (family, i) not in done]
    if not pending:
        return 0
    key = os.environ.get(api_key_env)
    if not key:
        raise RuntimeError(f"set the environment variable {api_key_env} to your provider's API "
                           "key; it is read at call time and never written to disk")
    url = base_url.rstrip("/") + "/chat/completions"
    host = urllib.parse.urlsplit(base_url).hostname or ""
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("a", encoding="utf-8") as fh:
        if out.stat().st_size and not _ends_with_newline(out):
            fh.write("\n")                    # a crash left a partial line; do not append onto it
        for index in pending:
            prompt = build_prompt(subj, index, seed)
            text = None
            for attempt in range(MAX_TRIES):
                try:
                    text = _chat_once(url, key, model, prompt, max_tokens)
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code != 429:
                        raise RuntimeError(f"teacher API returned HTTP {exc.code} for {host}") from None
                    if attempt + 1 < MAX_TRIES:
                        time.sleep(BACKOFF_BASE * 2 ** attempt)
            else:
                break                          # rate limited on every try: stop cleanly
            text = (text or "").strip()
            if sleep > 0:
                time.sleep(sleep)
            if not text or _digest(text) in digests:
                continue
            digests.add(_digest(text))
            row = {
                "family": family, "subject": subj, "text": text, "prompt": prompt,
                "model": model, "base_url": host, "seed_index": index,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
    return written


def to_shards(jsonl_paths: Iterable, shard_dir, tokenizer, window: int, split: str = "train",
              test_fraction: float = 0.01, *, vocab_rows: int) -> dict:
    """Pack teacher rows into ``{family}-{split}`` shards plus ``{family}-test``.

    Each text is laid out as ``BOS text EOS`` with the same supervision as the
    text shards in :mod:`iridium.training.prepare`: BOS is context only, the
    text and EOS are targets. Short items are packed into windows of ``window``
    ids. A row is held out for ``-test`` when a stable hash of its text falls
    under ``test_fraction``. Exact duplicates across all inputs are dropped.
    Writes ``teacher_manifest.json`` (models, hosts, row counts) beside the
    shards and returns it.
    """
    from ..codecs.spans import Sample, Span, text_span
    from ..config import TEXT_ID_OFFSET
    from ..data.shards import PackingWriter, ShardWriter
    from ..training.tasks import BOS, EOS
    from ..training.tokenizer_bridge import tokenizer_manifest
    import numpy as np

    if split == "test":
        raise ValueError("split must not be 'test': held-out rows always go to '{family}-test'")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    meta = {"split": split, "window": window, "teacher": True,
            "tokenizer": tokenizer_manifest(tokenizer)}
    writers: dict[tuple[str, str], PackingWriter] = {}

    def writer_for(family: str, name: str) -> PackingWriter:
        key = (family, name)
        if key not in writers:
            writers[key] = PackingWriter(
                ShardWriter(shard_dir / f"{family}-{name}", family, vocab_rows, dict(meta)),
                window)
        return writers[key]

    seen: set[str] = set()
    dropped = {"duplicate": 0, "empty": 0, "malformed": 0, "unknown_family": 0, "too_long": 0}
    models: set[str] = set()
    hosts: set[str] = set()
    inputs = []
    try:
        for path in jsonl_paths:
            inputs.append(Path(path).name)
            with Path(path).open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        dropped["malformed"] += 1
                        continue
                    family = row.get("family") if isinstance(row, dict) else None
                    if family not in TEACHER_FAMILIES:
                        dropped["unknown_family"] += 1
                        continue
                    text = str(row.get("text") or "").strip()
                    if not text:
                        dropped["empty"] += 1
                        continue
                    digest = _digest(text)
                    if digest in seen:
                        dropped["duplicate"] += 1
                        continue
                    seen.add(digest)
                    held = int(digest[:16], 16) / float(1 << 64) < test_fraction
                    name = "test" if held else split
                    spans = [Span("control", np.array([BOS], np.int64), supervised=False),
                             text_span(text, supervised=True, offset=TEXT_ID_OFFSET,
                                       tokenizer=tokenizer),
                             Span("control", np.array([EOS], np.int64), supervised=True)]
                    if not writer_for(family, name).add(Sample(spans)):
                        dropped["too_long"] += 1
                        continue
                    if row.get("model"):
                        models.add(str(row["model"]))
                    if row.get("base_url"):
                        hosts.add(str(row["base_url"]))
    finally:
        # Rows kept, not windows written: a window holds several packed rows.
        closed = {}
        for (family, name), pw in writers.items():
            rows_kept = pw.items
            pw.close()
            closed.setdefault(family, {})[name] = rows_kept
    families = {f: {name: closed[f].get(name, 0) for name in (split, "test")} for f in closed}
    manifest = {
        "format": 1,
        "window": window,
        "split": split,
        "test_fraction": test_fraction,
        "tokenizer": meta["tokenizer"],
        "families": families,
        "rows": sum(sum(v.values()) for v in families.values()),
        "models": sorted(models),
        "hosts": sorted(hosts),
        "inputs": inputs,
        "dropped": dropped,
        "terms": ("Check each provider's terms before training on these outputs. This file "
                  "records provenance only; it does not establish that the use is permitted."),
    }
    (shard_dir / "teacher_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
