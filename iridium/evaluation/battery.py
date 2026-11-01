"""A plain graded test of what a chat checkpoint can do: ``python -m iridium battery``.

Written after chat-36m's validation loss looked healthy while it got 0 of 20 on
simple arithmetic. Loss does not answer "does it work", so this asks, and checks
each answer by computing or running it wherever that is possible:

* **facts**: an answer counts only in the reply's first sentence.
* **arithmetic**: sums held out of training (:func:`held_out_arithmetic`).
* **code**: the produced function is run in the sandbox and its result compared.
  The templates asked for here are held out of ``code_exec`` training.
* **agent**: real tasks in a real temporary folder; what counts is the file or
  answer left at the end, not the model's claim about it.
* **media**: generated pictures and tones the foundation families teach.

Writing has no exact grader, so samples are printed for a person to read. The
battery is small and home-made: use it to compare runs, not as a benchmark.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

__all__ = ["FACTS", "first_sentence_hit", "extract_code", "run_battery"]

FACTS = [
    ("What is the capital of France?", "paris"), ("What is the capital of Japan?", "tokyo"),
    ("What is the capital of Italy?", "rome"), ("Who wrote Romeo and Juliet?", "shakespeare"),
    ("What planet is known as the Red Planet?", "mars"), ("What is the chemical symbol for gold?", "au"),
    ("What gas do plants absorb from the air?", "carbon dioxide"),
    ("What is the largest planet in the solar system?", "jupiter"),
    ("What is the freezing point of water in Celsius?", "0"), ("How many legs does a spider have?", "eight|8"),
    ("Who developed the theory of general relativity?", "einstein"),
    ("What organ pumps blood through the body?", "heart"),
]

WRITING = ("Write a short story about a cat who finds a key.", "Write a short poem about the ocean.",
           "Write the opening paragraph of a mystery novel.", "Explain how a battery works.")


def first_sentence_hit(reply: str, key: str) -> bool:
    first = re.split(r"(?<=[.!?])\s", reply.strip())[0].lower()
    return any(re.search(rf"(?<![\w.]){re.escape(k)}(?![\w])", first) for k in key.split("|"))


def extract_code(reply: str) -> str:
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", reply, re.S)
    return fenced.group(1) if fenced else reply


def _agent_tasks():
    """(request, starting files, check(workspace, answer) -> bool)."""
    def file_is(name, text):
        return lambda root, answer: (root / name).is_file() and text in (root / name).read_text("utf-8")

    def says(text):
        return lambda root, answer: text in answer.lower()

    return [
        ("Create hello.txt containing the word lantern.", {}, file_is("hello.txt", "lantern")),
        ("Create idea5.txt containing the word orbit.", {}, file_is("idea5.txt", "orbit")),
        ("What does secret.txt say?", {"secret.txt": "The password hint is quartz."}, says("quartz")),
        ("How many files are in this folder?", {"a.txt": "1", "b.txt": "2", "c.txt": "3"}, says("3")),
        ("In app.txt, change meadow to pebble.", {"app.txt": "color = meadow\nsize = 3\n"},
         file_is("app.txt", "color = pebble")),
        ("Which file mentions willow-key?", {"one.txt": "nothing here", "two.txt": "the willow-key is here"},
         says("two.txt")),
        ("Use Python to multiply 37 by 41.", {}, says("1517")),
        ("What is (12 + 30) * 7?", {}, says("294")),
    ]


@torch.inference_mode()
def run_battery(model, tokenizer, seeds: int = 3, temperature: float = 0.3,
                say: Callable[[str], None] = print) -> dict:
    from ..data.agent_corpus import (CODE_TEMPLATES, agent_system_prompt, audio_foundation_item,
                                     held_out_arithmetic, image_foundation_item, render_code_template)
    from ..codecs.spans import Sample
    from ..runtime.agent_tools import build_registry
    from ..runtime.chat import ChatSession, Turn
    from ..runtime.generate import generate
    from ..runtime.sandbox import Sandbox
    from ..runtime.tools import run_tool_loop

    def ask(question, seed, max_new=48):
        session = ChatSession(model, tokenizer=tokenizer, temperature=temperature,
                              max_new_tokens=max_new, seed=seed)
        return session.send(question)

    scores: dict[str, dict] = {}

    hits = sum(first_sentence_hit(ask(q, s), key) for q, key in FACTS for s in range(seeds))
    scores["facts"] = {"correct": hits, "total": len(FACTS) * seeds}

    sums = held_out_arithmetic(30)
    hits = sum(first_sentence_hit(ask(q, 0), str(value)) for q, value in sums)
    scores["arithmetic_held_out"] = {"correct": hits, "total": len(sums)}

    rng = np.random.default_rng(0)
    passed = total = 0
    for index in range(len(CODE_TEMPLATES)):
        if index % 5 != 4:
            continue
        request, _, call, expected = render_code_template(index, rng)
        name = call.split("(")[0]
        reply = ask(f"{request} Name it {name}.", 0, 160)
        check = f"{extract_code(reply)}\nemit({{'value': {call}}})\n"
        result = Sandbox().run(check, timeout=10)
        passed += bool(result.ok and result.payload.get("value") == json.loads(json.dumps(expected)))
        total += 1
    scores["code_held_out"] = {"correct": passed, "total": total}

    done = calls_made = 0
    tasks = _agent_tasks()
    for request, files, check in tasks:
        with tempfile.TemporaryDirectory(prefix="iridium-battery-") as tmp:
            root = Path(tmp)
            for name, text in files.items():
                (root / name).write_text(text, encoding="utf-8")
            registry = build_registry(root, groups=("files", "python", "calc", "fun"), seed=0)
            # The tool block the model trained on, whichever tools this run binds.
            messages = [Turn("system", agent_system_prompt()), Turn("user", request)]
            out = run_tool_loop(model, messages, registry, tokenizer=tokenizer, max_calls=5,
                                max_new_tokens=128, max_bytes=2800, temperature=temperature, seed=0)
            calls_made += len(out.calls)
            done += bool(out.stopped == "answer" and check(root, out.answer))
    scores["agent_tasks"] = {"correct": done, "total": len(tasks), "tool_calls_made": calls_made}

    # Media: re-open each generated sample at its reply and let the model answer.
    from ..data.tokenization import retokenize
    from ..training.tasks import Item  # noqa: F401
    hits = total = 0
    for maker in (image_foundation_item, audio_foundation_item):
        for seed in range(15):
            item = maker(np.random.default_rng(seed), "test")
            sample = retokenize(item.sample, tokenizer)
            prompt = next(i for i, s in enumerate(sample.spans) if s.supervised)
            reply = generate(model, Sample(sample.spans[:prompt], meta={}), max_new_tokens=8,
                             temperature=0.0, text_only=True, stop_ids=(2, 7, 9), tokenizer=tokenizer)
            hits += item.answer.lower().rstrip(".") in reply.text.lower()
            total += 1
    scores["media_foundation"] = {"correct": hits, "total": total}

    for name, row in scores.items():
        say(f"{name:<22} {row['correct']}/{row['total']}"
            + (f"  ({row['tool_calls_made']} tool calls)" if "tool_calls_made" in row else ""))
    say("\n--- writing samples, for reading")
    samples = {}
    for prompt in WRITING:
        samples[prompt] = ask(prompt, 0, 200)
        say(f"[{prompt}]\n{samples[prompt]}\n")
    scores["writing_samples"] = samples
    return scores
