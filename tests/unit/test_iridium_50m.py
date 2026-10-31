"""iridium-50m: the preset, its generated data, its tools and its training path."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from iridium.codecs.bank import TensorBatch, continuous_dims
from iridium.codecs.spans import collate
from iridium.config import IridiumConfig
from iridium.data.agent_corpus import (CODE_TEMPLATES, agent_episode_item, arithmetic_drill_item,
                                       audio_foundation_item, code_exec_item, foundation_image,
                                       held_out_arithmetic, image_foundation_item,
                                       is_test_pair, render_code_template)
from iridium.data.domains import DOMAIN_FAMILIES, GENERATED_SUBJECT, SUBJECTS
from iridium.model.iridium1 import Iridium1
from iridium.presets import get_preset
from iridium.runtime import chat
from iridium.runtime.agent_tools import (LocalFS, MemoryFS, WorkspaceError, WorkspaceTools,
                                         _check_public, build_registry, calculate, web_search)
from iridium.runtime.chat import Turn, conversation_spans
from iridium.runtime.sandbox import Sandbox
from iridium.runtime.tools import format_tool_call, run_tool_loop
from iridium.training.tasks import GENERATORS

PRESET = get_preset("iridium-50m")
CFG = PRESET.config


def small_config():
    from dataclasses import replace
    core = replace(CFG.core, d_model=64, n_layers=2, n_query_heads=4, d_head=16, d_ff=128)
    stacks = replace(CFG.stacks, d_model=64, core_d_model=64, n_layers=2, n_query_heads=4,
                     d_head=16, d_ff=128, min_depth=1)
    return replace(CFG, core=core, stacks=stacks, ple_dim=4, perception_layers=1, perception_rank=8,
                   max_seq_len=256, text_vocab_size=0,
                   codecs=replace(CFG.codecs, vocab_size=320))


def test_sizes():
    assert 48e6 < CFG.body_params < 52e6
    # The compressed core is the smaller part, and each stack has a subject.
    assert CFG.core.params < CFG.stacks.params / 3
    assert CFG.stacks.specializations == SUBJECTS
    media = sum(v for k, v in CFG.parameter_parts().items()
                if k in ("perceptual_encoders", "codec.image_encoder", "codec.audio_encoder"))
    assert 1.8e6 < media < 2.2e6
    assert not any("decoder" in k for k in CFG.parameter_parts())
    assert 90 < PRESET.tokens / CFG.body_params < 110


def test_counts_match_and_config_round_trips():
    assert IridiumConfig.from_dict(json.loads(json.dumps(CFG.to_dict()))) == CFG
    with torch.device("meta"):
        model = Iridium1(CFG)
    assert sum(p.numel() for p in model.parameters()) == CFG.n_params


def test_every_family_in_the_mix_has_a_source():
    assert sum(PRESET.mixture.values()) == pytest.approx(1.0)
    for family in PRESET.mixture:
        assert family in DOMAIN_FAMILIES or family in GENERATORS
    from iridium.data.chat_corpus import CHAT_SOURCES, CONVERSATION_LOADERS
    from iridium.data.text_corpus import SOURCES
    for spec in DOMAIN_FAMILIES.values():
        assert bool(spec.text_mix) != bool(spec.chat_mix)
        for key in spec.text_mix or ():
            assert key in SOURCES
        for key in spec.chat_mix or ():
            assert key in CHAT_SOURCES and key in CONVERSATION_LOADERS
            assert not CHAT_SOURCES[key].opt_in


def test_tool_choice_is_a_trained_target_where_generation_asks_for_it():
    spans = conversation_spans([Turn("user", "hi"), Turn("tool_call", "{}"),
                                Turn("tool_result", "{}"), Turn("assistant", "ok")])
    ids = [(int(s.payload[0]), s.supervised) for s in spans if s.modality == "control"]
    at = ids.index((chat.ASSISTANT, False))
    assert ids[at + 1] == (chat.TOOL_CALL, True)
    # The environment's marker is never a target.
    assert (chat.TOOL_RESULT, False) in ids


def test_arithmetic_held_out_really_is_held_out():
    from iridium.runtime.chat import TEXT_OFFSET
    for question, value in held_out_arithmetic(50):
        body = question.removeprefix("What is ").rstrip("?")
        parts = body.split()
        op = {"+": "+", "-": "-", "x": "*", "*": "*", "times": "*", "/": "/"}[parts[1]]
        assert is_test_pair(op, int(parts[0]), int(parts[2]))
        assert eval(f"{parts[0]} {op} {parts[2]}") == value      # noqa: S307
    item = arithmetic_drill_item(np.random.default_rng(0), "train")
    assert item.sample.meta["subject"] == "math" and len(item.sample) > 100


@pytest.mark.parametrize("index", range(len(CODE_TEMPLATES)))
def test_generated_code_really_works(index):
    rng = np.random.default_rng(index)
    for _ in range(3):
        _, code, call, expected = render_code_template(index, rng)
        scope: dict = {}
        exec(code, scope)                                         # noqa: S102
        assert eval(call, scope) == expected                      # noqa: S307


def test_code_held_out_templates_stay_out_of_training():
    held = {CODE_TEMPLATES[i][0] for i in range(len(CODE_TEMPLATES)) if i % 5 == 4}
    for seed in range(20):
        item = code_exec_item(np.random.default_rng(seed), "train")
        raw = bytes(int(t) - chat.TEXT_OFFSET for s in item.sample.spans if s.modality == "text"
                    for t in s.payload).decode()
        assert not any(request in raw for request in held)


@pytest.mark.parametrize("seed", range(12))
def test_agent_episodes_replay_on_a_real_folder(seed, tmp_path):
    """A generated transcript's file tool results match real files exactly."""
    from iridium.data import agent_corpus
    rng = np.random.default_rng(seed)
    episode = agent_corpus._EPISODES[seed % 6](rng, "TOOLS:")
    turns = {t.role for t in episode}
    assert "tool_call" in turns and episode[-1].role == "assistant"
    calls = [(json.loads(a.text), json.loads(b.text)) for a, b in zip(episode, episode[1:])
             if a.role == "tool_call"]
    # Rebuild the starting folder by asking the generator again with the same seed.
    fs_tools = {}
    rng = np.random.default_rng(seed)
    seen: list = []
    original = agent_corpus._Episode.__init__

    def capture(self, system, user, fs):
        seen.append(dict(fs.files))
        original(self, system, user, fs)

    agent_corpus._Episode.__init__ = capture
    try:
        agent_corpus._EPISODES[seed % 6](rng, "TOOLS:")
    finally:
        agent_corpus._Episode.__init__ = original
    root = LocalFS(tmp_path)
    for name, text in seen[0].items():
        root.write(name, text)
    registry = build_registry(tmp_path, groups=("files",))
    for call, result in calls:
        try:
            real = {"ok": True, "result": registry.call(call["name"], call["arguments"])}
        except WorkspaceError as exc:
            real = {"ok": False, "error": str(exc)}
        assert json.loads(json.dumps(real)) == result


def test_media_families_train_through_the_read_only_encoders():
    cfg = small_config()
    torch.manual_seed(0)
    model = Iridium1(cfg)
    assert not model.codecs.decoders
    items = [image_foundation_item(np.random.default_rng(0)),
             audio_foundation_item(np.random.default_rng(1))]
    batch = TensorBatch(collate([i.sample for i in items], continuous_dims(cfg.codecs)))
    losses, _ = model.losses(batch)
    total = sum(losses.values())
    assert math.isfinite(float(total))
    total.backward()
    for name in ("image", "audio"):
        grads = [p.grad for p in model.codecs.encoders[name].parameters()]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), name
    assert foundation_image("red", "square")[1] == (4, 4)


def test_subject_labels_supervise_the_router():
    cfg = small_config()
    model = Iridium1(cfg)
    item = agent_episode_item(np.random.default_rng(0))
    batch = TensorBatch(collate([item.sample], continuous_dims(cfg.codecs), max_length=256))
    losses, _ = model.losses(batch)
    assert float(losses["subject_routing"]) > 0


def test_shards_label_their_subject(tmp_path):
    from iridium.data.shards import Shard, ShardWriter
    writer = ShardWriter(tmp_path / "math_qa-train", "math_qa", 320)
    writer.add_ids(np.array([1, 20, 21]), np.array([0, 1, 1]))
    writer.close()
    assert Shard(tmp_path / "math_qa-train").sample(0).meta["subject"] == "math"


# -- tools ------------------------------------------------------------------


@pytest.mark.parametrize("fs_kind", ["local", "memory"])
def test_file_tools(fs_kind, tmp_path):
    tools = WorkspaceTools(LocalFS(tmp_path) if fs_kind == "local" else MemoryFS())
    tools.write_file("a/b.txt", "one\ntwo\n")
    assert tools.list_dir(".") == ["a/"] and tools.list_dir("a") == ["b.txt"]
    tools.edit_file("a/b.txt", "two", "three")
    assert tools.read_file("a/b.txt") == "one\nthree\n"
    assert tools.search_files("three") == ["a/b.txt:2: three"]
    with pytest.raises(WorkspaceError, match="exactly once"):
        tools.edit_file("a/b.txt", "missing", "x")
    with pytest.raises(WorkspaceError, match="no file"):
        tools.read_file("nope.txt")


@pytest.mark.parametrize("path", ["../x", "a/../../x", "/etc/passwd", "C:/Windows/x", "..\\x"])
def test_paths_cannot_leave_the_workspace(path, tmp_path):
    tools = WorkspaceTools(LocalFS(tmp_path / "w"))
    with pytest.raises(WorkspaceError):
        tools.write_file(path, "x")
    assert not (tmp_path / "x").exists()


def test_symlinks_cannot_leave_the_workspace(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    root = tmp_path / "w"
    root.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("no symlink permission here")
    tools = WorkspaceTools(LocalFS(root))
    with pytest.raises(WorkspaceError):
        tools.read_file("link/secret.txt")
    assert tools.search_files("s") == []


def test_python_runs_in_the_workspace_and_is_guarded(tmp_path):
    work = tmp_path / "w"
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    registry = build_registry(work, groups=("python",))
    good = registry.call("run_python", {"code": "open('made.txt','w').write('hi'); print(1 + 1)"})
    assert good["ok"] and good["stdout"].strip() == "2" and (work / "made.txt").exists()
    for code in (f"print(open({str(outside)!r}).read())",
                 f"open({str(outside)!r}, 'w').write('x')",
                 "import subprocess; subprocess.run(['whoami'])",
                 "import socket; socket.create_connection(('example.org', 80))",
                 f"import os; os.remove({str(outside)!r})"):
        bad = registry.call("run_python", {"code": code})
        assert not bad["ok"] and "not allowed" in bad["error"] or "outside" in bad["error"], code
    assert outside.read_text() == "private"
    # Stdlib imports still work under the guard.
    assert registry.call("run_python", {"code": "import json, math; print(math.floor(2.5))"})["ok"]


def test_python_timeout(tmp_path):
    from iridium.runtime.agent_tools import ToolSettings
    registry = build_registry(tmp_path, groups=("python",), settings=ToolSettings(python_timeout=2))
    assert not registry.call("run_python", {"code": "while True: pass"})["ok"]


def test_sandbox_runs_on_this_platform():
    result = Sandbox().run("emit({'x': 1})")
    assert result.ok and result.payload == {"x": 1}
    assert result.isolation["resource_limits"] == (sys.platform != "win32")


def test_groups_are_optional(tmp_path):
    assert build_registry(groups=("calc",)).names() == ["calculator"]
    assert "web_fetch" not in build_registry(tmp_path, groups=("files",))
    with pytest.raises(ValueError, match="workspace"):
        build_registry(groups=("files",))
    with pytest.raises(ValueError, match="unknown"):
        build_registry(tmp_path, groups=("shell",))


@pytest.mark.parametrize("expr,value", [("(12 + 30) * 7", 294), ("10 / 4", 2.5), ("-3 + 1", -2)])
def test_calculator(expr, value):
    assert calculate(expr) == value


@pytest.mark.parametrize("bad", ["__import__('os')", "1;2", "a+1", "2**99999"])
def test_calculator_refuses_code(bad):
    with pytest.raises((ValueError, SyntaxError)):
        calculate(bad)


@pytest.mark.parametrize("url", ["http://127.0.0.1/", "http://localhost/", "http://[::1]/",
                                 "http://169.254.169.254/latest", "http://10.0.0.1/",
                                 "file:///etc/passwd", "ftp://example.org/"])
def test_web_refuses_local_addresses_and_other_schemes(url):
    with pytest.raises(ValueError):
        _check_public(url)


def test_search_page_parsing():
    page = ('<a class="result__a" rel="x" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa">'
            'An <b>Example</b></a>')
    assert web_search("q", backend=lambda q: page) == [{"title": "An Example", "url": "https://example.org/a"}]
    with pytest.raises(ValueError):
        web_search("q", backend=lambda q: "<html></html>")


def test_tool_loop_runs_real_tools_and_keeps_the_system_block(tmp_path):
    from types import SimpleNamespace
    registry = build_registry(tmp_path, groups=("files",))
    call = format_tool_call("write_file", {"path": "x.txt", "content": "hello"})
    script = iter([SimpleNamespace(ids=[chat.TOOL_CALL] + [b + chat.TEXT_OFFSET for b in call.encode()], text=""),
                   SimpleNamespace(ids=[], text="Done.")])
    contexts = []

    def fake(model, sample, **kwargs):
        contexts.append(sample)
        return next(script)

    model = SimpleNamespace(cfg=SimpleNamespace(max_seq_len=1024))
    out = run_tool_loop(model, [Turn("system", "TOOLS: " + "x" * 400), Turn("user", "go")], registry,
                        generate_fn=fake, max_bytes=500)
    assert out.answer == "Done." and (tmp_path / "x.txt").read_text() == "hello"
    assert out.turns[0].role == "system"


# -- training path ------------------------------------------------------------


def test_time_budget_stops_cleanly_and_pruning_keeps_newest(tmp_path):
    from iridium.training.run_preset import _prune_rounds
    for i in range(5):
        (tmp_path / f"x-round{i}.pt").write_bytes(b"x")
    _prune_rounds(tmp_path, 2)
    assert sorted(p.name for p in tmp_path.glob("*.pt")) == ["x-round3.pt", "x-round4.pt"]


def test_small_model_trains_on_generated_families(tmp_path):
    from iridium.data.shards import MixedCorpus
    from iridium.training.trainer import TrainConfig, Trainer
    torch.manual_seed(0)
    from dataclasses import replace
    cfg = replace(small_config(), max_seq_len=1024)
    corpus = MixedCorpus.build({family: 8 for family in GENERATED_SUBJECT}, seed=0)
    trainer = Trainer(Iridium1(cfg), corpus, TrainConfig(steps=3, batch_size=2, log_every=1, max_length=1024),
                      out_dir=tmp_path)
    history = trainer.train()
    assert all(math.isfinite(h["total"]) for h in history if "total" in h)
