"""A small checkpoint should be usable in chat without corrupting history."""

from types import SimpleNamespace

import pytest
import torch

from iridium import cli
from iridium.config import get_config
from iridium.model.iridium1 import Iridium1
from iridium.runtime import chat


def _controls(sample):
    return [int(span.payload[0]) for span in sample.spans
            if span.modality == "control"]


def test_chat_budget_drops_orphaned_assistant_and_honors_override(monkeypatch):
    seen = []

    def fake_generate(_model, sample, **kwargs):
        seen.append((sample, kwargs))
        return SimpleNamespace(text="yes")

    monkeypatch.setattr("iridium.runtime.generate.generate", fake_generate)
    model = SimpleNamespace(cfg=SimpleNamespace(max_seq_len=32))
    session = chat.ChatSession(model, max_new_tokens=4)
    assert session.send("a" * 12) == "yes"
    assert session.send("b" * 12, max_new_tokens=6) == "yes"

    # The earlier user message no longer fits. Its assistant answer must not
    # become the first turn in the prompt merely because that answer is short.
    assert _controls(seen[-1][0]) == [chat.BOS, chat.USER, chat.ASSISTANT]
    assert seen[-1][1]["max_new_tokens"] == 6
    assert session.history() == [
        ("user", "a" * 12), ("assistant", "yes"),
        ("user", "b" * 12), ("assistant", "yes"),
    ]


def test_chat_rejects_oversize_without_changing_history(monkeypatch):
    monkeypatch.setattr("iridium.runtime.generate.generate", lambda *args, **kw:
                        pytest.fail("generation should not run"))
    session = chat.ChatSession(SimpleNamespace(cfg=SimpleNamespace(max_seq_len=32)),
                               max_new_tokens=4)
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        session.send("x" * 30)
    assert session.history() == []
    assert session.seed == 0


def test_failed_generation_does_not_commit_a_user_turn(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("model failed")

    monkeypatch.setattr("iridium.runtime.generate.generate", fail)
    session = chat.ChatSession(SimpleNamespace(cfg=SimpleNamespace(max_seq_len=32)),
                               max_new_tokens=4)
    with pytest.raises(RuntimeError, match="model failed"):
        session.send("hi")
    assert session.history() == []
    assert session.seed == 0


def test_chat_checkpoint_loader_accepts_plain_inference_file(tmp_path):
    cfg = get_config("tiny")
    original = Iridium1(cfg).eval()
    path = tmp_path / "chat-inference.pt"
    torch.save({
        "state_dict": original.state_dict(),
        "manifest": {"model_config": cfg.to_dict(), "completed_steps": 2},
    }, path)
    loaded, manifest = cli._load_chat_checkpoint(path, "cpu")
    assert loaded.cfg.name == cfg.name
    assert manifest["completed_steps"] == 2
    first_key = next(iter(original.state_dict()))
    torch.testing.assert_close(loaded.state_dict()[first_key],
                               original.state_dict()[first_key])


def test_legacy_checkpoint_loads_for_chat_and_finetuning_only_with_known_gap(tmp_path):
    from iridium.training.chat_finetune import load_initial_model

    cfg = get_config("tiny")
    state = Iridium1(cfg).state_dict()
    quantity = [key for key in state if key.startswith("codecs.encoders.quantity.")
                or key.startswith("codecs.decoders.quantity.")]
    assert len(quantity) == 5
    for key in quantity:
        state.pop(key)
    path = tmp_path / "legacy.pt"
    torch.save({"state_dict": state, "manifest": {"model_config": cfg.to_dict()}}, path)
    cli._load_chat_checkpoint(path, "cpu")
    load_initial_model(path)

    state.pop(next(iter(state)))
    torch.save({"state_dict": state, "manifest": {"model_config": cfg.to_dict()}}, path)
    with pytest.raises((ValueError, RuntimeError), match="missing"):
        cli._load_chat_checkpoint(path, "cpu")
    with pytest.raises((ValueError, RuntimeError), match="missing"):
        load_initial_model(path)


def test_chat_command_defaults_to_bundled_checkpoint():
    args = cli.build_parser().parse_args(["chat", "--prompt", "hello"])
    assert args.func is cli.cmd_chat
    assert args.checkpoint.endswith("nano-phase1-fp16.pt")
    assert args.max_new_tokens == 96
    assert cli._terminal_text("a\x1b[31m\n") == "a\\u001b[31m\n"
