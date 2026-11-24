"""Every batch must be built on the model's device — provable without a GPU.

This is the bug that shipped. `TensorBatch` defaults to CPU, so any evaluation
or generation helper that forgets `device=` works perfectly on a CPU machine
and dies on the first GPU with *"Expected all tensors to be on the same
device"*, raised inside an embedding lookup twelve frames from the line that
actually made the CPU tensor.

A CPU-only test suite cannot reproduce that by running it, which is precisely
why every test passed while the notebook failed. So these tests check the
*wiring* instead: they record the `device` argument of every `TensorBatch`
built during a call and demand it match the model's own device. That runs
anywhere, and it fails the moment someone drops the argument again.
"""

import torch
import pytest

from iridium.codecs import bank as bank_module
from iridium.codecs.bank import continuous_dims
from iridium.codecs.spans import Sample, collate, text_span
from iridium.config_builder import build
from iridium.evaluation.harness import evaluate
from iridium.model.iridium1 import Iridium1
from iridium.runtime.device import device_of, generator_for
from iridium.runtime.generate import generate
from iridium.training.datasets import build_corpus


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = build(d_model=64, core_layers=2, n_superstacks=2, superstack_layers=3,
                d_head=32, n_kv_heads=1, vocab_size=384, max_seq_len=256,
                name="iridium-1-device-probe")
    return Iridium1(cfg).eval()


@pytest.fixture
def recorded(monkeypatch):
    """Capture the ``device`` every TensorBatch is constructed with."""
    seen: list = []
    original = bank_module.TensorBatch.__init__

    def spy(self, batch, device=None, dtype=None):
        seen.append(device)
        return original(self, batch, device=device, dtype=dtype)

    monkeypatch.setattr(bank_module.TensorBatch, "__init__", spy)
    return seen


def _same(device, model) -> bool:
    if device is None:
        return False
    return torch.device(device) == device_of(model)


def test_generate_builds_its_batch_on_the_models_device(model, recorded):
    generate(model, Sample([text_span("depth")]), max_new_tokens=2)
    assert recorded, "generate() built no batch at all"
    assert all(_same(d, model) for d in recorded), recorded


def test_evaluation_builds_every_batch_on_the_models_device(model, recorded):
    """All five graders, in one pass — each has its own construction site, and
    each one of them was wrong."""
    corpus = build_corpus(20, seed=0, split="test", mixture={
        "channel_depth": 0.3, "channel_intervention": 0.2, "false_premise": 0.2,
        "field_rollout": 0.15, "scene_goal": 0.15,
    })
    evaluate(model, corpus, max_per_family=2)
    assert len(recorded) >= 5, f"only {len(recorded)} batches built; graders were skipped"
    assert all(_same(d, model) for d in recorded), recorded


def test_routing_analysis_builds_on_the_models_device(model, recorded):
    from iridium.evaluation.routing import analyse
    from iridium.training.datasets import BatchLoader
    corpus = build_corpus(8, seed=1, split="test", mixture={"channel_depth": 1.0})
    analyse(model, BatchLoader(corpus, model.cfg.codecs, 4, 0,
                               device=str(device_of(model))))
    assert recorded and all(_same(d, model) for d in recorded), recorded


def test_no_evaluation_path_builds_a_batch_without_a_device(model, recorded):
    """`device=None` means "wherever torch defaults", which is the bug."""
    corpus = build_corpus(10, seed=2, split="test", mixture={"channel_depth": 1.0})
    evaluate(model, corpus, max_per_family=2)
    generate(model, Sample([text_span("hello")]), max_new_tokens=2)
    assert None not in recorded, "a batch was built with no device at all"


# ---------------------------------------------------------------------------
# the generator, which fails the same way
# ---------------------------------------------------------------------------

def test_generator_for_matches_the_device():
    gen = generator_for("cpu", 0)
    assert gen.device.type == "cpu"


def test_generate_seeds_from_a_device_matched_generator(model, monkeypatch):
    """``torch.randn(..., device='cuda', generator=torch.Generator())`` raises,
    so sampling has to seed on the model's device, not on the default one."""
    from iridium.runtime import generate as generate_module

    seen: list = []
    original = generate_module.generator_for

    def spy(device, seed):
        seen.append(torch.device(device))
        return original(device, seed)

    # generate() is wrapped in @torch.no_grad(), so its __globals__ belong to
    # the decorator; patch the module the function actually looks names up in.
    monkeypatch.setattr(generate_module, "generator_for", spy)
    generate(model, Sample([text_span("x")]), max_new_tokens=2, temperature=0.8)
    assert seen == [device_of(model)], seen


# ---------------------------------------------------------------------------
# the property itself, checked directly
# ---------------------------------------------------------------------------

def test_device_of_reads_the_parameters(model):
    assert device_of(model) == next(model.parameters()).device


def test_device_of_survives_a_module_with_no_parameters():
    assert device_of(torch.nn.Identity()) == torch.device("cpu")


def test_a_batch_built_without_a_device_is_detectably_on_cpu(model):
    """The failure mode, stated as a property: a default-constructed batch is
    CPU, so on any non-CPU model it is a mismatch waiting to happen."""
    from iridium.codecs.bank import TensorBatch
    dims = continuous_dims(model.cfg.codecs)
    batch = TensorBatch(collate([Sample([text_span("abc")])], dims))
    assert batch.discrete.device == torch.device("cpu")
    explicit = TensorBatch(collate([Sample([text_span("abc")])], dims),
                           device=device_of(model))
    assert explicit.discrete.device == device_of(model)
