"""One resident model, many streams, nothing crossing between them.

The vision asked for a single always-on instance that every conversation
arrives at. The safety property that makes that acceptable is that a stream's
output must not depend on any other stream's existence or content — bit for
bit, not statistically.
"""

import numpy as np
import pytest
import torch

from iridium.codecs.spans import Sample, text_span
from iridium.config import get_config
from iridium.model.iridium1 import Iridium1
from iridium.runtime.registry import StreamRegistry
from iridium.runtime.scheduler import SchedulerPolicy
from iridium.runtime.service import IridiumService


@pytest.fixture(scope="module")
def service_factory():
    torch.manual_seed(0)
    model = Iridium1(get_config("tiny")).double().eval()

    def make():
        return IridiumService(model, SchedulerPolicy(token_budget=64, min_share=0.2))

    return make


def test_a_streams_output_does_not_depend_on_its_neighbours(service_factory):
    alone = service_factory()
    alone.admit("target", "alice", Sample([text_span("the quick brown fox")]))
    alone.run(max_ticks=8)
    solo = torch.cat([h for h in alone.registry.get("target").emitted], dim=0)

    crowded = service_factory()
    crowded.admit("noise-1", "mallory",
                  Sample([text_span("SECRET-CANARY-ABCDEF-0123456789")]))
    crowded.admit("target", "alice", Sample([text_span("the quick brown fox")]))
    crowded.admit("noise-2", "mallory", Sample([text_span("another unrelated stream")]))
    crowded.run(max_ticks=8)
    shared = torch.cat([h for h in crowded.registry.get("target").emitted], dim=0)

    assert torch.equal(solo, shared), (
        "a stream's output changed when other streams were present"
    )


def test_no_tensor_storage_is_reachable_from_two_streams(service_factory):
    service = service_factory()
    for i in range(4):
        service.admit(f"s{i}", f"owner{i % 2}", Sample([text_span(f"stream {i} text")]))
    service.run(max_ticks=6)
    report = service.registry.aliasing_report()
    assert report["isolated"], report["collisions"]
    assert report["distinct_storages"] > 0


def test_reading_another_owners_stream_raises(service_factory):
    service = service_factory()
    service.admit("private", "alice", Sample([text_span("hello")]))
    with pytest.raises(PermissionError):
        service.registry.get("private", owner="bob")


def test_closing_a_stream_releases_its_cache(service_factory):
    service = service_factory()
    service.admit("temp", "alice", Sample([text_span("some text here")]))
    service.run(max_ticks=4)
    assert service.registry.total_cache_bytes() > 0
    service.close("temp")
    assert service.registry.total_cache_bytes() == 0
    assert "temp" not in service.registry


def test_registry_evicts_rather_than_overflowing():
    registry = StreamRegistry(max_streams=3)
    for i in range(5):
        registry.open(f"s{i}", "owner")
    assert len(registry) <= 3
