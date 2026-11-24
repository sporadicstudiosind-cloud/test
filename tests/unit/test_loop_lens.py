import torch

from iridium.codecs.spans import Sample, text_span
from iridium.config import get_config
from iridium.model.iridium1 import Iridium1
from iridium.runtime.loop_lens import format_readings, read_loops


def _model():
    torch.manual_seed(0)
    cfg = get_config("tiny")
    return Iridium1(cfg)


def test_lens_is_read_only_and_reports_every_loop():
    model = _model()
    model.train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    sample = Sample([text_span("hello loop", offset=16)])
    loops = model.cfg.router.max_loops
    readings = read_loops(model, sample, n_loops=loops, top_k=2)
    assert len(readings) == len(sample)
    for r in readings:
        assert len(r.loops) == loops
        assert 1 <= r.settled_at <= loops
        assert abs(sum(lp.stop_p for lp in r.loops) - 1.0) < 1e-4
        assert all(len(lp.top) == 2 for lp in r.loops)
    assert readings[0].actual_next == "e"
    assert model.training
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert "loop 1:" in format_readings(readings)
