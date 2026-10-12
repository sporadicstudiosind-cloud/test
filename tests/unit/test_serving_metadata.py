"""What the server says about the weights it loaded must be true of those weights.

The caveat string is the most load-bearing text this project serves. It is the
sentence that stops somebody reading a 10 M model's generated text as an answer.
It was hardcoded to describe the one checkpoint that shipped with the repo,
which was fine until the studio notebooks started pointing
``IRIDIUM_CHECKPOINT`` at their own output — at which point the server went on
announcing "nano · 34 M · trained" and a stored sentence about 800 steps for a
model that was neither. Reciting a true-sounding sentence about a different run
is worse than saying nothing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "serve"))

from server import describe_caveat, describe_label, _si   # noqa: E402

from iridium.config_builder import build   # noqa: E402

FALLBACK = {"label": "catalogue label", "caveat": "catalogue caveat"}


@pytest.fixture(scope="module")
def cfg():
    return build(d_model=128, core_layers=2, n_superstacks=3,
                 superstack_layers=4, d_head=32, n_kv_heads=2,
                 vocab_size=384, max_seq_len=512, name="iridium-1-probe")


def manifest(**over):
    base = {
        "train_config": {"steps": 30, "batch_size": 8},
        "data_licences": "synthetic only",
        "evaluation": {"interpolation": {
            "a": {"accuracy": 0.1, "baseline": 0.3},
            "b": {"accuracy": 0.9, "baseline": 0.2},
        }},
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------

def test_label_names_the_loaded_geometry(cfg):
    label = describe_label(cfg, FALLBACK, manifest())
    assert "iridium-1-probe" in label
    assert "10.47 M" in label
    assert "trained" in label


def test_label_says_untrained_when_no_steps_were_run(cfg):
    label = describe_label(cfg, FALLBACK, manifest(train_config={}))
    assert "untrained" in label and "· trained" not in label


def test_label_falls_back_only_without_a_manifest(cfg):
    assert describe_label(cfg, FALLBACK, {}) == "catalogue label"


def test_label_does_not_report_a_size_the_model_is_not(cfg):
    """The specific regression: a 10 M model announced as 34 M."""
    assert "34 M" not in describe_label(cfg, FALLBACK, manifest())


# ---------------------------------------------------------------------------
# caveat
# ---------------------------------------------------------------------------

def test_caveat_reports_this_run_not_a_remembered_one():
    text = describe_caveat(FALLBACK, manifest())
    assert "30 training steps" in text
    assert "800 steps" not in text
    assert "synthetic only" in text


def test_caveat_counts_families_that_beat_the_baseline():
    text = describe_caveat(FALLBACK, manifest())
    assert "1 of 2 task families" in text


def test_caveat_says_the_text_is_noise_when_nothing_beat_the_baseline():
    m = manifest(evaluation={"interpolation": {
        "a": {"accuracy": 0.0, "baseline": 0.3},
        "b": {"accuracy": 0.2, "baseline": 0.2},
    }})
    text = describe_caveat(FALLBACK, m)
    assert "0 of 2" in text
    assert "noise" in text and "telemetry" in text


def test_caveat_does_not_claim_noise_when_the_model_learned():
    m = manifest(evaluation={"interpolation": {
        "a": {"accuracy": 0.9, "baseline": 0.2},
        "b": {"accuracy": 0.8, "baseline": 0.1},
    }})
    text = describe_caveat(FALLBACK, m)
    assert "2 of 2" in text
    assert "noise" not in text


def test_caveat_admits_when_nothing_was_graded():
    text = describe_caveat(FALLBACK, manifest(evaluation={}))
    assert "unverified" in text
    assert "task families" not in text


def test_caveat_admits_unrecorded_data_provenance():
    m = manifest()
    del m["data_licences"]
    assert "unrecorded" in describe_caveat(FALLBACK, m)


def test_caveat_survives_a_manifest_with_nulls():
    """Real manifests carry nulls where a stage did not run; a metadata helper
    that raises takes the whole serving path down with it."""
    m = {"train_config": None, "evaluation": None, "data_licences": None}
    text = describe_caveat(FALLBACK, m)
    assert text and text.endswith(".")


def test_caveat_falls_back_only_without_a_manifest():
    assert describe_caveat(FALLBACK, {}) == "catalogue caveat"


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (999, "999"), (10_469_509, "10.47 M"), (1_007_202_723, "1.01 B"), (34_000, "34.00 k"),
])
def test_si_formatting(n, expected):
    assert _si(n) == expected
