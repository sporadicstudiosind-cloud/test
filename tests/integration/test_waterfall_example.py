"""The originating scenario, run end to end, checked against the analytic laws."""

import json
from pathlib import Path

import pytest

from examples.waterfall import build
from iridium.physics.shallow_water import Channel


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("waterfall")
    channel = Channel(length=100.0, n_cells=120, slope=0.002, manning=0.030)
    return build(3.0, 2.0, channel, 0.5, out, n_frames=3), out


def test_the_report_releases_with_no_unbound_numerals(result):
    manifest, out = result
    assert manifest["reconciliation_findings"] == []
    assert manifest["status"] == "complete"


def test_every_numeral_in_the_report_came_from_the_table(result):
    manifest, out = result
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "{{metric" not in text, "an unsubstituted claim marker survived"
    assert f"{manifest['metrics']['depth_ratio_measured']:.4f}" in text


def test_the_answer_is_the_manning_law_not_doubling(result):
    manifest, _ = result
    m = manifest["metrics"]
    assert m["depth_ratio_measured"] == pytest.approx(m["depth_ratio_normal_law"], rel=3e-3)
    assert abs(m["depth_ratio_measured"] - 2.0) > 0.4
    assert m["depth_ratio_critical_law"] != pytest.approx(m["depth_ratio_normal_law"])


def test_artifacts_carry_an_evidence_level_each(result):
    manifest, out = result
    assert manifest["artifacts"]["report.md"] == "verified"
    # The vorticity field is decoration and must say so.
    assert manifest["artifacts"]["vorticity_frames.npy"] == "illustrative"
    assert manifest["omissions"], "a result with no stated omissions is suspect"
    for name in ("report.md", "manifest.json", "scene.py", "depth_after.npy"):
        assert (out / name).exists()


def test_the_emitted_blender_script_is_real_bpy(result):
    _, out = result
    script = (out / "scene.py").read_text(encoding="utf-8")
    assert "import bpy" in script
    assert "primitive_plane_add" in script
