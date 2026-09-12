"""Numeric reconciliation. Gate: §18.3 "Numeric reporting".

The regression fixture is the contradiction catalogued in the archived
blueprint reconciliation: a report that states one discharge in its setup and a
different one in its results table. Prose review missed it; this pass does not.
"""

import pytest

from iridium.contracts.metrics import (
    MetricEntry,
    MetricTable,
    ReconciliationError,
    check_dependency_staleness,
    reconcile,
    render,
)

REV = "rev_a1b2c3"


@pytest.fixture
def table() -> MetricTable:
    t = MetricTable(REV)
    t.add(MetricEntry("inlet.discharge.baseline", 3.0, "m3/s",
                      "fields/u.npy#inlet_face_integral", REV))
    t.add(MetricEntry("inlet.discharge.doubled", 6.0, "m3/s",
                      "fields/u_doubled.npy#inlet_face_integral", REV))
    t.add(MetricEntry("inlet.velocity.baseline", 1.5, "m/s",
                      "derived:Q/A", REV))
    t.add(MetricEntry("inlet.velocity.doubled", 3.0, "m/s",
                      "derived:Q/A", REV))
    t.add(MetricEntry("stored.volume.change", 12.4, "m3",
                      "fields/alpha.npy#volume_integral", REV))
    return t


def test_render_substitutes_exact_values(table):
    out = render(table, "Discharge rises to {{metric:inlet.discharge.doubled}}.")
    assert out == "Discharge rises to 6 m3/s."


def test_render_rejects_unknown_key(table):
    with pytest.raises(ReconciliationError, match="unknown metric key"):
        render(table, "{{metric:inlet.discharge.tripled}}")


def test_clean_report_reconciles(table):
    report = render(
        table,
        "Inlet discharge rises from {{metric:inlet.discharge.baseline}} to "
        "{{metric:inlet.discharge.doubled}}. Mean inlet velocity rises from "
        "{{metric:inlet.velocity.baseline}} to {{metric:inlet.velocity.doubled}}. "
        "Stored liquid volume changes by {{metric:stored.volume.change}}.",
    )
    assert reconcile(table, report) == []


def test_unsubstituted_placeholder_is_flagged(table):
    findings = reconcile(table, "Discharge is {{metric:inlet.discharge.baseline}}.")
    assert any(f.kind == "unbound" for f in findings)


def test_contradictory_discharge_is_caught(table):
    """The archived blueprint stated Q = 3.0 in setup and 3.66 in its results."""
    report = (
        "The supplied baseline has an inlet discharge of 3 m3/s. "
        "Doubling it gives 3.66 m3/s rising to 7.32 m3/s."
    )
    findings = reconcile(table, report)
    flagged = {f.span for f in findings}
    assert any("3.66" in s for s in flagged)
    assert any("7.32" in s for s in flagged)
    assert all(f.kind == "mismatch" for f in findings)


def test_fabricated_percentage_is_flagged(table):
    """A '+142% aeration' style claim with no backing metric must not pass."""
    findings = reconcile(table, "Spray detachment increases by 142 percent.")
    assert findings
    assert any("142" in f.span for f in findings)


def test_structural_numerals_are_not_flagged(table):
    report = render(
        table,
        "See Section 3 and Table 2. Discharge is {{metric:inlet.discharge.baseline}}.",
    )
    assert reconcile(table, report) == []


def test_declared_inputs_can_be_allowed(table):
    report = "Gravity was taken as 9.81 m/s2 and the domain width as 2 m."
    assert reconcile(table, report) != []
    assert reconcile(
        table, report, allow_bare=frozenset({"9.81 m/s2", "2 m"})
    ) == []


def test_rounded_values_accepted(table):
    t = MetricTable(REV)
    t.add(MetricEntry("froude", 0.478655, "1", "derived", REV))
    assert reconcile(t, "The inlet Froude number is 0.479.") == []


def test_stale_revision_detected(table):
    assert check_dependency_staleness(table, REV) == []
    stale = check_dependency_staleness(table, "rev_zzz")
    assert stale and stale[0].kind == "stale"


def test_table_rejects_foreign_revision():
    t = MetricTable(REV)
    with pytest.raises(ReconciliationError, match="belongs to revision"):
        t.add(MetricEntry("x", 1.0, "m", "src", "other_rev"))


def test_table_rejects_duplicate_key(table):
    with pytest.raises(ReconciliationError, match="duplicate"):
        table.add(MetricEntry("inlet.discharge.baseline", 99.0, "m3/s", "s", REV))


def test_every_entry_records_its_source(table):
    """Artifact lineage: a metric with no source cannot back a report claim."""
    for entry in table.entries.values():
        assert entry.source
        assert entry.scenario_revision == REV


def test_scientific_notation_rounds_by_significant_figures():
    """Regression: a rounded value in scientific notation must reconcile.

    Found by the first-slice run. Rounding 9.45711e-09 to "decimal places"
    gives zero, so the comparison has to be in significant figures.
    """
    t = MetricTable(REV)
    t.add(MetricEntry("drift", 9.45711e-09, "1", "src", REV))
    assert reconcile(t, "Mass drift is 9.457e-09 for the flux head.") == []
    assert reconcile(t, "Mass drift is 8.100e-09 for the flux head.") != []


def test_following_english_word_is_not_treated_as_a_unit():
    """Regression: 'for' after a numeral is prose, not a unit."""
    t = MetricTable(REV)
    t.add(MetricEntry("x", 0.0555, "1", "src", REV))
    assert reconcile(t, "The head reaches 0.0555 for the held-out split.") == []


def test_unregistered_unit_does_not_silently_match_wrong_dimension():
    t = MetricTable(REV)
    t.add(MetricEntry("q", 3.0, "m3/s", "src", REV))
    # Right number, wrong registered unit -> still flagged.
    assert reconcile(t, "The discharge is 3 kg/s.") != []
