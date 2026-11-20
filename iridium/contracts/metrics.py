"""Typed metric tables and deterministic numeric reconciliation.

Architecture §7.6 and §13.5: Iridium-1 writes the prose; a deterministic pass
checks that every numeric claim in that prose resolves to an entry in an exact
metric table, with a compatible unit. Unbound or contradicted numbers are
flagged before release.

This is the mechanism that catches the class of error catalogued in the
archived blueprint reconciliation: fluent, specific, mutually inconsistent
quantities that no amount of reading reliably detects.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .units import REGISTRY, Quantity, Unit, UnitError


class ReconciliationError(ValueError):
    pass


@dataclass(frozen=True)
class MetricEntry:
    """One exact value derived from stored arrays, with its provenance."""

    key: str
    value: float
    unit: str
    source: str                      # e.g. "fields/u.npy#inlet_face_integral"
    scenario_revision: str
    uncertainty: float | None = None

    def as_quantity(self) -> Quantity:
        try:
            return Quantity(self.value, REGISTRY[self.unit])
        except KeyError as exc:
            raise UnitError(f"unregistered unit: {self.unit}") from exc


@dataclass
class MetricTable:
    scenario_revision: str
    entries: dict[str, MetricEntry] = field(default_factory=dict)

    def add(self, entry: MetricEntry) -> None:
        if entry.scenario_revision != self.scenario_revision:
            raise ReconciliationError(
                f"entry {entry.key!r} belongs to revision "
                f"{entry.scenario_revision!r}, table is {self.scenario_revision!r}"
            )
        if entry.key in self.entries:
            raise ReconciliationError(f"duplicate metric key: {entry.key}")
        self.entries[entry.key] = entry

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def __getitem__(self, key: str) -> MetricEntry:
        return self.entries[key]


@dataclass(frozen=True)
class Finding:
    kind: str          # "unbound" | "mismatch" | "unknown_key" | "unit_mismatch"
    detail: str
    span: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.detail}  (in: {self.span!r})"


# A bound claim looks like {{metric:key}} or {{metric:key|3 significant digits}}.
_CLAIM = re.compile(r"\{\{metric:([A-Za-z0-9_.\-]+)(?:\|([^}]*))?\}\}")

# Anything that opens a claim. A marker that opens but does not parse - a typo
# in the format separator, an unclosed brace - would otherwise be invisible to
# both `render` and `reconcile`: the substitution never fires, so the numeral
# never appears in the prose, so the bare-number scan finds nothing to object
# to, and a report containing no numbers at all passes the release gate
# reporting zero findings. Found exactly that way. See `_MALFORMED` below.
_CLAIM_OPEN = re.compile(r"\{\{\s*metric\b[^}]*\}*")

# A bare number, optionally followed by a unit token. Used to find prose
# numerals that were never bound to the table.
_BARE_NUMBER = re.compile(
    r"(?<![\w.]) (-? \d{1,3}(?:,\d{3})+ (?:\.\d+)? | -? \d+ (?:\.\d+)? (?:[eE][-+]?\d+)? ) "
    r"[ \t]* ([A-Za-z][A-Za-z0-9_./·^%-]*)? ",
    re.VERBOSE,
)

# Numerals that are structural rather than factual claims.
_EXEMPT_CONTEXTS = re.compile(
    r"^(?:section|figure|table|step|revision|version|level|scenario|milestone|"
    r"stage|chapter|page|item|note|equation|eq|fig|no|id|m0|m1|m2)$",
    re.IGNORECASE,
)


def render(table: MetricTable, template: str) -> str:
    """Substitute ``{{metric:key}}`` claims with exact values from the table."""
    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in table:
            raise ReconciliationError(f"unknown metric key in template: {key}")
        entry = table[key]
        fmt = match.group(2)
        value = f"{entry.value:{fmt}}" if fmt else f"{entry.value:g}"
        return f"{value} {entry.unit}" if entry.unit != "1" else value

    return _CLAIM.sub(_sub, template)


def reconcile(
    table: MetricTable,
    report_text: str,
    *,
    rel_tol: float = 1e-9,
    allow_bare: frozenset[str] = frozenset(),
) -> list[Finding]:
    """Check a rendered report against the metric table.

    Returns findings; an empty list means the report is releasable under
    Architecture §18.3's "Numeric reporting" gate.

    Every numeral in the prose must either (a) equal a table value within
    ``rel_tol``, (b) appear in ``allow_bare`` as an explicitly declared input or
    structural constant, or (c) sit in a structural context such as "Section 3".
    """
    findings: list[Finding] = []

    for match in _CLAIM_OPEN.finditer(report_text):
        if not _CLAIM.fullmatch(match.group(0)):
            findings.append(
                Finding(
                    "malformed_claim",
                    match.group(0),
                    "claim marker did not parse, so no value was substituted; "
                    "the separator before a format spec is '|', as in "
                    "{{metric:key|.4f}}",
                )
            )

    for match in _CLAIM.finditer(report_text):
        findings.append(
            Finding(
                "unbound",
                f"unsubstituted claim placeholder for key {match.group(1)!r}",
                match.group(0),
            )
        )

    all_values = [(e.key, e.value, e.unit) for e in table.entries.values()]

    for match in _BARE_NUMBER.finditer(report_text):
        raw, unit_token = match.group(1), match.group(2)
        # The unit charset admits '.' and '/' so that "Pa.s" and "m3/s" parse,
        # which means sentence punctuation gets swept up. Strip it back off.
        if unit_token:
            unit_token = unit_token.rstrip(".,;:)")
            if not unit_token:
                unit_token = None
        text = f"{raw} {unit_token}" if unit_token else raw
        # The token after a numeral is only a *unit* if it is one. Otherwise it
        # is an ordinary English word ("9.457e-09 for the flux head") and the
        # numeral should be matched without a unit constraint.
        unit_token = unit_token if (unit_token and unit_token in REGISTRY) else None
        if raw in allow_bare or text in allow_bare:
            continue
        if unit_token and _EXEMPT_CONTEXTS.match(unit_token):
            continue
        prefix = report_text[max(0, match.start() - 24): match.start()].strip()
        last_word = prefix.split()[-1] if prefix.split() else ""
        if _EXEMPT_CONTEXTS.match(last_word.rstrip(".:")):
            continue

        number = float(raw.replace(",", ""))
        matched = any(
            _close(number, value, rel_tol)
            and (unit_token is None or unit == unit_token or unit == "1")
            for _key, value, unit in all_values
        )
        if not matched:
            near = [
                (key, value, unit)
                for key, value, unit in all_values
                if unit_token is not None and unit == unit_token
            ]
            if near:
                findings.append(
                    Finding(
                        "mismatch",
                        f"{number:g} {unit_token} does not match any table value "
                        f"with that unit; closest keys: "
                        + ", ".join(f"{k}={v:g}" for k, v, _ in near[:4]),
                        text,
                    )
                )
            else:
                findings.append(
                    Finding("unbound", f"numeral {number:g} is not bound to a metric", text)
                )

    return findings


def _significant_digits(written: float) -> int:
    """Significant digits actually present in a written value."""
    text = repr(float(written))
    mantissa = text.split("e")[0].split("E")[0].lstrip("-+")
    digits = mantissa.replace(".", "").lstrip("0")
    return max(1, len(digits.rstrip("0")) or 1)


def _close(a: float, b: float, rel_tol: float) -> bool:
    if math.isclose(a, b, rel_tol=rel_tol, abs_tol=0.0):
        return True
    # Reports round. Accept a written value that equals the table entry rounded
    # to the precision actually written. This must work in scientific notation
    # too: rounding 9.45711e-09 to "decimal places" gives zero, so the
    # comparison is done in significant figures.
    sig = _significant_digits(a)
    if b == 0.0:
        return a == 0.0
    try:
        return float(f"{b:.{sig}g}") == float(f"{a:.{sig}g}")
    except (ValueError, OverflowError):
        return False


def check_dependency_staleness(
    table: MetricTable, current_scenario_revision: str
) -> list[Finding]:
    """Architecture §11: a scenario edit marks derived quantities stale.

    Reporting a metric computed on a superseded revision is the failure mode
    where a correct-looking report is paired with a stale field.
    """
    if table.scenario_revision != current_scenario_revision:
        return [
            Finding(
                "stale",
                f"metric table is for revision {table.scenario_revision!r} but the "
                f"delivered scenario is {current_scenario_revision!r}",
                table.scenario_revision,
            )
        ]
    return []
