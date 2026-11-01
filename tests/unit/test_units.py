"""Unit and dimension semantics. Gate: Architecture §18.3 "Frames and units"."""

from fractions import Fraction

import pytest

from iridium.contracts.units import (
    DIMENSIONLESS,
    PRESSURE,
    REGISTRY,
    VELOCITY,
    VOLUME_FLOW,
    Dimension,
    Quantity,
    UnitError,
    parse_dimension,
)


def test_base_dimension_order_is_canonical():
    from iridium.contracts.units import BASE_DIMENSIONS

    assert BASE_DIMENSIONS == (
        "mass",
        "length",
        "time",
        "current",
        "temperature",
        "amount",
        "luminous_intensity",
    )


def test_velocity_is_length_over_time():
    assert Dimension.of(length=1) / Dimension.of(time=1) == VELOCITY
    assert str(VELOCITY) == "L·T^-1"


def test_rational_exponents_supported():
    root_length = Dimension.of(length=Fraction(1, 2))
    assert root_length * root_length == Dimension.of(length=1)


def test_pressure_algebra():
    force_per_area = Dimension.of(mass=1, length=1, time=-2) / Dimension.of(length=2)
    assert force_per_area == PRESSURE


@pytest.mark.parametrize(
    "unit,value,expected_si",
    [("m", 2.0, 2.0), ("mm", 2000.0, 2.0), ("km", 0.002, 2.0)],
)
def test_scale_conversion_round_trip(unit, value, expected_si):
    q = Quantity(value, REGISTRY[unit])
    assert q.to_si() == pytest.approx(expected_si, rel=1e-12)
    assert q.to("m").value == pytest.approx(expected_si, rel=1e-12)


def test_affine_units_need_an_offset():
    """degC is not a scaled kelvin. Architecture §5.1 calls this out explicitly."""
    celsius = REGISTRY["degC"]
    assert celsius.is_affine
    assert Quantity(0.0, celsius).to_si() == pytest.approx(273.15)
    assert Quantity(100.0, celsius).to("K").value == pytest.approx(373.15)
    # A pure scale conversion would give 0 K here; the offset is what prevents that.
    assert Quantity(0.0, celsius).to("K").value != 0.0


def test_fahrenheit_round_trip():
    q = Quantity(212.0, REGISTRY["degF"])
    assert q.to("degC").value == pytest.approx(100.0, abs=1e-9)
    assert q.to("K").value == pytest.approx(373.15, abs=1e-9)


def test_round_trip_through_every_registered_unit():
    for unit in REGISTRY.values():
        original = Quantity(7.25, unit)
        si = original.to_si()
        back = unit.from_si(si)
        assert back == pytest.approx(7.25, rel=1e-12, abs=1e-12), unit.name


def test_incompatible_conversion_rejected():
    with pytest.raises(UnitError):
        Quantity(1.0, REGISTRY["m"]).to("s")


def test_non_numeric_is_not_dimensionless():
    """The ORRERY correction: text and dimensionless numbers are different states."""
    text = Quantity(0.0, None)
    number = Quantity(0.5, REGISTRY["1"])

    assert not text.is_numeric
    assert not text.is_dimensionless          # units do not apply at all
    assert number.is_numeric
    assert number.is_dimensionless            # zero exponent vector

    with pytest.raises(UnitError):
        text.to_si()


def test_parse_dimension_distinguishes_null_from_empty():
    assert parse_dimension(None) is None
    assert parse_dimension({}) == DIMENSIONLESS
    assert parse_dimension({"length": 3, "time": -1}) == VOLUME_FLOW


def test_parse_dimension_rejects_unknown_base():
    with pytest.raises(UnitError):
        parse_dimension({"charge": 1})


def test_parse_dimension_sequence_length_enforced():
    with pytest.raises(UnitError):
        parse_dimension([1, 0, 0])
