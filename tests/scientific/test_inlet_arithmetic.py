"""The §11.2 inlet table, recomputed rather than transcribed.

Two things are being protected here. First, that the numbers in the
architecture document are derivable from its stated assumptions. Second, and
more importantly, the *scope* claims attached to them: doubling discharge does
not double every dimensionless group, and these inlet Froude numbers do not
cross 1.
"""

import math

import pytest

from iridium.contracts.units import (
    DIMENSIONLESS,
    REGISTRY,
    Dimension,
    Quantity,
)

# Stated assumptions of the synthetic inlet example.
AREA = 2.0            # m^2
LENGTH = 1.0          # m, hydraulic length
DEPTH = 1.0           # m, prescribed
RHO = 1000.0          # kg/m^3
MU = 0.001            # Pa.s
SIGMA = 0.072         # N/m
G = 9.81              # m/s^2
Q_OLD = 3.0           # m^3/s


def inlet_state(q: float) -> dict[str, float]:
    u = q / AREA
    return {
        "Q": q,
        "U": u,
        "mass_inflow": RHO * q,
        "Re": RHO * u * LENGTH / MU,
        "Fr": u / math.sqrt(G * DEPTH),
        "We": RHO * u**2 * LENGTH / SIGMA,
    }


BASE = inlet_state(Q_OLD)
DOUBLED = inlet_state(2 * Q_OLD)


@pytest.mark.parametrize(
    "key,expected",
    [
        ("Q", 3.0),
        ("U", 1.5),
        ("mass_inflow", 3000.0),
        ("Re", 1_500_000.0),
        ("We", 31_250.0),
    ],
)
def test_baseline_row(key, expected):
    assert BASE[key] == pytest.approx(expected, rel=1e-12)


def test_baseline_froude():
    assert BASE["Fr"] == pytest.approx(0.479, abs=5e-4)


@pytest.mark.parametrize(
    "key,expected",
    [
        ("Q", 6.0),
        ("U", 3.0),
        ("mass_inflow", 6000.0),
        ("Re", 3_000_000.0),
        ("We", 125_000.0),
    ],
)
def test_doubled_row(key, expected):
    assert DOUBLED[key] == pytest.approx(expected, rel=1e-12)


def test_doubled_froude():
    assert DOUBLED["Fr"] == pytest.approx(0.958, abs=5e-4)


def test_doubling_discharge_does_not_double_every_group():
    """Re and Fr scale with U; We scales with U squared."""
    assert DOUBLED["Re"] / BASE["Re"] == pytest.approx(2.0)
    assert DOUBLED["Fr"] / BASE["Fr"] == pytest.approx(2.0)
    assert DOUBLED["We"] / BASE["We"] == pytest.approx(4.0)


def test_inlet_froude_does_not_cross_one():
    """The counterexample to 'doubling inflow almost always approaches Fr = 1'.

    Both states are subcritical at the inlet under these assumptions. Whatever
    happens at a downstream fall is a separate question that this table cannot
    answer.
    """
    assert BASE["Fr"] < 1.0
    assert DOUBLED["Fr"] < 1.0


def test_dimensionless_groups_are_actually_dimensionless():
    """Check the group definitions against the unit algebra, not by eye."""
    velocity = Dimension.of(length=1, time=-1)
    density = Dimension.of(mass=1, length=-3)
    viscosity = Dimension.of(mass=1, length=-1, time=-1)
    surface_tension = Dimension.of(mass=1, time=-2)
    length = Dimension.of(length=1)
    accel = Dimension.of(length=1, time=-2)

    reynolds = density * velocity * length / viscosity
    weber = density * (velocity ** 2) * length / surface_tension
    froude_squared = (velocity ** 2) / (accel * length)

    assert reynolds == DIMENSIONLESS
    assert weber == DIMENSIONLESS
    assert froude_squared == DIMENSIONLESS


def test_mass_inflow_units_resolve():
    q = Quantity(Q_OLD, REGISTRY["m3/s"])
    rho = Quantity(RHO, REGISTRY["kg/m3"])
    product_dimension = rho.unit.dimension * q.unit.dimension
    assert product_dimension == REGISTRY["kg/s"].dimension
    assert rho.value * q.value == pytest.approx(BASE["mass_inflow"])


def test_scenario_b_bulk_temperature_rise():
    """§13.6 Scenario B's bounded arithmetic check: 1000 W into 0.05 kg/s."""
    power = 1000.0                  # W
    mass_flow = 0.05                # kg/s
    heat_capacity = 4000.0          # J/(kg.K)
    delta_t = power / (mass_flow * heat_capacity)
    assert delta_t == pytest.approx(5.0, rel=1e-12)

    # Units resolve to kelvin.
    watt = REGISTRY["W"].dimension
    kg_s = REGISTRY["kg/s"].dimension
    cp = REGISTRY["J/(kg.K)"].dimension
    assert watt / (kg_s * cp) == Dimension.of(temperature=1)
