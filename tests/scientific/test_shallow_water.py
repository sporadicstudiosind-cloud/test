"""Open-channel flow, and the answer to "what if I double the water?".

The solver is checked against the analytic laws, never against itself. Two
different exact answers exist and the tests keep them apart, because answering
with the wrong one - or with "it doubles" - is the failure mode the whole
scenario was chosen to expose.
"""

import numpy as np
import pytest

from iridium.physics.shallow_water import (
    Channel,
    ShallowWater1D,
    critical_depth,
    froude,
    intervention,
    normal_depth,
)

CHANNEL = Channel(length=100.0, n_cells=200, slope=0.002, manning=0.030)


@pytest.fixture(scope="module")
def solver():
    return ShallowWater1D(CHANNEL)


def test_manning_and_critical_depth_are_different_laws():
    q = 3.0
    assert normal_depth(q, 0.002, 0.030) != pytest.approx(critical_depth(q))
    assert (2.0 ** 0.6) != pytest.approx(2.0 ** (2.0 / 3.0))
    assert 2.0 ** 0.6 == pytest.approx(1.515717, abs=1e-6)
    assert 2.0 ** (2.0 / 3.0) == pytest.approx(1.587401, abs=1e-6)


@pytest.mark.parametrize("initial_depth", [0.4, 4.0])
def test_solver_finds_normal_depth_from_either_side(solver, initial_depth):
    """Starting at the answer proves only that it is a fixed point."""
    q = 3.0
    state, _, info = solver.run_to_steady(
        q, max_time=20000, tol=1e-8, initial_depth=initial_depth
    )
    assert info["converged"], info
    assert info["steps"] > 100, "converged suspiciously fast"
    measured = solver.diagnostics(state, q)["h_upstream"]
    assert measured == pytest.approx(normal_depth(q, CHANNEL.slope, CHANNEL.manning),
                                     rel=2e-3)


def test_doubling_the_discharge_does_not_double_the_depth(solver):
    result = intervention(CHANNEL, 3.0, 6.0, max_time=20000, tol=1e-8,
                          initial_depth=0.5)
    measured = result["depth_ratio_measured"]
    assert measured == pytest.approx(result["depth_ratio_normal_law"], rel=2e-3)
    assert measured == pytest.approx(1.5157, abs=2e-3)
    assert abs(measured - 2.0) > 0.4, "the depth does not double"
    assert abs(measured - result["depth_ratio_critical_law"]) > 0.05, (
        "normal depth and critical depth must not be conflated"
    )


def test_velocity_scales_as_the_remaining_two_fifths(solver):
    """``u = q/h``, so ``u`` ratio is ``2^1 / 2^0.6 = 2^0.4``."""
    result = intervention(CHANNEL, 2.0, 4.0, max_time=20000, tol=1e-8,
                          initial_depth=0.5)
    assert result["velocity_ratio_measured"] == pytest.approx(2.0 ** 0.4, rel=3e-3)


def test_steady_state_conserves_discharge_along_the_channel(solver):
    q = 4.0
    state, _, _ = solver.run_to_steady(q, max_time=20000, tol=1e-8, initial_depth=1.0)
    diagnostics = solver.diagnostics(state, q)
    assert diagnostics["discharge_error"] < 5e-3


def test_flow_stays_subcritical_for_these_parameters(solver):
    """The inlet Froude numbers do not cross 1, so no hydraulic jump forms."""
    for q in (1.0, 3.0, 8.0):
        state, _, _ = solver.run_to_steady(q, max_time=20000, tol=1e-8,
                                           initial_depth=1.0)
        fr = solver.diagnostics(state, q)["froude_upstream"]
        assert 0.0 < fr < 1.0, f"q={q} gave Fr={fr}"


def test_critical_depth_is_where_froude_is_one():
    q = 3.0
    h_c = critical_depth(q)
    u = q / h_c
    assert float(froude(np.array([h_c]), np.array([u]))[0]) == pytest.approx(1.0, rel=1e-9)


def test_normal_depth_is_undefined_on_a_flat_bed():
    with pytest.raises(ValueError):
        normal_depth(3.0, 0.0, 0.03)
