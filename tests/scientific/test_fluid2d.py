"""The Navier-Stokes solver, checked against an exact solution.

Taylor-Green is a closed-form solution of the *full nonlinear* 2-D equations:
the advection term is balanced exactly by the pressure gradient, so the flow
decays as pure diffusion while still satisfying the whole system. A solver with
a broken advection term or a broken projection does not reproduce it.
"""

import numpy as np
import pytest

from iridium.physics.fluid2d import NavierStokes2D, project, taylor_green
from iridium.physics.operators import (
    divergence,
    divergence_norm,
    kinetic_energy,
    project_divergence_free,
    vorticity,
)

NU = 0.05
N = 32


@pytest.fixture(scope="module")
def solver():
    return NavierStokes2D(N, nu=NU)


def test_reproduces_the_taylor_green_exact_solution(solver):
    state = taylor_green(N, 0.0, NU)
    dt, horizon = 0.005, 0.5
    for _ in range(int(horizon / dt)):
        state = solver.step(state, dt)
    exact = taylor_green(N, state.t, NU)
    err = float(np.sqrt(np.mean((state.u - exact.u) ** 2 + (state.v - exact.v) ** 2)))
    ref = float(np.sqrt(np.mean(exact.u ** 2 + exact.v ** 2)))
    assert err / ref < 1e-10, f"relative L2 error {err / ref:.3e}"


def test_energy_decay_matches_the_analytic_rate(solver):
    """Taylor-Green energy decays as ``exp(-4 nu t)``."""
    state = taylor_green(N, 0.0, NU)
    e0 = kinetic_energy(state.u, state.v)
    dt, horizon = 0.005, 0.4
    for _ in range(int(horizon / dt)):
        state = solver.step(state, dt)
    predicted = e0 * np.exp(-4.0 * NU * state.t)
    assert kinetic_energy(state.u, state.v) == pytest.approx(predicted, rel=1e-9)


def test_velocity_stays_divergence_free(solver):
    """Projection is a constraint, not a penalty: the residual is round-off."""
    rng = np.random.default_rng(0)
    u = rng.normal(size=(N, N))
    v = rng.normal(size=(N, N))
    state = project(type(taylor_green(N))(u, v, 0.0))
    for _ in range(50):
        state = solver.step(state, 0.002)
        assert divergence_norm(state.u, state.v) < 1e-10


def test_leray_projection_is_idempotent():
    rng = np.random.default_rng(1)
    u, v = rng.normal(size=(N, N)), rng.normal(size=(N, N))
    a = project_divergence_free(u, v)
    b = project_divergence_free(*a)
    assert np.allclose(a[0], b[0], atol=1e-12)
    assert np.allclose(a[1], b[1], atol=1e-12)


def test_a_curl_free_field_projects_to_zero():
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    phi = np.cos(xx) * np.sin(2 * yy)
    from iridium.physics.operators import ddx, ddy

    u, v = project_divergence_free(ddx(phi), ddy(phi))
    assert np.max(np.abs(u)) < 1e-12
    assert np.max(np.abs(v)) < 1e-12


def test_divergence_norm_is_unit_invariant():
    """The escalation criterion must not depend on the user's choice of units.

    An absolute threshold on ``||div u||``, as the source plan specified, fails
    this: the same flow in cm/s has 100x the divergence of the same flow in m/s.
    """
    rng = np.random.default_rng(2)
    u, v = project_divergence_free(rng.normal(size=(N, N)), rng.normal(size=(N, N)))
    u = u + 1e-6 * rng.normal(size=(N, N))
    base = divergence_norm(u, v, 2 * np.pi)
    scaled = divergence_norm(100 * u, 100 * v, 2 * np.pi)
    assert scaled == pytest.approx(base, rel=1e-9)
    raw = float(np.sqrt(np.mean(divergence(100 * u, 100 * v) ** 2)))
    raw_base = float(np.sqrt(np.mean(divergence(u, v) ** 2)))
    assert raw / raw_base == pytest.approx(100.0, rel=1e-6)


def test_vorticity_of_a_known_field():
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    u, v = np.cos(xx) * np.sin(yy), -np.sin(xx) * np.cos(yy)
    expected = -2.0 * np.cos(xx) * np.cos(yy)
    assert np.allclose(vorticity(u, v), expected, atol=1e-10)


def test_cfl_timestep_keeps_the_solver_stable(solver):
    state = taylor_green(N, 0.0, NU)
    dt = solver.cfl_dt(state)
    assert dt > 0
    for _ in range(100):
        state = solver.step(state, dt)
    assert np.isfinite(state.u).all()
    assert kinetic_energy(state.u, state.v) < 1.0
