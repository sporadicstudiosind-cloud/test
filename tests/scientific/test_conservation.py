"""Conservation mechanics. Gate: §18.3 "Conservation mechanics".

This suite is the executable form of the §3 correction to ORRERY's claim that
"a coarse conservative solve plus a learned correction cannot invent mass".

It cannot, *if* the correction is applied to oriented shared faces. It very much
can if the correction is applied per cell. Both are demonstrated below.
"""

import numpy as np
import pytest

from iridium.physics.fluxes import (
    Budget,
    ConservationError,
    Mesh1D,
    apply_cell_update,
    apply_face_update,
    cfl_number,
    check_admissible,
    integrate_budget,
    total_mass,
    upwind_flux,
)

MESH = Mesh1D(n_cells=64, dx=1.0 / 64)


def initial_field() -> np.ndarray:
    x = (np.arange(MESH.n_cells) + 0.5) * MESH.dx
    return 1.0 + 0.5 * np.sin(2 * np.pi * x)


def test_closed_periodic_advection_conserves_to_machine_precision():
    """FP64 closed fixture, 1e-10 relative tolerance per the gate table."""
    u = initial_field()
    m0 = total_mass(u, MESH)
    dt = 0.4 * MESH.dx / 1.0
    for _ in range(400):
        u = apply_face_update(u, upwind_flux(u, 1.0, periodic=True), MESH, dt)
    drift = abs(total_mass(u, MESH) - m0) / m0
    assert drift < 1e-10, f"relative mass drift {drift:.3e}"


def test_learned_face_correction_still_conserves():
    """A *large, arbitrary* learned correction on shared faces cannot add mass.

    The correction below is deliberately unphysical and of the same order as
    the flux itself. Conservation survives because face i is subtracted from
    one cell and added to its neighbour: the interior telescopes exactly.
    """
    rng = np.random.default_rng(0)
    u = initial_field()
    m0 = total_mass(u, MESH)
    dt = 0.2 * MESH.dx

    for _ in range(200):
        base = upwind_flux(u, 1.0, periodic=True)
        correction = rng.normal(scale=1.0, size=MESH.n_faces)
        correction[-1] = correction[0]          # periodic wrap shares one face
        u = apply_face_update(u, base + correction, MESH, dt)

    drift = abs(total_mass(u, MESH) - m0) / m0
    assert drift < 1e-10, f"relative mass drift {drift:.3e}"


def test_learned_cell_correction_does_not_conserve():
    """The contrast case: this is what the archived spec implicitly permitted."""
    rng = np.random.default_rng(0)
    u = initial_field()
    m0 = total_mass(u, MESH)
    dt = 0.2 * MESH.dx

    for _ in range(200):
        u = apply_face_update(u, upwind_flux(u, 1.0, periodic=True), MESH, dt)
        u = apply_cell_update(u, rng.normal(scale=1.0, size=MESH.n_cells), dt)

    drift = abs(total_mass(u, MESH) - m0) / m0
    assert drift > 1e-6, (
        "per-cell corrections are expected to break conservation; if this "
        "passes, the contrast the architecture relies on is not being tested"
    )


def test_face_flux_antisymmetry_is_what_makes_it_work():
    """Direct statement: interior divergence contributions sum to zero."""
    rng = np.random.default_rng(1)
    flux = rng.normal(size=MESH.n_faces)
    flux[-1] = flux[0]
    divergence = (flux[1:] - flux[:-1]) / MESH.dx
    assert abs(np.sum(divergence) * MESH.dx) < 1e-12


def test_wrong_face_count_rejected():
    with pytest.raises(ConservationError):
        apply_face_update(initial_field(), np.zeros(10), MESH, 0.001)


def test_open_domain_stores_mass_without_violating_conservation():
    """§9.4 and the §3 correction: an open waterfall domain is not closed.

    Requiring constant total mass here — as the archived evaluation section did
    — would fail a perfectly correct solve.
    """
    u = np.zeros(MESH.n_cells)
    dt = 0.2 * MESH.dx
    inflow, outflow = 2.0, 0.5

    stored_start = total_mass(u, MESH)
    for _ in range(100):
        # F is the flux in the +x direction through each face, so a positive
        # left-face flux carries mass *into* cell 0 and a positive right-face
        # flux carries it out of the last cell.
        flux = np.zeros(MESH.n_faces)
        flux[0] = inflow
        flux[-1] = outflow
        u = apply_face_update(u, flux, MESH, dt)

    budget = Budget(
        stored_start=stored_start,
        stored_end=total_mass(u, MESH),
        inflow=inflow * dt * 100,
        outflow=outflow * dt * 100,
        sources=0.0,
    )
    assert budget.actual_change > 0.0          # mass accumulated
    assert budget.closes(tolerance=1e-10, floor=1e-12)


def test_budget_detects_a_real_leak():
    leaky = Budget(stored_start=0.0, stored_end=5.0, inflow=2.0, outflow=0.5,
                   sources=0.0)
    assert not leaky.closes(tolerance=1e-6, floor=1e-12)
    assert leaky.absolute_error == pytest.approx(3.5)


def test_normalization_floor_prevents_meaningless_relative_error():
    """§9.4: a near-zero reference quantity must not manufacture a huge ratio."""
    tiny = Budget(stored_start=0.0, stored_end=1e-18, inflow=0.0, outflow=0.0,
                  sources=0.0)
    with pytest.raises(ConservationError):
        tiny.relative_error(floor=0.0)
    assert tiny.relative_error(floor=1.0) == pytest.approx(1e-18)


def test_integrate_budget_accounts_for_sources():
    u0 = np.zeros(MESH.n_cells)
    sources = np.full(MESH.n_cells, 3.0)
    dt = 0.01
    u1 = apply_face_update(u0, np.zeros(MESH.n_faces), MESH, dt, sources=sources)
    budget = integrate_budget(u0, u1, MESH, 0.0, 0.0, dt, sources=sources)
    assert budget.closes(tolerance=1e-12, floor=1e-12)


def test_manufactured_solution_converges_at_first_order():
    """§9.5: error must fall under refinement at an order consistent with the
    scheme. Upwind advection is first order, so the observed order should sit
    near 1 -- not at some value asserted in advance."""
    def run(n_cells: int) -> float:
        mesh = Mesh1D(n_cells, 1.0 / n_cells)
        x = (np.arange(n_cells) + 0.5) * mesh.dx
        u = np.sin(2 * np.pi * x)
        c, t_end = 1.0, 0.1
        dt = 0.4 * mesh.dx / c
        steps = int(round(t_end / dt))
        dt = t_end / steps
        for _ in range(steps):
            u = apply_face_update(u, upwind_flux(u, c, periodic=True), mesh, dt)
        exact = np.sin(2 * np.pi * (x - c * t_end))
        return float(np.sqrt(np.mean((u - exact) ** 2)))

    resolutions = [64, 128, 256, 512]
    errors = [run(n) for n in resolutions]
    assert all(b < a for a, b in zip(errors, errors[1:])), errors

    orders = [
        np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)
    ]
    # Three systematic refinement levels, observed order reported not assumed.
    assert 0.7 < orders[-1] < 1.3, f"observed orders {orders}"


def test_cfl_number():
    assert cfl_number(2.0, 0.1, 0.4) == pytest.approx(0.5)


def test_admissibility_reports_worst_violation():
    ok, worst = check_admissible(np.array([0.2, 0.9]), lower=0.0, upper=1.0)
    assert ok and worst == 0.0

    bad, worst = check_admissible(np.array([-0.3, 1.4]), lower=0.0, upper=1.0)
    assert not bad
    assert worst == pytest.approx(0.4)
