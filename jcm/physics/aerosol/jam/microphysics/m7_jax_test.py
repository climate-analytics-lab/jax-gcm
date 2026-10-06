"""The M7-JAX core adapter on columns (jcm[m7] extra)."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.requires_extra("m7")

NLEV, NCOL = 6, 3
DT = 720.0


def _fortran_ihpbl(dse, height, ustar, coriolis):
    """Literal transcription of vdiff.f90:737-759 for one column (1-based)."""
    klev = len(dse)
    zcor = max(abs(coriolis), 5.0e-5)
    zhdyn = min(height[0], 0.3 * ustar / zcor)
    ihpblc = ihpbld = klev
    for jk in range(klev - 1, 0, -1):
        zds = dse[jk - 1] - dse[klev - 1]
        zdz = height[jk - 1] - zhdyn
        if ihpblc == klev and zds > 0.0:
            ihpblc = jk
        if ihpbld == klev and zdz >= 0.0:
            ihpbld = jk
    return min(ihpblc, ihpbld)


def test_pbl_top_level_matches_the_fortran_loop():
    from jcm.physics.aerosol.jam.microphysics.m7_jax import pbl_top_level

    rng = np.random.default_rng(3)
    nlev, ncol = 12, 200
    height = np.sort(rng.uniform(10.0, 20000.0, (nlev, ncol)), axis=0)[::-1]
    dse = 3.0e5 + rng.normal(0.0, 400.0, (nlev, ncol)) + 2.0 * height * rng.uniform(-1, 1, ncol)
    ustar = rng.uniform(0.0, 0.8, ncol)
    coriolis = rng.uniform(-1.4e-4, 1.4e-4, ncol)
    got = np.asarray(pbl_top_level(dse, height, ustar, coriolis))
    want = [_fortran_ihpbl(dse[:, j], height[:, j], ustar[j], coriolis[j]) for j in range(ncol)]
    np.testing.assert_array_equal(got, want)


def test_clear_sky_humidity_is_hams():
    from jcm.physics.aerosol.jam.microphysics.m7_jax import clear_sky_relative_humidity

    q, qs = np.array([0.008, 0.008, 0.008, 0.02]), np.array([0.01, 0.01, 0.01, 0.01])
    cc = np.array([0.0, 0.5, 1.0, 0.3])
    got = np.asarray(clear_sky_relative_humidity(q, qs, cc))
    ccl = np.minimum(cc, 1 - 1e-10)
    want = np.clip(np.maximum(0, (q - qs * ccl) / (1 - ccl)) / qs, 0, 1)
    np.testing.assert_allclose(got, want, rtol=2e-7)  # float32 when x64 is off
    assert got[1] == pytest.approx(0.6)


def _column():
    import jax.numpy as jnp

    from jcm.physics_interface import PhysicsState

    from jcm.physics.aerosol.jam.microphysics.m7_jax import M7_CLASSES, M7_COMPONENTS
    from jcm.physics.aerosol.jam.tracer_layout import gas_name, mass_name, number_name

    rng = np.random.default_rng(7)
    shape = (NLEV, NCOL)
    p = np.linspace(2.0e4, 1.0e5, NLEV)[:, None] * np.ones(shape)
    t = np.linspace(225.0, 290.0, NLEV)[:, None] * np.ones(shape)
    rho = p / (287.0 * t)
    tracers = {}
    # Realistic continental mixing ratios [kg/kg] and numbers [1/kg].
    typical = {"so4": 1e-9, "bc": 2e-10, "oc": 1e-9, "ss": 1e-9, "du": 5e-9}
    for sp, cl in M7_COMPONENTS:
        tracers[mass_name(sp, cl)] = jnp.asarray(typical[sp] * rng.uniform(0.2, 2.0, shape))
    nnum = {"ns": 1e8, "ks": 1e9, "as": 3e8, "cs": 1e5, "ki": 5e8, "ai": 1e5, "ci": 1e4}
    for cl in M7_CLASSES:
        tracers[number_name(cl)] = jnp.asarray(nnum[cl] * rng.uniform(0.5, 2.0, shape))
    tracers[gas_name("h2so4")] = jnp.asarray(5e-12 * rng.uniform(0.5, 2.0, shape))
    state = PhysicsState(
        u_wind=jnp.zeros(shape), v_wind=jnp.zeros(shape), temperature=jnp.asarray(t),
        specific_humidity=jnp.asarray(0.6 * 0.622 * 611.0 * np.exp(17.27 * (t - 273.15) / (t - 35.86)) / p),
        geopotential=jnp.asarray(9.81 * np.linspace(11000.0, 100.0, NLEV)[:, None] * np.ones(shape)),
        normalized_surface_pressure=jnp.ones(NCOL), tracers=tracers)
    run = {k: jnp.zeros(shape) for k in tracers}
    run[gas_name("h2so4")] = jnp.asarray(2e-16 * rng.uniform(0.5, 2.0, shape))  # gas-chem production
    run[mass_name("bc", "ki")] = jnp.asarray(np.full(shape, 1e-15))             # an emission earlier this step
    diagnostics = {"_dt_seconds": DT, "air_density": jnp.asarray(rho),
                   "pressure_full": jnp.asarray(p), "height_full": jnp.asarray(
                       np.linspace(11000.0, 100.0, NLEV)[:, None] * np.ones(shape)),
                   "_tendency_run": {"tracers": run}}
    return state, diagnostics


def test_adapter_conserves_species_and_sulfur():
    import jax

    from jcm.physics.aerosol.jam.microphysics.m7_jax import (
        M7_COMPONENTS, M7JaxMicrophysics)
    from jcm.physics.aerosol.jam.tracer_layout import gas_name, mass_name

    core = M7JaxMicrophysics(nucleation_scheme=1, organic_scheme=1)
    core._coriolis = jax.numpy.asarray(2 * 7.292e-5 * np.sin([-0.5, 0.0, 0.8]))
    state, diag = _column()
    tend, out = core(state, diag, None, None)
    run = diag["_tendency_run"]["tracers"]

    def total(sp):
        return sum(np.asarray(tend.tracers[mass_name(s, c)]) for s, c in M7_COMPONENTS if s == sp)

    for sp in ("bc", "oc", "ss", "du"):                      # pure transfers between classes
        np.testing.assert_allclose(total(sp), 0.0, atol=1e-12 * typical_scale(sp))
    # Sulfur [molecules]: aerosol SO4 at 96.0631 g/mol, gas at H2SO4's molar mass.
    so4_mol = total("so4") / 96.0631
    gas_mol = np.asarray(tend.tracers[gas_name("h2so4")]) / 98.0784
    # The gas tendency the core returns excludes the running production
    # (summed tendency = core + run), so the closed sulfur budget is:
    np.testing.assert_allclose(so4_mol + gas_mol, 0.0,
                               atol=1e-10 * np.abs(np.asarray(run[gas_name("h2so4")]) / 98.0784).max())
    js = out["_jam_state"]
    assert js.r_wet.shape == (7, NLEV, NCOL)
    assert np.all(np.isfinite(np.asarray(js.r_wet))) and np.all(np.asarray(js.rho) > 0)
    # Something actually happened: new particles and ageing out of KI.
    assert np.asarray(tend.tracers["n_ns"]).max() > 0
    assert np.asarray(tend.tracers["n_ki"]).min() < 0


def typical_scale(sp):
    return {"bc": 2e-10, "oc": 1e-9, "ss": 1e-9, "du": 5e-9}[sp] / DT


def test_adapter_kazil_scheme_runs_end_to_end():
    """``nucleation_scheme=2`` (jax-gcm#1017 Kazil/GCR task) through the full
    adapter: needs HAM_INPUT_DIR (both the PARNUC and O'Brien tables -- see
    gcr_ionisation.py) AND an m7-jax release with load_kazil_lovejoy_table
    (not yet in the jcm[m7] pin at time of writing -- the lazy-import site
    in m7_jax.py's __init__ this exercises). Skips cleanly when either is
    missing, which is the common case (CI's extras-tests job and a plain
    checkout both lack one or the other right now).
    """
    import os

    ham_input_dir = os.environ.get("HAM_INPUT_DIR")
    if not ham_input_dir:
        pytest.skip("HAM_INPUT_DIR not set")
    try:
        import m7_jax.nucleation  # noqa: F401
        if not hasattr(m7_jax.nucleation, "load_kazil_lovejoy_table"):
            pytest.skip("installed m7-jax lacks load_kazil_lovejoy_table")
    except ImportError:
        pytest.skip("m7-jax not installed")

    import jax

    from jcm.physics.aerosol.jam.microphysics.m7_jax import M7JaxMicrophysics

    core = M7JaxMicrophysics(nucleation_scheme=2, organic_scheme=0)
    core._coriolis = jax.numpy.asarray(2 * 7.292e-5 * np.sin([-0.5, 0.0, 0.8]))
    core._lat = jax.numpy.asarray([-0.5, 0.0, 0.8])
    core._lon = jax.numpy.asarray([0.1, 1.5, -2.0])
    state, diag = _column()

    class _Solar:
        calendar_year = jax.numpy.asarray(2000.0)
        day_of_year = jax.numpy.asarray(100.0)
        tyear = jax.numpy.asarray(100.0 / 366.0)

    class _Forcing:
        solar = _Solar()
        forest_fraction = None

    tend, out = core(state, diag, _Forcing(), None)
    assert np.all(np.isfinite(np.asarray(tend.tracers["g_h2so4"])))
    assert np.all(np.isfinite(np.asarray(out["_jam_state"].r_wet)))


def test_adapter_refuses_kazil_without_ham_input_dir_and_wrong_population(monkeypatch):
    from jcm.physics.aerosol.jam.microphysics.m7_jax import M7JaxMicrophysics
    from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC

    # nucleation_scheme=2 is supported (jax-gcm#1017 Kazil/GCR task) given
    # HAM_INPUT_DIR; what is always refused is constructing it without one.
    monkeypatch.delenv("HAM_INPUT_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="HAM_INPUT_DIR"):
        M7JaxMicrophysics(nucleation_scheme=2)
    with pytest.raises(ValueError, match="M7 population"):
        M7JaxMicrophysics(spec=MAM4_SPEC)
