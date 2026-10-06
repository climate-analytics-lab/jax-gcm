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
    return {"so4": 1e-9, "bc": 2e-10, "oc": 1e-9, "ss": 1e-9, "du": 5e-9}[sp] / DT


def test_adapter_default_core_dtype_is_float64_and_bit_identical():
    """``core_dtype=None`` (the default) is unperturbed by the new option.

    #1017 task 3's invariant is that adding the float32 ``core_dtype``
    choice must not change the default path by even one bit. Compare the
    implicit default against an explicit ``core_dtype="float64"`` core on
    the same column with exact (not ``allclose``) equality.
    """
    import jax

    from jcm.physics.aerosol.jam.microphysics.m7_jax import (
        M7_CLASSES, M7_COMPONENTS, M7JaxMicrophysics)
    from jcm.physics.aerosol.jam.tracer_layout import gas_name, mass_name, number_name

    # Construct the core BEFORE building the column: the constructor is
    # what flips the process-wide jax_enable_x64 flag on, and `_column()`
    # builds its state with plain `jnp.asarray` -- calling it first would
    # silently hand back float32 state (jax_enable_x64 still off from
    # whatever an earlier test/import left it at) and make this test's
    # own `out_dtype` plumbing look broken when it is not.
    coriolis = jax.numpy.asarray(2 * 7.292e-5 * np.sin([-0.5, 0.0, 0.8]))
    core_default = M7JaxMicrophysics(nucleation_scheme=1, organic_scheme=1)
    core_default._coriolis = coriolis
    core_explicit = M7JaxMicrophysics(nucleation_scheme=1, organic_scheme=1, core_dtype="float64")
    core_explicit._coriolis = coriolis
    state, diag = _column()
    tend_default, _ = core_default(state, diag, None, None)
    tend_explicit, _ = core_explicit(state, diag, None, None)

    keys = ([mass_name(sp, cl) for sp, cl in M7_COMPONENTS]
            + [number_name(cl) for cl in M7_CLASSES] + [gas_name("h2so4")])
    for key in keys:
        a, b = tend_default.tracers[key], tend_explicit.tracers[key]
        assert a.dtype == jax.numpy.float64
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=key)


def test_adapter_float32_core_matches_float64_within_measured_tolerance():
    """``core_dtype="float32"`` vs the default float64 core, same column.

    Per-field tolerance is anchored to each field's own TYPICAL magnitude
    (the mixing-ratio/number scale ``_column()`` draws from for that
    species/class), not to the sample's own computed tendency: several
    class-transfer tendencies -- coarse-mode sea-salt in particular -- are
    themselves numerical noise around a true value of ~0 in this test
    (both the float64 and float32 values sit many orders of magnitude
    below the typical scale), so an atol keyed to ``abs(tend64).max()``
    would be demanding agreement between two unrelated noise floors
    instead of a physically meaningful comparison.

    Measured (this column, nucleation_scheme=1, organic_scheme=1): with
    ``atol = 1e-6 * typical_scale`` and ``rtol = 5e-3``, the worst
    per-field fraction of the allowed budget used is ~0.37 (h2so4 gas
    tendency, which has the fewest cancelling contributions). This test's
    atol is 10x looser than that measurement.
    """
    import jax

    from jcm.physics.aerosol.jam.microphysics.m7_jax import (
        M7_CLASSES, M7_COMPONENTS, M7JaxMicrophysics)
    from jcm.physics.aerosol.jam.tracer_layout import gas_name, mass_name, number_name

    typical_number = {"ns": 1e8, "ks": 1e9, "as": 3e8, "cs": 1e5, "ki": 5e8, "ai": 1e5, "ci": 1e4}
    rtol = 5e-3

    # Construct both cores before building the column -- see the comment
    # in test_adapter_default_core_dtype_is_float64_and_bit_identical.
    coriolis = jax.numpy.asarray(2 * 7.292e-5 * np.sin([-0.5, 0.0, 0.8]))
    core64 = M7JaxMicrophysics(nucleation_scheme=1, organic_scheme=1, core_dtype="float64")
    core64._coriolis = coriolis
    core32 = M7JaxMicrophysics(nucleation_scheme=1, organic_scheme=1, core_dtype="float32")
    core32._coriolis = coriolis
    state, diag = _column()
    tend64, _ = core64(state, diag, None, None)
    tend32, _ = core32(state, diag, None, None)

    for sp, cl in M7_COMPONENTS:
        key = mass_name(sp, cl)
        np.testing.assert_allclose(
            np.asarray(tend32.tracers[key]), np.asarray(tend64.tracers[key]),
            rtol=rtol, atol=1e-5 * typical_scale(sp), err_msg=key)
    for cl in M7_CLASSES:
        key = number_name(cl)
        np.testing.assert_allclose(
            np.asarray(tend32.tracers[key]), np.asarray(tend64.tracers[key]),
            rtol=rtol, atol=1e-5 * typical_number[cl] / DT, err_msg=key)
    key = gas_name("h2so4")
    np.testing.assert_allclose(
        np.asarray(tend32.tracers[key]), np.asarray(tend64.tracers[key]),
        rtol=rtol, atol=1e-5 * 5e-12 / DT, err_msg=key)


def test_adapter_float32_core_conserves_species_and_sulfur():
    """Species and sulfur conservation hold at float32 precision too.

    Measured residuals (core_dtype="float32", same column as the matching
    float64 test ``test_adapter_conserves_species_and_sulfur``): the
    bc/oc/ss/du cross-class transfer residuals are ~4e-7-9e-7 of their
    typical mass-tendency scale; the sulfur budget residual is ~2e-3 of
    the gas-chem production scale. This test's atol is an order of
    magnitude looser than each measurement (float64's equivalent test
    uses 1e-12/1e-10 of the same scales -- six orders of magnitude
    tighter, consistent with float32 vs float64 machine epsilon).
    """
    import jax

    from jcm.physics.aerosol.jam.microphysics.m7_jax import (
        M7_COMPONENTS, M7JaxMicrophysics)
    from jcm.physics.aerosol.jam.tracer_layout import gas_name, mass_name

    core = M7JaxMicrophysics(nucleation_scheme=1, organic_scheme=1, core_dtype="float32")
    core._coriolis = jax.numpy.asarray(2 * 7.292e-5 * np.sin([-0.5, 0.0, 0.8]))
    state, diag = _column()
    tend, out = core(state, diag, None, None)
    run = diag["_tendency_run"]["tracers"]

    def total(sp):
        return sum(np.asarray(tend.tracers[mass_name(s, c)]) for s, c in M7_COMPONENTS if s == sp)

    for sp in ("bc", "oc", "ss", "du"):                      # pure transfers between classes
        np.testing.assert_allclose(total(sp), 0.0, atol=1e-5 * typical_scale(sp))
    so4_mol = total("so4") / 96.0631
    gas_mol = np.asarray(tend.tracers[gas_name("h2so4")]) / 98.0784
    np.testing.assert_allclose(
        so4_mol + gas_mol, 0.0,
        atol=1e-2 * np.abs(np.asarray(run[gas_name("h2so4")]) / 98.0784).max())

    js = out["_jam_state"]
    assert js.r_wet.shape == (7, NLEV, NCOL)
    # The float32 core must not introduce NaN or an unphysical negative
    # state (#1017 task 3 STOP condition).
    for field in (js.r_dry, js.r_wet, js.rho, js.kappa, js.mass, js.number):
        assert np.all(np.isfinite(np.asarray(field)))
    assert np.all(np.asarray(js.rho) > 0)
    assert np.all(np.asarray(js.mass) >= 0) and np.all(np.asarray(js.number) >= 0)


def test_adapter_refuses_kazil_and_wrong_population():
    from jcm.physics.aerosol.jam.microphysics.m7_jax import M7JaxMicrophysics
    from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC

    with pytest.raises(NotImplementedError, match="Kazil"):
        M7JaxMicrophysics(nucleation_scheme=2)
    with pytest.raises(ValueError, match="M7 population"):
        M7JaxMicrophysics(spec=MAM4_SPEC)
