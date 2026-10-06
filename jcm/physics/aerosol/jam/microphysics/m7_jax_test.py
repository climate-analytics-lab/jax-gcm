"""The M7-JAX core adapter on columns (jcm[m7] extra)."""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.requires_extra("m7")

NLEV, NCOL = 6, 3
DT = 720.0

# nsnucl=2 end-to-end chain reference (jax-gcm#1017 Kazil/GCR task, W5):
# read_obrien_gcr_ipr -> gcr_ionization -> nucl_kazil_lovejoy, against the
# UNMODIFIED compiled Fortran on the REAL O'Brien and PARNUC tables. Built
# by /scr/dwatsonparris/ham-m7/w5/build_hamnucl2_chain.py (own scratch, not
# part of this repository); see hamnucl2_README.md/hamnucl2_provenance.json
# next to the npz for the exact build and the non-embedded-table rationale
# (the real PARNUC table is ~200MB; only the per-case 2^5-corner bracket
# m7_jax.nucleation.kazil_lovejoy's own bisection search touches is
# embedded, which reproduces the full-table call bit-for-bit).
_HAMNUCL2_REF = (Path(__file__).resolve().parents[4] / "data" / "test"
                 / "echam_cloud_reference" / "hamnucl2_chain.npz")


def _recompute_ion_pair_rate(ref, table):
    """Run gcr_ion_pair_rate per case against ``table`` (either the embedded
    or a freshly-read O'Brien table) -- stage 1 of the chain.
    """
    from jcm.physics.aerosol.jam.chemistry.gcr_ionisation import gcr_ion_pair_rate

    n = len(ref["in/ion_pair_rate"])
    recomputed = np.empty(n)
    for i in range(n):
        date = ref["meta/scenario_date"][i]
        year, month, day = int(date[0]), int(date[1]), int(date[2])
        doy = (datetime.date(year, month, day) - datetime.date(year, 1, 1)).days + 1
        lat_rad = np.radians(ref["meta/column_lat"][i])
        lon_rad = np.radians(ref["meta/column_lon"][i])
        pressure_1 = np.asarray([ref["meta/pressure"][i]])
        temperature_1 = np.asarray([ref["in/temperature"][i]])
        out = gcr_ion_pair_rate(
            lat_rad, lon_rad, pressure_1, temperature_1,
            float(ref["meta/scenario_psolact"][i]), table,
            year=float(year), day_of_year=float(doy), grav=9.80665)
        recomputed[i] = float(np.asarray(out)[0])
    return recomputed


def _write_real_format_obrien_file(path, vcr, mcd, ipr):
    """Write a solmin/solmax file in ``read_obrien_gcr_ipr``'s own blocked
    text format (mo_ham_gcrion.f90:120-206: one skipped header line, then
    per cutoff-rigidity block a skipped line, an "F11.6,A" cutoff-rigidity
    line, two skipped lines, and ``len(mcd)`` "mcd ipr" data lines),
    from REAL numbers (``vcr``/``mcd``/``ipr``, shape ``(len(vcr),
    len(mcd))``) rather than synthetic ones -- unlike
    gcr_ionisation_test.py's own ``_write_synthetic_format_test_file``,
    this one is used to give the MARKED adapter test (below) a fully
    offline ``HAM_INPUT_DIR`` built from the embedded hamnucl2 reference,
    not to test the parser's format tolerance. ``.17g`` round-trips any
    float64 exactly; the reader's own regex/``split()`` parsing does not
    care about field width.
    """
    lines = ["HEADER real_obrien_excerpt"]
    for block in range(len(vcr)):
        lines.append("skip")
        lines.append(f"{float(vcr[block]):.17g} GV vertical cutoff rigidity")
        lines.append("skip")
        lines.append("g cm-2      cm-3 s-1")
        for row in range(len(mcd)):
            lines.append(f"{float(mcd[row]):.17g} {float(ipr[block, row]):.17g}")
    Path(path).write_text("\n".join(lines) + "\n")


def _write_tiny_parnuc_netcdf(path, axes, log_pfr):
    """Write a PARNUC table ``load_kazil_lovejoy_table`` can read, from a
    REAL per-case excerpt (``axes`` shape ``(5, 2)``, ``log_pfr`` shape
    ``(2, 2, 2, 2, 2)``, both from the embedded hamnucl2 reference's
    ``in/kazil_table_axes``/``in/kazil_table_log_pfr``) -- a degenerate
    2-points-per-axis table, not the real ~200MB (40,40,40,20,40) one.
    ``kazil_lovejoy``'s own bracket search (``_bracket``, a clip then
    ``searchsorted``) always resolves a 2-element strictly-increasing axis
    to (0, 1) and then CLAMPS the query into ``[axis[0], axis[-1]]`` before
    bracketing, so this is valid for ANY query the adapter's column
    produces, not just the one the excerpt's own case was built from --
    this is why the adapter does not need to be fed a column that
    reproduces that case's exact (T, RH, H2SO4, sink, ion_pair_rate); see
    ``kazil_lovejoy``'s own docstring/source for the clamp-before-bracket
    order (mo_ham_m7_nucl.F90:320-668's own INTEGER search loop plus its
    ``MIN``/``MAX`` clamps, lines ~422-480).
    """
    from scipy.io import netcdf_file

    axis_names = ("temperature", "RH", "H2SO4", "ionization", "condensation_sink")
    with netcdf_file(path, "w") as nc:
        for name, values in zip(axis_names, axes):
            nc.createDimension(name, len(values))
            var = nc.createVariable(name, "d", (name,))
            var[:] = np.asarray(values, dtype=np.float64)
        pfr = nc.createVariable("pfr", "f", axis_names)
        pfr[:] = np.asarray(log_pfr, dtype=np.float32)


def _kazil_ham_input_dir(tmp_path, ref, case=0):
    """Build a fully offline ``HAM_INPUT_DIR`` (O'Brien text files + a tiny
    PARNUC netCDF) from ``hamnucl2_chain.npz``'s embedded real numbers, so
    the MARKED Kazil adapter test below needs neither the shared disk nor
    ``monkeypatch``ing past the adapter's real file-loading path. ``case``
    selects which of the 16 embedded cases' PARNUC corner excerpt to use
    (immaterial to correctness -- see ``_write_tiny_parnuc_netcdf``).
    """
    _write_real_format_obrien_file(
        tmp_path / "gcr_ipr_solmin.txt", ref["in/vertical_cutoff_rigidity"],
        ref["in/mass_column_density"], ref["in/ipr_solmin"])
    _write_real_format_obrien_file(
        tmp_path / "gcr_ipr_solmax.txt", ref["in/vertical_cutoff_rigidity"],
        ref["in/mass_column_density"], ref["in/ipr_solmax"])
    _write_tiny_parnuc_netcdf(
        tmp_path / "parnuc.15H2SO4.nc", ref["in/kazil_table_axes"][case],
        ref["in/kazil_table_log_pfr"][case])
    return tmp_path


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


@pytest.mark.parametrize("core_dtype", ["float64", "float32"])
def test_adapter_kazil_scheme_runs_end_to_end(core_dtype, tmp_path, monkeypatch):
    """``nucleation_scheme=2`` (jax-gcm#1017 Kazil/GCR task) through the full
    adapter, at both the default float64 core and the forward-only float32
    core (jax-gcm#1017 task 6 -- the two switches compose: the Kazil table
    and the GCR ion-pair tables follow the exact same float64-numpy-storage
    -plus-per-step-rebuild rule as the kappa table, see m7_jax.py's module
    docstring).

    Runs fully offline: ``HAM_INPUT_DIR`` is pointed (via ``monkeypatch``) at
    a ``tmp_path`` holding O'Brien text files and a tiny PARNUC netCDF built
    from the embedded ``hamnucl2_chain.npz`` reference (see
    ``_kazil_ham_input_dir``), so the adapter's own ``_ham_input_dir``/
    ``read_obrien_gcr_ipr``/``load_kazil_lovejoy_table`` file-loading path is
    genuinely exercised rather than skipped. This module's
    ``@pytest.mark.requires_extra("m7")`` is the only gate: m7-jax's pin
    (jax-gcm#1017's M7 preset) always has Kazil/Lovejoy support, so there is
    nothing further to skip on.
    """
    import jax

    from jcm.physics.aerosol.jam.microphysics.m7_jax import M7JaxMicrophysics

    with np.load(_HAMNUCL2_REF) as z:
        ref = {k: z[k] for k in z.files}
    monkeypatch.setenv("HAM_INPUT_DIR", str(_kazil_ham_input_dir(tmp_path, ref)))

    core = M7JaxMicrophysics(
        nucleation_scheme=2, organic_scheme=0, core_dtype=core_dtype)
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
    # The output cast (__call__, outside the scoped x64 context) always
    # restores float64 regardless of the core's own working dtype.
    assert tend.tracers["g_h2so4"].dtype == jax.numpy.float64


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


@pytest.mark.parametrize("core_dtype", ["float64", "float32"])
def test_kazil_lovejoy_chain_matches_fortran_reference(core_dtype):
    """nsnucl=2 end-to-end chain (jax-gcm#1017 Kazil/GCR task, W5):
    ``read_obrien_gcr_ipr -> gcr_ionization -> nucl_kazil_lovejoy``, against
    the UNMODIFIED compiled Fortran on the REAL O'Brien and PARNUC tables
    (``hamnucl2_chain.npz`` -- see its own README/provenance, next to it in
    ``jcm/data/test/echam_cloud_reference/``, for the exact build).

    Runs WITHOUT ``HAM_INPUT_DIR``: the real O'Brien table (small) is
    embedded whole, and the real PARNUC table (~200MB, far too large to
    commit) only as the per-case 2^5-corner bracket excerpt
    ``kazil_lovejoy``'s own bisection search would read from the full
    table for that case's query -- reproducing the full-table call
    bit-for-bit (verified when ``hamnucl2_chain.npz`` was built). The
    reference npz is committed to the repository, so its absence is a
    checkout problem, not something to skip past.
    """
    assert _HAMNUCL2_REF.exists(), f"{_HAMNUCL2_REF} is committed to the repository"

    import jax

    from jcm.physics.aerosol.jam.chemistry.gcr_ionisation import ObrienGcrTable

    with np.load(_HAMNUCL2_REF) as z:
        ref = {k: z[k] for k in z.files}
    n = len(ref["in/ion_pair_rate"])

    # Stage 1: jcm's own gcr_ion_pair_rate against the embedded real O'Brien
    # table must reproduce the ion_pair_rate stage 2 was built against (which
    # is itself hamgcr.npz's Fortran gcr_ionization output -- see
    # gcr_ionisation_test.py's own Fortran-parity test for that half).
    with jax.enable_x64(True):
        table = ObrienGcrTable(ref["in/vertical_cutoff_rigidity"], ref["in/mass_column_density"],
                                ref["in/ipr_solmin"], ref["in/ipr_solmax"])
        recomputed_ipr = _recompute_ion_pair_rate(ref, table)
    np.testing.assert_allclose(recomputed_ipr, ref["in/ion_pair_rate"], rtol=1e-9, atol=0)

    # Stage 2: feed that (jcm-recomputed, Fortran-matching) ion_pair_rate into
    # m7-jax's own kazil_lovejoy, against the per-case mini-table excerpt of
    # the real PARNUC table, at the requested core_dtype (jax-gcm#1017 task 6
    # -- M7JaxMicrophysics's own float32 forward-only core).
    from m7_jax.nucleation import KazilLovejoyTable, kazil_lovejoy

    with jax.enable_x64(core_dtype == "float64"):
        dt = jax.numpy.float64 if core_dtype == "float64" else jax.numpy.float32
        rate = np.empty(n)
        cluster = np.empty(n)
        for i in range(n):
            mini_table = KazilLovejoyTable(
                *(jax.numpy.asarray(ref["in/kazil_table_axes"][i, a, :], dtype=dt) for a in range(5)),
                jax.numpy.asarray(ref["in/kazil_table_log_pfr"][i], dtype=dt))
            r, s = kazil_lovejoy(
                dt(ref["in/temperature"][i]), dt(ref["in/relative_humidity_pct"][i]),
                dt(ref["in/h2so4"][i]), dt(ref["in/total_sink"][i]),
                dt(recomputed_ipr[i]), mini_table)
            rate[i], cluster[i] = float(r), float(s)

    # float64 matches to machine precision (measured 1.7e-16 when this
    # reference was built); float32 loses precision in the log-space
    # interpolation's exponentiation (measured 3.7e-6) -- both tolerances
    # below hold a comfortable margin over the measured values.
    tol = 1e-9 if core_dtype == "float64" else 1e-4
    np.testing.assert_allclose(rate, ref["out/rate"], rtol=tol, atol=0)
    np.testing.assert_allclose(cluster, ref["out/cluster_sulfate"], rtol=tol, atol=0)
    assert np.any(rate == 0.0)  # the clamped-low-ion_pair_rate cases (col 14) exercise lset_zero
    assert np.any(rate > 0.0)

# The live-HAM_INPUT_DIR staleness guard (comparing a fresh disk read of the
# real O'Brien/PARNUC tables against the embedded excerpts above) lives in
# gcr_ionisation_test.py instead of here: this module's
# @pytest.mark.requires_extra("m7") marker means CI's extras-tests job fails
# ANY skip of it (JCM_REQUIRE_EXTRAS=1 has no HAM_INPUT_DIR), so a test that
# can only run against the real shared disk cannot live in a marked module.
# The gcr_ionisation_test.py versions are unmarked, do not import m7_jax at
# all (so the extras scanner has nothing to flag), and skip on a reason that
# does not name the m7/m7_jax package.
