"""Galactic cosmic ray ionisation (jax-gcm#1017 Kazil/GCR task, Part B)."""

from __future__ import annotations

import datetime
import functools
import math
from pathlib import Path

import jax
import numpy as np
import pytest

# jcm's conftest resets jax_enable_x64 to the session default around every
# test (#729), so -- unlike m7-jax's own test files -- a module-level
# jax.config.update here would be silently undone; every x64-sensitive test
# below uses the ``with jax.enable_x64(True):`` context manager instead
# (the pattern aqueous_hamaqueous_reference_test.py/ham_freezing_reference_
# test.py already use for the same reason). gcr_ionisation.py's own kernels
# hardcode float64 casts internally (matching m7-jax's kazil_lovejoy, since
# this module's only real caller, M7JaxMicrophysics, already runs in
# float64 throughout); without x64 enabled those casts silently truncate to
# float32 with a UserWarning rather than erroring, which is what the
# solar_activity/dipole-axis tests below would otherwise fail on.
from jcm.physics.aerosol.jam.chemistry.gcr_ionisation import (  # noqa: E402
    ObrienGcrTable,
    _VCR_N,
    _MCD_N,
    _dipole_axis_coefficients,
    gcr_ion_pair_rate,
    geo2mag,
    read_obrien_gcr_ipr,
    solar_activity,
    vertical_cutoff_rigidity,
)

# --- reference-comparison skip: jax-gcm#1017 Kazil/GCR task STATUS 2026-10-06
# named this missing; the O'Brien tables (and the Kazil table, Part A) landed
# in HAM_INPUT_DIR mid-task, so fortran_harness/echam_hamgcr's build_reference_
# hamgcr.py was run and this reference now exists. The skip (naming HAM_INPUT_DIR
# and the npz) stays for a checkout where the data file was not copied in.
_HAMGCR_REF = (Path(__file__).resolve().parents[4] / "data" / "test"
               / "echam_cloud_reference" / "hamgcr.npz")
_gcr_data_skip = pytest.mark.skipif(
    not _HAMGCR_REF.exists(),
    reason=(
        "GCR reference data not staged: need HAM_INPUT_DIR (O'Brien "
        "solar-min/max tables) and jcm/data/test/echam_cloud_reference/"
        "hamgcr.npz (from the fortran_harness/echam_hamgcr 'make reference' "
        "target) -- see docs source for the exact command once both exist"
    ),
)


def _write_synthetic_format_test_file(path, rng):
    """Write a tiny-VALUE (not tiny-size) O'Brien file: the real format is fixed-
    size (15 cutoff-rigidity blocks of 110 mass-column-density rows each --
    read_obrien_gcr_ipr's own hardcoded vcr_n/mcd_n, not read from the file),
    but every number here is synthetic. Never used for a physics number --
    see the naming rule in kazil_gcr_TASK.md. Format inferred from the
    Fortran's own READ statements (mo_ham_gcrion.f90:120-206): one skipped
    header line, then per block a skipped line, an 'F11.6,A' cutoff-rigidity
    line, two skipped lines, and 110 'mcd ipr' data lines.
    """
    vcr = np.sort(rng.uniform(0.1, 20.0, _VCR_N))
    mcd = np.sort(rng.uniform(0.0, 1.0, _MCD_N))
    lines = ["HEADER synthetic_format_test"]
    for ji in range(_VCR_N):
        lines.append("skip")
        lines.append(f"{vcr[ji]:11.6f} GV (synthetic)")
        lines.append("skip")
        lines.append("skip")
        for jj in range(_MCD_N):
            ipr = rng.uniform(0.0, 50.0)
            lines.append(f"{mcd[jj]:.6f} {ipr:.6f}")
    path.write_text("\n".join(lines) + "\n")
    return vcr, mcd


def test_read_obrien_gcr_ipr_synthetic_format(tmp_path):
    rng = np.random.default_rng(0)
    vcr, mcd = _write_synthetic_format_test_file(tmp_path / "gcr_ipr_solmin.txt", rng)
    # solmax must share the same axes (read_obrien_gcr_ipr checks this).
    (tmp_path / "gcr_ipr_solmax.txt").write_text(
        (tmp_path / "gcr_ipr_solmin.txt").read_text())
    table = read_obrien_gcr_ipr(tmp_path)
    assert table.vertical_cutoff_rigidity.shape == (_VCR_N,)
    assert table.mass_column_density.shape == (_MCD_N,)
    assert table.ipr_solmin.shape == (_VCR_N, _MCD_N)
    assert table.ipr_solmax.shape == (_VCR_N, _MCD_N)
    np.testing.assert_allclose(np.asarray(table.vertical_cutoff_rigidity), vcr, atol=5e-7)
    np.testing.assert_allclose(np.asarray(table.mass_column_density), mcd, atol=5e-7)
    np.testing.assert_array_equal(np.asarray(table.ipr_solmin), np.asarray(table.ipr_solmax))


def test_read_obrien_gcr_ipr_accepts_every_named_file(tmp_path):
    """README_GCR.txt's own SOLMIN.txt/SOLMAX.txt names, not just ECHAM's."""
    rng = np.random.default_rng(1)
    _write_synthetic_format_test_file(tmp_path / "SOLMIN.txt", rng)
    (tmp_path / "SOLMAX.txt").write_text((tmp_path / "SOLMIN.txt").read_text())
    table = read_obrien_gcr_ipr(tmp_path)
    assert table.vertical_cutoff_rigidity.shape == (_VCR_N,)


def test_read_obrien_gcr_ipr_missing_directory_names_the_variable(tmp_path):
    with pytest.raises(FileNotFoundError, match="gcr_ipr_solmin"):
        read_obrien_gcr_ipr(tmp_path)


def test_read_obrien_gcr_ipr_rejects_mismatched_axes(tmp_path):
    rng = np.random.default_rng(2)
    _write_synthetic_format_test_file(tmp_path / "gcr_ipr_solmin.txt", rng)
    _write_synthetic_format_test_file(tmp_path / "gcr_ipr_solmax.txt", np.random.default_rng(3))
    with pytest.raises(ValueError, match="disagree"):
        read_obrien_gcr_ipr(tmp_path)


# --- solar_activity: a closed form of the date, transcribed independently -

def _native_solar_activity(year, day_of_year_fraction, days_in_year):
    """``mo_ham_gcrion.f90::solar_activity`` (526-557), transcribed
    independently of ``gcr_ionisation.solar_activity`` from the Fortran text
    (not copy-pasted from the port) as the test's own oracle.
    """
    ztime = year + (day_of_year_fraction - 1.0) / days_in_year
    return math.cos(2.0 * math.pi * (1991.0 - ztime) / 11.0)


@pytest.mark.parametrize("year,doy,days_in_year", [
    (1991, 1.0, 365), (1996, 184.5, 366), (2005, 100.25, 365), (2020, 366.0, 366),
])
def test_solar_activity_matches_native_formula(year, doy, days_in_year):
    tyear = (doy - 1.0) / days_in_year
    expected = _native_solar_activity(year, doy, days_in_year)
    with jax.enable_x64(True):
        actual = float(solar_activity(year, tyear))
    assert actual == pytest.approx(expected, rel=1e-13)


def test_solar_activity_peaks_at_solar_max_and_min():
    with jax.enable_x64(True):
        # 1 Jan 1991 (solar max, by construction): tyear=0.
        peak = float(solar_activity(1991, 0.0))
        # 2 Jul 1996 is 5.5 years later -- half an 11-year cycle -- solar min.
        trough = float(solar_activity(1996, 0.5))
    assert peak == pytest.approx(1.0, abs=1e-12)
    assert trough == pytest.approx(-1.0, abs=1e-3)


def test_solar_activity_bounded():
    rng = np.random.default_rng(4)
    years = rng.integers(1960, 2030, 50)
    tyears = rng.uniform(0.0, 1.0, 50)
    with jax.enable_x64(True):
        values = np.asarray([float(solar_activity(y, t)) for y, t in zip(years, tyears)])
    assert np.all(values >= -1.0 - 1e-12) and np.all(values <= 1.0 + 1e-12)


# --- geomagnetic dipole axis: physical sanity (Fortran parity is the ------
# harness's job, gated by _gcr_data_skip via test_gcr_ion_pair_rate_matches_
# native below once hamgcr.npz exists) --------------------------------------

def test_dipole_axis_coefficients_give_a_unit_vector():
    with jax.enable_x64(True):
        st0, ct0, sl0, cl0 = (float(x) for x in _dipole_axis_coefficients(2000.0, 1.0))
    assert st0**2 + ct0**2 == pytest.approx(1.0, abs=1e-12)
    assert sl0**2 + cl0**2 == pytest.approx(1.0, abs=1e-12)


def test_geo2mag_pole_tilt_matches_known_epoch_2000_value():
    # IGRF-2000's dipole tilt was ~10.5 degrees (geomagnetic north pole near
    # 79.5N) -- a published, well-known value, not a Fortran-derived one;
    # this is a physical sanity check on the reduced recalc port, not a
    # parity claim.
    with jax.enable_x64(True):
        mag_lat, _ = geo2mag(90.0, 0.0, 2000.0, 1.0)
        value = float(mag_lat)
    assert value == pytest.approx(79.5, abs=0.2)


def test_geo2mag_broadcasts_over_grids():
    lat = np.linspace(-90.0, 90.0, 7)
    lon = np.linspace(-180.0, 180.0, 7)
    with jax.enable_x64(True):
        mag_lat, mag_lon = geo2mag(lat, lon, 2010.0, 100.0)
        mag_lat, mag_lon = np.asarray(mag_lat), np.asarray(mag_lon)
    assert mag_lat.shape == (7,)
    assert mag_lon.shape == (7,)
    assert np.all(np.isfinite(mag_lat))
    assert np.all(np.isfinite(mag_lon))


def test_vertical_cutoff_rigidity_zero_at_poles_max_at_equator():
    with jax.enable_x64(True):
        at_pole = float(vertical_cutoff_rigidity(90.0))
        at_equator = float(vertical_cutoff_rigidity(0.0))
    assert at_pole == pytest.approx(0.0, abs=1e-10)
    assert at_equator == pytest.approx(14.9, abs=1e-10)


# --- gcr_ion_pair_rate: shape/finiteness now; Fortran parity once staged ---

def _synthetic_table(rng):
    vcr = np.sort(rng.uniform(0.1, 20.0, 15))
    mcd = np.sort(rng.uniform(0.0, 1.0, 110))
    return ObrienGcrTable(*(x for x in (
        vcr, mcd, rng.uniform(0.0, 50.0, (15, 110)), rng.uniform(0.0, 50.0, (15, 110)))))


def test_gcr_ion_pair_rate_shape_and_finite():
    rng = np.random.default_rng(5)
    tables = _synthetic_table(rng)
    lat = np.radians(np.array([0.0, 45.0, -60.0, 89.0]))
    lon = np.radians(np.array([0.0, 100.0, -50.0, 10.0]))
    pressure = np.array([[101325.0, 85000.0, 50000.0, 20000.0], [1000.0, 900.0, 800.0, 700.0]])
    temperature = np.full_like(pressure, 250.0)
    with jax.enable_x64(True):
        out = np.asarray(gcr_ion_pair_rate(lat, lon, pressure, temperature, 0.3, tables,
                                            year=2005.0, day_of_year=180.0))
    assert out.shape == pressure.shape
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


def test_gcr_ion_pair_rate_jit_matches_eager():
    rng = np.random.default_rng(6)
    tables = _synthetic_table(rng)
    lat = np.radians(np.array([10.0, -40.0]))
    lon = np.radians(np.array([20.0, -30.0]))
    pressure = np.array([[90000.0, 80000.0], [900.0, 800.0]])
    temperature = np.full_like(pressure, 230.0)
    fn = functools.partial(gcr_ion_pair_rate, year=1999.0, day_of_year=50.0)
    with jax.enable_x64(True):
        eager = np.asarray(fn(lat, lon, pressure, temperature, -0.5, tables))
        compiled = np.asarray(jax.jit(fn)(lat, lon, pressure, temperature, -0.5, tables))
    np.testing.assert_allclose(eager, compiled, rtol=1e-12)


def _day_of_year_1_based(year, month, day):
    """1-based day-of-year, matching ``get_year_day``'s integer part
    (``gcr_ionization``'s own ``idoy = aint(get_year_day(current_date))``).
    """
    return (datetime.date(int(year), int(month), int(day))
            - datetime.date(int(year), 1, 1)).days + 1


@functools.lru_cache(maxsize=None)
def _load_hamgcr_reference():
    with np.load(_HAMGCR_REF) as z:
        return {k: z[k] for k in z.files}


@_gcr_data_skip
def test_gcr_ion_pair_rate_matches_native_gcr_ionization():
    """``gcr_ion_pair_rate`` against the compiled, UNMODIFIED
    ``mo_ham_gcrion.f90::gcr_ionization`` (+ ``mo_geopack.f90``'s
    ``recalc``/``geo2mag`` it calls) on 28 designed columns (7 latitudes x
    4 longitudes, exercising the geomagnetic-offset effect) x 10 L47-like
    levels x 8 (solar-activity, date) scenarios -- including dates both
    inside and outside the IGRF table's 1965-2010 range (the ``recalc``
    year clamp/extrapolation) -- against the REAL O'Brien tables staged at
    build time (``fortran_harness/echam_hamgcr``'s ``build_reference_
    hamgcr.py``, which itself reads ``HAM_INPUT_DIR``; the npz bakes the
    table in, so this test itself does not need ``HAM_INPUT_DIR``).
    ``grav=9.80665`` reproduces ``mo_physical_constants.f90``'s own value
    (see :func:`gcr_ion_pair_rate`'s docstring for why jcm's own ``c.grav``
    differs and is not used here).
    """
    z = _load_hamgcr_reference()
    tables = ObrienGcrTable(z["in/vertical_cutoff_rigidity"], z["in/mass_column_density"],
                             z["in/ipr_solmin"], z["in/ipr_solmax"])
    lat = np.radians(z["meta/column_lat"])
    lon = np.radians(z["meta/column_lon"])
    pressure = z["in/pressure"].T    # (klev, ncols), this module's own convention
    temperature = z["in/temperature"].T
    with jax.enable_x64(True):
        for i, (psolact, date) in enumerate(zip(z["meta/scenario_psolact"], z["meta/scenario_date"])):
            year, month, day = date[0], date[1], date[2]
            doy = _day_of_year_1_based(year, month, day)
            out = gcr_ion_pair_rate(lat, lon, pressure, temperature, float(psolact), tables,
                                     year=float(year), day_of_year=float(doy), grav=9.80665)
            out = np.asarray(out).T  # back to (ncols, klev), the npz's own layout
            expected = z["out/pgcripr"][i]
            np.testing.assert_allclose(out, expected, rtol=1e-12, atol=0,
                                        err_msg=str(z["meta/scenario_names"][i]))
