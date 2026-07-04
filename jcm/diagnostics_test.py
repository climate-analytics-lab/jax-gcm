"""Unit tests for ``jcm.diagnostics``."""

import unittest

import numpy as np
import xarray as xr

from jcm.diagnostics import check_health, print_report


def _make_dataset(T_min: float, T_max: float, q_max_gkg: float = 15.0,
                  nan_frac: float = 0.0):
    """Build a synthetic xarray dataset for the health-check tests.

    ``q_max_gkg`` is the per-cell upper bound on specific_humidity in g/kg
    (matches the unit convention :func:`dynamics_state_to_physics_state` writes
    into the saved netCDF). Healthy tropical surface q runs ~10-25 g/kg.
    """
    nx, ny, nt = 4, 4, 2
    rng = np.random.default_rng(0)
    T = T_min + (T_max - T_min) * rng.random((nt, nx, ny))
    if nan_frac > 0:
        mask = rng.random(T.shape) < nan_frac
        T = np.where(mask, np.nan, T)
    q = q_max_gkg * rng.random((nt, nx, ny))
    return xr.Dataset({
        "temperature": (("time", "lon", "lat"), T),
        "specific_humidity": (("time", "lon", "lat"), q),
    })


class TestCheckHealth(unittest.TestCase):
    def test_healthy_dataset(self):
        ds = _make_dataset(T_min=240.0, T_max=310.0)
        ok, report = check_health(ds, chunk_idx=0, elapsed_days=10.0)
        self.assertTrue(ok)
        self.assertEqual(report["reasons"], [])
        self.assertGreater(report["T_max"], report["T_min"])

    def test_extreme_temperature_min(self):
        ds = _make_dataset(T_min=50.0, T_max=300.0)
        ok, report = check_health(ds, 0, 10.0)
        self.assertFalse(ok)
        self.assertTrue(any("T_min" in reason for reason in report["reasons"]))

    def test_extreme_temperature_max(self):
        ds = _make_dataset(T_min=240.0, T_max=600.0)
        ok, report = check_health(ds, 0, 10.0)
        self.assertFalse(ok)
        self.assertTrue(any("T_max" in reason for reason in report["reasons"]))

    def test_extreme_humidity(self):
        # 200 g/kg is unphysical (max physical tropical surface q ~30 g/kg),
        # well above the 100 g/kg threshold ``check_health`` flags.
        ds = _make_dataset(T_min=240.0, T_max=300.0, q_max_gkg=200.0)
        ok, report = check_health(ds, 0, 10.0)
        self.assertFalse(ok)
        self.assertTrue(any("q_max" in reason for reason in report["reasons"]))

    def test_print_report_handles_failed_run(self):
        # Smoke test that print_report doesn't raise on a failed report.
        ds = _make_dataset(T_min=50.0, T_max=600.0)
        _, report = check_health(ds, 1, 90.0)
        print_report(report)

    def test_any_nan_temperature_fails(self):
        # A single NaN in T should fail the run, not require a > 10% fraction.
        ds = _make_dataset(T_min=240.0, T_max=300.0, nan_frac=0.5)
        ok, report = check_health(ds, 0, 10.0)
        self.assertGreater(report["T_nan_frac"], 0)
        self.assertFalse(ok)
        self.assertTrue(any("NaN" in reason for reason in report["reasons"]))


class TestCheckHealthOptionalDiagnostics(unittest.TestCase):
    """Radiation / surface-T / convective-precip report fields.

    These variables are only present in some run configurations, so
    ``check_health`` picks them up opportunistically; the report values
    must be the plain means over the LAST timestep.
    """

    def _make_full_dataset(self):
        nx, ny, nt = 4, 4, 2
        ds = _make_dataset(T_min=240.0, T_max=310.0)
        dims = ("time", "lon", "lat")
        # Two timesteps with different values so we can verify the
        # report uses isel(time=-1), not a whole-run mean.
        olr = np.stack([np.full((nx, ny), 200.0), np.full((nx, ny), 240.0)])
        ds["radiation.toa_lw_up"] = (dims, olr)
        ds["radiation.surface_lw_down"] = (dims, np.full((nt, nx, ny), 350.0))
        ds["radiation.toa_sw_down"] = (dims, np.full((nt, nx, ny), 340.0))
        ds["radiation.toa_sw_up"] = (dims, np.full((nt, nx, ny), 100.0))
        sfc_T = np.stack([np.full((nx, ny), 285.0), np.full((nx, ny), 290.0)])
        ds["surface.surface_temperature"] = (dims, sfc_T)
        # A tendency-suffixed variable must NOT be picked as surface T.
        ds["surface.surface_temperature_tendency"] = (
            dims, np.full((nt, nx, ny), 1e9),
        )
        # 20 mm/day in kg m^-2 s^-1.
        precip = np.full((nt, nx, ny), 20.0 / 86400.0)
        ds["convection.precip_conv"] = (dims, precip)
        return ds

    def test_radiation_means_from_last_timestep(self):
        ds = self._make_full_dataset()
        ok, report = check_health(ds, 0, 10.0)
        self.assertTrue(ok)
        self.assertAlmostEqual(report["toa_lw_up_mean"], 240.0)
        self.assertAlmostEqual(report["surface_lw_down_mean"], 350.0)
        self.assertAlmostEqual(report["toa_sw_down_mean"], 340.0)
        self.assertAlmostEqual(report["toa_sw_up_mean"], 100.0)

    def test_surface_temperature_skips_tendency_vars(self):
        ds = self._make_full_dataset()
        _, report = check_health(ds, 0, 10.0)
        # 1e9-valued tendency variable must not leak into the report.
        self.assertAlmostEqual(report["sfc_T_mean"], 290.0)
        self.assertAlmostEqual(report["sfc_T_min"], 290.0)
        self.assertAlmostEqual(report["sfc_T_max"], 290.0)

    def test_convective_precip_converted_to_mm_day(self):
        ds = self._make_full_dataset()
        _, report = check_health(ds, 0, 10.0)
        self.assertAlmostEqual(report["precip_conv_mean_mmday"], 20.0, places=5)
        self.assertAlmostEqual(report["precip_conv_max_mmday"], 20.0, places=5)

    def test_print_report_covers_optional_sections(self):
        ds = self._make_full_dataset()
        _, report = check_health(ds, 3, 365.25)
        # Smoke: all optional sections present and printable.
        print_report(report)

