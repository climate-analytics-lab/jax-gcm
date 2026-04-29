"""Unit tests for ``jcm.diagnostics``."""

import unittest

import numpy as np
import pytest
import xarray as xr

from jcm.diagnostics import check_health, print_report


def _make_dataset(T_min: float, T_max: float, q_max: float = 0.01,
                  nan_frac: float = 0.0):
    nx, ny, nt = 4, 4, 2
    rng = np.random.default_rng(0)
    T = T_min + (T_max - T_min) * rng.random((nt, nx, ny))
    if nan_frac > 0:
        mask = rng.random(T.shape) < nan_frac
        T = np.where(mask, np.nan, T)
    q = q_max * rng.random((nt, nx, ny))
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
        ds = _make_dataset(T_min=240.0, T_max=300.0, q_max=0.2)
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


# Slow-marked companion — see jcm/runners_test.py for rationale.

@pytest.mark.slow
class TestCheckHealthSlow(TestCheckHealth):
    pass
