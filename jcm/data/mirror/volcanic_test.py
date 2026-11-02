"""Tests for the grid-free volcanic SO₂ product builder (#1190)."""

import os
import tempfile
import unittest

import numpy as np
import xarray as xr

from jcm.data.mirror.volcanic import (
    build_volcanic_product,
    degassing_sources,
    eruption_list,
    standard_atmosphere_height,
)

_AVOGADRO = 6.02214076e23


def _contvolcano(times=2, vary=False, ps=101325.0):
    """Build a CESM-shaped ``contvolcano`` file: 4 lat (poles included) × 8 lon, 3 levels."""
    lat = np.array([-90.0, -30.0, 30.0, 90.0])
    lon = np.arange(8) * 45.0
    p_int = np.array([30000.0, 60000.0, 90000.0, 101325.0])   # top first
    field = np.zeros((times, 3, 4, 8))
    field[:, 1, 2, 3] = 2.0e3     # molec cm-3 s-1, layer 600-900 hPa
    field[:, 2, 2, 3] = 1.0e3     # 900-1013 hPa
    if vary:
        field[1] *= 2.0
    return xr.Dataset(
        {"contvolcano": (("time", "lev", "lat", "lon"), field),
         "PS": (("time", "lat", "lon"), np.full((times, 4, 8), ps)),
         "P0": ((), 100000.0),
         "hyai": ("ilev", p_int / 100000.0), "hybi": ("ilev", np.zeros(4))},
        coords={"lat": lat, "lon": lon, "time": np.arange(times, dtype=float)})


def _utsvolcemis(days=1):
    t0 = np.array(["2010-04-14", "1991-06-15"], dtype="datetime64[ns]")
    return xr.Dataset(
        {"utsvolcemis": ("eruption_number", [1.0e7, 2.0e10]),
         "time_bnds": (("eruption_number", "bnds"),
                       np.stack([t0, t0 + np.timedelta64(days, "D")], axis=1))},
        coords={"lat": ("eruption_number", [63.6, 15.1]),
                "lon": ("eruption_number", [-19.6, 120.35]),
                "height": ("eruption_number", [9000.0, 25000.0]),
                "depth": ("eruption_number", [800.0, 2600.0]),
                "time": ("eruption_number", t0)})


class VolcanicProductTest(unittest.TestCase):
    def test_degassing_rate_is_molecules_times_layer_volume(self):
        src = degassing_sources(_contvolcano())
        self.assertEqual(src.sizes["degassing_source"], 1)
        z = standard_atmosphere_height(np.array([101325.0, 90000.0, 60000.0, 30000.0]))
        np.testing.assert_allclose(src["degassing_altitude_interface"], z)
        self.assertAlmostEqual(float(z[0]), 0.0, places=6)
        # Cell (30N, 135E): edges 0 and 60N, 45 degrees of longitude.
        area_cm2 = 6.371e6 ** 2 * np.deg2rad(45.0) * np.sin(np.deg2rad(60.0)) * 1e4
        kg = 64.066e-3 / _AVOGADRO
        dz_cm = np.diff(z) * 100.0
        expect = np.array([1.0e3 * dz_cm[0], 2.0e3 * dz_cm[1], 0.0]) * area_cm2 * kg
        np.testing.assert_allclose(src["degassing_so2"].values[0], expect, rtol=1e-12)
        self.assertEqual(float(src["degassing_lat"][0]), 30.0)
        self.assertEqual(float(src["degassing_lon"][0]), 135.0)

    def test_degassing_refuses_what_it_cannot_represent(self):
        with self.assertRaisesRegex(ValueError, "varies in time"):
            degassing_sources(_contvolcano(vary=True))
        with self.assertRaisesRegex(ValueError, "fixed 1013.25 hPa"):
            degassing_sources(_contvolcano(ps=95000.0))

    def test_eruptions_copied_and_checked(self):
        e = eruption_list(_utsvolcemis())
        np.testing.assert_allclose(e["eruption_lon"], [340.4, 120.35])
        np.testing.assert_allclose(e["eruption_so2"], [1.0e7, 2.0e10])
        with self.assertRaisesRegex(ValueError, "one day long"):
            eruption_list(_utsvolcemis(days=2))

    def test_product_round_trips_through_netcdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            dg, er = os.path.join(tmp, "dg.nc"), os.path.join(tmp, "er.nc")
            _contvolcano().to_netcdf(dg)
            _utsvolcemis().to_netcdf(er)
            out = os.path.join(tmp, "product.nc")
            build_volcanic_product(dg, er).to_netcdf(out)
            with xr.open_dataset(out) as ds:
                self.assertEqual(ds.sizes["eruption"], 2)
                self.assertIn("sha256", ds.attrs["degassing_source"])
                self.assertEqual(ds["eruption_time"].dtype.kind, "M")
                self.assertEqual(ds["degassing_so2"].attrs["units"], "kg s-1")


if __name__ == "__main__":
    unittest.main()
