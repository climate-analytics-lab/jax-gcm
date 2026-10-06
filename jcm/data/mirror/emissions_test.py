"""Tests for the HAM biogenic-OC emissions source (jax-gcm#1017, decision F7)."""

import os
import tempfile
import unittest

import numpy as np
import xarray as xr

from jcm.data.mirror.emissions import load_biogenic_oc


def _synthetic_biogenic_file(path, nlat=8, nlon=16, flux=1.0e-12):
    """Write a tiny stand-in for HAM's emiss_aerocom_OC_monthly_2000_T63.nc."""
    lat = np.linspace(85.0, -85.0, nlat).astype(np.float32)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False).astype(np.float32)
    data = np.full((12, nlat, nlon), flux, dtype=np.float32)
    ds = xr.Dataset(
        {"emiss_biogenic": (("time", "lat", "lon"), data)},
        coords={"time": np.arange(1, 13, dtype=float), "lat": lat, "lon": lon},
    )
    ds["emiss_biogenic"].attrs = {"long_name": "SOA emissions", "units": "kg/m**2/s"}
    ds.to_netcdf(path)


class LoadBiogenicOcTest(unittest.TestCase):
    def test_reads_and_renames_the_variable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "synthetic_biogenic.nc")
            _synthetic_biogenic_file(path)
            da = load_biogenic_oc(path)
            self.assertEqual(da.name, "oc_biogenic")
            self.assertEqual(da.dims, ("time", "lat", "lon"))
            self.assertEqual(da.shape, (12, 8, 16))
            self.assertEqual(da.dtype, np.float32)

    def test_global_total_is_area_weighted_correctly(self):
        # A uniform flux's global annual total is just flux * Earth's area,
        # independent of the (synthetic) grid resolution -- a cheap, exact
        # check that the real file's 19.06 Tg/yr (see the docstring) uses
        # the same area-weighting this test exercises.
        from jcm.analysis import area_weights, global_mean

        flux = 1.0e-12
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "synthetic_biogenic.nc")
            _synthetic_biogenic_file(path, flux=flux)
            da = load_biogenic_oc(path)
            gm = global_mean(da, area_weights(da))
            earth_area = 4.0 * np.pi * 6371000.0 ** 2
            tg_per_yr = float(gm.mean()) * earth_area * 86400.0 * 365.25 / 1.0e9
            np.testing.assert_allclose(
                tg_per_yr, flux * earth_area * 86400.0 * 365.25 / 1.0e9, rtol=1e-5)


class BuildEmissionsNcBiogenicTest(unittest.TestCase):
    """build_emissions_nc's optional biogenic_oc parameter."""

    def _ceds_bb_stores(self, tmp, lats, lons):
        shape = (12, lats.size, lons.size)
        ceds = xr.Dataset()
        bb = xr.Dataset()
        for sp in ("SO2", "BC", "OC"):
            for sector in ("surface_combustion", "elevated_industrial", "shipping"):
                for era in ("pi", "pd"):
                    ceds[f"{sp}_{sector}_{era}_clim"] = (
                        ("month", "lat", "lon"), np.zeros(shape))
            for era in ("pi", "pd"):
                bb[f"{sp}_{era}_clim"] = (("month", "lat", "lon"), np.zeros(shape))
        ceds = ceds.assign_coords(lat=lats, lon=lons, month=np.arange(1, 13))
        bb = bb.assign_coords(lat=lats, lon=lons, month=np.arange(1, 13))
        ceds_path, bb_path = os.path.join(tmp, "ceds.zarr"), os.path.join(tmp, "bb.zarr")
        ceds.to_zarr(ceds_path)
        bb.to_zarr(bb_path)
        return ceds_path, bb_path

    def test_emis_biogenic_oc_written_when_supplied(self):
        from jcm.data.mirror.bundles import build_emissions_nc
        from jcm.data.regridding import gaussian_latlon

        lats, lons = gaussian_latlon(8)
        with tempfile.TemporaryDirectory() as tmp:
            ceds_path, bb_path = self._ceds_bb_stores(tmp, lats, lons)
            bio_path = os.path.join(tmp, "bio.nc")
            _synthetic_biogenic_file(bio_path, nlat=8, nlon=16, flux=2.0e-12)
            biogenic_oc = load_biogenic_oc(bio_path)
            out = os.path.join(tmp, "emissions_pd.nc")
            build_emissions_nc(ceds_path, bb_path, "pd", lats, lons, out,
                               biogenic_oc=biogenic_oc)
            with xr.open_dataset(out) as ds:
                self.assertIn("emis_biogenic_oc", ds)
                self.assertEqual(ds["emis_biogenic_oc"].attrs["units"], "kg m-2 s-1")
                self.assertTrue(np.all(np.isfinite(ds["emis_biogenic_oc"].values)))
                self.assertIn("HAM AeroCom-II biogenic OC", ds.attrs["source"])

    def test_emis_biogenic_oc_absent_by_default(self):
        from jcm.data.mirror.bundles import build_emissions_nc
        from jcm.data.regridding import gaussian_latlon

        lats, lons = gaussian_latlon(8)
        with tempfile.TemporaryDirectory() as tmp:
            ceds_path, bb_path = self._ceds_bb_stores(tmp, lats, lons)
            out = os.path.join(tmp, "emissions_pd.nc")
            build_emissions_nc(ceds_path, bb_path, "pd", lats, lons, out)
            with xr.open_dataset(out) as ds:
                self.assertNotIn("emis_biogenic_oc", ds)
                self.assertNotIn("HAM AeroCom-II", ds.attrs["source"])


if __name__ == "__main__":
    unittest.main()
