"""The land-surface bundle convention survives every regrid (#672).

``lsm`` is the land share of a cell, ``glac`` the glacier share of the
LAND, and ``forest`` / ``snowc`` / ``alb`` describe the NON-glacier land
(``jcm.data.regridding.CONDITIONAL_FIELDS``). Each host that regrids these
fields — the Gaussian bundle builders, the pySES column sampler and the
spectral upsampler — goes through ``regrid_land_surface``. The fixture is a
checkerboard whose four source points around one target point are ocean,
snow-free 60 %-forest land, glacier, and fully snow-covered 60 %-forest
land; a bilinear regrid to the point between them weights the four equally,
so the exact area shares are known in closed form.
"""
import types
import unittest

import numpy as np
import xarray as xr

from jcm.data.regridding import regrid_land_surface

# Source grid: 2-degree spacing; the target (3N, 3E) sits at the centre of
# the (2..4)N x (2..4)E cell.
LAT = np.arange(-88.0, 90.0, 2.0)
LON = np.arange(0.0, 360.0, 2.0)
OCEAN, FOREST, GLACIER, SNOWY = (2.0, 2.0), (2.0, 4.0), (4.0, 2.0), (4.0, 4.0)

# Expected at the target: 3 of 4 points are land, 1 of those 3 glacier; the
# 2 non-glacier land points are both 60 % forest, one of them snow covered.
EXPECT = {"lsm": 0.75, "glac": 1.0 / 3.0, "forest": 0.6, "snowc": 0.5,
          "alb": 0.25, "stl": (260.0 + 250.0 + 262.0) / 3.0,
          # Soil wetness of the NON-glacier land: the glacier's 0 must not
          # dilute it (the vdiff counts the glacier as fully wet already).
          "soilw_am": 0.3}


def _source():
    """Build the (lat, lon) source fields.

    Values off their mask are deliberately wrong (ocean forest/snow 0,
    glacier albedo 0.8, ocean stl 290 K), so a dilution shows.
    """
    shape = (LAT.size, LON.size)
    f = {k: np.zeros(shape) for k in
         ("lsm", "glac", "forest", "snowc", "alb", "soilw_am")}
    f["stl"] = np.full(shape, 290.0)
    f["alb"][:] = 0.06                      # ocean albedo, must not leak

    def put(point, **values):
        i, j = int(np.argmin(abs(LAT - point[0]))), int(np.argmin(abs(LON - point[1])))
        for k, v in values.items():
            f[k][i, j] = v

    put(FOREST, lsm=1.0, forest=0.6, snowc=0.0, alb=0.2, stl=260.0,
        soilw_am=0.4)
    put(GLACIER, lsm=1.0, glac=1.0, alb=0.8, stl=250.0, soilw_am=0.0)
    put(SNOWY, lsm=1.0, forest=0.6, snowc=1.0, alb=0.3, stl=262.0,
        soilw_am=0.2)
    # A coastal row at 6N: ocean at 2E, fully snow-covered land at 4E.
    put((6.0, 4.0), lsm=1.0, snowc=1.0, alb=0.2, stl=265.0)
    return f


class TestGaussianBuilders(unittest.TestCase):
    def test_checkerboard_gives_exact_area_shares(self):
        from jcm.data.mirror.bundles import land_surface_fields

        f = _source()
        grid = dict(dims=("latitude", "longitude"),
                    coords={"latitude": LAT, "longitude": LON})
        era5 = xr.Dataset({"lsm": xr.DataArray(f["lsm"], **grid),
                           "cvh": xr.DataArray(f["forest"], **grid)})
        out = land_surface_fields(
            era5, xr.DataArray(f["glac"] > 0.5, **grid),
            {"snowc": xr.DataArray(f["snowc"], **grid),
             "alb": xr.DataArray(f["alb"], **grid),
             "forest": era5.cvh,
             "stl": xr.DataArray(f["stl"], **grid),
             "soilw_am": xr.DataArray(f["soilw_am"], **grid)},
            lats=np.array([3.0, 6.0]), lons=np.array([3.0]))
        for name, value in EXPECT.items():
            np.testing.assert_allclose(
                float(out[name].sel(lat=3.0, lon=3.0)), value, rtol=1e-12,
                err_msg=name)
        # Coastal half-ocean, fully snow-covered: snowc stays 1 (not 0.5).
        np.testing.assert_allclose(float(out["snowc"].sel(lat=6.0, lon=3.0)),
                                   1.0)
        np.testing.assert_allclose(float(out["lsm"].sel(lat=6.0, lon=3.0)),
                                   0.5)


class TestPysesColumnSampler(unittest.TestCase):
    def test_checkerboard_gives_exact_area_shares(self):
        from jcm.dycore.pyses.forcing import sample_forcing_to_columns

        f = _source()
        months = 12

        def monthly(a):
            return (("lon", "lat", "time"),
                    np.repeat(a.T[:, :, None], months, axis=2))

        ds = xr.Dataset(
            {"sst": monthly(np.full_like(f["stl"], 290.0)),
             "icec": monthly(np.zeros_like(f["stl"])),
             "stl": monthly(f["stl"]),
             "soilw_am": monthly(f["soilw_am"]),
             "snowc": monthly(f["snowc"]),
             "alb": (("lon", "lat"), f["alb"].T),
             "forest": (("lon", "lat"), f["forest"].T),
             "glac": (("lon", "lat"), f["glac"].T),
             "lsm": (("lon", "lat"), f["lsm"].T)},
            coords={"lon": LON, "lat": LAT, "time": np.arange(months)})
        monthly_out, static = sample_forcing_to_columns(
            ds, LON, LAT, np.array([3.0, 3.0]), np.array([3.0, 6.0]))
        self.assertAlmostEqual(float(static["glac"][0]), EXPECT["glac"], 12)
        self.assertAlmostEqual(float(static["forest"][0]), EXPECT["forest"], 12)
        self.assertAlmostEqual(float(static["alb"][0]), EXPECT["alb"], 12)
        np.testing.assert_allclose(monthly_out["snowc"][:, 0], EXPECT["snowc"])
        np.testing.assert_allclose(monthly_out["stl"][:, 0], EXPECT["stl"])
        # Soil wetness of the non-glacier land, diluted by neither the ocean
        # nor the glacier.
        np.testing.assert_allclose(monthly_out["soilw_am"][:, 0],
                                   EXPECT["soilw_am"])
        np.testing.assert_allclose(monthly_out["snowc"][:, 1], 1.0)
        # Ocean fields keep the plain bilinear sample.
        np.testing.assert_allclose(monthly_out["sst"][:, 0], 290.0)

    def test_file_without_lsm_keeps_the_plain_sample(self):
        from jcm.dycore.pyses.forcing import sample_forcing_to_columns

        f = _source()
        ds = xr.Dataset(
            {"snowc": (("lon", "lat", "time"), f["snowc"].T[:, :, None]),
             "alb": (("lon", "lat"), f["alb"].T)},
            coords={"lon": LON, "lat": LAT, "time": [0]})
        monthly_out, static = sample_forcing_to_columns(
            ds, LON, LAT, np.array([3.0]), np.array([3.0]))
        np.testing.assert_allclose(monthly_out["snowc"][0, 0], 0.25)
        np.testing.assert_allclose(static["alb"][0], (0.06 + 0.2 + 0.8 + 0.3) / 4)


class TestSpectralUpsampler(unittest.TestCase):
    def test_checkerboard_gives_exact_area_shares(self):
        from jcm.data.bc.interpolate import upsample_forcings_ds

        f = _source()
        ds = xr.Dataset(
            {name: (("lon", "lat"), f[name].T)
             for name in ("lsm", "glac", "forest", "alb", "stl")}
            | {"snowc": (("lon", "lat", "time"), f["snowc"].T[:, :, None]),
               "icec": (("lon", "lat", "time"), np.zeros_like(f["stl"]).T[:, :, None]),
               "soilw_am": (("lon", "lat", "time"), f["soilw_am"].T[:, :, None])},
            coords={"lon": LON, "lat": LAT, "time": [0]})
        grid = types.SimpleNamespace(latitudes=np.radians(np.array([3.0])),
                                     longitudes=np.radians(np.array([3.0])))
        out = upsample_forcings_ds(ds, grid)
        for name, value in EXPECT.items():
            np.testing.assert_allclose(np.asarray(out[name]).ravel()[0], value,
                                       rtol=1e-6, err_msg=name)


class TestPackagedT63File(unittest.TestCase):
    def test_convention_holds_and_regrid_is_the_identity(self):
        from pathlib import Path

        import jcm

        path = Path(jcm.__file__).parent / "data" / "bc" / "t63" / "forcing.nc"
        ds = xr.open_dataset(path).load()
        # Binary GLAC with no seasonal snow on it: conditional on the
        # non-glacier land by construction.
        self.assertTrue(set(np.unique(ds.glac.values)) <= {0.0, 1.0})
        self.assertEqual(float(ds.snowc.where(ds.glac > 0.5).max()), 0.0)
        fields = {k: ds[k] for k in ("glac", "forest", "snowc", "alb", "stl",
                                     "soilw_am")}
        out = regrid_land_surface(fields, ds.lsm, lambda da: da, glac=ds.glac)
        for name, value in fields.items():
            np.testing.assert_allclose(out[name].transpose(*value.dims).values,
                                       value.values, rtol=1e-6, atol=1e-7,
                                       err_msg=name)


if __name__ == "__main__":
    unittest.main()
