"""Tests for the conservative emissions regridder (#498)."""

import os
import unittest

import numpy as np

from jcm.data.emissions.regrid import build_regridder, model_grid
from jcm.physics.speedy.speedy_coords import get_speedy_coords

# The CESM CEDS ne30 reference files, used for an end-to-end conservation check
# when present (skipped otherwise so the suite stays self-contained).
_NE30_DIR = (
    "/glade/campaign/cesm/cesmdata/cseg/inputdata/atm/cam/chem/emis/cmip7/"
    "ne30/CEDS-CMIP-2025-04-18_20251030"
)
_NE30_BC = os.path.join(
    _NE30_DIR,
    "bc_a4-em-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_"
    "175001-202312_c20251030.nc",
)


def _regular_latlon(nlon: int, nlat: int):
    """Cell centres + (relative) areas for a regular lat/lon grid [deg]."""
    lon = (np.arange(nlon) + 0.5) * (360.0 / nlon)
    lat = -90.0 + (np.arange(nlat) + 0.5) * (180.0 / nlat)
    LON, LAT = np.meshgrid(lon, lat, indexing="ij")
    area = np.cos(np.deg2rad(LAT))            # ∝ cos(lat) dlon dlat (dlon,dlat const)
    return LON.ravel(), LAT.ravel(), area.ravel()


class RegridConservationTest(unittest.TestCase):
    def setUp(self):
        # A fine (1°) source coarsened onto the T31 model grid — the regime the
        # first-order conservative scheme is designed for.
        self.coords = get_speedy_coords(layers=8, spectral_truncation=31)
        self.dlon, self.dlat, self.darea = model_grid(self.coords)
        self.slon, self.slat, self.sarea = _regular_latlon(360, 180)
        self.R = build_regridder(self.slon, self.slat, self.sarea,
                                 self.dlon, self.dlat)

    def test_target_shape(self):
        out = self.R(np.ones(self.slon.size))
        self.assertEqual(out.shape, self.coords.horizontal.nodal_shape)

    def test_constant_field_preserved(self):
        # A spatially constant flux must come back constant (area-weighted mean
        # of a constant is that constant), everywhere the grid is covered.
        out = self.R(np.full(self.slon.size, 3.7))
        np.testing.assert_allclose(out, 3.7, rtol=1e-12)

    def test_global_mass_conserved(self):
        # A smooth non-constant field: global integral ∫f dΩ preserved to the
        # first-order binning accuracy (<1%).
        f = 1.0 + np.cos(np.deg2rad(self.slat)) ** 2 * np.sin(np.deg2rad(self.slon))
        out = self.R(f)
        src_int = np.sum(f * self.sarea) / np.sum(self.sarea)      # area-wt mean
        tgt_int = np.sum(out * self.darea) / np.sum(self.darea)
        self.assertLess(abs(tgt_int / src_int - 1.0), 1e-2)

    def test_leading_axes_preserved(self):
        # (time, level, n_source) regrids to (time, level, nlon, nlat).
        vals = np.random.default_rng(0).random((4, 3, self.slon.size))
        out = self.R(vals)
        self.assertEqual(out.shape, (4, 3, *self.coords.horizontal.nodal_shape))

    @unittest.skipUnless(os.path.exists(_NE30_BC), "ne30 reference file absent")
    def test_ne30_reference_conserves(self):
        # Regrid a real CESM CEDS field (unstructured ncol) onto T31 and check
        # the global emitted mass is preserved to first order.
        import xarray as xr

        ds = xr.open_dataset(_NE30_BC, decode_times=False)
        em = ds["emiss"].isel(time=-1).values
        R = build_regridder(ds["lon"].values, ds["lat"].values,
                            ds["area"].values, self.dlon, self.dlat)
        out = R(em)
        src_int = np.sum(em * ds["area"].values)
        tgt_int = np.sum(out * self.darea)
        self.assertLess(abs(tgt_int / src_int - 1.0), 5e-3)
        self.assertFalse(np.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
