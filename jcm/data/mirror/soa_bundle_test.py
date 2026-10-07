"""Mixed bulk/SOAG bundles preserve months and route through both readers."""

import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from jcm.data.mirror.bundles import add_cam6_soa


class TestSOAGBundle(unittest.TestCase):
    def test_matching_months_and_eras_reach_both_runtime_readers(self):
        from jcm.forcing import (read_anthropogenic_emissions,
                                 read_prescribed_aerosol_emissions)

        times = np.array([np.datetime64(f"2014-{m:02d}-01") for m in range(1, 13)])
        source = xr.Dataset(
            {"aero_emis_g_soag": (("time", "lon", "lat"),
                                   np.arange(1, 13)[:, None, None] * np.ones((12, 4, 2)))},
            coords={"time": times + np.timedelta64(15, "D")},
            attrs={"source": "NCAR", "source_sha256": "verified", "inventory_year": "period"})
        source.aero_emis_g_soag.attrs["units"] = "kg m-2 s-1"
        for era, period in [("pd", (2005, 2014)), ("pi", (1850, 1859))]:
            bulk = xr.Dataset(
                {"emis_surface_combustion_so2": (("time", "lon", "lat"),
                                                  np.ones((12, 4, 2)))},
                coords={"time": times, "lon": np.arange(4)*90., "lat": [-35.2644, 35.2644]},
                attrs={"era": era})
            with self.subTest(era=era), patch(
                    "jcm.data.emissions.cam6_soa.prepare_cam6_soa", return_value=source) as prepare:
                merged = add_cam6_soa(bulk)
                self.assertEqual(prepare.call_args.kwargs['year_range'], period)
                xr.testing.assert_identical(merged.emis_surface_combustion_so2,
                                            bulk.emis_surface_combustion_so2)
                np.testing.assert_array_equal(merged.aero_emis_g_soag[:, 0, 0], np.arange(1, 13))
                self.assertTrue(np.isfinite(merged.aero_emis_g_soag).all())
                self.assertEqual(merged.attrs['soag_source_sha256'], 'verified')
                self.assertIn('emis_surface_combustion_so2',
                              read_anthropogenic_emissions(merged, align_mode='wrap_year'))
                self.assertEqual(set(read_prescribed_aerosol_emissions(
                    merged, align_mode='wrap_year')), {'g_soag'})
