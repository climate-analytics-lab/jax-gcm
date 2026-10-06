"""CAM6 source conversion, conservative remapping and runtime injection."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from jcm.data.emissions.cam6_soa import SOURCES, prepare_cam6_soa
from jcm.data.emissions.prepare import molec_flux_to_mass_flux
from jcm.physics.speedy.speedy_coords import get_speedy_coords


class Cam6SoaTest(unittest.TestCase):
    def test_invalid_units_fluxes_and_calendars_fail_before_remapping(self):
        for invalid in ("units", "negative", "nonfinite", "calendar", "time"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                sources = {}
                for key in SOURCES:
                    path = Path(directory) / f"{key}.nc"
                    ds = xr.Dataset(
                        {"emiss_" + key: (("time", "lat", "lon"),
                                         np.ones((2, 2, 4)))},
                        coords={"time": [15.0, 45.0], "lat": [-45., 45.],
                                "lon": [0., 90., 180., 270.]})
                    field = ds["emiss_" + key]
                    field.attrs["units"] = "molecules/cm2/s"
                    ds.time.attrs.update(units="days since 2000-01-01",
                                         calendar="gregorian")
                    if key == "biogenic":
                        if invalid == "units":
                            field.attrs["units"] = "kg m-2 s-1"
                        elif invalid == "negative":
                            field.values[0, 0, 0] = -1.
                        elif invalid == "nonfinite":
                            field.values[0, 0, 0] = np.nan
                        elif invalid == "calendar":
                            ds.time.attrs["calendar"] = "noleap"
                        else:
                            attrs = dict(ds.time.attrs)
                            ds = ds.assign_coords(time=[16., 45.])
                            ds.time.attrs.update(attrs)
                    ds.to_netcdf(path)
                    sources[key] = str(path)
                with self.assertRaises(ValueError):
                    prepare_cam6_soa(None, sources)

    def test_carbon_equivalents_and_calendar_survive_preparation(self):
        # The physical 150 g/mol SOAG molecule must not multiply emissions
        # reported in carbon equivalents by 12.5. Yields/1.5 are also already
        # in these sources, so the mass flux is their sum without rescaling.
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        with tempfile.TemporaryDirectory() as directory:
            sources = {}
            for i, key in enumerate(SOURCES):
                path = Path(directory) / f"{key}.nc"
                ds = xr.Dataset(
                    {"emiss_" + key: (("time", "lat", "lon"),
                        np.full((2, 4, 8), (i + 1) * 1e10))},
                    coords={"time": [15.0, 45.0],
                            "lat": [-67.5, -22.5, 22.5, 67.5],
                            "lon": np.arange(8) * 45.0})
                ds["emiss_" + key].attrs.update(
                    molecular_weight=12.0, units="molecules/cm2/s")
                ds.time.attrs.update(units="days since 2000-01-01",
                                     calendar="gregorian")
                ds.to_netcdf(path)
                sources[key] = str(path)
            result = prepare_cam6_soa(coords, sources)
        expected = 6e10 * molec_flux_to_mass_flux(12.011)
        np.testing.assert_allclose(result.aero_emis_g_soag, expected,
                                   rtol=2e-14)
        self.assertEqual(result.aero_emis_g_soag.attrs["units"], "kg m-2 s-1")
        self.assertEqual(result.time.attrs["units"], "days since 2000-01-01")
        self.assertEqual(result.time.attrs["calendar"], "gregorian")
        from jcm.forcing import read_prescribed_aerosol_emissions
        fields = read_prescribed_aerosol_emissions(result, align_mode="wrap_year")
        self.assertEqual(set(fields), {"g_soag"})

    def test_incomplete_inventory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "anthro, biogenic and bb"):
            prepare_cam6_soa(None, {"biogenic": "missing.nc"})
