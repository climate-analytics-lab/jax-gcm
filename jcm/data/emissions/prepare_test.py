"""Tests for the emissions prepare pipeline (#498).

The unit-conversion check is self-contained; the end-to-end adapter checks run
against the on-disk CESM CMIP7 ne30 files when present (skipped otherwise).
"""

import os
import unittest

import numpy as np
import pytest

import jcm.constants as _const
from jcm.data.emissions.prepare import (
    cesm_bb4cmip7,
    cesm_cmip_anthro,
    molec_flux_to_mass_flux,
    prepare_emissions,
)
from jcm.physics.speedy.speedy_coords import get_speedy_coords

_NE30 = ("/glade/campaign/cesm/cesmdata/cseg/inputdata/atm/cam/chem/emis/cmip7/"
         "ne30")
_ANTHRO_DIR = os.path.join(_NE30, "CEDS-CMIP-2025-04-18_20251030")
_BB_DIR = os.path.join(_NE30, "DRES-CMIP-BB4CMIP7-2-0_smoothed_20251102")
_ANTHRO_SO2 = os.path.join(
    _ANTHRO_DIR,
    "SO2-em-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_"
    "175001-202312_c20251030.nc")


class UnitConversionTest(unittest.TestCase):
    def test_molec_to_mass_factor(self):
        # molec cm-2 s-1 -> kg m-2 s-1 is MW * 10 / N_A.
        na = _const.physical_constants.avogadro
        self.assertAlmostEqual(molec_flux_to_mass_flux(64.0), 64.0 * 10.0 / na)


def _synthetic_source(path, *, elevated=False, var="emiss", mw=64.0,
                      ncol=300, nt=3):
    """Write a tiny unstructured (ncol) source emission file like the CESM ones."""
    import xarray as xr

    rng = np.random.default_rng(0)
    lon = rng.uniform(0.0, 360.0, ncol)
    lat = rng.uniform(-89.0, 89.0, ncol)
    area = np.full(ncol, 4.0 * np.pi / ncol)  # steradians, sums to 4π
    coords = {"lon": ("ncol", lon), "lat": ("ncol", lat),
              "area": ("ncol", area), "time": np.arange(nt)}
    if elevated:
        nalt = 4
        data = {var: (("time", "altitude", "ncol"),
                      np.full((nt, nalt, ncol), 1.0e3))}
        coords["altitude_int"] = ("altitude_int",
                                  np.linspace(0.0, 0.4, nalt + 1))  # km
    else:
        data = {var: (("time", "ncol"), np.full((nt, ncol), 1.0e10))}
    ds = xr.Dataset(data, coords=coords)
    ds[var].attrs["molecular_weight"] = mw
    ds.to_netcdf(path)
    return path


class PrepareSyntheticTest(unittest.TestCase):
    """Fast coverage of the prepare pipeline on a tiny synthetic source."""

    def setUp(self):
        self.coords = get_speedy_coords(layers=8, spectral_truncation=31)
        self.nodal = self.coords.horizontal.nodal_shape

    def test_prepare_emissions_bulk(self):
        from jcm.data.emissions.prepare import Channel, prepare_emissions
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            src = _synthetic_source(os.path.join(d, "so2.nc"), mw=64.0)
            ds = prepare_emissions(
                (Channel("emis_surface_combustion_so2", src, molar_mass=64.0),),
                self.coords, time_index=0)
        v = ds["emis_surface_combustion_so2"]
        self.assertEqual(v.values.shape, self.nodal)
        self.assertEqual(v.attrs["units"], "kg m-2 s-1")
        self.assertTrue(np.all(np.isfinite(v.values)) and v.values.max() > 0)

    def test_prepare_speciated_surface_and_elevated(self):
        from jcm.data.emissions.prepare import (
            SpeciatedChannel, prepare_speciated_emissions)
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            surf = _synthetic_source(os.path.join(d, "a2.nc"), mw=115.0)
            elev = _synthetic_source(os.path.join(d, "ene.nc"), elevated=True,
                                     mw=115.0)
            ds = prepare_speciated_emissions(
                (SpeciatedChannel("m_so4_ait", surf),
                 SpeciatedChannel("m_so4_acc", elev, elevated=True)),
                self.coords, time_index=0)
        for t in ("m_so4_ait", "m_so4_acc"):
            arr = ds[f"aero_emis_{t}"].values
            self.assertEqual(arr.shape, self.nodal)
            self.assertTrue(np.all(np.isfinite(arr)) and arr.max() > 0)

    def test_adapter_builders_construct_channels(self):
        from jcm.data.emissions.prepare import (
            cesm_bb4cmip7, cesm_cmip_anthro, cesm_mam4_speciated)
        anthro = {c.contract_var for c in cesm_cmip_anthro("/d")}
        self.assertEqual(anthro, {"emis_surface_combustion_so2",
                                  "emis_surface_combustion_bc",
                                  "emis_surface_combustion_oc"})
        self.assertEqual({c.contract_var for c in cesm_bb4cmip7("/d")},
                         {"emis_biomass_burning_so2", "emis_biomass_burning_bc",
                          "emis_biomass_burning_oc"})
        spec = cesm_mam4_speciated("/d")
        tracers = {c.tracer for c in spec}
        self.assertEqual(tracers, {"m_so4_acc", "m_so4_ait", "m_bc_pcm",
                                   "m_poa_pcm", "n_acc", "n_ait", "n_pcm",
                                   "g_so2"})
        # The energy-sector sulfate is the elevated (mo_extfrc) channel.
        self.assertTrue(any(c.elevated for c in spec))


@pytest.mark.slow
@unittest.skipUnless(os.path.exists(_ANTHRO_SO2), "ne30 reference files absent")
class PrepareFromCesmTest(unittest.TestCase):
    def setUp(self):
        self.coords = get_speedy_coords(layers=8, spectral_truncation=31)
        self.nodal = self.coords.horizontal.nodal_shape
        # A single recent month keeps the regrid cheap.
        self.month = (2014 - 1750) * 12 + 6

    def test_anthro_adapter_produces_contract_vars(self):
        ds = prepare_emissions(cesm_cmip_anthro(_ANTHRO_DIR), self.coords,
                               time_index=self.month)
        for sp in ("so2", "bc", "oc"):
            v = f"emis_surface_combustion_{sp}"
            self.assertIn(v, ds.data_vars)
            arr = ds[v].values
            self.assertEqual(arr.shape, self.nodal)
            self.assertTrue(np.all(np.isfinite(arr)))
            self.assertGreater(float(arr.sum()), 0.0)
            self.assertEqual(ds[v].attrs["units"], "kg m-2 s-1")

    def test_anthro_so2_global_budget_plausible(self):
        # Sanity-check magnitude: a mid-2010s month of global anthropogenic SO2
        # annualises to ~100 Tg(SO2)/yr — right order, not off by 1000×.
        ds = prepare_emissions(cesm_cmip_anthro(_ANTHRO_DIR), self.coords,
                               time_index=self.month)
        _, _, area = __import__(
            "jcm.data.emissions.regrid", fromlist=["model_grid"]).model_grid(
            self.coords)
        r2 = self.coords.horizontal.radius ** 2
        kg_s = float(np.sum(ds["emis_surface_combustion_so2"].values * area * r2))
        tg_yr = kg_s * 3.1536e7 / 1e9
        self.assertGreater(tg_yr, 40.0)
        self.assertLess(tg_yr, 200.0)

    @unittest.skipUnless(os.path.isdir(_BB_DIR), "BB4CMIP7 files absent")
    def test_biomass_adapter_produces_contract_vars(self):
        ds = prepare_emissions(cesm_bb4cmip7(_BB_DIR), self.coords,
                               time_index=self.month)
        for sp in ("so2", "bc", "oc"):
            v = f"emis_biomass_burning_{sp}"
            self.assertIn(v, ds.data_vars)
            self.assertEqual(ds[v].values.shape, self.nodal)
            self.assertTrue(np.all(np.isfinite(ds[v].values)))


@pytest.mark.slow
@unittest.skipUnless(os.path.exists(_ANTHRO_SO2), "ne30 reference files absent")
class PrepareSpeciatedTest(unittest.TestCase):
    """CAM6-faithful pre-speciated path reproduces CESM's MAM4 emissions."""

    def setUp(self):
        from jcm.data.emissions.prepare import (
            cesm_mam4_speciated, prepare_speciated_emissions)
        self.coords = get_speedy_coords(layers=8, spectral_truncation=31)
        self.ds = prepare_speciated_emissions(
            cesm_mam4_speciated(_ANTHRO_DIR), self.coords,
            time_index=(2014 - 1750) * 12 + 6)

    def _tg_per_yr(self, tracer):
        from jcm.data.emissions.regrid import model_grid
        _, _, area = model_grid(self.coords)
        r2 = self.coords.horizontal.radius ** 2
        kg_s = float(np.sum(self.ds[f"aero_emis_{tracer}"].values * area * r2))
        return kg_s * 3.1536e7 / 1e9

    def test_produces_all_mam4_tracers(self):
        for t in ("m_so4_acc", "m_so4_ait", "m_bc_pcm", "m_poa_pcm",
                  "n_acc", "n_ait", "n_pcm", "g_so2"):
            v = f"aero_emis_{t}"
            self.assertIn(v, self.ds.data_vars)
            arr = self.ds[v].values
            self.assertEqual(arr.shape, self.coords.horizontal.nodal_shape)
            self.assertTrue(np.all(np.isfinite(arr)) and np.all(arr >= 0.0))

    def test_recovers_hammoz_primary_so4_fraction(self):
        # SO₄(115 g/mol, 1 S=32) primary + SO₂ gas(64, 1 S=32) should split as
        # the HAMMOZ/MAM4 2.5 % primary sulfate — recovered from CESM's files.
        s_so4 = (self._tg_per_yr("m_so4_acc")
                 + self._tg_per_yr("m_so4_ait")) * 32.0 / 115.0
        s_gas = self._tg_per_yr("g_so2") * 32.0 / 64.0
        self.assertAlmostEqual(s_so4 / (s_so4 + s_gas), 0.025, places=3)

    def test_total_sulfur_budget_plausible(self):
        s = ((self._tg_per_yr("m_so4_acc") + self._tg_per_yr("m_so4_ait"))
             * 32.0 / 115.0 + self._tg_per_yr("g_so2") * 32.0 / 64.0)
        self.assertGreater(s, 20.0)   # 2014 global anthropogenic S ~50 Tg/yr
        self.assertLess(s, 80.0)


if __name__ == "__main__":
    unittest.main()
