"""Tests for the HAM-specific emission sources (jax-gcm#1017).

Biogenic OC (decision F7) and the CEDS super-sector/subset channel builder
(F6). Synthetic CEDS-shaped inputs only (no Glade/input4MIPs access): the file
loaders (``xr.open_mfdataset``, ``glob.glob``) are monkeypatched so
``load_ceds_species`` runs its real sector-selection logic end to end
against an in-memory ``xr.Dataset`` with the same ``(time, sector, lat,
lon)`` layout the real CEDS files have.
"""

import os
import tempfile
import unittest

import numpy as np
import xarray as xr

from jcm.data.mirror import emissions as em
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
            # Super-sectors plus the HAM-sizing subset channels (F6) the
            # bundle builder regrids alongside them.
            for sector in ("surface_combustion", "elevated_industrial", "shipping",
                           *em.CEDS_SUBSET_SECTORS):
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


# CEDS sector indices this module documents: 0 AGR, 1 ENE, 2 IND, 3 TRA,
# 4 RCO, 5 SLV, 6 WST, 7 SHP.
N_SECTOR = 8


def _synthetic_ceds_dataset(species="SO2", seed=0):
    rng = np.random.default_rng(seed)
    time = np.array(["1850-01-01", "1850-02-01", "2005-06-01"], dtype="datetime64[ns]")
    lat = np.array([-45.0, -15.0, 15.0, 45.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    data = rng.random((time.size, N_SECTOR, lat.size, lon.size))
    da = xr.DataArray(
        data, dims=("time", "sector", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name=f"{species}_em_anthro",
    )
    return xr.Dataset({da.name: da})


def _patch_loader(monkeypatch, ds):
    monkeypatch.setattr(em.glob, "glob", lambda pattern: ["fake_0000.nc"])
    monkeypatch.setattr(em.xr, "open_mfdataset", lambda *a, **k: ds)


def test_subset_arrays_equal_the_corresponding_sector_slice(monkeypatch):
    ds = _synthetic_ceds_dataset("SO2")
    _patch_loader(monkeypatch, ds)
    arrays = {da.name: da for da in em.load_ceds_species("SO2")}

    raw = ds["SO2_em_anthro"]
    for name, idx in em.CEDS_SUBSET_SECTORS.items():
        assert len(idx) == 1, f"{name} is documented as a single-sector subset"
        expected = raw.isel(sector=idx).sum("sector").astype(np.float32)
        np.testing.assert_array_equal(
            arrays[f"SO2_{name}"].values, expected.values)


def test_super_sector_sums_are_unchanged_by_the_subset_addition(monkeypatch):
    """The subset channels are ADDITIONAL arrays, not a change to the

    existing super-sector sums -- each super-sector still sums exactly the
    sector indices ``CEDS_SUPER_SECTORS`` always named (RCO/ENE included).
    """
    ds = _synthetic_ceds_dataset("BC")
    _patch_loader(monkeypatch, ds)
    arrays = {da.name: da for da in em.load_ceds_species("BC")}

    raw = ds["BC_em_anthro"]
    for name, idx in em.CEDS_SUPER_SECTORS.items():
        expected = raw.isel(sector=idx).sum("sector").astype(np.float32)
        np.testing.assert_array_equal(
            arrays[f"BC_{name}"].values, expected.values)
    # Sanity: the two subsets are genuinely INSIDE a super-sector's index
    # list, not disjoint from every one of them.
    assert em.CEDS_SUBSET_SECTORS["residential"][0] in em.CEDS_SUPER_SECTORS["surface_combustion"]
    assert em.CEDS_SUBSET_SECTORS["energy"][0] in em.CEDS_SUPER_SECTORS["elevated_industrial"]


def test_load_ceds_species_returns_every_super_sector_and_subset_once(monkeypatch):
    ds = _synthetic_ceds_dataset("OC")
    _patch_loader(monkeypatch, ds)
    names = {da.name for da in em.load_ceds_species("OC")}
    expected = {f"OC_{n}" for n in em.CEDS_SUPER_SECTORS} | {
        f"OC_{n}" for n in em.CEDS_SUBSET_SECTORS}
    assert names == expected


def test_subset_sector_count_matches_the_module_docstring():
    """Pins the exact CEDS sector each subset channel reads (RCO=4, ENE=1),

    the HAM-sizing targets ``mo_ham_m7_emissions.f90:564-646`` keys on --
    a silent reindex here would misassign which flux gets HAM's biomass-
    like or energy/ships sizing.
    """
    assert em.CEDS_SUBSET_SECTORS == {"residential": [4], "energy": [1]}


if __name__ == "__main__":
    unittest.main()
