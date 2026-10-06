"""Unit tests for the CEDS super-sector/subset channel builder (#1017 F6).

Synthetic CEDS-shaped inputs only (no Glade/input4MIPs access): the file
loaders (``xr.open_mfdataset``, ``glob.glob``) are monkeypatched so
``load_ceds_species`` runs its real sector-selection logic end to end
against an in-memory ``xr.Dataset`` with the same ``(time, sector, lat,
lon)`` layout the real CEDS files have.
"""
import numpy as np
import xarray as xr

from jcm.data.mirror import emissions as em

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
