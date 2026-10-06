"""Unit tests for the emissions bundle writer's residential/energy subset

channels (#1017 F6). Synthetic CEDS-shaped Tier A zarr stores only (a
regular lat/lon grid, written to a temp dir) -- no Glade/input4MIPs access.
"""
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from jcm.data.mirror.bundles import _ANTHRO_SECTORS, _ANTHRO_SUBSETS, build_emissions_nc
from jcm.data.mirror.emissions import CEDS_SUBSET_SECTORS
from jcm.data.regridding import conservative_to_gaussian, gaussian_latlon

_SPECIES = ("SO2", "BC", "OC")
_NLAT_SRC, _NLON_SRC = 36, 72   # 5-degree regular source grid


def _synthetic_tier_a(tmp_path, seed=0):
    """Write tiny ``ceds_anthro.zarr``/``bb4cmip7.zarr`` stores with the

    exact ``<SPECIES>_<channel>_<era>_clim`` naming ``build_emissions_nc``
    reads, for every super-sector, both subsets, and biomass burning.
    """
    rng = np.random.default_rng(seed)
    lat = np.linspace(-87.5, 87.5, _NLAT_SRC)
    lon = np.arange(_NLON_SRC) * 5.0
    month = np.arange(1, 13)

    def field():
        # Strictly positive (a flux), (month, lat, lon).
        return rng.uniform(0.1, 1.0, (12, _NLAT_SRC, _NLON_SRC)).astype(np.float32)

    ceds_path = str(tmp_path / "ceds_anthro.zarr")
    bb_path = str(tmp_path / "bb4cmip7.zarr")
    ceds = xr.Dataset(coords={"lat": lat, "lon": lon, "month": month})
    bb = xr.Dataset(coords={"lat": lat, "lon": lon, "month": month})
    for sp in _SPECIES:
        for sector in _ANTHRO_SECTORS:
            da = xr.DataArray(field(), dims=("month", "lat", "lon"),
                              coords={"lat": lat, "lon": lon, "month": month})
            ceds[f"{sp}_{sector}_pd_clim"] = da
        for subset in _ANTHRO_SUBSETS:
            da = xr.DataArray(field(), dims=("month", "lat", "lon"),
                              coords={"lat": lat, "lon": lon, "month": month})
            ceds[f"{sp}_{subset}_pd_clim"] = da
        bb[f"{sp}_pd_clim"] = xr.DataArray(
            field(), dims=("month", "lat", "lon"),
            coords={"lat": lat, "lon": lon, "month": month})
    ceds.to_zarr(ceds_path)
    bb.to_zarr(bb_path)
    return ceds_path, bb_path, lat, lon


def test_subset_channels_are_written_for_every_species():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ceds_path, bb_path, lat, lon = _synthetic_tier_a(tmp_path)
        lats, lons = gaussian_latlon(8)
        out_path = str(tmp_path / "emissions_pd.nc")
        build_emissions_nc(ceds_path, bb_path, "pd", lats, lons, out_path)

        out = xr.open_dataset(out_path)
        for sp in ("so2", "bc", "oc"):
            for subset in CEDS_SUBSET_SECTORS:
                assert f"emis_{subset}_{sp}" in out.data_vars, \
                    f"emis_{subset}_{sp} missing from the bundle"


def test_subset_channel_values_match_a_direct_regrid_of_the_same_source():
    """The bundle's ``emis_residential_so2``/``emis_energy_so2`` must equal

    ``conservative_to_gaussian`` applied directly to the zarr's own
    ``SO2_residential_pd_clim``/``SO2_energy_pd_clim`` array -- the SAME
    regridding and naming contract the super-sector channels already use
    (deliverable 1's requirement), not a separate/approximate path.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ceds_path, bb_path, lat, lon = _synthetic_tier_a(tmp_path)
        lats, lons = gaussian_latlon(8)
        out_path = str(tmp_path / "emissions_pd.nc")
        build_emissions_nc(ceds_path, bb_path, "pd", lats, lons, out_path)

        ceds = xr.open_zarr(ceds_path)
        out = xr.open_dataset(out_path)
        for subset in CEDS_SUBSET_SECTORS:
            raw = ceds[f"SO2_{subset}_pd_clim"].load().values
            expected = conservative_to_gaussian(raw, lat, lon, lats, lons)
            got = out[f"emis_{subset}_so2"].transpose("time", "lon", "lat").values
            np.testing.assert_allclose(got, expected.transpose(0, 2, 1), rtol=1e-6)


def test_regridding_conserves_each_subset_channels_global_total():
    """Area-weighted global mean of a subset channel is preserved by the

    conservative regrid (same check as ``regridding_test.py``'s
    ``test_preserves_global_integral``, applied end to end through
    ``build_emissions_nc`` rather than the bare low-level function).
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ceds_path, bb_path, lat, lon = _synthetic_tier_a(tmp_path)
        lats, lons = gaussian_latlon(16)
        out_path = str(tmp_path / "emissions_pd.nc")
        build_emissions_nc(ceds_path, bb_path, "pd", lats, lons, out_path)

        ceds = xr.open_zarr(ceds_path)
        out = xr.open_dataset(out_path)
        # Regular 5-degree source bands: exact cell edges in sin(lat).
        edges = np.sin(np.deg2rad(np.linspace(-90.0, 90.0, _NLAT_SRC + 1)))
        src_weight = np.diff(edges)
        tgt_weight = np.polynomial.legendre.leggauss(16)[1]
        for subset in CEDS_SUBSET_SECTORS:
            raw = ceds[f"BC_{subset}_pd_clim"].load().values  # (month, lat, lon)
            src_total = (raw * src_weight[None, :, None]).sum() / raw.shape[-1]
            got = out[f"emis_{subset}_bc"].transpose("time", "lat", "lon").values
            tgt_total = (got * tgt_weight[None, :, None]).sum() / got.shape[-1]
            assert abs(tgt_total / src_total - 1.0) < 1e-6, subset
