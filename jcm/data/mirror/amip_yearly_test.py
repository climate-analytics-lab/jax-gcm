"""Unit tests for ``build_emissions_year``'s optional ``biogenic_oc`` parameter

(jax-gcm#1017 F7, jax-gcm#1045 review comment 4197639963): the transient
twin of ``bundles.build_emissions_nc``'s own ``biogenic_oc`` support
(``emissions_test.py::BuildEmissionsNcBiogenicTest``), feeding the SAME
single HAM AeroCom-II climatology onto a requested year's time axis rather
than slicing a transient series (there is none to slice).
"""

import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from jcm.data.mirror.amip_yearly import build_emissions_year
from jcm.data.mirror.bundles import _ANTHRO_SECTORS, _ANTHRO_SUBSETS
from jcm.data.regridding import gaussian_latlon

_SPECIES = ("SO2", "BC", "OC")
_NLAT_SRC, _NLON_SRC = 36, 72
_YEAR = 2000


def _synthetic_transient_ceds_bb(tmp_path: Path) -> tuple[str, str]:
    """Write a one-year slice of ``ceds_anthro.zarr``/``bb4cmip7.zarr``'s

    transient (non-``_clim``) variables -- what ``build_emissions_year``
    reads, keyed directly by species/sector with a real ``time`` axis
    (unlike ``build_emissions_nc``'s ``_<era>_clim`` + ``month`` convention).
    """
    lat = np.linspace(-87.5, 87.5, _NLAT_SRC)
    lon = np.arange(_NLON_SRC) * 5.0
    time = xr.date_range(f"{_YEAR}-01-01", periods=12, freq="MS")
    rng = np.random.default_rng(0)

    def field():
        return rng.uniform(0.1, 1.0, (12, _NLAT_SRC, _NLON_SRC)).astype(np.float32)

    ceds = xr.Dataset(coords={"lat": lat, "lon": lon, "time": time})
    bb = xr.Dataset(coords={"lat": lat, "lon": lon, "time": time})
    for sp in _SPECIES:
        for sector in _ANTHRO_SECTORS:
            ceds[f"{sp}_{sector}"] = (("time", "lat", "lon"), field())
        for subset in _ANTHRO_SUBSETS:
            ceds[f"{sp}_{subset}"] = (("time", "lat", "lon"), field())
        bb[sp] = (("time", "lat", "lon"), field())
    ceds_path, bb_path = str(tmp_path / "ceds.zarr"), str(tmp_path / "bb.zarr")
    ceds.to_zarr(ceds_path)
    bb.to_zarr(bb_path)
    return ceds_path, bb_path


def _synthetic_biogenic_oc() -> xr.DataArray:
    lat = np.linspace(-87.5, 87.5, _NLAT_SRC)
    lon = np.arange(_NLON_SRC) * 5.0
    rng = np.random.default_rng(1)
    data = rng.uniform(1.0e-13, 1.0e-12, (12, _NLAT_SRC, _NLON_SRC)).astype(np.float32)
    return xr.DataArray(data, dims=("time", "lat", "lon"),
                        coords={"lat": lat, "lon": lon}, name="oc_biogenic")


def test_emis_biogenic_oc_written_when_supplied():
    lats, lons = gaussian_latlon(8)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ceds_path, bb_path = _synthetic_transient_ceds_bb(tmp_path)
        out_path = str(tmp_path / f"{_YEAR}.nc")
        build_emissions_year(ceds_path, bb_path, _YEAR, lats, lons, out_path,
                             biogenic_oc=_synthetic_biogenic_oc())
        with xr.open_dataset(out_path) as ds:
            assert "emis_biogenic_oc" in ds.data_vars
            assert ds["emis_biogenic_oc"].attrs["units"] == "kg m-2 s-1"
            assert bool(np.all(np.isfinite(ds["emis_biogenic_oc"].values)))
            assert ds.sizes["time"] == 12


def test_emis_biogenic_oc_absent_by_default():
    lats, lons = gaussian_latlon(8)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ceds_path, bb_path = _synthetic_transient_ceds_bb(tmp_path)
        out_path = str(tmp_path / f"{_YEAR}.nc")
        build_emissions_year(ceds_path, bb_path, _YEAR, lats, lons, out_path)
        with xr.open_dataset(out_path) as ds:
            assert "emis_biogenic_oc" not in ds.data_vars
