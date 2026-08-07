"""Super-sectored CEDS + BB4CMIP7 emissions mirror (the #515 product).

Two Tier A stores, each at its source's native resolution ("always regrid
from the highest resolution available" — regridding happens once, at
bundle assembly, straight from these):

* ``ceds_anthro.zarr`` — CEDS-CMIP-2025-04-18 anthropogenic flux summed
  over the 8 CEDS sectors ("surface_combustion" super-sector), 0.5°,
  monthly 1850–2023, per species.
* ``bb4cmip7.zarr`` — DRES BB4CMIP7-2-0 open-burning flux
  ("biomass_burning" super-sector), 0.25°, monthly 1850–2023, per species.

Both stores also carry 12-month PI (1850–1859) and PD (2005–2014)
climatology arrays so bundles need no time arithmetic.
"""

from __future__ import annotations

import glob

import numpy as np
import xarray as xr

CEDS_ROOT = ("/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/"
             "CMIP7/CMIP/PNNL-JGCRI/CEDS-CMIP-2025-04-18/atmos/mon")
BB_ROOT = ("/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/"
           "CMIP7/CMIP/DRES/DRES-CMIP-BB4CMIP7-2-0/atmos/mon")

SPECIES = ("SO2", "BC", "OC", "NH3")
PI_YEARS = ("1850-01-01", "1859-12-31")
PD_YEARS = ("2005-01-01", "2014-12-31")


def _climatologies(da: xr.DataArray) -> dict[str, xr.DataArray]:
    out = {}
    for tag, (t0, t1) in (("pi", PI_YEARS), ("pd", PD_YEARS)):
        clim = (da.sel(time=slice(t0, t1)).groupby("time.month")
                .mean("time").astype(np.float32))
        out[f"{da.name}_{tag}_clim"] = clim
    return out


def load_ceds_species(species: str) -> xr.DataArray:
    """CEDS flux summed over sectors, monthly 1850–2023 (kg m-2 s-1)."""
    files = sorted(glob.glob(
        f"{CEDS_ROOT}/{species}_em_anthro/gn/*/*.nc"))
    files = [f for f in files if not f.split("gn_")[-1].startswith("17")]
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 120},
                           data_vars="minimal", coords="minimal",
                           compat="override")
    da = ds[f"{species}_em_anthro"].sel(time=slice("1850-01-01", None))
    return da.sum("sector").astype(np.float32).rename(species)


def load_bb_species(species: str) -> xr.DataArray:
    """BB4CMIP7 open-burning flux, monthly 1850–2023 (kg m-2 s-1)."""
    files = sorted(glob.glob(f"{BB_ROOT}/{species}/gn/*/*.nc"))
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 120},
                           data_vars="minimal", coords="minimal",
                           compat="override")
    da = ds[species].sel(time=slice("1850-01-01", None))
    return (da.rename({"latitude": "lat", "longitude": "lon"})
            .astype(np.float32))


def build_store(loader, species, out_path: str, source_attr: str) -> None:
    """Stream one species at a time into a zarr store."""
    for i, sp in enumerate(species):
        da = loader(sp)
        ds = xr.Dataset({sp: da.chunk({"time": 12}), **{
            k: v for k, v in _climatologies(da).items()}})
        ds.attrs["source"] = source_attr
        ds.attrs["units"] = "kg m-2 s-1"
        mode = "w" if i == 0 else "a"
        ds.to_zarr(out_path, mode=mode)
        print(f"  {sp} -> {out_path}", flush=True)
