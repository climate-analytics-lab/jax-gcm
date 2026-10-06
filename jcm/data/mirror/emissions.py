"""Super-sectored CEDS + BB4CMIP7 emissions mirror (the #515 product).

Two Tier A stores, each at its source's native resolution ("always regrid
from the highest resolution available" — regridding happens once, at
bundle assembly, straight from these):

* ``ceds_anthro.zarr`` — CEDS-CMIP-2025-04-18 anthropogenic flux summed
  into the model's three anthropogenic super-sectors (see
  ``jcm.physics.aerosol.jam.emissions.sectors``): ``surface_combustion``
  (AGR/TRA/RCO/SLV/WST), ``elevated_industrial`` (ENE/IND — 50 m
  injection) and ``shipping`` (SHP), PLUS two single-sector SUBSET
  channels (``CEDS_SUBSET_SECTORS``, jax-gcm#1017 F6) each already
  included in one of those sums: ``residential`` (RCO, sector 4, inside
  ``surface_combustion``) and ``energy`` (ENE, sector 1, inside
  ``elevated_industrial``) — HAM's M7 submodel sizes these two
  differently from the rest of their super-sector
  (``mo_ham_m7_emissions.f90:564-646``), so the model's M7 emission
  policy (``emissions/ham_sectors.py``) reads them when present to split
  that share out at its own size, rather than the whole super-sector's
  parent size. 0.5°, monthly 1850–2023, per species.
* ``bb4cmip7.zarr`` — DRES BB4CMIP7-2-0 open-burning flux
  ("biomass_burning" super-sector), 0.25°, monthly 1850–2023, per species.

Both stores also carry 12-month PI (1850–1859) and PD (2005–2014)
climatology arrays so bundles need no time arithmetic.
"""

from __future__ import annotations

import glob

import numpy as np
import xarray as xr

from jcm.data.mirror import sites

CEDS_ROOT = sites.input4mips(
    "CMIP7/CMIP/PNNL-JGCRI/CEDS-CMIP-2025-04-18/atmos/mon")
BB_ROOT = sites.input4mips("CMIP7/CMIP/DRES/DRES-CMIP-BB4CMIP7-2-0/atmos/mon")

SPECIES = ("SO2", "BC", "OC", "NH3")
PI_YEARS = ("1850-01-01", "1859-12-31")
PD_YEARS = ("2005-01-01", "2014-12-31")

# CEDS sector indices (file convention: 0 Agriculture, 1 Energy,
# 2 Industrial, 3 Transportation, 4 Residential/Commercial/Other,
# 5 Solvents, 6 Waste, 7 International Shipping) grouped into the
# model's anthropogenic super-sectors — the split carries the injection
# altitude distinction (elevated_industrial -> ~50 m stacks).
CEDS_SUPER_SECTORS = {
    "surface_combustion": [0, 3, 4, 5, 6],
    "elevated_industrial": [1, 2],
    "shipping": [7],
}

# Two single-sector SUBSET channels, each already included in one of the
# super-sectors above (NOT additional mass): HAM sizes these differently
# from the rest of their super-sector (``mo_ham_m7_emissions.f90:564-646``,
# jax-gcm#1017 F6) -- residential/commercial/other (RCO, sector 4, inside
# ``surface_combustion``) like biomass burning, and energy (ENE, sector 1,
# inside ``elevated_industrial``) with its primary SO4 in accumulation+
# coarse rather than Aitken+accumulation. ``emissions/ham_sectors.py``'s
# ``SECTOR_ROUTING`` already reads ``emis_residential_<sp>``/
# ``emis_energy_<sp>`` when present and falls back to the parent
# super-sector's own sizing (with a warning) when absent -- this dict is
# what lets the bundle builder actually supply them.
CEDS_SUBSET_SECTORS = {
    "residential": [4],
    "energy": [1],
}


def _climatologies(da: xr.DataArray) -> dict[str, xr.DataArray]:
    out = {}
    for tag, (t0, t1) in (("pi", PI_YEARS), ("pd", PD_YEARS)):
        clim = (da.sel(time=slice(t0, t1)).groupby("time.month")
                .mean("time").astype(np.float32))
        out[f"{da.name}_{tag}_clim"] = clim
    return out


def load_ceds_species(species: str) -> list[xr.DataArray]:
    """CEDS flux per anthropogenic super-sector AND subset, monthly 1850–2023.

    One ``<SPECIES>_<super_sector>`` array (kg m-2 s-1) per entry of
    ``CEDS_SUPER_SECTORS`` — the sector split carries the injection
    altitudes the model's emission terms apply — PLUS one
    ``<SPECIES>_<subset>`` array per entry of ``CEDS_SUBSET_SECTORS``: each
    subset's sector is already summed into one of the super-sectors above
    (it is the model's HAM sizing split, not a fifth injection group), so
    its own flux is carried alongside, never instead of, its parent's.
    Both go through the identical ``isel(sector=...).sum("sector")`` ->
    ``build_store``/``_climatologies`` path, so the subset channels get
    the same regridding and climatology treatment as every other channel
    this module writes.
    """
    files = sorted(glob.glob(
        f"{CEDS_ROOT}/{species}_em_anthro/gn/*/*.nc"))
    files = [f for f in files if not f.split("gn_")[-1].startswith("17")]
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 120},
                           data_vars="minimal", coords="minimal",
                           compat="override")
    da = ds[f"{species}_em_anthro"].sel(time=slice("1850-01-01", None))
    return [da.isel(sector=idx).sum("sector").astype(np.float32)
            .rename(f"{species}_{name}")
            for name, idx in {**CEDS_SUPER_SECTORS, **CEDS_SUBSET_SECTORS}.items()]


def load_bb_species(species: str) -> list[xr.DataArray]:
    """BB4CMIP7 open-burning flux, monthly 1850–2023 (kg m-2 s-1)."""
    files = sorted(glob.glob(f"{BB_ROOT}/{species}/gn/*/*.nc"))
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 120},
                           data_vars="minimal", coords="minimal",
                           compat="override")
    da = ds[species].sel(time=slice("1850-01-01", None))
    return [da.rename({"latitude": "lat", "longitude": "lon"})
            .astype(np.float32)]


def build_store(loader, species, out_path: str, source_attr: str) -> None:
    """Stream one species at a time into a zarr store (resumable)."""
    import os
    for sp in species:
        arrays = loader(sp)
        if os.path.exists(f"{out_path}/{arrays[-1].name}_pd_clim"):
            print(f"  {sp} already in {out_path}, skipping", flush=True)
            continue
        ds = xr.Dataset()
        for da in arrays:
            ds[da.name] = da.chunk({"time": 12})
            for k, v in _climatologies(da).items():
                ds[k] = v
        ds.attrs["source"] = source_attr
        ds.attrs["units"] = "kg m-2 s-1"
        mode = "w" if not os.path.exists(out_path) else "a"
        ds.to_zarr(out_path, mode=mode)
        print(f"  {sp} -> {out_path}", flush=True)
