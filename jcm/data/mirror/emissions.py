"""Super-sectored CEDS + BB4CMIP7 + HAM biogenic emissions mirror (#515, #1017).

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

:func:`load_biogenic_oc` (jax-gcm#1017, maintainer decision F7) is a third,
much simpler source: HAM's own AeroCom II biogenic-OC file, already a
single T63 monthly climatology (no PI/PD transient split -- HAM replays it
every model year regardless of era), feeding the M7 sector policy's
standalone ``"biogenic"`` class (``emissions/ham_sectors.py``) rather than
any of the four model super-sectors above.
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


#: HAM's reference run emits BIOGENIC OC from this AeroCom II file (the
#: emi_spec row ``BIOGENIC=EF_FILE, ..., emiss_biogenic, EF_LONLAT,
#: surface``). Already on the T63 model grid.
HAM_BIOGENIC_OC_FILE = (
    "/data/climate-analytics-lab-shared/ECHAM_emissions_v0006/hammoz/T63/"
    "aerocom_II/T63/2000/emiss_aerocom_OC_monthly_2000_T63.nc")


def load_biogenic_oc(path: str = HAM_BIOGENIC_OC_FILE) -> xr.DataArray:
    """HAM's biogenic OC source, a T63 monthly climatology (year 2000).

    Returns the raw ``emiss_biogenic`` variable [kg m-2 s-1] (12 monthly
    records, T63 gaussian grid, 192x96) renamed ``oc_biogenic`` -- a single
    AeroCom II climatology HAM replays every model year (unlike the CEDS/
    BB4CMIP7 sources above, which are a 1850-2023 transient series with a
    PI/PD split), so :func:`~jcm.data.mirror.bundles.build_emissions_nc`
    reads it once and uses the same climatology for every ``era``. The
    file's ``long_name`` ("SOA emissions") is a label inherited from its
    processing chain (``RG_source_file`` points at an AeroCom "SOA" 1x1
    file); HAM itself treats this mass as primary OC when its own SOA
    scheme is off (``nsoa /= 1``, ``mo_ham_m7_emissions.f90:596``), which is
    why the M7 sector policy's ``"biogenic"`` class (``emissions/
    ham_sectors.py``) applies no OM:OC scaling to it.

    Global annual total ~19.06 Tg/yr (area-weighted with
    ``jcm.analysis.area_weights``), matching the 19.1 Tg/yr AeroCom-II
    biogenic-OC source multiple models including ECHAM5-HAMMOZ use
    (Dentener et al. 2006, cited in Tsigaridis et al. 2014, ACP 14,
    10845-10895, https://doi.org/10.5194/acp-14-10845-2014) to 0.2%.

    Build/verify (one command, writes to a scratch path -- never uploads)::

        python -c "
        from jcm.data.mirror.emissions import load_biogenic_oc
        from jcm.analysis import area_weights, global_mean
        da = load_biogenic_oc()
        gm = global_mean(da, area_weights(da))
        print(float(gm.mean()) * 4*3.14159265*6371000.0**2 * 86400*365.25/1e9, 'Tg/yr')
        da.to_netcdf('/scr/<you>/oc_biogenic_t63.nc')"
    """
    ds = xr.open_dataset(path)
    return ds["emiss_biogenic"].rename("oc_biogenic").astype(np.float32)


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
