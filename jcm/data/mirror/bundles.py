"""Assemble per-grid boundary-condition bundles from the Tier A products.

For each Gaussian grid this produces the two files the spectral model
reads directly:

* ``terrain.nc`` — jcm-canonical layout (lowercase vars, ``(lon, lat)``
  axis order, ascending latitudes): GMTED2010 SSO statistics masked with
  the ERA5 invariant land-sea mask.
* ``forcing_<era>.nc`` — ``(lon, lat, time)`` 12-month climatology:
  SST + sea ice from PCMDI-AMIP-1-1-10 (PD 2005–2014; "PI" uses
  1870–1879, the earliest observed decade — flagged in the file attrs
  since no observational 1850 SST exists), land fields from the ERA5
  climatology (identical for both eras).

Unit translations into the conventions the packaged t63 files establish
(and ``_validate_bc_fields`` checks):

* ``sst``   = tos [degC] + 273.15, land filled with nearest-ocean value
* ``icec``  = siconc [%] / 100, land = 0
* ``stl``   = ERA5 stl1 [K]
* ``snowc`` = ERA5 sd [m w.e.] capped at 0.2, zeroed where snow persists
  year-round (ice sheets) — matching ECHAM's SN, which excludes glaciers:
  their high albedo lives in ``alb``, and blending toward fresh-snow
  albedo would darken them
* ``alb``   = per-cell minimum monthly ERA5 fal — the snow-free
  background albedo (snow brightening is applied dynamically from
  ``snowc``; an annual mean would double-count it)
* ``soilw_am`` = root-zone soil-water depth [m], Σᵢ swvlᵢ·Dᵢ with
  D = (0.07, 0.21, 0.72) m — matching the packaged field's ECHAM
  WS-bucket convention (land mean ≈ 0.16 m), not an availability
  fraction
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from jcm.data.regridding import (conservative_to_gaussian, fill_nearest,
                                 interp_to)

AMIP_ROOT = ("/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/"
             "CMIP7/CMIP/PCMDI/PCMDI-AMIP-1-1-10")
TOS = (f"{AMIP_ROOT}/ocean/mon/tos/gn/v20250807/"
       "tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_"
       "187001-202212.nc")
SICONC = (f"{AMIP_ROOT}/seaIce/mon/siconc/gn/v20250807/"
          "siconc_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_"
          "187001-202212.nc")

ERA_YEARS = {"pd": ("2005-01-01", "2014-12-31"),
             "pi": ("1870-01-01", "1879-12-31")}

_SOIL_D = np.array([0.07, 0.21, 0.72])   # ERA5 soil-layer depths [m]
SNOW_CAP_M = 0.2                   # packaged ECHAM SN climatology ceiling

SSO_FIELDS = ("orog", "orostd", "orosig", "orogam", "orothe",
              "oropic", "oroval")

# 12 month-start timestamps: interpolate_to_daily requires pd.infer_freq
# "MS"/"M", and align_mode='auto' then resolves to WRAP_YEAR (climatology)
# indexing, so the year itself is arbitrary.
CLIMO_TIME = np.array([np.datetime64(f"2014-{m:02d}-01")
                       for m in range(1, 13)])





def _monthly_clim(da: xr.DataArray, era: str) -> xr.DataArray:
    t0, t1 = ERA_YEARS[era]
    return (da.sel(time=slice(t0, t1)).groupby("time.month").mean("time")
            .rename(month="time"))


def _to_lonlat(da2d: xr.DataArray) -> tuple:
    """(lat, lon[, time]) DataArray -> jcm-canonical (lon, lat[, time])."""
    dims = ("lon", "lat") + tuple(d for d in da2d.dims
                                  if d not in ("lat", "lon"))
    return dims, da2d.transpose(*dims).values



_EMIS_SPECIES = ("so2", "bc", "oc")


def build_emissions_nc(ceds_zarr: str, bb_zarr: str, era: str,
                       lats, lons, out_path: str) -> None:
    """Per-grid emissions file in the model's 6-variable contract."""
    ceds = xr.open_zarr(ceds_zarr)
    bb = xr.open_zarr(bb_zarr)
    ds = xr.Dataset(coords={"lat": lats, "lon": lons,
                            "time": CLIMO_TIME})
    for sp in _EMIS_SPECIES:
        up = sp.upper()
        for prefix, store in (("surface_combustion", ceds),
                              ("biomass_burning", bb)):
            da = store[f"{up}_{era}_clim"].load()
            arr = conservative_to_gaussian(
                np.nan_to_num(da.values), da.lat.values, da.lon.values,
                lats, lons)
            ds[f"emis_{prefix}_{sp}"] = (
                ("time", "lon", "lat"), arr.transpose(0, 2, 1),
                {"units": "kg m-2 s-1"})
    ds.attrs = {
        "title": ("jax-gcm prescribed emissions (bulk per-super-sector "
                  "surface flux)"),
        "era": era,
        "source": "CEDS-CMIP-2025-04-18 + DRES-CMIP-BB4CMIP7-2-0",
    }
    ds.to_netcdf(out_path)
    print("wrote", out_path, flush=True)


def build_terrain(sso_path: str, era5_path: str, out_path: str) -> None:
    """Gaussian terrain.nc: GMTED SSO fields + fractional ERA5 land mask.

    ``lsm`` stays a fraction — ``TerrainData``'s ``fmask`` is consumed
    fractionally (coastal flux blending) and the packaged climatology is
    fractional too. SSO fields are zeroed only below 10% land (matching
    ``get_terrain``'s snap threshold): open-ocean cells lose the
    shoreline-step artifacts of the DEM while islands and coasts keep
    their orography.
    """
    sso = xr.open_dataset(sso_path)
    era5 = xr.open_dataset(era5_path)
    lats, lons = sso.lat.values, sso.lon.values
    lsm_frac = np.clip(interp_to(era5.lsm, lats, lons).values, 0.0, 1.0)
    keep = lsm_frac >= 0.1
    ds = xr.Dataset(coords={"lat": lats, "lon": lons})
    ds["lsm"] = (("lon", "lat"), lsm_frac.T)
    for name in SSO_FIELDS:
        ds[name] = (("lon", "lat"), np.where(keep, sso[name].values, 0.0).T)
    ds.attrs = dict(sso.attrs)
    ds.attrs["lsm_source"] = "ERA5 invariant land fraction (0.25 deg)"
    ds.to_netcdf(out_path)
    print("wrote", out_path, flush=True)


def build_forcing(era5_path: str, era: str, lats, lons,
                  out_path: str) -> None:
    era5 = xr.open_dataset(era5_path).rename(month="time")

    tos = xr.open_dataset(TOS).tos
    sic = xr.open_dataset(SICONC).siconc
    sst_c = _monthly_clim(tos, era)
    sst = fill_nearest(sst_c.values, sst_c.lat.values, sst_c.lon.values)
    sst_da = xr.DataArray(sst + 273.15, dims=("time", "lat", "lon"),
                          coords={"time": CLIMO_TIME,
                                  "lat": sst_c.lat, "lon": sst_c.lon})
    icec_c = _monthly_clim(sic, era) / 100.0
    icec_da = icec_c.fillna(0.0)

    soilw = (_SOIL_D[0] * era5.swvl1 + _SOIL_D[1] * era5.swvl2
             + _SOIL_D[2] * era5.swvl3)

    fields = {
        "sst": interp_to(sst_da, lats, lons),
        "icec": interp_to(icec_da, lats, lons).clip(0.0, 1.0),
        "stl": interp_to(era5.stl1, lats, lons),
        "soilw_am": interp_to(soilw, lats, lons).clip(min=0.0),
        "snowc": interp_to(
            era5.sd.clip(0.0, SNOW_CAP_M).where(
                era5.sd.min("time") < 0.5 * SNOW_CAP_M, 0.0),
            lats, lons).clip(min=0.0),
        "alb": interp_to(era5.fal.min("time"), lats, lons),
    }
    ds = xr.Dataset(coords={"lat": lats, "lon": lons,
                            "time": CLIMO_TIME})
    for name, da in fields.items():
        ds[name] = _to_lonlat(da)
    ds.attrs = {
        "source": ("SST/ice: PCMDI-AMIP-1-1-10; land: ERA5 monthly "
                   "climatology 2005-2014"),
        "era": era,
        "note": ("PI SST/ice is the 1870-1879 mean — the earliest "
                 "observed decade, not a true 1850 state"
                 if era == "pi" else "PD climatology 2005-2014"),
    }
    ds.to_netcdf(out_path)
    print("wrote", out_path, flush=True)
