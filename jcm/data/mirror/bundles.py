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
* ``snowc`` = ERA5 sd [m w.e.] (snow-cover factor ``min(1, snowc)``
  downstream)
* ``alb``   = annual-mean ERA5 fal
* ``soilw_am`` — evaporation-availability factor in [0, 1]: root-zone
  volumetric soil water relative to SPEEDY's field capacity
  (``swcap`` = 0.30), layers weighted by depth with the deep layer
  discounted:
  ``(D1·swvl1 + D2·swvl2 + w·D3·swvl3) / (swcap·(D1 + D2 + w·D3))``,
  D = (0.07, 0.21, 0.72) m, w = 0.3.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

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

SWCAP = 0.30                       # SPEEDY field capacity (vol. fraction)
_SOIL_D = np.array([0.07, 0.21, 0.72])   # ERA5 soil-layer depths [m]
_DEEP_W = 0.3

SSO_FIELDS = ("orog", "orostd", "orosig", "orogam", "orothe",
              "oropic", "oroval")

# 12 month-start timestamps: interpolate_to_daily requires pd.infer_freq
# "MS"/"M", and align_mode='auto' then resolves to WRAP_YEAR (climatology)
# indexing, so the year itself is arbitrary.
CLIMO_TIME = np.array([np.datetime64(f"2014-{m:02d}-01")
                       for m in range(1, 13)])


def gaussian_latlon(nlat: int):
    lats = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(nlat)[0]))
    return lats, np.arange(2 * nlat) * 360.0 / (2 * nlat)


def interp_to(da: xr.DataArray, lats, lons) -> xr.DataArray:
    """Bilinear regrid with periodic longitude wrap; lat clamped at ends."""
    latn, lonn = da.dims[-2], da.dims[-1]
    if float(da[latn][0]) > float(da[latn][-1]):
        da = da.isel({latn: slice(None, None, -1)})
    dlon = float(da[lonn][1] - da[lonn][0])
    wrapped = xr.concat(
        [da.isel({lonn: -1}).assign_coords(
            {lonn: float(da[lonn][0]) - dlon}),
         da,
         da.isel({lonn: 0}).assign_coords(
             {lonn: float(da[lonn][-1]) + dlon})], dim=lonn)
    # constant extension to the poles so Gaussian lats beyond the source's
    # first/last row interpolate instead of going NaN
    if float(wrapped[latn][0]) > -90.0:
        wrapped = xr.concat(
            [wrapped.isel({latn: 0}).assign_coords({latn: -90.0}), wrapped],
            dim=latn)
    if float(wrapped[latn][-1]) < 90.0:
        wrapped = xr.concat(
            [wrapped, wrapped.isel({latn: -1}).assign_coords({latn: 90.0})],
            dim=latn)
    out = wrapped.interp({latn: lats, lonn: lons}, method="linear")
    return out.rename({latn: "lat", lonn: "lon"})


def _fill_nearest(field: np.ndarray, lats, lons) -> np.ndarray:
    """Fill NaNs (land in ocean products) with the nearest valid value."""
    glon, glat = np.meshgrid(np.deg2rad(lons), np.deg2rad(lats))
    xyz = np.stack([np.cos(glat) * np.cos(glon),
                    np.cos(glat) * np.sin(glon), np.sin(glat)], -1)
    out = field.copy()
    for t in range(field.shape[0]):
        bad = ~np.isfinite(field[t])
        if not bad.any():
            continue
        tree = cKDTree(xyz[~bad])
        _, idx = tree.query(xyz[bad], workers=-1)
        out[t][bad] = field[t][~bad][idx]
    return out


def _monthly_clim(da: xr.DataArray, era: str) -> xr.DataArray:
    t0, t1 = ERA_YEARS[era]
    return (da.sel(time=slice(t0, t1)).groupby("time.month").mean("time")
            .rename(month="time"))


def _to_lonlat(da2d: xr.DataArray) -> tuple:
    """(lat, lon[, time]) DataArray -> jcm-canonical (lon, lat[, time])."""
    dims = ("lon", "lat") + tuple(d for d in da2d.dims
                                  if d not in ("lat", "lon"))
    return dims, da2d.transpose(*dims).values


def conservative_to_gaussian(field: np.ndarray, src_lats, src_lons,
                             lats, lons) -> np.ndarray:
    """Area-weighted binning of a regular-grid flux onto Gaussian cells.

    Each source cell contributes its cos(lat)-weighted value to the target
    cell containing its center — conservative in the flux-density sense,
    which is what per-m² emissions need (bilinear would smear point
    sources and lose mass).
    """
    lat_edges = np.concatenate([[-90.0], 0.5 * (lats[1:] + lats[:-1]),
                                [90.0]])
    nlat, nlon = lats.size, lons.size
    dlon = 360.0 / nlon
    lat_bin = np.clip(np.searchsorted(lat_edges, src_lats) - 1, 0, nlat - 1)
    lon_bin = ((np.asarray(src_lons) - (lons[0] - dlon / 2)) // dlon
               ).astype(int) % nlon
    flat = (lat_bin[:, None] * nlon + lon_bin[None, :]).ravel()
    w = np.cos(np.deg2rad(src_lats))[:, None].repeat(len(src_lons), 1).ravel()
    wsum = np.bincount(flat, weights=w, minlength=nlat * nlon)
    lead = field.shape[:-2]
    out = np.empty(lead + (nlat, nlon))
    for idx in np.ndindex(lead):
        num = np.bincount(flat, weights=w * field[idx].ravel(),
                          minlength=nlat * nlon)
        out[idx] = (num / np.maximum(wsum, 1e-30)).reshape(nlat, nlon)
    return out


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
    sso = xr.open_dataset(sso_path)
    era5 = xr.open_dataset(era5_path)
    lats, lons = sso.lat.values, sso.lon.values
    lsm_frac = interp_to(era5.lsm, lats, lons).values
    lsm = (lsm_frac > 0.5).astype(np.float64)
    ds = xr.Dataset(coords={"lat": lats, "lon": lons})
    ds["lsm"] = (("lon", "lat"), lsm.T)
    for name in SSO_FIELDS:
        ds[name] = (("lon", "lat"), np.where(lsm, sso[name].values, 0.0).T)
    ds.attrs = dict(sso.attrs)
    ds.attrs["lsm_source"] = "ERA5 invariant land-sea mask (>0.5)"
    ds.to_netcdf(out_path)
    print("wrote", out_path, flush=True)


def build_forcing(era5_path: str, era: str, lats, lons,
                  out_path: str) -> None:
    era5 = xr.open_dataset(era5_path).rename(month="time")

    tos = xr.open_dataset(TOS).tos
    sic = xr.open_dataset(SICONC).siconc
    sst_c = _monthly_clim(tos, era)
    sst = _fill_nearest(sst_c.values, sst_c.lat.values, sst_c.lon.values)
    sst_da = xr.DataArray(sst + 273.15, dims=("time", "lat", "lon"),
                          coords={"time": CLIMO_TIME,
                                  "lat": sst_c.lat, "lon": sst_c.lon})
    icec_c = _monthly_clim(sic, era) / 100.0
    icec_da = icec_c.fillna(0.0)

    d = _SOIL_D * np.array([1.0, 1.0, _DEEP_W])
    soilw = (d[0] * era5.swvl1 + d[1] * era5.swvl2 + d[2] * era5.swvl3) \
        / (SWCAP * d.sum())
    soilw = soilw.clip(0.0, 1.0)

    fields = {
        "sst": interp_to(sst_da, lats, lons),
        "icec": interp_to(icec_da, lats, lons).clip(0.0, 1.0),
        "stl": interp_to(era5.stl1, lats, lons),
        "soilw_am": interp_to(soilw, lats, lons),
        "snowc": interp_to(era5.sd.clip(0.0, 10.0), lats, lons),
        "alb": interp_to(era5.fal.mean("time"), lats, lons),
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
