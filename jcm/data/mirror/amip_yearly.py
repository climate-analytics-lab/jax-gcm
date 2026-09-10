"""Yearly transient AMIP bundles (issue #610).

One file per calendar year so a run downloads only the years it covers
and a new year appends without rewriting history:

* ``bundles/<grid>/forcing_amip/<year>.nc`` — PCMDI-AMIP **boundary**
  SST/sea-ice (``tosbcs``/``siconcbcs`` mid-month values, which
  reconstruct observed monthly means under the loader's
  ``by_date_interp`` linear interpolation), the ERA5 land climatology
  repeated on the year's time axis, and CR-CMIP global-annual-mean
  CO2/CH4/N2O as ``(time,)`` series (ppmv).
* ``bundles/<grid>/emissions_amip/<year>.nc`` — the year's monthly CEDS
  anthropogenic + BB4CMIP7 biomass-burning fluxes from the Tier-A zarrs.
* ``bundles/<grid>_l{47,95}/ozone_amip/<year>.nc`` — the year's monthly
  FZJ CMIP7 ozone on model hybrid levels.

Every writer pins the time encoding to one shared epoch
(``TIME_UNITS``): ``OzoneClimatology.from_file`` concatenates yearly
files on their *raw* time values and rejects mixed epochs.

Coverage: PCMDI-AMIP and FZJ ozone run 1870/1850–2022, CEDS/BB4CMIP7
1850–2023, CR GHGs 1750–2022 → full AMIP years are 1870–2022.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from jcm.data.mirror.bundles import (AMIP_ROOT, _ANTHRO_SECTORS,
                                     _EMIS_SPECIES, _to_lonlat,
                                     translate_land)
from jcm.data.regridding import (conservative_to_gaussian, fill_nearest,
                                 interp_to)

TOSBCS = (f"{AMIP_ROOT}/ocean/mon/tosbcs/gn/v20250807/"
          "tosbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_"
          "187001-202212.nc")
SICONCBCS = (f"{AMIP_ROOT}/seaIce/mon/siconcbcs/gn/v20250807/"
             "siconcbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_"
             "187001-202212.nc")

_GHG_ROOT = ("/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/"
             "CMIP7/CMIP/CR/CR-CMIP-1-0-0/atmos/yr")
_GHG_FILE = (_GHG_ROOT + "/{gas}/gm/v20250228/{gas}_input4MIPs_"
             "GHGConcentrations_CMIP_CR-CMIP-1-0-0_gm_1750-2022.nc")
#: unit -> ppmv conversion for the CR global-mean files.
_TO_PPMV = {"ppm": 1.0, "ppb": 1e-3, "ppt": 1e-6}

_FZJ_ROOT = ("/glade/campaign/cesm/cesmdata/input4MIPs_raw/input4MIPs/"
             "CMIP7/CMIP/FZJ/FZJ-CMIP-ozone-1-0/atmos/mon/vmro3/gn/"
             "v20250904")
#: (first_year, last_year) -> transient vmro3 chunk file.
_FZJ_CHUNKS = {
    (1829, 1849): "182901-184912", (1850, 1899): "185001-189912",
    (1900, 1949): "190001-194912", (1950, 1999): "195001-199912",
    (2000, 2022): "200001-202212",
}

#: Shared time-encoding epoch for every yearly file (see module docstring).
TIME_UNITS = "days since 1850-01-01"
_TIME_ENC = {"time": {"units": TIME_UNITS, "dtype": "float64"}}

AMIP_FIRST_YEAR, AMIP_LAST_YEAR = 1870, 2022


def _check_year(year: int) -> None:
    if not AMIP_FIRST_YEAR <= year <= AMIP_LAST_YEAR:
        raise ValueError(
            f"AMIP year {year} outside source coverage "
            f"[{AMIP_FIRST_YEAR}, {AMIP_LAST_YEAR}]")


def ghg_ppmv(gas: str, year: int) -> float:
    """CR-CMIP global-annual-mean concentration for ``year`` in ppmv."""
    ds = xr.open_dataset(_GHG_FILE.format(gas=gas))
    da = ds[gas].sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    if da.sizes.get("time", 0) != 1:
        raise ValueError(f"No unique {gas} entry for {year} in CR-CMIP file")
    scale = _TO_PPMV[str(da.attrs.get("units", ds[gas].attrs.get("units")))]
    return float(da.values.ravel()[0]) * scale


def build_forcing_year(era5_path: str, year: int, lats, lons,
                       out_path: str) -> None:
    """One year of AMIP boundary forcing on a Gaussian grid.

    SST/ice are the PCMDI ``*bcs`` mid-month values (12 steps at the
    source's real timestamps); the land fields repeat the ERA5 monthly
    climatology on that same axis (AMIP prescribes only SST/ice — land
    stays climatological); CO2/CH4/N2O ride along as constant-in-year
    ``(time,)`` series so ``by_date_interp`` blends year to year.
    """
    _check_year(year)
    era5 = xr.open_dataset(era5_path).rename(month="time")
    span = slice(f"{year}-01-01", f"{year}-12-31")

    tos = xr.open_dataset(TOSBCS).tosbcs.sel(time=span)
    sic = xr.open_dataset(SICONCBCS).siconcbcs.sel(time=span)
    times = tos.time.values
    if len(times) != 12:
        raise ValueError(f"tosbcs has {len(times)} steps for {year}")

    # Ocean-only field: fill land by nearest ocean neighbour before the
    # bilinear regrid (same treatment as the climatology bundles).
    sst_filled = fill_nearest(tos.values, tos.lat.values, tos.lon.values)
    sst_da = xr.DataArray(sst_filled + 273.15, dims=("time", "lat", "lon"),
                          coords={"time": times, "lat": tos.lat,
                                  "lon": tos.lon})
    icec_da = (sic / 100.0).fillna(0.0)

    # Land fields: the shared translation (bundles.translate_land),
    # re-stamped on this year's time axis (months align 1:1). The input
    # is the climatology, so its own window doubles as the fixed
    # ice-sheet-mask window.
    land = translate_land(era5, permanent_snow=era5.sd.min("time") >= 0.1)

    def _on_year_axis(da):
        return da.assign_coords(time=times)

    fields = {
        "sst": interp_to(sst_da, lats, lons),
        "icec": interp_to(icec_da, lats, lons).clip(0.0, 1.0),
        "stl": _on_year_axis(interp_to(land["stl"], lats, lons)),
        "soilw_am": _on_year_axis(
            interp_to(land["soilw_am"], lats, lons).clip(0.0, 1.0)),
        "snowc": _on_year_axis(
            interp_to(land["snowc"], lats, lons).clip(0.0, 1.0)),
        "alb": interp_to(era5.fal.min("time"), lats, lons),
    }
    ds = xr.Dataset(coords={"lat": lats, "lon": lons, "time": times})
    for name, da in fields.items():
        ds[name] = _to_lonlat(da)
    for gas in ("co2", "ch4", "n2o"):
        ds[gas] = (("time",),
                   np.full(12, ghg_ppmv(gas, year), dtype=np.float32),
                   {"units": "ppmv",
                    "source": "CR-CMIP-1-0-0 global annual mean"})
    ds.attrs = {
        "title": f"jax-gcm transient AMIP forcing, year {year}",
        "source": ("SST/ice: PCMDI-AMIP-1-1-10 tosbcs/siconcbcs "
                   "(mid-month boundary values — load with "
                   "align=by_date_interp); land: ERA5 monthly climatology "
                   "2005-2014; GHG: CR-CMIP-1-0-0"),
        "year": year,
    }
    ds.to_netcdf(out_path, encoding=_TIME_ENC)
    print("wrote", out_path, flush=True)


def build_emissions_year(ceds_zarr: str, bb_zarr: str, year: int, lats, lons,
                         out_path: str) -> None:
    """One year of monthly transient emissions on a Gaussian grid.

    Same channels as the climatology ``build_emissions_nc`` (three CEDS
    super-sectors + biomass burning per species), sliced from the
    transient Tier-A series instead of the era climatology.
    """
    _check_year(year)
    span = slice(f"{year}-01-01", f"{year}-12-31")
    ceds = xr.open_zarr(ceds_zarr)
    bb = xr.open_zarr(bb_zarr)
    times = ceds.time.sel(time=span).values
    if len(times) != 12:
        raise ValueError(f"CEDS zarr has {len(times)} steps for {year}")
    ds = xr.Dataset(coords={"lat": lats, "lon": lons, "time": times})
    for sp in _EMIS_SPECIES:
        up = sp.upper()
        channels = [(sector, ceds[f"{up}_{sector}"])
                    for sector in _ANTHRO_SECTORS]
        channels.append(("biomass_burning", bb[up]))
        for prefix, da in channels:
            da = da.sel(time=span).load()
            arr = conservative_to_gaussian(
                np.nan_to_num(da.values), da.lat.values, da.lon.values,
                lats, lons)
            ds[f"emis_{prefix}_{sp}"] = (
                ("time", "lon", "lat"), arr.transpose(0, 2, 1),
                {"units": "kg m-2 s-1"})
    ds.attrs = {
        "title": (f"jax-gcm prescribed emissions (bulk per-super-sector "
                  f"surface flux), year {year}"),
        "source": "CEDS-CMIP-2025-04-18 + DRES-CMIP-BB4CMIP7-2-0",
        "year": year,
    }
    ds.to_netcdf(out_path, encoding=_TIME_ENC)
    print("wrote", out_path, flush=True)


def load_ozone_year(year: int) -> xr.DataArray:
    """Load the year's 12 monthly ``vmro3`` fields from the FZJ chunks."""
    _check_year(year)
    for (y0, y1), tag in _FZJ_CHUNKS.items():
        if y0 <= year <= y1:
            path = (f"{_FZJ_ROOT}/vmro3_input4MIPs_ozone_CMIP_"
                    f"FZJ-CMIP-ozone-1-0_gn_{tag}.nc")
            da = xr.open_dataset(path).vmro3.sel(
                time=slice(f"{year}-01-01", f"{year}-12-31"))
            if da.sizes.get("time") != 12:
                raise ValueError(
                    f"FZJ chunk {tag} has {da.sizes.get('time')} steps "
                    f"for {year}")
            return da.transpose("time", "plev", "lat", "lon")
    raise ValueError(f"No FZJ ozone chunk covers {year}")


def regrid_ozone_year(da: xr.DataArray, lats: np.ndarray,
                      lons: np.ndarray) -> xr.Dataset:
    """Bilinear regrid keeping the year's real time axis.

    Same output contract as ``jcm.data.mirror.ozone.regrid_climatology``
    (the ``interpolate_ozone`` input format) but with datetimes instead
    of month indices, pinned to the shared epoch on write.
    """
    out = interp_to(da, lats, lons)
    plev_pa = out.plev.values * 100.0          # source file is hPa
    # The FZJ source uses a 365_day (cftime) calendar; normalise to plain
    # datetime64 mid-month stamps by calendar components (noleap dates are
    # always valid Gregorian dates) so every consumer sees one clock.
    times = np.array([np.datetime64(f"{d.year:04d}-{d.month:02d}-"
                                    f"{d.day:02d}")
                      for d in np.ravel(da.time.values)]) \
        if da.time.dtype == object else da.time.values
    ds = xr.Dataset({"O3": (("time", "plev", "lat", "lon"),
                            out.transpose("time", "plev", "lat", "lon").values,
                            {"units": "mole mole-1"})},
                    coords={"time": times, "plev": plev_pa,
                            "lat": lats, "lon": lons})
    ds.plev.attrs["units"] = "Pa"
    ds.attrs["source"] = "FZJ-CMIP-ozone-1-0 (input4MIPs CMIP7)"
    return ds
