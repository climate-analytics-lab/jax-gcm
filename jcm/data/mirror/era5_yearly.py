"""Yearly transient all-ERA5 boundary-condition bundles (issue #629).

``forcing_era5/<year>.nc`` prescribes every surface field from one
reanalysis: SST, sea ice and the land surface share ERA5's land-sea
mask, coastlines and sea-ice conventions, unlike ``forcing_amip`` which
combines PCMDI-AMIP ocean fields with a repeated ERA5 land climatology.
ERA5 also extends past PCMDI's 2022 endpoint (AIMIP runs to end-2024).
``forcing_amip`` remains the product for protocol runs that mandate
PCMDI SSTs.

Time axis: 12 **month-start** boundary values per year, to be loaded
with ``align=by_date_interp``:

* SST/ice — rectangular average of 6-hourly analyses over the window
  between the midpoints of the two adjacent months: the AIMIP
  construction (doi:10.5281/zenodo.16782372). This is *not* the
  mean-preserving Taylor et al. (2000) ``tosbcs`` construction — linear
  interpolation between these values damps monthly-mean amplitude by
  ~0.16 K RMS — but it makes calibration runs and an AIMIP submission
  share identically constructed boundary conditions.
* Land — monthly means blended onto month starts as
  ``0.5·(M_prev + M_cur)``. This is the same centred-window family, so
  land shares the ocean time axis without the half-month phase shift
  that stamping a monthly mean at the month start would introduce.

Two fields deliberately stay climatological (2005–2014): the ice-sheet
mask that zeroes ``snowc`` and the snow-free background albedo ``alb``
(per-year versions of both drift with how snowy the year was — see
``bundles.translate_land`` and the ``fal`` minimum in ``bundles``).

Land monthly means come from the pre-computed RDA product (d633001,
1979–2022) where it exists, and are reduced from the 6-hourly analyses
(d633000, 1940–present) outside that range — so any year from
``ERA5_FIRST_YEAR`` is buildable from data already on Glade.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import xarray as xr

from jcm.data.mirror.amip_yearly import _TIME_ENC, ghg_ppmv
from jcm.data.mirror.bundles import _to_lonlat, translate_land
from jcm.data.mirror.era5_land import FIELDS, RDA_MODA, _open_year
from jcm.data.regridding import fill_nearest, interp_to

RDA_AN_SFC = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc"

#: 6-hourly analysis fields for the ocean half of the bundle.
AN_CODES = {"sstk": "128_034_sstk", "ci": "128_031_ci"}

#: ERA5 starts 1940-01; a year build needs the previous December.
ERA5_FIRST_YEAR = 1941

#: Last year in the CR-CMIP GHG source; later years are extrapolated.
CR_LAST_YEAR = 2022


def _open_an_month(code: str, year: int, month: int,
                   every: int = 6) -> xr.DataArray:
    """One month of 6-hourly (00/06/12/18Z) 0.25° analysis, loaded."""
    (path,) = glob.glob(
        f"{RDA_AN_SFC}/{year}{month:02d}/e5.oper.an.sfc.{code}.ll025sc."
        f"{year}{month:02d}0100_*.nc")
    ds = xr.open_dataset(path)
    (name,) = [v for v in ds.data_vars if ds[v].ndim == 3]
    return ds[name].isel(time=slice(0, None, every)).load()


def _month_midpoint(year: int, month: int) -> np.datetime64:
    """Exact temporal midpoint of a calendar month (hour precision)."""
    start = np.datetime64(f"{year:04d}-{month:02d}-01")
    y2, m2 = (year + 1, 1) if month == 12 else (year, month + 1)
    end = np.datetime64(f"{y2:04d}-{m2:02d}-01")
    return start + (end - start).astype("timedelta64[h]") // 2


def _month_starts(year: int) -> np.ndarray:
    return np.array([np.datetime64(f"{year:04d}-{m:02d}-01")
                     for m in range(1, 13)])


def build_sstice_year(year: int) -> xr.Dataset:
    """12 month-start SST/ice boundary values (AIMIP centred window).

    The value stamped at the start of month ``m`` is the mean of the
    6-hourly analyses over ``[mid(m-1), mid(m))``, so each month is
    streamed once, split at its midpoint, and the second half combined
    with the next month's first half. Land cells are NaN throughout
    (ERA5's ocean mask is static) — filled at bundle assembly.
    """
    if year < ERA5_FIRST_YEAR:
        raise ValueError(f"ERA5 6-hourly analyses start 1940 — cannot "
                         f"build {year} (first buildable year is "
                         f"{ERA5_FIRST_YEAR})")
    months = [(year - 1, 12)] + [(year, m) for m in range(1, 13)]
    results: dict[str, list[np.ndarray]] = {v: [] for v in AN_CODES}
    prev_half: dict[str, tuple[np.ndarray, int]] = {}
    template = None
    for y, m in months:
        mid = _month_midpoint(y, m)
        for var, code in AN_CODES.items():
            da = _open_an_month(code, y, m)
            first = da.time.values < mid
            vals = da.values.astype(np.float64)
            if var in prev_half:
                psum, pn = prev_half[var]
                results[var].append(
                    (psum + vals[first].sum(axis=0)) / (pn + first.sum()))
            prev_half[var] = (vals[~first].sum(axis=0), int((~first).sum()))
            template = da
    ds = xr.Dataset(
        {var: (("time", "latitude", "longitude"),
               np.stack(results[var]).astype(np.float32))
         for var in AN_CODES},
        coords={"time": _month_starts(year),
                "latitude": template.latitude,
                "longitude": template.longitude})
    ds.attrs = {
        "source": "ERA5 6-hourly surface analyses (NCAR RDA d633000)",
        "construction": ("month-start boundary values: rectangular mean "
                         "of 6-hourly (00/06/12/18Z) analyses between "
                         "adjacent month midpoints (AIMIP construction)"),
        "year": year,
    }
    return ds


def _land_month_mean(code: str, year: int, month: int) -> xr.DataArray:
    """One (1, lat, lon) land monthly mean, stamped at the month start.

    The RDA monthly-mean product where the year has one; otherwise
    reduced from the 6-hourly analyses (evenly sampled diurnal phase, so
    the monthly mean agrees with the moda product to ~0.01 K).
    """
    if Path(f"{RDA_MODA}/{year}").is_dir():
        return _open_year(code, year).isel(time=[month - 1]).load()
    da = _open_an_month(code, year, month).mean("time")
    return da.expand_dims(
        time=[np.datetime64(f"{year:04d}-{month:02d}-01")])


def build_land_year(year: int) -> xr.Dataset:
    """13 land monthly means: Dec(year-1), then Jan–Dec of ``year``.

    Thirteen so the assembly can blend adjacent pairs onto the 12
    month-start boundary values. Carries every ``era5_land`` field
    including ``skt`` (evaluation only — ``stl`` is what jcm prescribes,
    since ``SpeedySurfaceFlux`` adds its own diagnostic skin response).
    """
    if year < ERA5_FIRST_YEAR:
        raise ValueError(f"first buildable year is {ERA5_FIRST_YEAR}")
    months = [(year - 1, 12)] + [(year, m) for m in range(1, 13)]
    out = {}
    for var, code in FIELDS.items():
        out[var] = xr.concat([_land_month_mean(code, y, m)
                              for y, m in months], dim="time")
    ds = xr.Dataset(out)
    moda = {y for y, _ in months if Path(f"{RDA_MODA}/{y}").is_dir()}
    ds.attrs = {
        "source": ("ERA5 monthly means (NCAR RDA d633001)"
                   + ("" if moda == {year - 1, year} else
                      "; missing years reduced from 6-hourly analyses "
                      "(d633000)")),
        "year": year,
    }
    return ds


def _blend_to_month_starts(da: xr.DataArray,
                           times: np.ndarray) -> xr.DataArray:
    """13 monthly means -> 12 month-start values, 0.5·(prev + cur)."""
    v = 0.5 * (da.isel(time=slice(None, -1)).values
               + da.isel(time=slice(1, None)).values)
    coords = {d: da[d] for d in da.dims if d != "time"}
    return xr.DataArray(v, dims=da.dims, coords={**coords, "time": times})


def _extrapolate_linear(years: np.ndarray, values: np.ndarray,
                        target: int) -> float:
    """Least-squares linear continuation of an annual series."""
    slope, intercept = np.polyfit(years, values, 1)
    return float(slope * target + intercept)


def ghg_ppmv_extended(gas: str, year: int) -> tuple[float, str]:
    """CR-CMIP value in coverage; last-decade linear trend beyond it.

    The extrapolation error over the 2 years AIMIP needs past 2022 is
    well under 1 ppm CO2 (the growth rate is nearly linear on that
    scale); the provenance string is stamped in the file so nobody
    mistakes it for an observed concentration.
    """
    if year <= CR_LAST_YEAR:
        return ghg_ppmv(gas, year), "CR-CMIP-1-0-0 global annual mean"
    fit_years = np.arange(CR_LAST_YEAR - 9, CR_LAST_YEAR + 1)
    vals = np.array([ghg_ppmv(gas, int(y)) for y in fit_years])
    note = (f"linear extrapolation of the CR-CMIP-1-0-0 "
            f"{fit_years[0]}-{fit_years[-1]} trend (source ends "
            f"{CR_LAST_YEAR})")
    return _extrapolate_linear(fit_years, vals, year), note


def build_forcing_year(clim_path: str, sstice_path: str, year: int,
                       lats, lons, out_path: str,
                       land: xr.Dataset | None = None) -> None:
    """One year of all-ERA5 boundary forcing on a Gaussian grid.

    ``clim_path`` is the 2005–2014 land climatology (``era5`` stage):
    it supplies the invariants (``cvh``/``cvl``), the fixed ice-sheet
    mask and the background albedo. ``sstice_path`` is the year's
    ``build_sstice_year`` output; ``land`` the year's
    ``build_land_year`` output (built on the fly when omitted).
    """
    clim = xr.open_dataset(clim_path)
    sstice = xr.open_dataset(sstice_path)
    if land is None:
        land = build_land_year(year)
    land = xr.merge([land, clim[["cvh", "cvl"]]])
    times = sstice.time.values

    # Ocean fields: nearest-ocean fill under the (static) NaN land mask
    # before the bilinear regrid; ice concentration is simply 0 on land.
    sst_filled = fill_nearest(sstice.sstk.values.astype(np.float64),
                              sstice.latitude.values,
                              sstice.longitude.values)
    sst_da = xr.DataArray(sst_filled, dims=("time", "latitude", "longitude"),
                          coords={"time": times,
                                  "latitude": sstice.latitude,
                                  "longitude": sstice.longitude})
    icec_da = sstice.ci.fillna(0.0)

    # Land: translate the 13 monthly means with the climatological
    # ice-sheet mask, then blend onto the month-start axis.
    tl = translate_land(land, permanent_snow=clim.sd.min("month") >= 0.1)
    fields = {
        "sst": interp_to(sst_da, lats, lons),
        "icec": interp_to(icec_da, lats, lons).clip(0.0, 1.0),
        "stl": interp_to(_blend_to_month_starts(tl["stl"], times),
                         lats, lons),
        "soilw_am": interp_to(
            _blend_to_month_starts(tl["soilw_am"], times),
            lats, lons).clip(0.0, 1.0),
        "snowc": interp_to(
            _blend_to_month_starts(tl["snowc"], times),
            lats, lons).clip(0.0, 1.0),
        # Background albedo stays climatological, like the ice-sheet
        # mask: a per-year minimum drifts with how snowy the year was.
        "alb": interp_to(clim.fal.min("month"), lats, lons),
    }
    ds = xr.Dataset(coords={"lat": lats, "lon": lons, "time": times})
    for name, da in fields.items():
        ds[name] = _to_lonlat(da)
    for gas in ("co2", "ch4", "n2o"):
        value, note = ghg_ppmv_extended(gas, year)
        ds[gas] = (("time",), np.full(12, value, dtype=np.float32),
                   {"units": "ppmv", "source": note})
    ds.attrs = {
        "title": f"jax-gcm transient ERA5 forcing, year {year}",
        "source": ("ERA5 (NCAR RDA): SST/ice from 6-hourly sstk/ci "
                   "(d633000), land from monthly means (d633001; "
                   "6-hourly reduction outside 1979-2022); GHG: "
                   "CR-CMIP-1-0-0"),
        "construction": (
            "month-start boundary values, load with align=by_date_interp. "
            "SST/ice: mean of 6-hourly analyses between adjacent month "
            "midpoints (AIMIP construction, cf. "
            "doi:10.5281/zenodo.16782372 — NOT the mean-preserving PCMDI "
            "tosbcs construction). Land: 0.5*(prev+cur) monthly means. "
            "snowc ice-sheet mask and background albedo are "
            "climatological 2005-2014."),
        "year": year,
    }
    ds.to_netcdf(out_path, encoding=_TIME_ENC)
    print("wrote", out_path, flush=True)
