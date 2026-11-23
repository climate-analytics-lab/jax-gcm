r"""Post-process JCM output into AtmosphereBench / CMIP6 CMOR NetCDF files.

Reads one or more JCM netCDF files — or a JCM run directory containing
``*_monthly_YYYY-MM.nc`` and/or ``*_day*.nc`` chunk files — and emits **one
variable per file** following the AtmosphereBench submission layout and CMIP6
CMOR naming convention::

    <out>/<experiment>/<member>/          # member experiments: amip, lgm, amip-hist
        <var>_Amon_<model>_<experiment>_<member>_<grid>_<YYYYMM>-<YYYYMM>.nc
        <var>_day_<model>_<experiment>_<member>_<grid>_<YYYYMMDD>-<YYYYMMDD>.nc
        sftlf_fx_<model>_<experiment>_<member>_<grid>.nc
        areacella_fx_<model>_<experiment>_<member>_<grid>.nc

    <out>/<experiment>/                   # single-member experiments (amip-4xCO2, amip-p4K, ...)
        <var>_Amon_<model>_<experiment>_<member>_<grid>_<YYYYMM>-<YYYYMM>.nc
        ...

Pass ``--flat`` to write directly into ``<out>`` (useful for ``atmospherebench dev``).

Examples
--------
Convert a completed JCM run directory (auto-detecting monthly and daily files)::

    python tools/atmospherebench_cmor.py /path/to/jcm-run \
        --model JCM-ECHAM-1M --experiment amip --member r1i1p1f1 \
        --out submission/

Convert explicit monthly files only::

    python tools/atmospherebench_cmor.py run_monthly_1979-*.nc \
        --model JCM-ECHAM-1M --experiment amip --freq mon --out submission/

"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections.abc import Callable, Sequence

import numpy as np
import xarray as xr

# Allow running directly as ``python tools/atmospherebench_cmor.py`` without
# requiring ``PYTHONPATH=.``.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jcm.analysis import area_weights, column_burden, to_pressure_levels  # noqa: E402

#: Earth mean radius [m] used for exact spherical ``areacella`` grid areas.
EARTH_RADIUS_M = 6.37122e6

#: Standard gravity [m s-2] for converting geopotential [m2 s-2] to height [m].
STANDARD_GRAVITY = 9.80665

#: Experiments whose AtmosphereBench layout places files under ``<exp>/<member>/``.
MEMBER_EXPERIMENTS = frozenset({"amip", "lgm", "amip-hist"})

#: Standard CMIP6 19-level pressure grid [hPa] for monthly 3D fields (covers
#: 850/700/200 hPa and the full tropospheric/stratospheric column for MSU).
DEFAULT_MONTHLY_PLEV_HPA: tuple[float, ...] = (
    1000.0, 925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0,
    200.0, 150.0, 100.0, 70.0, 50.0, 30.0, 20.0, 10.0, 5.0, 1.0,
)

#: Compact pressure levels [hPa] for daily 3D fields (geostrophic balance at
#: 850 hPa and MJO diagnostics at 850/200 hPa).
DEFAULT_DAILY_PLEV_HPA: tuple[float, ...] = (850.0, 200.0)

#: Default daily variable set requested by AtmosphereBench.
DEFAULT_DAILY_VARS: tuple[str, ...] = (
    "pr", "tas", "tasmax", "tasmin", "rlut", "ua", "zg",
)

#: 3D CMOR variables that live on ``plev`` when exported.
PLEV_VARS = frozenset({"ta", "hus", "hur", "ua", "va", "zg"})

#: Source variable(s) needed before calling ``to_pressure_levels`` for each 3D CMOR var.
_PLEV_SOURCE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "ta": ("temperature",),
    "hus": ("specific_humidity",),
    "hur": ("relative_humidity",),
    "ua": ("u_wind",),
    "va": ("v_wind",),
    "zg": ("height_full", "geopotential"),
}

_VARIANT_RE = re.compile(r"^r\d+i\d+p\d+f\d+$")
_MONTHLY_FILE_RE = re.compile(r"_monthly_\d{4}-\d{2}\.nc$")
_CHUNK_FILE_RE = re.compile(r"_day(\d+(?:\.\d+)?)\.nc$")

#: CF metadata for every CMOR variable this converter can emit:
#: ``cmor_name -> (units, standard_name, long_name)``.
CMOR_META: dict[str, tuple[str, str, str]] = {
    "tas": ("K", "air_temperature", "Near-Surface Air Temperature"),
    "tasmax": ("K", "air_temperature", "Daily Maximum Near-Surface Air Temperature"),
    "tasmin": ("K", "air_temperature", "Daily Minimum Near-Surface Air Temperature"),
    "ts": ("K", "surface_temperature", "Surface Temperature"),
    "pr": ("kg m-2 s-1", "precipitation_flux", "Precipitation"),
    "evspsbl": ("kg m-2 s-1", "water_evapotranspiration_flux", "Evaporation Including Sublimation and Transpiration"),
    "hfls": ("W m-2", "surface_upward_latent_heat_flux", "Surface Upward Latent Heat Flux"),
    "hfss": ("W m-2", "surface_upward_sensible_heat_flux", "Surface Upward Sensible Heat Flux"),
    "rsdt": ("W m-2", "toa_incoming_shortwave_flux", "TOA Incident Shortwave Radiation"),
    "rsut": ("W m-2", "toa_outgoing_shortwave_flux", "TOA Outgoing Shortwave Radiation"),
    "rlut": ("W m-2", "toa_outgoing_longwave_flux", "TOA Outgoing Longwave Radiation"),
    "rsutcs": ("W m-2", "toa_outgoing_shortwave_flux_assuming_clear_sky", "TOA Outgoing Clear-Sky Shortwave Radiation"),
    "rlutcs": ("W m-2", "toa_outgoing_longwave_flux_assuming_clear_sky", "TOA Outgoing Clear-Sky Longwave Radiation"),
    "rsds": ("W m-2", "surface_downwelling_shortwave_flux_in_air", "Surface Downwelling Shortwave Radiation"),
    "rsus": ("W m-2", "surface_upwelling_shortwave_flux_in_air", "Surface Upwelling Shortwave Radiation"),
    "rlds": ("W m-2", "surface_downwelling_longwave_flux_in_air", "Surface Downwelling Longwave Radiation"),
    "rlus": ("W m-2", "surface_upwelling_longwave_flux_in_air", "Surface Upwelling Longwave Radiation"),
    "clt": ("%", "cloud_area_fraction", "Total Cloud Cover Percentage"),
    "prw": ("kg m-2", "atmosphere_mass_content_of_water_vapor", "Water Vapor Path"),
    "ps": ("Pa", "surface_air_pressure", "Surface Air Pressure"),
    "psl": ("Pa", "air_pressure_at_mean_sea_level", "Sea Level Pressure"),
    "huss": ("1", "specific_humidity", "Near-Surface Specific Humidity"),
    "uas": ("m s-1", "eastward_wind", "Eastward Near-Surface Wind"),
    "vas": ("m s-1", "northward_wind", "Northward Near-Surface Wind"),
    "ta": ("K", "air_temperature", "Air Temperature"),
    "hus": ("1", "specific_humidity", "Specific Humidity"),
    "hur": ("%", "relative_humidity", "Relative Humidity"),
    "ua": ("m s-1", "eastward_wind", "Eastward Wind"),
    "va": ("m s-1", "northward_wind", "Northward Wind"),
    "zg": ("m", "geopotential_height", "Geopotential Height"),
    "sftlf": ("%", "land_area_fraction", "Percentage of the Grid Cell Occupied by Land (Including Lakes)"),
    "areacella": ("m2", "cell_area", "Grid-Cell Area for Atmospheric Grid Variables"),
}

#: Direct JCM -> CMOR mappings with fallback priority (first present key wins):
#: ``cmor_name -> ((jcm_name, scale), ...)``.
_DIRECT_SOURCES: dict[str, tuple[tuple[str, float], ...]] = {
    "ts": (("surface.surface_temperature", 1.0), ("surface.skin_temperature", 1.0)),
    "evspsbl": (
        ("surface.evaporation", 1.0),
        ("surface_exchange.evaporation", 1.0),
        ("vertical_diffusion.surface_evaporation", 1.0),
    ),
    "hfls": (("surface.latent_heat_flux", 1.0), ("surface_exchange.latent_heat_flux", 1.0)),
    "hfss": (
        ("surface.sensible_heat_flux_dry", 1.0),
        ("surface.sensible_heat_flux", 1.0),
        ("surface_exchange.sensible_heat_flux", 1.0),
    ),
    "rsdt": (("radiation.toa_sw_down", 1.0),),
    "rsut": (("radiation.toa_sw_up", 1.0),),
    "rlut": (("radiation.toa_lw_up", 1.0), ("longwave_rad.ftop", 1.0)),
    "rsutcs": (("radiation.toa_sw_up_clear", 1.0), ("clouds.toa_sw_up_clear", 1.0)),
    "rlutcs": (("radiation.toa_lw_up_clear", 1.0), ("clouds.toa_lw_up_clear", 1.0)),
    "rsds": (("radiation.surface_sw_down", 1.0), ("surface_exchange.downward_shortwave_flux", 1.0)),
    "rsus": (("radiation.surface_sw_up", 1.0),),
    "rlds": (("radiation.surface_lw_down", 1.0), ("surface_exchange.downward_longwave_flux", 1.0)),
    "rlus": (("radiation.surface_lw_up", 1.0),),
    "clt": (
        ("clouds.total_cloud_cover", 100.0),
        ("radiation.total_cloud_cover", 100.0),
        ("shortwave_rad.cloudc", 100.0),
    ),
    "ps": (("surface_pressure", 1.0), ("surface_exchange.surface_pressure", 1.0)),
    "psl": (("aerocom_psl", 1.0),),
    "huss": (("surface.humidity_2m", 1.0),),
    "uas": (("vertical_diffusion.wind_10m_u", 1.0), ("aerocom_uas", 1.0)),
    "vas": (("vertical_diffusion.wind_10m_v", 1.0), ("aerocom_vas", 1.0)),
    "tasmax": (("tasmax", 1.0),),
    "tasmin": (("tasmin", 1.0),),
}


def _extract_tas(ds: xr.Dataset) -> xr.DataArray | None:
    for key in ("surface.temperature_2m", "aerocom_tas"):
        if key in ds.data_vars:
            return ds[key]
    if "temperature" in ds.data_vars and "level" in ds["temperature"].dims:
        return ds["temperature"].isel(level=0, drop=True)
    return None


def _extract_pr(ds: xr.Dataset) -> xr.DataArray | None:
    if "surface_exchange.precipitation" in ds.data_vars:
        return ds["surface_exchange.precipitation"]
    echam_parts = [
        ds[k]
        for k in ("clouds.precip_rain", "clouds.precip_snow", "convection.precip_conv")
        if k in ds.data_vars
    ]
    if echam_parts:
        out = echam_parts[0]
        for part in echam_parts[1:]:
            out = out + part
        return out
    speedy_parts = [
        ds[k]
        for k in ("condensation.precls", "convection.precnv")
        if k in ds.data_vars
    ]
    if speedy_parts:
        out = speedy_parts[0]
        for part in speedy_parts[1:]:
            out = out + part
        # SPEEDY stores precipitation in g m-2 s-1 (= 1e-3 kg m-2 s-1).
        return out * 1e-3
    return None


def _extract_prw(ds: xr.Dataset) -> xr.DataArray | None:
    if "aerocom_prw" in ds.data_vars:
        return ds["aerocom_prw"]
    if "specific_humidity" not in ds.data_vars or "level" not in ds["specific_humidity"].dims:
        return None
    if "pressure_thickness" not in ds.data_vars and "pressure_half" not in ds.data_vars:
        return None
    return column_burden(ds, "specific_humidity")


def _extract_sftlf(ds: xr.Dataset) -> xr.DataArray | None:
    for key in ("surface_exchange.tile_fraction.2", "vertical_diffusion.surface_fraction.2"):
        if key in ds.data_vars:
            da = ds[key]
            if "time" in da.dims:
                da = da.isel(time=0, drop=True)
            return (da * 100.0).clip(0.0, 100.0)
    return None


def _compute_areacella(ds: xr.Dataset) -> xr.DataArray | None:
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return None
    lat = ds.coords["lat"]
    lon = ds.coords["lon"]
    w = area_weights(lat)
    w_norm = w / w.sum()
    total_area_m2 = 4.0 * np.pi * EARTH_RADIUS_M**2
    cell_area_by_lat = w_norm.values * (total_area_m2 / lon.size)
    area_2d = np.broadcast_to(cell_area_by_lat[:, None], (lat.size, lon.size)).astype(np.float64)
    return xr.DataArray(area_2d, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


_DERIVED_EXTRACTORS: dict[str, Callable[[xr.Dataset], xr.DataArray | None]] = {
    "tas": _extract_tas,
    "pr": _extract_pr,
    "prw": _extract_prw,
}


def _check_token(flag: str, value: str) -> str:
    if not value or "_" in value or "/" in value:
        raise ValueError(
            f"{flag}={value!r}: CMOR filename tokens must be non-empty and "
            "must not contain '_' or '/' (use '-' instead)."
        )
    return value


def _check_member(member: str) -> str:
    if not _VARIANT_RE.match(member):
        raise ValueError(
            f"--member={member!r}: expected CMIP variant label like 'r1i1p1f1'."
        )
    return member


def _date_ymd(val) -> tuple[int, int, int]:
    """Extract ``(year, month, day)`` from a numpy datetime64, Timestamp, or cftime scalar."""
    if hasattr(val, "year") and hasattr(val, "month") and hasattr(val, "day"):
        return int(val.year), int(val.month), int(val.day)
    if isinstance(val, np.datetime64):
        s = str(np.datetime_as_string(val, unit="D"))
        y, m, d = s.split("-")
        return int(y), int(m), int(d)
    raise TypeError(f"Unsupported time coordinate value: {val!r} ({type(val).__name__})")


def _time_stamp(ds: xr.Dataset, table: str) -> str:
    """Format ``<start>-<end>`` CMOR date token from ``ds['time']``."""
    if "time" not in ds.coords or ds.sizes.get("time", 0) == 0:
        raise ValueError("Dataset has no non-empty 'time' coordinate.")
    t0 = ds.coords["time"].values[0]
    t1 = ds.coords["time"].values[-1]
    y0, m0, d0 = _date_ymd(t0)
    y1, m1, d1 = _date_ymd(t1)
    if table == "Amon":
        return f"{y0:04d}{m0:02d}-{y1:04d}{m1:02d}"
    return f"{y0:04d}{m0:02d}{d0:02d}-{y1:04d}{m1:02d}{d1:02d}"


def cmor_filename(
    var: str,
    table: str,
    model: str,
    experiment: str,
    member: str,
    grid_label: str,
    date_stamp: str | None = None,
) -> str:
    """Return the CMIP6 / AtmosphereBench CMOR filename."""
    if table == "fx" or not date_stamp:
        return f"{var}_{table}_{model}_{experiment}_{member}_{grid_label}.nc"
    return f"{var}_{table}_{model}_{experiment}_{member}_{grid_label}_{date_stamp}.nc"


def _standardize_da(da: xr.DataArray, cmor: str) -> xr.DataArray:
    """Transpose to CMOR dim order, sort lat ascending, drop extra coords, set CF attrs."""
    # Drop JCM auxiliary coordinates that are not dimensions of this variable.
    extra_coords = [c for c in da.coords if c not in da.dims]
    if extra_coords:
        da = da.drop_vars(extra_coords)

    ordered_dims = [d for d in ("time", "plev", "lat", "lon") if d in da.dims]
    remaining = [d for d in da.dims if d not in ordered_dims]
    da = da.transpose(*ordered_dims, *remaining)

    if "lat" in da.dims and da.sizes["lat"] > 1:
        lat_vals = da.coords["lat"].values
        if lat_vals[0] > lat_vals[-1]:
            da = da.isel(lat=slice(None, None, -1))

    units, std_name, long_name = CMOR_META[cmor]
    da = da.rename(cmor)
    da.attrs = {
        "units": units,
        "standard_name": std_name,
        "long_name": long_name,
    }
    return da


def _set_coord_cf_attrs(out: xr.Dataset) -> None:
    """Ensure every CMOR coordinate carries the CF attributes AtmosphereBench expects."""
    if "time" in out.coords:
        out["time"].attrs.setdefault("standard_name", "time")
        out["time"].attrs.setdefault("long_name", "time")
        out["time"].attrs["axis"] = "T"
        if "time_bounds" in out.data_vars:
            out["time"].attrs["bounds"] = "time_bounds"
    if "lat" in out.coords:
        out["lat"].attrs["standard_name"] = "latitude"
        out["lat"].attrs["units"] = "degrees_north"
        out["lat"].attrs["axis"] = "Y"
        out["lat"].attrs.setdefault("long_name", "latitude")
    if "lon" in out.coords:
        out["lon"].attrs["standard_name"] = "longitude"
        out["lon"].attrs["units"] = "degrees_east"
        out["lon"].attrs["axis"] = "X"
        out["lon"].attrs.setdefault("long_name", "longitude")
    if "plev" in out.coords:
        out["plev"].attrs["standard_name"] = "air_pressure"
        out["plev"].attrs["units"] = "Pa"
        out["plev"].attrs["positive"] = "down"
        out["plev"].attrs["axis"] = "Z"
        out["plev"].attrs.setdefault("long_name", "pressure")


def _extract_plev_vars(
    ds: xr.Dataset,
    requested_plev_vars: Sequence[str],
    plevs_hpa: Sequence[float],
) -> dict[str, xr.DataArray]:
    """Extract and pressure-interpolate 3D variables onto ``plevs_hpa`` [hPa]."""
    if not requested_plev_vars:
        return {}

    # Select only the needed 3D source variables before calling to_pressure_levels
    # so we avoid interpolating dozens of internal diagnostics.
    chosen_src: dict[str, tuple[str, float]] = {}
    for cmor in requested_plev_vars:
        for src in _PLEV_SOURCE_CANDIDATES[cmor]:
            if src in ds.data_vars:
                scale = (
                    100.0
                    if cmor == "hur"
                    else (1.0 / STANDARD_GRAVITY if src == "geopotential" else 1.0)
                )
                chosen_src[cmor] = (src, scale)
                break
    if not chosen_src:
        return {}

    needs_interp = any("level" in ds[src].dims for src, _ in chosen_src.values())
    if needs_interp:
        if "pressure_full" not in ds.data_vars:
            return {}
        sub_vars = {src for src, _ in chosen_src.values()} | {"pressure_full"}
        interp_ds = to_pressure_levels(ds[sorted(sub_vars)], list(plevs_hpa))
    else:
        interp_ds = ds

    out: dict[str, xr.DataArray] = {}
    for cmor, (src, scale) in chosen_src.items():
        if src not in interp_ds.data_vars or "plev" not in interp_ds[src].dims:
            continue
        da = interp_ds[src]
        if scale != 1.0:
            da = da * scale
        out[cmor] = da
    return out


def _write_single_var(
    da: xr.DataArray,
    *,
    cmor: str,
    table: str,
    model: str,
    experiment: str,
    member: str,
    grid_label: str,
    frequency: str,
    outdir: pathlib.Path,
    date_stamp: str | None = None,
    time_bounds: xr.DataArray | None = None,
    dry_run: bool = False,
) -> str:
    """Write one standardized CMOR variable NetCDF file and return its filename."""
    std_da = _standardize_da(da, cmor)
    fname = cmor_filename(cmor, table, model, experiment, member, grid_label, date_stamp)
    if dry_run:
        return fname

    data_vars: dict[str, xr.DataArray] = {cmor: std_da}
    if time_bounds is not None and "time" in std_da.dims:
        tb = time_bounds
        extra = [c for c in tb.coords if c not in tb.dims]
        if extra:
            tb = tb.drop_vars(extra)
        data_vars["time_bounds"] = tb

    out = xr.Dataset(data_vars)
    _set_coord_cf_attrs(out)
    out.attrs = {
        "Conventions": "CF-1.8",
        "model_id": model,
        "source_id": model,
        "experiment_id": experiment,
        "variant_label": member,
        "table_id": table,
        "frequency": frequency,
        "grid_label": grid_label,
        "comment": "Produced by tools/atmospherebench_cmor.py from JCM output.",
    }
    encoding: dict[str, dict] = {cmor: {"zlib": True, "complevel": 4}}
    if "time" in out.coords:
        encoding["time"] = {"units": "days since 1850-01-01", "dtype": "float64"}
        if "time_bounds" in out.data_vars:
            encoding["time_bounds"] = {"units": "days since 1850-01-01", "dtype": "float64"}
    out.to_netcdf(outdir / fname, encoding=encoding)
    return fname


def convert_dataset(
    ds: xr.Dataset,
    *,
    model: str,
    experiment: str,
    member: str = "r1i1p1f1",
    table: str = "Amon",
    grid_label: str = "gn",
    outdir: pathlib.Path,
    variables: Sequence[str] | None = None,
    plevs_hpa: Sequence[float] | None = None,
    write_fx: bool = True,
    dry_run: bool = False,
) -> list[str]:
    """Convert one JCM Dataset into AtmosphereBench CMOR files; return written filenames."""
    _check_token("--model", model)
    _check_token("--experiment", experiment)
    _check_token("--grid-label", grid_label)
    _check_member(member)
    if table not in ("Amon", "day"):
        raise ValueError(f"Unsupported CMOR table {table!r}; expected 'Amon' or 'day'.")

    frequency = "mon" if table == "Amon" else "day"
    if variables is None:
        if table == "day":
            target_vars = list(DEFAULT_DAILY_VARS)
        else:
            target_vars = [v for v in CMOR_META if v not in ("sftlf", "areacella", "tasmax", "tasmin")]
    else:
        unknown = [v for v in variables if v not in CMOR_META]
        if unknown:
            raise ValueError(f"Unknown CMOR variable(s): {', '.join(unknown)}")
        target_vars = list(variables)

    if plevs_hpa is None:
        plevs_hpa = DEFAULT_DAILY_PLEV_HPA if table == "day" else DEFAULT_MONTHLY_PLEV_HPA

    outdir.mkdir(parents=True, exist_ok=True)
    date_stamp = _time_stamp(ds, table)
    time_bounds = ds["time_bounds"] if "time_bounds" in ds.data_vars else None

    extracted: dict[str, xr.DataArray] = {}
    plev_requested: list[str] = []
    for cmor in target_vars:
        if cmor in ("sftlf", "areacella"):
            continue
        if cmor in PLEV_VARS:
            plev_requested.append(cmor)
            continue
        if cmor in _DERIVED_EXTRACTORS:
            da = _DERIVED_EXTRACTORS[cmor](ds)
            if da is not None:
                extracted[cmor] = da
            continue
        for src, scale in _DIRECT_SOURCES.get(cmor, ()):
            if src in ds.data_vars:
                da = ds[src]
                extracted[cmor] = da * scale if scale != 1.0 else da
                break

    extracted.update(_extract_plev_vars(ds, plev_requested, plevs_hpa))

    written: list[str] = []
    for cmor, da in sorted(extracted.items()):
        written.append(
            _write_single_var(
                da,
                cmor=cmor,
                table=table,
                model=model,
                experiment=experiment,
                member=member,
                grid_label=grid_label,
                frequency=frequency,
                outdir=outdir,
                date_stamp=date_stamp,
                time_bounds=time_bounds,
                dry_run=dry_run,
            )
        )

    if write_fx:
        fx_vars: dict[str, xr.DataArray] = {}
        sftlf = _extract_sftlf(ds)
        if sftlf is not None:
            fx_vars["sftlf"] = sftlf
        areacella = _compute_areacella(ds)
        if areacella is not None:
            fx_vars["areacella"] = areacella
        for cmor, da in sorted(fx_vars.items()):
            written.append(
                _write_single_var(
                    da,
                    cmor=cmor,
                    table="fx",
                    model=model,
                    experiment=experiment,
                    member=member,
                    grid_label=grid_label,
                    frequency="fx",
                    outdir=outdir,
                    date_stamp=None,
                    time_bounds=None,
                    dry_run=dry_run,
                )
            )

    return written


def discover_inputs(
    paths: Sequence[pathlib.Path],
    freq: str = "auto",
) -> dict[str, list[pathlib.Path]]:
    """Resolve input files or a JCM run directory into ``{'Amon': [...], 'day': [...]}``."""
    if len(paths) == 1 and paths[0].is_dir():
        run_dir = paths[0]
        monthly = sorted(p for p in run_dir.glob("*_monthly_*.nc") if _MONTHLY_FILE_RE.search(p.name))
        chunks = sorted(
            (p for p in run_dir.glob("*_day*.nc") if _CHUNK_FILE_RE.search(p.name)),
            key=lambda p: float(_CHUNK_FILE_RE.search(p.name)[1]),  # type: ignore[index]
        )
        out: dict[str, list[pathlib.Path]] = {}
        if freq in ("auto", "both", "mon") and monthly:
            out["Amon"] = monthly
        if freq in ("auto", "both", "day") and chunks:
            out["day"] = chunks
        if not out:
            raise FileNotFoundError(
                f"No *_monthly_YYYY-MM.nc or *_day*.nc files found in {run_dir}"
            )
        return out

    files = [pathlib.Path(p) for p in paths]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Input file(s) not found: {', '.join(missing)}")
    if freq == "day":
        return {"day": files}
    if freq == "mon":
        return {"Amon": files}
    if all(_CHUNK_FILE_RE.search(p.name) for p in files):
        return {"day": files}
    return {"Amon": files}


def resolve_output_dir(
    out_root: pathlib.Path,
    experiment: str,
    member: str,
    *,
    flat: bool = False,
) -> pathlib.Path:
    """Return the AtmosphereBench submission directory for ``(experiment, member)``."""
    if flat:
        return out_root
    if experiment in MEMBER_EXPERIMENTS:
        return out_root / experiment / member
    return out_root / experiment


def _open_inputs(files: Sequence[pathlib.Path]) -> xr.Dataset:
    if len(files) == 1:
        return xr.open_dataset(files[0])
    return xr.open_mfdataset(
        [str(p) for p in files],
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        type=pathlib.Path,
        help="JCM run directory or one or more JCM output NetCDF files",
    )
    ap.add_argument("--model", required=True, help="Model name for CMOR filenames (no underscores)")
    ap.add_argument(
        "--experiment",
        default="amip",
        help="AtmosphereBench experiment name (e.g. amip, amip-4xCO2, amip-p4K; default: %(default)s)",
    )
    ap.add_argument(
        "--member",
        default="r1i1p1f1",
        help="CMIP variant label (default: %(default)s)",
    )
    ap.add_argument(
        "--grid-label",
        default="gn",
        help="CMOR grid label (default: %(default)s)",
    )
    ap.add_argument(
        "--freq",
        default="auto",
        choices=("auto", "both", "mon", "day"),
        help="Which frequencies to export (default: %(default)s)",
    )
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("atmospherebench_submit"),
        help="Submission root directory (default: %(default)s)",
    )
    ap.add_argument(
        "--flat",
        action="store_true",
        help="Write NetCDF files directly into --out instead of <out>/<experiment>[/<member>]/",
    )
    ap.add_argument(
        "--variables",
        help="Optional comma-separated subset of CMOR variables to export",
    )
    ap.add_argument(
        "--plevs",
        help="Comma-separated pressure levels in hPa for monthly 3D fields (default: 19 CMIP6 levels)",
    )
    ap.add_argument(
        "--daily-plevs",
        help="Comma-separated pressure levels in hPa for daily 3D fields (default: 850,200)",
    )
    ap.add_argument(
        "--no-fx",
        action="store_true",
        help="Skip writing fixed grid fields (sftlf, areacella)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print filenames that would be written without writing NetCDF files",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    variables = (
        [v.strip() for v in args.variables.split(",") if v.strip()]
        if args.variables
        else None
    )
    monthly_plevs = (
        [float(x) for x in args.plevs.split(",") if x.strip()]
        if args.plevs
        else None
    )
    daily_plevs = (
        [float(x) for x in args.daily_plevs.split(",") if x.strip()]
        if args.daily_plevs
        else None
    )

    grouped = discover_inputs(args.inputs, freq=args.freq)
    outdir = resolve_output_dir(args.out, args.experiment, args.member, flat=args.flat)

    all_written: list[str] = []
    fx_written = False
    for table in ("Amon", "day"):
        if table not in grouped:
            continue
        with _open_inputs(grouped[table]) as ds:
            written = convert_dataset(
                ds,
                model=args.model,
                experiment=args.experiment,
                member=args.member,
                table=table,
                grid_label=args.grid_label,
                outdir=outdir,
                variables=variables,
                plevs_hpa=monthly_plevs if table == "Amon" else daily_plevs,
                write_fx=(not args.no_fx and not fx_written),
                dry_run=args.dry_run,
            )
            fx_written = fx_written or (not args.no_fx)
            all_written.extend(written)

    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {len(all_written)} file(s) to {outdir}")
    for fname in all_written:
        print(" ", fname)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
