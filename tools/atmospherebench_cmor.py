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

Shared variable mappings, CF metadata, constants, and the atomic single-variable
NetCDF writer live in :mod:`jcm.cmor_converter` (shared with
``tools/aerocom_cmor.py``).

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
import dataclasses
import datetime
import pathlib
import re
import sys
import warnings
from collections.abc import Collection, Sequence

import numpy as np
import xarray as xr

from jcm.analysis import validate_pressure_levels
from jcm.cmor_converter import (
    CMOR_META,
    FX_VARS,
    PLEV_VARS,
    cmor_variable,
    date_ymd,
    extract_direct_variable,
    extract_plev_variable,
    extract_tas,
    get_time_bounds,
    median_step_days,
    time_bounds_names,
    to_cmip_layout,
    validate_token,
    validate_variant_label,
    write_cmor_variable,
)

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

#: Default monthly variable set: every time-varying CMOR variable except the
#: daily extremes.
DEFAULT_MONTHLY_VARS: tuple[str, ...] = tuple(
    v for v in CMOR_META if v not in FX_VARS and v not in ("tasmax", "tasmin")
)

#: Default daily variable set requested by AtmosphereBench.
DEFAULT_DAILY_VARS: tuple[str, ...] = (
    "pr", "tas", "tasmax", "tasmin", "rlut", "ua", "zg",
)


@dataclasses.dataclass(frozen=True)
class _Table:
    """Per-CMOR-table defaults for the time-varying variables."""

    frequency: str
    plevs_hpa: tuple[float, ...]
    variables: tuple[str, ...]
    #: Leading digits of ``YYYYMMDD`` kept in the filename date stamp.
    stamp_digits: int


_TABLES: dict[str, _Table] = {
    "Amon": _Table("mon", DEFAULT_MONTHLY_PLEV_HPA, DEFAULT_MONTHLY_VARS, 6),
    "day": _Table("day", DEFAULT_DAILY_PLEV_HPA, DEFAULT_DAILY_VARS, 8),
}

_MONTHLY_FILE_RE = re.compile(r"^(?P<prefix>.+)_monthly_\d{4}-\d{2}\.nc$")
_CHUNK_FILE_RE = re.compile(r"^(?P<prefix>.+)_day(?P<day>\d+(?:\.\d+)?)\.nc$")


def _resample_subdaily_to_daily(ds: xr.Dataset) -> xr.Dataset:
    """Aggregate a sub-daily JCM Dataset into daily means plus ``tasmax``/``tasmin``."""
    ds_no_bounds = ds.drop_vars(time_bounds_names(ds))

    # Extract raw sub-daily near-surface temperature before averaging so we can
    # compute daily tasmax and tasmin.
    tas_raw = extract_tas(ds_no_bounds)
    daily = ds_no_bounds.resample(time="1D").mean(keep_attrs=True)
    if tas_raw is not None:
        if "tasmax" not in daily.data_vars:
            daily["tasmax"] = tas_raw.resample(time="1D").max(keep_attrs=True)
        if "tasmin" not in daily.data_vars:
            daily["tasmin"] = tas_raw.resample(time="1D").min(keep_attrs=True)

    tvals = daily.coords["time"].values
    if len(tvals) > 0:
        one_day = (
            np.timedelta64(1, "D")
            if isinstance(tvals[0], np.datetime64)
            else datetime.timedelta(days=1)
        )
        tb = np.stack([tvals, tvals + one_day], axis=-1)
        daily["time_bounds"] = xr.DataArray(tb, dims=("time", "bounds"))
        daily["time"].attrs["bounds"] = "time_bounds"
    return daily


def _match_table_cadence(ds: xr.Dataset, table: str, resample_daily: bool) -> xr.Dataset:
    """Check the cadence of ``ds`` against ``table``, resampling sub-daily input to daily."""
    step_days = median_step_days(ds, get_time_bounds(ds))
    if step_days is None:
        return ds
    if table == "Amon":
        if step_days < 25.0:
            raise ValueError(
                f"Expected monthly data for table='Amon', but median time step is "
                f"{step_days:.2f} days. Pass --freq day for daily/sub-daily inputs."
            )
        return ds
    if step_days > 1.5:
        raise ValueError(
            f"Expected daily or sub-daily data for table='day', but median "
            f"time step is {step_days:.2f} days."
        )
    if step_days >= 0.9:
        return ds
    if not resample_daily:
        raise ValueError(
            f"Input for table='day' has sub-daily cadence ({step_days * 24:.2f} h). "
            "Enable daily resampling (--resample-daily) to aggregate to daily means "
            "and tasmax/tasmin."
        )
    print(
        f"INFO tools/atmospherebench_cmor.py: resampling sub-daily input "
        f"({step_days * 24:.2f} h cadence) to daily means (plus tasmax/tasmin).",
        file=sys.stderr,
    )
    return _resample_subdaily_to_daily(ds)


def _time_stamp(
    ds: xr.Dataset,
    stamp_digits: int,
    time_bounds: xr.DataArray | None = None,
) -> str:
    """Format the ``<start>-<end>`` CMOR date token from ``time_bounds`` (or ``ds['time']``)."""
    if "time" not in ds.coords or ds.sizes.get("time", 0) == 0:
        raise ValueError("Dataset has no non-empty 'time' coordinate.")
    starts = ds.coords["time"].values if time_bounds is None else time_bounds.values[:, 0]
    return "-".join(
        "{:04d}{:02d}{:02d}".format(*date_ymd(t))[:stamp_digits]
        for t in (starts[0], starts[-1])
    )


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


def _slice_years(
    ds: xr.Dataset,
    start_year: int | None,
    end_year: int | None,
) -> xr.Dataset:
    if start_year is None and end_year is None:
        return ds
    if "time" not in ds.coords:
        raise ValueError("Cannot apply --start-year/--end-year: dataset has no 'time' coordinate.")
    years = ds["time"].dt.year
    mask = xr.ones_like(years, dtype=bool)
    if start_year is not None:
        mask = mask & (years >= start_year)
    if end_year is not None:
        mask = mask & (years <= end_year)
    sliced = ds.isel(time=np.nonzero(np.asarray(mask.values))[0])
    if sliced.sizes.get("time", 0) == 0:
        raise ValueError(
            f"Year filter [{start_year}, {end_year}] left zero timesteps."
        )
    return sliced


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
    fx_vars: Collection[str] | None = None,
    resample_daily: bool = True,
    start_year: int | None = None,
    end_year: int | None = None,
    dry_run: bool = False,
) -> tuple[dict[str, str], list[str]]:
    """Convert one JCM Dataset into AtmosphereBench CMOR files.

    ``variables`` selects the CMOR variables to export (default: the table's
    standard set). ``fx_vars`` selects the fixed grid fields to write alongside
    them: ``None`` means every fixed field, or just those named in an explicit
    ``variables``; pass an empty collection to write none.

    Returns ``(written, missing)``: the filename written for each CMOR
    variable, and the sorted names of the target variables the input could not
    supply.
    """
    validate_token("--model", model)
    validate_token("--experiment", experiment)
    validate_token("--grid-label", grid_label)
    validate_variant_label(member)
    if table not in _TABLES:
        raise ValueError(f"Unsupported CMOR table {table!r}; expected 'Amon' or 'day'.")
    spec = _TABLES[table]

    explicit_plevs = plevs_hpa is not None
    plevs = (
        validate_pressure_levels(plevs_hpa).tolist()
        if explicit_plevs
        else list(spec.plevs_hpa)
    )

    if variables is None:
        target_vars = list(spec.variables)
    else:
        unknown = [v for v in variables if v not in CMOR_META]
        if unknown:
            raise ValueError(f"Unknown CMOR variable(s): {', '.join(unknown)}")
        target_vars = list(dict.fromkeys(variables))
    if fx_vars is None:
        fx_vars = FX_VARS if variables is None else target_vars

    ds = _slice_years(ds, start_year, end_year)
    ds = _match_table_cadence(ds, table, resample_daily)
    time_bounds = get_time_bounds(ds)

    if not dry_run:
        outdir.mkdir(parents=True, exist_ok=True)
    date_stamp = _time_stamp(ds, spec.stamp_digits, time_bounds)

    # Stream one variable at a time (time-varying first, then fx) to keep peak
    # memory low on long runs: (variable, table, frequency, date stamp, bounds).
    jobs = [
        (v, table, spec.frequency, date_stamp, time_bounds)
        for v in sorted(set(target_vars) - FX_VARS)
    ] + [(v, "fx", "fx", None, None) for v in sorted(FX_VARS.intersection(fx_vars))]

    written: dict[str, str] = {}
    missing: list[str] = []
    for cmor, tbl, frequency, stamp, bounds in jobs:
        if cmor in PLEV_VARS:
            da = extract_plev_variable(ds, cmor, plevs, subset_preinterp=explicit_plevs)
        else:
            da = extract_direct_variable(ds, cmor)
        if da is None:
            missing.append(cmor)
            continue
        fname = cmor_filename(cmor, tbl, model, experiment, member, grid_label, stamp)
        written[cmor] = write_cmor_variable(
            cmor_variable(to_cmip_layout(da), cmor, table=tbl),
            outdir / fname,
            {
                "Conventions": "CF-1.8",
                "model_id": model,
                "source_id": model,
                "experiment_id": experiment,
                "variant_label": member,
                "table_id": tbl,
                "variable_id": cmor,
                "realm": "atmos",
                "frequency": frequency,
                "grid_label": grid_label,
                "comment": "Produced by tools/atmospherebench_cmor.py from JCM output.",
            },
            time_bounds=bounds,
            dry_run=dry_run,
        )

    missing.sort()
    if missing and variables is not None:
        warnings.warn(
            f"Requested CMOR variable(s) not available in input ({table}): "
            f"{', '.join(missing)}",
            UserWarning,
            stacklevel=2,
        )
    return written, missing


def _classify_inputs(
    files: Sequence[pathlib.Path],
) -> tuple[list[tuple[pathlib.Path, re.Match]], list[tuple[pathlib.Path, re.Match]]]:
    """Split ``files`` into monthly files and chunk files (ordered by day), by filename."""
    monthly = [(p, m) for p in files if (m := _MONTHLY_FILE_RE.match(p.name))]
    chunks = sorted(
        ((p, m) for p in files if (m := _CHUNK_FILE_RE.match(p.name))),
        key=lambda item: float(item[1].group("day")),
    )
    return monthly, chunks


def discover_inputs(
    paths: Sequence[pathlib.Path],
    freq: str = "auto",
    prefix: str | None = None,
) -> dict[str, list[pathlib.Path]]:
    """Resolve input files or a JCM run directory into ``{'Amon': [...], 'day': [...]}``."""
    if len(paths) == 1 and paths[0].is_dir():
        run_dir = paths[0]
        monthly, chunks = _classify_inputs(sorted(run_dir.glob("*.nc")))
        if prefix is not None:
            monthly = [(p, m) for p, m in monthly if m.group("prefix") == prefix]
            chunks = [(p, m) for p, m in chunks if m.group("prefix") == prefix]
        else:
            prefixes = {m.group("prefix") for _, m in monthly + chunks}
            if len(prefixes) > 1:
                raise ValueError(
                    f"Multiple JCM run prefixes found in {run_dir}: "
                    f"{', '.join(sorted(prefixes))}. Pass --prefix or explicit file paths."
                )

        out: dict[str, list[pathlib.Path]] = {}
        if freq != "day" and monthly:
            out["Amon"] = [p for p, _ in monthly]
        if freq != "mon" and chunks:
            out["day"] = [p for p, _ in chunks]
        if freq == "both" and len(out) < 2:
            missing_tables = [t for t in _TABLES if t not in out]
            raise FileNotFoundError(
                f"--freq both requested, but missing {', '.join(missing_tables)} files in {run_dir}"
            )
        if not out:
            raise FileNotFoundError(
                f"No *_monthly_YYYY-MM.nc or *_day*.nc files found in {run_dir}"
            )
        return out

    files = [pathlib.Path(p) for p in paths]
    dirs = [str(p) for p in files if p.is_dir()]
    if dirs:
        raise ValueError(
            f"Pass either a single run directory or explicit NetCDF files, not both: {', '.join(dirs)}"
        )
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Input file(s) not found: {', '.join(missing)}")
    if freq == "day":
        return {"day": files}
    if freq == "mon":
        return {"Amon": files}

    monthly, chunks = _classify_inputs(files)
    chunk_files = [p for p, _ in chunks]
    if monthly and chunks and len(monthly) + len(chunks) == len(files):
        return {"Amon": [p for p, _ in monthly], "day": chunk_files}
    if freq == "both":
        raise FileNotFoundError(
            "--freq both requires both *_monthly_YYYY-MM.nc and *_day*.nc inputs."
        )
    if len(chunks) == len(files):
        return {"day": chunk_files}
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
    try:
        return xr.open_mfdataset(
            [str(p) for p in files],
            combine="by_coords",
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )
    except ImportError:
        # Fallback when dask is not installed.
        datasets = [xr.open_dataset(p) for p in files]
        try:
            combined = xr.concat(
                datasets,
                dim="time",
                data_vars="minimal",
                coords="minimal",
                compat="override",
            )
            return combined.sortby("time")
        finally:
            for d in datasets:
                d.close()


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
        "--prefix",
        help="Optional JCM filename prefix when a directory contains multiple runs",
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
        "--start-year",
        type=int,
        help="Optional first calendar year (inclusive) to include",
    )
    ap.add_argument(
        "--end-year",
        type=int,
        help="Optional last calendar year (inclusive) to include",
    )
    ap.add_argument(
        "--resample-daily",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resample sub-daily chunk files to daily means plus tasmax/tasmin (default: enabled)",
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


def _split_csv(raw: str) -> list[str]:
    """Split a comma-separated CLI value, dropping empty items."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_plev_arg(raw: str | None, flag: str) -> list[float] | None:
    """Parse a ``--plevs``-style value, failing before any input is opened."""
    if raw is None:
        return None
    parts = _split_csv(raw)
    if not parts:
        raise ValueError(f"{flag} must be a non-empty comma-separated list of pressures in hPa.")
    try:
        vals = [float(x) for x in parts]
    except ValueError as exc:
        raise ValueError(f"{flag}={raw!r}: could not parse pressure levels as floats.") from exc
    return validate_pressure_levels(vals).tolist()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    variables = _split_csv(args.variables) if args.variables else None
    plevs = {
        "Amon": _parse_plev_arg(args.plevs, "--plevs"),
        "day": _parse_plev_arg(args.daily_plevs, "--daily-plevs"),
    }

    grouped = discover_inputs(args.inputs, freq=args.freq, prefix=args.prefix)
    outdir = resolve_output_dir(args.out, args.experiment, args.member, flat=args.flat)

    # Fixed fields are written once, from the first table that can supply them.
    remaining_fx: set[str] = set()
    if not args.no_fx:
        remaining_fx = set(FX_VARS if variables is None else FX_VARS.intersection(variables))

    all_written: list[str] = []
    missing_by_table: dict[str, list[str]] = {}
    for table in _TABLES:
        if table not in grouped:
            continue
        with _open_inputs(grouped[table]) as ds:
            written, missing = convert_dataset(
                ds,
                model=args.model,
                experiment=args.experiment,
                member=args.member,
                table=table,
                grid_label=args.grid_label,
                outdir=outdir,
                variables=variables,
                plevs_hpa=plevs[table],
                fx_vars=set(remaining_fx),
                resample_daily=args.resample_daily,
                start_year=args.start_year,
                end_year=args.end_year,
                dry_run=args.dry_run,
            )
        remaining_fx -= written.keys()
        all_written.extend(written.values())
        if missing:
            missing_by_table[table] = missing

    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {len(all_written)} file(s) to {outdir}")
    for fname in all_written:
        print(" ", fname)
    for table, miss in missing_by_table.items():
        print(f"\n{len(miss)} target {table} variable(s) unavailable in input:")
        print("  " + ", ".join(miss))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
