"""Post-process JCM output into AeroCom phase-4 submission files.

Reads the netCDF a JCM run writes and emits **one variable per file** under
the AeroCom naming convention, with CMOR variable names and units.

    python tools/aerocom_cmor.py run_day30.nc --model JCM-v2 \
        --experiment AP4-CTRL-PD-NudClim --period 2010 --out submission/

Filename convention (aci-baseline / imo2020 form)::

    aerocom_<ModelName>_<ExperimentName>_<VariableName>_<VertCoord>_<Period>_<Freq>.nc

AP4-CTRL uses a phase-4 variant with ``aerocom4_`` and the simulation name
folded into the experiment field; pass ``--convention aerocom4`` for that.

``<ModelName>`` may not contain underscores (use ``-``) and is capped at 20
characters — enforced here rather than discovered at submission.

Vertical orientation
--------------------
JCM writes output **surface-first** (level index 0 is the surface; see
CLAUDE.md). This script does not reorder levels by default, because the
protocol specifies the *coordinate type* (``ModelLevel``/``Surface``/
``Column``/``TOA``) rather than an ordering, and models submit on their
native grid. Pass ``--flip-levels`` to write TOA-first if your AeroCom
contact asks for it, and record the choice in the submission notes.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import xarray as xr

# jcm name -> (CMOR name, units, vertical coordinate type, scale, offset)
#
# ``scale``/``offset`` convert jcm units to the units AeroCom requests.
# Only variables JCM actually writes are listed; the remaining requested
# fields need model-side work (see jax-gcm#581).
NAME_MAP: dict[str, tuple[str, str, str, float, float]] = {
    # --- radiative fluxes (TOA / surface) ---
    "radiation.toa_lw_up": ("rlut", "W m-2", "TOA", 1.0, 0.0),
    "radiation.toa_lw_up_clear": ("rlutcs", "W m-2", "TOA", 1.0, 0.0),
    "radiation.toa_sw_up": ("rsut", "W m-2", "TOA", 1.0, 0.0),
    "radiation.toa_sw_up_clear": ("rsutcs", "W m-2", "TOA", 1.0, 0.0),
    "radiation.toa_sw_down": ("rsdt", "W m-2", "TOA", 1.0, 0.0),
    "radiation.surface_sw_down": ("rsds", "W m-2", "Surface", 1.0, 0.0),
    "radiation.surface_lw_down": ("rlds", "W m-2", "Surface", 1.0, 0.0),
    "radiation.surface_sw_up": ("rsus", "W m-2", "Surface", 1.0, 0.0),
    "radiation.surface_lw_up": ("rlus", "W m-2", "Surface", 1.0, 0.0),
    # --- turbulent fluxes ---
    "surface.latent_heat_flux": ("hfls", "W m-2", "Surface", 1.0, 0.0),
    "surface.sensible_heat_flux": ("hfss", "W m-2", "Surface", 1.0, 0.0),
    # --- basic state ---
    "surface_pressure": ("ps", "Pa", "Surface", 1.0, 0.0),
    "surface.surface_temperature": ("ts", "K", "Surface", 1.0, 0.0),
    "temperature": ("ta", "K", "ModelLevel", 1.0, 0.0),
    "specific_humidity": ("hus", "kg kg-1", "ModelLevel", 1.0, 0.0),
    "relative_humidity": ("hur", "%", "ModelLevel", 100.0, 0.0),
    "u_wind": ("ua", "m s-1", "ModelLevel", 1.0, 0.0),
    "v_wind": ("va", "m s-1", "ModelLevel", 1.0, 0.0),
    "pressure_full": ("pfull", "Pa", "ModelLevel", 1.0, 0.0),
    "geopotential": ("zg", "m", "ModelLevel", 1.0, 0.0),
    "air_density": ("rho", "kg m-3", "ModelLevel", 1.0, 0.0),
    "layer_thickness": ("dzhalf", "m", "ModelLevel", 1.0, 0.0),
    # --- clouds ---
    "radiation.total_cloud_cover": ("clt", "1", "Column", 1.0, 0.0),
    "clouds.cloud_fraction": ("cl", "1", "ModelLevel", 1.0, 0.0),
    "qc": ("clw", "kg kg-1", "ModelLevel", 1.0, 0.0),
    "qi": ("cli", "kg kg-1", "ModelLevel", 1.0, 0.0),
    "clouds.droplet_number": ("cdnc3d", "m-3", "ModelLevel", 1.0, 0.0),
    # jcm carries effective radii in microns; AeroCom asks for metres.
    "clouds.r_eff_liq": ("cdr3d", "m", "ModelLevel", 1e-6, 0.0),
    "clouds.r_eff_ice": ("icr3d", "m", "ModelLevel", 1e-6, 0.0),
    # --- precipitation ---
    "clouds.precip_rain": ("prlr", "kg m-2 s-1", "Surface", 1.0, 0.0),
    "clouds.precip_snow": ("prls", "kg m-2 s-1", "Surface", 1.0, 0.0),
    "convection.precip_conv": ("prc", "kg m-2 s-1", "Surface", 1.0, 0.0),
    # --- aerosol optics ---
    "aerosol.aod_total": ("od550aer", "1", "Column", 1.0, 0.0),
    "aerosol.angstrom": ("angstrm", "1", "Column", 1.0, 0.0),
    # --- boundary layer ---
    "vertical_diffusion.pbl_height": ("hdtcbl", "m", "Surface", 1.0, 0.0),
    # --- AerocomDiagnostics term output ---
    "aerocom_clt": ("clt", "1", "Column", 1.0, 0.0),
    "aerocom_ttop": ("ttop", "K", "Column", 1.0, 0.0),
    "aerocom_cdr": ("cdr", "m", "Column", 1.0, 0.0),
    "aerocom_icr": ("icr", "m", "Column", 1.0, 0.0),
    "aerocom_cdnc": ("cdnc", "m-3", "Column", 1.0, 0.0),
    "aerocom_lcc": ("lcc", "1", "Column", 1.0, 0.0),
    "aerocom_icc": ("icc", "1", "Column", 1.0, 0.0),
    "aerocom_cod": ("cod", "1", "Column", 1.0, 0.0),
    "aerocom_codliq": ("codliq", "1", "Column", 1.0, 0.0),
    "aerocom_codice": ("codice", "1", "Column", 1.0, 0.0),
    "aerocom_lwp": ("lwp", "kg m-2", "Column", 1.0, 0.0),
    "aerocom_iwp": ("iwp", "kg m-2", "Column", 1.0, 0.0),
    "aerocom_cllvi": ("cllvi", "kg m-2", "Column", 1.0, 0.0),
    "aerocom_clivi": ("clivi", "kg m-2", "Column", 1.0, 0.0),
    "aerocom_prw": ("prw", "kg m-2", "Column", 1.0, 0.0),
    "aerocom_cdnum": ("cdnum", "m-2", "Column", 1.0, 0.0),
    "aerocom_icnum": ("icnum", "m-2", "Column", 1.0, 0.0),
    "aerocom_albedo": ("albedo", "1", "Column", 1.0, 0.0),
    "aerocom_lts": ("lts", "K", "Surface", 1.0, 0.0),
    "aerocom_u200": ("u200", "m s-1", "Surface", 1.0, 0.0),
    "aerocom_v200": ("v200", "m s-1", "Surface", 1.0, 0.0),
    "aerocom_u700": ("u700", "m s-1", "Surface", 1.0, 0.0),
    "aerocom_v700": ("v700", "m s-1", "Surface", 1.0, 0.0),
    "aerocom_N70": ("N70", "m-3", "ModelLevel", 1.0, 0.0),
    "aerocom_N100": ("N100", "m-3", "ModelLevel", 1.0, 0.0),
    "aerocom_PM1": ("PM1", "kg m-3", "ModelLevel", 1.0, 0.0),
    "aerocom_PM10": ("PM10", "kg m-3", "ModelLevel", 1.0, 0.0),
}

# Aerosol burdens are emitted per tracer as ``aerocom_burden_m_<spec>_<mode>``;
# AeroCom wants them summed per species as ``burden_<spec>``.
BURDEN_SPECIES = ("so4", "bc", "oc", "poa", "soa", "ss", "du", "moa")

VALID_VERT = ("Surface", "TOA", "Column", "ModelLevel")


def _check_model_name(name: str) -> str:
    if "_" in name:
        raise SystemExit(
            f"--model {name!r}: underscores are not allowed in <ModelName> "
            "(AeroCom convention); use '-' instead.")
    if len(name) > 20:
        raise SystemExit(
            f"--model {name!r}: <ModelName> is capped at 20 characters "
            f"(got {len(name)}).")
    return name


def _filename(convention, model, experiment, var, vert, period, freq):
    if convention == "aerocom4":
        # phase-4 form: aerocom4_<Model>_<Exp>-<Sim>_<Var>_<Vert>_<Year>_<Freq>.nc
        return f"aerocom4_{model}_{experiment}_{var}_{vert}_{period}_{freq}.nc"
    return f"aerocom_{model}_{experiment}_{var}_{vert}_{period}_{freq}.nc"


def _collect_burdens(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """Sum per-tracer column burdens into per-species totals."""
    out: dict[str, xr.DataArray] = {}
    for spec in BURDEN_SPECIES:
        parts = [ds[v] for v in ds.data_vars
                 if v.startswith("aerocom_burden_m_") and f"_{spec}_" in v]
        if parts:
            total = parts[0]
            for p in parts[1:]:
                total = total + p
            out[f"burden_{spec}"] = total
    return out


def convert(
    ds: xr.Dataset,
    model: str,
    experiment: str,
    period: str,
    freq: str,
    outdir: pathlib.Path,
    convention: str = "aerocom",
    flip_levels: bool = False,
    dry_run: bool = False,
) -> tuple[list[str], list[str]]:
    """Write one file per mapped variable; return (written, skipped)."""
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    candidates: dict[str, tuple[xr.DataArray, str, str]] = {}
    for src, (cmor, units, vert, scale, offset) in NAME_MAP.items():
        if src not in ds.data_vars:
            continue
        da = ds[src] * scale + offset
        candidates[cmor] = (da, units, vert)
    for name, da in _collect_burdens(ds).items():
        candidates[name] = (da, "kg m-2", "Column")

    for cmor, (da, units, vert) in sorted(candidates.items()):
        assert vert in VALID_VERT, vert
        if flip_levels and "level" in da.dims:
            da = da.isel(level=slice(None, None, -1))
        da = da.rename(cmor)
        da.attrs.update(units=units, standard_name=cmor,
                        comment=("Produced by tools/aerocom_cmor.py from JCM "
                                 "output; see jax-gcm#581 for coverage."))
        fname = _filename(convention, model, experiment, cmor, vert, period, freq)
        if not dry_run:
            out = xr.Dataset({cmor: da})
            out.attrs.update(
                model_id=model, experiment_id=experiment, frequency=freq,
                vertical_coordinate_type=vert,
                level_order=("TOA-first" if flip_levels else "surface-first"),
            )
            out.to_netcdf(outdir / fname)
        written.append(fname)

    mapped_srcs = {s for s in NAME_MAP if s in ds.data_vars}
    skipped = sorted(set(ds.data_vars) - mapped_srcs
                     - {v for v in ds.data_vars if v.startswith("aerocom_burden_")})
    return written, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=pathlib.Path, help="JCM output netCDF")
    ap.add_argument("--model", required=True,
                    help="<ModelName>: no underscores, <=20 chars")
    ap.add_argument("--experiment", required=True,
                    help="e.g. all_2000, CTRL, 20shp, AP4-CTRL-PD-NudClim")
    ap.add_argument("--period", required=True, help="e.g. 2010")
    ap.add_argument("--freq", default="monthly",
                    choices=["timeinvariant", "hourly", "3hourly", "daily", "monthly"])
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("aerocom_submit"))
    ap.add_argument("--convention", default="aerocom", choices=["aerocom", "aerocom4"])
    ap.add_argument("--flip-levels", action="store_true",
                    help="write TOA-first instead of JCM's native surface-first")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written without writing")
    args = ap.parse_args(argv)

    _check_model_name(args.model)
    ds = xr.open_dataset(args.input, decode_times=False)
    written, skipped = convert(
        ds, args.model, args.experiment, args.period, args.freq, args.out,
        convention=args.convention, flip_levels=args.flip_levels,
        dry_run=args.dry_run)

    print(f"{'would write' if args.dry_run else 'wrote'} {len(written)} "
          f"variable file(s) to {args.out}")
    for f in written:
        print("  ", f)
    if skipped:
        print(f"\n{len(skipped)} JCM variable(s) had no AeroCom mapping "
              "(expected — most are internal diagnostics):")
        print("   " + ", ".join(skipped[:15])
              + (" ..." if len(skipped) > 15 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
