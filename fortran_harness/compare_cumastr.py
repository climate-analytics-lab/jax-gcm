"""Run the same column through ECHAM/ICON ``cumastr`` (Fortran) and the
JAX Tiedtke-Nordeng port, then diff every output field.

Usage:
    python compare_cumastr.py --nc icon_t42_l8_baseline.nc --time 30 --col 0

Picks one column at the requested time index, dumps it to a Fortran-
unformatted binary file, calls ``build/cumastr_driver``, reads the
result, and compares against the JAX scheme run on the same column.
Reports max-abs and mean-abs errors per output field.
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr


HERE = Path(__file__).resolve().parent
DRIVER = HERE / "build" / "cumastr_driver"

# Fortran sequential unformatted records have a 4-byte length prefix and
# a 4-byte length suffix around the data. We hand-roll the I/O because
# numpy / scipy don't have a streamlined writer. wp = 8 bytes (real64).


def _write_record(f, data: np.ndarray):
    """Write one Fortran sequential-unformatted record."""
    blob = np.ascontiguousarray(data).tobytes(order="F")
    f.write(struct.pack("<i", len(blob)))
    f.write(blob)
    f.write(struct.pack("<i", len(blob)))


def _read_record(f, dtype, count) -> np.ndarray:
    nb_in = struct.unpack("<i", f.read(4))[0]
    expect = np.dtype(dtype).itemsize * count
    if nb_in != expect:
        raise ValueError(
            f"record length {nb_in} != expected {expect} for "
            f"dtype={dtype}, count={count}"
        )
    blob = f.read(nb_in)
    nb_out = struct.unpack("<i", f.read(4))[0]
    if nb_out != nb_in:
        raise ValueError(
            f"record length mismatch: head {nb_in}, tail {nb_out}"
        )
    return np.frombuffer(blob, dtype=dtype, count=count)


def write_input_file(
    path: Path, kproma: int, klev: int, dtime: float,
    eta_full: np.ndarray,
    state: dict,
):
    """Pack a single-column input file matching the cumastr_driver layout."""
    with open(path, "wb") as f:
        # Record 1: kproma, klev (two 4-byte ints in one record)
        ints = np.array([kproma, klev], dtype="<i4")
        _write_record(f, ints)
        # Record 2: dtime
        _write_record(f, np.array([dtime], dtype="<f8"))
        # Record 3: eta_full(klev)
        _write_record(f, eta_full.astype("<f8"))
        # Records 4-15: 2D fields
        for k in (
            "pten", "pqen", "pxen", "puen", "pven", "pverv",
            "papp1", "paphp1", "pgeo", "pgeoh", "pzf", "pzh",
        ):
            arr = state[k].astype("<f8")
            _write_record(f, arr)
        # Record 16: pmref
        _write_record(f, state["pmref"].astype("<f8"))
        # Record 17: pqte
        _write_record(f, state["pqte"].astype("<f8"))
        # Record 18: pqhfla(kproma)
        _write_record(f, state["pqhfla"].astype("<f8"))
        # Record 19: pthvsig
        _write_record(f, state["pthvsig"].astype("<f8"))
        # Record 20: ldland — Fortran LOGICAL is 4 bytes; True=-1, False=0
        ld = np.where(state["ldland"], np.int32(1), np.int32(0)).astype("<i4")
        _write_record(f, ld)


def read_output_file(path: Path, kproma: int, klev: int) -> dict:
    out = {}
    with open(path, "rb") as f:
        out["ktype"] = _read_record(f, "<i4", kproma)
        out["kctop"] = _read_record(f, "<i4", kproma)
        for k in ("pq_cnv", "pqte_cnv", "pvom_cnv", "pvol_cnv",
                  "pxtecl",  "pxteci"):
            arr = _read_record(f, "<f8", kproma * klev)
            out[k] = arr.reshape(klev, kproma).T  # Fortran ordering -> (kproma, klev)
        for k in ("prsfc", "pssfc", "ptop",
                  "pcon_dtrl", "pcon_dtri", "pcon_iqte"):
            out[k] = _read_record(f, "<f8", kproma)
    return out


def derive_state_from_xarray(ds: xr.Dataset, time_idx: int, col_idx: int,
                              dtime: float = 1800.0) -> dict:
    """Pick one column from the netcdf and assemble cumastr inputs."""
    nlev = ds.sizes["sigma"] if "sigma" in ds.sizes else ds.sizes["level"]
    nlon = ds.sizes["longitude"] if "longitude" in ds.sizes else ds.sizes["lon"]
    nlat = ds.sizes["latitude"] if "latitude" in ds.sizes else ds.sizes["lat"]
    ncols = nlon * nlat
    if not (0 <= col_idx < ncols):
        raise ValueError(f"col_idx={col_idx} outside [0, {ncols})")

    ilat = col_idx // nlon
    ilon = col_idx %  nlon

    # Snapshot at time_idx for one column — keep level axis
    snap = ds.isel(time=time_idx).isel({
        "longitude" if "longitude" in ds.sizes else "lon": ilon,
        "latitude"  if "latitude"  in ds.sizes else "lat": ilat,
    })

    # Pull what we can. Fields not in the dataset get sane defaults.
    T = np.asarray(snap["temperature"])     # (klev,) in K
    # Dataset stores specific_humidity in g/kg (matches the rescaling at the
    # ICON-physics boundary, see commit 08e3fc5). cumastr expects kg/kg.
    q = np.asarray(snap["specific_humidity"]) * 1.0e-3
    u = np.asarray(snap["u_wind"])
    v = np.asarray(snap["v_wind"])

    # Pressure: half/full from sigma + surface pressure, if not in dataset.
    p0 = 101325.0
    sigma_centers = np.asarray(ds["sigma"].values) if "sigma" in ds.coords \
        else np.linspace(1.0, 1.0/nlev, nlev) * (1 - 0.5/nlev)
    sigma_bnds = np.linspace(0.0, 1.0, nlev + 1)
    p_full = sigma_centers * p0
    p_half = sigma_bnds * p0

    omega = np.zeros(nlev)
    qc = np.zeros(nlev)  # not always saved; use 0 if not
    if "tracers.qc" in ds:
        qc += np.asarray(snap["tracers.qc"]) * 1.0e-3
    if "tracers.qi" in ds:
        qc += np.asarray(snap["tracers.qi"]) * 1.0e-3
    qte = np.zeros(nlev)

    # Build geometric height + geopotential from hypsometric assuming T
    # (ad hoc; cumastr only uses pgeoh-pgeo deltas on heights for cuasc).
    # Use d(geo) = -R_d T_v d(ln p)
    Rd = 287.04
    Tv = T * (1 + 0.608 * q)
    p_full_safe = np.maximum(p_full, 1.0)
    p_half_safe = np.maximum(p_half, 1.0)
    dpln = np.diff(np.log(p_half_safe))
    dgeo = Rd * Tv * dpln  # geopotential thickness of each layer (positive)
    geo_full = np.zeros(nlev)
    geo_full[-1] = 0.5 * dgeo[-1]
    for k in range(nlev - 2, -1, -1):
        geo_full[k] = geo_full[k+1] + 0.5 * (dgeo[k+1] + dgeo[k])
    geo_half = np.zeros(nlev + 1)
    geo_half[-1] = 0.0
    for k in range(nlev - 1, -1, -1):
        geo_half[k] = geo_half[k+1] + dgeo[k]

    grav = 9.80665
    z_full = geo_full / grav
    z_half = geo_half / grav

    # Reference layer mass = (p_half[k+1] - p_half[k]) / g
    pmref = np.diff(p_half) / grav

    # Surface latent heat flux: pull from the dataset if it's there,
    # otherwise fall back to a moderate tropical value (1e-4 kg/m²/s ≈ 250 W/m²).
    if "surface.latent_heat_flux" in ds.data_vars:
        # ECHAM convention: pqhfla in kg/m²/s = LHF / (Lv ≈ 2.5e6)
        lhf = float(snap["surface.latent_heat_flux"])
        pqhfla = np.array([abs(lhf) / 2.5e6])
    else:
        pqhfla = np.array([1.0e-4])
    pthvsig = np.array([1.0])     # std dev of virtual pot T — typical PBL value
    ldland = np.array([False])

    # Reshape to (kproma=1, klev) — Fortran column-major needs careful layout.
    def col(x):
        return np.asarray(x).reshape(1, -1)

    return {
        "pten":   col(T),
        "pqen":   col(np.maximum(q, 0.0)),
        "pxen":   col(qc),
        "puen":   col(u),
        "pven":   col(v),
        "pverv":  col(omega),
        "papp1":  col(p_full),
        "paphp1": col(p_half),
        "pgeo":   col(geo_full),
        "pgeoh":  col(geo_half),
        "pzf":    col(z_full),
        "pzh":    col(z_half),
        "pmref":  col(pmref),
        "pqte":   col(qte),
        "pqhfla": pqhfla,
        "pthvsig":pthvsig,
        "ldland": ldland,
    }, sigma_centers


def run_jax_cumastr_equivalent(state: dict, dtime: float):
    """Run the JAX Tiedtke-Nordeng convection scheme on the same column.

    Returns the same field names as the Fortran output. Fields the JAX
    side doesn't compute are returned as None.
    """
    import jax.numpy as jnp
    from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
        get_convection_tendencies,
    )
    from jcm.physics.icon.parameters import Parameters

    # The JAX scheme expects (klev, ...) state. Strip kproma=1.
    def s(name):
        return jnp.asarray(state[name][0]) if state[name].ndim == 2 \
               else jnp.asarray(state[name][0])

    p = Parameters.default()
    raise NotImplementedError(
        "Plumb the JAX-side cumastr through `tiedtke_nordeng_full` here. "
        "Stub out for the first commit; we'll wire it up once the Fortran "
        "side is producing sane numbers."
    )


def report_diff(jax_out: dict | None, fort_out: dict):
    print(f"\n{'field':>12s}  {'shape':>10s}  {'fmin':>14s}  {'fmax':>14s}"
          f"  {'jmin':>14s}  {'jmax':>14s}  {'maxabs':>14s}  {'meanabs':>14s}")
    for k, fv in fort_out.items():
        fv = np.asarray(fv)
        if jax_out is None or k not in jax_out:
            print(f"{k:>12s}  {str(fv.shape):>10s}  {fv.min():>14.4e}  {fv.max():>14.4e}"
                  f"  {'-':>14s}  {'-':>14s}  {'-':>14s}  {'-':>14s}")
            continue
        jv = np.asarray(jax_out[k])
        diff = np.abs(fv - jv)
        print(f"{k:>12s}  {str(fv.shape):>10s}  {fv.min():>14.4e}  {fv.max():>14.4e}"
              f"  {jv.min():>14.4e}  {jv.max():>14.4e}"
              f"  {diff.max():>14.4e}  {diff.mean():>14.4e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", required=True, help="netcdf with the JAX run")
    parser.add_argument("--time", type=int, default=30, help="time index")
    parser.add_argument("--col",  type=int, default=0,  help="flat column index")
    parser.add_argument("--dtime", type=float, default=1800.0)
    parser.add_argument("--input", default="/tmp/cumastr_in.bin")
    parser.add_argument("--output", default="/tmp/cumastr_out.bin")
    parser.add_argument("--no-jax", action="store_true",
                        help="skip JAX side, only run Fortran")
    args = parser.parse_args()

    ds = xr.open_dataset(args.nc)
    state, eta_full = derive_state_from_xarray(ds, args.time, args.col, args.dtime)
    nlev = state["pten"].shape[1]
    print(f"Loaded column {args.col} at time={args.time} (klev={nlev}). "
          f"T={state['pten'][0]} K\n  q (g/kg)={state['pqen'][0]*1000}")

    write_input_file(Path(args.input), kproma=1, klev=nlev,
                     dtime=args.dtime, eta_full=eta_full, state=state)
    res = subprocess.run(
        [str(DRIVER), args.input, args.output],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print("=== driver stdout ===");  print(res.stdout)
        print("=== driver stderr ===");  print(res.stderr)
        raise SystemExit(res.returncode)
    print(res.stdout.strip())

    fort_out = read_output_file(Path(args.output), kproma=1, klev=nlev)

    jax_out = None
    if not args.no_jax:
        try:
            jax_out = run_jax_cumastr_equivalent(state, args.dtime)
        except NotImplementedError as e:
            print(f"\n[skip JAX side] {e}")

    report_diff(jax_out, fort_out)


if __name__ == "__main__":
    main()
