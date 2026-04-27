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

    Builds JAX config to match Fortran ECHAM6.3 ``__ICON__`` defaults so
    any remaining discrepancy is a port bug, not a parameter choice.
    Returns a dict with the same field names as the Fortran driver
    output for ``report_diff`` to consume.
    """
    import jax.numpy as jnp
    from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
        tiedtke_nordeng_convection, ConvectionParameters,
    )

    Rd, cp, grav = 287.04, 1004.64, 9.80665

    def s(name):
        return jnp.asarray(state[name][0])

    T = s("pten")
    q = s("pqen")
    qc_total = s("pxen")        # total cloud condensate (kg/kg)
    u = s("puen")
    v = s("pven")
    p = s("papp1")
    z_half = s("pzh")

    # Layer thickness (m, positive). ECHAM convention: index 0 = top so
    # z_half[k] > z_half[k+1] and the upward-positive thickness is
    # z_half[k] - z_half[k+1].
    layer_thickness = z_half[:-1] - z_half[1:]
    Tv = T * (1.0 + 0.608 * q)
    rho = p / (Rd * Tv)
    # Split total cloud water into liquid/ice on a 235-273.15 K mixed-phase
    # ramp (matches Sundqvist convention).
    fliq = jnp.clip((T - 235.0) / (273.15 - 235.0), 0.0, 1.0)
    qc_liq = qc_total * fliq
    qi_ice = qc_total * (1.0 - fliq)

    # Match Fortran parameters.  Notable mismatches in the JAX defaults
    # vs Fortran ECHAM-ICON:
    #   cprcon  = 1.4e-3 (JAX)   vs 2.5e-4 (Fortran __ICON__)
    #   cmfctop = 0.33  (JAX)    vs 0.20    (Fortran)
    #   cmfdeps = 0.33  (JAX)    vs 0.30    (Fortran)
    config = ConvectionParameters.default(
        dt_conv=dtime,
        entrpen=1.0e-4, entrscv=3.0e-3, entrmid=1.0e-4,
        entrdd=2.0e-4,
        tau=7200.0, cmfcmax=1.0, cmfcmin=1.0e-10,
        cprcon=2.5e-4, cevapcu=2.0e-5,
        cmfctop=0.20, cmfdeps=0.30,
    )

    # Build the same updraft + tendency machinery the wrapper does, so
    # we can inspect the un-adjusted ``calculate_tendencies`` output
    # alongside the post-saturation-adjustment final tendency.
    from jcm.physics.convection.tiedtke_nordeng.updraft import calculate_updraft
    from jcm.physics.convection.tiedtke_nordeng.downdraft import calculate_downdraft
    from jcm.physics.convection.tiedtke_nordeng.flux_tendencies import (
        calculate_tendencies as _calc_tend, mass_flux_closure,
    )
    from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
        find_cloud_base as _find_cb, calculate_cape_cin as _cape_cin,
    )

    cloud_base, _has_cb = _find_cb(T, q, p, config)
    cape, cin = _cape_cin(T, q, p, layer_thickness, cloud_base, config)
    conv_type = 1 if float(cape) > 1000 else (2 if float(cape) > 100 else 0)
    cloud_depth = 5 if conv_type == 2 else 35
    ktop_ceil = jnp.maximum(cloud_base - cloud_depth, jnp.array(2))
    mfb = mass_flux_closure(cape, cin, jnp.array(0.0), conv_type, config)
    upd = calculate_updraft(
        T, q, p, layer_thickness, rho, cloud_base, ktop_ceil,
        conv_type, mfb, config,
    )
    dwn = calculate_downdraft(
        T, q, p, layer_thickness, rho, upd, jnp.array(0.0),
        cloud_base, ktop_ceil, config,
    )
    raw_tend = _calc_tend(
        T, q, u, v, p, rho, layer_thickness, upd, dwn,
        cloud_base, ktop_ceil, dtime, config,
    )
    raw_dtedt = np.asarray(raw_tend.dtedt)
    print(f"  raw calculate_tendencies dtedt: max|.|={np.max(np.abs(raw_dtedt)):.3e} K/s "
          f"({np.max(np.abs(raw_dtedt)) * 86400:.2f} K/day)")
    print(f"    levels with |dtedt|>1e-12: "
          f"{np.where(np.abs(raw_dtedt) > 1e-12)[0].tolist()}")
    print(f"    raw_dtedt[10:30] = {raw_dtedt[10:30]}")
    print(f"    raw_dtedt[30:46] = {raw_dtedt[30:46]}")

    tend, jstate = tiedtke_nordeng_convection(
        T, q, p, layer_thickness, rho, u, v, qc_liq, qi_ice, dtime, config,
    )

    # JAX-side internal diagnostics — print directly so we can see what
    # the JAX scheme thought about cloud base / top vs Fortran's choice.
    print(f"  JAX internal: kbase={int(jstate.kbase)} (~{float(p[int(jstate.kbase)]) / 100:.1f} hPa)  "
          f"ktop={int(jstate.ktop)} (~{float(p[int(jstate.ktop)]) / 100:.1f} hPa)  "
          f"ktype={int(jstate.ktype)}  prate={float(jstate.prate):.3e} kg/m^2/s")
    print(f"  JAX max(mfu)={float(jnp.max(jstate.mfu)):.4e} kg/m^2/s  "
          f"max(tu)={float(jnp.max(jstate.tu)):.2f} K  "
          f"max(qu)={float(jnp.max(jstate.qu)) * 1000:.3f} g/kg")
    # Where does the updraft mass flux actually live?
    mfu_arr = np.asarray(jstate.mfu)
    nz = np.nonzero(mfu_arr)[0]
    if nz.size > 0:
        print(f"  JAX mfu nonzero at levels {nz.min()}..{nz.max()}: "
              f"values [{mfu_arr[nz.min()]:.3e} .. {mfu_arr[nz.max()]:.3e}]")
    # Where does the heating tendency actually live (post-mask)?
    dtedt_arr = np.asarray(tend.dtedt)
    nz = np.nonzero(dtedt_arr)[0]
    if nz.size > 0:
        print(f"  JAX dtedt nonzero at levels {nz.min()}..{nz.max()}: "
              f"max |dT/dt| = {np.max(np.abs(dtedt_arr)):.3e} K/s "
              f"= {np.max(np.abs(dtedt_arr)) * 86400:.1f} K/day")

    pq_cnv   = (cp * tend.dtedt).reshape(1, -1)
    pqte_cnv = tend.dqdt.reshape(1, -1)
    pvom     = tend.dudt.reshape(1, -1)
    pvol     = tend.dvdt.reshape(1, -1)
    pxtecl   = tend.dqc_dt.reshape(1, -1)
    pxteci   = tend.dqi_dt.reshape(1, -1)

    # Surface precip split — frozen at the surface if T_surf < 273.15.
    surface_T = float(T[-1])
    rate = float(tend.precip_conv)
    prsfc = np.array([rate if surface_T >= 273.15 else 0.0])
    pssfc = np.array([rate if surface_T <  273.15 else 0.0])

    ktop = int(jstate.ktop)
    if 0 <= ktop < len(p):
        ptop = np.array([float(p[ktop])])
    else:
        ptop = np.array([99999.0])

    return {
        "ktype":     np.array([int(jstate.ktype)]),
        "kctop":     np.array([ktop + 1]),    # JAX 0-idx → Fortran 1-idx
        "pq_cnv":    np.asarray(pq_cnv),
        "pqte_cnv":  np.asarray(pqte_cnv),
        "pvom_cnv":  np.asarray(pvom),
        "pvol_cnv":  np.asarray(pvol),
        "pxtecl":    np.asarray(pxtecl),
        "pxteci":    np.asarray(pxteci),
        "prsfc":     prsfc,
        "pssfc":     pssfc,
        "ptop":      ptop,
        # JAX doesn't expose detrainment-budget scalars; leave NaN.
        "pcon_dtrl": np.array([np.nan]),
        "pcon_dtri": np.array([np.nan]),
        "pcon_iqte": np.array([np.nan]),
    }


def report_diff(jax_out: dict | None, fort_out: dict):
    print(f"\n{'field':>12s}  {'shape':>10s}  {'fmin':>14s}  {'fmax':>14s}"
          f"  {'jmin':>14s}  {'jmax':>14s}  {'maxabs':>14s}  {'meanabs':>14s}")
    for k, fv in fort_out.items():
        fv = np.asarray(fv, dtype=float)
        if jax_out is None or k not in jax_out:
            print(f"{k:>12s}  {str(fv.shape):>10s}  "
                  f"{fv.min():>14.4e}  {fv.max():>14.4e}  "
                  f"{'-':>14s}  {'-':>14s}  {'-':>14s}  {'-':>14s}")
            continue
        jv = np.asarray(jax_out[k], dtype=float)
        if np.all(np.isnan(jv)):
            print(f"{k:>12s}  {str(fv.shape):>10s}  "
                  f"{fv.min():>14.4e}  {fv.max():>14.4e}  "
                  f"{'(JAX N/A)':>14s}  {'-':>14s}  {'-':>14s}  {'-':>14s}")
            continue
        diff = np.abs(fv - jv)
        print(f"{k:>12s}  {str(fv.shape):>10s}  "
              f"{fv.min():>14.4e}  {fv.max():>14.4e}  "
              f"{jv.min():>14.4e}  {jv.max():>14.4e}  "
              f"{diff.max():>14.4e}  {diff.mean():>14.4e}")


def report_per_level(jax_out: dict, fort_out: dict, field: str):
    """For 2-D fields (e.g. pq_cnv), show per-level F vs J + diff."""
    if field not in fort_out or field not in jax_out:
        return
    fv = np.asarray(fort_out[field], dtype=float).ravel()
    jv = np.asarray(jax_out[field], dtype=float).ravel()
    if fv.size != jv.size or np.all(np.isnan(jv)):
        return
    print(f"\n  per-level diff for ``{field}``")
    print(f"    {'k':>3s}  {'fortran':>14s}  {'jax':>14s}  {'diff':>14s}")
    for k in range(fv.size):
        if abs(fv[k]) < 1e-15 and abs(jv[k]) < 1e-15:
            continue
        print(f"    {k:>3d}  {fv[k]:>14.4e}  {jv[k]:>14.4e}  "
              f"{(fv[k] - jv[k]):>14.4e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", help="netcdf with the JAX run")
    parser.add_argument("--time", type=int, default=30, help="time index")
    parser.add_argument("--col",  type=int, default=0,  help="flat column index")
    parser.add_argument("--rce", action="store_true",
                        help="Use a synthetic 47-level tropical RCE column "
                             "instead of reading from --nc")
    parser.add_argument("--dtime", type=float, default=1800.0)
    parser.add_argument("--input", default="/tmp/cumastr_in.bin")
    parser.add_argument("--output", default="/tmp/cumastr_out.bin")
    parser.add_argument("--no-jax", action="store_true",
                        help="skip JAX side, only run Fortran")
    args = parser.parse_args()

    if args.rce:
        sys.path.insert(0, str(HERE))
        from test_rce_column import tropical_rce_sounding
        state, eta_full = tropical_rce_sounding(klev=47)
        print("Synthetic tropical RCE column (47 levels)")
    else:
        if not args.nc:
            raise SystemExit("--nc is required unless --rce is given")
        ds = xr.open_dataset(args.nc)
        state, eta_full = derive_state_from_xarray(
            ds, args.time, args.col, args.dtime,
        )
        print(f"Loaded column {args.col} at time={args.time}.")
    nlev = state["pten"].shape[1]
    print(f"  klev={nlev}  surface T={state['pten'][0,-1]:.2f} K  "
          f"surface q={state['pqen'][0,-1]*1000:.3f} g/kg")

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
    if jax_out is not None:
        for fname in ("pq_cnv", "pqte_cnv"):
            report_per_level(jax_out, fort_out, fname)


if __name__ == "__main__":
    main()
