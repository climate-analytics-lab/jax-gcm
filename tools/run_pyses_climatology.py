"""Chunked long-run driver for pySES CAM-SE + ECHAM climatologies.

The pySES backend is not wired into the Hydra runner yet, so this script
reproduces the production "long run" pattern (chunked integration, per-chunk
netCDF + checkpoint + health report) with the Python API. Log lines mirror
the jcm runner's so the usual monitoring greps work unchanged.

Usage (one GPU per run):

    CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python tools/run_pyses_climatology.py --config 1m \
        --nx 30 --days 365 --chunk-days 10 --save-interval 5 \
        --physics-dt 1800 --nu-top 2.5e4 \
        --prefix /scr/.../ne30_rrtmgp_1m

Configs: 1m | 2m | 2m-jam (ECHAM + RRTMGP throughout).
"""

import argparse
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("PYSES_BACKEND", "jax")

import numpy as np


def build(args):
    import jax.numpy as jnp

    import jcm
    from jcm.dycore.pyses import PysesCamSEDycore, build_forcing
    from jcm.model import Model
    from jcm.physics.echam.echam_terms import echam_physics

    bc_dir = Path(jcm.__file__).resolve().parent / "data" / "bc" / "t63"

    kwargs = dict(radiation_scheme="rrtmgp")
    if args.config == "1m":
        kwargs.update(cloud_scheme="1m")
    elif args.config == "2m":
        kwargs.update(cloud_scheme="2m")
    elif args.config == "2m-jam":
        kwargs.update(cloud_scheme="2m", aerosol_module="jam")
    else:
        raise SystemExit(f"unknown --config {args.config}")

    dycore = PysesCamSEDycore(
        nx=args.nx, npt=4, dt_seconds=args.physics_dt,
        nu_top=args.nu_top, n_sponge=args.n_sponge,
        physics_dtype=jnp.float32,
        terrain_file=str(bc_dir / "terrain.nc"),
        tracer_substeps=args.tracer_split,
        dyn_substeps_per_tracer=args.dyn_split,
        hypervis_scale=args.hypervis_scale,
        coupling=args.coupling,
        hypervis=args.hypervis,
    )
    if abs(args.hines_rms - 1.0) > 1e-12:
        from jcm.physics.gravity_waves.hines import HinesParameters
        kwargs.update(hines=HinesParameters(
            rms_launch_wind=args.hines_rms))
    physics = echam_physics(**kwargs)

    # Upper-atmosphere temperature relaxation: the ~1 Pa finite lid sits far
    # outside the radiation schemes' validity and refrigerates without it
    # (the Laplacian nu_top sponge only damps horizontal structure — the
    # first ne30 attempt cooled the lid mean 187 K -> 117 K in 20 days).
    # Reference profile = USSA-1976 at the level reference mid-pressures.
    from jcm.physics.dissipation.upper_temperature_relaxation import (
        UpperTemperatureRelaxation,
    )
    from jcm.dycore.pyses.initial_states import (
        ussa_pressure, ussa_temperature,
    )
    hybrid = dycore.coords.vertical
    a = np.asarray(hybrid.a_boundaries, dtype=float)
    b = np.asarray(hybrid.b_boundaries, dtype=float)
    p_mid = 0.5 * (a[:-1] + a[1:]) + 0.5 * (b[:-1] + b[1:]) * 101325.0
    zs = np.linspace(0.0, 84000.0, 4000)
    ps = np.asarray(ussa_pressure(zs))
    z_of_p = np.interp(np.log(p_mid), np.log(ps[::-1]), zs[::-1])
    t_ref = np.asarray(ussa_temperature(z_of_p))
    uv_tau = (args.uv_sponge_hours * 3600.0
              if args.uv_sponge_hours > 0 else None)
    physics = physics + UpperTemperatureRelaxation(
        t_ref, n_levels=args.t_sponge_levels,
        timescale_s=args.t_sponge_hours * 3600.0,
        wind_timescale_s=uv_tau)
    print(f"[sponge] upper-T relaxation: top {args.t_sponge_levels} levels, "
          f"tau {args.t_sponge_hours:g} h at lid (x2.5/level), "
          f"T_ref lid {t_ref[0]:.1f} K, "
          f"uv Rayleigh tau {args.uv_sponge_hours:g} h at lid")

    model = Model(
        dycore=dycore,
        time_step=dycore.dt_seconds / 60.0,  # minutes
        physics=physics,
    )
    forcing = build_forcing(str(bc_dir / "forcing.nc"), dycore)
    # TEMPORARY boundary conditions: bilinearly downscaled from the bundled
    # T63 (192x96) climatology onto the ne{nx} columns at build time. Swap
    # for native-resolution files when available.
    print(f"[bc] terrain+forcing: bilinear downscale from T63 files in "
          f"{bc_dir} (temporary until native ne{args.nx} boundary data)")
    return dycore, model, forcing


def health_report(ds):
    """Print the runner-style NaN/range summary for a chunk dataset."""
    nan_vars = 0
    total = 0
    for name, da in ds.data_vars.items():
        total += 1
        if bool(np.isnan(da.values).any()):
            nan_vars += 1
    print(f"NaN vars:    {nan_vars}/{total}")
    if "temperature" in ds:
        t = ds["temperature"].values
        print(f"Temperature: {np.nanmin(t):.1f} - {np.nanmax(t):.1f} K "
              f"(mean {np.nanmean(t):.1f} K, NaN {np.isnan(t).mean():.1%})")
    if "specific_humidity" in ds:
        q = ds["specific_humidity"].values
        print(f"Humidity:    max {np.nanmax(q):.4g}, mean {np.nanmean(q):.4g} "
              f"(NaN {np.isnan(q).mean():.1%})")
    return nan_vars


def gpu_memory_line():
    import jax

    stats = jax.local_devices()[0].memory_stats() or {}
    peak = stats.get("peak_bytes_in_use", 0) / 2**30
    inuse = stats.get("bytes_in_use", 0) / 2**30
    return f"GPU memory: peak {peak:.2f} GiB, in use {inuse:.2f} GiB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["1m", "2m", "2m-jam"])
    ap.add_argument("--nx", type=int, default=30)
    ap.add_argument("--days", type=float, default=365.0)
    ap.add_argument("--chunk-days", type=float, default=10.0)
    ap.add_argument("--save-interval", type=float, default=5.0)
    ap.add_argument("--physics-dt", type=float, default=1800.0)
    ap.add_argument("--nu-top", type=float, default=2.5e4)
    ap.add_argument("--n-sponge", type=int, default=8)
    ap.add_argument("--tracer-split", type=int, default=9,
                    help="floor on pySES tracer subcycles per coupling "
                         "interval; pySES's own CFL sizing assumes 120 m/s "
                         "winds, which the winter polar-night jet exceeds "
                         "(day-127 blow-up of the first ne30 1m run). The "
                         "default 9 gives ~216 m/s headroom at ne30/dt=1800; "
                         "-1 restores pySES's CFL-derived count")
    ap.add_argument("--t-sponge-levels", type=int, default=8)
    ap.add_argument("--t-sponge-hours", type=float, default=6.0)
    ap.add_argument("--hines-rms", type=float, default=1.0,
                    help="Hines rms_launch_wind (m/s; documented typical "
                         "range 0.5-2, ECHAM default 1.0). The T63-tuned "
                         "default under-decelerates the ne30 winter vortex "
                         "(breakdown-season blow-ups); raising it is the "
                         "physics-side interim until the CAM frontal GW "
                         "scheme has a pySES frontogenesis provider")
    ap.add_argument("--coupling", default="lump_all",
                    choices=["lump_all", "dribble_all", "hybrid"],
                    help="physics-dynamics coupling: lump_all = one "
                         "forward-Euler kick (pySES prototype default; the "
                         "proven ne30 winter-vortex destabilizer), "
                         "dribble_all / hybrid = CAM-SE se_ftype 0 / 2 "
                         "(hybrid lumps tracers, dribbles u/v/T)")
    ap.add_argument("--hypervis", default="tensor",
                    choices=["tensor", "quasi_uniform"],
                    help="hyperviscosity family: quasi_uniform matches "
                         "CAM-SE production (divergence damped "
                         "nu_div_factor=2.5x harder - the front-killer); "
                         "tensor is the variable-resolution config with "
                         "no divergence enhancement")
    ap.add_argument("--hypervis-scale", type=float, default=0.5,
                    help="pySES ad_hoc_scale on the interior tensor "
                         "hyperviscosity (library default 0.5); raise for "
                         "under-dissipated vortex-edge fronts")
    ap.add_argument("--dyn-split", type=int, default=-1,
                    help="floor on pySES dynamics subcycles per tracer step; "
                         "-1 keeps the CFL-derived count (sized for 342 m/s "
                         "gravity waves + ~120 m/s advection). The ne30 "
                         "winter stratospheric jet (~29 hPa, 6-h means >100 "
                         "m/s) broke that margin at dt_dyn=100 s (day-146 "
                         "blow-up); 3 gives dt_dyn ~67 s")
    ap.add_argument("--uv-sponge-hours", type=float, default=12.0,
                    help="Rayleigh wind-damping timescale at the lid (h), "
                         "ramped x2.5/level over the same t-sponge levels; "
                         "<= 0 disables. Nothing else damps the MEAN wind at "
                         "the 1 Pa lid, and undamped lid jets (~100 m/s "
                         "5-day means) preceded both ne30 blow-ups")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="restore from <prefix>.ckpt and continue")
    args = ap.parse_args()

    import jax

    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(f"{prefix}.ckpt")

    t0 = time.time()
    dycore, model, forcing = build(args)
    print(f"[setup] config={args.config} ne{args.nx} L{dycore.nlev} "
          f"ncols={dycore.coords.horizontal.nodal_shape[1]} "
          f"dt={dycore.dt_seconds:.0f}s nu_top={args.nu_top:g} "
          f"tracer_split={args.tracer_split} dyn_split={args.dyn_split} "
          f"coupling={args.coupling} hypervis={args.hypervis} "
          f"({time.time() - t0:.1f}s)")

    model.bootstrap_state(None)
    day_done = 0.0
    if args.resume and ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            payload = pickle.load(f)
        # Plain pickle of the device-fetched pytrees (numpy leaves +
        # dataclass containers); flax/msgpack can't serialize the
        # tree_math structs in the physics carry.
        model._final_dycore_state = payload["dycore_state"]
        model._final_physics_state = payload["physics_state"]
        day_done = payload["day_done"]
        print(f"[resume] restored checkpoint at day {day_done:.1f}")

    total_wall = 0.0
    while day_done < args.days - 1e-6:
        chunk = min(args.chunk_days, args.days - day_done)
        t_chunk = time.time()
        preds = model.resume(
            forcing=forcing,
            save_interval=args.save_interval,
            total_time=chunk,
            output_averages=True,
        )
        ds = preds.to_xarray()
        day_done += chunk
        wall = time.time() - t_chunk
        total_wall += wall

        out_nc = f"{prefix}_day{int(round(day_done))}.nc"
        ds.to_netcdf(out_nc)
        print("Run completed.")
        nan_vars = health_report(ds)
        print(f"Saved {out_nc}")

        # NaN gate BEFORE the checkpoint write: the first ne30 campaign's 1m
        # run aborted at day 130 AFTER overwriting the only checkpoint with
        # the NaN state, losing the restartable day-120 state (outputs are
        # 5-day means, so netCDFs can't reconstruct one). Keep the previous
        # checkpoint as .prev anyway, as a second line of defence.
        if nan_vars > 0:
            print(f"[ABORT] {nan_vars} NaN variables at day {day_done:.1f} — "
                  "checkpoint NOT overwritten; restart from "
                  f"{ckpt_path} (last clean chunk).")
            raise SystemExit(2)

        if ckpt_path.exists():
            ckpt_path.replace(f"{ckpt_path}.prev")
        with open(ckpt_path, "wb") as f:
            pickle.dump({
                "day_done": day_done,
                "dycore_state": jax.device_get(model._final_dycore_state),
                "physics_state": jax.device_get(model._final_physics_state),
            }, f)
        print(f"Saved checkpoint to {ckpt_path}")
        rate = chunk / (wall / 3600.0)
        print(f"Wall: {wall:.1f}s this chunk, {total_wall:.0f}s total "
              f"({rate:.1f} sim days/hr)")
        print(gpu_memory_line(), flush=True)

    print(f"[done] {args.days:.0f} days in {total_wall / 3600.0:.2f} h "
          f"({args.days / (total_wall / 3600.0):.1f} sim days/hr overall)")


if __name__ == "__main__":
    main()
