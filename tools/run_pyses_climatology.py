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
    )
    physics = echam_physics(**kwargs)
    model = Model(
        dycore=dycore,
        time_step=dycore.dt_seconds / 60.0,  # minutes
        physics=physics,
    )
    forcing = build_forcing(str(bc_dir / "forcing.nc"), dycore)
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
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="restore from <prefix>.ckpt and continue")
    args = ap.parse_args()

    from flax import serialization

    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(f"{prefix}.ckpt")

    t0 = time.time()
    dycore, model, forcing = build(args)
    print(f"[setup] config={args.config} ne{args.nx} L{dycore.nlev} "
          f"ncols={dycore.coords.horizontal.nodal_shape[1]} "
          f"dt={dycore.dt_seconds:.0f}s nu_top={args.nu_top:g} "
          f"({time.time() - t0:.1f}s)")

    model.bootstrap_state(None)
    day_done = 0.0
    if args.resume and ckpt_path.exists():
        template = (model._final_dycore_state, model._final_physics_state)
        with open(ckpt_path, "rb") as f:
            payload = pickle.load(f)
        restored = serialization.from_bytes(template, payload["state_bytes"])
        model._final_dycore_state, model._final_physics_state = restored
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

        with open(ckpt_path, "wb") as f:
            pickle.dump({
                "day_done": day_done,
                "state_bytes": serialization.to_bytes(
                    (model._final_dycore_state, model._final_physics_state)),
            }, f)
        print(f"Saved checkpoint to {ckpt_path}")
        rate = chunk / (wall / 3600.0)
        print(f"Wall: {wall:.1f}s this chunk, {total_wall:.0f}s total "
              f"({rate:.1f} sim days/hr)")
        print(gpu_memory_line(), flush=True)

        if nan_vars > 0:
            print(f"[ABORT] {nan_vars} NaN variables at day {day_done:.1f} — "
                  "stopping so the previous checkpoint stays clean.")
            raise SystemExit(2)

    print(f"[done] {args.days:.0f} days in {total_wall / 3600.0:.2f} h "
          f"({args.days / (total_wall / 3600.0):.1f} sim days/hr overall)")


if __name__ == "__main__":
    main()
