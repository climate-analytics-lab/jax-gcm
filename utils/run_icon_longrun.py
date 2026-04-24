"""Long ICON simulation with periodic health checks.

Runs in 90-day chunks using model.resume() to avoid recompilation.
Between chunks, checks atmosphere health (NaN, moisture, temperature, fluxes)
and logs progress. Stops early if the atmosphere explodes.

Usage:
    CUDA_VISIBLE_DEVICES=6 ~/micromamba/envs/jcm/bin/python utils/run_icon_longrun.py \
        --years 3 --radiation grey --sigma
"""

import argparse
import time
import json
import logging

logging.basicConfig(level=logging.INFO)

CHUNK_DAYS = 90.0
SAVE_INTERVAL = 30.0  # monthly snapshots within each chunk


def check_health(ds, chunk_idx, elapsed_days):
    """Check atmosphere health after a chunk. Returns (ok, report_dict)."""
    import numpy as np

    report = {"chunk": chunk_idx, "elapsed_days": elapsed_days}

    # NaN count
    nan_vars = [v for v in ds.data_vars if ds[v].isnull().any()]
    report["n_nan_vars"] = len(nan_vars)
    report["n_total_vars"] = len(ds.data_vars)

    # Temperature
    T = ds["temperature"] if "temperature" in ds else None
    if T is not None:
        T_vals = T.isel(time=-1).values
        report["T_min"] = float(np.nanmin(T_vals))
        report["T_max"] = float(np.nanmax(T_vals))
        report["T_mean"] = float(np.nanmean(T_vals))
        report["T_nan_frac"] = float(np.isnan(T_vals).mean())

    # Specific humidity
    q = ds["specific_humidity"] if "specific_humidity" in ds else None
    if q is not None:
        q_vals = q.isel(time=-1).values
        report["q_max_gkg"] = float(np.nanmax(q_vals)) * 1000
        report["q_mean_gkg"] = float(np.nanmean(q_vals)) * 1000
        report["q_nan_frac"] = float(np.isnan(q_vals).mean())

    # Radiation fluxes (last snapshot)
    for key in ["radiation.toa_lw_up", "radiation.surface_lw_down",
                "radiation.toa_sw_down", "radiation.toa_sw_up"]:
        if key in ds:
            vals = ds[key].isel(time=-1).values
            short = key.split(".")[-1]
            report[f"{short}_mean"] = float(np.nanmean(vals))

    # Surface temperature
    sfc_t_key = next((v for v in ds.data_vars if "surface_temperature" in v
                      and "tendency" not in v), None)
    if sfc_t_key:
        sfc = ds[sfc_t_key].isel(time=-1).values
        report["sfc_T_min"] = float(np.nanmin(sfc))
        report["sfc_T_max"] = float(np.nanmax(sfc))
        report["sfc_T_mean"] = float(np.nanmean(sfc))

    # Convection
    if "convection.precip_conv" in ds:
        precip = ds["convection.precip_conv"].isel(time=-1).values
        report["precip_conv_mean_mmday"] = float(np.nanmean(precip)) * 86400
        report["precip_conv_max_mmday"] = float(np.nanmax(precip)) * 86400

    # Health checks
    ok = True
    reasons = []

    if report.get("T_nan_frac", 0) > 0.1:
        ok = False
        reasons.append(f"T NaN fraction {report['T_nan_frac']:.1%}")
    if report.get("T_min", 200) < 100:
        ok = False
        reasons.append(f"T_min={report['T_min']:.0f}K (< 100K)")
    if report.get("T_max", 300) > 500:
        ok = False
        reasons.append(f"T_max={report['T_max']:.0f}K (> 500K)")
    if report.get("q_max_gkg", 0) > 100:
        ok = False
        reasons.append(f"q_max={report['q_max_gkg']:.1f} g/kg (> 100)")

    report["ok"] = ok
    report["reasons"] = reasons
    return ok, report


def print_report(report):
    """Pretty-print a health check report."""
    days = report["elapsed_days"]
    years = days / 365.25
    print(f"\n{'='*60}")
    print(f"  Chunk {report['chunk']} | Day {days:.0f} ({years:.2f} yr) | "
          f"{'OK' if report['ok'] else 'FAILED: ' + '; '.join(report['reasons'])}")
    print(f"{'='*60}")
    print(f"  NaN vars:    {report['n_nan_vars']}/{report['n_total_vars']}")
    if "T_min" in report:
        print(f"  Temperature: {report['T_min']:.1f} - {report['T_max']:.1f} K "
              f"(mean {report['T_mean']:.1f} K, NaN {report.get('T_nan_frac',0):.1%})")
    if "q_max_gkg" in report:
        print(f"  Humidity:    max {report['q_max_gkg']:.2f} g/kg, "
              f"mean {report['q_mean_gkg']:.4f} g/kg (NaN {report.get('q_nan_frac',0):.1%})")
    if "sfc_T_mean" in report:
        print(f"  Surface T:   {report['sfc_T_min']:.1f} - {report['sfc_T_max']:.1f} K "
              f"(mean {report['sfc_T_mean']:.1f} K)")
    if "toa_lw_up_mean" in report:
        print(f"  TOA LW up:   {report['toa_lw_up_mean']:.1f} W/m²")
    if "surface_lw_down_mean" in report:
        print(f"  SFC LW down: {report['surface_lw_down_mean']:.1f} W/m²")
    if "toa_sw_up_mean" in report:
        print(f"  TOA SW up:   {report['toa_sw_up_mean']:.1f} W/m²")
    if "precip_conv_mean_mmday" in report:
        print(f"  Precip conv: mean {report['precip_conv_mean_mmday']:.4f} mm/day, "
              f"max {report['precip_conv_max_mmday']:.2f} mm/day")
    print()


def main():
    parser = argparse.ArgumentParser(description="Long ICON run with health monitoring")
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--radiation", default="grey", choices=["grey", "emulated"])
    parser.add_argument("--sigma", action="store_true",
                        help="Use equidistant sigma levels")
    parser.add_argument("--output", default=None)
    parser.add_argument("--chunk_days", type=float, default=CHUNK_DAYS)
    parser.add_argument("--save_interval", type=float, default=SAVE_INTERVAL)
    parser.add_argument("--dt_min", type=float, default=30.0,
                        help="Timestep in minutes (default 30). Try 10-12 to avoid CFL issues.")
    parser.add_argument("--jw_ref_temp", action="store_true",
                        help="Use Jablonowski-Williamson lapse-rate reference T "
                             "(instead of isothermal 288K) in semi-implicit scheme")
    parser.add_argument("--diffusion_scale", type=float, default=1.0,
                        help="Multiply default SPEEDY diffusion timescales by this factor. "
                             "Values < 1 = stronger diffusion. Default 1.0 (SPEEDY tuning).")
    parser.add_argument("--sponge_levels", type=int, default=0,
                        help="Number of top levels covered by the upper sponge (0 = off).")
    parser.add_argument("--sponge_timescale_h", type=float, default=3.0,
                        help="Rayleigh timescale at TOA for the upper sponge (hours).")
    parser.add_argument("--sponge_enspodi", type=float, default=2.0,
                        help="Softening factor per level away from TOA for the sponge.")
    args = parser.parse_args()

    total_days = args.years * 365.25
    n_chunks = int(total_days / args.chunk_days) + 1
    if args.output is None:
        args.output = f"icon_t85_47lev_{args.radiation}_{args.years:.0f}yr"

    import jax
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

    # Build model
    import sys
    sys.path.insert(0, "utils")
    from run_icon_simulation import build_model
    model = build_model(radiation_scheme=args.radiation, use_sigma=args.sigma,
                        time_step_min=args.dt_min, jw_ref_temp=args.jw_ref_temp,
                        diffusion_scale=args.diffusion_scale,
                        sponge_levels=args.sponge_levels,
                        sponge_timescale_h=args.sponge_timescale_h,
                        sponge_enspodi=args.sponge_enspodi)
    print(f"Timestep: {args.dt_min:.1f} min ({args.dt_min*60:.0f} s)")

    # Initial run (triggers JIT compilation)
    print(f"\n=== Starting {args.years:.0f}-year run: {n_chunks} chunks of "
          f"{args.chunk_days:.0f} days ===")
    print(f"Save interval: {args.save_interval:.0f} days")

    all_reports = []
    total_elapsed = 0.0
    elapsed_sim_days = 0.0

    for i in range(n_chunks):
        remaining = total_days - elapsed_sim_days
        chunk_days = min(args.chunk_days, remaining)
        if chunk_days <= 0:
            break

        t0 = time.perf_counter()
        if i == 0:
            preds = model.run(
                save_interval=args.save_interval, total_time=chunk_days,
            )
        else:
            preds = model.resume(
                save_interval=args.save_interval, total_time=chunk_days,
            )

        # Block until done
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            preds._predictions,
        )
        chunk_elapsed = time.perf_counter() - t0
        total_elapsed += chunk_elapsed
        elapsed_sim_days += chunk_days

        ds = preds.to_xarray()

        # Health check
        ok, report = check_health(ds, i, elapsed_sim_days)
        report["wall_seconds"] = chunk_elapsed
        all_reports.append(report)
        print_report(report)

        # Save checkpoint every chunk
        nc_path = f"{args.output}_day{int(elapsed_sim_days)}.nc"
        ds.to_netcdf(nc_path)
        print(f"  Saved {nc_path}")

        if not ok:
            print(f"\n*** STOPPING: atmosphere unhealthy at day {elapsed_sim_days:.0f} ***")
            break

        sdph = elapsed_sim_days / (total_elapsed / 3600)
        print(f"  Wall time: {chunk_elapsed:.1f}s this chunk, "
              f"{total_elapsed:.0f}s total ({sdph:.0f} sim days/hr)")

    # Summary
    print(f"\n{'='*60}")
    print(f"  COMPLETED: {elapsed_sim_days:.0f}/{total_days:.0f} days "
          f"in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}")

    # Save reports
    report_path = f"{args.output}_reports.json"
    with open(report_path, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"Saved health reports to {report_path}")

    # Plot final climatology if we have data
    if elapsed_sim_days > 30:
        try:
            from run_icon_simulation import plot_climatology
            # Load the last checkpoint
            import xarray as xr
            last_nc = f"{args.output}_day{int(elapsed_sim_days)}.nc"
            ds_final = xr.open_dataset(last_nc)
            plot_climatology(ds_final, output_prefix=args.output)
        except Exception as e:
            print(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
