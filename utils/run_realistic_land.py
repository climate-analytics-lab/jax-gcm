"""T85x47 ICON-physics 1-year realistic-land + orography stability test.

Mirrors run_moist_perturbation.py but with:
- Realistic terrain (orography + land-sea mask) loaded from
  jcm/data/bc/t30/clim/terrain.nc and bilinear-interpolated to T85.
- Realistic forcing (SST, sea-ice concentration, snow, soil moisture,
  land surface T) from jcm/data/bc/t30/clim/forcing.nc.
- A realistic atmospheric initial state (288 K surface, 6.5 K/km lapse,
  60 % RH below 200 hPa) instead of the isothermal-rest + moisture
  Gaussian. Compatible with the 212-314 K land surface T range without
  the day-1 ΔT shock that crashed the moist-perturbation init.

Usage::

    CUDA_VISIBLE_DEVICES=6 ~/micromamba/envs/jcm/bin/python \\
        utils/run_realistic_land.py --days 30 \\
        --surface_layer_scheme echam_louis \\
        --output icon_t85_47_realland_louis_30d.nc
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import jax

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


HERE = Path(__file__).resolve().parent
T30_BC = HERE.parent / "jcm" / "data" / "bc" / "t30" / "clim"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=30.0,
                        help="Total simulation days")
    parser.add_argument("--save_interval", type=float, default=1.0,
                        help="Save interval in days")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--radiation", default="grey",
                        choices=["grey", "emulated", "rrtmgp"])
    parser.add_argument("--surface_layer_scheme", default="businger_dyer",
                        choices=["businger_dyer", "echam_louis"])
    parser.add_argument("--terrain_file", default=str(T30_BC / "terrain.nc"))
    parser.add_argument("--forcing_file", default=str(T30_BC / "forcing.nc"))
    parser.add_argument("--time_step_min", type=float, default=15.0,
                        help="Time step in minutes (smaller for orography)")
    args = parser.parse_args()

    if args.output is None:
        args.output = (
            f"icon_t85_47_realland_{args.surface_layer_scheme}_"
            f"{int(args.days)}d.nc"
        )

    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

    sys.path.insert(0, str(HERE))
    from run_icon_simulation import build_model, inject_realistic_profile

    model = build_model(
        radiation_scheme=args.radiation,
        surface_layer_scheme=args.surface_layer_scheme,
        terrain_file=args.terrain_file,
        time_step_min=args.time_step_min,
    )
    print(f"Surface-layer scheme: {args.surface_layer_scheme}")
    print(f"Terrain file: {args.terrain_file}")
    print(f"Forcing file: {args.forcing_file}")
    print(f"Time step:    {float(model.dt_si.m):.0f} s "
          f"({float(model.dt_si.m)/60:.1f} min)")

    inject_realistic_profile(model)

    from jcm.forcing import ForcingData
    forcing = ForcingData.from_file(args.forcing_file, coords=model.coords)
    print(f"Forcing SST  range = {float(forcing.sea_surface_temperature.min()):.1f} – "
          f"{float(forcing.sea_surface_temperature.max()):.1f} K")
    print(f"Forcing stl  range = {float(forcing.stl_am.min()):.1f} – "
          f"{float(forcing.stl_am.max()):.1f} K")

    t0 = time.perf_counter()
    preds = model.resume(
        forcing=forcing,
        save_interval=args.save_interval,
        total_time=args.days,
    )
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        preds,
    )
    elapsed = time.perf_counter() - t0
    print(f"Run took {elapsed:.1f} s ({elapsed/60:.1f} min) for "
          f"{args.days:.0f} days at dt={float(model.dt_si.m):.0f} s")

    ds = preds.to_xarray()
    ds.to_netcdf(args.output)
    print(f"Saved {args.output}")

    # Reuse the moist-perturbation health summary
    from run_moist_perturbation import health_summary
    health_summary(ds, label=f"After {args.days:.0f} days")


if __name__ == "__main__":
    main()
