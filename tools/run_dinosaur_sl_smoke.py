"""Smoke/positivity test for the dinosaur semi-Lagrangian dycore + JAM.

The #521 stress case: sharp CEDS emission sources on a near-zero background,
which the Eulerian spectral core turns into Gibbs ringing (negative tracers →
NaN microphysics at T63+). The semi-Lagrangian core carries every jcm tracer
NODALLY with the quasi-monotone limiter, so tracers must stay exactly
non-negative.

Run with dinosaur PR#135 on the path:

    PYTHONPATH=~/dinosaur-sl:$PWD JAX_PLATFORMS=cpu \
      python tools/run_dinosaur_sl_smoke.py --advection semi_lagrangian --steps 12

Compare --advection eulerian (with the positivity tracer_filter DISABLED)
to see the ringing this run is designed to kill.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--advection", default="semi_lagrangian",
                    choices=["semi_lagrangian", "eulerian"])
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--dt", type=float, default=900.0)
    ap.add_argument("--grid", default="t42_sigma",
                    choices=["t42_sigma", "t63_hybrid"],
                    help="t63_hybrid is the #521 stress resolution (the "
                         "Eulerian core NaN'd there); needs the T63 "
                         "emissions file and terrain")
    ap.add_argument("--spmd", default=None,
                    help="dinosaur SPMD mesh as 'x,y,z' (e.g. '2,1,1' for a "
                         "2-GPU longitude split)")
    ap.add_argument("--emissions-file", default=None)
    ap.add_argument("--off-centering", type=float, default=0.0,
                    help="SL Crank-Nicolson off-centering (IFS uses ~0.2 for "
                         "stability over orography)")
    ap.add_argument("--aquaplanet", action="store_true",
                    help="flat terrain (isolates orography from blow-ups)")
    ap.add_argument("--no-emissions", action="store_true")
    ap.add_argument("--save-nc", default=None,
                    help="write the trajectory here for offline first-NaN "
                         "forensics")
    ap.add_argument("--save-every", type=int, default=1,
                    help="steps between saved frames when --save-nc is set "
                         "(1 = every step; larger keeps the compile small)")
    args = ap.parse_args()

    import jax

    print(f"[env] devices: {jax.devices()}")
    import dinosaur
    print(f"[env] dinosaur from: {dinosaur.__file__}")

    # Pre-x64 phase, mirroring the pyses driver: flip before physics build.
    jax.config.update("jax_enable_x64", True)
    import numpy as np

    import mam4_jax.coag  # noqa: F401
    from jcm.physics.echam.echam_terms import echam_physics

    physics = echam_physics(
        radiation_scheme="rrtmgp", cloud_scheme="2m", aerosol_module="jam",
        jam_microphysics="mam4_jax", jam_anthropogenic=True,
    )

    from dinosaur.sigma_coordinates import SigmaCoordinates

    from jcm.dycore.dinosaur.dycore import DinosaurDycore
    from jcm.forcing import default_forcing, read_anthropogenic_emissions
    from jcm.model import Model
    from jcm.terrain import TerrainData
    from jcm.utils import get_coords

    spmd = tuple(int(x) for x in args.spmd.split(",")) if args.spmd else None
    runs_dir = "/glade/u/home/duncanwp/jax-gcm/runs"
    if args.grid == "t63_hybrid":
        from jcm.physics.echam.echam_levels import get_echam_levels
        coords = get_coords(get_echam_levels(47), spectral_truncation=63,
                            spmd_mesh=spmd)
        if args.aquaplanet:
            terrain = TerrainData.from_coords(coords)
        else:
            bc = Path(__file__).resolve().parent.parent / "jcm" / "data" / "bc" / "t63"
            terrain = TerrainData.from_file(str(bc / "terrain.nc"), coords)
        emissions = args.emissions_file or (
            f"{runs_dir}/emissions_echam_t63_l47_hybrid_2014.nc")
    else:
        coords = get_coords(SigmaCoordinates.equidistant(8),
                            spectral_truncation=42, spmd_mesh=spmd)
        terrain = TerrainData.from_coords(coords)
        emissions = args.emissions_file or (
            f"{runs_dir}/emissions_echam_t42_l8_sigma_2014.nc")
    args.emissions_file = emissions
    tracer_specs = {s.name: s for s in physics.required_tracers()}
    dycore = DinosaurDycore(
        coords=coords, terrain=terrain, dt_seconds=args.dt,
        tracer_specs=tracer_specs, advection=args.advection,
        sl_options={"off_centering": args.off_centering},
    )
    model = Model(dycore=dycore, time_step=args.dt / 60.0, physics=physics)

    forcing = default_forcing(coords.horizontal)
    if not args.no_emissions:
        import xarray as xr
        with xr.open_dataset(args.emissions_file) as ds:
            forcing = forcing.copy(
                anthropogenic_emissions=read_anthropogenic_emissions(ds))
    print(f"[cfg] advection={args.advection} grid={args.grid} "
          f"dt={args.dt:.0f}s steps={args.steps} spmd={spmd} "
          f"offc={args.off_centering} aqua={args.aquaplanet} "
          f"emis={not args.no_emissions}")

    model.bootstrap_state(None)
    t0 = time.time()
    step_days = args.dt / 86400.0
    # With --save-nc, save every --save-every steps so first-NaN forensics
    # can bisect in time; otherwise a single end frame.
    save_days = (args.save_every * step_days if args.save_nc
                 else args.steps * step_days)
    preds = model.resume(
        forcing=forcing, save_interval=save_days,
        total_time=args.steps * step_days, output_averages=False,
    )
    ds_out = preds.to_xarray()
    if args.save_nc:
        ds_out.to_netcdf(args.save_nc)
        print(f"[save] wrote {args.save_nc}")
    print(f"[run] {args.steps} steps in {time.time() - t0:.1f}s (incl. compile)")
    # Steady-state timing segment (already compiled).
    t1 = time.time()
    model.resume(forcing=forcing, save_interval=args.steps * step_days,
                 total_time=args.steps * step_days, output_averages=False)
    jax.block_until_ready(model._final_dycore_state)
    wall = time.time() - t1
    per_step = wall / args.steps
    print(f"[perf] steady: {per_step*1000:.0f} ms/step -> "
          f"{3600.0 / (per_step * 86400.0 / args.dt):.1f} sim days/hr")

    # The verdict: JAM tracer minima. Under SL+monotone these must be >= 0
    # (exactly); under Eulerian they go negative from ringing.
    bad = []
    checked = 0
    for name in sorted(ds_out.data_vars):
        if not (name.startswith(("m_", "n_", "g_", "mc_", "nc_"))):
            continue
        v = np.asarray(ds_out[name].values)
        checked += 1
        vmin = float(np.nanmin(v))
        has_nan = bool(np.isnan(v).any())
        if vmin < 0 or has_nan:
            bad.append((name, vmin, has_nan))
    print(f"[positivity] checked {checked} tracer fields")
    if bad:
        print(f"[positivity] {len(bad)} fields NEGATIVE or NaN:")
        for name, vmin, has_nan in bad[:10]:
            print(f"   {name}: min={vmin:.3e} nan={has_nan}")
    else:
        print("[positivity] ALL tracer fields non-negative and finite ✓")
    nan_names = [n for n, v in ds_out.data_vars.items()
                 if bool(np.isnan(v.values).any())]
    print(f"[health] NaN vars: {len(nan_names)}/{len(ds_out.data_vars)}")
    if nan_names:
        print(f"[health] NaN fields: {nan_names[:12]}")
    # Emission signal actually present?
    if "m_bc_pcm" in ds_out:
        print(f"[signal] m_bc_pcm max: {float(np.nanmax(ds_out['m_bc_pcm'].values)):.3e}")
    print("SL SMOKE DONE" if not bad and not nan_names else "SL SMOKE PROBLEMS")


if __name__ == "__main__":
    main()
