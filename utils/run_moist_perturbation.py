"""T85x47 ICON-physics moist-perturbation stability test.

Runs the model from an isothermal-rest initial state plus a localized
1 g/kg Gaussian moisture perturbation. This is the same setup that
gave the "10 days stable, NaN day 11" baseline before the
fortran-harness branch fixes (see .claude/moist_run_debug_log.md).
We use it here as a regression test: with the convection + cloud
fixes from the harness branch, does the failure day move?

Usage:
    CUDA_VISIBLE_DEVICES=6 ~/micromamba/envs/jcm/bin/python \\
        utils/run_moist_perturbation.py --days 5 --output icon_moist_5d.nc

If the 5-day smoke test stays stable, retry with --days 30.
"""

import argparse
import logging
import sys
import time

import jax
import jax.numpy as jnp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_perturbed_initial_state(model, perturbation_amplitude_kgkg=1e-3,
                                   random_seed=0):
    """Build an initial modal state with a 1 g/kg Gaussian humidity blob.

    Mirrors ``Model._prepare_initial_modal_state(humidity_perturbation=True)``
    but lets us pick the amplitude (the hardcoded ``1e-2`` there is
    10 g/kg, which over-pumps convection and produced 260 m/s winds in
    the day-11 cascade tests; ``1e-3`` = 1 g/kg is the stable baseline
    documented in moist_run_debug_log.md).
    """
    from dinosaur import primitive_equations
    from dinosaur import primitive_equations_states
    from dinosaur.hybrid_coordinates import HybridCoordinates
    import pint
    units = pint.UnitRegistry()

    state = model.default_state_fn(jax.random.PRNGKey(random_seed))
    if not isinstance(model.coords.vertical, HybridCoordinates):
        from jcm.constants import p0
        state.log_surface_pressure = model.coords.horizontal.to_modal(
            model.coords.horizontal.to_nodal(state.log_surface_pressure)
            - jnp.log(model.physics_specs.nondimensionalize(p0 * units.pascal))
        )
    state.tracers = {
        "specific_humidity": (
            perturbation_amplitude_kgkg
            * primitive_equations_states.gaussian_scalar(
                model.coords, model.physics_specs
            )
        ),
    }
    return primitive_equations.State(**state.asdict(), sim_time=0.0)


def health_summary(ds, label):
    """Quick health check after a run completes."""
    import numpy as np

    nan_vars = [v for v in ds.data_vars if ds[v].isnull().any()]
    print(f"\n=== {label} ===")
    print(f"  NaN variables: {len(nan_vars)} / {len(ds.data_vars)}")
    if nan_vars:
        print(f"    First few: {nan_vars[:5]}")
    if "temperature" in ds:
        T = ds["temperature"].isel(time=-1).values
        print(f"  T (final): min={float(np.nanmin(T)):.1f} K  "
              f"max={float(np.nanmax(T)):.1f} K  "
              f"mean={float(np.nanmean(T)):.1f} K")
    if "specific_humidity" in ds:
        q = ds["specific_humidity"].isel(time=-1)
        # The netCDF saves q in g/kg (see ``attrs['units']``) — don't
        # multiply by 1000 again. Detect the unit attribute to be safe.
        units = q.attrs.get("units", "kg/kg")
        scale = 1.0 if units == "g/kg" else 1000.0
        qv = q.values * scale
        print(f"  q (final, g/kg): min={float(np.nanmin(qv)):.4f}  "
              f"max={float(np.nanmax(qv)):.3f}  "
              f"mean={float(np.nanmean(qv)):.3f}")
    for wind in ("u_wind", "v_wind", "u", "v"):
        if wind in ds:
            w = ds[wind].isel(time=-1).values
            print(f"  {wind} (final): max|.|={float(np.nanmax(np.abs(w))):.1f} m/s")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=5.0,
                        help="Total simulation days")
    parser.add_argument("--save_interval", type=float, default=1.0,
                        help="Save interval in days")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--amplitude_gkg", type=float, default=1.0,
                        help="Gaussian humidity perturbation amplitude (g/kg)")
    parser.add_argument("--radiation", default="grey",
                        choices=["grey", "emulated", "rrtmgp"],
                        help="grey is the cheapest; use for stability checks")
    parser.add_argument("--surface_layer_scheme", default="businger_dyer",
                        choices=["businger_dyer", "echam_louis"],
                        help="Which surface-layer exchange-coeff scheme to use")
    parser.add_argument("--terrain_file", default=None,
                        help="Path to a netCDF with orog+lsm (interpolated to "
                             "the working grid). If None, runs aquaplanet.")
    parser.add_argument("--forcing_file", default=None,
                        help="Path to a forcing.nc with stl/icec/sst/alb/"
                             "soilw_am/snowc. Required when --terrain_file "
                             "is set so land/sea-ice tiles get realistic "
                             "boundary conditions.")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"icon_t85_47_moist_{int(args.days)}d.nc"

    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

    # Use the same build_model() as run_icon_simulation.py (default
    # T85x47, hybrid coords, ICON physics, 30-min timestep).
    sys.path.insert(0, "utils")
    from run_icon_simulation import build_model
    model = build_model(
        radiation_scheme=args.radiation,
        surface_layer_scheme=args.surface_layer_scheme,
        terrain_file=args.terrain_file,
    )
    print(f"Surface-layer scheme: {args.surface_layer_scheme}")
    if args.terrain_file:
        print(f"Terrain file: {args.terrain_file}")
    print(f"Model built: timestep = {float(model.dt_si.m):.1f} s "
          f"({float(model.dt_si.m)/60:.1f} min)")

    initial_state = build_perturbed_initial_state(
        model, perturbation_amplitude_kgkg=args.amplitude_gkg * 1e-3,
    )
    print(f"Initial state: isothermal rest + {args.amplitude_gkg:.1f} g/kg "
          f"Gaussian humidity perturbation")

    forcing = None
    if args.forcing_file:
        from jcm.forcing import ForcingData
        forcing = ForcingData.from_file(args.forcing_file, coords=model.coords)
        print(f"Forcing file: {args.forcing_file}")
        print(f"  SST     range = {float(forcing.sea_surface_temperature.min()):.1f} – "
              f"{float(forcing.sea_surface_temperature.max()):.1f} K")
        print(f"  stl_am  range = {float(forcing.stl_am.min()):.1f} – "
              f"{float(forcing.stl_am.max()):.1f} K")

    t0 = time.perf_counter()
    preds = model.run(
        initial_state=initial_state,
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

    health_summary(ds, label=f"After {args.days:.0f} days")


if __name__ == "__main__":
    main()
