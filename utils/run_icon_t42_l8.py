"""T42 × 8 sigma ICON baseline — smoke test before swapping in SPEEDY terms.

Smaller resolution: 128×64 grid, 8 equidistant sigma levels. Default state,
no perturbation, no sponge, uniform del² diffusion, dt=30 min. Goal: see
whether full ICON physics is stable at this scale, and if not, when it
blows up. The result tells us whether the day-12 NaN at T85×47 is a
resolution-specific issue or a fundamental physics-coupling problem.

Usage:
    CUDA_VISIBLE_DEVICES=6 python utils/run_icon_t42_l8.py [--days 30]
"""
import argparse
import sys
import time
import logging

logging.basicConfig(level=logging.INFO)


def _make_speedy_in_icon_adapter(speedy_term):
    """Wrap a SPEEDY term so it can run in ICON's column-vectorized pipeline.

    Two impedance mismatches to bridge:

    1. Column vectorization: ``ComposableIconPhysics`` reshapes state to
       ``(nlev, ncols)`` before iterating terms; SPEEDY terms expect the
       full 3-D ``(nlev, nlon, nlat)`` form. The adapter reshapes the
       state in, calls SPEEDY, reshapes the tendencies out.

    2. Diagnostics namespace clash: SPEEDY and ICON both store sub-structs
       under ``_<name>`` keys (e.g. ``_convection``) with disjoint typed
       schemas. Letting SPEEDY read whatever ICON wrote there crashes
       (different attributes). The adapter calls the SPEEDY term with a
       *clean* diagnostics dict and discards SPEEDY's writes — the
       wrapped term acts as a black box from ICON's side.

    The black-box treatment means SPEEDY internal coupling (e.g.
    SpeedyVerticalDiffusion reading ``data.convection.se``) won't work
    unless the upstream SPEEDY term is also swapped in. For the swap
    bisection that's fine: each test only swaps one term.
    """
    import jax.numpy as jnp
    from flax import nnx
    from jcm.physics.physics_term import PhysicsTerm
    from jcm.physics_interface import PhysicsState, PhysicsTendency

    class SpeedyInIconAdapter(PhysicsTerm):
        name = f"adapted_{speedy_term.name}"
        category = speedy_term.category

        def __init__(self):
            self._inner = speedy_term
            self._coords_cached = False

        def cache_coords(self, coords):
            self._inner.cache_coords(coords)
            self._nlev, self._nlon, self._nlat = coords.nodal_shape
            self._ncols = self._nlon * self._nlat
            self._coords_cached = True

        def __call__(self, state, diagnostics, forcing, terrain):
            def to_3d(x):
                return x.reshape(self._nlev, self._nlon, self._nlat)

            def to_2d(x):
                return x.reshape(self._nlev, self._ncols)

            def reshape_2d_field(x):
                if hasattr(x, "ndim") and x.ndim == 2 and x.shape == (
                    self._nlev, self._ncols,
                ):
                    return to_3d(x)
                if hasattr(x, "ndim") and x.ndim == 1 and x.shape == (
                    self._ncols,
                ):
                    return x.reshape(self._nlon, self._nlat)
                return x

            state_3d = PhysicsState(
                u_wind=reshape_2d_field(state.u_wind),
                v_wind=reshape_2d_field(state.v_wind),
                temperature=reshape_2d_field(state.temperature),
                specific_humidity=reshape_2d_field(state.specific_humidity),
                geopotential=reshape_2d_field(state.geopotential),
                normalized_surface_pressure=reshape_2d_field(
                    state.normalized_surface_pressure
                ),
                tracers={k: reshape_2d_field(v) for k, v in state.tracers.items()},
            )
            # Pass _date through; otherwise use a clean dict so SPEEDY's
            # _data_from_diagnostics doesn't choke on ICON-typed sub-structs.
            isolated = {}
            if "_date" in diagnostics:
                isolated["_date"] = diagnostics["_date"]
            tend_3d, _ = self._inner(state_3d, isolated, forcing, terrain)

            def reshape_tend_field(x):
                if hasattr(x, "ndim") and x.ndim == 3:
                    return to_2d(x)
                return x

            tend_2d = PhysicsTendency(
                u_wind=reshape_tend_field(tend_3d.u_wind),
                v_wind=reshape_tend_field(tend_3d.v_wind),
                temperature=reshape_tend_field(tend_3d.temperature),
                specific_humidity=reshape_tend_field(tend_3d.specific_humidity),
                tracers={
                    k: reshape_tend_field(v) for k, v in tend_3d.tracers.items()
                },
            )
            return tend_2d, diagnostics

    return SpeedyInIconAdapter()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=30.0)
    parser.add_argument("--dt_min", type=float, default=30.0)
    parser.add_argument("--radiation", default="grey", choices=["grey", "emulated"])
    parser.add_argument("--save_interval", type=float, default=1.0)
    parser.add_argument("--output", default="icon_t42_l8_baseline")
    parser.add_argument(
        "--swap",
        default=None,
        choices=[None, "convection", "clouds", "vertical_diffusion"],
        help="Replace one ICON term with the SPEEDY equivalent.",
    )
    parser.add_argument(
        "--use_speedy",
        action="store_true",
        help="Use the full SPEEDY physics package instead of ICON. Implies "
             "no ICON term swaps.",
    )
    parser.add_argument(
        "--remove",
        default=None,
        choices=[None, "convection", "clouds", "vertical_diffusion",
                 "gravity_waves", "surface", "radiation"],
        help="Remove one ICON term (no replacement) to test if it's the "
             "destabiliser.",
    )
    args = parser.parse_args()

    sys.path.insert(0, "/data/dwatsonparris/jax-gcm/utils")
    import jax
    import numpy as np
    from run_icon_simulation import build_model

    physics_label = "SPEEDY" if args.use_speedy else "ICON"
    print(f"T42 x 8 sigma {physics_label} baseline ({args.days} days, dt={args.dt_min}min)"
          + (f", swapped term={args.swap}" if args.swap else ""))
    if args.use_speedy:
        from jcm.model import Model
        from jcm.utils import get_coords
        from dinosaur.sigma_coordinates import SigmaCoordinates
        from jcm.physics.speedy.speedy_terms import speedy_physics
        from jcm.diffusion import DiffusionFilter
        vertical = SigmaCoordinates.equidistant(8)
        coords = get_coords(vertical, spectral_truncation=42)
        physics = speedy_physics()
        model = Model(coords=coords, physics=physics, time_step=args.dt_min,
                      diffusion=DiffusionFilter.default(),
                      log_level=logging.INFO)
        print(f"Grid: {coords.horizontal.nodal_shape}, "
              f"{coords.nodal_shape[0]} levels")
        print(f"Model created. Timestep: {model.dt_si}")
    else:
        model = build_model(
            radiation_scheme=args.radiation,
            use_sigma=True,
            time_step_min=args.dt_min,
            jw_ref_temp=False,
            sponge_levels=0,
            nlev=8,
            spectral_truncation=42,
        )

    if args.remove is not None and not args.use_speedy:
        model.physics = model.physics.remove(args.remove)
        print(f"Removed category={args.remove!r}")

    if args.swap is not None and not args.use_speedy:
        if args.swap == "convection":
            from jcm.physics.speedy.speedy_terms import SpeedyConvection
            speedy_term = SpeedyConvection()
        elif args.swap == "clouds":
            # SPEEDY's cloud-side analog of ICON's clouds-and-microphysics is
            # large-scale condensation (the term that produces precipitation
            # from supersaturation). The diagnostic SpeedyClouds is purely a
            # radiation input and doesn't touch q/T tendencies.
            from jcm.physics.speedy.speedy_terms import SpeedyLargeScaleCondensation
            speedy_term = SpeedyLargeScaleCondensation()
        elif args.swap == "vertical_diffusion":
            from jcm.physics.speedy.speedy_terms import SpeedyVerticalDiffusion
            speedy_term = SpeedyVerticalDiffusion()
        else:
            raise ValueError(args.swap)
        adapter = _make_speedy_in_icon_adapter(speedy_term)
        model.physics = model.physics.replace(args.swap, adapter)
        # The replacement term was constructed after the Model __init__
        # called cache_coords on every term, so it has no coords cached.
        # Re-run cache_coords on the new physics composition to fix.
        model.physics.cache_coords(model.coords)
        print(f"Replaced category={args.swap!r} with {speedy_term.__class__.__name__}")

    t0 = time.perf_counter()
    preds = model.run(save_interval=args.save_interval, total_time=args.days)
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        preds._predictions,
    )
    wall = time.perf_counter() - t0
    print(f"Wall: {wall:.0f} s ({args.days / (wall / 86400):.0f} sim-days/sec)")

    ds = preds.to_xarray()
    ds.to_netcdf(f"{args.output}.nc")
    print(f"Saved {args.output}.nc")

    print(f'\n{"day":>4} {"|u|max":>8} {"Tmin":>7} {"Tmax":>7} '
          f'{"qmax":>8} {"qmean":>8} {"qmin":>10} {"prc":>10}')
    for tt in range(len(ds.time)):
        u = ds["u_wind"].isel(time=tt).values
        if np.any(np.isnan(u)):
            print(f"{tt:>4} NaN")
            continue
        T = ds["temperature"].isel(time=tt).values
        q = ds["specific_humidity"].isel(time=tt).values
        precip_keys = [k for k in ds.data_vars
                       if k in ("convection.precip_conv", "clouds.precip_rain")]
        if precip_keys:
            pc = sum(ds[k].isel(time=tt).values for k in precip_keys)
            prc_str = f"{np.nanmax(pc) * 86400:>10.4f}"
        else:
            prc_str = f"{'-':>10}"
        print(f"{tt:>4} {np.nanmax(np.abs(u)):>8.2f} "
              f"{np.nanmin(T):>7.2f} {np.nanmax(T):>7.2f} "
              f"{np.nanmax(q):>8.4f} {np.nanmean(q):>8.4f} "
              f"{np.nanmin(q):>10.6f} {prc_str}")


if __name__ == "__main__":
    main()
