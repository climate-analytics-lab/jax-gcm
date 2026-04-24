"""Moisture / convection diagnostic probe.

Runs a T85 x 47 hybrid aquaplanet with full ICON physics, a pre-moistened
initial state (so convection ignites in ~5 days instead of ~20), and saves
a rich set of diagnostics (precip_conv, precip_rain, precip_snow,
evaporation, latent_heat_flux, tke, pbl_height, TOA/SFC radiation).

Usage:
    CUDA_VISIBLE_DEVICES=5 python utils/run_moisture_diag.py \
        --total_days 30 --save_interval 1 --output moisture_probe
"""

import argparse
import logging
import time
import sys

logging.basicConfig(level=logging.INFO)


def build_moist_initial_state(model, q_surface_gkg=10.0, scale_height_km=2.0,
                              rh_cap=0.9):
    """Build an initial PhysicsState with a realistic moist tropical profile.

    q(k) = min(q_surface * exp(-z(k) / H_q), rh_cap * q_sat(p, T))

    Capping at ``rh_cap * q_sat`` avoids supersaturated initial states when
    the prescribed profile overshoots saturation at cold levels. JW surface
    T is ~288 K so saturation q at surface is only ~10 g/kg — 15 g/kg
    surface humidity supersaturates and immediately NaNs the vdiff scheme.

    Args:
        model: Model instance (already constructed; gives coords, physics).
        q_surface_gkg: Target surface specific humidity in g/kg.
        scale_height_km: e-folding height for humidity decay with altitude.
        rh_cap: Maximum relative humidity allowed in the profile.

    Returns:
        PhysicsState suitable for ``model.run(initial_state=...)``.

    """
    import jax.numpy as jnp
    import numpy as np
    from jcm.physics_interface import dynamics_state_to_physics_state

    # Get the default initial modal state (isothermal 288 K or JW profile).
    modal_state = model._prepare_initial_modal_state(None, random_seed=0)
    ps = dynamics_state_to_physics_state(modal_state, model.primitive)

    # Altitude profile from geopotential.
    grav = 9.81
    height = np.asarray(ps.geopotential) / grav  # meters
    H_q_m = scale_height_km * 1000.0
    # NOTE: PhysicsState.specific_humidity is stored in g/kg in this
    # codebase (see ``dynamics_state_to_physics_state``:
    #   q = physics_specs.dimensionalize(q, units.gram / units.kilogram).m
    # and the inverse conversion in physics_state_to_dynamics_state).
    # So q_profile stays in g/kg throughout; saturation cap is applied
    # in kg/kg then scaled up at the end.
    q_surf_kgkg = q_surface_gkg * 1e-3  # kg/kg for sat comparison
    q_exp_kgkg = q_surf_kgkg * np.exp(-np.abs(height) / H_q_m)

    # Saturation cap: Tetens formula (water only; good enough for init).
    T = np.asarray(ps.temperature)
    p_pa = np.asarray(ps.normalized_surface_pressure) * 1e5
    from dinosaur.hybrid_coordinates import HybridCoordinates
    v = model.coords.vertical
    if isinstance(v, HybridCoordinates):
        a = np.asarray(v.a_centers)
        b = np.asarray(v.b_centers)
        p_level = a[:, None, None] + b[:, None, None] * p_pa[None, :, :]
    else:
        sigma = np.asarray(v.centers)
        p_level = sigma[:, None, None] * p_pa[None, :, :]

    e_sat = 610.78 * np.exp(17.27 * (T - 273.15) / (T - 35.86))
    eps_q = 0.622
    q_sat_kgkg = eps_q * e_sat / (p_level - (1 - eps_q) * e_sat)

    q_profile_kgkg = np.minimum(q_exp_kgkg, rh_cap * q_sat_kgkg)
    q_profile_kgkg = np.clip(q_profile_kgkg, 1e-8, 0.03)
    # PhysicsState expects g/kg.
    q_profile_gkg = q_profile_kgkg * 1000.0

    new_ps = ps.copy(specific_humidity=jnp.asarray(q_profile_gkg, dtype=jnp.float32))
    return new_ps


def main():
    parser = argparse.ArgumentParser(description="Moisture-convection diagnostic probe")
    parser.add_argument("--total_days", type=float, default=30.0)
    parser.add_argument("--save_interval", type=float, default=1.0,
                        help="Save snapshots every N days (default 1).")
    parser.add_argument("--dt_min", type=float, default=3.0)
    parser.add_argument("--output", default="moisture_probe")
    parser.add_argument("--q_surface_gkg", type=float, default=15.0,
                        help="Initial surface specific humidity (g/kg).")
    parser.add_argument("--sponge_levels", type=int, default=5)
    parser.add_argument("--sponge_timescale_h", type=float, default=3.0)
    parser.add_argument("--sigma", action="store_true", help="Use equidistant sigma levels")
    args = parser.parse_args()

    sys.path.insert(0, "utils")
    from run_icon_simulation import build_model

    model = build_model(
        radiation_scheme="grey",
        use_sigma=args.sigma,
        time_step_min=args.dt_min,
        jw_ref_temp=True,
        sponge_levels=args.sponge_levels,
        sponge_timescale_h=args.sponge_timescale_h,
    )
    print(f"Timestep: {args.dt_min:.1f} min")
    print(f"Initial q profile: {args.q_surface_gkg:.1f} g/kg surface, exponential decay")

    # Initialise model with the moist state.
    initial_ps = build_moist_initial_state(
        model, q_surface_gkg=args.q_surface_gkg,
    )

    t0 = time.perf_counter()
    preds = model.run(
        initial_state=initial_ps,
        save_interval=args.save_interval,
        total_time=args.total_days,
    )
    import jax
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        preds._predictions,
    )
    wall = time.perf_counter() - t0
    print(f"Wall time: {wall:.0f} s")

    ds = preds.to_xarray()
    print(f"Output variables: {sorted(ds.data_vars)}")
    ds.to_netcdf(f"{args.output}.nc")
    print(f"Saved {args.output}.nc ({ds.nbytes / 1e6:.0f} MB)")

    # Quick summary of moisture budget — all terms reduced to a scalar per
    # time before indexing, so the ``float(...)`` call always gets a 0-d
    # array regardless of the variable's native shape.
    def _gmean(da):
        return da.mean(dim=[d for d in da.dims if d != "time"])

    if "surface.evaporation" in ds and "clouds.precip_rain" in ds:
        evap = _gmean(ds["surface.evaporation"])
        precip_rain = _gmean(ds["clouds.precip_rain"])
        precip_snow = _gmean(ds["clouds.precip_snow"]) if "clouds.precip_snow" in ds else None
        precip_conv = _gmean(ds["convection.precip_conv"]) if "convection.precip_conv" in ds else None
        print("\nGlobal mean moisture budget (kg/m²/s):")
        print(f"  {'day':>5} {'evap':>10} {'precip_rain':>12} "
              f"{'precip_snow':>12} {'precip_conv':>12} {'E-P':>10}")
        for i, _ in enumerate(ds.time.values):
            day = float(i) * args.save_interval
            e = float(evap.isel(time=i))
            pr = float(precip_rain.isel(time=i))
            ps_ = float(precip_snow.isel(time=i)) if precip_snow is not None else 0.0
            pc = float(precip_conv.isel(time=i)) if precip_conv is not None else 0.0
            ep = e - pr - ps_ - pc
            print(f"  {day:>5.1f} {e:>10.2e} {pr:>12.2e} {ps_:>12.2e} {pc:>12.2e} {ep:>10.2e}")


if __name__ == "__main__":
    main()
