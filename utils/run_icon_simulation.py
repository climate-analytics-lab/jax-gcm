"""Run ICON simulation at T85 x 47 levels with NN radiation emulator.

Usage:
    # 1-day smoke test
    CUDA_VISIBLE_DEVICES=6 python utils/run_icon_simulation.py --mode smoke

    # 30-day full run
    CUDA_VISIBLE_DEVICES=6 python utils/run_icon_simulation.py --mode full

    # Plot results from saved netcdf
    python utils/run_icon_simulation.py --mode plot --input icon_t85_47lev_30day.nc
"""

import argparse
import time
import logging

logging.basicConfig(level=logging.INFO)


def build_model(radiation_scheme="emulated", use_sigma=False, time_step_min=30.0,
                jw_ref_temp=False, diffusion_scale=1.0, sponge_levels=0,
                sponge_timescale_h=3.0, sponge_enspodi=2.0,
                nlev=47, spectral_truncation=85,
                surface_layer_scheme="businger_dyer",
                terrain_file=None):
    """Construct ICON Model with the specified resolution and radiation scheme."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from jcm.model import Model
    from jcm.utils import get_coords
    from jcm.physics.icon.icon_levels import get_icon_levels
    from jcm.physics.icon.icon_terms import icon_physics
    from jcm.physics.icon.parameters import Parameters
    from jcm.physics.radiation.radiation_types import RadiationParameters
    from jcm.physics.radiation.rrtmgp_nn import (
        init_emulator_weights,
        InputScaling,
    )

    if use_sigma:
        from dinosaur.sigma_coordinates import SigmaCoordinates
        vertical = SigmaCoordinates.equidistant(nlev)
        print(f"Using {nlev} equidistant sigma levels")
    else:
        # Native ICON hybrid coordinates (a + b*P_s): thick lower layers and
        # thinner upper layers should help CFL vs equidistant sigma at T85.
        vertical = get_icon_levels(nlev)
        print(f"Using {nlev} ICON hybrid coordinates")
    coords = get_coords(vertical, spectral_truncation=spectral_truncation)
    print(f"Grid: {coords.horizontal.nodal_shape}, "
          f"{coords.nodal_shape[0]} levels, "
          f"{coords.horizontal.nodal_shape[0] * coords.horizontal.nodal_shape[1]} columns")

    # Surface-layer scheme override applied to the vdiff parameters.
    from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (
        VDiffParameters,
    )
    base_params = Parameters.default()
    vdiff_params_override = VDiffParameters.default(
        surface_layer_scheme=surface_layer_scheme,
    )

    if radiation_scheme == "emulated":
        # NN emulator radiation with random weights
        key = jax.random.key(0)
        emulator_wts = init_emulator_weights(
            sw_features=7, lw_features=7, units=16, key=key,
        )
        sw_scaling = InputScaling(x_max=jnp.ones(7))
        lw_scaling = InputScaling(x_max=jnp.ones(7))

        rad_params = RadiationParameters.default(
            emulator_weights=emulator_wts,
            sw_scaling=sw_scaling,
            lw_scaling=lw_scaling,
        )
        params = Parameters(
            radiation=rad_params,
            convection=base_params.convection,
            clouds=base_params.clouds,
            microphysics=base_params.microphysics,
            gravity_waves=base_params.gravity_waves,
            vertical_diffusion=vdiff_params_override,
            surface=base_params.surface,
            aerosol=base_params.aerosol,
        )
    else:
        # Even non-emulated runs need to pick up the scheme override
        params = Parameters(
            radiation=base_params.radiation,
            convection=base_params.convection,
            clouds=base_params.clouds,
            microphysics=base_params.microphysics,
            gravity_waves=base_params.gravity_waves,
            vertical_diffusion=vdiff_params_override,
            surface=base_params.surface,
            aerosol=base_params.aerosol,
        )

    physics = icon_physics(
        parameters=params,
        radiation_scheme=radiation_scheme,
        checkpoint_terms=False,
    )

    # Optional upper sponge layer — Rayleigh drag on (u,v) in the top N
    # levels, ECHAM-style. Strongly stabilises T85 hybrid runs; leave off
    # for sigma unless climatology still shows sporadic extreme events.
    if sponge_levels > 0:
        from jcm.physics.dissipation import UpperSponge
        physics = physics + UpperSponge(
            n_sponge_levels=sponge_levels,
            sponge_timescale_s=sponge_timescale_h * 3600.0,
            enspodi=sponge_enspodi,
        )
        print(f"Upper sponge: {sponge_levels} top levels, "
              f"tau_top={sponge_timescale_h:.1f}h, enspodi={sponge_enspodi}")

    # Diffusion: use uniform-order SPEEDY defaults (temp 24h, vor_q 12h,
    # div 2h, del²). The level-dependent ``echam_t85_l47`` profile (del² at
    # TOA → del⁸ in the troposphere) destabilises moist runs at T85×47 —
    # a 30-day default-state test on dt=3 min NaN'd at day 12 with the
    # ECHAM profile but lasted to day 26 with the uniform default. The
    # level-dep path also has a known JIT bug at order ≥ 4 (see
    # ``jcm.diffusion`` module docstring). Stick with default until the
    # level-dep filter is debugged and re-tuned.
    from jcm.diffusion import DiffusionFilter
    base_diff = DiffusionFilter.default()
    diffusion = DiffusionFilter(
        div_timescale=base_diff.div_timescale * diffusion_scale,
        div_order=base_diff.div_order,
        vor_q_timescale=base_diff.vor_q_timescale * diffusion_scale,
        vor_q_order=base_diff.vor_q_order,
        temp_timescale=base_diff.temp_timescale * diffusion_scale,
        temp_order=base_diff.temp_order,
    )
    if diffusion_scale != 1.0:
        print(f"Diffusion timescales scaled by {diffusion_scale}x "
              f"(div {diffusion.div_timescale/3600:.1f}h, "
              f"vor_q {diffusion.vor_q_timescale/3600:.1f}h, "
              f"temp {diffusion.temp_timescale/3600:.1f}h)")

    # Optional realistic terrain (orography + land-sea mask) loaded from a
    # bilinear-interpolated boundary-condition file. Default behaviour is
    # the original aquaplanet (TerrainData.aquaplanet inside Model).
    if terrain_file is not None:
        from jcm.terrain import TerrainData
        terrain = TerrainData.from_coords(
            coords, terrain_file=terrain_file, interpolate=True,
        )
        print(f"Terrain: orog max={float(terrain.orog.max()):.0f} m, "
              f"fmask sum={float(terrain.fmask.sum()):.0f} cells, "
              f"lfluxland={bool(terrain.lfluxland)}")
        model = Model(coords=coords, physics=physics, time_step=time_step_min,
                      terrain=terrain, diffusion=diffusion,
                      log_level=logging.INFO)
    else:
        model = Model(coords=coords, physics=physics, time_step=time_step_min,
                      diffusion=diffusion, log_level=logging.INFO)

    if jw_ref_temp:
        # Swap the semi-implicit reference temperature from isothermal 288K to
        # a JW-style lapse-rate profile (210K top → 288K surface). This keeps
        # the semi-implicit scheme's linearization close to the actual
        # atmospheric state, avoiding instability when the stratosphere cools.
        #
        # dinosaur's steady_state_jw doesn't behave well with HybridCoordinates
        # (the eta->sigma transform produces NaN in the analytical solution),
        # so build the 1D lapse-rate profile directly from the sigma-equivalent
        # centers of the vertical grid.
        from dinosaur import primitive_equations
        from dinosaur.hybrid_coordinates import HybridCoordinates
        p0_pa = 101325.0
        if isinstance(coords.vertical, HybridCoordinates):
            sigma_centers = np.array(coords.vertical.get_sigma_centers(p0_pa))
        else:
            sigma_centers = np.array(coords.vertical.centers)
        # Approximate hypsometric height at each level (T=288 isothermal)
        z = 8400.0 * np.log(1.0 / np.clip(sigma_centers, 1e-4, 1.0))
        # Standard lapse rate up to 12 km, isothermal above
        T_sfc, gamma = 288.0, 6.5e-3
        T_K = np.maximum(T_sfc - gamma * z, 210.0)
        # Nondimensionalize for dinosaur's PrimitiveEquations
        # (keep as numpy — dinosaur calls np.unique on T_ref internally)
        from dinosaur.scales import units
        jw_ref_T = np.array([
            float(model.physics_specs.nondimensionalize(t * units.kelvin)) for t in T_K
        ], dtype=np.float32)
        print(f"Using lapse-rate reference T profile: {float(T_K.min()):.1f} - "
              f"{float(T_K.max()):.1f} K (replaces 288 K isothermal)")
        if isinstance(coords.vertical, HybridCoordinates):
            model.primitive = primitive_equations.PrimitiveEquationsHybrid(
                reference_temperature=jw_ref_T,
                orography=model.truncated_orography,
                coords=coords,
                physics_specs=model.physics_specs,
                hpa_quantity=units.pascal,
                humidity_key='specific_humidity',
            )
        else:
            model.primitive = primitive_equations.PrimitiveEquations(
                reference_temperature=jw_ref_T,
                orography=model.truncated_orography,
                coords=coords,
                physics_specs=model.physics_specs,
                humidity_key='specific_humidity',
            )

    print(f"Model created. Timestep: {model.dt_si}")
    return model


def inject_realistic_profile(model):
    """Set up model with realistic T + humidity profile, using internal state path.

    Modifies model._final_modal_state in-place after the default isothermal
    init, then calls model.resume(). This ensures the exact same code path as
    the default — avoids NaN from the State-based model.run path.
    """
    import jax.numpy as jnp
    from dinosaur.hybrid_coordinates import HybridCoordinates

    # First, trigger the default state setup
    model._final_modal_state = model._prepare_initial_modal_state(
        physics_state=None, random_seed=0
    )
    state = model._final_modal_state

    nlon, nlat = model.coords.horizontal.nodal_shape
    p0_pa = 101325.0
    if isinstance(model.coords.vertical, HybridCoordinates):
        sigma = jnp.asarray(model.coords.vertical.get_sigma_centers(p0_pa))
    else:
        sigma = jnp.asarray(model.coords.vertical.centers)
    nlev = sigma.size

    # Standard-atmosphere T(sigma)
    p = sigma * p0_pa
    T_sfc = 288.0
    gamma = 6.5e-3
    z = 8400.0 * jnp.log(p0_pa / p)
    # Cap minimum T to stay within ~40K of reference (288K) for semi-implicit stability
    T_profile = jnp.maximum(T_sfc - gamma * z, 250.0)

    # If the model has nonzero orography, the default isothermal-rest
    # init (uniform log_surface_pressure) creates a hydrostatic
    # mismatch — the model thinks there's air below ground level on
    # tall mountains, producing instant ageostrophic spin-up that
    # NaN's the run. Adjust log_surface_pressure to the standard-
    # atmosphere hydrostatic balance ``P_s(x,y) = p0·exp(-g·h/(R·T))``
    # before injecting the T/q profile.
    orog = jnp.asarray(model.terrain.orog)   # (nlon, nlat) in m
    if jnp.any(orog > 1.0):
        from dinosaur.scales import units
        Rd, grav, T_ref_avg = 287.04, 9.80665, 260.0
        ps_pa_nodal = p0_pa * jnp.exp(-grav * orog / (Rd * T_ref_avg))
        # log_surface_pressure stored as ln(P_s in Pa) for hybrid coords.
        # Nondimensionalise via physics_specs to match dycore convention.
        # ``physics_specs.nondimensionalize`` is a scalar function, but we
        # can apply it on the array as a single per-unit scale factor.
        scale = float(model.physics_specs.nondimensionalize(1.0 * units.pascal))
        ps_nodal = ps_pa_nodal * scale
        log_ps_nodal = jnp.log(ps_nodal)
        state.log_surface_pressure = model.coords.horizontal.to_modal(
            log_ps_nodal[None, ...]   # add level dim
        )
        print(f"Hydrostatic-adjusted P_s: "
              f"{float(ps_pa_nodal.min()):.0f} – "
              f"{float(ps_pa_nodal.max()):.0f} Pa "
              f"(orography max {float(orog.max()):.0f} m)")

    # Inject T profile as temperature_variation on top of model ref temperature
    T_ref = jnp.asarray(model.primitive.reference_temperature)
    T_var_profile = T_profile - T_ref
    T_var_nodal = jnp.broadcast_to(
        T_var_profile[:, None, None], (nlev, nlon, nlat)
    ).astype(state.temperature_variation.dtype)
    state.temperature_variation = model.coords.horizontal.to_modal(T_var_nodal)

    # Humidity: 60% RH below 200 hPa
    es = 611.2 * jnp.exp(17.67 * (T_profile - 273.15) / (T_profile - 29.65))
    q_sat = 0.622 * es / jnp.maximum(p - es, 1.0)
    rh = jnp.where(p > 20000.0, 0.6, 0.0)
    q_profile = jnp.clip(rh * q_sat, 1e-8, 0.03)
    q_dtype = state.tracers["specific_humidity"].dtype
    q_nodal = jnp.broadcast_to(
        q_profile[:, None, None], (nlev, nlon, nlat)
    ).astype(q_dtype)
    state.tracers = {"specific_humidity": model.coords.horizontal.to_modal(q_nodal)}

    model._final_modal_state = state
    print(f"Injected realistic T profile: surface={T_sfc}K, top={float(T_profile[0]):.1f}K")
    print(f"Moisture: max q={float(q_profile.max())*1000:.1f} g/kg")


def run_smoke_test(model, total_days=1.0, save_interval=1.0, label="SMOKE TEST",
                   use_resume=False):
    """Run a short simulation to verify configuration."""
    import jax

    print(f"\n=== {label} ({total_days} day(s), save every {save_interval*24:.2f}h) ===")
    t0 = time.perf_counter()
    if use_resume:
        predictions = model.resume(save_interval=save_interval, total_time=total_days)
    else:
        predictions = model.run(save_interval=save_interval, total_time=total_days)
    # Block until done
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        predictions._predictions,
    )
    elapsed = time.perf_counter() - t0
    print(f"1-day run completed in {elapsed:.1f}s (includes JIT compilation)")

    ds = predictions.to_xarray()
    print(f"\nDataset variables ({len(ds.data_vars)}):")
    for name in sorted(ds.data_vars):
        arr = ds[name]
        has_nan = bool(arr.isnull().any())
        print(f"  {name:40s} {str(arr.shape):30s} NaN={has_nan}")

    # Check key fields
    temp = ds["temperature"] if "temperature" in ds else None
    if temp is not None:
        print(f"\nTemperature range: {float(temp.min()):.1f} - {float(temp.max()):.1f} K")

    n_nan_vars = sum(1 for v in ds.data_vars if ds[v].isnull().any())
    print(f"\nVariables with NaN: {n_nan_vars}/{len(ds.data_vars)}")
    return ds


def run_full(model, total_days=30.0, output_path="icon_t85_47lev_30day.nc",
             save_interval=None):
    """Run full simulation with periodic snapshot output."""
    import jax

    if save_interval is None:
        save_interval = 1.0 if total_days <= 30 else 30.0

    print(f"\n=== {total_days:.0f}-DAY FULL RUN ===")
    print(f"Output: every {save_interval:.0f} days -> {output_path}")

    t0 = time.perf_counter()
    predictions = model.run(
        save_interval=save_interval,
        total_time=total_days,
        output_averages=False,
    )
    # Block until done
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        predictions._predictions,
    )
    elapsed = time.perf_counter() - t0

    simulated_days_per_hour = total_days / (elapsed / 3600)
    print(f"Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Simulated days per wall-clock hour: {simulated_days_per_hour:.1f}")

    ds = predictions.to_xarray()
    ds.to_netcdf(output_path)
    print(f"Saved to {output_path}")
    return ds, elapsed


def plot_climatology(ds, output_prefix="icon_t85_47lev"):
    """Plot time-mean maps of precipitation, TOA radiation, and surface temperature."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Get lat/lon — dataset uses (time, level, lon, lat) ordering
    if "lon" in ds.coords:
        lon = ds.lon.values
        lat = ds.lat.values
    else:
        lon = np.arange(ds.sizes.get("longitude", ds.sizes.get("lon", 1)))
        lat = np.arange(ds.sizes.get("latitude", ds.sizes.get("lat", 1)))

    def _to_latlon(field):
        """Squeeze and transpose (lon, lat) -> (lat, lon) for pcolormesh."""
        v = field.values.squeeze()
        if v.ndim == 2 and v.shape == (len(lon), len(lat)):
            return v.T
        return v

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={"projection": None})

    # 1. Precipitation
    ax = axes[0]
    precip_vars = [v for v in ["convection.precip_conv", "clouds.precip_rain", "clouds.precip_snow"]
                   if v in ds]
    if precip_vars:
        total_precip = sum(ds[v].mean(dim="time") for v in precip_vars)
        # Convert to mm/day (from kg/m2/s): multiply by 86400
        precip_mm = total_precip * 86400
        im = ax.pcolormesh(lon, lat, _to_latlon(precip_mm), cmap="Blues", shading="nearest")
        plt.colorbar(im, ax=ax, label="mm/day")
        ax.set_title(f"Precipitation ({', '.join(precip_vars)})")
    else:
        ax.text(0.5, 0.5, "No precip variables found\n" + str(list(ds.data_vars)[:10]),
                transform=ax.transAxes, ha="center", fontsize=8)
        ax.set_title("Precipitation (not found)")

    # 2. TOA radiation balance
    ax = axes[1]
    sw_down_key = next((v for v in ds.data_vars if "toa_sw_down" in v), None)
    sw_up_key = next((v for v in ds.data_vars if "toa_sw_up" in v), None)
    lw_up_key = next((v for v in ds.data_vars if "toa_lw_up" in v), None)
    if sw_down_key and sw_up_key and lw_up_key:
        net_toa = (ds[sw_down_key] - ds[sw_up_key] - ds[lw_up_key]).mean(dim="time")
        im = ax.pcolormesh(lon, lat, _to_latlon(net_toa), cmap="RdBu_r",
                          shading="nearest", vmin=-150, vmax=150)
        plt.colorbar(im, ax=ax, label="W/m²")
        ax.set_title("Net TOA Radiation")
    else:
        available = [v for v in ds.data_vars if "toa" in v or "radiation" in v.lower()]
        ax.text(0.5, 0.5, f"TOA vars found: {available}", transform=ax.transAxes,
                ha="center", fontsize=8, wrap=True)
        ax.set_title("Net TOA (not found)")

    # 3. Surface temperature
    ax = axes[2]
    sfc_temp_key = next((v for v in ds.data_vars if "surface_temperature" in v
                        and "tendency" not in v), None)
    if sfc_temp_key:
        sfc_t = ds[sfc_temp_key].mean(dim="time")
        im = ax.pcolormesh(lon, lat, _to_latlon(sfc_t), cmap="RdYlBu_r",
                          shading="nearest")
        plt.colorbar(im, ax=ax, label="K")
        ax.set_title("Surface Temperature")
    else:
        available = [v for v in ds.data_vars if "surface" in v or "temp" in v.lower()]
        ax.text(0.5, 0.5, f"Surface temp vars: {available}", transform=ax.transAxes,
                ha="center", fontsize=8, wrap=True)
        ax.set_title("Surface Temp (not found)")

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.tight_layout()
    out_path = f"{output_prefix}_climatology.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved climatology plot to {out_path}")
    plt.close()


def plot_diurnal_radiation(ds, output_prefix="icon_diurnal"):
    """Plot diurnal cycle of SW/LW TOA + surface fluxes.

    Produces:
      - Equator (lat≈0) zonal mean time series of key fluxes
      - Hovmöller (time vs lon) of TOA SW down and net TOA
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def find(*keys):
        for k in keys:
            for v in ds.data_vars:
                if all(p in v for p in k.split("+")):
                    return v
        return None

    sw_down = find("toa_sw_down")
    sw_up = find("toa_sw_up")
    lw_up = find("toa_lw_up")
    sfc_lw_down = find("surface+lw+down", "sfc_lw_down", "lw_down")
    sfc_sw_down = find("surface+sw+down", "sfc_sw_down", "sw_down")

    print("Detected flux vars:",
          {k: v for k, v in [("toa_sw_down", sw_down), ("toa_sw_up", sw_up),
                             ("toa_lw_up", lw_up), ("sfc_lw_down", sfc_lw_down),
                             ("sfc_sw_down", sfc_sw_down)]})

    time = ds.time.values
    lon = ds.lon.values if "lon" in ds.coords else np.arange(ds.sizes.get("lon", 1))
    lat = ds.lat.values if "lat" in ds.coords else np.arange(ds.sizes.get("lat", 1))

    # Panel 1: global-mean time series of each flux
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for name, var in [("TOA SW down", sw_down), ("TOA SW up", sw_up),
                      ("TOA LW up", lw_up), ("SFC LW down", sfc_lw_down),
                      ("SFC SW down", sfc_sw_down)]:
        if var is None:
            continue
        arr = ds[var]
        # cosine-lat weighted global mean
        lat_rad = np.deg2rad(lat) if lat.max() > 3.2 else lat
        w = np.cos(lat_rad)
        # find lat axis
        dims = arr.dims
        lat_dim = next((d for d in dims if "lat" in d), None)
        lon_dim = next((d for d in dims if "lon" in d), None)
        if lat_dim and lon_dim:
            gm = arr.weighted(
                arr[lat_dim].copy(data=w) if lat_dim in arr.coords else
                ds[lat_dim].copy(data=w)
            ).mean(dim=[lat_dim, lon_dim]) if False else \
                (arr * w).sum(dim=lat_dim) / w.sum() if lat_dim else arr
            gm = gm.mean(dim=lon_dim) if lon_dim in gm.dims else gm
        else:
            gm = arr.mean()
        ax.plot(time, np.asarray(gm).squeeze(), label=name)
    ax.set_xlabel("Time")
    ax.set_ylabel("W/m²")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Global-mean radiative fluxes (diurnal cycle)")

    # Panel 2: Hovmöller of TOA SW down (time vs lon) at equator
    ax = axes[1]
    if sw_down is not None:
        arr = ds[sw_down]
        lat_dim = next((d for d in arr.dims if "lat" in d), None)
        if lat_dim is not None:
            ieq = int(np.argmin(np.abs(lat)))
            arr_eq = arr.isel({lat_dim: ieq}).squeeze()
            lon_dim = next((d for d in arr_eq.dims if "lon" in d), None)
            data = arr_eq.transpose("time", lon_dim).values if lon_dim else arr_eq.values
            im = ax.pcolormesh(lon, time, data, cmap="inferno", shading="nearest")
            plt.colorbar(im, ax=ax, label="W/m²")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Time")
            ax.set_title("TOA SW down at equator (Hovmöller)")

    plt.tight_layout()
    out = f"{output_prefix}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved diurnal plot to {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run ICON T85x47 simulation")
    parser.add_argument("--mode", choices=["smoke", "full", "plot", "diurnal"], default="smoke")
    parser.add_argument("--initial", choices=["isothermal", "jw"], default="isothermal",
                        help="Initial condition: isothermal rest or Jablonowski-Williamson")
    parser.add_argument("--days", type=float, default=30.0, help="Total days for full run")
    parser.add_argument("--input", type=str, default=None, help="NetCDF file for plot mode")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--radiation", type=str, default="emulated",
                        choices=["grey", "emulated", "rrtmgp"])
    parser.add_argument("--save_interval", type=float, default=None,
                        help="Days between saves (default: 1 for <=30 days, 30 for longer)")
    parser.add_argument("--sigma", action="store_true",
                        help="Use equidistant sigma levels instead of ICON hybrid levels")
    args = parser.parse_args()
    if args.output is None:
        args.output = f"icon_t85_47lev_{args.radiation}_{int(args.days)}day.nc"

    if args.mode == "plot":
        import xarray as xr
        path = args.input or args.output
        print(f"Loading {path}...")
        ds = xr.open_dataset(path)
        plot_climatology(ds)
        return

    import jax
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

    model = build_model(radiation_scheme=args.radiation, use_sigma=args.sigma)

    use_resume = False
    if args.initial == "jw":
        inject_realistic_profile(model)
        use_resume = True

    if args.mode == "smoke":
        ds = run_smoke_test(model, use_resume=use_resume)
    elif args.mode == "diurnal":
        # 1-day run, hourly output — to inspect diurnal cycle of radiation
        out = args.output.replace(".nc", "_diurnal.nc")
        ds = run_smoke_test(model, total_days=1.0,
                            save_interval=1.0/24.0, label="DIURNAL CYCLE",
                            use_resume=use_resume)
        ds.to_netcdf(out)
        print(f"Saved diurnal output to {out}")
        plot_diurnal_radiation(ds, output_prefix=out.replace(".nc", ""))
    elif args.mode == "full":
        ds, elapsed = run_full(model, total_days=args.days, output_path=args.output,
                               save_interval=args.save_interval)
        plot_climatology(ds)


if __name__ == "__main__":
    main()
