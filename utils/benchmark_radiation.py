"""Benchmark radiation schemes: grey vs NN emulator (vs RRTMGP if available).

Profiles the full compute_tendencies call with each radiation scheme at
T85 x 40 levels (32768 columns) — the target production resolution.

Usage:
    python utils/benchmark_radiation.py [--truncation 85] [--nlev 40]
    JAX_PLATFORMS=cpu python utils/benchmark_radiation.py
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from jcm.utils import get_coords
from jcm.terrain import TerrainData
from jcm.physics_interface import PhysicsState
from jcm.forcing import ForcingData
from jcm.date import DateData


def _block(out):
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        out,
    )


def _time_fn(fn, n_warmup, n_repeats):
    for _ in range(n_warmup):
        out = fn()
        _block(out)
    times_ms = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        out = fn()
        _block(out)
        times_ms.append((time.perf_counter() - t0) * 1000)
    return times_ms


def make_icon_setup(nlev, spectral_truncation, radiation_scheme="grey"):
    """Build IconPhysics + inputs for the given radiation scheme."""
    from jcm.physics.icon.icon_physics import IconPhysics
    from jcm.physics.icon.parameters import Parameters

    sigma_boundaries = np.linspace(0, 1, nlev + 1)
    coords = get_coords(sigma_boundaries, spectral_truncation=spectral_truncation)
    terrain = TerrainData.aquaplanet(coords)

    kwargs = dict(checkpoint_terms=False, radiation_scheme=radiation_scheme)

    if radiation_scheme == "emulated":
        from jcm.physics.radiation.rrtmgp_nn import (
            init_emulator_weights,
            InputScaling,
        )
        from jcm.physics.radiation.radiation_types import RadiationParameters

        key = jax.random.key(0)
        emulator_wts = init_emulator_weights(
            sw_features=7, lw_features=7, units=16, key=key,
        )
        # Default scaling (no normalization — fine for benchmarking)
        sw_scaling = InputScaling(x_max=jnp.ones(7))
        lw_scaling = InputScaling(x_max=jnp.ones(7))

        params = Parameters.default()
        rad_params = RadiationParameters.default(
            emulator_weights=emulator_wts,
            sw_scaling=sw_scaling,
            lw_scaling=lw_scaling,
        )
        # tree_math.struct supports functional update via constructor
        params = Parameters(
            convection=params.convection,
            clouds=params.clouds,
            microphysics=params.microphysics,
            gravity_waves=params.gravity_waves,
            radiation=rad_params,
            vertical_diffusion=params.vertical_diffusion,
            surface=params.surface,
            aerosol=params.aerosol,
        )
        kwargs["parameters"] = params

    physics = IconPhysics(**kwargs)
    physics.cache_coords(coords)

    nodal = coords.horizontal.nodal_shape
    shape_3d = (nlev, nodal[0], nodal[1])
    state = PhysicsState(
        temperature=jnp.ones(shape_3d) * 280.0,
        specific_humidity=jnp.ones(shape_3d) * 0.01,
        u_wind=jnp.zeros(shape_3d),
        v_wind=jnp.zeros(shape_3d),
        geopotential=jnp.zeros(shape_3d),
        normalized_surface_pressure=jnp.ones(nodal),
        tracers={"qc": jnp.zeros(shape_3d), "qi": jnp.zeros(shape_3d)},
    )
    forcing = ForcingData.zeros(nodal)
    date = DateData.zeros()
    return physics, state, forcing, terrain, date, coords


def main():
    parser = argparse.ArgumentParser(description="Benchmark radiation schemes")
    parser.add_argument("--truncation", type=int, default=85)
    parser.add_argument("--nlev", type=int, default=40)
    parser.add_argument("--n_warmup", type=int, default=2)
    parser.add_argument("--n_repeats", type=int, default=10)
    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(f"Config: T{args.truncation} x {args.nlev} levels, "
          f"{args.n_warmup} warmup, {args.n_repeats} repeats\n")

    # Determine available schemes
    schemes = ["grey", "emulated"]
    try:
        import rrtmgp  # noqa: F401
        schemes.append("rrtmgp")
    except ImportError:
        print("Note: rrtmgp not installed, skipping RRTMGP benchmark\n")

    results = {}
    for scheme in schemes:
        print(f"--- {scheme.upper()} ---")
        print("  Setting up...")
        physics, state, forcing, terrain, date, coords = make_icon_setup(
            nlev=args.nlev, spectral_truncation=args.truncation,
            radiation_scheme=scheme,
        )
        nodal = coords.horizontal.nodal_shape
        ncols = nodal[0] * nodal[1]
        print(f"  Grid: {nodal}, {args.nlev} levels, {ncols} columns")

        @jax.jit
        def run(physics=physics, state=state, forcing=forcing,
                terrain=terrain, date=date):
            return physics.compute_tendencies(state, forcing, terrain, date)

        print("  Warmup (includes JIT)...")
        times = _time_fn(run, args.n_warmup, args.n_repeats)
        results[scheme] = times

        a = np.array(times)
        print(f"  Result: {a.mean():.2f} +/- {a.std():.2f} ms  "
              f"(min {a.min():.2f}, max {a.max():.2f})\n")

    # Summary
    print("=" * 60)
    print(f"SUMMARY  (T{args.truncation} x {args.nlev} levels, "
          f"{ncols} columns, {jax.default_backend()})")
    print("=" * 60)
    print(f"  {'Scheme':<15} {'Mean (ms)':>10} {'Std':>8} {'vs grey':>10}")
    print(f"  {'-'*43}")
    grey_mean = np.mean(results["grey"]) if "grey" in results else 1.0
    for scheme, times in results.items():
        a = np.array(times)
        ratio = f"{a.mean() / grey_mean:.2f}x"
        print(f"  {scheme:<15} {a.mean():>10.2f} {a.std():>8.2f} {ratio:>10}")


if __name__ == "__main__":
    main()
