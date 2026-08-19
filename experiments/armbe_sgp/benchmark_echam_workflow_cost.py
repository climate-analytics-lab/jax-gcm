"""Benchmark prescribed-state physics cost for SPEEDY and ECHAM T63L47."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import xarray as xr

from jcm.date import DateData
from jcm.forcing import ForcingData
from jcm.physics_interface import PhysicsState
from jcm.terrain import TerrainData
from jcm.utils import get_coords, load_states_from_xarray


SCHEMES = ("speedy-t31l8", "echam-grey-t63l47", "echam-rrtmgp-t63l47")


def block_tree(tree):
    """Block until every asynchronous JAX leaf has materialized."""
    return jax.tree_util.tree_map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready")
        else value,
        tree,
    )


def _speedy_state(coords) -> PhysicsState:
    """Return a deterministic, physically plausible full-grid SPEEDY state."""
    nlev = coords.nodal_shape[0]
    horizontal_shape = coords.horizontal.nodal_shape
    shape = (nlev, *horizontal_shape)

    def profile(top, bottom):
        values = jnp.linspace(top, bottom, nlev).reshape((nlev, 1, 1))
        return jnp.broadcast_to(values, shape)

    return PhysicsState(
        u_wind=jnp.full(shape, 5.0),
        v_wind=jnp.zeros(shape),
        temperature=profile(220.0, 290.0),
        specific_humidity=profile(0.01, 10.0),
        geopotential=profile(50_000.0, 0.0),
        normalized_surface_pressure=jnp.ones(horizontal_shape),
    )


def _build_speedy():
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.physics.speedy.speedy_terms import speedy_physics

    coords = get_speedy_coords(layers=8, spectral_truncation=31)
    terrain = TerrainData.from_file(
        Path("jcm/data/bc/t30/clim/terrain.nc"), coords=coords
    )
    forcing = ForcingData.from_file(
        Path("jcm/data/bc/t30/clim/forcing.nc"), coords=coords
    )
    return coords, terrain, forcing, speedy_physics(checkpoint_terms=False), _speedy_state(coords)


def _build_echam(*, rrtmgp: bool, compute_cre: bool):
    from jcm.physics.echam.echam_levels import get_echam_levels
    from jcm.physics.echam.echam_terms import echam_physics

    coords = get_coords(get_echam_levels(47), spectral_truncation=63)
    terrain = TerrainData.from_file(Path("jcm/data/bc/t63/terrain.nc"), coords=coords)
    forcing = ForcingData.from_file(Path("jcm/data/bc/t63/forcing.nc"), coords=coords)
    with xr.open_dataset("jcm/data/test/echam_t63l47/spinup_state.nc") as dataset:
        state = load_states_from_xarray(dataset, tracer_vars={"qc": "qc", "qi": "qi"})

    if rrtmgp:
        from jcm.physics.radiation.rrtmgp import RRTMGPRadiation

        radiation = RRTMGPRadiation(compute_cre=compute_cre)
    else:
        radiation = "grey"
    physics = echam_physics(
        radiation_scheme=radiation,
        cloud_scheme="1m",
        checkpoint_terms=False,
    )
    return coords, terrain, forcing, physics, state


def _device_metadata() -> dict[str, object]:
    device = jax.devices()[0]
    return {
        "backend": jax.default_backend(),
        "device_kind": device.device_kind,
        "device_count_visible": len(jax.devices()),
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def benchmark(scheme: str, *, repeats: int, compute_cre: bool) -> dict[str, object]:
    """Time one compiled independent full-grid physics evaluation."""
    if repeats < 1:
        raise ValueError("repeats must be positive")

    setup_start = time.perf_counter()
    if scheme == "speedy-t31l8":
        coords, terrain, forcing, physics, state = _build_speedy()
        dt_seconds = 1800.0
    elif scheme == "echam-grey-t63l47":
        coords, terrain, forcing, physics, state = _build_echam(
            rrtmgp=False, compute_cre=False
        )
        dt_seconds = 720.0
    elif scheme == "echam-rrtmgp-t63l47":
        coords, terrain, forcing, physics, state = _build_echam(
            rrtmgp=True, compute_cre=compute_cre
        )
        dt_seconds = 720.0
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    physics.cache_coords(coords)
    physics.dt_seconds = dt_seconds
    date = DateData.set_date(
        model_time=jdt.to_datetime("2019-01-07"),
        model_step=jnp.int32(0),
        dt_seconds=dt_seconds,
    )
    forcing_now = forcing.select(date)
    block_tree((state, terrain, forcing_now))
    setup_seconds = time.perf_counter() - setup_start

    @jax.jit
    def evaluate(current_state):
        return physics.compute_tendencies(current_state, forcing_now, terrain)

    first_start = time.perf_counter()
    result = evaluate(state)
    block_tree(result)
    first_call_seconds = time.perf_counter() - first_start

    steady_seconds = []
    for _ in range(repeats):
        run_start = time.perf_counter()
        result = evaluate(state)
        block_tree(result)
        steady_seconds.append(time.perf_counter() - run_start)

    leaves = [np.asarray(value) for value in jax.tree_util.tree_leaves(result)]
    all_finite = all(np.all(np.isfinite(value)) for value in leaves)
    output_bytes = sum(value.nbytes for value in leaves)
    nlev = int(coords.nodal_shape[0])
    ncols = int(np.prod(coords.horizontal.nodal_shape))
    median_seconds = float(np.median(steady_seconds))
    return {
        "scheme": scheme,
        "cloud_microphysics": "ECHAM 1-moment" if scheme.startswith("echam") else "SPEEDY diagnostic",
        "radiation": (
            f"RRTMGP all-sky plus clear-sky CRE={compute_cre}"
            if scheme == "echam-rrtmgp-t63l47"
            else "ECHAM grey two-stream"
            if scheme == "echam-grey-t63l47"
            else "SPEEDY shortwave/longwave"
        ),
        "grid": {
            "levels": nlev,
            "horizontal_shape": list(coords.horizontal.nodal_shape),
            "columns": ncols,
            "three_dimensional_cells": nlev * ncols,
        },
        "timings_seconds": {
            "setup_and_host_to_device": setup_seconds,
            "compile_plus_first_call": first_call_seconds,
            "steady_calls": steady_seconds,
            "steady_median": median_seconds,
            "steady_mean": float(np.mean(steady_seconds)),
            "steady_min": float(np.min(steady_seconds)),
            "steady_max": float(np.max(steady_seconds)),
        },
        "derived_cost": {
            "full_grid_states_per_hour_serial": 3600.0 / median_seconds,
            "estimated_hours_for_960_states_serial": 960.0 * median_seconds / 3600.0,
            "three_dimensional_cells_per_second": nlev * ncols / median_seconds,
        },
        "result": {
            "all_finite": bool(all_finite),
            "materialized_output_megabytes": output_bytes / 1.0e6,
        },
        "device": _device_metadata(),
        "method": (
            "Reusable JIT around one independent full-grid physics call; repeated "
            "calls use the same state and force all output leaves to materialize. "
            "The 960-state estimate assumes serial calls and excludes ERA5 remapping."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=SCHEMES, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--compute-cre",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include RRTMGP's additional clear-sky solve (default: false)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = benchmark(args.scheme, repeats=args.repeats, compute_cre=args.compute_cre)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized, flush=True)


if __name__ == "__main__":
    main()
