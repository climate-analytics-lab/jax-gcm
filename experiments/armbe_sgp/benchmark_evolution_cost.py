"""Benchmark continuous SPEEDY and ECHAM full-model evolution cost."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import jax
import numpy as np
import xarray as xr
from dinosaur.scales import units

from jcm.diffusion import DiffusionFilter
from jcm.dycore.dinosaur.dycore import DinosaurDycore
from jcm.forcing import ForcingData
from jcm.model import Model
from jcm.terrain import TerrainData
from jcm.utils import get_coords, load_states_from_xarray


SCHEMES = ("speedy-t31l8", "echam-rrtmgp-t63l47")


def block_tree(tree):
    """Block until every asynchronous JAX leaf has materialized."""
    return jax.tree_util.tree_map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready")
        else value,
        tree,
    )


def _execution_tree(model: Model, predictions):
    return (
        predictions._predictions,
        model._final_dycore_state,
        model._final_physics_state,
    )


def _all_finite(tree) -> bool:
    for value in jax.tree_util.tree_leaves(tree):
        if not hasattr(value, "dtype"):
            continue
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.inexact) and not np.all(np.isfinite(array)):
            return False
    return True


def _echam_radiation_step(model: Model) -> int | None:
    carry = model._final_physics_state
    if not isinstance(carry, dict) or "radiation" not in carry:
        return None
    return int(np.asarray(carry["radiation"].step))


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
    model = Model(
        coords=coords,
        terrain=terrain,
        physics=speedy_physics(checkpoint_terms=False),
        time_step=30.0,
    )
    return model, forcing, None


def _load_echam_spinup():
    with xr.open_dataset("jcm/data/test/echam_t63l47/spinup_state.nc") as dataset:
        return load_states_from_xarray(
            dataset,
            tracer_vars={"qc": "qc", "qi": "qi"},
        )


def _build_echam(*, compute_cre: bool):
    from jcm.physics.echam.echam_levels import get_echam_levels
    from jcm.physics.echam.echam_terms import echam_physics
    from jcm.physics.radiation.rrtmgp import RRTMGPRadiation

    coords = get_coords(get_echam_levels(47), spectral_truncation=63)
    terrain = TerrainData.from_file(Path("jcm/data/bc/t63/terrain.nc"), coords=coords)
    forcing = ForcingData.from_file(Path("jcm/data/bc/t63/forcing.nc"), coords=coords)
    physics = echam_physics(
        radiation_scheme=RRTMGPRadiation(compute_cre=compute_cre),
        cloud_scheme="1m",
        checkpoint_terms=False,
    )
    tracer_specs = {spec.name: spec for spec in physics.required_tracers()}
    dycore = DinosaurDycore(
        coords=coords,
        terrain=terrain,
        dt_seconds=12.0 * 60.0,
        tracer_specs=tracer_specs,
        diffusion=DiffusionFilter.echam_t63_l47(),
    )
    model = Model(dycore=dycore, physics=physics, time_step=12.0)
    return model, forcing, _load_echam_spinup()


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


def benchmark(
    scheme: str,
    *,
    cycle_steps: int,
    repeats: int,
    compute_cre: bool,
) -> dict[str, object]:
    """Time continuously evolving, compiled full-model blocks."""
    if cycle_steps < 1 or repeats < 1:
        raise ValueError("cycle_steps and repeats must be positive")

    setup_start = time.perf_counter()
    if scheme == "speedy-t31l8":
        model, forcing, initial_state = _build_speedy()
    elif scheme == "echam-rrtmgp-t63l47":
        model, forcing, initial_state = _build_echam(compute_cre=compute_cre)
    else:
        raise ValueError(f"unknown scheme: {scheme}")
    setup_seconds = time.perf_counter() - setup_start

    dt_seconds = float(model.dt_si.to(units.second).m)
    cycle_days = float(model.dt_si.to(units.day).m) * cycle_steps
    cycle_simulated_hours = cycle_steps * dt_seconds / 3600.0

    first_start = time.perf_counter()
    predictions = model.run(
        initial_state=initial_state,
        forcing=forcing,
        save_interval=cycle_days,
        total_time=cycle_days,
        output_averages=False,
    )
    block_tree(_execution_tree(model, predictions))
    first_cycle_seconds = time.perf_counter() - first_start
    radiation_step_after_first_cycle = _echam_radiation_step(model)

    resume_warm_start = time.perf_counter()
    predictions = model.resume(
        forcing=forcing,
        save_interval=cycle_days,
        total_time=cycle_days,
        output_averages=False,
    )
    block_tree(_execution_tree(model, predictions))
    resume_warm_cycle_seconds = time.perf_counter() - resume_warm_start
    radiation_step_after_warm_cycle = _echam_radiation_step(model)

    steady_seconds = []
    for _ in range(repeats):
        run_start = time.perf_counter()
        predictions = model.resume(
            forcing=forcing,
            save_interval=cycle_days,
            total_time=cycle_days,
            output_averages=False,
        )
        block_tree(_execution_tree(model, predictions))
        steady_seconds.append(time.perf_counter() - run_start)

    final_tree = _execution_tree(model, predictions)
    final_radiation_step = _echam_radiation_step(model)
    median_seconds = float(np.median(steady_seconds))
    nlev = int(model.coords.nodal_shape[0])
    ncols = int(np.prod(model.coords.horizontal.nodal_shape))
    return {
        "scheme": scheme,
        "configuration": {
            "time_step_seconds": dt_seconds,
            "cycle_steps": cycle_steps,
            "cycle_simulated_hours": cycle_simulated_hours,
            "compute_cre": compute_cre if scheme.startswith("echam") else None,
            "echam_radiation_interval_seconds": (
                7200.0 if scheme.startswith("echam") else None
            ),
            "radiation_step_after_first_cycle": radiation_step_after_first_cycle,
            "radiation_step_after_warm_cycle": radiation_step_after_warm_cycle,
            "radiation_step_after_timed_cycles": final_radiation_step,
            "expected_echam_cycle_composition": (
                "one full RRTMGP step plus nine cached-radiation steps"
                if scheme.startswith("echam") and cycle_steps == 10
                else None
            ),
            "output": "one end-of-cycle snapshot",
        },
        "grid": {
            "levels": nlev,
            "horizontal_shape": list(model.coords.horizontal.nodal_shape),
            "columns": ncols,
            "three_dimensional_cells": nlev * ncols,
        },
        "timings_seconds": {
            "setup": setup_seconds,
            "compile_plus_first_cycle": first_cycle_seconds,
            "resume_warm_cycle": resume_warm_cycle_seconds,
            "steady_cycles": steady_seconds,
            "steady_cycle_median": median_seconds,
            "steady_cycle_mean": float(np.mean(steady_seconds)),
            "steady_cycle_min": float(np.min(steady_seconds)),
            "steady_cycle_max": float(np.max(steady_seconds)),
            "effective_step_from_cycle_median": median_seconds / cycle_steps,
            "estimated_simulated_day": median_seconds * 24.0 / cycle_simulated_hours,
        },
        "result": {
            "all_final_state_and_diagnostic_leaves_finite": _all_finite(final_tree),
            "continuously_evolved_timed_steps": repeats * cycle_steps,
        },
        "device": _device_metadata(),
        "method": (
            "Model.run compiles and advances the first cycle; one blocked resume cycle "
            "warms the continuous-carry executable before repeated timed Model.resume "
            "calls. Timings include dynamics, physics, tendency application, state "
            "update, and one saved snapshot per cycle."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=SCHEMES, required=True)
    parser.add_argument("--cycle-steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--compute-cre",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include RRTMGP's additional clear-sky solve (default: false)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = benchmark(
        args.scheme,
        cycle_steps=args.cycle_steps,
        repeats=args.repeats,
        compute_cre=args.compute_cre,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized, flush=True)


if __name__ == "__main__":
    main()
