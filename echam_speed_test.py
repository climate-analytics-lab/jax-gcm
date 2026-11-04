"""ECHAM speed test: T31/L47, 15-minute steps, stock Sundqvist and RRTMGP.

Usage:
    python echam_speed_test.py [--total_time DAYS] [--save_interval DAYS] [--n_repeats N]

Runs a jcm simulation, pre-compiles, then times repeated runs.
Results are printed as JSON for easy aggregation. Uses built-in dry JW
initialization, flat aquaplanet and constant forcing; no external datasets.
"""

import argparse
import json
import platform
import time

import jax
import numpy as np

from jcm.model import Model
from jcm.forcing import ForcingData
from jcm.initial_states import jw_state
from jcm.physics.echam.echam_levels import get_echam_levels
from jcm.physics.echam.echam_terms import echam_physics
from jcm.utils import get_coords


def block_until_ready(predictions):
    """Block until all arrays in a ModelPredictions object are materialized."""
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        predictions._predictions,
    )
    return predictions


def get_device_info():
    """Gather JAX device and platform info."""
    devices = jax.devices()
    device = devices[0]
    return {
        "platform": jax.default_backend(),
        "device_kind": device.device_kind,
        "num_devices": len(devices),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "hostname": platform.node(),
    }


def run_speed_test(total_time=5.0, save_interval=1.0, n_repeats=3, spectral_truncation=31):
    """Run speed test and return results dict."""
    device_info = get_device_info()
    print(f"Device info: {json.dumps(device_info, indent=2)}")

    # --- Model setup ---
    print(f"Creating ECHAM model (T{spectral_truncation}, 47 layers, dt=15min)...")
    coords = get_coords(get_echam_levels(47), spectral_truncation=spectral_truncation)
    model = Model(coords=coords, physics=echam_physics(coords=coords), time_step=15.0)
    initial = jw_state(model, rh=0.0)
    forcing = ForcingData.zeros(coords.horizontal.nodal_shape)
    print("Model created.")

    # --- Warmup / compile ---
    print(f"Warmup run ({total_time} days, save every {save_interval} days)...")
    t0 = time.perf_counter()
    predictions = model.run(initial_state=initial, forcing=forcing,
                                save_interval=save_interval, total_time=total_time)
    block_until_ready(predictions)
    compile_time = time.perf_counter() - t0
    print(f"Warmup (includes compile): {compile_time:.2f}s")

    # --- Timed runs ---
    print(f"Running {n_repeats} timed iterations...")
    times = []
    for i in range(n_repeats):
        t0 = time.perf_counter()
        predictions = model.run(initial_state=initial, forcing=forcing,
                                save_interval=save_interval, total_time=total_time)
        block_until_ready(predictions)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Run {i + 1}/{n_repeats}: {elapsed:.3f}s")

    times = np.array(times)
    results = {
        **device_info,
        "spectral_truncation": spectral_truncation,
        "total_time_days": total_time,
        "save_interval_days": save_interval,
        "n_repeats": n_repeats,
        "compile_time_s": round(compile_time, 3),
        "mean_s": round(float(times.mean()), 3),
        "std_s": round(float(times.std()), 3),
        "min_s": round(float(times.min()), 3),
        "max_s": round(float(times.max()), 3),
        "all_times_s": [round(float(t), 3) for t in times],
    }

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="jax-gcm ECHAM speed test")
    parser.add_argument("--total_time", type=float, default=5.0, help="Simulation length in days")
    parser.add_argument("--save_interval", type=float, default=1.0, help="Save interval in days")
    parser.add_argument("--n_repeats", type=int, default=3, help="Number of timed repeats")
    parser.add_argument("--spectral_truncation", type=int, default=31, help="Horizontal truncation (e.g. 31 or 63)")
    args = parser.parse_args()

    results = run_speed_test(
        total_time=args.total_time,
        save_interval=args.save_interval,
        n_repeats=args.n_repeats,
        spectral_truncation=args.spectral_truncation,
    )
