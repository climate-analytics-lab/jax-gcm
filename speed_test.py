"""Speed test for jax-gcm on different hardware architectures.

Usage:
    python speed_test.py [--total_time DAYS] [--save_interval DAYS] [--n_repeats N]

Runs a jcm simulation, pre-compiles, then times repeated runs.
Results are printed as JSON for easy aggregation.
"""

import argparse
import json
import platform
import time

import jax
import numpy as np

from jcm.model import Model
from jcm.physics.speedy.speedy_coords import get_speedy_coords


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


def run_speed_test(total_time=360.0, save_interval=30.0, n_repeats=5):
    """Run speed test and return results dict."""
    device_info = get_device_info()
    print(f"Device info: {json.dumps(device_info, indent=2)}")

    # --- Model setup ---
    print("Creating model (T31, 8 layers, dt=30min)...")
    model = Model(coords=get_speedy_coords(), time_step=30.0)
    print("Model created.")

    # --- Warmup / compile ---
    print(f"Warmup run ({total_time} days, save every {save_interval} days)...")
    t0 = time.perf_counter()
    predictions = model.run(save_interval=save_interval, total_time=total_time)
    block_until_ready(predictions)
    compile_time = time.perf_counter() - t0
    print(f"Warmup (includes compile): {compile_time:.2f}s")

    # --- Timed runs ---
    print(f"Running {n_repeats} timed iterations...")
    times = []
    for i in range(n_repeats):
        t0 = time.perf_counter()
        predictions = model.run(save_interval=save_interval, total_time=total_time)
        block_until_ready(predictions)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Run {i + 1}/{n_repeats}: {elapsed:.3f}s")

    times = np.array(times)
    results = {
        **device_info,
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
    parser = argparse.ArgumentParser(description="jax-gcm speed test")
    parser.add_argument("--total_time", type=float, default=360.0, help="Simulation length in days")
    parser.add_argument("--save_interval", type=float, default=30.0, help="Save interval in days")
    parser.add_argument("--n_repeats", type=int, default=5, help="Number of timed repeats")
    args = parser.parse_args()

    results = run_speed_test(
        total_time=args.total_time,
        save_interval=args.save_interval,
        n_repeats=args.n_repeats,
    )
