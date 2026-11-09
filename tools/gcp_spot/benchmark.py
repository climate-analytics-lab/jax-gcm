"""Run the staged ECHAM benchmark, saving timings and checking finite outputs."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "jax-gcm"))
import echam_speed_test as benchmark


def main():
    out = Path(os.environ["RESULT_DIR"])
    versions = {dist.metadata["Name"]: dist.version
                for dist in importlib.metadata.distributions() if dist.metadata["Name"]}
    (out / "environment.json").write_text(json.dumps({
        "python": platform.python_version(),
        "packages": versions,
        "devices": [str(d) for d in jax.devices()],
        "jax_enable_x64": jax.config.jax_enable_x64,
    }, indent=2) + "\n")
    if jax.default_backend() != "gpu":
        raise RuntimeError("CUDA GPU required; refusing CPU benchmark")

    # Fail before compilation if a wheel is missing its radiation lookup data.
    from jcm.physics.radiation.rrtmgp import _ensure_rrtmgp
    _ensure_rrtmgp()

    # The timing loop blocks via this helper. Check only its final predictions,
    # after all timed runs, to avoid adding host transfers to the measurements.
    original_block = benchmark.block_until_ready
    last_predictions = None

    def capture(predictions):
        nonlocal last_predictions
        last_predictions = original_block(predictions)
        return last_predictions

    benchmark.block_until_ready = capture
    try:
        results = benchmark.run_speed_test(
            total_time=float(os.environ["TOTAL_DAYS"]),
            save_interval=float(os.environ["SAVE_DAYS"]),
            n_repeats=int(os.environ["REPEATS"]),
            spectral_truncation=int(os.environ["TRUNCATION"]),
        )
    finally:
        benchmark.block_until_ready = original_block
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    nonfinite = 0
    checked = 0
    for leaf in jax.tree_util.tree_leaves(last_predictions._predictions):
        if not hasattr(leaf, "dtype"):
            continue
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.inexact):
            checked += array.size
            nonfinite += int(np.count_nonzero(~np.isfinite(array)))
    validation = {"checked_elements": checked, "nonfinite_elements": nonfinite,
                  "finite": checked > 0 and nonfinite == 0}
    (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    if not validation["finite"]:
        raise RuntimeError(f"Invalid model predictions: {validation}")


if __name__ == "__main__":
    main()
