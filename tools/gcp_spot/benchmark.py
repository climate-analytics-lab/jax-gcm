"""Run the staged ECHAM benchmark, saving timings and checking finite outputs."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import csv
import resource
import subprocess
import threading
import time

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "jax-gcm"))
import echam_speed_test as benchmark

PHASE = "setup"


class CpuTelemetry:
    """Sample process resources; CPU percent uses 100 percent per logical CPU."""

    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.path = Path(os.environ["RESULT_DIR"])
        self.samples = []
        hardware = {
            "logical_cpus": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "lscpu": subprocess.check_output(["lscpu"], text=True),
            "memory_total_bytes": psutil.virtual_memory().total,
            "thread_settings": {key: os.environ.get(key) for key in
                                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS")},
        }
        (self.path / "cpu-hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
        self.process.cpu_percent()

        def sample():
            with (self.path / "cpu-telemetry.csv").open("w") as file:
                writer = csv.writer(file)
                writer.writerow(["unix_time", "phase", "process_cpu_percent", "rss_bytes", "threads"])
                while not self.stop.wait(1):
                    row = [time.time(), PHASE, self.process.cpu_percent(),
                           self.process.memory_info().rss, self.process.num_threads()]
                    self.samples.append(row)
                    writer.writerow(row)
                    file.flush()

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=5)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        summary = {"process_user_seconds": usage.ru_utime, "process_system_seconds": usage.ru_stime,
                   "max_rss_kib": usage.ru_maxrss, "samples": len(self.samples),
                   "cpu_percent_units": "100 percent per logical CPU; includes compiler threads"}
        (self.path / "cpu-resources.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    global PHASE
    out = Path(os.environ["RESULT_DIR"])
    versions = {dist.metadata["Name"]: dist.version
                for dist in importlib.metadata.distributions() if dist.metadata["Name"]}
    (out / "environment.json").write_text(json.dumps({
        "python": platform.python_version(),
        "packages": versions,
        "devices": [str(d) for d in jax.devices()],
        "jax_enable_x64": jax.config.jax_enable_x64,
    }, indent=2) + "\n")
    expected = os.environ.get("EXPECTED_BACKEND", "gpu")
    if jax.default_backend() != expected:
        raise RuntimeError(f"Expected {expected} backend; got {jax.default_backend()}")

    # Fail before compilation if a wheel is missing its radiation lookup data.
    from jcm.physics.radiation.rrtmgp import _ensure_rrtmgp
    _ensure_rrtmgp()

    # The timing loop blocks via this helper. Check only its final predictions,
    # after all timed runs, to avoid adding host transfers to the measurements.
    original_block = benchmark.block_until_ready
    last_predictions = None
    completed_calls = 0

    def capture(predictions):
        nonlocal last_predictions, completed_calls
        global PHASE
        last_predictions = original_block(predictions)
        completed_calls += 1
        PHASE = f"timed_repeat_{completed_calls}"
        return last_predictions

    benchmark.block_until_ready = capture
    PHASE = "warmup_and_compilation"
    try:
        results = benchmark.run_speed_test(
            total_time=float(os.environ["TOTAL_DAYS"]),
            save_interval=float(os.environ["SAVE_DAYS"]),
            n_repeats=int(os.environ["REPEATS"]),
            spectral_truncation=int(os.environ["TRUNCATION"]),
        )
    finally:
        benchmark.block_until_ready = original_block
    PHASE = "validation"
    results["expected_backend"] = expected
    results["sample_std_s"] = (float(np.std(results["all_times_s"], ddof=1))
                               if len(results["all_times_s"]) > 1 else None)
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
    if os.environ.get("EXPECTED_BACKEND") == "cpu":
        with CpuTelemetry():
            main()
    else:
        main()
