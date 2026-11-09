"""One continuous T63/L47 month, with daily diagnostics and five-day checkpoints."""

import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).parent / "jax-gcm"))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def integrate_day(model, initial, forcing, day):
    """Reset once at day one, then continue the complete model carry/clock."""
    kwargs = dict(forcing=forcing, total_time=1.0, save_interval=1.0,
                  output_averages=False)
    if day == 1:
        return model.run(initial_state=initial, **kwargs)
    return model.resume(**kwargs)


def recovery_start(report, restored_days, truncation, step):
    """Validate provenance/continuity before accepting a checkpoint as progress."""
    day = int(restored_days)
    if not 0 < day < 30 or float(day) != restored_days:
        raise ValueError("Recovery checkpoint must be at a whole day before day 30")
    if (report["simulated_days"] != day or report["total_days"] != 30
            or report["spectral_truncation"] != truncation or report["layers"] != 47
            or report["time_step_minutes"] != 15 or step != day * 96):
        raise ValueError("Recovery checkpoint/report configuration or clock mismatch")
    if len(report["chunks"]) != day or any(
            chunk["day"] != expected or not chunk["finite"]
            for expected, chunk in enumerate(report["chunks"], 1)):
        raise ValueError("Recovery report must contain every validated preceding day")
    return day + 1


def main():
    import jax
    import numpy as np
    from jcm.checkpoint import load_checkpoint, save_checkpoint
    from jcm.forcing import ForcingData
    from jcm.initial_states import jw_state
    from jcm.model import Model
    from jcm.physics.echam.echam_levels import get_echam_levels
    from jcm.physics.echam.echam_terms import echam_physics
    from jcm.utils import get_coords

    out = Path(os.environ["RESULT_DIR"])
    uri = os.environ.get("RESULT_URI")
    devices = jax.devices()
    if jax.default_backend() != "gpu":
        raise RuntimeError("CUDA GPU required")
    if float(os.environ["TOTAL_DAYS"]) != 30:
        raise ValueError("This runner integrates exactly 30 days")
    truncation = int(os.environ["TRUNCATION"])
    environment = {
        "python": platform.python_version(), "jax_enable_x64": jax.config.jax_enable_x64,
        "devices": [str(device) for device in devices],
        "device_kind": devices[0].device_kind,
        "packages": {dist.metadata["Name"]: dist.version for dist in
                     importlib.metadata.distributions() if dist.metadata["Name"]},
    }
    atomic_json(out / "environment.json", environment)
    report = dict(complete=False, simulated_days=0, total_days=30,
                  spectral_truncation=truncation, layers=47,
                  time_step_minutes=15, chunk_days=1, initialization="dry analytic JW",
                  chunks=[], warmup_includes_compile_s=None)
    atomic_json(out / "progress.json", report)

    def upload(path, name):
        if uri:
            subprocess.run(["gcloud", "--quiet", "storage", "cp", str(path),
                            f"{uri}/{name}"], check=True, timeout=180)

    def execution_tree(model, predictions):
        return predictions._predictions, model.dycore_state, model.physics_carry

    def block(tree):
        jax.tree_util.tree_map(lambda x: x.block_until_ready()
                              if hasattr(x, "block_until_ready") else x, tree)

    def finite(tree):
        checked = nonfinite = 0
        for leaf in jax.tree_util.tree_leaves(tree):
            if not hasattr(leaf, "dtype"):
                continue
            try:
                floating = np.issubdtype(np.dtype(leaf.dtype), np.inexact)
            except TypeError:
                continue  # Typed random keys and other non-numeric metadata.
            if floating:
                array = np.asarray(leaf)
                checked += array.size
                nonfinite += int(np.count_nonzero(~np.isfinite(array)))
        return dict(checked_elements=checked, nonfinite_elements=nonfinite,
                    finite=checked > 0 and nonfinite == 0)

    print(json.dumps({"phase": "setup", "device": environment["device_kind"],
                      "truncation": truncation}), flush=True)
    coords = get_coords(get_echam_levels(47), spectral_truncation=truncation)
    model = Model(coords=coords, physics=echam_physics(coords=coords), time_step=15.0)
    initial = jw_state(model, rh=0.0)
    forcing = ForcingData.zeros(coords.horizontal.nodal_shape)
    start = time.perf_counter()
    predictions = integrate_day(model, initial, forcing, 1)
    block(execution_tree(model, predictions))
    report["warmup_includes_compile_s"] = time.perf_counter() - start
    if not finite(execution_tree(model, predictions))["finite"]:
        raise RuntimeError("Nonfinite warmup output")
    del predictions
    gc.collect()
    print(json.dumps({"phase": "warmup_complete", "seconds": report["warmup_includes_compile_s"]}), flush=True)

    first_day = 1
    resume_from = os.environ.get("RESUME_FROM")
    if resume_from:
        prior = json.loads((out.parent / "resume-progress.json").read_text())
        restored_days = float(load_checkpoint(model, out.parent / "resume.msgpack"))
        first_day = recovery_start(prior, restored_days, truncation,
                                   int(np.asarray(model.run_state.step)))
        if not finite((model.dycore_state, model.physics_carry))["finite"]:
            raise RuntimeError("Nonfinite restored model state")
        prior.setdefault("recoveries", []).append({
            "from": resume_from, "resumed_at_day": restored_days,
            "warmup_includes_compile_s": report["warmup_includes_compile_s"],
        })
        report = prior
        report["complete"] = False
        report["current_attempt_start_day"] = first_day
        atomic_json(out / "progress.json", report)
        print(json.dumps({"phase": "checkpoint_restored", "next_day": first_day}), flush=True)

    production_start = time.perf_counter()
    for day in range(first_day, 31):
        start = time.perf_counter()
        predictions = integrate_day(model, initial, forcing, day)
        block(execution_tree(model, predictions))
        seconds = time.perf_counter() - start
        validation = finite(execution_tree(model, predictions))
        step = int(np.asarray(model.run_state.step))
        if step != day * 96:
            raise RuntimeError(f"Run clock discontinuity at day {day}: step {step}")
        chunk = dict(day=day, integration_s=seconds, step=step, **validation)
        report["chunks"].append(chunk)
        report["simulated_days"] = day
        atomic_json(out / "progress.json", report)
        print(json.dumps(chunk), flush=True)
        upload(out / "progress.json", f"days/{day:02d}/progress.json")
        if not validation["finite"]:
            raise RuntimeError(f"Nonfinite predictions or carried state at day {day}")
        del predictions
        gc.collect()
        if day % 5 == 0:
            save_checkpoint(model, out / "checkpoint.msgpack", elapsed_days=day)
            upload(out / "checkpoint.msgpack", f"days/{day:02d}/checkpoint.msgpack")

    report["complete"] = True
    report["production_wall_s_including_validation_and_io"] = time.perf_counter() - production_start
    report["production_wall_scope"] = f"current attempt, days {first_day}-30"
    report["integration_total_s"] = sum(chunk["integration_s"] for chunk in report["chunks"])
    report["gpu_hours_per_365_day_year"] = report["integration_total_s"] / 30 * 365 / 3600
    atomic_json(out / "progress.json", report)
    atomic_json(out / "results.json", report)
    atomic_json(out / "validation.json", {"finite": True, "days_checked": 30})
    print(json.dumps({"phase": "complete", **{key: report[key] for key in
          ("simulated_days", "integration_total_s", "gpu_hours_per_365_day_year")}}), flush=True)


if __name__ == "__main__":
    main()
