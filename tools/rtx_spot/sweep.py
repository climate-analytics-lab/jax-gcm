"""Orchestrate existing benchmark/profile tools; never implement a model loop."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ["RESULT_DIR"])
SCRATCH = OUT.parent / "model-output"
PLAN = json.loads((Path(__file__).with_name("plan.json")).read_text())


def arrays(directory):
    """Yield every restart array plus every named netCDF variable/coordinate."""
    import numpy as np
    import xarray as xr
    from flax.serialization import msgpack_restore

    checkpoint = msgpack_restore((directory / "final.ckpt").read_bytes())
    for group in ("dycore", "physics", "clock"):
        for name, value in sorted(checkpoint[group].items()):
            yield f"checkpoint/{group}/{name}", np.asarray(value)
    yield "checkpoint/elapsed_days", np.asarray(checkpoint["elapsed_days"])
    for path in sorted(directory.glob("state_day*.nc")):
        with xr.open_dataset(path, decode_times=False, mask_and_scale=False) as ds:
            for name in sorted(ds.variables):
                yield f"{path.name}/{name}", ds[name].values


def fingerprints(directory):
    """Hash original array bytes, preserving signed zeros and all dtype bits."""
    import numpy as np

    result = {}
    for name, array in arrays(directory):
        numeric = np.issubdtype(array.dtype, np.number)
        result[name] = {
            "shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(
                json.dumps(array.tolist(), sort_keys=True).encode() if array.dtype.hasobject
                else array.tobytes(order="C")
            ).hexdigest(),
            "finite": bool(np.isfinite(array).all()) if numeric else True,
            "elements": int(array.size),
        }
    if not any(n.startswith("state_day") for n in result):
        raise RuntimeError("No model output found; refusing an empty comparison")
    return result


def compare(reference, candidate):
    """Require the same keys, shapes, dtypes and bytes, with no nonfinite data."""
    names = sorted(set(reference) | set(candidate))
    differences = [n for n in names if reference.get(n) != candidate.get(n)]
    finite = all(v["finite"] for v in candidate.values())
    return {"bitwise_equal": not differences and finite, "finite": finite,
            "arrays": len(candidate), "different_arrays": differences,
            "elements": sum(v["elements"] for v in candidate.values())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).with_name("plan.json"), OUT / "plan.json")
    env_base = dict(os.environ, PYTHONPATH=str(ROOT), JAX_ENABLE_X64="0",
                    XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1", MAM4_JAX_ENABLE_X64="0")
    env_base.pop("XLA_FLAGS", None)
    # Attribution disables CUDA graphs; its time is never quoted as throughput.
    profile_command = [sys.executable, str(ROOT / "tools/profile_terms.py"), "--preset",
                       PLAN["preset"], "--gpu", "0", "--outdir", str(OUT / "profile"),
                       "--label", "baseline", "--extra", f"run.output={SCRATCH}/profile.nc",
                       f"run.output_prefix={SCRATCH}/profile"]
    profile_code = subprocess.run(profile_command, env=env_base, cwd=ROOT, check=False).returncode
    (OUT / "profile-status.json").write_text(json.dumps({"exit_code": profile_code, "command": profile_command}))
    reference = None
    summary = []
    for label in PLAN["sequence"]:
        variant = PLAN["variants"][label.rsplit("-", 1)[0]]
        env = dict(env_base, **variant["env"])
        data = SCRATCH / label
        data.mkdir()
        command = [sys.executable, str(ROOT / "tools/benchmark.py"),
                   "--python", sys.executable, "--preset", PLAN["preset"],
                   "--days", str(PLAN["days"]), "--chunk-days", str(PLAN["chunk_days"]),
                   "--gpu", "0", "--f32", "--outdir", str(OUT / "throughput"),
                   "--label", label, "--scratch-root", str(SCRATCH / "temporary"),
                   "--extra", f"run.output_prefix={data}/state",
                   f"++run.checkpoint_path={data}/final.ckpt", *variant["extra"]]
        print(f"START {label} {time.strftime('%FT%TZ', time.gmtime())}", flush=True)
        code = subprocess.run(command, env=env, cwd=ROOT, check=False).returncode
        report_path = OUT / "throughput" / label / "result.json"
        item = {"label": label, "exit_code": code, "command": command,
                "environment_overrides": variant["env"], "extra": variant["extra"]}
        if report_path.exists():
            item["benchmark"] = json.loads(report_path.read_text())
        if code == 0:
            fingerprint = fingerprints(data)
            (OUT / f"{label}-arrays.json").write_text(json.dumps(fingerprint, indent=2))
            if reference is None:
                if label != "baseline-a":
                    raise RuntimeError("Initial baseline did not succeed")
                reference = fingerprint
            item["numerics"] = compare(reference, fingerprint)
        elif label == "baseline-a":
            raise RuntimeError("Baseline failed; refusing candidate measurements")
        summary.append(item)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        print("RESULT", json.dumps({k: v for k, v in item.items() if k != "benchmark"}), flush=True)
        shutil.rmtree(data)

if __name__ == "__main__":
    main()
