r"""Reproducible jcm throughput benchmark.

A thin wrapper around ``python -m jcm.main``: it builds a Hydra command line,
runs it, samples GPU telemetry alongside, and parses the per-chunk wall times
into a throughput number with an explicit convergence criterion. It does
**not** reimplement any part of the run loop — chunking, health gates and
checkpointing all stay in ``jcm.runners`` (see the "No bespoke run scripts"
rule in CLAUDE.md).

Why this exists rather than eyeballing the log:

* jcm's ``N sim days/hr`` log line is **cumulative including compile time**.
  Reading it as the throughput understates a run by 2-5x early on, and it
  keeps drifting upward for the whole run. A 5.3x "regression" was once filed
  against jax-rrtmgp on the strength of chunk 1 of a run that settled 22x
  faster. This tool ignores that line entirely and uses ``Wall: Xs this
  chunk``.
* Chunk times settle over 3-4 chunks (XLA autotuning, cache warming, host
  allocator). This tool requires two consecutive chunks agreeing within a
  tolerance before it reports a number, and says so when they never do.
* **High GPU utilisation does not prove steady state.** XLA's autotuner keeps
  the device at 95%+ while still picking kernels, so "the GPU is busy" is not
  evidence that a chunk time is converged. Only chunk-to-chunk agreement is.

Usage::

    python tools/benchmark.py --preset t63-echam-rrtmgp --months 1 --gpu 1
    python tools/benchmark.py --preset t63-echam-jam --months 12 --gpu 3 \\
        --label jam-baseline --pythonpath /path/to/lib/worktree

Results land in ``<outdir>/<label>/`` as ``report.md``, ``result.json``,
``run.log`` and ``gpu.csv``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_PY = "/home/dwatsonparris/micromamba/envs/jcm/bin/python"
DEFAULT_OUTDIR = pathlib.Path("/scr/dwatsonparris/benchmarks")

# Chunk-to-chunk agreement required before a rate is called converged.
CONVERGENCE_TOL = 0.03      # 3 % between consecutive chunks
GPU_SAMPLE_SECONDS = 10.0

_WALL_RE = re.compile(r"Wall:\s*([0-9.]+)s this chunk")
_NAN_RE = re.compile(r"NaN vars:\s*(\d+)\s*/\s*(\d+)")
_SAVED_RE = re.compile(r"Saved .*_day(\d+)\.nc")

# Validated stable configurations. Each is the *known-good* override set for
# that grid: an isothermal cold start with no sponge goes NaN within days at
# L47, so these are not interchangeable with a bare `grid=` override.
_T63_COMMON = [
    "grid=echam_t63_l47_hybrid",
    "init=jw", "init.rh=0.0",
    "terrain=from_file", f"terrain.file={REPO}/jcm/data/bc/t63/terrain.nc",
    "forcing=from_file", f"forcing.file={REPO}/jcm/data/bc/t63/forcing.nc",
    ("forcing.ozone_file="
     f"{REPO}/jcm/data/bc/T63L47_ozone_picontrol_latflip.nc"),
    "run=longrun",
    "run.time_step=12",
    "run.sponge.levels=10", "run.sponge.timescale_h=1.5",
    "run.sponge.enspodi=2.0", "+run.sponge.target_T_K=270",
]

PRESETS: dict[str, list[str]] = {
    "t63-echam-rrtmgp": ["physics=echam-rrtmgp", *_T63_COMMON],
    "t63-echam-rrtmgp-2m": ["physics=echam-rrtmgp-2m", *_T63_COMMON],
    "t63-echam-jam": ["physics=echam-jam", *_T63_COMMON],
    "t63-echam-jam-aerocom": ["physics=echam-jam-aerocom", *_T63_COMMON],
    "t63-echam-jam-aerocom-optics": [
        "physics=echam-jam-aerocom-optics", *_T63_COMMON],
}


def _gpu_sampler(gpu: int, out_path: pathlib.Path, stop: threading.Event):
    """Sample memory/utilisation until ``stop`` is set."""
    query = ("--query-gpu=timestamp,memory.used,memory.total,"
             "utilization.gpu,utilization.memory,power.draw")
    with out_path.open("w") as fh:
        fh.write("timestamp,mem_used_mib,mem_total_mib,util_gpu_pct,"
                 "util_mem_pct,power_w\n")
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--id={gpu}", query,
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=20, check=False)
                if out.stdout.strip():
                    fh.write(out.stdout.strip() + "\n")
                    fh.flush()
            except (OSError, subprocess.SubprocessError):
                pass    # a transient nvidia-smi failure must not kill the run
            stop.wait(GPU_SAMPLE_SECONDS)


def _summarize_gpu(csv_path: pathlib.Path) -> dict:
    """Peak memory and steady-state utilisation from the telemetry."""
    if not csv_path.exists():
        return {}
    mem, util, power, total = [], [], [], []
    for line in csv_path.read_text().splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            mem.append(float(parts[1]))
            total.append(float(parts[2]))
            util.append(float(parts[3]))
            power.append(float(parts[5]))
        except ValueError:
            continue
    if not mem:
        return {}
    # Drop the first 20 % of samples from the utilisation/power stats: that
    # window is compile, where the GPU is idle or autotuning and neither
    # number reflects the steady-state workload. Peak MEMORY keeps the whole
    # series, since an allocation spike during compile is still a real
    # requirement for provisioning.
    warm = max(1, len(util) // 5)
    return {
        "peak_mem_mib": max(mem),
        "peak_mem_gib": round(max(mem) / 1024, 2),
        "mem_total_gib": round(max(total) / 1024, 2) if total else None,
        "median_util_pct": round(statistics.median(util[warm:] or util), 1),
        "median_power_w": round(statistics.median(power[warm:] or power), 1),
        "n_samples": len(mem),
    }


def _analyse_chunks(walls: list[float], days_per_chunk: int) -> dict:
    """Reduce per-chunk wall times to a converged throughput.

    Chunk 1 always contains compilation and is discarded. A rate is only
    reported as converged once two consecutive chunks agree to
    ``CONVERGENCE_TOL``; otherwise the last chunk is reported explicitly
    flagged as unconverged rather than silently passed off as the answer.
    """
    if len(walls) < 2:
        return {"converged": False, "reason": "fewer than 2 chunks completed",
                "chunk_walls_s": walls}
    post_compile = walls[1:]
    converged_at = None
    for i in range(1, len(post_compile)):
        a, b = post_compile[i - 1], post_compile[i]
        if abs(a - b) / max(a, b) <= CONVERGENCE_TOL:
            converged_at = i
            break
    if converged_at is None:
        chosen = post_compile[-1]
        conv = False
        reason = (f"no two consecutive chunks agreed within "
                  f"{CONVERGENCE_TOL:.0%}; reporting the last chunk")
    else:
        chosen = post_compile[converged_at]
        conv = True
        reason = (f"chunks {converged_at + 1} and {converged_at + 2} "
                  f"(post-compile) agreed within {CONVERGENCE_TOL:.0%}")
    return {
        "converged": conv,
        "reason": reason,
        "chunk_walls_s": walls,
        "compile_chunk_s": walls[0],
        "steady_chunk_s": chosen,
        "s_per_sim_day": round(chosen / days_per_chunk, 2),
        "sim_days_per_hour": round(3600.0 / (chosen / days_per_chunk), 1),
        "sim_years_per_day": round(
            86400.0 / (chosen / days_per_chunk) / 365.0, 2),
    }


def run(args) -> dict:
    preset = PRESETS[args.preset]
    days = args.days if args.days else args.months * 30
    chunk = args.chunk_days
    outdir = pathlib.Path(args.outdir) / (args.label or args.preset)
    outdir.mkdir(parents=True, exist_ok=True)

    overrides = [
        *preset,
        f"run.total_time={days}",
        f"run.chunk_days={chunk}",
        # save_interval must be <= chunk_days or the chunk write dies with an
        # IndexError from to_xarray() on an empty time axis.
        f"run.save_interval={min(args.save_interval, chunk)}",
        f"run.output_prefix={outdir}/state",
        *args.extra,
    ]
    cmd = [args.python, "-m", "jcm.main", *overrides]

    env_note = {}
    import os
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.pythonpath:
        env["PYTHONPATH"] = args.pythonpath
        env_note["PYTHONPATH"] = args.pythonpath

    log_path = outdir / "run.log"
    gpu_path = outdir / "gpu.csv"
    stop = threading.Event()
    sampler = threading.Thread(target=_gpu_sampler,
                               args=(args.gpu, gpu_path, stop), daemon=True)
    sampler.start()

    t0 = time.time()
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh,
                              stderr=subprocess.STDOUT, check=False)
    wall_total = time.time() - t0
    stop.set()
    sampler.join(timeout=30)

    log = log_path.read_text()
    walls = [float(m) for m in _WALL_RE.findall(log)]
    nan_hits = [(int(a), int(b)) for a, b in _NAN_RE.findall(log)]
    last_day = max((int(d) for d in _SAVED_RE.findall(log)), default=0)

    result = {
        "label": args.label or args.preset,
        "preset": args.preset,
        "months": args.months,
        "requested_days": days,
        "completed_days": last_day,
        "chunk_days": chunk,
        "gpu_index": args.gpu,
        "gpu_name": _gpu_name(args.gpu),
        "exit_code": proc.returncode,
        "total_wall_s": round(wall_total, 1),
        "nan_any": any(n > 0 for n, _ in nan_hits),
        "nan_max_vars": max((n for n, _ in nan_hits), default=0),
        "nan_total_vars": nan_hits[0][1] if nan_hits else None,
        "overrides": overrides,
        "env": env_note,
        **_analyse_chunks(walls, chunk),
        "gpu": _summarize_gpu(gpu_path),
    }
    (outdir / "result.json").write_text(json.dumps(result, indent=2))
    (outdir / "report.md").write_text(_report(result))
    return result


def _gpu_name(idx: int) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={idx}", "--query-gpu=name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _report(r: dict) -> str:
    g = r.get("gpu") or {}
    lines = [
        f"# jcm benchmark — {r['label']}",
        "",
        f"- preset: `{r['preset']}`",
        f"- GPU {r['gpu_index']}: {r['gpu_name']}",
        f"- requested {r['requested_days']} d, "
        f"completed {r['completed_days']} d",
        f"- exit code: {r['exit_code']}",
        "",
        "## Throughput",
        "",
    ]
    if r.get("converged") is None:
        lines.append("No chunk timings parsed — the run did not get that far.")
    elif r.get("s_per_sim_day"):
        status = "converged" if r["converged"] else "**NOT CONVERGED**"
        lines += [
            f"**{r['sim_days_per_hour']} sim days/hr** "
            f"({r['s_per_sim_day']} s per sim day, "
            f"{r['sim_years_per_day']} sim years/day) — {status}",
            "",
            f"- {r['reason']}",
            f"- compile chunk: {r['compile_chunk_s']} s (discarded)",
            f"- per-chunk walls: {r['chunk_walls_s']}",
        ]
    else:
        lines.append(f"Not enough chunks: {r.get('reason')}")
    lines += ["", "## GPU", ""]
    if g:
        lines += [
            f"- peak memory: {g['peak_mem_gib']} GiB",
            f"- median utilisation (post-compile): {g['median_util_pct']} %",
            f"- median power: {g['median_power_w']} W",
            f"- samples: {g['n_samples']}",
        ]
    else:
        lines.append("No GPU telemetry collected.")
    lines += ["", "## Health", ""]
    if r["nan_any"]:
        lines.append(
            f"**NaN detected** — up to {r['nan_max_vars']}/"
            f"{r['nan_total_vars']} variables. Timing from a run that blew up "
            "is not a valid benchmark: fix the configuration and re-run.")
    else:
        lines.append("No NaN reported by the health gate.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--preset", required=True, choices=sorted(PRESETS))
    p.add_argument("--months", type=int, default=1,
                   help="1 for the short benchmark, 12 for the long one "
                        "(a 'month' is 30 days here)")
    p.add_argument("--days", type=int, default=None,
                   help="explicit sim-day count; overrides --months")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--chunk-days", type=int, default=5,
                   help="5 gives several post-compile chunks in a 30-day run")
    p.add_argument("--save-interval", type=int, default=5)
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--python", default=DEFAULT_PY)
    p.add_argument("--pythonpath", default=None,
                   help="prepend a library worktree (editable-install A/B)")
    p.add_argument("--extra", nargs="*", default=[],
                   help="additional raw Hydra overrides")
    args = p.parse_args(argv)

    if not shutil.which("nvidia-smi"):
        print("warning: nvidia-smi not found; no GPU telemetry",
              file=sys.stderr)
    r = run(args)
    print((pathlib.Path(args.outdir) /
           (args.label or args.preset) / "report.md").read_text())
    return 0 if r["exit_code"] == 0 and not r["nan_any"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
