#!/usr/bin/env python
"""Generate a PBS job script for a jcm run on Derecho.

    python mkjob.py --name my_run --days 30 > runs/my_run.pbs
    python mkjob.py --name bench --bench --days 20 > runs/bench.pbs

Writes the script to stdout. ``--check`` composes the Hydra config first
(cheap, catches +/++ prefix errors before a queue slot is burned).
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

# Site defaults; every one is overridable by env var or flag so the skill
# works for any user/checkout on Derecho. See reference/data_paths.md.
HOME = os.path.expanduser("~")
USER = os.environ.get("USER", "")
SCRATCH = os.environ.get("SCRATCH", f"/glade/derecho/scratch/{USER}")
DEFAULT_REPO = os.environ.get("JCM_REPO", f"{HOME}/jax-gcm-pyses")
DEFAULT_VENV = os.environ.get("JCM_VENV", f"{HOME}/.venvs/jaxgcm")
DEFAULT_DINOSAUR = os.environ.get("JCM_DINOSAUR", f"{HOME}/dinosaur-sl")
DEFAULT_ACCOUNT = os.environ.get("PBS_ACCOUNT", "UCSD0085")
JAM_INPUTS = os.environ.get("JAM_INPUTS", f"{SCRATCH}/jam_inputs")


def grid_layers(grid: str) -> str:
    """Layer count parsed out of a grid name (echam_t63_l47_hybrid -> "47")."""
    return "".join(c for c in grid.split("_l")[-1] if c.isdigit()) or "?"


def grid_truncation(grid: str) -> str:
    """Spectral truncation parsed out of a grid name (..._t63_... -> "t63")."""
    if "_t" not in grid:
        return "t63"
    return "t" + "".join(c for c in grid.split("_t")[-1].split("_")[0]
                         if c.isdigit())


# Prepared aux inputs (purge-eligible on scratch — see reference/data_paths.md).
# EVERY one of these is grid-specific, in one of two ways, and both were got
# wrong here before:
#   * oxidants are LEVEL-resolved — the runner validates the level count, so a
#     hard-coded L47 file killed every L95 job after it reached the queue front;
#   * dms/dust are HORIZONTALLY resolved — read_dms_seawater/read_dust_source
#     validate lat/lon against the model Gaussian grid, so the T63 files are
#     not usable on T106/T119 despite having no vertical axis. (An earlier
#     comment here claimed they were "grid-independent" because they are
#     surface fields. That conflated the vertical with the horizontal.)
# Deriving all of them means check_inputs() catches a missing or wrong-grid
# file BEFORE qsub rather than the job dying in the queue.
def aux_files(grid: str) -> dict:
    lev, tr = grid_layers(grid), grid_truncation(grid)
    return {
        "dms_file": f"{JAM_INPUTS}/dms_lana2011_climo_{tr}.nc",
        "dust_file": f"{JAM_INPUTS}/dust_erodibility_cam_f19_{tr}.nc",
        "oxidants_file":
            f"{JAM_INPUTS}/oxidants_cam_echam_l{lev}_2014_{tr}.nc",
    }


# Level count of the packaged ozone climatology, per truncation directory
# under jcm/data/bc/. `_resolve_auto_ozone` only ever picks from these.
PACKAGED_OZONE_LEVELS = {"t63": "47", "t30": "8"}


def check_ozone(a) -> str | None:
    """Return an explicit ozone override, or None to leave ``auto``.

    ``forcing.ozone_file: auto`` resolves a PACKAGED climatology and silently
    falls back to RRTMGP's ANALYTIC profile when none matches the grid. That
    surrogate carries ~7.6x the tropospheric ozone column, so an L95 or
    T106/T119 job would run to completion and quietly produce radiation that
    cannot be compared with anything — no preflight failure, no warning the
    watcher looks for. Refuse instead, unless the run is aquaplanet (no
    prescribed forcing at all) or an explicit --ozone was supplied.
    """
    if a.ozone:
        return a.ozone
    if a.aquaplanet:
        return None
    tr, lev = grid_truncation(a.grid), grid_layers(a.grid)
    if PACKAGED_OZONE_LEVELS.get(tr) == lev:
        return None          # auto will resolve the packaged file correctly
    sys.exit(
        f"NO PACKAGED OZONE for {a.grid} ({tr}, L{lev}).\n"
        "  forcing.ozone_file=auto would fall back to the ANALYTIC profile "
        "(~7.6x the tropospheric ozone column), which runs fine and silently "
        "invalidates the radiation.\n"
        "  Pass --ozone /path/to/ozone_<grid>.nc — see reference/data_paths.md "
        "(e.g. ozone_cam6chem_2005-2014_t63_l95.nc), or prepare one with\n"
        f"    python -m jcm.data.bc.interpolate_ozone --in T63_ozone_picontrol.nc"
        f" --out ozone_{tr}_l{lev}.nc --nlevels {lev}"
    )


def default_emissions(grid: str) -> str:
    """Emissions file for a grid.

    ``_validate_emissions_grid`` rejects fields whose horizontal shape differs
    from the model grid, so the T63 default cannot be reused on T85/T106/T119 —
    the job would pass check_inputs() and then die after reaching the queue
    front. Derived here so a missing file is caught before qsub instead.
    """
    # $JCM_EMISSIONS pins ONE file, so honour it (an explicit site choice) but
    # say so when it does not name the requested grid -- silently reusing a
    # T63 file on T106 is the failure this function exists to prevent.
    pinned = os.environ.get("JCM_EMISSIONS")
    if pinned:
        if grid not in os.path.basename(pinned):
            print(f"warning: $JCM_EMISSIONS={pinned} does not name grid "
                  f"{grid}; _validate_emissions_grid will reject it if the "
                  "horizontal shape differs", file=sys.stderr)
        return pinned
    root = os.environ.get("JCM_EMISSIONS_DIR", f"{HOME}/jax-gcm/runs")
    return f"{root}/emissions_{grid}_2014.nc"


def check_inputs(a) -> None:
    """Fail before qsub if a required input file is missing (scratch purges)."""
    needed = [("emissions", a.emissions), *aux_files(a.grid).items()]
    missing = [f"{k}: {v}" for k, v in needed if not os.path.exists(v)]
    if missing:
        sys.exit("MISSING INPUT FILES (scratch is purge-eligible; see "
                 "reference/data_paths.md to regenerate):\n  " +
                 "\n  ".join(missing))


def build_overrides(a) -> list[str]:
    """Hydra overrides for one variant, in composition order."""
    ov = [f"physics={a.physics}", f"grid={a.grid}"]
    if a.physics.endswith("jam"):
        ov.append(f"physics.jam_microphysics={a.microphysics}")
    if a.radiation:
        ov.append(f"physics.radiation_scheme={a.radiation}")
    if not a.aquaplanet:
        ov += [
            "terrain=from_file",
            f"terrain.file={a.repo}/jcm/data/bc/t63/terrain.nc",
            "forcing=from_file",
            f"forcing.file={a.repo}/jcm/data/bc/t63/forcing.nc",
        ]
        if a.physics.endswith("jam") and not a.no_emissions:
            ov.append(f"forcing.emissions_file={a.emissions}")
            if a.ozone:
                ov.append(f"forcing.ozone_file={a.ozone}")
            ov += [f"forcing.{k}={v}"
                   for k, v in aux_files(a.grid).items()]
    ov += ["init=jw", "init.rh=0.0", "run=longrun"]
    if a.physics.endswith("jam"):
        ov.append("diffusion.tracer_positivity=true")
    if a.advection == "semi_lagrangian":
        ov += ["+advection=semi_lagrangian",
               f"+sl_off_centering={a.off_centering}"]
    if a.gpus > 1:
        if a.gpus % 2 == 0:
            default_mesh = f"[{a.gpus // 2},2,1]"
        else:
            default_mesh = f"[{a.gpus},1,1]"
        mesh = a.mesh or default_mesh
        ov.append(f'"+grid.spmd_mesh={mesh}"')
    ov += [
        f"run.time_step={a.dt}",
        f"run.total_time={a.days}",
        f"run.save_interval={a.save_every}",
        f"run.chunk_days={a.chunk_days}",
        "run.output_averages=true",
        "run.log_level=INFO",
    ]
    if a.extra:
        ov += shlex.split(a.extra)
    return ov


def check_compose(a, overrides) -> None:
    """Compose the config on the login node; fail loudly before qsub."""
    cmd = [sys.executable, "-m", "jcm.main",
           *[o.strip('"') for o in overrides
             if not o.startswith(("run.output", "hydra.run.dir", "+run.checkpoint"))],
           "--cfg", "job"]
    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "PYTHONPATH": f"{a.dinosaur}:{a.repo}"}
    r = subprocess.run(cmd, cwd=a.repo, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("COMPOSE FAILED:\n" + (r.stderr or r.stdout)[-2000:])
    layers = grid_layers(a.grid)
    trunc = a.grid.split("_t")[-1].split("_")[0] if "_t" in a.grid else "?"
    print(f"# compose OK; also verify coords build:\n"
          f"#   JAX_PLATFORMS=cpu python -c \"from jcm.utils import get_coords;"
          f" from jcm.physics.echam.echam_levels import get_echam_levels;"
          f" get_coords(vertical_coords=get_echam_levels({layers}),"
          f" spectral_truncation={trunc})\"", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--days", type=float, default=30)
    p.add_argument("--dt", type=float, default=15, help="minutes")
    p.add_argument("--grid", default="echam_t63_l47_hybrid")
    p.add_argument("--physics", default="echam-jam")
    p.add_argument("--microphysics", default="mam4_jax")
    p.add_argument("--radiation", default=None,
                   help="override radiation_scheme, e.g. grey")
    p.add_argument("--advection", default="semi_lagrangian",
                   choices=["semi_lagrangian", "eulerian"])
    p.add_argument("--off-centering", type=float, default=0.2)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--mesh", default=None, help="explicit spmd_mesh, e.g. [2,2,1]")
    p.add_argument("--cpus", type=int, default=16)
    p.add_argument("--mem", default=None, help="default 160GB, 200GB if gpus>1")
    p.add_argument("--hours", type=float, default=6)
    p.add_argument("--queue", default="main",
                   help="main (routes to gpu) or gpudev for short debugging")
    p.add_argument("--account", default=DEFAULT_ACCOUNT)
    p.add_argument("--chunk-days", type=float, default=5)
    p.add_argument("--save-every", type=float, default=5)
    p.add_argument("--aquaplanet", action="store_true")
    p.add_argument("--no-emissions", action="store_true")
    p.add_argument("--ozone", default=None,
                   help="explicit ozone file; required for any grid without a "
                        "packaged climatology (non-T63L47), since `auto` would "
                        "silently fall back to the analytic profile")
    p.add_argument("--emissions", default=None,
                   help="anthropogenic emissions netCDF")
    p.add_argument("--resume", action="store_true",
                   help="keep an existing checkpoint in the run dir")
    p.add_argument("--bench", action="store_true",
                   help="variant matrix (reference + grey) with settled rates")
    p.add_argument("--bench-variant", action="append", default=[],
                   metavar="TAG:OVERRIDES", help="extra --bench variant")
    p.add_argument("--extra", default="", help="raw Hydra overrides")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--venv", default=DEFAULT_VENV)
    p.add_argument("--dinosaur", default=DEFAULT_DINOSAUR)
    p.add_argument("--check", action="store_true",
                   help="compose the config before emitting the script")
    a = p.parse_args()

    # Grid-derived defaults, resolved before any check runs.
    if a.emissions is None:
        a.emissions = default_emissions(a.grid)
    # Refuses (with instructions) when the grid has no packaged climatology.
    a.ozone = check_ozone(a)

    a.mem = a.mem or ("200GB" if a.gpus > 1 else "160GB")
    frac = 0.85 if a.gpus > 1 else 0.93
    rundir = f"{SCRATCH}/jam_runs/{a.name}"
    if a.physics.endswith("jam") and not a.aquaplanet and not a.no_emissions:
        check_inputs(a)
    overrides = build_overrides(a)
    if a.check:
        check_compose(a, overrides)

    marker = f"{a.name.upper()}_COMPLETE"
    ov = " \\\n    ".join(overrides)
    head = f"""#!/bin/bash
# Generated by derecho-jcm-runs/mkjob.py — edit freely, but keep -m abe and
# `set -euo pipefail` (a failed run must not look successful).
#PBS -N {a.name}
#PBS -A {a.account}
#PBS -q {a.queue}
#PBS -l select=1:ncpus={a.cpus}:ngpus={a.gpus}:mem={a.mem}:gpu_type=a100
#PBS -l walltime={int(a.hours):02d}:{int((a.hours % 1) * 60):02d}:00
#PBS -m abe
#PBS -j oe
#PBS -o {a.name}.log
set -euo pipefail
REPO={a.repo}
RUNDIR={rundir}
mkdir -p "$RUNDIR"
source {a.venv}/bin/activate
cd "$REPO"
export PYTHONPATH={a.dinosaur}:$REPO
export JAX_PLATFORMS=cuda,cpu
export MAM4_JAX_ENABLE_X64=0
export XLA_PYTHON_CLIENT_MEM_FRACTION={frac}
echo "host: $(hostname -f)"; nvidia-smi -L | head -{a.gpus}
echo "jcm: $(git -C $REPO rev-parse --short HEAD)"
"""
    if not a.resume:
        head += 'rm -f "$RUNDIR"/checkpoint.msgpack\n'

    if not a.bench:
        body = f"""
python -u -m jcm.main \\
    {ov} \\
    run.output={a.name}.nc \\
    run.output_prefix=$RUNDIR/{a.name} \\
    +run.checkpoint_path=$RUNDIR/checkpoint.msgpack \\
    hydra.run.dir=$RUNDIR
echo "{marker}"
"""
    else:
        variants = ["reference:"] + ["grey:physics.radiation_scheme=grey"] \
            + a.bench_variant
        skill_dir = os.path.dirname(os.path.abspath(__file__))
        body = f'\nSKILL_DIR={skill_dir}\nCHUNK_DAYS={a.chunk_days}\n'
        body += f'COMMON="{" ".join(o.strip(chr(34)) for o in overrides)}"\n'
        body += """
FAILED_VARIANTS=0

run_variant () {
  local tag="$1"; shift
  local d="$RUNDIR/$tag"; mkdir -p "$d"
  echo "######## VARIANT: $tag ########"
  ( python -u -m jcm.main $COMMON "$@" run.output=$tag.nc \\
      run.output_prefix=$d/$tag hydra.run.dir=$d > $d/run.log 2>&1 ) &
  local pid=$!
  ( sleep 420; for i in $(seq 6); do kill -0 $pid 2>/dev/null || break
      echo "  [$tag] $(nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu --format=csv,noheader | head -1)"
      sleep 30; done ) &
  if ! wait $pid; then
    echo "  VARIANT $tag FAILED"
    FAILED_VARIANTS=$((FAILED_VARIANTS + 1))
  fi
  # A variant that ran but went unhealthy is also a failure: the driver
  # returns normally after tripping the health gate. Each variant's output is
  # redirected to its own run.log, so the outer watcher never sees these
  # per-chunk health lines -- they have to be checked HERE. A non-zero
  # "NaN vars: N/M" counts even with no "unhealthy" line, because the health
  # line is printed per chunk before the gate reacts, and a NaN in a
  # non-gated variable never produces one at all.
  if grep -aqE "unhealthy|Traceback" "$d/run.log" 2>/dev/null \
     || grep -aoE "NaN vars:[[:space:]]*[0-9]+" "$d/run.log" 2>/dev/null \
        | grep -avqE "NaN vars:[[:space:]]*0$"; then
    echo "  VARIANT $tag UNHEALTHY"
    grep -aE "unhealthy|NaN vars:[[:space:]]*[1-9]" "$d/run.log" | tail -2
    FAILED_VARIANTS=$((FAILED_VARIANTS + 1))
  fi
  # Non-zero here only means "not enough chunks to quote a rate", which is
  # not a job failure, so it stays tolerated.
  python "$SKILL_DIR/settled_rate.py" --chunk-days $CHUNK_DAYS "$d/run.log" || true
}
"""
        for v in variants:
            tag, _, extra = v.partition(":")
            body += f'run_variant {tag} {extra}\n'
        # Withhold the completion marker if any variant failed, so the
        # monitor cannot read a half-failed matrix as a finished benchmark.
        body += (f'\nif [ "$FAILED_VARIANTS" -ne 0 ]; then\n'
                 f'  echo "BENCH FAILED: $FAILED_VARIANTS variant(s)"\n'
                 f'  exit 1\n'
                 f'fi\n'
                 f'echo "{marker}"\n')

    print(head + body)
    print(f"# submit: qsub <this file>\n"
          f"# watch : scripts/watch_job.sh <jobid> {a.name}.log {marker}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
