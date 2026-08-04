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
EMISSIONS = os.environ.get(
    "JCM_EMISSIONS", f"{HOME}/jax-gcm/runs/emissions_echam_t63_l47_hybrid_2014.nc")

# Prepared aux inputs (purge-eligible on scratch — see reference/data_paths.md).
AUX_FILES = {
    "dms_file": f"{JAM_INPUTS}/dms_lana2011_climo_t63.nc",
    "dust_file": f"{JAM_INPUTS}/dust_erodibility_cam_f19_t63.nc",
    "oxidants_file": f"{JAM_INPUTS}/oxidants_cam_echam_l47_2014_t63.nc",
}


def check_inputs(a) -> None:
    """Fail before qsub if a required input file is missing (scratch purges)."""
    needed = [("emissions", a.emissions), *AUX_FILES.items()]
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
            ov += [f"forcing.{k}={v}" for k, v in AUX_FILES.items()]
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
    layers = "".join(c for c in a.grid.split("_l")[-1] if c.isdigit()) or "?"
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
    p.add_argument("--emissions", default=EMISSIONS,
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
  wait $pid || echo "  VARIANT $tag FAILED"
  python "$SKILL_DIR/settled_rate.py" --chunk-days $CHUNK_DAYS "$d/run.log" || true
}
"""
        for v in variants:
            tag, _, extra = v.partition(":")
            body += f'run_variant {tag} {extra}\n'
        body += f'echo "{marker}"\n'

    print(head + body)
    print(f"# submit: qsub <this file>\n"
          f"# watch : scripts/watch_job.sh <jobid> {a.name}.log {marker}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
