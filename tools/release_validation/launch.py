"""Generate (and optionally submit) the release-validation matrix runs.

    python tools/release_validation/launch.py --repo . [--members a,b] [--submit]

Each member of ``matrix.yaml`` becomes a PBS job running a full-output
year on one A100. Per-grid inputs resolve automatically inside jcm
(``terrain=auto``, ``forcing.ozone_file=auto`` — see
``runners._resolve_auto_terrain``/``_resolve_auto_ozone``); JAM members
additionally need the aux inputs staged per
``jcm/data/mirror/SOURCES.md`` (dms/dust/oxidants + emissions on the
model grid) via the ``JAM_INPUTS``/``JCM_EMISSIONS`` environment.
Health-check finished runs with ``health.py``; run ``scm_check.py`` for
the SCM member (CPU, no PBS needed).
"""
import argparse
import os
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).parent
HOME = os.environ["HOME"]


def jam_aux(repo: str, grid: str, levels: str) -> list[str]:
    inputs = os.environ.get(
        "JAM_INPUTS", "/glade/derecho/scratch/" + os.environ.get("USER", "")
        + "/jam_inputs")
    token = grid.split("_")[1]        # echam_t63_l95_hybrid -> t63
    # Emissions are horizontal-only (12-month 2-D fields), so every level
    # set of a horizontal grid shares the L47-named prep_emissions output.
    emis = os.environ.get(
        "JCM_EMISSIONS",
        f"{HOME}/jax-gcm/runs/emissions_echam_{token}_l47_hybrid_2014.nc")
    ov = [
        f"forcing.emissions_file={emis}",
        f"forcing.dms_file={inputs}/dms_lana2011_climo_{token}.nc",
        f"forcing.dust_file={inputs}/dust_erodibility_cam_f05_{token}.nc",
        f"forcing.oxidants_file={inputs}/oxidants_cam_echam_{levels}_2014_{token}.nc",
    ]
    for o in ov:
        path = o.split("=", 1)[1]
        if not Path(path).exists():
            raise SystemExit(
                f"missing JAM input {path} — regenerate per "
                "jcm/data/mirror/SOURCES.md (scratch is purge-eligible)")
    return ov


def overrides(name: str, m: dict, d: dict, rundir: str) -> list[str]:
    grid = m["grid"]
    base = [
        f"physics={m['physics']}", f"grid={grid}",
        "terrain=auto", "forcing=from_file",
        "forcing.file=jcm/data/bc/t63/forcing.nc",
        f"run.time_step={d['time_step_min']}",
        f"run.total_time={d['days']}.0",
        f"run.save_interval={d['save_interval']}",
        f"run.chunk_days={d['chunk_days']}",
        "run.output_averages=true", "run.log_level=INFO",
        f"run.output={name}.nc",
        f"run.output_prefix={rundir}/{name}",
        f"run.checkpoint_path={rundir}/checkpoint.msgpack",
    ]
    if m.get("profile") != "speedy":
        # ECHAM year-run profile: dry JW spin-up + the longrun sponge
        # (all keys under the DEFAULT run group, hence no '+' prefixes
        # except the group-replacing longrun's own additions).
        base = ["init=jw", "init.rh=0.0", "run=longrun",
                "+advection=semi_lagrangian", "+sl_off_centering=0.2"] + [
            o.replace("run.chunk_days", "run.chunk_days")
            for o in base if not o.startswith(("run.checkpoint",))
        ] + [f"+run.checkpoint_path={rundir}/checkpoint.msgpack"]
    if m.get("jam_inputs"):
        base += jam_aux(".", grid, m["jam_inputs"])
    return base


PBS = """#!/bin/bash
#PBS -N {name}
#PBS -A {account}
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=1:mem=160GB:gpu_type=a100
#PBS -l walltime={hours}:00:00
#PBS -m abe
#PBS -j oe
#PBS -o {logdir}/{name}.log
set -euo pipefail
source {venv}/bin/activate
export PYTHONPATH={dinosaur}:{repo}
export JAX_PLATFORMS=cuda,cpu
export MAM4_JAX_ENABLE_X64=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.93
export JAX_COMPILATION_CACHE_DIR=${{SCRATCH}}/jcm-jax-cache
mkdir -p {rundir}
cd {repo}
python -u -m jcm.main \\
    {ovs}
echo {marker}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--members", default=None,
                    help="comma-separated subset (default: all)")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--account",
                    default=os.environ.get("PBS_ACCOUNT", "UCSD0085"))
    a = ap.parse_args()

    cfg = yaml.safe_load(open(HERE / "matrix.yaml"))
    d = cfg["defaults"]
    repo = str(Path(a.repo).resolve())
    scratch = os.environ.get("SCRATCH", f"{HOME}/scratch")
    venv = os.environ.get("JCM_VENV", f"{HOME}/.venvs/jaxgcm")
    dinosaur = os.environ.get("JCM_DINOSAUR", f"{HOME}/dinosaur-sl")
    outdir = Path(repo) / "runs"
    outdir.mkdir(exist_ok=True)

    wanted = a.members.split(",") if a.members else list(cfg["members"])
    for name in wanted:
        m = cfg["members"][name]
        tag = "mx_" + name.replace("-", "_")
        rundir = f"{scratch}/jam_runs/{tag}"
        ovs = " \\\n    ".join(
            overrides(tag, m, d, rundir) + [f"hydra.run.dir={rundir}"])
        job = PBS.format(
            name=tag, account=a.account, hours=m.get("hours", d["hours"]),
            logdir=str(outdir), venv=venv, dinosaur=dinosaur, repo=repo,
            rundir=rundir, ovs=ovs, marker=f"{tag.upper()}_COMPLETE",
        )
        path = outdir / f"{tag}.pbs"
        path.write_text(job)
        print("wrote", path)
        if a.submit:
            subprocess.run(["qsub", str(path)], check=True)


if __name__ == "__main__":
    main()
