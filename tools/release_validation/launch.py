"""Generate (and optionally submit) the release-validation matrix runs.

    python tools/release_validation/launch.py --repo . [--members a,b] \
        [--tag SHA] [--resume] [--submit]

Each member of ``matrix.yaml`` references a validated preset in
``tools/benchmark.py``'s ``PRESETS`` (the single home of known-good
override sets) and becomes a PBS job running a full-output year on one
A100. Per-grid inputs resolve automatically inside jcm (``terrain=auto``,
``forcing.ozone_file=auto``); JAM members additionally need the aux
inputs staged per
``jcm/data/mirror/SOURCES.md`` (dms/dust/oxidants + emissions on the
model grid) via the ``JAM_INPUTS``/``JCM_EMISSIONS`` environment.
Each run directory is namespaced by ``--tag`` (default: the launched
repo's HEAD short SHA), because a release-validation member is a *fresh*
year: a fixed rundir let a second matrix run silently resume the first
one's checkpoint. Health-check finished runs with ``health.py``; run
``scm_check.py`` for the SCM member (CPU, no PBS needed).
"""
import argparse
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
HOME = os.environ["HOME"]
sys.path.insert(0, str(HERE.parent))
from benchmark import PRESETS  # noqa: E402


def jam_aux(grid: str, levels: str) -> list[str]:
    inputs = os.environ.get(
        "JAM_INPUTS", "/glade/derecho/scratch/" + os.environ.get("USER", "")
        + "/jam_inputs")
    token = grid.split("_")[1]        # echam_t63_l95_hybrid -> t63
    # Emissions are horizontal-only (12-month 2-D fields), so every level
    # set of a horizontal grid shares the L47-named prep_emissions output.
    emis = os.environ.get(
        "JCM_EMISSIONS",
        f"{HOME}/jax-gcm/runs/emissions_echam_{token}_l47_hybrid_2014.nc")
    # The oxidant source (cam/waccm) is the preparer's choice —
    # prep_jam_aux_inputs recommends waccm at L95 — so match any source
    # rather than hardcoding one.
    ox = sorted(Path(inputs).glob(
        f"oxidants_*_echam_{levels}_2014_{token}.nc"))
    ov = [
        f"forcing.emissions_file={emis}",
        f"forcing.dms_file={inputs}/dms_lana2011_climo_{token}.nc",
        f"forcing.dust_file={inputs}/dust_erodibility_cam_f05_{token}.nc",
    ]
    if ox:
        ov.append(f"forcing.oxidants_file={ox[-1]}")
    else:
        raise SystemExit(
            f"no oxidants_*_echam_{levels}_2014_{token}.nc under {inputs} — "
            "regenerate per jcm/data/mirror/SOURCES.md (scratch is "
            "purge-eligible)")
    for o in ov[:3]:
        path = o.split("=", 1)[1]
        if not Path(path).exists():
            raise SystemExit(
                f"missing JAM input {path} — regenerate per "
                "jcm/data/mirror/SOURCES.md (scratch is purge-eligible)")
    return ov


def _preset_grid(preset_name: str) -> str | None:
    """Return the grid config a preset composes to.

    PRESETS entries are now a thin ``+configuration=<name>`` shim, so the grid is
    inside the configuration yaml rather than a ``grid=`` override string. Compose
    the preset (pure Hydra, no model build) and read the chosen grid group.
    """
    from pathlib import Path

    import jcm
    from hydra import compose, initialize_config_dir
    cfgdir = str(Path(jcm.__file__).resolve().parent / "config")
    with initialize_config_dir(config_dir=cfgdir, version_base=None):
        cfg = compose(config_name="config", overrides=PRESETS[preset_name],
                      return_hydra_config=True)
    return cfg.hydra.runtime.choices.get("grid")


TAG_UNSAFE = re.compile(r"[^A-Za-z0-9_]")


def check_tag(tag: str) -> str:
    """Return an explicit ``--tag``, refusing anything unsafe to embed.

    The tag becomes a path segment, a PBS job name and the completion
    marker, so a branch-style ``feature/foo`` writes the job into a
    directory that does not exist and a tag carrying spaces or shell
    metacharacters produces malformed PBS directives. Reject rather than
    rewrite: folding ``a/b`` and ``a_b`` onto one tag would put two
    launches in one rundir, the checkpoint collision this tool exists to
    prevent (#701).
    """
    if not tag or TAG_UNSAFE.search(tag):
        raise SystemExit(
            f"--tag {tag!r} is not a usable run tag: it names a run "
            "directory, a PBS job and the completion marker, so it must be "
            "non-empty and made only of letters, digits and underscores.")
    return tag


def repo_tag(repo: str | Path) -> str:
    """Return the run tag for ``repo``: its HEAD short SHA where there is one.

    Naming the rundir after the commit makes the provenance already recorded
    in each output (``jcm=<sha>``) match the integration that produced it.
    Outside a git checkout (a tarball, an exported tree) fall back to the UTC
    date, which still separates one day's launch from the next.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        out = ""
    tag = out or "nogit_" + datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%d")
    # Same safe character set check_tag demands of an explicit --tag, but
    # derived rather than typed, so rewrite quietly instead of failing.
    return TAG_UNSAFE.sub("_", tag)


def check_fresh(rundir: str, resume: bool) -> None:
    """Refuse to (re)launch into a rundir that already holds a checkpoint.

    ``run_chunked`` resumes from ``checkpoint_path`` when it exists, so a
    second launch into a populated directory continues someone else's
    integration and reports a healthy year for a run that never happened
    (#701). Crashing on a mismatched physics composition is the lucky case.
    """
    ckpt = Path(rundir) / "checkpoint.msgpack"
    if ckpt.exists() and not resume:
        raise SystemExit(
            f"{ckpt} already exists — this member has been launched under "
            "this tag before, and starting here would resume that run "
            "instead of integrating a fresh year. Either pass --resume to "
            "continue it deliberately, or launch under a new --tag (the "
            "default is the repo HEAD short SHA).")


def overrides(name: str, m: dict, d: dict, rundir: str) -> list[str]:
    # Presets may carry their own output plumbing (the pyses ones set
    # run.checkpoint_path); ours must win, so strip conflicting keys
    # rather than duplicating the override on the CLI.
    preset = [o for o in PRESETS[m["preset"]]
              if not o.startswith(("run.checkpoint_path", "run.output",
                                   "hydra.run.dir"))]
    grid = _preset_grid(m["preset"]) if m.get("jam_inputs") else None
    if m.get("jam_inputs") and grid is None:
        raise SystemExit(
            f"preset {m['preset']} composes no grid; JAM aux "
            "input resolution needs one.")
    ovs = [*preset,
           f"run.total_time={d['days']}.0",
           f"run.save_interval={d['save_interval']}",
           f"run.chunk_days={d['chunk_days']}",
           "run.output_averages=true", "run.log_level=INFO",
           f"run.output={name}.nc",
           f"run.output_prefix={rundir}/{name}",
           # checkpoint_path is now a universal run key (the run schema is one
           # base -- #640), so a plain override sets it on every run group.
           f"run.checkpoint_path={rundir}/checkpoint.msgpack"]
    if m.get("jam_inputs"):
        ovs += jam_aux(grid, m["jam_inputs"])
    return ovs


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
export PYTHONPATH={repo}
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


def main(argv=None):
    """Write (and optionally submit) one PBS job per matrix member."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--members", default=None,
                    help="comma-separated subset (default: all)")
    ap.add_argument("--tag", default=None,
                    help="rundir/job namespace, letters/digits/underscore "
                         "only (default: repo HEAD short SHA)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing run rather than refusing to "
                         "start on top of its checkpoint")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--account",
                    default=os.environ.get("PBS_ACCOUNT", "UCSD0085"))
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(HERE / "matrix.yaml"))
    d = cfg["defaults"]
    repo = str(Path(a.repo).resolve())
    scratch = os.environ.get("SCRATCH", f"{HOME}/scratch")
    venv = os.environ.get("JCM_VENV", f"{HOME}/.venvs/jaxgcm")
    outdir = Path(repo) / "runs"
    outdir.mkdir(exist_ok=True)

    run_tag = check_tag(a.tag) if a.tag is not None else repo_tag(repo)
    print(f"run tag: {run_tag}")

    wanted = a.members.split(",") if a.members else list(cfg["members"])
    # The tag namespaces the rundir, the job name, the outputs and the log
    # together, so one launch's artefacts never mix with another's. Vet every
    # member before writing anything: a matrix launch that would resume a
    # previous one should fail whole, not half-submitted.
    plan = [(name, f"mx_{name.replace('-', '_')}_{run_tag}") for name in wanted]
    for _, tag in plan:
        check_fresh(f"{scratch}/jam_runs/{tag}", a.resume)

    for name, tag in plan:
        m = cfg["members"][name]
        rundir = f"{scratch}/jam_runs/{tag}"
        ovs = " \\\n    ".join(
            overrides(tag, m, d, rundir) + [f"hydra.run.dir={rundir}"])
        job = PBS.format(
            name=tag, account=a.account, hours=m.get("hours", d["hours"]),
            logdir=str(outdir), venv=venv, repo=repo,
            rundir=rundir, ovs=ovs, marker=f"{tag.upper()}_COMPLETE",
        )
        path = outdir / f"{tag}.pbs"
        path.write_text(job)
        print("wrote", path)
        if a.submit:
            subprocess.run(["qsub", str(path)], check=True)


if __name__ == "__main__":
    main()
