"""Generate (and optionally submit) the release-validation matrix runs.

    python tools/release_validation/launch.py --repo . [--members a,b] \
        [--tag SHA] [--resume] [--submit]

Each member of ``matrix.yaml`` references a validated preset in
``tools/benchmark.py``'s ``PRESETS`` (the single home of known-good
override sets) and becomes a PBS job running a full-output year on one
A100. Per-grid inputs resolve automatically inside jcm (``terrain=auto``,
``forcing.ozone_file=auto``) and are PREFETCHED here, on the submitting
(networked) node, so a member whose inputs are unavailable refuses at submit
time instead of after hours of GPU; JAM members additionally need the aux
inputs staged per
``jcm/data/mirror/SOURCES.md`` (dms/oxidants + emissions on the model
grid) via the ``JAM_INPUTS``/``JCM_EMISSIONS`` environment. The five
Tegen dust bundles are mirror products and are fetched here, on the
login node, so the compute nodes need no network.
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
from benchmark import (  # noqa: E402
    PRESETS, _hf_fetch, _preset_data_files)


#: The Tegen dust inputs (#802). Mirror products, so unlike the other JAM aux
#: files they are not produced by the local prep tool.
_DUST_KEYS = ("dust_file", "dust_preferential_file", "dust_soil_types_file",
              "dust_regions_file", "dust_roughness_file")


def dust_overrides(token: str) -> list[str]:
    """Fetch the five dust bundles HERE and pass their local cache paths.

    ``auto`` would resolve — and therefore download — inside the PBS job, where
    there is no internet: the member would abort before integrating on any
    cache that was not already warm. Fetching on the login node at generation
    time and baking in concrete paths is the same contract the rest of this
    launcher uses for its aux inputs.
    """
    from jcm.data import mirror_manifest as mm
    from jcm.data.remote import fetch
    manifest = mm.load_manifest()
    out = []
    for key in _DUST_KEYS:
        product = mm.product_for_key(manifest, key)
        rel = mm.bundle_path(manifest, product, token, None)
        try:
            out.append(f"forcing.{key}={fetch(rel)}")
        except Exception as exc:                              # noqa: BLE001
            raise SystemExit(
                f"could not fetch the dust bundle {rel} for {token}: {exc}. "
                "Release validation generates jobs on a login node precisely "
                "so the compute nodes need no network; fix the fetch here "
                "rather than letting the member abort in the queue.") from exc
    return out


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
        *dust_overrides(token),
    ]
    if ox:
        ov.append(f"forcing.oxidants_file={ox[-1]}")
    else:
        raise SystemExit(
            f"no oxidants_*_echam_{levels}_2014_{token}.nc under {inputs} — "
            "regenerate per jcm/data/mirror/SOURCES.md (scratch is "
            "purge-eligible)")
    for o in ov[:2]:
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


def prefetch(ovs: list[str]) -> list[str]:
    """Download every input a member resolves to; return the unavailable ones.

    Mirrors ``tools/benchmark.py``'s pre-GPU preflight, including the
    lazily-resolved ``auto`` bundles (emissions, ozone) the literal-path walk
    cannot see. A matrix member is a ten-hour GPU job, so an input that cannot
    be resolved must fail here rather than inside model construction on a
    compute node with no network (#774).
    """
    missing = []
    try:
        paths = _preset_data_files(ovs)
    except Exception as e:                  # noqa: BLE001 — reported, not raised
        # Enumeration itself reaches the mirror manifest (the ``auto`` ozone and
        # emission bundles), so it can fail for the same reasons a fetch can.
        # Report it as this member's problem rather than aborting the whole
        # plan with a traceback.
        return [f"could not enumerate inputs ({type(e).__name__}: {e})"]
    for path in paths:
        if path.startswith("hf://"):
            try:
                _hf_fetch(path[len("hf://"):])
            except Exception as e:              # unreachable or absent
                missing.append(f"{path}  ({type(e).__name__}: {e})")
        elif not Path(path).exists():
            missing.append(path)
    return missing


#: Written into each member's rundir at launch: the data-mirror commit the
#: run reads, so a ``--resume`` continues on exactly the same inputs.
MIRROR_RECORD = "mirror_revision.json"


def mirror_commit(rundir: str, resume: bool, force: bool) -> tuple[str, bool]:
    """Return ``(commit, opt_in)`` for one member (recorded after preflight).

    The commit is part of what a member is: the same code and config read
    different inputs at another commit. A fresh launch records this process's
    commit (the pin, or ``JCM_MIRROR_REVISION``). ``--resume`` reuses the
    recorded one, so a jcm update that moved the pin cannot switch a running
    member's inputs; an explicit different ``JCM_MIRROR_REVISION`` is refused
    unless ``force``, which records the new commit and opts the job in to
    resuming a checkpoint written at the old one (``run_chunked`` otherwise
    refuses it).
    """
    import json

    from jcm.data.remote import REVISION_ENV, mirror_revision, revision_source
    commit, source = mirror_revision(), revision_source()
    record = Path(rundir) / MIRROR_RECORD
    if resume and record.exists():
        recorded = json.loads(record.read_text())["commit"]
        if recorded == commit or (source == "pinned" and not force):
            return recorded, False
        if not force:
            raise SystemExit(
                f"{REVISION_ENV}={commit} differs from the mirror commit "
                f"{recorded} recorded in {record}; resuming on it would change "
                "the run's boundary inputs mid-integration. Unset it to "
                "continue on the recorded commit, or pass "
                "--force-mirror-revision to switch deliberately.")
    return commit, force and resume


def write_mirror_record(rundir: str, commit: str) -> None:
    """Record ``commit`` in ``rundir`` (after the preflight has passed)."""
    import json

    from jcm.data.remote import revision_source
    record = Path(rundir) / MIRROR_RECORD
    if record.exists() and json.loads(record.read_text())["commit"] == commit:
        return      # a resume on the recorded commit keeps the launch record
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({
        "requested": commit, "source": revision_source(), "commit": commit,
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds")}, indent=1))


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
           f"run.checkpoint_path={rundir}/checkpoint.msgpack",
           # Permanent archives alongside the rotating checkpoint, so a
           # member whose failure develops slowly still has a state from
           # before the onset to restart from.
           "run.archive_ckpt_every="
           f"{m.get('archive_ckpt_every', d.get('archive_ckpt_every', 0))}"]
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
export JCM_MIRROR_REVISION={mirror_revision}
{mirror_optin}mkdir -p {rundir}
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
    ap.add_argument("--force-mirror-revision", action="store_true",
                    help="with --resume, switch a run to the explicitly set "
                         "JCM_MIRROR_REVISION although it was started at "
                         "another mirror commit (its inputs change mid-run)")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="skip the input preflight (offline job generation)")
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
    # Each member's mirror commit, resolved (and recorded) before the
    # prefetch; the job exports it, as a PBS job does not inherit this shell.
    mirror = {tag: mirror_commit(f"{scratch}/jam_runs/{tag}", a.resume,
                                 a.force_mirror_revision)
              for _, tag in plan}

    # Preflight every member's inputs before writing any job: a matrix launch
    # that cannot resolve an input should fail whole, on the node that still
    # has network, not member by member inside a queued job.
    if not a.no_prefetch:
        missing = {}
        for name, tag in plan:
            # Prefetch at the commit this member's job will read.
            os.environ["JCM_MIRROR_REVISION"] = mirror[tag][0]
            unavailable = prefetch(
                overrides(tag, cfg["members"][name], d,
                          f"{scratch}/jam_runs/{tag}"))
            if unavailable:
                missing[name] = unavailable
        if missing:
            report = "\n".join(f"  {name}:\n    " + "\n    ".join(paths)
                                for name, paths in missing.items())
            raise SystemExit(
                "these members reference inputs that are not available "
                f"here:\n{report}\nStage them per "
                "jcm/data/mirror/SOURCES.md, or pass --no-prefetch to "
                "generate the jobs anyway.")

    # Only now, with every input available, is the commit this launch's.
    for _, tag in plan:
        write_mirror_record(f"{scratch}/jam_runs/{tag}", mirror[tag][0])

    for name, tag in plan:
        m = cfg["members"][name]
        rundir = f"{scratch}/jam_runs/{tag}"
        os.environ["JCM_MIRROR_REVISION"] = mirror[tag][0]  # dust fetch
        ovs = " \\\n    ".join(
            overrides(tag, m, d, rundir) + [f"hydra.run.dir={rundir}"])
        job = PBS.format(
            name=tag, account=a.account, hours=m.get("hours", d["hours"]),
            logdir=str(outdir), venv=venv, repo=repo,
            rundir=rundir, ovs=ovs, marker=f"{tag.upper()}_COMPLETE",
            mirror_revision=mirror[tag][0],
            mirror_optin=("export JCM_ALLOW_MIRROR_REVISION_CHANGE=1\n"
                          if mirror[tag][1] else ""),
        )
        path = outdir / f"{tag}.pbs"
        path.write_text(job)
        print("wrote", path)
        if a.submit:
            subprocess.run(["qsub", str(path)], check=True)


if __name__ == "__main__":
    main()
