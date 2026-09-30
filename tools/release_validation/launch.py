"""Generate (and optionally submit) the release-validation matrix runs.

    python tools/release_validation/launch.py --repo . [--members a,b] \
        [--tag SHA] [--resume] [--submit]                       # Derecho, PBS
    python tools/release_validation/launch.py --site nautilus [--members a,b] \
        [--pin jcm=REF] [--tag T] [--resume] [--submit]         # Kubernetes
    python tools/release_validation/launch.py --site nautilus --fetch \
        [--members a,b] [--tag T]            # copy finished runs off the PVC

One member, one arm (the #682 retune): ``--days N``, ``--init STATE``
(``init=from_state``), ``--suffix NAME`` and ``--extra OVERRIDE ...`` (raw
Hydra overrides appended last); the suffix lands in the run's name, so an arm
has its own rundir, Job and label.

Each member of ``matrix.yaml`` references a validated preset in
``tools/benchmark.py``'s ``PRESETS`` (the single home of known-good
override sets) and becomes a job running a full-output year on one A100 — a
PBS job on Derecho, or a Kubernetes Job built by the production-run engine in
``.claude/skills/kubernetes-jcm-runs/scripts/mkrun.py``. Both sites take one
override list from :func:`member_overrides`. Per-grid inputs resolve
automatically inside jcm (``terrain=auto``, ``forcing.ozone_file=auto``) and are
PREFETCHED here, on the submitting (networked) node, so a member whose inputs
are unavailable refuses at submit time instead of after hours of GPU; JAM
members additionally get their aux inputs as concrete paths — the present-day
climatological mirror bundles (``emissions_pd``, ``dms``, ``oxidants_pd`` and
the five Tegen dust bundles). A PBS compute node has no network, so those are
local cache paths fetched here on the login node; a pod has network, so they
stay ``hf://`` paths it resolves at the same mirror commit.
Each run directory is namespaced by ``--tag`` (default: the launched
repo's HEAD short SHA), because a release-validation member is a *fresh*
year: a fixed rundir let a second matrix run silently resume the first
one's checkpoint. Health-check finished runs with ``health.py``; run
``scm_check.py`` for the SCM member (CPU, no PBS needed).
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
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


def dust_overrides(token: str, fetch_here: bool = True) -> list[str]:
    """Fetch the five dust bundles HERE and pass their local cache paths.

    ``auto`` would resolve — and therefore download — inside the PBS job, where
    there is no internet: the member would abort before integrating on any
    cache that was not already warm. Fetching on the login node at generation
    time and baking in concrete paths is the same contract the rest of this
    launcher uses for its aux inputs.

    ``fetch_here=False`` names the same bundles as ``hf://`` paths instead, for
    a job that has network (a Kubernetes pod): a local cache path would not
    exist there, and the pod resolves the URL at the mirror commit the job
    exports. The prefetch still checks every one of them before submitting.
    """
    from jcm.data import mirror_manifest as mm
    from jcm.data.remote import fetch
    manifest = mm.load_manifest()
    out = []
    for key in _DUST_KEYS:
        product = mm.product_for_key(manifest, key)
        rel = mm.bundle_path(manifest, product, token, None)
        if not fetch_here:
            out.append(f"forcing.{key}=hf://{rel}")
            continue
        try:
            out.append(f"forcing.{key}={fetch(rel)}")
        except Exception as exc:                              # noqa: BLE001
            raise SystemExit(
                f"could not fetch the dust bundle {rel} for {token}: {exc}. "
                "Release validation generates jobs on a login node precisely "
                "so the compute nodes need no network; fix the fetch here "
                "rather than letting the member abort in the queue.") from exc
    return out


#: JAM aux inputs besides dust: the present-day (2005-2014) climatology
#: products ``auto`` resolves to (``emissions_pd``, ``dms``, ``oxidants_pd``).
_JAM_PD_KEYS = ("emissions_file", "dms_file", "oxidants_file")


def jam_aux(grid: str, levels: str, fetch_here: bool = True) -> list[str]:
    """Present-day climatological JAM inputs, fetched HERE as concrete paths.

    The same ``*_pd`` mirror climatologies the presets resolve with ``auto``
    (CEDS/BB4CMIP emissions and PD oxidants averaged over 2005-2014, the Lana
    DMS climatology) plus the five dust bundles — so a release member runs
    under the same climatological present-day AMIP forcing as ``forcing_pd``.
    Fetched on the (networked) generating node and baked in as local cache
    paths, so the compute job needs no network. ``fetch_here=False``: the
    same bundles as ``hf://`` paths, for a job that resolves them itself (see
    :func:`dust_overrides`).
    """
    from jcm.data import mirror_manifest as mm
    from jcm.data.remote import fetch
    token = grid.split("_")[1]        # echam_t63_l95_hybrid -> t63
    nlev = int(levels.lstrip("l"))    # l95 -> 95
    manifest = mm.load_manifest()
    out = []
    for key in _JAM_PD_KEYS:
        product = mm.product_for_key(manifest, key)
        rel = mm.bundle_path(manifest, product, token, nlev)
        if not fetch_here:
            out.append(f"forcing.{key}=hf://{rel}")
            continue
        try:
            out.append(f"forcing.{key}={fetch(rel)}")
        except Exception as exc:                              # noqa: BLE001
            raise SystemExit(
                f"could not fetch the present-day JAM input {rel} for "
                f"{token}/{levels}: {exc}. Generate the jobs on a node with "
                "network so the compute nodes need none.") from exc
    return out + dust_overrides(token, fetch_here)


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
    """Return ``(commit, opt_in, source)`` for one member (recorded later).

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

    from jcm.data.remote import (
        REVISION_ENV, requested_revision, revision_source)
    commit, source = requested_revision(), revision_source()
    record = Path(rundir) / MIRROR_RECORD
    if resume and record.exists():
        recorded = json.loads(record.read_text())["commit"]
        if recorded == commit or (source == "pinned" and not force):
            # Re-issuing a forced resume on the commit it already recorded
            # (a forced launch that died before its first checkpoint)
            # regenerates the opt-in.
            return recorded, force and recorded == commit, source
        if not force:
            raise SystemExit(
                f"{REVISION_ENV}={commit} differs from the mirror commit "
                f"{recorded} recorded in {record}; resuming on it would change "
                "the run's boundary inputs mid-integration. Unset it to "
                "continue on the recorded commit, or pass "
                "--force-mirror-revision to switch deliberately.")
    return commit, force and resume, source


def write_mirror_record(rundir: str, commit: str, source: str) -> None:
    """Record ``commit`` in ``rundir`` (after the preflight has passed)."""
    import json

    record = Path(rundir) / MIRROR_RECORD
    if record.exists() and json.loads(record.read_text())["commit"] == commit:
        return      # a resume on the recorded commit keeps the launch record
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({
        "requested": commit, "source": source, "commit": commit,
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds")}, indent=1))


def overrides(name: str, m: dict, d: dict, rundir: str,
              fetch_here: bool = True) -> list[str]:
    """One matrix member's run overrides (``fetch_here``: see :func:`jam_aux`).

    ``jcm-monitor`` builds its runs from this function too, which is why the
    arm settings are layered on by :func:`member_overrides` rather than here.
    """
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
           # The release matrix scores per-chunk files (health.py), so it
           # opts out of run/longrun.yaml's calendar-month stream (#901).
           "run.monthly_means=false", "run.save_chunks=true",
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
        ovs += jam_aux(grid, m["jam_inputs"], fetch_here)
    return ovs


def init_overrides(init: str | None, fetch_here: bool) -> list[str]:
    """``init=from_state`` from ``--init`` (a warm-started arm), else nothing.

    The state is restored but the clock starts at zero, so ``--days`` counts
    from the warm start (``jcm/config/init/from_state.yaml``). A PBS job has no
    network, so an ``hf://`` state is fetched here and passed as its cache
    path, like the aux inputs; a pod resolves the URL itself, and a path is
    passed through for jcm to check at startup (a ``/runs/...`` state lives on
    the cluster volume, which this node cannot see).
    """
    if not init:
        return []
    if fetch_here and init.startswith("hf://"):
        from jcm.data.remote import fetch
        try:
            init = fetch(init[len("hf://"):])
        except Exception as exc:                              # noqa: BLE001
            raise SystemExit(f"could not fetch --init {init}: {exc}") from exc
    elif fetch_here and not Path(init).exists():
        raise SystemExit(f"--init {init} does not exist here, and a PBS job "
                         "reads the same filesystem.")
    return ["init=from_state", f"init.file={init}"]


def member_overrides(run: str, m: dict, d: dict, rundir: str, *,
                     init: str | None = None, extra=(),
                     fetch_here: bool = True) -> list[str]:
    """Return the complete override list one job runs, on either site.

    The member's overrides, Hydra's run directory in the rundir (so
    ``.hydra/`` lands next to the output, where scoring reads the run's own
    configuration), the warm start, and last the arm's raw ``extra``
    overrides — last, so an arm's setting wins over anything before it.
    """
    return (overrides(run, m, d, rundir, fetch_here)
            + [f"hydra.run.dir={rundir}"]
            + init_overrides(init, fetch_here) + list(extra))


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


def run_name(member: str, tag: str, suffix: str | None = None) -> str:
    """``mx_<member>_<tag>[_<suffix>]``: the rundir, job and label of one run.

    The suffix is what separates the arms of one member under one tag, so
    each arm writes its own rundir and checkpoint.
    """
    return (f"mx_{member.replace('-', '_')}_{tag}"
            + (f"_{suffix}" if suffix else ""))


def check_suffix(suffix: str | None) -> str | None:
    """Return ``--suffix`` when it is safe to embed (the tag's character set)."""
    if suffix is not None and (not suffix or TAG_UNSAFE.search(suffix)):
        raise SystemExit(
            f"--suffix {suffix!r} is not usable: it extends the run name "
            "(rundir, job name, label), so it must be non-empty and made only "
            "of letters, digits and underscores.")
    return suffix


def prefetchable(ovs: list[str]) -> list[str]:
    """``ovs`` without the ``hydra.*`` keys, which the input walk composes without."""
    return [o for o in ovs if not o.startswith("hydra.")]


# ---------------------------------------------------------------------------
# Kubernetes (--site nautilus). The Job is the one mkrun.py builds for every
# production run (clone pinned code, GPU check, checkpoint resume, health and
# day-count gates); this door supplies the member's overrides, the pinned
# commit's own environment and a guard on the rundir.

_K8S_SCRIPTS = (Path(__file__).resolve().parents[2] / ".claude" / "skills"
                / "kubernetes-jcm-runs" / "scripts")
if str(_K8S_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_K8S_SCRIPTS))
import fetch_run  # noqa: E402  (kubernetes-jcm-runs/scripts)
import mkrun  # noqa: E402
import sites  # noqa: E402

#: Where every pod clones jcm from; a pin must be a commit this remote has.
JCM_URL = mkrun.REPOS["jcm"][0]
#: The Job-name prefix mkrun.py gives production runs, so matrix members list
#: beside them; ``--job-prefix`` sets another for a campaign that shares the
#: namespace and wants its Jobs told apart.
JOB_PREFIX = "jcm-run"
#: The launch definition, kept in the local record dir and written by the pod
#: into the run directory on the volume.
LAUNCH_RECORD = "launch.json"
_DNS_LABEL = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?\Z")

#: The versions a pod records in PROVENANCE on every attempt.
_RECORDED_DISTS = ("jcm", "jax", "jaxlib", "jax-cuda12-plugin", "jax-rrtmgp",
                   "mam4-jax", "dinosaur", "flax")


def pinned_setup(rundir: str) -> str:
    """Shell that installs the pinned commit's own dependency set in the pod.

    The image is built from a jcm release older than the commit under
    validation (its jax-rrtmgp and dinosaur are below the commit's pins, and
    it has no MAM4-JAX), and a validation run validates that commit under its
    pins: requirements.txt plus the ``mam4`` extra (MAM4-JAX pinned exactly),
    which is what CI installs. The image's CUDA jax is held fixed by a
    constraint, so a pin needing another jax fails the install loudly instead
    of replacing the CUDA wheels with CPU ones; the GPU check that follows is
    the backstop either way. The resolved versions go into the rundir's
    PROVENANCE on every attempt, because an eviction retry can land on a
    newer ``:latest`` image.
    """
    dists = ", ".join(repr(d) for d in _RECORDED_DISTS)
    return f"""# The pinned commit's own dependency set (requirements.txt + the mam4 extra),
# not the image's older release; the image's CUDA jax build is held by a
# constraint so no pin can quietly replace it (the GPU check below backs that).
pip freeze 2>/dev/null | grep -iE '^(jax|jaxlib|jax[-_]cuda12[-_]plugin|jax[-_]cuda12[-_]pjrt)==' > /tmp/cuda-jax.txt || true
pip install --no-cache-dir --disable-pip-version-check -c /tmp/cuda-jax.txt -e '/work/jcm[mam4]' 2>&1 | tail -2
python - <<'PYDEPS' | tee -a {rundir}/PROVENANCE
from importlib import metadata
def version(d):
    try:
        return metadata.version(d)
    except metadata.PackageNotFoundError:
        return "absent"
print("deps:", " ".join(f"{{d}}=={{version(d)}}" for d in ({dists})))
PYDEPS"""


def _git(repo, *args) -> str | None:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                       text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def remote_refs(url: str) -> dict[str, str]:
    """``{ref: sha}`` for every branch and tag on ``url``; a tag maps to its commit."""
    r = subprocess.run(["git", "ls-remote", "--heads", "--tags", url],
                       capture_output=True, text=True, timeout=120)
    if r.returncode:
        raise SystemExit(f"git ls-remote {url} failed: {r.stderr.strip()}")
    refs = {}
    for line in r.stdout.splitlines():
        sha, _, name = line.partition("\t")
        # ``<tag>^{}`` (the peeled commit) follows ``<tag>`` and replaces it.
        refs[name.removesuffix("^{}")] = sha
    return refs


def resolve_pin(repo, ref: str | None = None, url: str = JCM_URL) -> str:
    """Return the full SHA of the jcm commit the pod will clone; refuse one ``url`` lacks.

    The pod clones from GitHub, not from this checkout, so an unpushed commit
    fails only after a node has been found and the image pulled. A branch or
    tag name means the one on ``url`` (what mkrun.py resolves); a SHA, HEAD
    (the default: the launched checkout) or a local-only name is resolved
    here, and must be reachable from a branch or tag on ``url``. Reachability
    is decided from this clone's objects, so a commit pushed since the last
    ``git fetch`` reads as unpushed until it is fetched.
    """
    refs = remote_refs(url)
    if ref not in (None, "HEAD"):
        for name in (f"refs/heads/{ref}", f"refs/tags/{ref}"):
            if name in refs:
                return refs[name]
    sha = _git(repo, "rev-parse", "--verify", "--quiet",
               f"{ref or 'HEAD'}^{{commit}}")
    if not sha:
        raise SystemExit(f"cannot resolve --pin jcm={ref} as a branch or tag "
                         f"on {url} or a commit in {repo} (a commit that is "
                         "on GitHub but not in this clone needs a git fetch "
                         "first).")
    if sha in refs.values():
        return sha
    r = subprocess.run(["git", "-C", str(repo), "rev-list", "--ignore-missing",
                        "--max-count=1", sha, "--not", *sorted(set(refs.values()))],
                       capture_output=True, text=True)
    if r.returncode == 0 and not r.stdout.strip():
        return sha
    raise SystemExit(
        f"jcm {ref or 'HEAD'} = {sha[:12]} is not on {url}: no branch or tag "
        "there contains it. The pod clones from GitHub, so push it first (or "
        "`git fetch` if it already is — reachability is checked against this "
        "clone's copy of those refs).")


def job_name(prefix: str, run: str) -> str:
    """Return the Kubernetes Job name for ``run``; refuse rather than truncate.

    A Job name must be a DNS label (at most 63 of ``a-z0-9-``, alphanumeric at
    both ends). Truncating would fold two long arm names onto one Job.
    """
    name = f"{prefix}-{run}".lower().replace("_", "-")
    if len(name) > 63 or not _DNS_LABEL.match(name):
        raise SystemExit(
            f"Job name {name!r} ({len(name)} characters) is not a valid "
            "Kubernetes name: at most 63 of a-z, 0-9 and '-', alphanumeric at "
            "both ends. Shorten --suffix, --tag or --job-prefix.")
    return name


def launch_definition(member: str, run: str, job: str, image: str,
                      pins: dict, days: int, ovs: list[str]) -> dict:
    """Everything that defines what one Kubernetes run integrates, with its digest.

    The code pin, the image, the override list, the target length and the
    Job that runs it. The digest is what the pod compares against the
    rundir's own record (:func:`rundir_guard`), so two launches that would
    integrate different things can never share a run directory. The mirror
    commit is deliberately not in it: jcm itself refuses a checkpoint written
    at another commit unless the job opts in (``--force-mirror-revision``).
    """
    body = {"member": member, "run": run, "job": job, "image": image,
            "pins": pins, "days": days, "overrides": list(ovs)}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode())
    return {**body, "digest": digest.hexdigest()[:16]}


def read_launch(local: Path) -> dict | None:
    """Return the launch definition recorded in ``local`` (None: there is none)."""
    record = Path(local) / LAUNCH_RECORD
    return json.loads(record.read_text()) if record.exists() else None


def rundir_guard(rundir: str, checkpoint: str, defn: dict) -> str:
    """Shell that ties the volume's run directory to ONE launch definition.

    The first attempt of a launch writes the definition into the rundir; every
    later attempt, eviction retry or ``--resume`` of the same definition
    passes, and a Job with a different definition — a reused tag launched
    from another machine, or relaunched with other overrides after the first
    Job was deleted — is refused instead of silently continuing someone
    else's integration and reporting it as its own (#701). The generating
    node cannot see the volume, so this check has to run in the pod; it runs
    before cloning, so a refusal costs seconds. The record is written to a
    temporary name and renamed, so a pod killed mid-write cannot leave a
    truncated record that would refuse the launch's own retry.
    """
    record = json.dumps(defn, indent=1, sort_keys=True)
    return f"""if [ -f {rundir}/{LAUNCH_RECORD} ]; then
  if ! grep -qF '"digest": "{defn["digest"]}"' {rundir}/{LAUNCH_RECORD}; then
    echo "FATAL: {rundir} holds a different launch (its {LAUNCH_RECORD}); refusing"
    echo "       to integrate this one on top of it. Resume that launch as"
    echo "       recorded, or launch this one under a new --tag or --suffix."
    exit 1
  fi
elif [ -f {checkpoint} ]; then
  echo "FATAL: {checkpoint} exists but no {LAUNCH_RECORD} records which launch"
  echo "       wrote it; refusing to resume an integration of unknown origin."
  exit 1
else
  cat > {rundir}/.{LAUNCH_RECORD}.tmp <<'LAUNCH'
{record}
LAUNCH
  mv {rundir}/.{LAUNCH_RECORD}.tmp {rundir}/{LAUNCH_RECORD}
fi
"""


def k8s_job(defn: dict, site: dict, mirror: tuple, retries: int,
            memory: str = "64Gi") -> dict:
    """Build the Kubernetes Job for one launch definition.

    ``mirror`` is :func:`mirror_commit`'s ``(commit, opt_in, source)``.
    ``retries`` and ``memory`` are how the Job runs, not what it integrates,
    so they are not part of the definition and a ``--resume`` may change them.
    """
    rundir = f"/runs/{defn['run']}"
    checkpoint = f"{rundir}/checkpoint.msgpack"
    commit, optin = mirror[0], mirror[1]
    env = [{"name": "JCM_MIRROR_REVISION", "value": commit}]
    if optin:
        env.append({"name": "JCM_ALLOW_MIRROR_REVISION_CHANGE", "value": "1"})
    env += [
        # The pod has the card to itself; the PBS job sets the same fraction.
        {"name": "XLA_PYTHON_CLIENT_MEM_FRACTION", "value": "0.93"},
        # The Job's log is the tee'd stdout; unbuffered, so `kubectl logs -f`
        # follows the run and an evicted attempt keeps its last lines.
        {"name": "PYTHONUNBUFFERED", "value": "1"},
    ]
    return mkrun.job_manifest(
        site={**site, "image": defn["image"]}, job_name=defn["job"],
        label=defn["run"], rundir=rundir, checkpoint=checkpoint,
        overrides=defn["overrides"], days=defn["days"],
        resolved={"jcm": (JCM_URL, defn["pins"]["jcm"])},
        setup=pinned_setup(rundir), python_env="MAM4_JAX_ENABLE_X64=0",
        retries=retries, gpus=1, cpu=8, memory=memory, env=tuple(env),
        guard=rundir_guard(rundir, checkpoint, defn))


def _kubectl(site: dict, *args, input=None, timeout=300):
    return subprocess.run(["kubectl", "-n", site["namespace"], *args],
                          input=input, capture_output=True, text=True,
                          timeout=timeout)


def job_status(site: dict, name: str) -> str | None:
    """``None`` when the Job does not exist, else complete/failed/active."""
    r = _kubectl(site, "get", "job", name, "-o", "json")
    if r.returncode:
        if "NotFound" in r.stderr:
            return None
        raise SystemExit(f"kubectl get job {name} failed: {r.stderr.strip()}")
    conds = {c["type"] for c in json.loads(r.stdout).get("status", {})
             .get("conditions", []) if c.get("status") == "True"}
    # A Job with no condition yet (pods pending or running) is unfinished.
    return ("complete" if "Complete" in conds else
            "failed" if "Failed" in conds else "active")


def submit_jobs(site: dict, jobs: list[dict], resume: bool) -> None:
    """Apply ``jobs``, having vetted all of them first.

    Jobs are immutable, so a resume must replace the finished Job of the same
    name (its output stays on the volume, its log in run.log); a running Job
    is never touched, and a fresh launch never replaces anything.
    """
    replace = []
    for job in jobs:
        name = job["metadata"]["name"]
        status = job_status(site, name)
        if status == "active":
            raise SystemExit(f"Job {name} is still running; nothing submitted.")
        if status is not None and not resume:
            raise SystemExit(
                f"Job {name} already exists ({status}); nothing submitted. "
                "--resume continues its run from the checkpoint; a fresh run "
                "needs a new --tag or --suffix.")
        if status is not None:
            replace.append((name, status))
    for name, status in replace:
        r = _kubectl(site, "delete", "job", name, "--wait=true")
        if r.returncode:
            raise SystemExit(f"kubectl delete job {name} failed: "
                             f"{r.stderr.strip()}")
        print(f"# replaced the {status} Job {name} to resume its run",
              file=sys.stderr)
    for job in jobs:
        r = _kubectl(site, "apply", "-f", "-", input=json.dumps(job))
        if r.returncode:
            raise SystemExit(f"kubectl apply of {job['metadata']['name']} "
                             f"failed: {r.stderr.strip()}")
        print(f"# {r.stdout.strip()}", file=sys.stderr)


def _main_k8s(a, cfg: dict, repo: str, scratch: str) -> None:
    """Emit (and optionally apply) one Kubernetes Job per member; or --fetch."""
    site = sites.get(a.site)
    d = dict(cfg["defaults"], **({"days": a.days} if a.days else {}))
    wanted = a.members.split(",") if a.members else list(cfg["members"])
    pins = dict(x.split("=", 1) for x in a.pin)
    if set(pins) - {"jcm"}:
        raise SystemExit("--pin takes only jcm=<ref>: every other dependency "
                         "comes from the pinned commit's own requirements.")
    if (a.resume or a.fetch) and (a.pin or a.days or a.init or a.extra):
        raise SystemExit(
            "--resume and --fetch act on a recorded launch, whose definition "
            "(--pin, --days, --init, --extra) cannot change; a different "
            "definition is a new run: give it a new --tag or --suffix.")
    if a.fetch and (a.submit or a.resume):
        raise SystemExit("--fetch copies runs; it does not launch them.")
    if a.with_checkpoints and not a.fetch:
        raise SystemExit("--with-checkpoints is an option of --fetch.")
    sha = None
    if not (a.resume or a.fetch):
        sha = resolve_pin(repo, pins.get("jcm"))
        print(f"# jcm pinned at {sha}", file=sys.stderr)
    # The tag names the code under test, as on the PBS path: the launched
    # checkout's HEAD by default, the pinned commit when one is given.
    if a.tag is not None:
        run_tag = check_tag(a.tag)
    elif sha and "jcm" in pins:
        run_tag = TAG_UNSAFE.sub(
            "_", _git(repo, "rev-parse", "--short", sha) or sha[:8])
    else:
        run_tag = repo_tag(repo)
    suffix = check_suffix(a.suffix)
    print(f"# run tag: {run_tag}", file=sys.stderr)
    if sha and "jcm" in pins and sha != _git(repo, "rev-parse", "HEAD"):
        # Warned, not refused: pinning a commit other than the checkout is
        # legitimate, but matrix.yaml's lengths and PRESETS' names are read
        # here, so they are this checkout's, not the pin's.
        print(f"# NOTE: the code is {sha[:12]} but the member definitions "
              "(matrix.yaml, PRESETS) come from this checkout's HEAD",
              file=sys.stderr)

    plan = []
    for member in wanted:
        run = run_name(member, run_tag, suffix)
        local = Path(scratch) / "jam_runs" / run
        recorded = read_launch(local)
        if a.fetch:
            plan.append((member, local, recorded or {"run": run, "job": job_name(
                a.job_prefix, run)}))
            continue
        if a.resume:
            if recorded is None:
                raise SystemExit(
                    f"no recorded launch for {run} in {local}: --resume "
                    "continues a launch made from this machine; pass the "
                    "--tag (and --suffix) it was launched under.")
            plan.append((member, local, recorded))
            continue
        rundir = f"/runs/{run}"
        ovs = member_overrides(run, cfg["members"][member], d, rundir,
                               init=a.init, extra=a.extra, fetch_here=False)
        defn = launch_definition(member, run, job_name(a.job_prefix, run),
                                 site["image"], {"jcm": sha}, int(d["days"]),
                                 ovs)
        if recorded is not None and recorded["digest"] != defn["digest"]:
            raise SystemExit(
                f"{run} was already launched with a different definition "
                f"({local / LAUNCH_RECORD}). --resume continues it as "
                "recorded; a new definition needs a new --tag or --suffix "
                "(or delete that record if it was never submitted).")
        plan.append((member, local, defn))

    if a.fetch:
        problems = {}
        for member, local, defn in plan:
            status = job_status(site, defn["job"])
            if status == "active":
                print(f"# NOTE: Job {defn['job']} is still running; this copy "
                      "is a snapshot of an unfinished run", file=sys.stderr)
            bad = fetch_run.fetch_run(
                defn["run"], local, site=site, pod=f"{defn['job']}-fetch",
                with_checkpoints=a.with_checkpoints)
            if bad:
                problems[member] = bad
        if problems:
            raise SystemExit("incomplete copies:\n" + "\n".join(
                f"  {m}: " + "; ".join(p) for m, p in problems.items()))
        return

    # The same mirror-commit rules as the PBS path, recorded in the local
    # record dir: the pod exports the commit the launch was generated at.
    mirror = {defn["run"]: mirror_commit(str(local), a.resume,
                                         a.force_mirror_revision)
              for _, local, defn in plan}
    commits = {m[0] for m in mirror.values()}
    if len(commits) > 1:
        raise SystemExit(
            f"these members read different mirror commits "
            f"({', '.join(sorted(commits))}); launch them separately.")
    os.environ["JCM_MIRROR_REVISION"] = commits.pop()
    if not a.no_prefetch:
        missing = {member: bad for member, _, defn in plan
                   if (bad := prefetch(prefetchable(defn["overrides"])))}
        if missing:
            report = "\n".join(f"  {name}:\n    " + "\n    ".join(paths)
                               for name, paths in missing.items())
            raise SystemExit(
                "these members reference inputs that are not available "
                f"here:\n{report}\nThe pod would fail on them too; stage them "
                "per jcm/data/mirror/SOURCES.md, or pass --no-prefetch.")

    jobs = []
    for _, local, defn in plan:
        local.mkdir(parents=True, exist_ok=True)
        write_mirror_record(str(local), *mirror[defn["run"]][::2])
        (local / LAUNCH_RECORD).write_text(
            json.dumps(defn, indent=1, sort_keys=True))
        job = k8s_job(defn, site, mirror[defn["run"]], a.retries, a.memory)
        (local / "job.json").write_text(json.dumps(job, indent=2))
        print(f"# {defn['member']}: Job {defn['job']} -> PVC "
              f"{site['runs_pvc']}:/runs/{defn['run']} (record: {local})",
              file=sys.stderr)
        jobs.append(job)
    if a.submit:
        submit_jobs(site, jobs, a.resume)
    else:
        print("\n---\n".join(json.dumps(j, indent=2) for j in jobs))


def main(argv=None):
    """Write (and optionally submit) one job per matrix member."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--site", default="derecho",
                    choices=["derecho", *sorted(sites.SITES)],
                    help="derecho: PBS scripts under <repo>/runs; a "
                         "Kubernetes site profile (sites.py): Job manifests")
    ap.add_argument("--members", default=None,
                    help="comma-separated subset (default: all)")
    ap.add_argument("--tag", default=None,
                    help="rundir/job namespace, letters/digits/underscore "
                         "only (default: repo HEAD short SHA; with --pin, "
                         "the pinned commit's)")
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
    arm = ap.add_argument_group("one arm of a member (e.g. the #682 retune)")
    arm.add_argument("--days", type=int, default=None,
                     help="run length instead of matrix.yaml's")
    arm.add_argument("--init", default=None, metavar="STATE",
                     help="warm start: init=from_state init.file=STATE "
                          "(hf://..., or /runs/... on the cluster volume)")
    arm.add_argument("--suffix", default=None,
                     help="appended to the run name, so the arm has its own "
                          "rundir, job and label")
    arm.add_argument("--extra", nargs="+", action="extend", default=[],
                     metavar="OVERRIDE",
                     help="raw Hydra overrides appended last, e.g. "
                          "+physics.convection.trigger_cape=150.0")
    k8s = ap.add_argument_group("Kubernetes sites")
    k8s.add_argument("--pin", action="append", default=[], metavar="jcm=REF",
                     help="the jcm commit the pod clones (default: the "
                          "launched repo's HEAD); must be on GitHub")
    k8s.add_argument("--retries", type=int, default=20,
                     help="Job backoffLimit; each retry resumes from the "
                          "checkpoint, so this is the eviction budget")
    k8s.add_argument("--memory", default="64Gi",
                     help="the pod's host memory (mkrun.py's default); an "
                          "OOM-killed container restarts at its checkpoint "
                          "and dies there again, so raise it and --resume")
    k8s.add_argument("--job-prefix", default=JOB_PREFIX,
                     help=f"Job-name prefix (default {JOB_PREFIX})")
    k8s.add_argument("--fetch", action="store_true",
                     help="copy each member's run directory off the volume "
                          "into $SCRATCH/jam_runs/<run>, for health.py")
    k8s.add_argument("--with-checkpoints", action="store_true",
                     help="with --fetch, also copy the checkpoints (a "
                          "year's archives run to GBs)")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(HERE / "matrix.yaml"))
    repo = str(Path(a.repo).resolve())
    scratch = os.environ.get("SCRATCH", f"{HOME}/scratch")
    if a.site != "derecho":
        return _main_k8s(a, cfg, repo, scratch)
    if a.pin or a.fetch or a.with_checkpoints:
        raise SystemExit("--pin, --fetch and --with-checkpoints are for a "
                         "Kubernetes --site; a PBS job runs --repo as it is.")

    d = dict(cfg["defaults"], **({"days": a.days} if a.days else {}))
    venv = os.environ.get("JCM_VENV", f"{HOME}/.venvs/jaxgcm")
    outdir = Path(repo) / "runs"
    outdir.mkdir(exist_ok=True)

    run_tag = check_tag(a.tag) if a.tag is not None else repo_tag(repo)
    suffix = check_suffix(a.suffix)
    print(f"run tag: {run_tag}")

    wanted = a.members.split(",") if a.members else list(cfg["members"])
    # The tag namespaces the rundir, the job name, the outputs and the log
    # together, so one launch's artefacts never mix with another's. Vet every
    # member before writing anything: a matrix launch that would resume a
    # previous one should fail whole, not half-submitted.
    plan = [(name, run_name(name, run_tag, suffix)) for name in wanted]
    for _, tag in plan:
        check_fresh(f"{scratch}/jam_runs/{tag}", a.resume)
    # Each member's mirror commit, resolved (and recorded) before the
    # prefetch; the job exports it, as a PBS job does not inherit this shell.
    mirror = {tag: mirror_commit(f"{scratch}/jam_runs/{tag}", a.resume,
                                 a.force_mirror_revision)
              for _, tag in plan}
    # This process reads the mirror at one commit (jcm.data.remote), so a
    # launch's members must share it; resume differing ones separately.
    commits = {m[0] for m in mirror.values()}
    if len(commits) > 1:
        raise SystemExit(
            f"these members read different mirror commits "
            f"({', '.join(sorted(commits))}); launch them separately.")
    os.environ["JCM_MIRROR_REVISION"] = commits.pop()

    # A PBS run's complete override list: JAM aux inputs and an hf:// warm
    # start are fetched here, as the compute node has no network.
    ovs = {tag: member_overrides(tag, cfg["members"][name], d,
                                 f"{scratch}/jam_runs/{tag}", init=a.init,
                                 extra=a.extra)
           for name, tag in plan}

    # Preflight every member's inputs before writing any job: a matrix launch
    # that cannot resolve an input should fail whole, on the node that still
    # has network, not member by member inside a queued job.
    if not a.no_prefetch:
        missing = {}
        for name, tag in plan:
            unavailable = prefetch(prefetchable(ovs[tag]))
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
        write_mirror_record(f"{scratch}/jam_runs/{tag}", *mirror[tag][::2])

    for name, tag in plan:
        m = cfg["members"][name]
        rundir = f"{scratch}/jam_runs/{tag}"
        job = PBS.format(
            name=tag, account=a.account, hours=m.get("hours", d["hours"]),
            logdir=str(outdir), venv=venv, repo=repo, rundir=rundir,
            # Quoted, so an --extra carrying shell metacharacters reaches
            # Hydra as written; a plain key=value passes through unchanged.
            ovs=" \\\n    ".join(shlex.quote(o) for o in ovs[tag]),
            marker=f"{tag.upper()}_COMPLETE",
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
