#!/usr/bin/env python
"""Generate a Nautilus Job for a PRODUCTION jcm run (output kept, resumable).

    python mkrun.py --name pd-year | kubectl apply -f -          # 12 calendar months
    python mkrun.py --name pd-year --months 24 | kubectl apply -f -
    python mkrun.py --name bench-90d --days 90 | kubectl apply -f -  # fixed length

By default a run is 12 calendar months from ``--start-time`` under the
present-day climatological AMIP forcing (the mirror's ``*_pd`` bundles: PCMDI
AMIP SST/sea ice and CEDS/BB4CMIP emissions averaged 2005-2014, PD ozone and
oxidants), written as calendar-month means (``run.monthly_means``, #901) —
``<name>_monthly_YYYY-MM.nc`` — from daily interval means that are streamed,
not kept.

Different from `mkjob.py` in every way that matters:

* the OUTPUT is the point, so netCDF goes to a persistent volume rather than
  a pod-local scratch that is thrown away;
* the run must survive eviction. Nautilus pods are not guaranteed a node for
  days, so the Job restarts and jcm resumes from its checkpoint instead of
  starting the year again;
* the health gate STAYS ON. A benchmark may deliberately measure an unstable
  configuration; a production year that goes NaN should stop, not burn a
  week of GPU producing garbage.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shlex
import subprocess
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import sites as site_profile  # noqa: E402  (local module)


REPOS = {
    "jcm": ("https://github.com/climate-analytics-lab/jax-gcm",
            "feat/derecho-runs-skill"),
    "jax-rrtmgp": ("https://github.com/climate-analytics-lab/jax-rrtmgp",
                   "main"),
    # Required by every echam-jam preset and NOT present in the published
    # image, so it has to be cloned like the others. Pinned to main: the
    # configure_gas_netprod work the dev box ran from a local branch was
    # merged (and developed further) upstream, so main is the same code plus
    # later fixes — verified by diffing, the only differences were comments.
    "mam4-jax": ("https://github.com/reflective-org/MAM4-JAX", "main"),
}



def resolve_refs(pins: dict) -> dict:
    """Resolve each ref to the SHA it currently points at.

    A production run that is resumed days later MUST come back on the same
    code — cloning a branch name would silently restart it on whatever has
    since been merged, half way through a year.
    """
    out = {}
    for d, (url, ref) in REPOS.items():
        ref = pins.get(d, ref)
        r = subprocess.run(["git", "ls-remote", url, ref],
                           capture_output=True, text=True, timeout=120)
        sha = r.stdout.split()[0] if r.stdout.split() else (
            ref if len(ref) >= 7 and all(c in "0123456789abcdef" for c in ref)
            else None)
        if not sha:
            raise SystemExit(f"cannot resolve {ref!r} in {url}")
        out[d] = (url, sha)
    return out


def grid_tokens(grid: str) -> tuple[str, str]:
    """``echam_t63_l47_hybrid`` -> (``t63``, ``l47``), the mirror bundle keys."""
    parts = grid.split("_")
    if len(parts) < 3 or not parts[1].startswith("t") or not parts[2].startswith("l"):
        raise SystemExit(f"cannot derive mirror bundle names from grid {grid!r}; "
                         "expected echam_t<N>_l<M>_<vertical>")
    return parts[1], parts[2]


def forcing_overrides(a) -> list:
    """Present-day climatological AMIP inputs from the data mirror.

    Surface forcing, terrain and ozone are named explicitly (the surface file
    has no ``auto``; an ozone left at ``auto`` on a grid without a mirrored
    climatology falls back to the analytic profile, ~7.6x the tropospheric
    column). Emissions/DMS/oxidants/dust resolve ``auto`` to the same ``_pd``
    climatologies. ``--ozone`` overrides the ozone file.
    """
    token, lev = grid_tokens(a.grid)
    ozone = a.ozone or f"hf://bundles/{token}_{lev}/ozone_pd.nc"
    return ["terrain=from_file",
            f"terrain.file=hf://bundles/{token}/terrain.nc",
            "forcing=from_file",
            f"forcing.file=hf://bundles/{token}/forcing_pd.nc",
            f"forcing.ozone_file={ozone}"]


def target_days(a) -> int:
    """Sim-days the run must reach: --days, or --months from --start-time."""
    if a.days:
        return a.days
    start = datetime.date.fromisoformat(a.start_time)
    year, month0 = divmod(start.month - 1 + a.months, 12)
    return (start.replace(year=start.year + year, month=month0 + 1)
            - start).days


def length_overrides(a) -> list:
    length = f"run.total_time={a.days}" if a.days else \
        f"run.total_time={a.months}months"
    return [length, f"run.start_time={a.start_time}",
            f"run.chunk_days={a.chunk_days}",
            f"run.save_interval={a.save_interval}",
            "run.output_averages=true",
            f"run.monthly_means={'true' if a.monthly_means else 'false'}",
            f"run.save_chunks={'true' if a.save_chunks else 'false'}"]


def extra_pip_setup(site: dict) -> str:
    """Shell that installs what the jcm image lacks (``extra_pip``, sites.py).

    ``mkrun.py``'s ``setup`` for :func:`job_manifest`. A door that needs a
    different environment passes its own instead.
    """
    return f"""# MAM4-JAX declares diffrax and matplotlib; neither is in the jcm image, and
# the JAM condensation backend imports diffrax at module load. Installing it
# here can in principle drag jax with it, which would silently swap the CUDA
# build for a CPU one — so the install is followed by a hard GPU check rather
# than trusting it. A CPU fallback would "work" and report timings 100x slow.
pip install --no-cache-dir {' '.join(repr(x) for x in site['extra_pip'])} 2>&1 | tail -2"""


def checkpoint_of(overrides: list[str]) -> str:
    """Return the checkpoint jcm will write: the LAST ``run.checkpoint_path``.

    Hydra applies overrides in order, so a later ``++run.checkpoint_path`` (a
    door's ``--extra``) wins over the one the door composed.
    """
    ckpts = [o.split("=", 1)[1] for o in overrides
             if o.lstrip("+").startswith("run.checkpoint_path=")]
    if not ckpts:
        raise ValueError("the overrides set no run.checkpoint_path, so the run "
                         "could not resume after an eviction")
    return ckpts[-1]


def job_manifest(*, site: dict, job_name: str, label: str, rundir: str,
                 overrides: list[str], days: int,
                 resolved: dict, setup: str, python_env: str,
                 retries: int, gpus: int, cpu: int, memory: str,
                 gpu_product: str | None = None, env: tuple = (),
                 guard: str = "") -> dict:
    """Build the production-run Job; every door that keeps its output uses it.

    The Job clones ``resolved`` (``{dir: (url, sha)}``) at their pinned SHAs,
    runs ``setup`` (the shell lines that install what the image lacks), refuses
    to run on a CPU fallback, runs ``python -m jcm.main`` with ``overrides``
    (each shell-quoted; ``python_env`` is the ``VAR=value`` prefix for that
    command) teeing into ``<rundir>/run.log``, and then fails the Job unless
    THIS attempt stayed healthy and reached day ``days`` — because
    ``run_chunked`` returns normally when its health gate trips. Each retry
    resumes from the checkpoint the overrides name (:func:`checkpoint_of`),
    so ``retries`` (``backoffLimit``) is the eviction budget. ``guard`` is
    shell run right after the rundir exists and before anything is cloned,
    for a door that must refuse a rundir before spending the attempt; ``env``
    adds container env entries after the base ones.
    """
    checkpoint = checkpoint_of(overrides)
    clone = "\n".join(
        f'git clone --filter=blob:none --no-checkout {url} /work/{d} '
        f'&& git -C /work/{d} fetch --depth 1 origin {sha} '
        f'&& git -C /work/{d} checkout --detach {sha}'
        for d, (url, sha) in resolved.items()
    )
    # Quoted, so an override carrying shell metacharacters (a Hydra list
    # ``[a,b]``, quotes, ``~key``) reaches Hydra as written; the plain
    # ``key=value`` overrides every door emits pass through unchanged.
    overrides = " ".join(shlex.quote(o) for o in overrides)
    script = f"""set -euo pipefail
echo "=== node $NODE_NAME | $(nvidia-smi --query-gpu=name --format=csv,noheader) | attempt $(date -u +%FT%TZ) ==="
mkdir -p /work {rundir}
{guard}{clone}
cd /work/jcm
{setup}
python - <<'PYCHK'
import sys, jax
d = jax.devices()
print("jax devices after install:", d)
if not any(x.platform == "gpu" for x in d):
    sys.exit("FATAL: no GPU visible to jax after pip install — the CUDA "
             "build was replaced. Refusing to run: timings would be "
             "meaningless and the failure would look like a slow run.")
PYCHK
for d in {' '.join(resolved)}; do
  echo "$d @ $(git -C /work/$d rev-parse HEAD)" | tee -a {rundir}/PROVENANCE
done
# jcm resumes automatically when the checkpoint exists, so a pod that was
# evicted mid-year picks up from the last completed chunk rather than
# starting over. That is what makes a multi-day run viable here.
if [ -f "{checkpoint}" ]; then
  echo "=== resuming from $(ls -la {shlex.quote(checkpoint)} | awk '{{print $5}}') byte checkpoint ==="
fi
# run.log is append-only ACROSS pod restarts (that is what makes the
# eviction-resume design debuggable), so every gate below must read only THIS
# attempt's slice. Grepping the cumulative log gets both verdicts wrong:
#   * completion: an attempt that integrated NOTHING inherits the previous
#     attempt's "_day365.nc" line and the Job is marked Complete. That is how
#     a no-op resume — e.g. a stale or foreign checkpoint already at/past the
#     target — reports success having done no work.
#   * health: one bad chunk that a later attempt already recovered from fails
#     the Job forever.
# Record the byte offset first and slice from it.
ATTEMPT_START=$(stat -c%s "{rundir}/run.log" 2>/dev/null || echo 0)
set +e
{python_env} \\
  python -m jcm.main {overrides} 2>&1 | tee -a {rundir}/run.log
RC=${{PIPESTATUS[0]}}
set -e
tail -c +$((ATTEMPT_START + 1)) "{rundir}/run.log" > /tmp/attempt.log

# jcm.runners.run_chunked BREAKS OUT of its loop and returns NORMALLY when
# bail_on_unhealthy trips, so jcm.main exits 0 even though the year stopped
# at the first bad chunk. Without this check Kubernetes marks a 365-day Job
# Complete after 30 days of output — the worst kind of failure, because it
# looks like success. Verify the health verdict and the day count.
#
# Match the messages runners.py actually EMITS on a bad chunk ("atmosphere
# unhealthy at ...", "(unhealthy chunk)"), not a bare "unhealthy". The bare
# pattern also matched the Hydra config echo `bail_on_unhealthy: true`, which
# every run prints at INFO, so a perfectly healthy run failed its own gate —
# observed on the first nudged run, where 3 of 3 days completed with 0 NaN
# and the Job was still marked Failed.
if grep -qiE "atmosphere unhealthy|unhealthy chunk|NaN vars: *[1-9]" /tmp/attempt.log; then
  echo "FATAL: health gate tripped — run stopped early, not complete"
  grep -iE "atmosphere unhealthy|unhealthy chunk|NaN vars: *[1-9]" /tmp/attempt.log | tail -3
  exit 1
fi
# `|| true` is load-bearing under `set -euo pipefail`: "no output this
# attempt" is a state we must INSPECT, but a no-match grep exits 1 and
# pipefail propagates that out of the command substitution, which would abort
# the script before the empty-LAST branch below could run — marking a genuine
# completion-restart as failed. Same for RESUMED.
# Progress = the furthest chunk end this attempt reached: a saved chunk file
# (``_dayN.nc``) or, when only monthly means are written (save_chunks=false),
# the per-chunk health report ``Chunk K | Day N (...)``.
LAST=$( (grep -oE "_day[0-9]+\\.nc" /tmp/attempt.log | grep -oE "[0-9]+";
         grep -oE "\\| Day [0-9]+ \\(" /tmp/attempt.log | grep -oE "[0-9]+") \\
       | sort -n | tail -1 || true)
if [ -z "$LAST" ]; then
  # No output this attempt. Distinguish the one benign case — the run was
  # already finished and the pod merely restarted — from a no-op resume,
  # which must NOT look like success.
  RESUMED=$(grep -oE "Resumed from checkpoint .* at sim-day [0-9.]+" \\
            /tmp/attempt.log | grep -oE "[0-9.]+$" | tail -1 || true)
  # A restart from the final checkpoint still does work: it flushes the
  # final month when its file was not written before the pod died. That
  # flush writes no chunk file or health line, so only the exit code says
  # whether it succeeded — a failed flush must fail the Job, not pass as
  # "already complete" with the final month missing.
  if [ -n "$RESUMED" ] && [ "${{RESUMED%%.*}}" -ge {days} ] && [ "$RC" -ne 0 ]; then
    echo "FATAL: resumed at the final checkpoint (day $RESUMED) but jcm.main"
    echo "       exited rc=$RC — the pending final flush did not complete."
    exit $RC
  fi
  if [ -n "$RESUMED" ] && [ "${{RESUMED%%.*}}" -ge {days} ]; then
    echo "=== already complete: checkpoint at day $RESUMED of {days}, nothing to do ==="
    exit 0
  fi
  echo "FATAL: this attempt wrote no output and resumed at day ${{RESUMED:-0}}"
  echo "       of {days} — no progress made. Check for a stale or foreign"
  echo "       checkpoint at {checkpoint}."
  exit 1
fi
if [ "$LAST" -lt {days} ]; then
  echo "FATAL: reached day $LAST of {days} — incomplete"
  exit 1
fi
echo "=== finished $(date -u +%FT%TZ), day $LAST of {days}, rc=$RC ==="
exit $RC
"""
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": job_name, "namespace": site["namespace"],
                     "labels": {"jcm-run": label}},
        "spec": {
            # Survive eviction: each retry re-runs the script, which resumes
            # from the checkpoint. Contrast the benchmark generator, where a
            # retry would silently re-time on a different node.
            "backoffLimit": retries,
            # No TTL — production output and its Job history are kept until
            # deliberately removed.
            "template": {
                "spec": {
                    "restartPolicy": "OnFailure",
                    "nodeSelector": (
                        {"nvidia.com/gpu.product": gpu_product}
                        if gpu_product
                        else dict(site["gpu_selector"])),
                    "containers": [{
                        "name": "run", "image": site["image"],
                        "command": ["/bin/bash", "-c", script],
                        "env": [
                            {"name": "NODE_NAME", "valueFrom": {"fieldRef": {
                                "fieldPath": "spec.nodeName"}}},
                            {"name": "JAX_PLATFORMS", "value": "cuda,cpu"},
                            # ERA5 nudging targets are pulled from GCS and
                            # regridded to the model grid at startup — GBs of
                            # download and minutes of work for a multi-month
                            # window. Cache on the runs PVC, not the pod's
                            # ephemeral disk, so an eviction retry and a
                            # sibling run in the same sweep reuse it instead
                            # of repeating the fetch on every attempt.
                            {"name": "JCM_ERA5_CACHE",
                             "value": "/runs/_era5-cache"},
                            *env,
                        ],
                        "resources": {
                            "limits": {site["gpu_resource"]: gpus,
                                       "cpu": str(cpu), "memory": memory},
                            "requests": {site["gpu_resource"]: gpus,
                                         "cpu": str(cpu), "memory": memory},
                        },
                        "volumeMounts": [
                            {"name": "runs", "mountPath": "/runs"},
                            {"name": "work", "mountPath": "/work"},
                            {"name": "dshm", "mountPath": "/dev/shm"},
                        ],
                    }],
                    "volumes": [
                        {"name": "runs", "persistentVolumeClaim": {
                            "claimName": site["runs_pvc"]}},
                        {"name": "work", "emptyDir": {}},
                        {"name": "dshm", "emptyDir": {"medium": "Memory"}},
                    ],
                    # NOTE: priorityClassName is deliberately UNSET. The
                    # namespace bans every named class at 0 pods (including
                    # "default"), so setting one gets the pod refused by
                    # quota. An unnamed pod runs at priority 0 and is fine.
                }
            },
        },
    }


def build(a, resolved) -> dict:
    """Build the production Job for ``mkrun.py``'s command line (``a``)."""
    S = site_profile.get(a.site)
    rundir = f"/runs/{a.name}"
    checkpoint = f"{rundir}/{a.name}.ckpt"
    overrides = [
        f"physics={a.physics}",
        f"grid={a.grid}",
        "init=jw", "init.rh=0.0",
        *forcing_overrides(a),
        "run=longrun",
        *length_overrides(a),
        f"run.time_step={a.dt}",
        f"run.output_prefix={rundir}/{a.name}",
        f"++run.checkpoint_path={checkpoint}",
        # Stop on NaN. The opposite of the benchmark default: a year that has
        # gone unstable should not keep consuming a GPU.
        "++run.bail_on_unhealthy=true",
        *a.extra,
    ]
    return job_manifest(
        site=S, job_name=f"jcm-run-{a.name}".lower().replace("_", "-")[:60],
        label=a.name, rundir=rundir, overrides=overrides, days=target_days(a), resolved=resolved,
        setup=extra_pip_setup(S),
        python_env=("PYTHONPATH="
                    + ":".join(f"/work/{d}" for d in ("jax-rrtmgp", "mam4-jax"))
                    + f" MAM4_JAX_ENABLE_X64={'0' if a.f32 else '1'}"),
        retries=a.retries, gpus=a.gpus, cpu=a.cpu, memory=a.memory,
        gpu_product=a.gpu_product)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--name", required=True, help="run name; also the outdir")
    p.add_argument("--months", type=int, default=12,
                   help="run length in calendar months from --start-time")
    p.add_argument("--days", type=int, default=None,
                   help="fixed run length in days instead of --months")
    p.add_argument("--start-time", default="2000-01-01",
                   help="model start date (climatological forcing replays "
                        "every year; the date sets the calendar)")
    p.add_argument("--physics", default="echam-jam")
    p.add_argument("--grid", default="echam_t63_l47_hybrid")
    p.add_argument("--dt", type=int, default=12, help="minutes")
    p.add_argument("--chunk-days", type=int, default=5,
                   help="health-gated chunk length; its daily saves are what "
                        "sits in device memory")
    p.add_argument("--save-interval", type=int, default=1,
                   help="interval-mean length in days (must tile the months "
                        "for monthly means)")
    p.add_argument("--no-monthly-means", dest="monthly_means",
                   action="store_false", default=True,
                   help="do not stream calendar-month means")
    p.add_argument("--save-chunks", action="store_true", default=None,
                   help="also keep every chunk's interval means (_dayN.nc); "
                        "hundreds of GB for a JAM year of daily means. "
                        "Default: off with monthly means, on without them")
    p.add_argument("--no-save-chunks", dest="save_chunks",
                   action="store_false")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--cpu", type=int, default=8)
    p.add_argument("--memory", default="64Gi")
    p.add_argument("--f32", action="store_true", default=True)
    p.add_argument("--no-f32", dest="f32", action="store_false")
    p.add_argument("--site", default="nautilus",
                   help="site profile from sites.py")
    p.add_argument("--ozone", default=None,
                   help="ozone file; REQUIRED for any grid without a packaged "
                        "climatology (i.e. anything but T63L47)")
    p.add_argument("--gpu-product", default=None)
    p.add_argument("--retries", type=int, default=20,
                   help="Job backoffLimit; each retry resumes from the "
                        "checkpoint, so this is eviction tolerance")
    p.add_argument("--pin", action="append", default=[], metavar="REPO=REF",
                   help="pin a repo to an exact ref/SHA, e.g. jcm=abc1234")
    p.add_argument("--extra", nargs="*", default=[],
                   help="raw Hydra overrides appended last")
    a = p.parse_args()
    if a.save_chunks is None:
        # Without monthly means the chunk files are the run's only output.
        a.save_chunks = not a.monthly_means

    pins = dict(x.split("=", 1) for x in a.pin)
    resolved = resolve_refs(pins)
    for d, (_, sha) in resolved.items():
        print(f"# {d} pinned at {sha[:12]}", file=sys.stderr)
    print(json.dumps(build(a, resolved), indent=2))
    print(f"# output -> PVC {site_profile.get(a.site)['runs_pvc']}:"
          f"/runs/{a.name} (kept)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
