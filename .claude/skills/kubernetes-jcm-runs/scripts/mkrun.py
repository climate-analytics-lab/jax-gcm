#!/usr/bin/env python
"""Generate a Nautilus Job for a PRODUCTION jcm run (output kept, resumable).

    python mkrun.py --name pi-control --days 365 | kubectl apply -f -
    python mkrun.py --name pi-control --days 365 --resume | kubectl apply -f -

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
import json
import subprocess
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import sites as site_profile  # noqa: E402  (local module)


REPOS = {
    "jcm": ("https://github.com/climate-analytics-lab/jax-gcm",
            "feat/derecho-runs-skill"),
    # shoyer's semi-lagrangian branch, NOT neuralgcm/dinosaur main: the
    # SemiLagrangianPrimitiveEquationsHybrid class the SL dycore needs lives
    # in PR #135 and is not upstream. Pointing at upstream main gets a clone
    # that imports fine and then fails at model construction with
    # AttributeError — which is how this was found.
    "dinosaur-sl": ("https://github.com/shoyer/dinosaur", "semi-lagrangian"),
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


# Only T63L47 ozone is packaged in the repo. `forcing.ozone_file: auto`
# resolves that and SILENTLY falls back to the analytic profile otherwise —
# ~7.6x the tropospheric column, so the run completes and its radiation is
# simply wrong. For a production year that is the worst possible failure
# mode, so anything else must supply --ozone explicitly.
PACKAGED_OZONE = {"echam_t63_l47_hybrid"}


def ozone_override(a) -> list:
    if a.ozone:
        return [f"forcing.ozone_file={a.ozone}"]
    if a.grid in PACKAGED_OZONE:
        return []          # `auto` resolves the packaged file correctly
    raise SystemExit(
        f"grid {a.grid} has no packaged ozone climatology, and leaving it at "
        "`auto` silently falls back to the ANALYTIC profile (~7.6x the "
        "tropospheric column) — the run would finish and be scientifically "
        "wrong.\nPass --ozone <file reachable from the pod>, or use "
        f"{sorted(PACKAGED_OZONE)[0]}.")


def build(a, resolved) -> dict:
    S = site_profile.get(a.site)
    name = f"jcm-run-{a.name}".lower().replace("_", "-")[:60]
    rundir = f"/runs/{a.name}"
    clone = "\n".join(
        f'git clone --filter=blob:none --no-checkout {url} /work/{d} '
        f'&& git -C /work/{d} fetch --depth 1 origin {sha} '
        f'&& git -C /work/{d} checkout --detach {sha}'
        for d, (url, sha) in resolved.items()
    )
    pythonpath = ":".join(
        f"/work/{d}" for d in ("dinosaur-sl", "jax-rrtmgp", "mam4-jax"))
    overrides = " ".join([
        f"physics={a.physics}",
        f"grid={a.grid}",
        "init=jw", "init.rh=0.0",
        "terrain=from_file",
        "terrain.file=/work/jcm/jcm/data/bc/t63/terrain.nc",
        "forcing=from_file",
        "forcing.file=/work/jcm/jcm/data/bc/t63/forcing.nc",
        *ozone_override(a),
        "run=longrun",
        f"run.total_time={a.days}",
        f"run.time_step={a.dt}",
        f"run.chunk_days={a.chunk_days}",
        f"run.save_interval={a.save_interval}",
        f"run.output_prefix={rundir}/{a.name}",
        f"++run.checkpoint_path={rundir}/{a.name}.ckpt",
        # Stop on NaN. The opposite of the benchmark default: a year that has
        # gone unstable should not keep consuming a GPU.
        "++run.bail_on_unhealthy=true",
        *a.extra,
    ])
    script = f"""set -euo pipefail
echo "=== node $NODE_NAME | $(nvidia-smi --query-gpu=name --format=csv,noheader) | attempt $(date -u +%FT%TZ) ==="
mkdir -p /work {rundir}
{clone}
cd /work/jcm
# MAM4-JAX declares diffrax and matplotlib; neither is in the jcm image, and
# the JAM condensation backend imports diffrax at module load. Installing it
# here can in principle drag jax with it, which would silently swap the CUDA
# build for a CPU one — so the install is followed by a hard GPU check rather
# than trusting it. A CPU fallback would "work" and report timings 100x slow.
pip install --no-cache-dir {' '.join(repr(x) for x in S['extra_pip'])} 2>&1 | tail -2
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
if [ -f "{rundir}/{a.name}.ckpt" ]; then
  echo "=== resuming from $(ls -la {rundir}/{a.name}.ckpt | awk '{{print $5}}') byte checkpoint ==="
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
PYTHONPATH={pythonpath} MAM4_JAX_ENABLE_X64={"0" if a.f32 else "1"} \\
  python -m jcm.main {overrides} 2>&1 | tee -a {rundir}/run.log
RC=${{PIPESTATUS[0]}}
set -e
tail -c +$((ATTEMPT_START + 1)) "{rundir}/run.log" > /tmp/attempt.log

# jcm.runners.run_chunked BREAKS OUT of its loop and returns NORMALLY when
# bail_on_unhealthy trips, so jcm.main exits 0 even though the year stopped
# at the first bad chunk. Without this check Kubernetes marks a 365-day Job
# Complete after 30 days of output — the worst kind of failure, because it
# looks like success. Verify the health verdict and the day count.
if grep -qiE "unhealthy|NaN vars: *[1-9]" /tmp/attempt.log; then
  echo "FATAL: health gate tripped — run stopped early, not complete"
  grep -iE "unhealthy|NaN vars: *[1-9]" /tmp/attempt.log | tail -3
  exit 1
fi
LAST=$(grep -oE "_day[0-9]+\\.nc" /tmp/attempt.log | grep -oE "[0-9]+" \\
       | sort -n | tail -1)
if [ -z "$LAST" ]; then
  # No output this attempt. Distinguish the one benign case — the run was
  # already finished and the pod merely restarted — from a no-op resume,
  # which must NOT look like success.
  RESUMED=$(grep -oE "Resumed from checkpoint .* at sim-day [0-9.]+" \\
            /tmp/attempt.log | grep -oE "[0-9.]+$" | tail -1)
  if [ -n "$RESUMED" ] && [ "${{RESUMED%%.*}}" -ge {a.days} ]; then
    echo "=== already complete: checkpoint at day $RESUMED of {a.days}, nothing to do ==="
    exit 0
  fi
  echo "FATAL: this attempt wrote no output and resumed at day ${{RESUMED:-0}}"
  echo "       of {a.days} — no progress made. Check for a stale or foreign"
  echo "       checkpoint at {rundir}/{a.name}.ckpt."
  exit 1
fi
if [ "$LAST" -lt {a.days} ]; then
  echo "FATAL: reached day $LAST of {a.days} — incomplete"
  exit 1
fi
echo "=== finished $(date -u +%FT%TZ), day $LAST of {a.days}, rc=$RC ==="
exit $RC
"""
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": name, "namespace": S["namespace"],
                     "labels": {"jcm-run": a.name}},
        "spec": {
            # Survive eviction: each retry re-runs the script, which resumes
            # from the checkpoint. Contrast the benchmark generator, where a
            # retry would silently re-time on a different node.
            "backoffLimit": a.retries,
            # No TTL — production output and its Job history are kept until
            # deliberately removed.
            "template": {
                "spec": {
                    "restartPolicy": "OnFailure",
                    "nodeSelector": (
                        {"nvidia.com/gpu.product": a.gpu_product}
                        if a.gpu_product
                        else dict(S["gpu_selector"])),
                    "containers": [{
                        "name": "run", "image": S["image"],
                        "command": ["/bin/bash", "-c", script],
                        "env": [
                            {"name": "NODE_NAME", "valueFrom": {"fieldRef": {
                                "fieldPath": "spec.nodeName"}}},
                            {"name": "JAX_PLATFORMS", "value": "cuda,cpu"},
                        ],
                        "resources": {
                            "limits": {S["gpu_resource"]: a.gpus,
                                       "cpu": str(a.cpu), "memory": a.memory},
                            "requests": {S["gpu_resource"]: a.gpus,
                                         "cpu": str(a.cpu), "memory": a.memory},
                        },
                        "volumeMounts": [
                            {"name": "runs", "mountPath": "/runs"},
                            {"name": "work", "mountPath": "/work"},
                            {"name": "dshm", "mountPath": "/dev/shm"},
                        ],
                    }],
                    "volumes": [
                        {"name": "runs", "persistentVolumeClaim": {
                            "claimName": S["runs_pvc"]}},
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--name", required=True, help="run name; also the outdir")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--physics", default="echam-jam")
    p.add_argument("--grid", default="echam_t63_l47_hybrid")
    p.add_argument("--dt", type=int, default=12, help="minutes")
    p.add_argument("--chunk-days", type=int, default=30)
    p.add_argument("--save-interval", type=int, default=5)
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
