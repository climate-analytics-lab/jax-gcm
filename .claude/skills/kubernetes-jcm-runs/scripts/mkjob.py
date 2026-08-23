#!/usr/bin/env python
"""Generate a Nautilus (NRP) Kubernetes Job manifest for a jcm benchmark.

    python mkjob.py --preset ma-t63-l47 --months 1 | kubectl apply -f -
    python mkjob.py --sweep                        | kubectl apply -f -

Writes a manifest to stdout so it can be inspected before it is applied —
`| kubectl apply -f -` only once it looks right.

Why a generator rather than a checked-in manifest: the GPU selector, the
pinned dependency SHAs and the benchmark arguments all have to move
together, and a stale hand-edited YAML is how a sweep ends up measuring the
wrong thing. See SKILL.md for the platform facts these defaults encode.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import sites as site_profile  # noqa: E402  (local module)


# Repos cloned at pinned refs inside the pod. The published image carries a
# RELEASE of jcm, so branch code (and the benchmark harness itself) has to
# come from git. Pinning refs is what makes a rerun comparable — see the
# provenance discussion in jax-gcm#591.
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

# The a100 quota spans FOUR products: 80GB-PCIe, SXM4-80GB, PCIE-40GB and a
# 1g.10gb MIG slice. Selecting on memory keeps both 80GB variants (31 nodes)
# and excludes the 40GB and MIG ones, which would OOM the L95 configs.
# --gpu-product pins a single product when a comparison needs strict
# identity: SXM4 is a 400 W part against PCIe's 300 W, so single-GPU
# throughput differs even though both have 80 GB.

# Grid x levels resolution sweep. ne30 is the pySES CAM-SE backend rather
# than a spectral truncation, so this compares BOTH resolution and dycore
# under one physics package. T119 is not here: the data mirror carries t63
# and t106 only, so its presets are still dev-box-local (pod_runnable
# excludes them anyway, but listing them would just print six skip lines).
DEFAULT_SWEEP = [
    "ma-t63-l47", "ma-t63-l95",
    "ma-t106-l47", "ma-t106-l95",
    "ma-ne30-l47", "ma-ne30-l95",
]


def _preset_overrides(preset: str) -> list[str]:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]
                           / "tools"))
    from benchmark import PRESETS
    return PRESETS.get(preset, [])


def pip_commands(preset: str, site: dict) -> list[str]:
    """``pip install`` command lines this preset needs, in order.

    The image runs ``pip install -e .`` with NO extras (see the repo
    Dockerfile) off a release that predates the data mirror, so anything
    optional or newer than that release is simply absent.

    pySES is the one that needs care. ``dycore=pyses_*`` presets import it
    at model construction and would die after the image pull and the clone,
    but it declares ``torch>=2.12.0`` — 2-3 GB of wheels that also ship
    their own nvidia-* CUDA libraries, which can shadow the ones jax's CUDA
    build resolves against. The dev-box environment that produced the
    validated ne30 runs has **no torch at all**, so the dependency is unused
    on this code path. Install it ``--no-deps`` with its one
    genuinely-needed non-jax dependency (frozendict) named explicitly;
    numpy/jax/jaxlib/setuptools are already in the image.

    Keep the pin at or above the pyproject floor (0.1.3.1) — older wheels are
    much slower on GPU and have an upstream tracer-hyperviscosity bug.

    Derived from the preset rather than a hardcoded list, and added only
    where needed — putting a dycore on the spectral runs' path would make
    their timings depend on something they never import.
    """
    cmds = [" ".join(repr(x) for x in site["extra_pip"])]
    if any(o.startswith("dycore=pyses") for o in _preset_overrides(preset)):
        cmds.append("'frozendict>=2.4.7'")
        cmds.append("--no-deps 'pyses==0.1.3.1'")
    return [f"pip install --no-cache-dir {c} 2>&1 | tail -2" for c in cmds]


def pod_runnable(preset: str) -> tuple[bool, str]:
    """Can this preset's inputs exist inside a pod?

    Presets referencing prepared boundary data by absolute path (the
    T63L95/T106/T119 ozone under the dev box's scratch) cannot run here: the
    manifest mounts no such host path and sets no JCM_BC_DIR, so the
    harness's missing-input preflight rejects them. Submitting those wastes
    a queue slot and reads as a platform failure rather than absent data.

    Everything under the repo is fine — the pod clones it. This check is
    deliberately derived from the preset rather than a hardcoded exclusion
    list, so a preset becomes runnable automatically once its data is
    reachable (e.g. via the HF mirror).
    """
    for o in _preset_overrides(preset):
        key, _, val = o.partition("=")
        if not key.endswith((".file", "_file")) or val in ("auto", "null", ""):
            continue
        if val.startswith("/") and "/jcm/data/" not in val:
            return False, val
    return True, ""


def resolve_refs() -> dict:
    """Turn each repo's ref into the SHA it points at right now."""
    out = {}
    for d, (url, ref) in REPOS.items():
        r = subprocess.run(["git", "ls-remote", url, ref],
                           capture_output=True, text=True, timeout=120)
        sha = r.stdout.split()[0] if r.stdout.split() else None
        if not sha:
            raise SystemExit(f"cannot resolve {ref} in {url}")
        out[d] = (url, sha)
    return out


def job(preset: str, a) -> dict:
    S = site_profile.get(a.site)
    # --suffix distinguishes a rerun. Kubernetes Jobs are IMMUTABLE, so
    # re-submitting a preset whose previous Job still exists (TTL is 24 h) is
    # rejected outright; the alternative is deleting the earlier Job, which
    # throws away the record of a run someone may still be reading. The
    # suffix also lands in the report label, so the two results sit side by
    # side on the PVC instead of the second overwriting the first.
    tag = f"{preset}-{a.suffix}" if a.suffix else preset
    name = f"jcm-bench-{tag}".lower().replace("_", "-")[:60]
    # Resolve every ref to a SHA at GENERATION time and clone that exact
    # commit. Cloning a branch name means the code depends on when the pod
    # happened to start: a job submitted seconds before a push silently runs
    # the previous commit, which is how the first Nautilus run executed
    # without the fix it was submitted to test. It also makes reruns
    # unreproducible, which is the whole point of jax-gcm#591.
    # rm -rf first: an in-place container restart re-runs the script with
    # /work intact and a bare clone dies on the existing dir (see mkrun).
    clone = "\n".join(
        f'rm -rf /work/{d} && '
        f'git clone --filter=blob:none --no-checkout {url} /work/{d} '
        f'&& git -C /work/{d} fetch --depth 1 origin {sha} '
        f'&& git -C /work/{d} checkout --detach {sha} '
        f'&& echo "{d} @ {sha}"'
        for d, (url, sha) in a._resolved.items()
    )
    pythonpath = ":".join(
        f"/work/{d}" for d in ("dinosaur-sl", "jax-rrtmgp", "mam4-jax"))
    bench = (
        f"python /work/jcm/tools/benchmark.py --preset {preset} "
        f"--months {a.months} --gpu 0 --chunk-days {a.chunk_days} "
        f"--save-interval {a.save_interval or a.chunk_days} "
        f"--label {tag}-nautilus "
        f"--outdir /reports --scratch-root /scratch "
        f"--pythonpath {pythonpath}"
        + (" --f32" if a.f32 else "")
        + (" --allow-unhealthy" if a.allow_unhealthy else "")
    )
    script = f"""set -euo pipefail
echo "=== node: $NODE_NAME  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
mkdir -p /work /scratch /reports
{clone}
cd /work/jcm
# Re-point the image's editable jcm install at the clone and hard-gate the
# import paths — the editable finder otherwise beats cwd AND PYTHONPATH and
# the benchmark times RELEASE code under a pinned-SHA label (see mkrun).
pip install --no-cache-dir --no-deps -q -e /work/jcm
PYTHONPATH={pythonpath} python - <<'PYPATH'
import os, sys
import jcm, dinosaur
for mod, want in ((jcm, "/work/jcm"), (dinosaur, "/work/dinosaur-sl")):
    p = os.path.dirname(mod.__file__)
    if not p.startswith(want):
        sys.exit("FATAL: %s imports from %s, not %s."
                 % (mod.__name__, p, want))
print("import paths OK:", os.path.dirname(jcm.__file__))
PYPATH
# Packages the image lacks: it runs `pip install -e .` with no extras, off a
# release that predates the data mirror. diffrax (MAM4-JAX's condensation
# backend imports it at module load), huggingface_hub (resolves the hf://
# boundary data) and, for the pySES presets, the dycore itself. Any of these
# can in principle drag jax with it, which would silently swap the CUDA build
# for a CPU one — so the install is followed by a hard GPU check rather than
# trusting it. A CPU fallback would "work" and report timings 100x slow.
{chr(10).join(pip_commands(preset, S))}
python - <<'PYCHK'
import sys, jax
d = jax.devices()
print("jax devices after install:", d)
if not any(x.platform == "gpu" for x in d):
    sys.exit("FATAL: no GPU visible to jax after pip install — the CUDA "
             "build was replaced. Refusing to run: timings would be "
             "meaningless and the failure would look like a slow run.")
PYCHK
# The pod has exactly one GPU, so --gpu 0 is unambiguous and the
# free-GPU gate is a no-op: Kubernetes already gave us exclusive use.
# `set -e` would abort here on a nonzero exit — which is precisely the case
# the epilogue below exists for. Capture the status, always emit the
# diagnostics, and re-exit with it at the end. Observed: every failed job so
# far produced NO report/log output, so debugging meant reading the PVC by
# hand.
set +e
{bench}
BENCH_RC=$?
set -e
echo "=== done (benchmark rc=$BENCH_RC) ==="
exit $BENCH_RC
"""
    selector = dict(S["gpu_selector"])
    if a.gpu_product:
        selector = {"nvidia.com/gpu.product": a.gpu_product}

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": S["namespace"]},
        "spec": {
            # A benchmark that failed should be inspected, not silently
            # retried on a different node with different timings.
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "nodeSelector": selector,
                    "containers": [{
                        "name": "bench",
                        "image": S["image"],
                        "command": ["/bin/bash", "-c", script],
                        "env": [
                            {"name": "NODE_NAME", "valueFrom": {
                                "fieldRef": {"fieldPath": "spec.nodeName"}}},
                            # f32 for the MAM4 core is set by --f32 on the
                            # harness, which exports it; kept here as the
                            # default for anything else in the image.
                            {"name": "JAX_PLATFORMS", "value": "cuda,cpu"},
                            # --env passthrough. Some memory/perf knobs are
                            # environment variables rather than Hydra keys --
                            # JCM_RRTMGP_COL_CHUNKS above all, which splits the
                            # radiation column vmap and is what makes ne30 L95
                            # fit on one 80 GB card. Recorded in the manifest,
                            # so the run's configuration stays inspectable.
                            *[{"name": k, "value": v}
                              for k, v in (e.split("=", 1) for e in a.env)],
                        ],
                        "resources": {
                            "limits": {
                                S["gpu_resource"]: 1,
                                "cpu": str(a.cpu),
                                "memory": a.memory,
                            },
                            "requests": {
                                S["gpu_resource"]: 1,
                                "cpu": str(a.cpu),
                                "memory": a.memory,
                            },
                        },
                        "volumeMounts": [
                            {"name": "reports", "mountPath": "/reports"},
                            {"name": "work", "mountPath": "/work"},
                            # Model netCDF lands here and dies with the pod:
                            # a benchmark needs the timings, not the fields.
                            {"name": "scratch", "mountPath": "/scratch"},
                            {"name": "dshm", "mountPath": "/dev/shm"},
                        ],
                    }],
                    "volumes": [
                        {"name": "reports",
                         "persistentVolumeClaim": {"claimName": S["reports_pvc"]}},
                        {"name": "work", "emptyDir": {}},
                        {"name": "scratch", "emptyDir": {}},
                        # JAX/XLA wants more than the 64 MB default /dev/shm.
                        {"name": "dshm", "emptyDir": {"medium": "Memory"}},
                    ],
                }
            },
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--preset")
    p.add_argument("--sweep", action="store_true",
                   help=f"emit one Job per preset: {' '.join(DEFAULT_SWEEP)}")
    p.add_argument("--months", type=int, default=1)
    p.add_argument("--chunk-days", type=int, default=5)
    p.add_argument("--f32", action="store_true", default=True)
    p.add_argument("--no-f32", dest="f32", action="store_false")
    p.add_argument("--allow-unhealthy", action="store_true")
    p.add_argument("--site", default="nautilus",
                   help="site profile from sites.py")
    p.add_argument("--gpu-product", default=None,
                   help="pin an exact product, e.g. NVIDIA-A100-80GB-PCIe; "
                        "default selects any 80GB A100 by memory label")
    p.add_argument("--env", action="append", default=[], metavar="K=V",
                   help="extra container env var; repeatable. e.g. "
                        "--env JCM_RRTMGP_COL_CHUNKS=8 to split the radiation "
                        "column vmap (ncols must divide evenly)")
    p.add_argument("--save-interval", type=int, default=None,
                   help="days between output writes; defaults to --chunk-days "
                        "(one write per chunk). Lower it to shrink the "
                        "per-outer-step observation buffer: the inner lax.scan "
                        "STACKS every step between writes, so ne30 L95 needs a "
                        "50 GiB allocation at save-interval 5 and OOMs an 80 GB "
                        "card. Chunk timing is unaffected, so a run reduced "
                        "this way stays comparable.")
    p.add_argument("--suffix", default=None,
                   help="distinguish a rerun: appended to the Job name and "
                        "the report label, so it neither collides with an "
                        "existing Job nor overwrites its report")
    p.add_argument("--cpu", type=int, default=8)
    p.add_argument("--memory", default="64Gi")
    a = p.parse_args()

    if not a.sweep and not a.preset:
        p.error("need --preset or --sweep")
    presets = DEFAULT_SWEEP if a.sweep else [a.preset]
    if a.sweep:
        keep, drop = [], []
        for x in presets:
            ok, why = pod_runnable(x)
            (keep if ok else drop).append((x, why))
        for x, why in drop:
            print(f"# SKIP {x}: needs {why}, which no pod can see",
                  file=sys.stderr)
        if drop:
            print(f"# {len(drop)} preset(s) excluded — mount the data or "
                  "serve it from the HF mirror to include them",
                  file=sys.stderr)
        presets = [x for x, _ in keep]
        if not presets:
            raise SystemExit("no preset in the sweep can run in a pod")
    a._resolved = resolve_refs()
    for d, (_, sha) in a._resolved.items():
        print(f"# {d} pinned at {sha[:12]}", file=sys.stderr)

    docs = [job(x, a) for x in presets]
    print("\n---\n".join(json.dumps(d, indent=2) for d in docs))
    print(f"# {len(docs)} job(s); quota is 8 concurrent A100s",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
