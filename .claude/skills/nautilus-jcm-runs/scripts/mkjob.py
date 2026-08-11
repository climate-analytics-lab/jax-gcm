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

DEFAULT_SWEEP = [
    "ma-t63-l47", "ma-t63-l95", "ma-t106-l47",
    "ma-t106-l95", "ma-t119-l47", "ma-t119-l95",
]


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
    name = f"jcm-bench-{preset}".lower().replace("_", "-")[:60]
    # Resolve every ref to a SHA at GENERATION time and clone that exact
    # commit. Cloning a branch name means the code depends on when the pod
    # happened to start: a job submitted seconds before a push silently runs
    # the previous commit, which is how the first Nautilus run executed
    # without the fix it was submitted to test. It also makes reruns
    # unreproducible, which is the whole point of jax-gcm#591.
    clone = "\n".join(
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
        f"--save-interval {a.chunk_days} --label {preset}-nautilus "
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
# The pod has exactly one GPU, so --gpu 0 is unambiguous and the
# free-GPU gate is a no-op: Kubernetes already gave us exclusive use.
{bench}
# Stamp which card actually ran it — the a100 quota spans two 80GB
# products with different power limits, so the report is not
# interpretable without this.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \\
    > /reports/{preset}-nautilus/gpu_product.txt || true
# Echo the report to stdout so `kubectl logs` retrieves it. Reports are a
# few KB, so there is no reason to make anyone mount the PVC to read one —
# and per NRP's data-movement guidance kubectl cp is for small files only,
# with S3 for anything bulky. This removes the exec-into-a-pod dance
# entirely for the common case.
echo "===== REPORT BEGIN ====="
cat /reports/{preset}-nautilus/report.md 2>/dev/null || echo "(no report)"
echo "===== REPORT END ====="
# On failure, surface the model's own log too. Without this the pod log
# shows only the harness's "truncated" verdict and the actual traceback
# stays on the PVC — which cost several debug cycles here.
if ! grep -q "sim days/hr" /reports/{preset}-nautilus/report.md 2>/dev/null; then
  echo "===== RUN LOG TAIL (run did not produce a rate) ====="
  tail -40 /reports/{preset}-nautilus/run.log 2>/dev/null || true
fi
echo "=== done ==="
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
                   help="site profile from site.py")
    p.add_argument("--gpu-product", default=None,
                   help="pin an exact product, e.g. NVIDIA-A100-80GB-PCIe; "
                        "default selects any 80GB A100 by memory label")
    p.add_argument("--cpu", type=int, default=8)
    p.add_argument("--memory", default="64Gi")
    a = p.parse_args()

    if not a.sweep and not a.preset:
        p.error("need --preset or --sweep")
    presets = DEFAULT_SWEEP if a.sweep else [a.preset]
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
