---
name: kubernetes-jcm-runs
description: Run jcm on a Kubernetes GPU cluster — generate benchmark and production Job manifests, pick a comparable GPU, survive eviction, collect results. Currently configured for Nautilus/NRP; other clusters are a site profile in sites.py. Use for runs that need dedicated GPUs, especially when the shared dev workstation is saturated.
---

# Running jcm on Kubernetes

The Kubernetes site layer. Read `jcm-run` for the model/Hydra layer and
`jcm-benchmark` for throughput methodology and reference numbers; this file
covers only what running on a cluster adds.

**Why use a cluster.** Kubernetes gives a pod *exclusive* use of the GPUs it
requests. That is the guarantee benchmarking needs and the one a shared
workstation cannot offer — there, a neighbour can land on your card mid-run
and quietly corrupt the timing (`devbox-jcm-runs`). Quota permitting, a
multi-config sweep also runs in parallel, finishing in the time of its
slowest member rather than the sum.

**Portability.** The Job *shape* — clone pinned SHAs, install what the image
lacks, run jcm, write to a volume, refuse or survive failure — is generic.
Everything cluster-specific lives in `scripts/sites.py` as a profile:
namespace, GPU resource name and selector, storage class, PVC names, image,
and the extra pip packages. A second cluster is a new dict there plus
`--site <name>`, **not** a fork of the generators. Only Nautilus is defined;
`sites.py` documents what a GKE/EKS profile would have to supply (most
notably `nvidia.com/gpu` rather than a vendor quota bucket).

## Submitting

```bash
S=.claude/skills/kubernetes-jcm-runs/scripts
python $S/mkjob.py --preset ma-t63-l47 --f32              # inspect first
python $S/mkjob.py --preset ma-t63-l47 --f32 | kubectl apply -f -
python $S/mkjob.py --sweep --f32 | kubectl apply -f -     # parallel sweep
```

Always look at the manifest before applying it. `kubectl apply
--dry-run=server -f -` validates against the API without creating anything.

| flag | why |
|---|---|
| `--suffix TAG` | Jobs are **immutable**, so a rerun collides with the previous Job (TTL 24 h). The suffix goes into the Job name *and* the report label, so the two results sit side by side instead of the second overwriting the first. |
| `--save-interval N` | Days between writes. Defaults to `--chunk-days`. |
| `--env K=V` | Container env var, repeatable — for knobs that are env-only rather than Hydra keys. |
| `--gpu-product P` | Pin an exact GPU product; by default any card the site selector admits. |
| `--site NAME` | Site profile from `sites.py`. |

`--sweep` skips presets whose inputs no pod can reach, and says which and
why. That check is derived from the preset's own file overrides, so a preset
becomes runnable automatically once its data is reachable rather than
needing an exclusion list edited.

## What the generated Job does, and why

- **Clones pinned refs at startup.** The published image contains a
  *release*, so branch code — including the benchmark harness itself — has
  to come from git. Every ref is resolved to a SHA at *generation* time:
  cloning a branch name makes the code depend on when the pod happened to
  start, and a job submitted seconds before a push silently runs the
  previous commit.
- **Installs what the image lacks.** It is built from a release with `pip
  install -e .` and no extras, so anything newer or optional is added at pod
  start (see `extra_pip` in the site profile, plus per-preset additions).
  A **hard GPU check follows every install**, because a pip-induced CPU
  fallback would still "work" and report timings ~100× slow.
- **`backoffLimit: 0`** for benchmarks. A failed benchmark must be
  inspected, not silently retried onto a different node with different
  timings.
- **Model output to an `emptyDir`** that dies with the pod; only reports and
  logs reach the PVC. A benchmark needs the timings, not the fields.
- **`/dev/shm` as a memory-backed `emptyDir`.** The 64 MB Kubernetes default
  is far too small for JAX/XLA.
- **`--gpu 0`** — the pod sees exactly one GPU, so the harness's free-GPU
  gate is a no-op. Kubernetes has already guaranteed exclusivity, which is
  the whole reason to run here.
- **Remote boundary data is prefetched** before the GPU is claimed. jcm
  resolves `hf://` paths lazily during model construction, which is after
  the telemetry sampler starts, so a multi-GB bundle would otherwise
  download inside the timed region.

## Production runs — `mkrun.py`

Benchmarks and production runs want opposite things, so they have separate
generators. Do not use `mkjob.py` for a run whose output you intend to keep.

```bash
python $S/mkrun.py --name pi-control --days 365 | kubectl apply -f -
python $S/mkrun.py --name aci-2yr --days 730 --physics echam-jam-aci \
    --pin jcm=abc1234 | kubectl apply -f -
```

| | `mkjob.py` (benchmark) | `mkrun.py` (production) |
|---|---|---|
| model output | pod-local `emptyDir`, **discarded** | runs PVC, **kept** |
| health gate | `--allow-unhealthy` optional | **always on** — stop on NaN |
| `backoffLimit` | **0** — inspect a failure | **20** — survive eviction |
| on retry | would re-time on another node | **resumes from checkpoint** |
| TTL | 24 h | none |

**Eviction tolerance is the whole design.** A pod is not guaranteed a node
for days, so a long run must assume interruption. jcm resumes automatically
when its checkpoint exists, so a restarted pod picks up at the last
completed chunk instead of restarting the year — that is what makes
multi-day runs viable here at all. `--retries` is effectively the eviction
budget.

Pin the code for anything long: `--pin jcm=<sha>`. A run resumed days later
must come back on the *same* code; a branch name would silently restart it
on whatever has since been merged, mid-year.

Output lands under `/runs/<name>/` alongside a `PROVENANCE` file recording
the SHA of every repo, appended on each attempt so a resumed run shows its
full history.

**A run that stops early can still exit 0.** `run_chunked` returns normally
when the health gate trips, so Kubernetes would mark a truncated year
`Complete`. `mkrun.py` therefore checks the health verdict *and* the day
count reached, and fails the Job if either falls short.

## Watching and collecting

```bash
kubectl get pods -l job-name=<job>
kubectl logs -f -l job-name=<job>
kubectl get jobs                        # COMPLETIONS column

python $S/fetch_reports.py            # summary table from job logs
python $S/fetch_reports.py --copy ./  # also copy them locally
python $S/fetch_reports.py --from-pvc # once a job's TTL has expired
```

`fetch_reports.py` reads job logs by default — no pod, no exec, no volume
mount. `--from-pvc` falls back to a throwaway pod that mounts the reports
volume, which is what you need after the 24 h TTL removes the Job.

**A pod that vanished is not a pod that succeeded.** Check the Job's
`COMPLETIONS`, and read the report — the harness refuses to quote a rate for
a truncated or NaN'd run, so a report with no throughput line means the run
failed even if Kubernetes says `Completed`.

For anything sizeable, push to object storage from a pod rather than pulling
multi-GB netCDF through `kubectl cp`, which is slow and has no resume.

---

# Site profile: Nautilus (NRP)

The National Research Platform. Everything below is specific to it.

## Access

```bash
export PATH=/data/dwatsonparris/micromamba/bin:$PATH   # kubectl lives here
kubectl config current-context     # -> nautilus
kubectl get pods                   # namespace: climate-analytics
```

Authentication is OIDC via an exec plugin, which needs
**`kubectl-oidc_login`** ([int128/kubelogin](https://github.com/int128/kubelogin))
on `PATH`. Note conda-forge's `kubelogin` package is *Azure's* tool and is
not the same thing.

The token caches under `~/.kube/cache/oidc-login/`. When it expires, re-run
the device-code flow — interactive, cannot be automated:

```bash
kubectl oidc-login get-token --grant-type=device-code --skip-open-browser \
  --oidc-issuer-url=https://authentik.nrp-nautilus.io/application/o/k8s/ \
  --oidc-client-id=<client-id> --oidc-extra-scope=profile,offline_access
```

`--skip-open-browser` matters: without it kubelogin tries to launch a
browser on a headless box and hangs.

## The A100 trap — read this before requesting a GPU

`nvidia.com/a100` is a *quota bucket*, not a card type. It spans four
products:

| product | nodes | memory |
|---|---|---|
| NVIDIA-A100-SXM4-80GB | 22 | 80 GB |
| NVIDIA-A100-80GB-PCIe | 9 | 80 GB |
| NVIDIA-A100-PCIE-40GB | 8 | 40 GB |
| NVIDIA-A100-80GB-PCIe-MIG-1g.10gb | 1 | **9.7 GB** |

A bare `nvidia.com/a100: 1` request can land on any of them. The 40 GB card
OOMs the larger configs and the MIG slice OOMs almost everything — and a
benchmark that lands on a different product each run produces numbers that
cannot be compared at all. `mkjob.py` therefore selects on the **memory
label**, keeping both 80 GB variants and excluding the other two:

```yaml
nodeSelector:
  nvidia.com/gpu.memory: "81920"
```

**PCIe vs SXM4 was measured, not assumed: 0.3 % apart** on an identical
config, despite SXM4 being a 400 W part against PCIe's 300 W. jcm is memory-
and launch-bound rather than power-bound at these sizes, so the extra
headroom buys nothing. Do not spend scheduling flexibility pinning a product
without a reason; `--gpu-product` is there if a comparison genuinely needs
it, and every job records what it ran on (`gpu_product.txt`) so a surprising
number can be checked rather than re-litigated.

## Resources

- **Image** `ghcr.io/climate-analytics-lab/jcm:latest` — public, GPU-ready
  (`nvidia/cuda:12.6.3-base` + `jax[cuda12]`). Tags include releases
  (`v2.0.1`) and `manual-<sha>` builds.
- **`jcm-bench` PVC** — 50 Gi RWX `rook-cephfs`, so parallel pods share it;
  holds reports and logs.
- **`jcm-runs` PVC** — 500 Gi RWX, production output.
- **Quota: 8 concurrent A100s** across the namespace, shared with anything
  else running there. `kubectl describe resourcequota a100-limit` shows
  current use; a ninth pod sits `Pending` rather than failing. Pod cap is
  200, well above the GPU limit.
- No H100/H200/GH200 quota (all 0), so those selectors will never schedule.

## Gotchas

- **Jobs are immutable.** Re-applying a changed manifest under the same name
  fails; `kubectl delete job <name>` first, or use `--suffix`.
- **`ttlSecondsAfterFinished: 86400`** removes benchmark Jobs after a day.
  Collect reports before then, or read the PVC with `--from-pvc`.
- **Image pull is slow on first use on a node** (multi-GB CUDA layers).
  Budget a few minutes before the container starts; that is not the model
  compiling.
- **Do not set `priorityClassName`.** The namespace bans every named class
  at **0 pods** — including `default` — via the `high-priority-ban` and
  `low-priority-ban` quotas. A pod that names one is refused by quota; an
  unnamed pod runs at priority 0 and is fine. Both generators leave it unset
  deliberately.
- **No node-read permission.** `kubectl get node` is forbidden, so a pod's
  GPU product must come from the pod itself, not the node label.

## Related skills

- `jcm-benchmark` — methodology and reference numbers; everything there applies here
- `jcm-run` — config groups and Hydra traps
- `devbox-jcm-runs` — the shared workstation, and why it is worse for timing
- `derecho-jcm-runs` — the PBS alternative (40 GB cards, so memory limits differ)
