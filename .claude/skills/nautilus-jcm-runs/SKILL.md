---
name: nautilus-jcm-runs
description: Run jcm on the Nautilus / NRP Kubernetes cluster — authenticate, generate GPU Job manifests, pick the right A100 variant, collect results. Use for benchmarks or batch runs that need dedicated GPUs, especially when the shared dev workstation is saturated.
---

# Running jcm on Nautilus (NRP)

Site layer for the National Research Platform Kubernetes cluster. Read
`jcm-run` for the model/Hydra layer and `jcm-benchmark` for throughput
methodology; this file covers only what is specific to Nautilus.

**Why use it.** Kubernetes gives a pod *exclusive* use of the GPUs it
requests. That is the guarantee benchmarking needs and the one the shared
dev workstation cannot offer — there, a neighbour can land on your card
mid-run and quietly corrupt the timing. The quota here is **8 concurrent
A100s**, so a six-config sweep runs in parallel in the time of its slowest
member rather than in sequence.

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
the device-code flow — this is interactive and cannot be automated:

```bash
kubectl oidc-login get-token --grant-type=device-code --skip-open-browser \
  --oidc-issuer-url=https://authentik.nrp-nautilus.io/application/o/k8s/ \
  --oidc-client-id=<client-id> --oidc-extra-scope=profile,offline_access
```

`--skip-open-browser` matters: without it kubelogin tries to launch a
browser on a headless box and hangs. Device-code lets you authenticate from
any machine.

## The A100 trap — read this before requesting a GPU

`nvidia.com/a100` is a *quota bucket*, not a card type. It spans four
different products:

| product | nodes | memory |
|---|---|---|
| NVIDIA-A100-SXM4-80GB | 22 | 80 GB |
| NVIDIA-A100-80GB-PCIe | 9 | 80 GB |
| NVIDIA-A100-PCIE-40GB | 8 | 40 GB |
| NVIDIA-A100-80GB-PCIe-MIG-1g.10gb | 1 | **9.7 GB** |

A bare `nvidia.com/a100: 1` request can land on any of them. The 40 GB card
OOMs every L95 config (they peak at ~60 GiB) and the MIG slice OOMs almost
everything — and a benchmark that lands on a different product each run
produces numbers that cannot be compared at all.

`mkjob.py` therefore selects on the **memory label**, which keeps both
80 GB variants and excludes the other two:

```yaml
nodeSelector:
  nvidia.com/gpu.memory: "81920"
```

**PCIe vs SXM4: measured, not assumed.** SXM4 is a 400 W part against PCIe's
300 W, so it is reasonable to expect a difference — but for this workload
there is none worth worrying about:

| | card | sim days/hr | peak GiB |
|---|---|---|---|
| dev box | A100 80GB PCIe | 174.6 | 8.64 |
| Nautilus | A100 SXM4-80GB | **175.2** | 8.56 |

`t63-echam-rrtmgp`, f32, same pinned rrtmgp — **0.3 % apart**. jcm at this
size is memory- and launch-bound rather than power-bound (~86 % utilisation
at well under the card's rating), so the extra 100 W buys nothing. Do not
spend scheduling flexibility pinning a product without a reason.

The job still records which product it ran on
(`/reports/<label>/gpu_product.txt`) — cheap, and it means a future
surprising number can be checked rather than re-litigated. `--gpu-product`
pins one if a comparison genuinely needs it.

## Submitting

```bash
S=.claude/skills/nautilus-jcm-runs/scripts
python $S/mkjob.py --preset ma-t63-l47 --f32              # inspect first
python $S/mkjob.py --preset ma-t63-l47 --f32 | kubectl apply -f -
python $S/mkjob.py --sweep --f32 | kubectl apply -f -     # all six, parallel
```

Always look at the manifest before applying it. `kubectl apply
--dry-run=server -f -` validates against the API without creating anything.

## What the generated Job does, and why

- **Image** `ghcr.io/climate-analytics-lab/jcm:latest` — public, GPU-ready
  (`nvidia/cuda:12.6.3-base` + `jax[cuda12]`). Tags include releases
  (`v2.0.1`) and `manual-<sha>` builds.
- **Clones pinned refs at startup.** The published image contains a
  *release*, so branch code — including the benchmark harness itself — has
  to come from git. Pinning the refs is what makes a rerun comparable.
- **`backoffLimit: 0`.** A failed benchmark must be inspected, not silently
  retried onto a different node with different timings.
- **Model output to an `emptyDir`** that dies with the pod; only reports and
  logs reach the PVC. A benchmark needs the timings, not the fields.
- **`/dev/shm` as a memory-backed `emptyDir`.** The 64 MB Kubernetes default
  is far too small for JAX/XLA.
- **`--gpu 0`** — the pod sees exactly one GPU, so the harness's free-GPU
  gate is a no-op here. Kubernetes has already guaranteed exclusivity, which
  is the whole reason to run here.

## Watching and collecting

```bash
kubectl get pods -l job-name=jcm-bench-<preset>
kubectl logs -f -l job-name=jcm-bench-<preset>
kubectl get jobs                        # COMPLETIONS column
```

Reports land on the `jcm-bench` PVC (50 Gi, RWX `rook-cephfs`, so parallel
pods share it). To read them, run a throwaway pod that mounts the PVC:

```bash
python $S/fetch_reports.py            # prints every report found
python $S/fetch_reports.py --copy ./  # copies them locally
```

**A pod that vanished is not a pod that succeeded.** Check the Job's
`COMPLETIONS`, and read the report — the harness refuses to quote a rate for
a truncated or NaN'd run, so a report with no throughput line means the run
failed even if Kubernetes says `Completed`.

## Gotchas

- **Jobs are immutable.** Re-applying a changed manifest under the same name
  fails; `kubectl delete job <name>` first.
- **`ttlSecondsAfterFinished: 86400`** cleans up after a day. Collect
  reports before then, or they persist on the PVC but the pod logs do not.
- **Image pull is slow on first use on a node** (multi-GB CUDA layers).
  Budget a few minutes before the container starts; that is not the model
  compiling.
- **Quota is 8 A100s across the namespace**, shared with anything else
  running there. `kubectl describe resourcequota a100-limit` shows current
  use; a ninth pod sits `Pending` rather than failing.
- No H100/H200/GH200 quota (all 0), so those selectors will never schedule.

## Reference point

`t63-echam-rrtmgp`, f32, 30 days, rrtmgp pinned at `848da33` (minor-gas scan
fix merged), A100-SXM4-80GB: **175.2 sim days/hr** (20.55 s/sim-day,
11.5 sim-years/day), 8.56 GiB peak, ~86 % utilisation. Chunks converged to
0.5 %. Image pull plus git clone plus compile took ~6 min before the first
chunk.

## Related skills

- `jcm-benchmark` — methodology; everything there applies here
- `devbox-jcm-runs` — the shared workstation, and why it is worse for this
- `derecho-jcm-runs` — the PBS alternative
