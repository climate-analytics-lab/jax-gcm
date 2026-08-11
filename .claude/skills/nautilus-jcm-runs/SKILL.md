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

Useful flags:

| flag | why |
|---|---|
| `--suffix TAG` | Jobs are **immutable**, so a rerun collides with the previous Job (TTL 24 h). The suffix goes into the Job name *and* the report label, so the two results sit side by side instead of the second overwriting the first. |
| `--save-interval N` | Days between writes. Defaults to `--chunk-days`. |
| `--env K=V` | Container env var, repeatable. Some knobs are env-only — `JCM_RRTMGP_COL_CHUNKS` splits the radiation column vmap. |
| `--gpu-product P` | Pin an exact A100 variant; default takes any 80 GB one. |

Boundary data comes from the Hugging Face mirror as `hf://bundles/...`
paths (#590). The harness **prefetches** them before claiming the GPU:
jcm resolves them lazily during model construction, which is after the
telemetry sampler starts, so a 2 GB bundle would otherwise download inside
the timed region.

The image is built from a *release* with `pip install -e .` and no extras,
so anything newer or optional is installed at pod start —
`huggingface_hub` (a jcm requirement only since the mirror), `diffrax`
(MAM4-JAX), and for pySES presets the dycore itself. pySES goes in
`--no-deps`: it declares `torch>=2.12.0`, 2–3 GB of wheels shipping their
own nvidia CUDA libraries that can shadow the ones jax resolves against,
and the environment behind the validated ne30 runs has no torch at all.
A hard GPU check follows every install, because a pip-induced CPU
fallback would still "work" and report timings 100× slow.

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
| model output | pod-local `emptyDir`, **discarded** | `jcm-runs` PVC, **kept** |
| health gate | `--allow-unhealthy` optional | **always on** — stop on NaN |
| `backoffLimit` | **0** — inspect a failure | **20** — survive eviction |
| on retry | would re-time on another node | **resumes from checkpoint** |
| TTL | 24 h | none |

**Eviction tolerance is the whole design.** A pod here is not guaranteed a
node for days, so a year-long run must assume it will be interrupted. jcm
resumes automatically when its checkpoint exists, so a restarted pod picks up
at the last completed chunk instead of restarting the year — that is what
makes multi-day runs viable on this platform at all. `--retries` is
effectively the eviction budget.

Pin the code for anything long: `--pin jcm=<sha>`. A run resumed days later
must come back on the *same* code, and cloning a branch name would silently
restart it on whatever has since been merged, mid-year.

Output lands on the `jcm-runs` PVC (500 Gi RWX) under `/runs/<name>/`,
alongside a `PROVENANCE` file recording the SHA of every repo, appended on
each attempt so a resumed run shows its full history.

### Getting the data out

```bash
kubectl exec -it <reader-pod> -- ls -la /runs/<name>/     # inspect
kubectl cp <reader-pod>:/runs/<name>/foo_day30.nc ./      # single file
```

For anything sizeable, run a pod that pushes to object storage or the HF
mirror rather than pulling multi-GB netCDF through `kubectl cp`, which is
slow and has no resume.

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
- **Do not set `priorityClassName`.** The namespace bans every named class at
  **0 pods** — including `default` — via the `high-priority-ban` and
  `low-priority-ban` quotas. A pod that names one is refused by quota; an
  unnamed pod runs at priority 0 and is fine. Both generators leave it unset
  deliberately.
- Pod cap is 200 (`reached-quota`), well above the 8-A100 GPU limit.

## Reference points

`t63-echam-rrtmgp`, f32, 30 days, rrtmgp pinned at `848da33` (minor-gas scan
fix merged), A100-SXM4-80GB: **175.2 sim days/hr** (20.55 s/sim-day,
11.5 sim-years/day), 8.56 GiB peak, ~86 % utilisation. Chunks converged to
0.5 %. Image pull plus git clone plus compile took ~6 min before the first
chunk.

### MA resolution sweep (2026-08-11)

`physics=echam-jam` + 2M + semi-Lagrangian, f32, 30 days, A100-80GB, all
boundary data grid-native from the mirror. Every entry converged to within
0.15 % on its last two chunks, completed 30/30 days, zero NaN.

| | L47 | L95 | peak GiB (L47 / L95) |
|---|---|---|---|
| **T63** | **127.1** | **65.8** | 16.6 / 32.6 |
| **T106** | **57.4** | **27.6** | 32.7 / 60.2 |

Cross-checks against the dev box within 1–4 % (T106L47 58.0→57.4,
T106L95 27.9→27.6), so these are the platform's numbers, not a node's.

Two things the shape of that table tells you:

- **Levels cost linearly** (1.93–2.08× for 2.02× the levels); **resolution
  is sublinear** (T106 has 2.78× T63's columns for 2.21–2.38× the cost) as
  utilisation climbs 88 % → 96 %. T63L47 leaves the card partly idle; the
  big configs are the efficient ones.
- **T106L95 at 60.2 GiB is the largest config that fits one card.**

### pySES ne30 is memory-bound, not compute-bound

ne30 L95 does **not** fit an 80 GB A100. XLA is explicit that this is a
whole-program floor rather than one fixable op:

```
hlo_rematerialization: Can't reduce memory use below 51.28GiB;
  only reduced to 61.92GiB, down from 65.83GiB originally
```

Per cell, the spectral path gets *more* efficient with size (19 → 12
GiB/Mcell from T63L47 to T106L95) while pySES sits at 30–50. ne30L95 has
**42 % of T106L95's cells and needs more memory than it**. So do not read
this as "ne30L95 is too big" — the same 30-day workload is 2.5–4× more
memory-hungry through this backend, and ne30L47 has been living at the edge
too (51 GiB, and a dev-box run that truncated at 26/30 days).

Levers already ruled out by measurement, so they are not worth re-trying:
`--save-interval` 5→1 *raised* the request to 54.39 GiB (more writes,
~1.1 GiB each); `--chunk-days` 5→1 left it at 50.45 GiB;
`JCM_RRTMGP_COL_CHUNKS=8` left it at 50.08 GiB. The remaining option is
multi-GPU sharding (pySES gained it in #575) — which stops being a
single-card number comparable with the table above.

## Related skills

- `jcm-benchmark` — methodology; everything there applies here
- `devbox-jcm-runs` — the shared workstation, and why it is worse for this
- `derecho-jcm-runs` — the PBS alternative
