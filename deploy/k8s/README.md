# Running JCM on Kubernetes (NRP Nautilus example)

This directory holds example manifests for running JCM as a non-interactive
batch job on the [NRP Nautilus](https://nrp.ai/documentation/) Kubernetes
cluster. The same manifests work on any cluster with NVIDIA GPU operators
installed; only the storage class and registry path need adjustment.

The container `ENTRYPOINT` is `python -m jcm.main`, so anything you pass
as args is treated as Hydra overrides — see [`jcm/config/`](../../jcm/config/)
for the available config groups.

## Files

| File | Purpose |
| --- | --- |
| [`job.yaml`](./job.yaml) | One-shot batch Job for a JCM simulation run (GPU). |
| [`interactive-pod.yaml`](./interactive-pod.yaml) | Long-lived pod for `kubectl exec` debugging (no GPU by default). |
| [`pvc.yaml`](./pvc.yaml) | Optional 50 GiB PVC for persisting outputs across Job runs. |

Each manifest uses `${VAR}` placeholders that `envsubst` expands at apply
time, so secrets and per-run parameters never get committed.

## Prerequisites

1. **`kubectl` + `kubelogin`.** NRP uses OIDC, which requires the
   `kubelogin` plugin (separate from `kubectl`):
   ```bash
   brew install int128/kubelogin/kubelogin   # macOS
   # or: go install github.com/int128/kubelogin/cmd/kubelogin@latest
   ```
   Verify with `kubectl auth whoami` against the `nautilus` context.

2. **Namespace.** NRP issues a per-group namespace. Pin it for the current
   context so you don't have to pass `-n` every time:
   ```bash
   kubectl config set-context --current --namespace=<your-namespace>
   ```
   For the Climate Analytics Lab, the namespace is `climate-analytics`.

3. **A pushed image.** NRP nodes need to pull JCM from a registry they can
   reach. Released versions are published to GHCR automatically by
   [`build_docker_image.yaml`](../../.github/workflows/build_docker_image.yaml)
   on every GitHub Release, tagged `:latest`, `:vX.Y.Z`, and `:<short-sha>`:

   ```
   ghcr.io/climate-analytics-lab/jcm:latest
   ```

   To cut a release: tag a commit on `main` and create a Release in the
   GitHub UI (or `gh release create vX.Y.Z`). The workflow guards
   against publishing from off-main branches.

   For ad-hoc dev images use `workflow_dispatch` (Actions tab → Build
   Docker image → Run workflow). Those publish as `:manual-<short-sha>`
   so they can't shadow a real release.

   **Local fallback** when you need a build before a release:
   ```bash
   # Apple Silicon needs --platform; native amd64 hosts can omit it.
   docker buildx build --platform=linux/amd64 \
     -t ghcr.io/climate-analytics-lab/jcm:dev-$(git rev-parse --short HEAD) \
     --push .
   ```
   Expect ~40 min under QEMU emulation on M-series Macs (vs ~5 min via
   GHA cache hits).

   Alternative registry: **NRP's GitLab**,
   `gitlab-registry.nrp-nautilus.io/<group>/jcm` — lives inside the
   cluster so pulls are fast, but needs you to set up that GitLab repo
   first.

   `envsubst` is in `gettext` on macOS: `brew install gettext`.

## Submitting a Job

```bash
export JCM_IMAGE=ghcr.io/climate-analytics-lab/jcm:latest
export JCM_JOB_NAME=jcm-icon-t31-$(date +%s)
export JCM_HYDRA_ARGS='physics=icon run.total_time=24'

envsubst < deploy/k8s/job.yaml | kubectl apply -f -
```

**For reproducible runs, prefer an immutable tag** (`:vX.Y.Z` or
`:<short-sha>`) over `:latest`. With `:latest`, Kubernetes pulls on
every Job (correct, since `:latest` is mutable) but you can't tell
afterwards which build of the code actually ran. With an immutable
tag, K8s reuses the cached image (saving the ~4 GiB pull) and the
provenance is unambiguous. `:latest` is best reserved for quick
iteration during development.

Watch it run:

```bash
kubectl get jobs,pods -l app=jcm
kubectl logs -f -l job-name=$JCM_JOB_NAME --tail=200
```

Copy outputs out when it's done (if you didn't mount a PVC):

```bash
POD=$(kubectl get pod -l job-name=$JCM_JOB_NAME -o jsonpath='{.items[0].metadata.name}')
kubectl cp "$POD:/app/outputs" ./outputs-from-$JCM_JOB_NAME
```

Delete a finished Job manually if you don't want to wait for the TTL:

```bash
kubectl delete job $JCM_JOB_NAME
```

## Interactive debugging

```bash
export JCM_IMAGE=ghcr.io/climate-analytics-lab/jcm:latest
envsubst < deploy/k8s/interactive-pod.yaml | kubectl apply -f -
kubectl exec -it jcm-shell -- bash
# ...poke around, run `python -m jcm.main physics=speedy run.total_time=1`, etc.
kubectl delete pod jcm-shell    # please don't forget — see NRP etiquette
```

## Picking a GPU

The default `nvidia.com/gpu: 1` request lets the scheduler pick any
available GPU. For specific models, edit `job.yaml`:

- **A100 / H100 / H200** — switch the resource key to
  `nvidia.com/a100`, `nvidia.com/h100`, `nvidia.com/h200`. These nodes
  are reservation-gated; check the NRP cluster resources page first.
- **A40 / RTX A6000 / RTX 6000 Blackwell** — use `nvidia.com/a40`,
  `nvidia.com/rtxa6000`, `nvidia.com/rtx6000bw`.
- **MIG slice** — `nvidia.com/mig-small` for a 1g.10gb A100 slice
  (cheaper, fine for T31 smoke tests).
- **A specific consumer GPU model** — use `nodeAffinity` on
  `nvidia.com/gpu.product` (commented example in `job.yaml`).

For **Grace Hopper / ARM64** add the `nautilus.io/arm64` toleration and
build an `arm64` image (the Dockerfile is portable; pass
`--platform=linux/arm64` to `docker buildx build`).

## Persisting outputs

By default Hydra writes runs to `/app/outputs/YYYY-MM-DD/HH-MM-SS/`
inside the container. Without a PVC, those vanish when the Job's pod is
deleted. Apply `pvc.yaml` once per namespace, then uncomment the
`outputs` volume blocks in `job.yaml` to mount the PVC at `/app/outputs`.

If you need to share outputs across Jobs running concurrently, change
`accessModes` in `pvc.yaml` from `ReadWriteOnce` to `ReadWriteMany`
(NRP's `rook-cephfs` storage class supports both).

## Resource sizing

The defaults in `job.yaml` (1 GPU, 4–8 CPU, 16–32 GiB memory, 4 GiB
`/dev/shm`) are sized for T31–T63 single-device runs. For larger
truncations or longer integrations:

- **Multi-GPU SPMD** — bump `nvidia.com/gpu` (up to 2 in a Pod, 8 in a
  Job per NRP policy) and match the JCM sharding config in
  `jcm/config/`.
- **Big shared-memory ops** — raise `dshm.sizeLimit`. Without an
  explicit limit NRP caps `/dev/shm` at 64 MB, which JAX collectives
  outgrow quickly.

## Etiquette

NRP is a shared resource. Two things to keep in mind:

1. **Delete finished pods.** The TTL on the Job does this automatically
   after 24 h, but a successful run can be deleted immediately with
   `kubectl delete job <name>`.
2. **Don't sit on idle GPUs.** Use Jobs (which terminate) rather than
   long-running Pods for actual work; the `interactive-pod.yaml`
   template deliberately omits the GPU request for this reason.
