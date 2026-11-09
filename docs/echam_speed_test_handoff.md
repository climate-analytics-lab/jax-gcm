# Handoff: L4 versus RTX PRO 6000 ECHAM benchmarks

## Source and scope

Branch: `bench/echam-speed-test` in `climate-analytics-lab/jax-gcm`.
Entry point: `echam_speed_test.py`. Cloud launch/setup, validation, cleanup and
background supervision: `tools/gcp_spot/`.

The original model revision was `a1b790260445382c3282867c8e43b6bbf48cfbe6`.
Branch commits `28ac9963` and `a456e792` added the standalone timing benchmark and
resolution selection. This handoff does not change model physics or that timing
entry point. Cloud helpers come from Terraform commit
`e74fe6367752ad75877f3541ea398d73c6f00931`, with the default source path adapted
to this checkout and recorded package constraints wired into VM startup.

The original reviewed source archive SHA256 was:

```text
cfff63a802c40671facbfdfc6e397452d5b7eec9f6d1a5763ab757b708149cfe
```

A new archive includes new docs/tools and therefore has a different hash even
when its model physics is identical. Record the actual revision, dirty status
and uploaded archive hash. This is an idealized throughput experiment, not a
realistic historical-climate simulation or production climate configuration.

## Model and timing protocol

- One GPU; **T63/L47**, 15-minute timestep, float32 (`JAX_ENABLE_X64=0`).
- Default `echam_physics`: **ECHAM one-moment**, not two-moment, microphysics;
  Sundqvist cloud fraction; RRTMGP radiation enabled at its native cadence.
- Analytic dry Jablonowski–Williamson initialization (`rh=0`), flat aquaplanet,
  constant forcing via `ForcingData.zeros`. No external climate datasets are
  provisioned. Source, packages and radiation lookup data still need setup.
- A full-length warmup includes compilation. Each timed repeat restarts from the
  same initial atmosphere, rather than continuing the preceding integration.
- Timers enclose `model.run` and block on GPU results. Installation, provisioning,
  validation and uploads are excluded. Output cadence must match between GPUs.
- `compile_time_s` is **warmup simulation plus compilation**, not compilation
  alone. `std_s` is population standard deviation. A zero with one repeat is not
  a variance estimate; individual measurements are in `all_times_s`.

| Experiment | Days per repeat | Save interval | Timed repeats |
|---|---:|---:|---:|
| Matched local RTX versus cloud L4 | 5 | 1 day | 3 |
| Cloud full L4 versus full RTX | 30 | 30 days, endpoint only | 1 |

Do not mix the daily-checkpoint/recovery experiment with the uninterrupted
30-day results. Any interrupted timing trial must restart with a fresh warmup;
preemption/recovery time must never be combined into an uninterrupted measurement.

## Recorded results and environments

| Device / experiment | Warmup including compilation | Timed seconds |
|---|---:|---|
| Local RTX PRO 6000, 5 days | 114.968 | 24.421, 24.581, 24.570; mean 24.524 |
| Cloud L4, 5 days | 375.776 | 127.507, 127.823, 127.720; mean 127.683 |
| Cloud L4, 30 days | 1004.918 | 759.826 |
| Cloud RTX PRO 6000, 30 days | 213.296 | 138.500 |

The warmed full-cloud RTX was 5.486 times faster. Both full-month jobs finished
without preemption/recovery, passed a 148,174,848-element final-output finite
check, and had their VMs and boot disks deleted. The wrapper checks the final
timed output, not every integration step. It does not establish climate fidelity.

Cloud Python was 3.12.3; local matched Python was 3.12.12. Both used JAX 0.10.2,
dinosaur 1.5.1 and jax-rrtmgp 0.7.0. The full-cloud L4 and RTX `pip freeze` files
were identical after excluding the machine-local editable jcm path. Those pins
are in `tools/gcp_spot/constraints.txt`; install jcm from the checkout/archive.
Older local environments using jax-rrtmgp 0.4.0 are not comparable.

The local RTX was shared with other work reserving substantial VRAM. It disabled
JAX preallocation and used an 11% memory limit; an earlier 10%-limit allocation
failure was excluded. Full-cloud runs used default preallocation. These memory
settings and Python versions are part of the comparison's limits.

## Local reproduction

Use Linux, Python 3.12 and a compatible NVIDIA driver. Choose an available GPU.

```bash
git clone --branch bench/echam-speed-test https://github.com/climate-analytics-lab/jax-gcm.git
cd jax-gcm
python3.12 -m venv .venv-benchmark
source .venv-benchmark/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -c tools/gcp_spot/constraints.txt -e . 'jax[cuda12]==0.10.2'
python -m pip check

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

env -u LD_LIBRARY_PATH python -u echam_speed_test.py \
  --spectral_truncation 63 --total_time 5 --save_interval 1 --n_repeats 3

env -u LD_LIBRARY_PATH python -u echam_speed_test.py \
  --spectral_truncation 63 --total_time 30 --save_interval 30 --n_repeats 1
```

The historical 11% limit was specific to the shared 96-GB local GPU. Do not apply
it to the 24-GB L4. Record competing workloads and allocator settings. The bare
entry point prints timing JSON; the cloud wrapper adds validation and artifacts.

## Shared infrastructure and authentication

Support resources are maintained in the separate
[Terraform repository](https://github.com/climate-analytics-lab/climate-analytics-lab-terraform),
merged in [PR #1](https://github.com/climate-analytics-lab/climate-analytics-lab-terraform/pull/1).
`projects/fluid-script-453604-u5/spot_jobs.tf` owns the private results bucket,
runtime service account and bucket-scoped object read/create grants.

Keep deployed definitions merged into that repository's main branch: an apply
from a checkout missing those definitions can remove the resources. Do not manage
them from a second Terraform state. Individual benchmark VMs are CLI-managed,
ephemeral jobs outside Terraform state. Do not commit state, credentials, plans,
generated archives or raw results.

Obtain local configuration with `terraform output -json spot_job_config` from
the initialized Terraform project, or save this as `/tmp/cal-spot-config.json`:

```json
{
  "project": "fluid-script-453604-u5",
  "bucket": "fluid-script-453604-u5-spot-jobs",
  "service_account": "jax-gcm-spot-runner@fluid-script-453604-u5.iam.gserviceaccount.com",
  "image": "https://www.googleapis.com/compute/v1/projects/ml-images/global/images/common-cu129-ubuntu-2404-nvidia-580-v20261019",
  "zone": "us-central1-c"
}
```

`gcloud auth login` authenticates cloud launchers; reauthentication may recur.
Terraform ADC authentication is separate. The submitter needs bucket access,
Compute create/list/delete and runtime-service-account use permission. The image
is historically pinned and may be deprecated/eventually unavailable; changing it
creates a new environment comparison and must be recorded.

## Launch full-GPU cloud trials

From this checkout, after authentication and support provisioning:

```bash
# Full L4: original month run used us-central1-a.
python -u tools/gcp_spot/launch.py --config /tmp/cal-spot-config.json \
  --source . --machine-type g2-standard-8 --zone us-central1-a \
  --total-days 30 --save-days 30 --repeats 1 --truncation 63 \
  --output /tmp/echam-spot-results/l4

# Full RTX PRO 6000 Blackwell: not a fractional VM.
python -u tools/gcp_spot/launch.py --config /tmp/cal-spot-config.json \
  --source . --machine-type g4-standard-48 --zone us-central1-c \
  --total-days 30 --save-days 30 --repeats 1 --truncation 63 \
  --output /tmp/echam-spot-results/rtx
```

For five-day reproduction, substitute `--total-days 5 --save-days 1 --repeats 3`.
For within-VM month variance, use `--repeats 3`; the original full-cloud month
comparison had only one timed repeat on each GPU.

Each command stays attached until completion. The launcher uses Spot only,
selects Hyperdisk Balanced for G4, and deletes its VM and auto-delete boot disk.
A two-hour server-side DELETE limit bounds running VMs. There is no silent
on-demand fallback. Add `--dry-run` to inspect without cloud calls.

Source snapshots contain tracked working-tree files plus the named benchmark
and its documentation, with size and symlink checks and remote SHA256 verification.
Track any new required files before launching. Recorded package constraints are
sent as VM metadata, including when an older prepared archive is used.

To use the original reviewed archive, replace `--source .` with
`--source-archive /tmp/source.tar.gz --source-sha256 <SHA256 above>`.

For detached execution, first create a frozen source archive:

```bash
mkdir -p /tmp/echam-background
python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, 'tools/gcp_spot')
from launch import snapshot
sha, info = snapshot(Path('.').resolve(), Path('/tmp/echam-background/source.tar.gz'))
print(sha)
PY

# Replace SHA256 with the printed value; use --gpu rtx for the other trial.
nohup python -I -u tools/gcp_spot/background.py --gpu l4 --continuous \
  --config /tmp/cal-spot-config.json \
  --source-archive /tmp/echam-background/source.tar.gz --source-sha256 SHA256 \
  --directory /tmp/echam-background/l4 \
  </dev/null >/tmp/echam-background/l4.log 2>&1 &
```

The supervisor writes atomic `status.json`, retries bounded capacity stockouts
and collects successful artifacts. `--continuous` uses one warmup and one timed
month with endpoint output. Its separate `month` mode has daily checkpoints and
is not the reported throughput protocol. A fresh preemption retry, if explicitly
enabled, always starts a fresh trial. Avoid a local script named `queue.py`,
which can shadow the Python standard library; use isolated mode as shown.

## Results and cleanup

The launcher prints its VM name and unique GCS prefix. Minute logs are uploaded
while running. The completion marker follows the durable result bundle. A failed
finite check or a stopped/preempted VM without completion is not a valid trial.
Inspect `results.json`, `validation.json`, `environment.json`, `pip-freeze.txt`,
`source.json`, `nvidia-smi.txt` and `job.log` inside the downloaded bundle.

Original month run IDs:

- L4: `20261106-054847-6f78a3d8`, `us-central1-a`.
- RTX: `20261106-054906-105d3d58`, `us-central1-c`.

Cloud bundles are at
`gs://fluid-script-453604-u5-spot-jobs/runs/<run-id>/output/results.tar.gz`;
source archives are at `runs/<run-id>/source.tar.gz` in the same bucket.
**Objects expire after 30 days**; these are not permanent archive links.
Workspace copies are under
`runs/results/gcp-spot-continuous-month/20261106T054847Z/` and
`runs/results/jcm-speed-test/t63-matched-cloud-20261106T014854Z/`.

The original full-cloud jobs did not record peak VRAM; NVIDIA-SMI snapshots were
taken before computation. Separate local checks measured about 4.6 GiB added
usage, not a verified full-month cloud peak. No bottleneck profiling was done.

If a submitter disappears, explicitly check deletion. Stopped VMs lose their
running-duration timer and can retain disks:

```bash
gcloud compute instances delete INSTANCE --zone=ZONE \
  --project=fluid-script-453604-u5
```

For a 365-day simulated year, `GPU-hours/year = mean_seconds/3600 * 365/days`.
Multiply by the full-VM hourly price for compute-only USD/year:

| Full VM | Hours/year | Historical Spot USD/hour | Implied USD/year |
|---|---:|---:|---:|
| L4, g2-standard-8 | 2.56793 | 0.510832 | 1.31 |
| RTX, g4-standard-48 | 0.46808 | 1.6808 | 0.79 |

Those prices are previously quoted rates, not current quotes. Costs exclude
setup, warmup, failed attempts, disks, storage and networking. Fractional follow-ups
used different drivers, Ubuntu 22.04 and JAX 0.10.0; their code/presets are in
the Terraform repository's `scripts/spot/fractional/` and `scripts/spot/configs/`.
Do not substitute fractional shapes into this full-GPU setup.

## Verification without paid VMs

```bash
python -m unittest discover -s tools/gcp_spot -p 'test_*.py'
bash -n tools/gcp_spot/startup.sh
python tools/gcp_spot/launch.py --source . --dry-run \
  --machine-type g4-standard-48 --total-days 30 --save-days 30 \
  --repeats 1 --truncation 63
```

Tests mock cloud calls and check source isolation, hashes, cleanup on
success/failure/preemption, full-RTX disk selection, constraints metadata and the
continuous protocol. They do not assert performance or launch paid resources.
