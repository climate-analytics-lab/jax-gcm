---
name: derecho-jcm-runs
description: Submit, monitor and benchmark jax-gcm (jcm) simulations on NCAR Derecho's PBS queues. Use when running any jcm model integration, timestep/resolution sweep, performance benchmark, or GPU job on Derecho — covers job-script generation, queue/account selection, environment setup, and reliable completion monitoring.
---

# Running jcm on Derecho

**Layering.** This is the Derecho/PBS site layer. The site-agnostic model
layer (config groups, Hydra traps, stability overrides) is `jcm-run`, and
throughput methodology is `jcm-benchmark`; both apply here too. The
shared-workstation counterpart is `devbox-jcm-runs` — worth a glance for the
contrast, since there GPUs are self-allocated rather than scheduled — and the
Kubernetes counterpart is `kubernetes-jcm-runs`.

**Derecho's A100s are 40 GB**, half the cluster and dev-box cards. Throughput
matches at equal work, but the memory ceiling does not: the reference table
in `jcm-benchmark` is measured on 80 GB, and only its smallest configs fit
here.

Generate a PBS script with `scripts/mkjob.py`, sanity-check the config, submit,
then monitor with the patterns below. Every default here was established by a
real campaign; the failure modes listed are ones that have actually happened.

## 1. Generate and submit

```bash
python scripts/mkjob.py --name my_run --days 30 > runs/my_run.pbs
qsub runs/my_run.pbs
```

Common flags (see `python scripts/mkjob.py --help` for all):

| flag | default | notes |
|---|---|---|
| `--gpus N` | 1 | >1 adds `+grid.spmd_mesh` and drops the memory fraction |
| `--queue` | `main` | `main` routes to `gpu`; `gpudev` for <1 h debugging |
| `--hours` | 6 | walltime |
| `--grid` | `echam_t63_l47_hybrid` | any `jcm/config/grid/*.yaml` stem |
| `--dt` | 15 | minutes |
| `--off-centering` | `0.2` | SL off-centering; transport is always semi-Lagrangian |
| `--physics` | `echam-jam` | `echam-rrtmgp-2m` for no aerosol |
| `--radiation` | (config default) | `grey` for a cheap-radiation A/B |
| `--aquaplanet` | off | skips terrain/forcing files |
| `--resume` | off | reuse the run dir's checkpoint |
| `--data` | `mirror` | HF bundles, prefetched at generation; `local` = legacy prepared files |
| `--era` | `pd` | `pd` (2005–2014) or `pi` (1850s) mirror climatologies |
| `--emissions` | (local mode) | legacy emissions file for `--data local` |
| `--extra "k=v ..."` | — | raw Hydra overrides appended last |

## 2. Always pre-flight before burning a queue slot

Two checks, both cheap, both catch failures that otherwise waste a job:

```bash
# (a) Hydra composition — catches +/++ prefix errors and unknown keys
JAX_PLATFORMS=cpu python -m jcm.main <exact overrides> --cfg job >/dev/null

# (b) coords constructibility — --cfg job does NOT build coords, so an
#     invalid spectral truncation only fails at runtime
JAX_PLATFORMS=cpu python -c "
from jcm.utils import get_coords
from jcm.physics.echam.echam_levels import get_echam_levels
get_coords(vertical_coords=get_echam_levels(<layers>), spectral_truncation=<T>)"
```

`mkjob.py --check` runs (a) for you and prints the command for (b).

Hydra override prefixes are a recurring trap: a key that already exists in the
composed config takes no `+`; one that does not, requires it. `run=longrun`
*replaces* the whole run group, so `run.checkpoint_path` needs `+` under it but
not under the default run config.

## 3. Environment (baked into generated scripts)

```bash
source ~/.venvs/jaxgcm/bin/activate
export PYTHONPATH=~/dinosaur-sl:$REPO     # SL dinosaur; jcm worktree wins over the venv's editable install
export JAX_PLATFORMS=cuda,cpu
export MAM4_JAX_ENABLE_X64=0              # f32 MAM4 core (forward-only); f64 default is much slower
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.93   # 0.85 when ngpus>1 — 0.93 starves CUDA command buffers
```

Overridable site paths: `JCM_REPO`, `JCM_VENV`, `JCM_DINOSAUR`, `JAM_INPUTS`,
`JCM_EMISSIONS`, `PBS_ACCOUNT`, `SCRATCH`.

Transport is always semi-Lagrangian (the Eulerian path was removed) and
requires a dinosaur carrying PR #135 (`JCM_DINOSAUR`, `~/dinosaur-sl` by
default). Without it the dycore raises a clear install-instruction error;
there is no fallback.

## 4. Input data

`reference/data_paths.md` lists every data source. The default is the
**HF data mirror** (`--data mirror --era pd|pi`): `mkjob.py` derives every
bundle path from `--grid` — terrain, forcing, emissions, DMS, dust, plus
level-resolved ozone and oxidants from `bundles/<grid>_l<levels>/` — and
prefetches them on the login node at generation time, baking the local
cache paths into the job. Compute nodes need no internet, and every
grid/level combination the mirror carries (t63/t106 × l47/l95) works the
same way: no packaged-grid special cases, no purge-eligible scratch
files, and a grid/level mismatch fails at generation, not in the queue.

`--data local` keeps the legacy prepared-file behaviour (`JAM_INPUTS` /
`JCM_EMISSIONS`, existence-checked before qsub). Its inputs are all
grid-specific — level-resolved (ozone, oxidants) or horizontally
validated (emissions, DMS, dust) — and **ozone is the dangerous one**:
`forcing.ozone_file: auto` resolves only a *packaged* climatology
(T63L47) and silently falls back to an ANALYTIC profile with ~7.6x the
tropospheric ozone column on any other grid. Prefer the mirror.

## 5. PBS facts specific to this machine

- GPU account is **UCSD0085** (UCSD0044 is casper-only and is rejected).
- `gpu_type=a100` must be **inside the select chunk**, not a separate `-l`.
- `-q main` is a routing queue that lands GPU jobs in `gpu`; `gpudev` exists for
  short interactive-style debugging.
- **`qsub -v VAR=x` does not reach the job environment here** — generated
  scripts hardcode their variables.
- Keep `#PBS -m abe` so job mail keeps working (it was silently lost once when a
  script was derived by `sed` from one that omitted it).
- Use `set -euo pipefail`; without `-e` a failed run still reaches a trailing
  `touch DONE` and looks successful.

## 6. Monitoring (`scripts/watch_job.sh`)

```bash
scripts/watch_job.sh <jobid> <logfile> "<completion marker>"
```

Use it as the command of a persistent `Monitor`. It encodes four lessons:

1. **Read the log once per check.** Grep the log into a variable, then both
   decide *and* report from those same bytes. Live NFS logs give stale re-reads,
   which produced repeated phantom "failures" whose detail printed empty.
2. **Debounce**: a failure signature must persist across two checks.
3. **3-strike `qstat`**: PBS requeues and transient `qstat` errors otherwise look
   like a vanished job.
4. **File existence is not success**: the driver writes chunk netCDFs *before*
   the NaN check. Verify the `NaN vars: 0/N` health line instead.

Filter Lmod's "unknown module" noise — it is harmless on these nodes.

## 7. Reading throughput correctly

Full methodology is in `jcm-benchmark`; the short version is that the
`N sim days/hr` line in the log is **cumulative and includes compile**, so it
must not be quoted. Use `Wall: X s this chunk`, discard chunk 1, and quote a
rate only once the last two chunks agree.

```bash
scripts/settled_rate.py <log> [--dt 15]   # per-chunk walls + convergence-checked rate
```

That script and `tools/benchmark.py` share `tools/chunk_timing.py`, so the
same run cannot yield two different answers. `settled_rate.py` reads a log
that already exists (what you want for a job back from the queue);
`benchmark.py` drives a run and samples GPU telemetry alongside (what you want
on an interactive box).

Log locations differ by job type: a plain run writes to the PBS `-o` file
(`<name>.log` in the submit directory); `--bench` variants write to
`$RUNDIR/<tag>/run.log`. A 10-day run yields only two chunks and the analyzer
will correctly refuse to quote a rate — allow >= 20 days (4 chunks) for a
number worth reporting.

Reference points at T63L47, JAM + SL, dt=15, one A100-40GB: **151 s per 5 days
= 119 days/hr**, of which radiation is ~78%. Grey radiation gives ~34 s / 533
days/hr. See `docs/source/design/dinosaur_sl_jam_configuration.md` in the repo.

## 8. Benchmarking or debugging a performance difference

`--bench` emits a variant-matrix job (reference / grey radiation / any extra
override sets) with convergence checks and GPU sampling under load. When
comparing machines, capture on both: `nvidia-smi` static specs, clocks/power
under load, `Clocks Event Reasons`, dependency provenance including git HEADs of
editable installs, and the **dtypes the model actually runs in** — a config flag
is not enough, since one f64 input promotes whole subgraphs. Power draw is
diagnostic: high power at max clocks with low throughput indicates FP64 units
engaging.

## 9. Memory guidance (A100-40GB)

T63L47 JAM fits comfortably at fraction 0.93 with 1 saved frame per chunk (2
frames OOM'd). T63L95 fits on one GPU. T106L95 does **not** — use 4 GPUs with
`+grid.spmd_mesh=[2,2,1]` and fraction 0.85. Valid spectral truncations are
21, 31, 42, 63, 85, 106, 119, 170, 213, 340, 425 — **T127 does not exist**.
