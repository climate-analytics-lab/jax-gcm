---
name: devbox-jcm-runs
description: Run jcm on the shared UCSD dev workstation (8x A100-80GB, no scheduler) — find a genuinely free GPU, avoid stomping on colleagues' jobs, environment and scratch paths, and the etiquette/traps specific to an unscheduled multi-tenant box. Use for any local jcm run, benchmark or long integration on this machine.
---

# Running jcm on the shared dev workstation

Site layer for `sn4622116170`: 8x NVIDIA A100 80GB PCIe, **no batch scheduler**.
Read `jcm-run` for the model/Hydra layer and `jcm-benchmark` for throughput
methodology — this file only covers what is specific to this machine. (The
Derecho equivalent is `derecho-jcm-runs`; the contrast matters, see below.)

## The defining difference from Derecho: nothing allocates GPUs for you

On Derecho, PBS hands you exclusive GPUs and queueing is the scheduler's
problem. Here **you self-allocate**, several colleagues run directly on the
same box at the same time, and nothing stops two people picking the same card.
Everything below follows from that.

**Always run on a free card.** For benchmarks this is absolute — a contended
card does not fail loudly, it returns a plausible number that is simply wrong.

## Finding a free GPU

```bash
python tools/gpu_util.py            # every GPU, its memory, and its tenants
python tools/gpu_util.py --free     # free indices; exit 1 if none
python tools/gpu_util.py --wait 3600  # block until one frees, print its index
```

**Do not judge freeness by utilisation.** A real reading from this box:

```
GPU 0  busy   36113/81920 MiB    0%  python(414 MiB) x4, python(34422 MiB)
GPU 3  busy   21854/81920 MiB    0%  python(1318 MiB) ... python(14994 MiB)
GPU 4  FREE       5/81920 MiB    0%
```

GPUs 0 and 3 both read **0 % utilisation** while carrying five or six of
someone else's processes and tens of GB. Utilisation is instantaneous; a
parked job shows 0 % until it wakes.

Nor by memory alone. A colleague's Jupyter kernel sat on GPU 1 for weeks
holding ~1.2 GiB, so the card read `1182 MiB, 0 %` — which looks like idle
driver overhead. It was judged free and a 3-hour benchmark was started on top
of it. **Require no compute apps *and* < ~2 GiB resident**; `gpu_util.py`
enforces exactly that, and `tools/benchmark.py` refuses to start otherwise.

Note `nvidia-smi --query-compute-apps` reports GPU **uuids**, not indices, so
the two queries must be joined on uuid — the reason this is a module and not a
shell one-liner.

## Etiquette on a shared box

- **Never kill a process you did not start.** Other users' jobs are visible
  (`/home/lepeng/...`, `/data/j2wilke/...`, `/data/jamadan/...`).
- **Never `pkill -f <pattern>`.** The pattern matches your own shell and other
  people's jobs; it has killed this session's own shell twice. Kill explicit
  PIDs you have confirmed are yours — e.g. filter on your run's output prefix:
  `ps -eo pid,args | grep "[m]yprefix" | awk '{print $1}'`.
- Long runs: `nohup` + `CUDA_VISIBLE_DEVICES=<idx>` so a dropped session does
  not take the run with it.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` — otherwise one process grabs the whole
  HBM and locks out everyone including your own second run.
- Do not raise `XLA_PYTHON_CLIENT_MEM_FRACTION` here the way the Derecho skill
  does; that machine gives you the node, this one does not.

## Environment

```bash
PY=/home/dwatsonparris/micromamba/envs/jcm/bin/python   # NOT on PATH
# or: eval "$(micromamba shell hook --shell bash)" && micromamba activate jcm
```

`jcm`, `jax-rrtmgp`, `dinosaur` and `mam4-jax` are **editable installs, so the
working tree is the running code**. Check what you are actually running:

```bash
git -C /data/dwatsonparris/jax-rrtmgp rev-parse --abbrev-ref HEAD
```

To A/B a library version, **never `git checkout` in the shared clone** — that
silently changes the code under a colleague's running job. Use a worktree and
`PYTHONPATH`:

```bash
git -C /data/dwatsonparris/jax-rrtmgp worktree add \
    /data/dwatsonparris/jax-rrtmgp-worktrees/<name> origin/<branch>
PYTHONPATH=/data/dwatsonparris/jax-rrtmgp-worktrees/<name> $PY -m jcm.main ...
# verify it took:
PYTHONPATH=<worktree> $PY -c "import rrtmgp,os; print(os.path.dirname(rrtmgp.__file__))"
```

## Storage

- **Write run outputs, checkpoints and logs to `/scr/dwatsonparris/`** — fast
  ephemeral NVMe. `/data` is near-full and is not the place for netCDF output.
- Copy finals back to `/data` when a campaign is done; `/scr` is scratch.
- Repo lives at `/data/dwatsonparris/jax-gcm`, with worktrees under
  `/data/dwatsonparris/jcm-worktrees/`.

## Boundary data

Terrain, forcing and ozone are **packaged in the repo** (`jcm/data/bc/t63/`)
and `forcing.ozone_file: auto` resolves the right one — pass no ozone
override. See `jcm-run` for the full explanation and the ANALYTIC-fallback
warning to watch for. There is no scratch-purge concern here, unlike Derecho's
prepared JAM aux inputs.

## Reference points (this machine)

T63L47, ECHAM + RRTMGP, real orography/SSTs, dt=12 min, one A100-80GB,
measured with `tools/benchmark.py` over 30 days:

| build | s/sim day | sim days/hr | sim years/day |
|---|---|---|---|
| jax-rrtmgp with the unbounded minor-gas scan | 72.2 | 49.9 | 3.3 |
| jax-rrtmgp `perf/minor-gas-scan-bound` | 20.5 | 174.6 | 11.5 |

Peak memory 8.6 GiB; ~85 % utilisation at ~223 W of the card's 300 W rating —
i.e. not compute-bound. Three independent A/B pairs agreed to within 2.5 %
(jax-rrtmgp#22).
