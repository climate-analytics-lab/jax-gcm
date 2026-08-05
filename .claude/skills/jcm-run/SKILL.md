---
name: jcm-run
description: Launch a jcm model run through the built-in Hydra configs — config groups, the validated stable T63L47 overrides, Hydra override traps, and watching for startup failures. Site-agnostic; pair with devbox-jcm-runs (shared workstation) or derecho-jcm-runs (NCAR PBS) for machine specifics.
---

# Running jcm

**Layering.** This skill is the site-agnostic model layer. For where to run,
see the machine skill: `devbox-jcm-runs` (shared UCSD workstation, no
scheduler, you pick the GPU) or `derecho-jcm-runs` (NCAR Derecho, PBS
allocates GPUs). For throughput measurement see `jcm-benchmark`.

Every runnable configuration goes through `python -m jcm.main` with Hydra
groups and overrides. **Never write a bespoke driver script** — see the "No
bespoke run scripts" rule in `CLAUDE.md`. If a configuration is worth
repeating, it becomes a config file under `jcm/config/<group>/`.

## Config groups

| group | options |
|---|---|
| `physics` | `speedy`, `held_suarez`, `echam`, `echam-rrtmgp`, `echam-rrtmgp-2m`, `echam-rrtmgp-2m-cosp`, `echam-strong-conv`, `echam-jam`, `echam-jam-aerocom`, `echam-jam-aerocom-optics`, `echam-jam-aci` |
| `grid` | `speedy_t31_l8`, `held_suarez_t31_l8`, `echam_t42_l8_sigma`, `echam_t63_l47_hybrid`, `echam_t85_l47_hybrid`, `echam_t63_l95_hybrid`, `echam_t106_l95_hybrid`, `echam_t119_l95_hybrid` |
| `init` | `isothermal`, `balanced_isothermal`, `jw` |
| `terrain` | `aquaplanet`, `from_file` |
| `forcing` | `default`, `from_file` |
| `run` | `default`, `smoke`, `longrun`, `pyses_year` |
| `dycore` | `dinosaur`, `pyses_ne30l47` |
| `diffusion` | `default`, `strong` |

Discover current options rather than trusting this table if it looks stale:
`ls jcm/config/<group>/`.

## The validated T63L47 ECHAM launch

This is the known-stable production baseline. **Use it as the starting point
for any T63L47 run** — the pieces below are not optional decoration, they are
what keeps the run from going NaN (see "Stability" below).

```bash
PY=/home/dwatsonparris/micromamba/envs/jcm/bin/python
REPO=/data/dwatsonparris/jax-gcm
TS=$(date +%y%m%d_%H%M%S)

COMMON="physics=echam-rrtmgp \
        grid=echam_t63_l47_hybrid \
        init=jw init.rh=0.0 \
        terrain=from_file terrain.file=$REPO/jcm/data/bc/t63/terrain.nc \
        forcing=from_file forcing.file=$REPO/jcm/data/bc/t63/forcing.nc \
        run=longrun \
        run.time_step=12 run.save_interval=5 run.chunk_days=30"

PREFIX=myrun_$TS
nohup env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    $PY -m jcm.main $COMMON \
        run.total_time=365 \
        run.output_prefix=$PREFIX \
        +run.checkpoint_path=${PREFIX}.ckpt \
    > $REPO/run_logs/${PREFIX}.log 2>&1 &
echo "$PREFIX PID=$!"
```

Write outputs to `/scr/dwatsonparris/...` for anything large — `/data` is
near-full. Logs conventionally go in `run_logs/` at the repo root.

## Pre-flight: where to run

GPU selection is machine-specific and lives in the site skills:

- **`devbox-jcm-runs`** — shared workstation. You self-allocate, so you must
  verify a card is genuinely free (`python tools/gpu_util.py`) and avoid
  stomping on colleagues. Utilisation and memory each *individually* look
  idle for a parked job; both must be checked.
- **`derecho-jcm-runs`** — PBS allocates exclusive GPUs; pre-flight the Hydra
  composition before spending a queue slot instead.

`JAX_PLATFORMS=cpu` is for **unit tests only**. Anything beyond ~5 simulated
days belongs on a GPU.

## Stability: why the overrides matter

A T63L47 ECHAM run started from an isothermal cold start with no sponge
**will go NaN within a few days**. The stable recipe needs:

- `init=jw init.rh=0.0` — Jablonowski-Williamson balanced initial state.
  `init=isothermal` on a real-orography grid is not a viable start.
- `terrain=from_file` + `forcing=from_file` — real orography/land-sea mask and
  SSTs.
- `run=longrun` — this already carries the **settled production sponge**
  (`levels=10, timescale_h=1.5, enspodi=2.0, damp_temperature=true,
  target_T_K=250`, rationale in `run/longrun.yaml`). Do **not** re-specify
  those on the command line: duplicating them invites drift from the
  validated values, and `+run.sponge.target_T_K=...` now fails outright with
  `An item is already at 'run.sponge.target_T_K'` because the key exists.
  The sponge is what arrests the cold-cap runaway at L47 — the absolute
  target catches the m=0 zonal mean that pure zonal-mean damping cannot
  touch, without which the top level drifts ~4 K/hr to NaN before the first
  save. Level-dependent diffusion alone does **not** do this.

## Hydra gotchas

- **`run.time_step` is in MINUTES**, not seconds.
- **`save_interval` must be ≤ `chunk_days`.** Otherwise a chunk contains zero
  output times and the chunk write dies with a confusing
  `IndexError: index 0 is out of bounds for axis 0 with size 0` from
  `predictions.to_xarray()`. This is easy to hit when shortening a run for a
  quick test and forgetting to shorten `save_interval` with it.
- `run/longrun.yaml` has **no** `checkpoint_path` key, so adding one needs
  Hydra's add syntax: `+run.checkpoint_path=...`. Plain
  `run.checkpoint_path=...` fails with `Key 'checkpoint_path' is not in
  struct`. `run/default.yaml` does define it.
- **Ozone**: `forcing.ozone_file: auto` is the shipped default and resolves a
  packaged climatology matching the grid (`jcm/data/bc/t63/ozone.nc` — already
  on L47 levels, already S→N). Leave it alone. Confirm in the log:
  `forcing.ozone_file=auto resolved to .../t63/ozone.nc`. If instead you see a
  warning about the **ANALYTIC** profile, the grid did not match and the run
  has ~7.6× the tropospheric ozone column — a large clear-sky OLR bias, and
  not a valid basis for any radiation comparison.
  Regenerate for a new grid with
  `python -m jcm.data.bc.interpolate_ozone --in T63_ozone_picontrol.nc --out
  jcm/data/bc/<grid>/ozone.nc --nlevels 47`.

## Watching a run

One `tail -F` with an alternation that covers **both** progress and every
failure signature — a filter that only matches success is silent through a
crash, which reads identically to "still running":

```bash
tail -F -n 0 run_logs/PREFIX.log 2>/dev/null | grep -E --line-buffered \
  "Saved .*_day[0-9]+\.nc|Wall: |NaN vars|unhealthy|Traceback|Error|FAILED|Killed|OOM|CUDA_ERROR|HydraException"
```

`NaN vars: N/239` is the health-check line. Parse the count — do **not** grep
for the bare string `nan`, which matches unrelated output.

Beware: `specific_humidity` is **labelled** `g/kg` in the health check and
netCDF but the values are `kg/kg`. It looks 1000× too dry and isn't.

## Failure modes worth recognising

- **Chunk write crashes after "Run completed"** — a diagnostic emitted a shape
  `data_to_xarray` has no dims for. The write order is `to_xarray →
  check_health → to_netcdf → save_checkpoint`, so a crash here loses the whole
  chunk with no checkpoint. Fix by adding the dotted key to
  `ComposablePhysics._EXCLUDED_OUTPUT_KEYS` or registering a band coord.
- **Editable installs**: `jcm`, `jax-rrtmgp`, `dinosaur` and `mam4-jax` are
  installed editable, so **the working tree is the running code**. Check
  `git -C <repo> rev-parse --abbrev-ref HEAD` before trusting any result. The
  site skills cover how to A/B a library version safely (worktree +
  `PYTHONPATH`, never `git checkout` in a shared clone).
