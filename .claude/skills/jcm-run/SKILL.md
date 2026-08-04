---
name: jcm-run
description: Launch a jcm model run through the built-in Hydra configs — pick config groups, apply the validated T63L47 overrides, isolate a GPU, and watch for startup failures. Use whenever running `python -m jcm.main`, including as the basis for benchmarking or a production sweep.
---

# Running jcm

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
        forcing.ozone_file=$REPO/jcm/data/bc/T63L47_ozone_picontrol_latflip.nc \
        run=longrun \
        run.time_step=12 run.save_interval=5 run.chunk_days=30 \
        run.sponge.levels=10 run.sponge.timescale_h=1.5 run.sponge.enspodi=2.0"

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

## Pre-flight: pick a genuinely free GPU

The box is shared, and stacking runs invalidates any timing and can OOM the
other tenant. Check **both** memory and running compute apps — neither column
alone is sufficient:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

Pick an index with no compute app *and* near-zero used memory. Set
`CUDA_VISIBLE_DEVICES=<idx>` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` so
concurrent runs on *different* GPUs don't each grab the whole HBM.

`JAX_PLATFORMS=cpu` is for **unit tests only**. Anything beyond ~5 simulated
days belongs on a GPU.

## Stability: why the overrides matter

A T63L47 ECHAM run started from an isothermal cold start with no sponge
**will go NaN within a few days**. The stable recipe needs:

- `init=jw init.rh=0.0` — Jablonowski-Williamson balanced initial state.
  `init=isothermal` on a real-orography grid is not a viable start.
- `terrain=from_file` + `forcing=from_file` — real orography/land-sea mask and
  SSTs.
- `run.sponge.levels=10` with `damp_temperature` — the upper sponge is what
  arrests the cold-cap runaway at L47. Level-dependent diffusion alone does
  **not** do this.
- For ICs that are not radiatively equilibrated (JW-dry), also set
  `+run.sponge.target_T_K=270` — the zonal-mean relaxation by construction
  cannot touch the m=0 mode, and without an absolute target the zonal-mean
  top-level T drifts several K/hr straight to NaN.

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
- The ozone file's latitudes are checked against the model grid to 0.001°.
  The CMIP6-style file ships N→S while the model grid is S→N — use the
  `..._latflip.nc` sibling. Never overwrite the original; the shared BC files
  are not regenerable in one step.
- The conda env is not on `PATH`. Invoke
  `/home/dwatsonparris/micromamba/envs/jcm/bin/python` directly, or
  `eval "$(micromamba shell hook --shell bash)" && micromamba activate jcm`.

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
  `git -C <repo> rev-parse --abbrev-ref HEAD` before trusting a result, and
  use a git worktree + `PYTHONPATH=<worktree>` to A/B a library version
  without disturbing anyone else's runs on the shared box.
