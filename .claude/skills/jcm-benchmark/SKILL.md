---
name: jcm-benchmark
description: Measure jcm throughput reproducibly — short (1 month) or long (12 month) runs on a validated stable config, with GPU memory/utilisation logging and an explicit convergence criterion. Use when comparing hardware, library versions (jax-rrtmgp, dinosaur), physics presets, or checking for a performance regression.
---

# Benchmarking jcm

Builds on the `jcm-run` skill — read that first for config groups, GPU
selection and the Hydra gotchas. This skill is about getting a throughput
number that is *actually true*, which is harder than it looks.

## Run it

```bash
PY=/home/dwatsonparris/micromamba/envs/jcm/bin/python

# Short: 1 month (30 d), ~6 chunks. The default for any A/B.
$PY tools/benchmark.py --preset t63-echam-rrtmgp --months 1 --gpu 1 \
    --label baseline

# Long: 12 months. Use for provisioning and drift, not for A/B.
$PY tools/benchmark.py --preset t63-echam-jam --months 12 --gpu 3 \
    --chunk-days 30 --label jam-year

# A/B a library version without touching the shared editable install
$PY tools/benchmark.py --preset t63-echam-rrtmgp --months 1 --gpu 1 \
    --label rrtmgp-perf \
    --pythonpath /data/dwatsonparris/jax-rrtmgp-worktrees/perf
```

Presets are in `tools/benchmark.py:PRESETS`. Each carries the **validated
stable override set** for its grid, not just `physics=`/`grid=` — see
"Configuration" below. Results land in
`/scr/dwatsonparris/benchmarks/<label>/` as `report.md`, `result.json`,
`run.log` and `gpu.csv`.

Runs take tens of minutes; launch with `run_in_background` and watch with a
Monitor on `run.log`, or the tool call will time out.

## Methodology — the part that matters

**Never quote jcm's `N sim days/hr` log line.** It is *cumulative including
compile time*, so it understates throughput by 2-5x early and drifts upward
for the whole run. A 5.3x regression was once filed against jax-rrtmgp on the
strength of chunk 1 of a run that settled 22x faster; it had to be retracted.
`tools/benchmark.py` ignores that line and parses `Wall: Xs this chunk`.

**Discard chunk 1 — it contains compilation.** For T63L47 ECHAM+JAM, compile
alone is several minutes.

**Require convergence.** Chunk times settle over 3-4 chunks (XLA autotuning,
cache warming, allocator). The tool reports a rate only once two consecutive
chunks agree within 3%, and marks the result `NOT CONVERGED` otherwise rather
than quietly reporting the last chunk. If you see that flag, run more chunks —
do not report the number.

**GPU utilisation is not a convergence signal.** XLA's autotuner keeps the
device at 95%+ while it is still *choosing kernels*. "The GPU is pegged" is
not evidence a chunk time is steady-state. Only chunk-to-chunk agreement is.

**One run per GPU.** Stacking invalidates the timing and can OOM the other
tenant. Check for free GPUs with both memory *and* compute-apps queries (see
`jcm-run`). Pick a genuinely idle card even if it means waiting.

**A NaN'd run is not a benchmark.** The tool reports `nan_any` and exits
non-zero. Timing from a run that blew up mid-flight is meaningless — fix the
configuration and re-run rather than quoting the chunks before the blow-up.

## Configuration

The presets are not `physics=X grid=Y` — they carry the whole known-stable
override set, because a T63L47 run from an isothermal cold start with no
sponge **goes NaN within days**, which silently destroys the benchmark. Each
T63 preset pins `init=jw init.rh=0.0`, real terrain/forcing/ozone from file,
and the upper sponge with `target_T_K=270`.

Adding a preset: put the *complete* validated override set in `PRESETS`, and
verify it completes the short benchmark NaN-free before using it for
comparisons.

`--save-interval` is clamped to `--chunk-days` by the tool: a chunk with zero
output times dies in `to_xarray()` with an opaque
`IndexError: index 0 is out of bounds for axis 0 with size 0`.

## A/B'ing a library version

`jcm`, `jax-rrtmgp`, `dinosaur` and `mam4-jax` are **editable installs — the
working tree is the running code.** Never `git checkout` a different branch in
the shared clone to benchmark it; that silently changes the code under anyone
else's concurrent run.

Instead make a worktree and point the benchmark at it:

```bash
git -C /data/dwatsonparris/jax-rrtmgp worktree add \
    /data/dwatsonparris/jax-rrtmgp-worktrees/perf origin/<branch>
$PY tools/benchmark.py ... --pythonpath /data/dwatsonparris/jax-rrtmgp-worktrees/perf
```

Verify the override actually took before trusting the result:

```bash
PYTHONPATH=<worktree> python -c "import rrtmgp,os; print(os.path.dirname(rrtmgp.__file__))"
```

Run baseline and candidate on the **same GPU model**, ideally the same card,
and interleave if the box is noisy. Report both absolute numbers, not just the
ratio.

**Benchmark A/Bs must attribute to a single change.** A comparison across a
merge commit that bundled two independent changes is not evidence about
either one — that error is precisely what made the jax-rrtmgp#22 diagnosis
wrong (the clamps were blamed; the real cost was a `while_loop`→`scan` rewrite
in the same PR). If the delta spans more than one change, bisect before
attributing.

## Interpreting the report

- `s_per_sim_day` — the primary comparable number.
- `sim_years_per_day` — for provisioning ("can we afford a 100-year run?").
- `peak_mem_gib` — includes the compile-time allocation spike, which is a real
  provisioning requirement. f32 is required above T63 on 80 GB cards.
- `median_util_pct` / `median_power_w` — computed over post-compile samples
  only. Power well below the card's rating alongside high utilisation
  indicates a memory-bound or launch-bound workload, not a compute-bound one.

## Known reference points

Re-measure rather than trusting these; they are here to catch order-of-
magnitude mistakes, and some were taken before the jax-rrtmgp minor-gas scan
fix (which changes radiation cost substantially).

- Radiation is ~87% of an ECHAM+RRTMGP step, so any RRTMGP change dominates.
- The AeroCom diagnostic groups cost ~9.4% together at T63L47.
- f32 (`MAM4_JAX_ENABLE_X64=0`) is **required** above T63 and is forward-only:
  MAM4 microphysics gradients are non-finite in f32.
