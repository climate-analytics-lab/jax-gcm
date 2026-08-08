---
name: jcm-benchmark
description: Measure jcm throughput reproducibly — short (1 month) or long (12 month) runs on a validated stable config, with GPU memory/utilisation logging and an explicit convergence criterion. Use when comparing hardware, library versions (jax-rrtmgp, dinosaur), physics presets, or checking for a performance regression.
---

# Benchmarking jcm

Builds on `jcm-run` (config groups, Hydra traps) and the machine skill for
your site — `devbox-jcm-runs` or `derecho-jcm-runs`. This skill is about getting a throughput
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
`tools/benchmark.py` ignores that line and parses `Wall: Xs this chunk`. The
reduction lives in `tools/chunk_timing.py`, shared with the Derecho skill's
`settled_rate.py` so a run cannot get two different answers depending on
which tool read it.

**Discard chunk 1 — it contains compilation.** For T63L47 ECHAM+JAM, compile
alone is several minutes.

**Require convergence.** Chunk times settle over 3-4 chunks (XLA autotuning,
cache warming, allocator). A rate is quoted only once the **last two** chunks
agree within 3% — judged on the last two rather than the first agreeing pair,
because a noisy series can match by accident early while still drifting, and
what matters is that the run had settled by the time it ended. Otherwise the
result is marked `NOT CONVERGED`; if you see that flag, run more chunks rather
than reporting the number. A 30-day run at `--chunk-days 5` gives 6 chunks,
which is comfortably enough; 10 days gives 2 and will correctly refuse.

**GPU utilisation is not a convergence signal.** XLA's autotuner keeps the
device at 95%+ while it is still *choosing kernels*. "The GPU is pegged" is
not evidence a chunk time is steady-state. Only chunk-to-chunk agreement is.

### Run on a genuinely free GPU. Always. No exceptions.

This is the hardest rule here. Simulations should always run on free cards;
**benchmarks especially**, because a contended card does not fail loudly — it
returns a plausible-looking number that is simply wrong, and you will not be
able to tell from the report. Every other precaution in this document is
wasted if the card was shared.

On a shared, unscheduled box, verify the target card is genuinely idle —
**both** no compute apps and near-zero resident memory, since either signal
alone looks idle for a parked job:

```bash
python tools/gpu_util.py          # every GPU and its tenants
python tools/gpu_util.py --free   # free indices; exit 1 if none
```

`tools/benchmark.py` calls the same check as a hard pre-flight gate and
refuses to start on a busy card. On a scheduled machine (Derecho) this is the
scheduler's job instead — see `derecho-jcm-runs`.

**Never stack two runs on one card.** It invalidates both timings and can OOM
the other tenant.

**Do not start other GPU work anywhere on the box mid-A/B.** This is the
non-obvious one. Even on a *different, genuinely free* card, a concurrent job
competes for host CPU, PCIe bandwidth and the allocator. If it overlaps only
the candidate arm and not the baseline arm, it contaminates exactly one side
of the comparison — an asymmetry that shows up as a fake speedup or
regression and is invisible in the report. Halving your wall time is not
worth a result you then cannot trust. Let the whole interleaved sequence
finish on one card before starting anything else.

The corollary: **parallelising an A/B across cards is not a shortcut.** If
you must (because a card is going away), run *both* arms on card A and both
on card B, and report the pairs separately — never baseline on A against
candidate on B.

**Compare like with like on chunk size.** Every chunk boundary costs a host
sync, a health check and a netCDF write. The short benchmark uses 5-day
chunks (to get ~6 chunks and so detect convergence) while a 12-month run uses
30-day chunks, so the short number carries slightly more per-day write
overhead than production. That is a couple of percent, and it cancels
entirely in an A/B at the same `--chunk-days` — but do not quote a
5-day-chunk number as the production throughput.

**The run must be a real one.** This is a production configuration made
shorter and more instrumented, not a reduced-physics proxy: real orography,
real SSTs, the packaged CMIP6 ozone, JW init and the production sponge. Check
the log line `forcing.ozone_file=auto resolved to .../t63/ozone.nc` — if it
warns about the **ANALYTIC** ozone profile instead, the radiation is seeing
~7.6x the tropospheric ozone column and the benchmark is measuring the wrong
workload.

**A NaN'd run is not a benchmark.** The tool reports `nan_any` and exits
non-zero. Timing from a run that blew up mid-flight is meaningless — fix the
configuration and re-run rather than quoting the chunks before the blow-up.

## Configuration

The presets are not `physics=X grid=Y` — they carry the whole known-stable
override set, because a T63L47 run from an isothermal cold start with no
sponge **goes NaN within days**, which silently destroys the benchmark. Each
T63 preset pins `init=jw init.rh=0.0`, real terrain and forcing from file,
and `run=longrun` — which carries the settled production sponge
(`target_T_K=250`). Ozone comes from `ozone_file: auto`, the shipped default,
which resolves the packaged CMIP6 climatology; the preset deliberately does
not override it.

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

**Pin EVERY editable install, not just the one under test.** Pinning only the
library you are varying leaves the others free to move mid-sweep — and they
do: a six-config sweep here had jax-rrtmgp switch from a feature branch to
`main` between config 1 and config 2 because someone merged a PR, so the two
were measuring different radiation code. Nothing failed; the numbers were
simply incomparable. It was caught only because the report records resolved
SHAs. Put all of `jcm`, `jax-rrtmgp`, `dinosaur` and `mam4-jax` on
`--pythonpath` as pinned worktrees for any run whose numbers you intend to
compare across hours.

**Pre-flight by BUILDING the model, not by composing the config.**
`--cfg job` resolves the config and stops; it never calls `build_model()`, so
backend-specific rejections pass it and then fail instantly on the GPU. Two
have bitten here: an invalid spectral truncation (coords are not built by
`--cfg job`), and `init=jw` on the pySES backend, which rejects it because it
initialises from its own resting USSA-1976 state.

```bash
JAX_PLATFORMS=cpu python -c "
from hydra import compose, initialize_config_dir
from jcm.runners import build_model
import os
with initialize_config_dir(config_dir=os.path.abspath('jcm/config'), version_base=None):
    cfg = compose(config_name='config', overrides=[...])
build_model(cfg)"
```

**Do not port overrides between backends by analogy.** The pySES preset needed
*fewer* overrides than the spectral one, not the same set: it ignores `grid`
and `run.time_step`, does its own tracer sub-cycling instead of
semi-Lagrangian advection, interpolates the packaged boundary fields onto its
columns, and rejects `init=jw`. Carrying that last one over "for consistency"
is exactly how it got in.

**Check the precision the configuration needs before launching.** f32
(`MAM4_JAX_ENABLE_X64=0`) is required above T63 and is forward-only. It
changes both memory and speed, so it cannot be switched on partway through a
sweep without making the halves incomparable — decide once, for all configs.
Memory scales with both axes: T63L47 18 GiB, T63L95 61 GiB, T106L47 62 GiB
on an 80 GiB card, so T106L95 needs both scalings at once and will not fit in
f64.

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
