# Bias-correction training and evaluation drivers

The scripts that produce and score a trained bias-correction term
(`jcm/physics/bias_correction`, documented in `docs/source/bias_correction.rst`).
They are drivers, not library code: each one chdirs to the repo root, expects
a GPU for everything except the taper and the tables, and is run once per
experiment rather than imported. Every argument that reproduces the shipped
term `jcm/data/bias_correction/online_term_t31_big_insol_vt.npz` is given
below in full.

| stage | script | what it does | GPU |
|---|---|---|---|
| 0 | `build_era5_cache.py` | per-year ERA5 targets and nudged start states, on disk outside the repo (about 1.1 GB per year, resumable) | yes |
| 1 | `train_offline_multiyear.py` | supervised warm start on recorded nudging tendencies | yes |
| 2 | `train_online_multiyear.py` | rollout curriculum, backprop through 6 h, 12 h, 24 h and 48 h free runs | yes |
| 3 | `finetune_climatology.py` | fine-tune against a 30-day free-run climatology, optionally adding a context feature | yes |
| 4 | `add_surface_taper.py` | post-hoc near-surface temperature taper; makes a `_vt` term | no |
| 5 | `evaluate_term.py` | free run, mean fields, the five scorecard metrics against ERA5 | yes |
| | `holdout_table.py` | scorecard for every term in an eval directory, with the noise floors | no |
| | `perturb_term.py` | weight-perturbed copies of a term, to measure the perturbation floor | no |
| | `run_longeval.sh` | 21-year free run cut into 7-year blocks, the sampling floor | yes |

## Reproduce the shipped term

Run from the repo root. Each script also runs from any other directory, since
it resolves the root from its own location; `JCM_REPO` overrides that.

```bash
# 0. cache. Lives outside the repo, default ~/era5_cache_t31.
python tools/bias_correction/build_era5_cache.py --years 1995-2015
python tools/bias_correction/build_era5_cache.py --years 2016-2022   # held-out span

# 1. offline warm start
python tools/bias_correction/train_offline_multiyear.py \
    --years 1995-2015 --val-years 2016-2022 --hidden 256,256,256 \
    --out jcm/data/bias_correction/offline_term_t31_big.npz

# 2. rollout curriculum. The update count matters, see below.
python tools/bias_correction/train_online_multiyear.py \
    jcm/data/bias_correction/offline_term_t31_big.npz \
    jcm/data/bias_correction/online_term_t31_big.npz \
    --years 1995-2015 --start-pool 21 --updates 500,500,500,250

# 3. climatology fine-tune, 30-day window, adding the insolation input
python tools/bias_correction/finetune_climatology.py \
    jcm/data/bias_correction/online_term_t31_big.npz \
    jcm/data/bias_correction/online_term_t31_big_insol.npz \
    --years 1995-2015 --start-pool 21 --n-steps 1440 --rollout-steps 48 \
    --lam 1.0 --updates 210 --lr 2.5e-5 --pole-floor 0.3 --context insol

# 4. surface taper, no GPU
CUDA_VISIBLE_DEVICES="" python tools/bias_correction/add_surface_taper.py \
    jcm/data/bias_correction/online_term_t31_big_insol.npz \
    jcm/data/bias_correction/online_term_t31_big_insol_vt.npz 0.7 1.0

# 5. score on the held-out years
M4_YEAR=2016 M4_DAYS=2645 M4_OUT_DIR=$PWD/eval_out_holdout \
    M4_TERM=jcm/data/bias_correction/online_term_t31_big_insol_vt.npz \
    M4_TAG=big_insol_vt python tools/bias_correction/evaluate_term.py
python tools/bias_correction/holdout_table.py --dir eval_out_holdout
```

Drop `--context insol` in step 3 for the no-context term, or use
`--context fmask` for the land-mask term. Drop `--hidden 256,256,256` in
step 1 for the 8,352-parameter network.

Three settings were each found by a term going non-finite in the 450-day
evaluation while its training log looked healthy: the rollout stage needs
about 1750 updates, not a count scaled to the start pool; the climatology
window must be 30 days rather than 90 when the start pool spans 21 years
(90 days gives gradient norms of 1e5 to 1e7 and the loss rises); rollout starts
must be spread over the seasonal cycle as well as over years.

## Environment knobs

All drivers read the same variables, so one setting drives the chain. The
`M4_` prefix is the milestone the evaluator was written in and is kept because
every recorded run and the shell drivers use it.

| variable | default | meaning |
|---|---|---|
| `M4_TRUNC` | 31 | spectral truncation |
| `M4_DT_MIN` | 30 | model step in minutes |
| `M4_TERM` | the shipped term | `.npz` to evaluate; a comma-separated list evaluates the ensemble mean |
| `M4_TAG` | `term` | output tag for `eval_fields_<tag>.nc` |
| `M4_OUT_DIR` | `<repo>/eval_out` | where field files are read and written |
| `M4_YEAR`, `M4_DAYS` | 2001, 450 | run start year and length; the canonical protocol is pinned in `jcm/physics/bias_correction/eval_protocol.py` |
| `M4_REF_YEARS`, `M4_REF2_YEAR` | derived | ERA5 reference span and spot-check year |
| `M4_BLOCK_YEARS`, `M4_CHUNK_DAYS` | unset | score a long run in consecutive blocks; integrate it in segments |
| `JCM_REPO` | script location | repo root override |
| `JCM_ERA5_URL` | the 64x32 WeatherBench2 store | ERA5 source for the cache builder |

Shared GPUs: set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (the training drivers
set it themselves) and pin a card with `CUDA_VISIBLE_DEVICES`.

## Measuring the floors before ranking two terms

A 7-year free-run mean carries the model's own internal variability, so a
difference between two terms is only a result if it clears the floors.
`perturb_term.py` measures the weight-perturbation floor, `run_longeval.sh`
the sampling floor from consecutive blocks of one long run, and
`holdout_table.py` reports both next to the seed spread and runs a paired
within-seed comparison. On the 2016-2022 holdout the perturbation floor is
0.06 to 0.12 K, the 7-year block floor 0.02 to 0.06 K, and reseeding the same
recipe moves 500 hPa by up to 2.8 K. Comparisons against plain SPEEDY clear all
of these by a wide margin; rankings between corrected terms mostly do not.
