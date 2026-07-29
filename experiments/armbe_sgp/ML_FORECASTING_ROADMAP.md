# Differentiable SCM Forecasting Roadmap

## Goal

Turn SPEEDY single-column experiments into a differentiable forecasting problem:

```text
SPEEDY cloud parameters
  -> free 30-minute physics rollouts of configurable duration
  -> model-to-ARMBE cloud observation operator
  -> QC/coverage mask
  -> RMSE loss
  -> JAX gradient + optimizer
```

The first target is to test and calibrate the SPEEDY cloud-cover closure against
ARMBE while preserving an honest held-out forecasting evaluation. Forecast
horizon is an experiment parameter, with initial cases such as one day, one
week, and one month.

## Prerequisites

### 1. Experiment policies

- Physics model runs in 30-minute timesteps. Horizon (for evaluation and resetting against observational data)
  is configurable.
- Quality Control: Flagged hours (non-0 QC) are omitted from evaluation, since they are rare (< 3%).
- Missing data timesteps are omitted (25% missing, pretty rough)
- Record valid hourly sample counts and dropped-window reasons in experiment manifest. 

## Cache And Evaluation Usage

Build the offline cache before invoking JEM-Cal. The YAML config requires
`atm`; `cldrad` is required for the default `cloud_fraction` target. The timing
fields below are the supported defaults and must satisfy: horizon is divisible
by the physics timestep, horizon includes one observation cadence, and stride is
a multiple of observation cadence.

```yaml
atm: "data/sgparmbeatmC1.c1.20180101.000000.nc"
cldrad: "data/sgparmbecldradC1.c1.20180101.000000.nc"
start: "2018-09-03"
end: "2018-10-02"
nlev: 8
physics_dt_minutes: 30
horizon_minutes: 360
stride_minutes: 360
observation_cadence_minutes: 360
target:
  observation: cloud_fraction
  model: shortwave_rad.cloudc
  reduction: trajectory
```

```bash
python forecast_cache.py --config cache_config.yaml --cache outputs/cache_2018_fall
PYTHONPATH=/path/to/jax-gcm:/path/to/JEM-Cal/src \
  python /path/to/JEM-Cal/examples/evaluate_jcm_armbe.py --config evaluation_config.yaml
```

The evaluator config requires `cache` and `out_dir`; select `all` or one named,
half-open chronological split. Do not randomly split overlapping or adjacent
windows. Keep training, validation, and held-out test intervals disjoint in
their underlying time ranges, and leave the test block untouched while choosing
parameters.

```yaml
cache: "outputs/cache_2018_fall"
out_dir: "outputs/eval_2018_fall_test"
split: test
batch_size: 8
splits:
  train: {start: "2018-09-03", end: "2018-09-20"}
  validation: {start: "2018-09-20", end: "2018-09-26"}
  test: {start: "2018-09-26", end: "2018-10-02"}
```

The cache writes `windows.nc` (initial states, prescribed surface-temperature
record, targets, and QC mask), `config.json`, `recipe.json`, and `manifest.json`.
The evaluator writes `predictions.nc`, `lead_metrics.csv`, and `run_manifest.json`
even when execution fails. The cache manifest records its resolved configuration,
recipe (including resolved second-based timing), retained-state count, and window count. The evaluator manifest records
the resolved configuration, selected windows, output paths, and available git
revisions.

The surface-temperature record is prescribed at every forecast step as SPEEDY
land `stl_am`. ARMBE `temperature_sfc` is a 2 m air-temperature proxy, not a
verified skin temperature. Consequently these are boundary-forced atmospheric
forecasts, not fully free land-surface forecasts: future surface observations
remain available during each rollout and this caveat must accompany results.


### 2. Define the Cloud Observation Operator

The current SPEEDY comparison uses `shortwave_rad.cloudc`. It is diagnosed from
relative humidity and precipitation. SPEEDY separately diagnoses `cloudstr`, a
low-level stratiform quantity used in shortwave optical properties.

ARMBE `tot_cld` is an observed narrow-field-of-view total cloud fraction. A
first loss can compare QC-passed, consistently aggregated `cloudc` and
`tot_cld`, but this must be labelled as an imperfect observation operator.

Do not define total model cloud as `cloudc + cloudstr` without an explicit
cloud-overlap model. Save both diagnostics and assess the following hypotheses
against QC-passed `tot_cld` before choosing a training target:

\[
C_{\mathrm{max}} = \max(C_{\mathrm{cloudc}}, C_{\mathrm{cloudstr}}),
\]

\[
C_{\mathrm{random}} = C_{\mathrm{cloudc}} + C_{\mathrm{cloudstr}}
                       - C_{\mathrm{cloudc}}C_{\mathrm{cloudstr}},
\]

\[
C_{\mathrm{disjoint}} = \min(1,
                               C_{\mathrm{cloudc}} + C_{\mathrm{cloudstr}}).
\]

These are maximum-overlap, random-overlap, and zero-overlap bounds, not known
properties of the current SPEEDY closure. `cloudstr` is a low-level shortwave
optical diagnostic, so even the best of these may not be a faithful all-sky
cloud-fraction observation operator.

## Batched Forecast Core

Replace the Python loop that constructs one SCM per six-hour window with a pure
JAX free-rollout function:

- Batch initial states with shape `(n_windows, n_levels)`.
- Accept `horizon_minutes` and `stride_minutes` as experiment configuration.
  The resolved configuration converts these to seconds before computing
  `horizon_steps = horizon_seconds / physics_dt_seconds`.
- Advance `horizon_steps` with `jax.lax.scan`; the horizon is static for each
  compiled experiment but can be changed between runs.
- Evaluate independent rollout cases in parallel with `jax.vmap`.
- Reset atmospheric state, tracers, and physics carry per independent rollout
  case, not per six-hour observation interval.
- Hold surface pressure and geopotential at their observed initial state unless
  an explicitly different experiment is introduced.
- Select forcing using every rollout step's real calendar time. Observed
  surface-temperature forcing may vary during the rollout; ARMBE atmospheric
  profiles must not be injected after initialization.

The batched function must return per-step cloud diagnostics, interval means,
trajectory states, and final profiles without converting arrays to NumPy inside
the differentiable path.

The rollout start times are selected with `stride_minutes`. Starts may overlap,
but train/validation/test splits must be chronological and non-overlapping in
their underlying time ranges to prevent leakage.

### Surface Boundary Condition

Phase one prescribes the time-varying ARMBE `temperature_sfc` record as
SPEEDY's `stl_am` land-temperature boundary condition throughout every rollout.
The field is a 2 m air-temperature proxy, not a verified skin temperature.

This is a boundary-forced atmospheric forecast, not a fully free forecast:
future surface observations are available to the model during the rollout. The
experiment manifest and every result must label this choice explicitly. The
current SPEEDY SCM has no prognostic land/skin-temperature equation, so a fully
free land-surface forecast requires separate physics development. A future
frozen-initial-temperature sensitivity can be added as a distinct experiment.

## Trainable Cloud Parameters

Start with the existing SPEEDY shortwave cloud parameters:

| Parameter | Meaning |
|---|---|
| `rhcl1` | Relative-humidity threshold for cloud onset |
| `rhcl2` | Relative-humidity threshold for saturated RH cloud contribution |
| `qacl` | Minimum specific humidity for an eligible cloud layer |
| `wpcl` | Precipitation-cloud weight |
| `pmaxcl` | Cap on precipitation contributing to cloud cover |

Use transformed unconstrained optimizer variables so that:

\[
0 < RH_{cl1} < RH_{cl2} \leq 1, \qquad q_{acl} > 0, \qquad
w_p > 0, \qquad p_{max} > 0.
\]

Leave low-cloud/stability parameters for a later sensitivity after the main
closure and observation operator are validated.

## Losses and Baselines

Initial objective:

\[
\mathcal{L}_{cloud} =
\sqrt{\frac{1}{N}\sum_{i \in \mathrm{valid}}
       (C_{model,i} - C_{ARMBE,i})^2}.
\]

The valid set includes only QC-passed, sufficiently sampled observations.
Loss can be evaluated at the forecast endpoint, across every valid observed
snapshot in the trajectory, or both. Report lead-time-dependent losses rather
than only one aggregate value.

Keep the following as secondary diagnostics rather than initial optimization
objectives:

- Surface and TOA radiation.
- Precipitation.
- Surface turbulent fluxes.
- Final temperature, humidity, and wind profile errors.

Required baselines:

1. Default SPEEDY cloud parameters.
2. Persistence for final-profile forecasts.
3. A simple RH-only cloud closure, if useful for interpretation.

Use daily means for radiation scores because SPEEDY shortwave has no local
solar diurnal cycle. Cloud loss may use hourly or six-hour aggregation only
when model and observation use identical intervals.

Run a horizon sweep, for example six hours, one day, one week, and one month.
Longer free rollouts are deliberately harder because this SCM has no dynamics,
advection, or large-scale forcing. A future forcing-constrained rollout using
VARANAL must be a separately labelled experiment.

## Evaluation Splits

Do not randomly split adjacent windows because weather regimes are temporally
correlated.

- Use chronological training, validation, and held-out test blocks.
- Keep the test period untouched during parameter selection.
- Treat other ARM sites as future cross-site test sets, not merely extra
  training samples.
- Record data products, QC rules, date ranges, splits, parameter priors, and
  code revision with every optimization run.

## Gradient Verification

Before fitting any parameter:

1. Evaluate `jax.value_and_grad(loss)` with respect to cloud parameters.
2. Require finite loss and gradients.
3. Compare selected gradients with finite differences.
4. Inspect gradient coverage across cloud regimes.

The current closure contains clipping, maxima, and `argmax`, so it is only
piecewise differentiable. Do not introduce smooth replacements preemptively;
use them only if gradient checks show sparse or unstable optimization behavior.

## First Milestone

Implement a batched, QC-masked 30-minute free-rollout loss for the default
cloud parameters using a fixed chronological train/validation/test split. Make
the forecast horizon and start stride configurable, beginning with six-hour
and one-day runs. Produce lead-time-resolved default-loss baselines before
optimizing any parameter.
