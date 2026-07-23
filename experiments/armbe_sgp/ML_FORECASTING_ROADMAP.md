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

### 1. Use a 30-Minute Physics Timestep

The prescribed-state diagnostic runner must not use a six-hour physics step.
It should hold each available observed profile for twelve 30-minute substeps.
The experiment configuration must state whether physics carry and tracers reset
at each observed profile or persist within a contiguous segment.

The existing independent six-hour hindcast is a prototype of this approach, but
it resets after every six-hour window. The forecasting core must generalize it
to longer free rollouts without a reset inside the requested horizon.

### 2. Define Missing-Data Policy

- Do not interpolate missing atmospheric profiles.
- Split prescribed-state runs into contiguous profile segments and reset carry
  between segments.
- A valid initial profile permits a free rollout. Missing observed profiles
  inside that rollout do not alter model integration; they only mask the
  corresponding intermediate or endpoint scores.
- Reset atmospheric state, tracers, and physics carry only between independent
  rollout cases, never at an internal six-hour observation time.
- Apply ARMBE QC flags before scoring each target field.
- Record valid hourly sample counts and dropped-window reasons in archives and
  manifests.

### 3. Define the Cloud Observation Operator

The current SPEEDY comparison uses `shortwave_rad.cloudc`. It is diagnosed from
relative humidity and precipitation. SPEEDY separately diagnoses `cloudstr`, a
low-level stratiform quantity used in shortwave optical properties.

ARMBE `tot_cld` is an observed narrow-field-of-view total cloud fraction. A
first loss can compare QC-passed, consistently aggregated `cloudc` and
`tot_cld`, but this must be labelled as an imperfect observation operator.

Do not define total model cloud as `cloudc + cloudstr` without an explicit
cloud-overlap model.

## Batched Forecast Core

Replace the Python loop that constructs one SCM per six-hour window with a pure
JAX free-rollout function:

- Batch initial states with shape `(n_windows, n_levels)`.
- Accept `horizon_seconds` and `stride_seconds` as experiment configuration.
  With a fixed 30-minute physics step, `horizon_steps = horizon_seconds / 1800`.
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

The rollout start times are selected with `stride_seconds`. Starts may overlap,
but train/validation/test splits must be chronological and non-overlapping in
their underlying time ranges to prevent leakage.

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
