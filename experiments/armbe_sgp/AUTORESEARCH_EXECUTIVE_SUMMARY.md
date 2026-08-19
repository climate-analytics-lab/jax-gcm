# Executive Summary: Foundation for Cloud-Physics Autoresearch

## What Has Been Built

The current infrastructure covers the full research loop:

```text
ARM observations and quality control
  -> SPEEDY-compatible atmospheric states
  -> reproducible train/validation/test blocks
  -> configurable feature tables
  -> symbolic-regression search and frontier evaluation
  -> train-only coefficient calibration
  -> online JAX/SPEEDY implementation and parity tests
  -> prescribed-state ERA5 cloud and RSUT evaluation
  -> mechanism counterfactuals and reproducibility ledger
```

The observational substrate contains 79,409 valid hourly samples from 15 ARM
site-facilities. Profiles are mapped to the eight SPEEDY sigma levels, and each
sample can be passed through the model's one-step diagnostics to generate
features that are available at runtime. Whole site-month blocks define the
current training, validation, and test sets, with 54,229, 16,267, and 8,913
samples, respectively.

The evaluation layer independently diagnoses physics on same-time ERA5 states.
It compares both total cloud and top-of-atmosphere reflected shortwave radiation
(RSUT), uses area weighting and paired daily-window uncertainty, and now covers
both a 120-window reference benchmark and a 240-window 2011-2020 extension. This
separates cloud-amount skill from radiative consequences before candidates are
advanced to more expensive forecast or coupled tests.

## Recommended pre-fixed items (mfisher)

### 1. State Reconstruction And Model Context

ARM profiles must be converted into model-compatible states before online
features can be evaluated. Potential knobs include:

- vertical grid and interpolation method;
- treatment of profile edges and missing winds;
- geopotential reconstruction;
- station versus model-grid terrain and land fraction;
- surface and climatological forcing;
- one-step diagnostic duration and initialization of physics carry.

These choices can materially alter derived features. They should normally be
locked within a campaign and changed only as explicit ablations.

### 2. Data Splits And Outer Validation

The current split uses whole `(site, year-month)` blocks, stratified by month.
This prevents direct row-level leakage but is not a leave-site-out or strict
future-climate test. Future outer protocols can use:

- leave-site-out or leave-region-out folds;
- chronological per-site holdouts;
- later-year temporal holdouts;
- weather-event embargoes across block boundaries;
- nested cross-validation over site-month blocks;
- disjoint ERA5 development and final evaluation periods.

### 3. Cloud Architecture Model Integration

An equation's output can be coupled to SPEEDY in several ways. Relevant knobs
include:

- output clipping and behavior for invalid values;
- prediction of one total-cloud component versus multiple components;
- cloud-top diagnosis and vertical placement;
- overlap assumptions;
- cloud optical properties and albedo;
- interaction with convection, condensation, and boundary-layer schemes;
- diagnostic versus prognostic memory.

The pilot changed cloud amount but retained SPEEDY's cloud-top and optical
treatment, while setting the separate stratocumulus component to zero. This
implementation choice, not just the symbolic equation, proved central to RSUT.

### 4. Evaluation Ladder

Candidate evaluation progresses through increasingly expensive gates:

1. Offline ARM train/validation/outer-test performance.
2. Online/offline feature and equation parity.
3. Same-time ERA5 cloud amount, distributions, and regime errors.
4. Radiatively weighted and common-operator diagnostics.
5. Full RSUT and other energy-budget metrics.
6. Short independent forecasts and tendency stability.
7. Longer coupled forecasts or climate simulations.

The current work establishes gates 1-5. Forecast stability and coupled climate
impact remain future stages. Evaluation knobs include dates, regions, variables,
weighting, uncertainty unit, surface forcing, lead times, and acceptance
thresholds.

### 5. Reproducibility And Computational Controls

Every autoresearch iteration should record immutable input artifacts, source
revision, environment, hardware, full configuration, random seeds, sampled row
indices, complete symbolic frontier, fitted coefficients, online equation,
per-window metrics, timings, and rejected or failed candidates.

Computational knobs include parallelism, deterministic mode, state caching,
feature caching, and staged fidelity. ERA5 remapping currently dominates runtime;
caching prepared T30 sigma-level states will make repeated closure and
counterfactual evaluation practical without changing the science.

## Recommended autoresearch items (mfisher)

The principal contribution of the decision ledger is to make the research
degrees of freedom explicit. A future agent should vary these knobs deliberately
rather than silently changing several parts of the experiment at once.

### 1. Scientific Target And Objective

The pilot target is hourly ARMBECLDRAD column-total cloud fraction, optimized
with unweighted mean squared error. This is only one possible formulation.
Future campaigns can vary:

- target variables: total cloud, low/mid/high cloud occurrence, layer cloud
  fraction, cloud component proxies, radiative fluxes, or jointly defined
  cloud-radiation targets;
- target aggregation: instantaneous, hourly, regime-conditioned, vertically
  aggregated, or temporally persistent quantities;
- loss functions: MSE, robust losses, relative or heteroscedastic losses, and
  multi-objective cloud-plus-radiation criteria;
- sample weighting: equal rows, equal sites, equal climate regimes, seasonal
  balance, cloud-regime balance, or radiative/insolation weighting;
- physical penalties: bounds, monotonicity, dimensional consistency,
  conservation, smoothness, and numerical-safety penalties.

This is the highest-level autoresearch choice. The pilot demonstrates that a
better scalar total-cloud objective does not necessarily produce a better
radiative parameterization.

### 2. Observational Data And Sampling

The present `tot_cld` target is read directly from ARMBE; it is not a vertical
sum of layer cloud fractions. More sophisticated component targets will require
verified product definitions because column occurrence and layer occurrence are
not interchangeable.

### 3. Feature Space

The completed search matrix establishes three nested reference groups:

- 5 baseline SPEEDY features: maximum RH, precipitation, stability, lowest-level
  RH, and land fraction;
- 14 features: additional fixed-sigma RH summaries, vertical RH structure, and
  inversion/lapse-rate diagnostics;
- 18 features: additional saturation deficit, precipitable water, low-level wind,
  and boundary-layer shear.

Future feature knobs include profile summaries, vertical basis functions,
cloud-regime indicators, memory or tendency information, surface coupling,
convection and condensation diagnostics, and features tailored to component or
radiative targets. Every candidate feature must have the same offline and online
definition and must be available when the parameterization executes.

Feature preprocessing is also searchable. The pilot used raw documented units
with no scaling, learned embedding, dimensional constraint, or feature
preselection. Future campaigns may compare normalization, dimensional analysis,
safe transformations, constrained feature groups, and learned summaries while
preserving an interpretable runtime path.

### 4. Symbolic Search Configuration

The pilot executed a controlled 12-cell matrix spanning:

- 5, 14, or 18 features;
- 2,000 or 5,000 sampled training rows;
- `maxsize/maxdepth` of `20/5` or `50/8`.

The initial PySR configuration used 80 iterations, 12 populations of 33
expressions, arithmetic operators, and a selected set of nonlinear operators.
Searchable knobs include:

- operator vocabulary, operator costs, and nested constraints;
- expression size, depth, and explicit parsimony;
- training sample size, batching, and curriculum;
- iterations, populations, population size, and early stopping;
- dimensional and argument constraints;
- expression templates or modular subexpressions;
- deterministic execution and independent random-seed ensembles;
- numerical-safety and physical-plausibility constraints during search.

The existing matrix has one evolutionary run per configuration, not a seed
ensemble. Recurrence across seeds or folds should become an explicit acceptance
criterion rather than an informal preference.

### 5. Frontier Selection And Model Choice

Within each pilot run, every Pareto-frontier expression was reevaluated on the
full validation set, and minimum validation RMSE selected the within-run model.
The final implemented equation was then chosen by human judgment favoring
compactness, interpretability, numerical safety, and recurrence across searches.

Future autoresearch must predeclare a mechanical cross-run rule. Possible knobs
include:

- validation error versus expression complexity;
- practical-equivalence thresholds;
- recurrence across seeds, sites, or folds;
- cloud and radiation as separate Pareto objectives;
- rejection gates for nonfinite behavior, unsafe algebra, unavailable features,
  or poor online parity;
- preference for expressions with physically interpretable monotonic responses.

This selection layer is as important as the symbolic search itself. Without a
predeclared rule, repeated validation inspection creates meta-overfitting even
when each individual fit uses training rows correctly.

### 6. Coefficient Calibration

The pipeline can freeze an expression's structure and fit a small number of
identifiable coefficients on all training rows. Calibration knobs include:

- which constants are free;
- parameterization and identifiability;
- bounds and initialization;
- optimizer, precision, regularization, and convergence criteria;
- cloud-only versus multi-objective calibration;
- sequential structure-then-coefficient fitting versus joint optimization.

The pilot's coefficient calibration changed validation RMSE negligibly and
slightly worsened test RMSE. This is useful negative evidence: additional
decimal-level tuning is less important than target and architecture design.


## Initial Demonstration Result

The first pipeline demonstration selected the compact equation

```text
tanh(
  (rh_low_mean + rh_mid_mean)
  * (rh_vertical_range^3 + rh_high_mean)
)
```

It uses only four RH summaries despite access to a broader feature set. On ARM,
test cloud RMSE improves from 0.34379 for calibrated SPEEDY to 0.31109, about
9.5%. Train-only coefficient fitting did not improve test generalization.

The result transfers to prescribed ERA5 states:

| Benchmark | Scheme | Cloud RMSE | RSUT RMSE (W m-2) |
| --- | --- | ---: | ---: |
| 120 windows, 2016-2020 | Calibrated SPEEDY | 0.21112 | **40.738** |
| 120 windows, 2016-2020 | Nested RH | **0.19035** | 43.067 |
| 240 windows, 2011-2020 | Calibrated SPEEDY | 0.21074 | **40.559** |
| 240 windows, 2011-2020 | Nested RH | **0.19077** | 43.099 |

In the 240-window benchmark, nested RH reduces cloud RMSE by 9.5% but increases
RSUT RMSE by 2.540 W m-2, or 6.3%. Both paired differences are robust across
daily windows.

## Why Better Cloud RMSE Gives Worse RSUT

This apparent contradiction is the most important lesson from the pilot for
future autoresearch design.

Native SPEEDY separately diagnoses generic RH/precipitation cloud (`cloudc`) and
stability-driven boundary-layer stratocumulus (`cloudstr`). They are placed and
treated differently by radiation. The nested-RH implementation predicts one
total-cloud component and sets `cloudstr=0`.

A 240-window diagnostic held nested RH's improved total cloud fixed but
applied calibrated SPEEDY's local `cloudstr/(cloudc+cloudstr)` ratio. RSUT RMSE
fell from 43.099 to 41.572 W m-2, removing 60.1% of the RSUT penalty in all 240
windows. Replacing cloud top alone was negligible. Incoming-solar-weighted cloud
RMSE improved by 10.3%, and a common-shortwave-operator cloud proxy improved by
9.8%; they favored nested RH in 239 and 235 windows, respectively. Thus nested RH
is not merely improving cloud in radiatively
unimportant locations; the loss of component placement is the dominant tested
mechanism.

The SPEEDY ratio is an empirical model diagnostic, not an observed physical
partition, and the ratio test remains post-hoc. Its value is diagnostic: it
shows that future autoresearch should expose component structure, vertical
placement, optical treatment, and radiation-aware objectives as searchable
dimensions rather than optimizing only a scalar total-cloud target.

## Recommended Autoresearch Program (chatgpt)

1. Freeze and version the existing data, split, online-parity, and ERA5
   evaluation infrastructure as the reference substrate.
2. Establish a fresh outer holdout before any new adaptive search.
3. Define a richer target suite, prioritizing radiatively distinct cloud
   structure and separately reported cloud and radiation objectives.
4. Expand only model-available features, with explicit ablations by feature
   family and verified online definitions.
5. Predeclare a cross-run selection rule combining generalization, recurrence,
   complexity, safety, and multi-objective performance.
6. Run independent seed/fold ensembles rather than one evolutionary search per
   configuration.
7. Advance candidates through the staged evaluation ladder, rejecting those
   that improve offline cloud RMSE but fail online, radiative, or stability
   gates.
8. Cache remapped ERA5 states and use staged-fidelity subsets so computation is
   spent on search rather than repeated preprocessing.
