# Nested-RH 120-Window Benchmark: Autoresearch Decision Ledger

## Purpose

This document catalogs the decisions that produced the 120-window ERA5 cloud
and outgoing shortwave radiation (RSUT) benchmark. It is written as context for
an autoresearch agent: it records not only the final equation and scores, but
also the upstream data, feature, split, search, calibration, implementation, and
evaluation choices that define what those scores mean.

The benchmark is the end of this dependency chain:

```text
paired ARMBE observations
  -> reconstructed SPEEDY L8 states and one-step diagnostics
  -> nested feature tables and blocked splits
  -> PySR structure search and validation selection
  -> train-only coefficient calibration
  -> online SPEEDY closure implementation
  -> prescribed-state ERA5 cloud and RSUT evaluation
```

### Decision labels

- **Locked**: preserve for an exact comparison with the reported benchmark.
- **Searchable**: an autoresearch loop may vary this, but must create a new run
  identity. Within-run frontier selection can reproduce the locked minimum-
  validation-RMSE rule. Cross-run selection of the implemented equation was
  human judgment, so a future loop must predeclare its own mechanical rule.
- **Inherited**: a software or wrapper default, not a separately justified
  scientific choice. Pin it before claiming strict reproducibility.
- **Limitation**: known uncertainty, approximation, or provenance gap.

## Canonical Artifacts

| Role | Path |
| --- | --- |
| Pooled cache | `outputs/cache_armbe_unified_paired_standard_random_month_blocks/samples.nc` |
| Cache manifest | `outputs/cache_armbe_unified_paired_standard_random_month_blocks/manifest.json` |
| Site terrain map | `outputs/cache_armbe_unified_paired_standard_random_month_blocks/t30_site_terrain.json` |
| Nested feature manifest | `outputs/symbolic_features_unified_t30_nested/manifest.json` |
| Search configuration | `PYSR_SEARCH_CONFIGURATION.md` |
| Implemented equation's original 2,000-row frontier evaluation | `outputs/symbolic_features_unified_t30_nested/group_14_humidity_stability/pysr_richer_80/evaluation.json` |
| Implemented equation's recurring 5,000-row frontier evaluation | `outputs/symbolic_features_unified_t30_nested/group_14_humidity_stability/pysr_richer_80_n5000/evaluation.json` |
| Calibration metrics | `outputs/symbolic_features_unified_t30_nested/group_14_humidity_stability/fast_calibration_all_train/metrics.json` |
| Processed ERA5 RSUT targets | `outputs/era5_rsut_t30_120window_2016_2020.nc` |
| 120-window report | `outputs/era5_nested_rh_diagnostic_120window.json` |
| General experiment history | `UNIFIED_EXPERIMENT.md` |

Primary implementations are `unified_cache.py`,
`export_unified_symbolic_features.py`, `export_nested_symbolic_features.py`,
`evaluate_symbolic_frontier.py`, `calibrate_unified_cloud.py`,
`evaluate_speedy_era5_smoke.py`, `evaluate_speedy_era5_diagnostic.py`, and
`jcm/physics/radiation/speedy_shortwave.py`.

## 1. Scientific Task

### 1.1 Discovery task

**Locked for reproduction:** learn a compact diagnostic mapping from a SPEEDY-
compatible atmospheric state to observed column-total cloud cover.

- Observation target: ARMBECLDRAD `tot_cld`.
- Search loss: unweighted mean squared error over sampled training rows.
- External model-selection metric: unweighted validation RMSE over all
  validation rows.
- The target was not transformed, standardized, clipped, class-balanced, or
  site-weighted.
- One row therefore receives one vote; long site records contribute more than
  short records.

### 1.2 ERA5 evaluation task

**Locked for the 120-window comparison:** diagnose physics independently on
same-time ERA5 atmospheric states and compare:

- total model cloud `cloudc + cloudstr` with ERA5 `total_cloud_cover`; and
- model outgoing TOA shortwave with ERA5 RSUT.

This is not a forecast, hindcast, free run, or stability test. No atmospheric
state is advanced between analyses.

### 1.3 Baseline and candidates

The benchmark evaluates exactly three schemes:

1. `calibrated_speedy`: the native SPEEDY cloud diagnosis with four parameters
   calibrated on the ARMBE training split.
2. `sr_nested_rh`: the selected uncalibrated symbolic structure.
3. `sr_nested_rh_calibrated`: the same symbolic structure with three
   train-fitted coefficients.

The older `sr_total_cloudc` closure exists in the code but is not part of this
benchmark.

## 2. ARMBE Dataset Decisions

### 2.1 Source and pairing

**Locked:**

- ARM archive order: `267892`.
- Resolved local root:
  `/data/MOSAIC/jax-gcm/experiments/armbe_sgp/data/order-267892/ftp.archive.arm.gov/fisherm1/267892`.
- Products: standard-resolution ARMBEATM atmospheric state and ARMBECLDRAD
  cloud/radiation data.
- Only `*.c1` datastream directories were considered.
- ARMBEATM directories containing `hires` were excluded.
- Files were paired by the `.<YYYYMMDD>.` date in their names.
- A date was retained only when both ATM and CLDRAD files existed.
- When `.nc` and `.cdf` represented the same start date, `.nc` was preferred;
  filename was the secondary ordering key.
- Paired products were merged with an inner join on time.
- The primary `time` coordinate was decoded explicitly; auxiliary `time_frac`
  was not trusted as CF time.
- Input time had to be one-dimensional, nonempty, increasing, and duplicate-free.

### 2.2 Row and target filtering

**Locked:** one sample is one `(timestamp, site_facility)` pair. After file-level
validation and state reconstruction, a timestamp was retained only when:

- surface pressure was finite;
- every interpolated SPEEDY-level temperature and humidity value was finite;
- `tot_cld` was finite; and
- `qc_tot_cld == 0`.

Missing interpolated winds did not reject a timestamp because they were replaced
with zero.

The resulting dataset contains 79,409 rows from 15 site-facilities:
`anxM1`, `awrM1`, `corM1`, `enaC1`, `epcM1`, `gucM1`, `houM1`, `maoM1`,
`nimM1`, `nsaC1`, `oliM1`, `sgpC1`, `twpC1`, `twpC2`, and `twpC3`.

`tot_cld` was read directly; it was not constructed by summing the vertical
cloud profile. In a representative standard ENA file its metadata describes it
as a narrow-field-of-view, radar/MPL hourly total cloud fraction with dimensions
`(time,)` and units 1. The same file's `cld_frac` is a distinct radar/MPL hourly
layer cloud fraction with dimensions `(time,height)` and units percent. Both are
derived from the underlying 10-second ARSCL observations, but they summarize
different events: cloud occurrence anywhere in a column versus occurrence in a
particular height layer. A vertical sum double-counts multilayer and vertically
extended cloud and is not a cloud fraction. For ENA 2018, its mean is 41.4 and
its maximum is 384 after conversion of layer percentages to fractions, whereas
mean `tot_cld` is 0.730. Even the vertical maximum is not equivalent to total
cover because clouds can occur at different heights in different sub-hourly
profiles. The file metadata does not state the exact temporal/column aggregation
algorithm, so that must be verified in ARSCL/ARMBE product documentation before
constructing new component targets.

Profile/file validation also required:

- strictly monotonic finite pressure coordinates with at least two levels;
- pressure coordinates in `1..1100 hPa`;
- finite surface pressure in `300..1100 hPa`;
- physically screened temperature and moisture ranges.

The cache manifest records excluded files and reasons.

### 2.3 ARMBE state reconstruction

**Locked:** profiles were mapped to the exact SPEEDY L8 full-level sigma centers:

```text
0.025, 0.095, 0.20, 0.34, 0.51, 0.685, 0.835, 0.95
```

The half-level boundaries are:

```text
0, 0.05, 0.14, 0.26, 0.42, 0.60, 0.77, 0.90, 1.0
```

Choices:

- Use physical sigma `fsg`, not logarithmic sigma.
- Convert source pressure to observed `p / surface_pressure`.
- Drop invalid profile levels and require at least two finite source points.
- Sort by sigma and use linear `np.interp`.
- Edge-clamp outside the observed pressure range rather than extrapolating.
- Interpret temperature as Celsius and add `273.15` when its maximum is below
  `100`; enforce the global `150..350 K` screen.
- Moisture source priority: specific humidity, then dew point, then RH.
- Convert source `g kg-1` specific humidity to `kg kg-1` when needed for
  processing, then export SPEEDY humidity in `g kg-1` and floor at zero.
- Dew-point conversion uses Magnus/Tetens vapor pressure and `epsilon=0.622`.
- RH-derived moisture clips source RH to `0..100%`.
- Replace remaining missing winds with zero.
- Reconstruct geopotential rather than using an observed profile:

```text
geopotential(sigma) = -rd * mean(column_temperature) * log(sigma)
```

- Normalize surface pressure by `p0 = 100000 Pa`.
- Prefer observed surface temperature; fill per-file missing values with the
  file mean; use `295 K` only if the variable is absent.

**Limitation:** edge clamping, zero-filled winds, and reconstructed geopotential
are modeling choices, not direct observations.

### 2.4 Site terrain and one-step diagnostics

**Locked:**

- Convert station longitude to `0..360 degrees`.
- Select the nearest cell from `jcm/data/bc/t30/clim/terrain.nc`.
- Use that cell's orography and fractional land mask, not station elevation or
  local land type.
- Set global `lfluxland=true`.
- Diagnose every row independently for one `1800 s` SPEEDY step.
- Start every row with fresh tracers and physics carry.
- Calendar: Gregorian.
- Export batch size: 64, processed as site-specific chunks. This is part of the
  scientific/reproducibility configuration because each chunk uses its own mean
  observed surface temperature as SST. Batch size, row order, and chunk
  boundaries can therefore change exported diagnostics.

Offline forcing used by these one-step feature diagnostics:

| Quantity | Choice |
| --- | ---: |
| Albedo | 0.20 |
| Soil water | 0.30 |
| Snow | 0 |
| Sea ice | 0 |
| CO2 | 407 |
| Surface temperature | same-time observed/fallback value |
| SST | mean observed surface temperature within each site-specific export chunk of at most 64 rows |

**Limitation:** coastal and island sites can map to ocean or fractional native
T30 cells. This was deliberate for model compatibility.

## 3. Feature Decisions

### 3.1 General feature policy

**Locked:**

- PySR received raw feature columns in their documented units.
- There was no train-fitted scaling, normalization, imputation, dimensional
  constraint, feature preselection, or learned embedding.
- All 13 newly derived columns had to be finite for every exported row.
- The three groups were strictly nested and used identical rows and split labels.

### 3.2 Five archived baseline features

| Feature | Definition |
| --- | --- |
| `rh_cloudc_max` | Maximum diagnosed RH at PBL-top sigma 0.835 and reference sigmas 0.34, 0.51, 0.685 |
| `precip_mm_day` | `86.4 * (precnv + precls)` |
| `gse` | SPEEDY gross-stability diagnostic exported directly |
| `rh_lowest` | RH at the lowest SPEEDY level |
| `fmask` | nearest-T30 fractional land mask for the site |

These archived columns were preserved exactly. Later RH clipping did not alter
them.

### 3.3 Added RH construction

Added RH features were recomputed from the archived L8 state using SPEEDY's
warm/cold Tetens saturation formulation. Diagnostic RH was clipped to `[0,1.2]`
before interpolation. The upper bound prevents a small number of finite,
edge-clamped cold/low-pressure levels from producing RH of order `100..10000`.

Profiles were sorted by physical sigma and linearly sampled at:

```text
0.20, 0.34, 0.51, 0.685, 0.835, 0.95
```

### 3.4 Fourteen-feature humidity/stability group

This group contained the five baseline features plus:

| Feature | Definition | Units |
| --- | --- | --- |
| `rh_cloudc_mean` | mean RH at sigma 0.34, 0.51, 0.685, 0.835 | 1 |
| `rh_low_mean` | mean RH at sigma 0.685, 0.835, 0.95 | 1 |
| `rh_mid_mean` | mean RH at sigma 0.34, 0.51 | 1 |
| `rh_high_mean` | mean RH at sigma 0.20, 0.34 | 1 |
| `rh_low_mid_gradient` | `rh_low_mean - rh_mid_mean` | 1 |
| `rh_vertical_range` | max minus min RH over all six fixed sigmas | 1 |
| `low_level_stability` | potential temperature at sigma 0.685 minus surface potential temperature | K |
| `maximum_inversion_strength` | largest positive adjacent upward potential-temperature increase over sigma 0.51, 0.685, 0.835, 0.95 | K |
| `low_level_lapse_rate` | `(T_0.95 - T_0.835) / (z_0.835 - z_0.95 in km)` | K km-1 |

### 3.5 Eighteen-feature group

The 18-feature group added four more variables to the 14-feature group:

| Feature | Definition | Units |
| --- | --- | --- |
| `column_saturation_deficit` | sigma-layer-thickness-weighted mean of `max(1-RH,0)` over native levels at sigma >= 0.2 | 1 |
| `precipitable_water` | pressure-integrated specific humidity | kg m-2 |
| `low_level_wind_speed` | mean wind speed at sigma 0.685, 0.835, 0.95 | m s-1 |
| `boundary_layer_wind_shear` | vector wind difference between sigma 0.685 and 0.95 | m s-1 |

### 3.6 Features actually used by the selected equation

The selected structure came from the 14-feature search but uses only four of the
available variables:

```text
rh_low_mean
rh_mid_mean
rh_high_mean
rh_vertical_range
```

This distinction matters: the search was allowed to use stability, baseline,
and other RH features, but its validation-selected expression discarded them.

## 4. Split and Leakage Decisions

### 4.1 Primary split

**Locked:**

- Split unit: whole `(site_facility, year-month)` blocks.
- Nominal fractions: 70% train, 20% validation, 10% test.
- Base seed: `20260731`.
- Per-site seed: `20260731 + crc32(site_facility)`.
- Split separately within each site.
- Stratify by calendar month.
- Weight assignment targets by the number of QC-valid samples in each block.
- Greedily assign randomized blocks to minimize squared deviation from target
  weighted counts; break exact ties with the same site RNG.
- Repair missing split labels when enough blocks exist.
- Singleton calendar-month strata use random assignment with probabilities
  `0.7/0.2/0.1`.

Realized counts:

| Split | Rows | Fraction |
| --- | ---: | ---: |
| Train | 54,229 | 68.29% |
| Validation | 16,267 | 20.49% |
| Test | 8,913 | 11.22% |

### 4.2 Leakage controls

- No site-month block crosses splits.
- Splits were created before feature export.
- Baseline and nested exporters checked row/time alignment and copied labels.
- PySR saw only sampled training rows.
- Every saved frontier candidate was evaluated on the full validation split.
- Within a run, test was evaluated only for the validation-selected equation.
- Coefficient calibration used all training rows and no validation/test rows.

### 4.3 Residual generalization risks

**Limitations:**

- Sites occur in all splits; this is not leave-one-site-out evaluation.
- Different years of the same site/calendar month can occur in different splits.
- `fmask` can partially encode site identity.
- No embargo prevents weather events crossing month boundaries.
- The same validation split was reused across 12 nested searches and earlier
  exploratory searches, creating cross-run meta-overfitting risk.
- Test results were repeatedly inspected during exploration. The test split is
  no longer pristine for future adaptive research decisions.
- A chronological per-site cache exists but was not used for this reported
  search.

An autoresearch loop should not treat another improvement on these reused
validation/test sets as confirmatory without a new outer holdout protocol.

## 5. PySR Search Decisions

### 5.1 Search matrix

Twelve searches covered the Cartesian product of:

- feature groups: 5, 14, and 18 features;
- training samples: 2,000 and 5,000 rows; and
- limits: `maxsize/maxdepth = 20/5` and `50/8`.

There was one evolutionary run per matrix cell. These were controlled
configuration comparisons, not multiple independent random-seed replicates.

### 5.2 Training-row sampling

**Locked:**

```text
np.random.default_rng(20260731).choice(
    n_train, size=sample_size, replace=False
)
```

- Sampling happened once before `model.fit`.
- All populations and coefficient fits within a run saw the same selected rows.
- PySR batching was disabled.
- Identical table order and seed gave the same draw across feature groups and
  complexity settings.
- With the installed NumPy behavior, the 5,000-row draw contains all 2,000 rows
  in the smaller draw plus 3,000 additional rows.

### 5.3 Experiment-selected settings

| Setting | Value |
| --- | --- |
| PySR version | 1.5.10 |
| Random state | `20260731` |
| Iterations | 80 |
| Populations | 12 |
| Population size | 33 |
| Cycles per iteration | 200 |
| Binary operators | `+`, `-`, `*`, `/` |
| Unary operators | `square`, `cube`, `sqrt_abs`, `log_abs`, `exp`, `tanh`, `relu` |
| Operator costs | `/=3`; `square=3`; `cube=3`; `sqrt_abs=3`; `log_abs=3`; `exp=5`; `tanh=3`; `relu=3`; all others 1 |
| Nested constraints | no `exp` in `exp`; no `log_abs` in `log_abs`; no `sqrt_abs` in `sqrt_abs` |
| Original limits | `maxsize=20`, `maxdepth=5` |
| Expanded limits | `maxsize=50`, `maxdepth=8` |

There was no template, feature preselection, dimensional constraint, general
argument constraint, size warmup, timeout, evaluation cap, or early-stop rule.

### 5.4 Settings passed by the unversioned AgentSR wrapper

These were passed explicitly by the wrapper but were not separately selected
for this scientific problem:

| Setting | Value |
| --- | --- |
| Constant optimization | enabled |
| Optimization probability | 0.14 |
| Optimizer | BFGS with backtracking |
| Optimizer restarts | 2 |
| Optimizer iterations | 8 |
| Optimizer function-call limit | backend default 10,000 |
| Warm start | false |
| Denoising | false |
| Parallelism | multithreading |

**Limitation:** the AgentSR package version, source revision, wrapper path, and
original command are not recorded with the copied search artifacts. Serialized
effective PySR settings are available, but wrapper behavior beyond them is not
independently reproducible.

### 5.5 Inherited PySR 1.5.10 settings

**Inherited and therefore version-sensitive:**

| Area | Effective value |
| --- | --- |
| Objective | unweighted MSE, `L2DistLoss` |
| Loss scaling | logarithmic |
| Explicit parsimony | 0.0 |
| Adaptive complexity | enabled, scaling 1040.0 |
| Tournament | 15 candidates, rank probability 0.982 |
| Survival | accepted offspring replace oldest population member |
| Crossover probability | 0.0259 |
| Simulated annealing | disabled |
| Migration | population and hall-of-fame migration enabled |
| Migration fractions | 0.00036 population, 0.0614 hall of fame |
| Migration candidates | top 12 |
| Simplification | enabled; failed mutations skipped |
| Constant perturbation/negation | 0.129 / 0.00743 |
| Numeric precision | 32 bit |
| Fast/turbo/bumper modes | disabled |
| Deterministic mode | false |
| PySR model selection | `best`, not used for scientific selection |

Mutation weights were: rotate tree 4.26, add node 2.47, delete node 0.870,
mutate operator 0.293, do nothing 0.273, swap operands 0.198, mutate constant
0.0346, insert node 0.0112, simplify 0.00209, randomize 0.000502, and optimize
constants as a mutation 0.0.

**Limitation:** fixed `random_state` reproduces the sampled rows but does not
guarantee a bitwise-identical evolutionary trajectory because multithreading was
enabled and deterministic mode was false.

Recorded environment: Python 3.11.14, PySR 1.5.10, Julia 1.12.6,
SymbolicRegression.jl 1.11.3, and h5py 3.16.0. NumPy, hardware, thread count,
BLAS, and exact original command lines are not recorded in the copied search
artifacts.

### 5.6 Frontier evaluation and selection

**Locked procedure:**

1. Parse every equation in `result.json["pareto_frontier"]` with SymPy.
2. Evaluate it on all validation rows.
3. Reject nonfinite outputs or incorrect output shape.
4. Compute RMSE, MAE, prediction-minus-target bias, Pearson correlation, and R2.
5. Select minimum validation RMSE, without a complexity penalty.
6. Evaluate only that selected equation on test.

No prediction clipping was used. PySR score, training loss, adaptive complexity
cost, and AgentSR `best_equation` did not choose the scientific result. Exact
validation-RMSE ties would resolve to the first frontier candidate encountered;
there was no additional tie rule.

### 5.7 Search outcomes and human choice

The complete controlled comparison was:

| Train rows | Limits | Group | Selected complexity | Validation RMSE | Test RMSE |
| ---: | --- | --- | ---: | ---: | ---: |
| 2,000 | 20/5 | 5-feature baseline | 15 | 0.31721 | 0.32974 |
| 2,000 | 20/5 | 14-feature humidity/stability | 13 | 0.29895 | 0.31109 |
| 2,000 | 20/5 | 18-feature moisture/wind | 18 | 0.29901 | 0.31103 |
| 2,000 | 50/8 | 5-feature baseline | 37 | 0.31440 | 0.32613 |
| 2,000 | 50/8 | 14-feature humidity/stability | 23 | 0.29537 | 0.31056 |
| 2,000 | 50/8 | 18-feature moisture/wind | 49 | 0.29547 | 0.30925 |
| 5,000 | 20/5 | 5-feature baseline | 19 | 0.31609 | 0.32656 |
| 5,000 | 20/5 | 14-feature humidity/stability | 13 | 0.29895 | 0.31109 |
| 5,000 | 20/5 | 18-feature moisture/wind | 15 | 0.30047 | 0.31200 |
| 5,000 | 50/8 | 5-feature baseline | 46 | 0.31317 | 0.32441 |
| 5,000 | 50/8 | 14-feature humidity/stability | 37 | 0.29630 | 0.30744 |
| 5,000 | 50/8 | 18-feature moisture/wind | 36 | **0.29407** | **0.30716** |

The 5,000-row expanded 18-feature run achieved the lowest later validation RMSE
but selected a complexity-36 expression with nested powers and division by
precipitable water. Expanded searches repeatedly reused validation and were
treated as exploratory. Test values in this table are reports, not selection
criteria.

The implemented structure was a human scientific choice favoring compactness,
interpretability, numerical safety, and recurrence across the 2,000- and
5,000-row 20/5 searches:

```text
tanh(
  (rh_low_mean + rh_mid_mean)
  * (rh_vertical_range^3 + rh_high_mean)
)
```

PySR complexity: 13. Validation RMSE: 0.29895144. Test RMSE: 0.31109056.

**Limitation:** there was no predeclared scalar objective combining validation
error, complexity, numerical safety, physical plausibility, and recurrence
across searches. The final preference is documented scientific judgment, not a
mechanically encoded winner.

## 6. Calibration Decisions

### 6.1 Symbolic coefficient parameterization

The structure was fixed and expanded into two identifiable basis terms:

```text
tanh(
  a_vertical * (rh_low_mean + rh_mid_mean) * rh_vertical_range^3
  + a_high * (rh_low_mean + rh_mid_mean) * rh_high_mean
  + bias
)
```

This avoids a non-identifiable product of separate scale factors on both sides
of the original multiplication.

### 6.2 Symbolic optimizer

**Locked:**

| Choice | Value |
| --- | --- |
| Fit rows | all 54,229 training rows |
| Weighting/batching | none; dense full-training objective |
| Objective | MSE |
| Gradient | dense JAX gradient of JIT-compiled objective |
| Optimizer | SciPy L-BFGS-B |
| Initial parameters | `(1, 1, 0)` |
| Bounds | `a_vertical=[0,5]`, `a_high=[0,5]`, `bias=[-5,5]` |
| Maximum iterations | 1,000 |
| `ftol` | `1e-12` |
| `gtol` | `1e-8` |

The artifact does not record JAX version, backend, `jax_enable_x64`, or objective
dtype. Saved predictions indicate float32 output while SciPy reports parameters
at double-precision display. These details must be pinned for a strict rerun.

The fit converged in 9 iterations and 28 function evaluations to:

```text
a_vertical = 1.0803934227259704
a_high     = 0.9798508872916644
bias       = -0.0008863295063258815
```

Calibration changed validation RMSE from 0.29895144 to 0.29874928 and test RMSE
from 0.31109056 to 0.31122764. It did not demonstrate improved test
generalization.

### 6.3 Calibrated SPEEDY baseline

The baseline fit varied four native cloud-diagnosis parameters using the same
training-only L-BFGS-B setup:

```text
R = clip((rh_cloudc_max - rhcl1) / (1 - rhcl1), 0, 1)^2
P = min(10, precip_mm_day)
cloudc = min(1, wpcl * sqrt(max(1e-9, P)) + R)
S = clip((gse - 0.25) / 0.15, 0, 1)
cloudstr_sea = S * max(clsmax - 1.2 * cloudc, 0)
cloudstr_land = max(cloudstr_sea, clsminl) * rh_lowest
cloudstr = min(1, cloudstr_sea + fmask * (cloudstr_land - cloudstr_sea))
target_prediction = cloudc + cloudstr
```

The archived comparison used the literal sum with no overlap rule or final clip
of `cloudc + cloudstr`.

| Parameter | Initial | Bounds | Fitted |
| --- | ---: | ---: | ---: |
| `rhcl1` | 0.30 | [0.10, 0.60] | 0.32162740151353536 |
| `wpcl` | 0.20 | [0.05, 0.60] | 0.05 |
| `clsmax` | 0.60 | [0.30, 0.90] | 0.6399201885756207 |
| `clsminl` | 0.15 | [0, 0.40] | 0.0 |

The implementation verified that initial parameters reproduced the archived
literal `cloudc + cloudstr` formula within `rtol=atol=2e-6`. `wpcl` and
`clsminl` reached lower bounds. The fit converged in 6 iterations and 19
evaluations.

## 7. Online SPEEDY Implementation Choices

### 7.1 Feature parity

**Locked:** online RH is clipped to `[0,1.2]`, sampled at sigma 0.20, 0.34,
0.51, 0.685, 0.835, and 0.95, and reduced to the same low/mid/high means and
vertical range. On L8 these six target sigmas are exact model centers.

The implementation supports other vertical resolutions with linear sigma
interpolation. Tests compare L8, L16, and L24 smooth-profile output within an
absolute tolerance of 0.05.

### 7.2 Output handling

- Both nested closures are one-component total-cloud closures.
- Both set `cloudstr = 0`.
- Online total cloud is clipped to `[0,1]`.
- A NaN cloud prediction is mapped to 1 before clipping.
- Offline PySR evaluation and calibration did not apply this final explicit
  clipping, although `tanh` upper-bounds positive outputs.

### 7.3 Radiation retained from SPEEDY

The symbolic equation changes cloud amount only. SPEEDY still controls:

- cloud-top diagnosis, including its convective-top and humidity/`qacl` gates;
  these gates do not gate the nested symbolic cloud amount itself;
- cloud optical properties and cloud albedo;
- water-vapor, aerosol, and dry-air absorption;
- surface albedo; and
- the two-band shortwave solver.

Nested closures use default SPEEDY parameters rather than the four calibrated
baseline cloud-amount parameters. For nested cloud amount, `wpcl`, `pmaxcl`,
`clsmax`, `clsminl`, and stratiform albedo are bypassed or inactive because
`cloudstr=0`. Default `rhcl1=0.30`, `qacl=0.20 g kg-1`, and convective `iptop`
remain active in cloud-top diagnosis. Cloud albedo `albcl=0.43` and the remaining
shortwave optical and absorption parameters remain active.

This is why improved total-cloud RMSE does not necessarily improve RSUT.

## 8. ERA5 120-Window Evaluation Choices

### 8.1 Date sampling and independence

**Locked:**

- Years: 2016 through 2020 inclusive.
- Dates: day 7 and day 21 of every month.
- Daily windows: 120.
- Analysis times per day: 00, 06, 12, and 18 UTC.
- Prescribed atmospheric states: 480.
- Calendar: Gregorian.
- Model physics metadata timestep: 1800 s.
- Bootstrap unit: one daily window, treated as exchangeable by the implemented
  i.i.d. bootstrap.

The dates are fixed systematic strata, not random date draws. Grid cells and the
four times within a day are not treated as independent bootstrap samples.

### 8.2 Model grid

- Coordinates: `get_speedy_coords(layers=8, spectral_truncation=31)`.
- Horizontal grid: 96 longitudes by 48 Gaussian latitudes.
- Longitude spacing: 3.75 degrees, from 0 through 356.25 degrees.
- Vertical grid: SPEEDY L8 sigma centers listed above.
- Packaged boundary files are named T30 even though model coordinates are often
  described as T31 spectral truncation.

### 8.3 ERA5 stores and variables

Atmospheric state, local six-hourly 240x121 store:

```text
/public/wb2/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr
```

Variables: temperature, specific humidity, zonal wind, meridional wind,
geopotential, and surface pressure.

Cloud target, local six-hourly 64x32 store:

```text
/public/wb2/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr
```

Variable: `total_cloud_cover`. The 240x121 state store did not provide usable
finite cloud cover.

RSUT source, public hourly 0.25-degree store accessed anonymously with `gcsfs`:

```text
gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr
```

The local coarse TOA shortwave fields were all NaN, so RSUT was remotely read
and reduced to the small local T30 cache listed under canonical artifacts.

### 8.4 ERA5 atmospheric remapping

For each of 480 states:

1. Sort source latitude ascending.
2. Map longitude modulo 360 and sort.
3. Linearly interpolate fields and surface pressure horizontally to 96x48.
4. Compute column-dependent target pressure `sigma * surface_pressure`.
5. Clamp target pressure to the available ERA5 pressure-level range.
6. Linearly interpolate each pressure-level field vertically.
7. Convert specific humidity from `kg kg-1` to `g kg-1` and floor at zero.
8. Normalize surface pressure by 100,000 Pa.

Horizontal interpolation precedes vertical interpolation. There is no balance
adjustment or terrain/geopotential reconciliation. The source begins at 50 hPa,
so SPEEDY's approximately 25 hPa top level is edge-clamped.

### 8.5 Cloud remapping

- Select same-time ERA5 `total_cloud_cover` at all four analyses.
- Sort latitude and modulo/sort longitude.
- Add periodic copies of the last source longitude at `lon-360` and the first at
  `lon+360`.
- Linearly interpolate to the model grid.

The periodic extension is necessary because plain xarray interpolation treated
the model's 356.25-degree longitude as outside the coarse source ending at
354.375 degrees. The corrected 120-window run has every cloud target finite.

### 8.6 RSUT definition

ERA5 uses downward-positive net TOA shortwave. The target was therefore:

```text
RSUT = mean_top_downward_short_wave_radiation_flux
       - mean_top_net_short_wave_radiation_flux
```

For each day, this was computed at 00/06/12/18 UTC, averaged over those four
times, then linearly interpolated to the model grid.

Model RSUT was:

```text
shortwave_rad.fsol - shortwave_rad.ftop
```

and was likewise averaged over four independently prescribed states. SPEEDY
uses daily-mean insolation for the date, not an instantaneous diurnal solar
zenith angle.

### 8.7 Terrain and surface forcing

Terrain:

```text
jcm/data/bc/t30/clim/terrain.nc
```

Forcing:

```text
jcm/data/bc/t30/clim/forcing.nc
```

The forcing file provides a 12-month 1981 climatology of land temperature, SST,
sea ice, snow, soil water, and annual-mean bare-land albedo. Monthly fields are
padded across year boundaries and linearly interpolated to 365 daily values.
Wrap-year indexing is used. In leap years, the 365-value climatology is indexed
by Gregorian fraction of year rather than exact 1981 month/day, so bin boundaries
are fractionally shifted. These are not same-date ERA5 surface conditions.

### 8.8 Metrics

For finite prediction-target pairs, per-window area-weighted metrics are:

```text
RMSE = sqrt(sum(w * (prediction-target)^2) / sum(w))
bias = sum(w * (prediction-target)) / sum(w)
```

Weights are SPEEDY Gaussian quadrature weights, broadcast over longitude, time,
and leading dimensions.

- Cloud metrics pool four analyses and all grid cells within each day.
- RSUT metrics use one four-time daily-mean field per day.
- Reported mean metrics are arithmetic means of 120 per-window metrics, not one
  globally pooled RMSE.
- Distribution means/standard deviations pool values with area weights.
- Percentage RMSE divides mean window RMSE by either the ERA5 weighted mean or
  ERA5 weighted standard deviation. It is not MAPE.

Evaluation points per scheme:

| Quantity | Count |
| --- | ---: |
| Independently initialized prescribed-state daily windows | 120 |
| Prescribed states | 480 |
| Finite cloud grid comparisons | 2,211,840 |
| Finite daily RSUT grid comparisons | 552,960 |

### 8.9 Bootstrap

The artifact's intervals reproduce the script defaults:

- draws: 10,000;
- paired resampling of whole daily windows with replacement;
- statistic: mean candidate-minus-calibrated-SPEEDY per-window RMSE difference;
- interval: percentile 2.5th and 97.5th quantiles;
- base seed: `20260731`.

Derived seeds are `20260741/20260742` for uncalibrated nested cloud/RSUT and
`20260751/20260752` for calibrated nested cloud/RSUT. Only RMSE differences have
bootstrap intervals; bias differences do not.

**Limitation:** these are empirical i.i.d.-window bootstrap intervals for the
fixed 120-date set, not design-based or temporal-dependence-adjusted confidence
intervals. Resampling does not preserve month/year strata or use multi-window
temporal blocks.

**Provenance limitation:** the original 120-window JSON did not serialize the
bootstrap draw count, seed, command line, software versions, git revision, or
timings. These values are reconstructed from the script defaults and reproduce
the saved intervals. Later script versions serialize stage timings.

## 9. Reported 120-Window Results

### 9.1 Mean errors

| Scheme | Cloud RMSE | Cloud bias | RSUT RMSE (W m-2) | RSUT bias (W m-2) |
| --- | ---: | ---: | ---: | ---: |
| Calibrated SPEEDY | 0.21112008 | 0.03131538 | **40.73818** | 10.23317 |
| Nested RH | 0.19035075 | **0.02856295** | 43.06745 | **8.72488** |
| Calibrated nested RH | **0.19023242** | 0.03479891 | 43.12679 | 9.36521 |

### 9.2 Paired differences from calibrated SPEEDY

| Candidate | Metric | Mean difference | Paired 95% interval |
| --- | --- | ---: | ---: |
| Nested RH | cloud RMSE | -0.02076933 | [-0.02228269, -0.01927846] |
| Calibrated nested RH | cloud RMSE | -0.02088766 | [-0.02232707, -0.01952214] |
| Nested RH | RSUT RMSE | +2.32927 W m-2 | [+2.13939, +2.51870] |
| Calibrated nested RH | RSUT RMSE | +2.38861 W m-2 | [+2.20269, +2.57500] |

Both nested closures improve cloud amount and worsen RSUT spatial RMSE. The
calibrated symbolic variant is not materially better than the uncalibrated one.

### 9.3 Post-hoc fixed-total partition counterfactual

A follow-up diagnostic used 240 dates (days 7 and 21 of each month in
2011-2020). SPEEDY does not natively partition a separately diagnosed total: it
diagnoses generic RH/precipitation cloud `cloudc` and stability-driven PBL
stratocumulus `cloudstr` separately. The counterfactual derived the local ratio
`r = cloudstr/(cloudc+cloudstr)` from calibrated SPEEDY, then replaced nested
RH's one-component cloud by `cloudc=(1-r)C_nested` and
`cloudstr=r*C_nested`. It therefore held nested total cloud and cloud top fixed
while changing only component treatment, then reran SPEEDY shortwave radiation.
Total cloud agreed with the original nested field to `5.96e-8` or better.

| Configuration | Cloud RMSE | RSUT RMSE (W m-2) | RSUT bias (W m-2) |
| --- | ---: | ---: | ---: |
| Calibrated SPEEDY | 0.21074 | **40.55906** | 9.96827 |
| Nested RH, one component | **0.19077** | 43.09910 | **8.64800** |
| Nested RH, SPEEDY-like partition | **0.19077** | 41.57207 | 8.99641 |
| Nested RH, SPEEDY-like partition and SPEEDY cloud top | **0.19077** | 41.57026 | 8.96991 |

The partitioned counterfactual improved RSUT RMSE in all 240 windows. Its mean
difference from one-component nested RH was -1.5270 W m-2, removing 60.1% of the
original nested-versus-baseline degradation. It remained +1.0130 W m-2 worse
than calibrated SPEEDY. This supports component
placement as a major mechanism, not a complete explanation. Adding calibrated
SPEEDY's cloud top after applying the ratio changes RMSE by only -0.0018 W m-2.
The baseline ratio is an empirical model diagnostic, not an observed or
conserved physical partition, and this remains a post-hoc diagnostic. The
artifact is `outputs/era5_partition_counterfactual_240window.json`.

### 9.4 Radiative weighting and cloud-top counterfactuals

The same 240-window diagnostic tested whether ordinary cloud RMSE overstates the
nested closure's improvement because errors occur where they matter less to
shortwave radiation.

| Cloud-amount metric | Calibrated SPEEDY | Nested RH | Nested reduction |
| --- | ---: | ---: | ---: |
| Ordinary cloud RMSE | 0.21074 | 0.19077 | 9.48% |
| Incoming-solar-weighted cloud RMSE | 0.21187 | 0.19000 | 10.32% |
| Common-operator radiative proxy RMSE | 20.9882 W m-2 | 18.9339 W m-2 | 9.79% |

The metrics improved in 238, 239, and 235 of 240 windows, respectively. The
common-operator proxy passes
candidate and ERA5 total cloud through the same nested-state cloud top and
one-component SPEEDY shortwave operator. It therefore includes insolation,
atmospheric transmission, surface albedo, and nonlinear shortwave sensitivity,
but deliberately excludes differences in component placement. The actual RSUT
degradation is consequently not evidence that nested cloud-amount errors are
concentrated in radiatively unimportant locations.

Changing only the nested closure's cloud-top index to calibrated SPEEDY's
diagnosis changed mean RSUT RMSE by +0.0806 W m-2. After applying the SPEEDY-like
component partition, changing cloud top contributed only -0.0018 W m-2. The
nested closure retains default `rhcl1=0.30`, whereas calibrated SPEEDY fitted
`rhcl1=0.3216274`; `rhcl1` remains active in cloud-top diagnosis even though the
nested equation bypasses native cloud-amount diagnosis. This is an unintended
parameter confound, but the full cloud-top substitution shows that it is
negligible for this subset. Component placement is the dominant tested
mechanism; fixed cloud optics and error cancellation remain possible causes of
the residual +1.0130 W m-2 partitioned-versus-baseline gap.

### 9.5 Expanded 240-window confirmation

The same-time benchmark was extended to days 7 and 21 of every month from 2011
through 2020: 240 independent daily windows, 960 prescribed states, 4,423,680
finite cloud comparisons, and 1,105,920 finite daily RSUT comparisons per
scheme.

| Closure | Cloud RMSE | RSUT RMSE (W m-2) | RSUT bias (W m-2) |
| --- | ---: | ---: | ---: |
| Calibrated SPEEDY | 0.21073943 | **40.55906** | 9.96827 |
| Nested RH | 0.19077021 | 43.09910 | **8.64800** |
| Calibrated nested RH | **0.19062888** | 43.15247 | 9.28932 |

For nested RH, the paired cloud-RMSE difference from calibrated SPEEDY is
-0.019969 (95% interval -0.021070 to -0.018883), while the paired RSUT-RMSE
difference is +2.5400 W m-2 (95% interval +2.4064 to +2.6688). This larger
sample confirms rather than weakens the 120-window conclusion. The run took
4472.6 seconds before serialization, dominated by atmospheric-state preparation
(3427.9 seconds) and RSUT-target preparation (975.8 seconds). Artifacts are
`outputs/era5_nested_rh_diagnostic_240window.json` and
`outputs/era5_rsut_t30_240window_2011_2020.nc`.

## 10. Autoresearch Contract

### 10.1 Choices to lock for comparable iterations

Unless a branch explicitly studies one of these choices, preserve:

1. ARMBE source order, standard products, pairing, QC, and row identity.
2. Whole site-month splits and all split labels.
3. Train-only search and coefficient fitting.
4. Within each frontier, full-validation minimum-RMSE selection with no
   prediction clipping. The later cross-run human choice is not a reproducible
   locked rule.
5. No use of test or ERA5 benchmark scores to fit coefficients.
6. SPEEDY-compatible feature definitions, sigma values, units, and RH clipping.
7. The calibrated SPEEDY baseline parameters.
8. ERA5 dates, state/cloud/RSUT stores, remapping, terrain, and forcing.
9. Same-time prescribed-state execution rather than forecast rollout.
10. Gaussian area weighting and daily-window bootstrap unit.
11. Candidate-minus-baseline sign convention.

### 10.2 Searchable axes

An autoresearch loop may explore these, but each change needs an explicit run
identifier and ablation against the locked reference:

- feature subsets or new model-available features;
- operator vocabulary and operator costs;
- dimensional constraints and safe argument constraints;
- training sample size;
- PySR iterations, populations, population size, maxsize, and maxdepth;
- independent seeds and deterministic/serial execution;
- explicit complexity or numerical-safety penalties;
- coefficient parameterization, bounds, and regularization;
- final cloud bounds implemented inside the symbolic structure;
- multi-objective selection using cloud and radiation diagnostics;
- cloud optical properties or cloud-top coupling, if the research question is
  expanded beyond cloud-amount closure.

### 10.3 Do not optimize directly against the reported holdouts

The existing validation, test, and 120-window ERA5 results have been repeatedly
observed. A defensible autonomous loop should create an outer protocol before
large adaptive searches. Options include:

- leave-site-out outer folds;
- later-year temporal holdout;
- disjoint ERA5 development and final date ranges; or
- nested cross-validation by site-month block.

This is a required design decision, not a cosmetic reporting improvement.

### 10.4 Required record for every new iteration

At minimum, save:

- immutable input artifact paths and checksums;
- source revision and dirty-worktree diff identifier;
- Python, NumPy, JAX, PySR, Julia, and SymbolicRegression.jl versions;
- hardware, thread count, and deterministic mode;
- full CLI/configuration, random seeds, and sampled row indices;
- exact feature names, definitions, units, and preprocessing;
- split manifest and counts;
- complete frontier, not only the selected equation;
- validation selection rule fixed before evaluation;
- train/validation/test metrics with clipping policy;
- coefficient calibration objective, bounds, convergence, and fitted values;
- online equation and parity-test result;
- ERA5 dates, stores, cache provenance, forcing, and remapping;
- per-window metrics and paired uncertainty;
- per-stage and total runtime;
- failures, rejected expressions, nonfinite counts, and numerical hazards.

### 10.5 Suggested decision order

The original study did not predeclare a formal multi-objective autoresearch
policy. A future agent should define one before searching. A conservative order
consistent with the scientific judgments made here is:

1. Reject leakage, nonfinite behavior, unit inconsistency, and unavailable
   online features.
2. Select structure using only the designated development protocol.
3. Require improvement to recur across seeds/folds rather than one frontier.
4. Prefer lower complexity and safer algebra when errors are practically tied.
5. Fit coefficients on training data only after structure selection.
6. Require online/offline feature parity and bounded output.
7. Evaluate cloud and RSUT separately; do not assume better cloud implies better
   radiation.
8. Touch the final outer holdout only once per frozen research campaign.

### 10.6 Known unresolved choices

- No formal practical-equivalence threshold defines when RMSE differences are
  too small to justify complexity.
- No site-balanced or climate-regime-balanced objective was tested.
- No independent seed ensemble quantified PySR search instability.
- No clean outer holdout remains under the current repeated-exploration history.
- No contemporaneous ERA5 surface forcing was used for RSUT.
- No forecast stability or coupled climate impact was evaluated.
- The nested closure removes the distinct stratiform cloud component while
  retaining SPEEDY cloud-top and optical assumptions.
- Better cloud amount but worse RSUT indicates the next research target may need
  radiatively relevant cloud placement/optics, not further tuning of total cloud
  amount alone.

## 11. Reproduction Entry Point

With the processed RSUT cache present, the 120-window diagnostic command is:

```bash
JAX_PLATFORMS=cpu /data/MOSAIC/.venv/bin/python \
  experiments/armbe_sgp/evaluate_speedy_era5_diagnostic.py \
  --start-year 2016 \
  --end-year 2020 \
  --state-store \
    /public/wb2/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr \
  --cloud-store \
    /public/wb2/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr \
  --rsut-store \
    gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr \
  --rsut-target-cache \
    experiments/armbe_sgp/outputs/era5_rsut_t30_120window_2016_2020.nc \
  --terrain jcm/data/bc/t30/clim/terrain.nc \
  --forcing jcm/data/bc/t30/clim/forcing.nc \
  --scheme calibrated_speedy \
  --scheme sr_nested_rh \
  --scheme sr_nested_rh_calibrated \
  --bootstrap-draws 10000 \
  --seed 20260731 \
  --output \
    experiments/armbe_sgp/outputs/era5_nested_rh_diagnostic_120window.json
```

The current script defaults to the same numerical choices, but explicit flags
avoid silent changes to defaults or scheme ordering. This command reproduces the
numerical procedure only under the original unrecorded source state; it cannot
reproduce the saved JSON byte-for-byte because the current script adds timing
fields that were absent from the 120-window artifact. A strict future rerun must
pin the environment and record the source revision and worktree diff.
