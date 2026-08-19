# Unified Paired ARMBE Experiment Record

Append material decisions and evaluation results here. This record defines the
pooled multi-site experiment separately from the earlier SGP-only experiment.

## 2026-08-04: Dataset and Split

The observational source is ARM archive order `267892`, restricted to paired,
standard-resolution ARMBEATM and ARMBECLDRAD records. The cache contains one
finite `tot_cld` observation with `qc_tot_cld == 0` for each retained
`(timestamp, site-facility)` pair. It has 79,409 samples across 15 sites.

The primary split assigns whole site-facility year-month blocks randomly with
seed `20260731`, stratified by calendar month and weighted by QC-valid sample
count. The realized pooled counts are 54,229 train (68.29%), 16,267 validation
(20.49%), and 8,913 test (11.22%). A chronological per-site cache is retained
only as a separate temporal-transfer benchmark.

## 2026-08-04: SPEEDY Terrain and Surface Boundary Condition

Use the nearest native T30 Gaussian-grid cell from the packaged climatological
SPEEDY terrain file, `jcm/data/bc/t30/clim/terrain.nc`, for every station.
Longitudes are converted from the ARMBE convention to 0--360 degrees before
nearest-cell selection. `orog_m` and `fmask` are taken directly from that cell.

Set `lfluxland=true`, matching `TerrainData.from_file` for the packaged T30
terrain. This is a global SPEEDY land-flux enable switch, not a field provided
by a particular grid cell. Fractional and ocean cells still use the selected
`fmask` in the surface-flux calculation.

This deliberately represents the native T30 grid-cell surface, not the local
station surface. Consequently, small islands and coastal sites can resolve to
ocean or fractional-land cells. The resulting prescribed values, including the
source coordinates and selected T30 coordinates, are in
`outputs/cache_armbe_unified_paired_standard_random_month_blocks/t30_site_terrain.json`.

## 2026-08-04: T30 SPEEDY Features and Symbolic Regression

Each observation was passed through one independent 1,800-second SPEEDY
diagnostic using the prescribed terrain for its site. Every sample starts from
its observed ARMBE profile with fresh tracer and physics carry. The exported
features and raw baseline are in `outputs/symbolic_features_unified_t30`.

The literal, unclipped `cloudc + cloudstr` T30 SPEEDY baseline has validation
RMSE 0.3943, MAE 0.2774, bias 0.1365, r 0.5258, and R-squared 0.1093. Its test
RMSE is 0.3961. It exceeds one for 39.65% of validation samples.

PySR used the same richer 80-iteration configuration as the SGP search and a
deterministic 2,000-row training subsample with seed `20260731`. Selection was
strictly by validation RMSE across the Pareto frontier. The selected equation
is:

```text
tanh(rh_cloudc_max^4 + sqrt(abs(gse)) * rh_lowest^2)
```

It has validation RMSE 0.3173, MAE 0.2367, bias -0.0020, r 0.6510, and
R-squared 0.4231. Held-out test RMSE is 0.3316, MAE 0.2474, bias -0.0043,
r 0.6038, and R-squared 0.3576. The complexity-16 frontier equation has nearly
identical validation RMSE, 0.3174, so this strict selection should not be read
as strong evidence for the extra complexity.

The complete frontier and split-safe evaluation are in
`outputs/symbolic_features_unified_t30/agentsr_pysr_richer_search/`.

## 2026-08-04: Train-Only Readout Calibration

Calibration used all 54,229 training samples only. Validation and test samples
were never used to fit coefficients. The raw SPEEDY baseline fit the four
physical cloud-diagnosis parameters while holding the observed-profile inputs
and site-specific native-T30 terrain fixed. The exported humidity,
precipitation, stability, and lowest-level RH inputs are upstream of those
parameters in a one-step diagnostic, and the calibration script verifies that
its default formula reproduces the archived SPEEDY raw sum before fitting.

The fitted raw SPEEDY parameters are:

```text
rhcl1   = 0.321627
wpcl    = 0.050000  (lower bound)
clsmax  = 0.639920
clsminl = 0.000000  (lower bound)
```

The calibrated SPEEDY baseline improves validation RMSE from 0.3943 to 0.3300
and test RMSE from 0.3961 to 0.3438. Validation r rises from 0.5258 to 0.6203
and R-squared from 0.1093 to 0.3760. It produces no values above one because
the fitted land stratiform contribution no longer pushes the literal raw sum
over the valid cloud-fraction range.

For the selected symbolic structure, fit only three coefficients on training
data:

```text
tanh(1.100547 * rh_cloudc_max^4
     + 0.673422 * sqrt(abs(gse)) * rh_lowest^2
     + 0.084255)
```

This improves validation RMSE from 0.3173 to 0.3149 and test RMSE from 0.3316
to 0.3265. The validation result is descriptive because the uncalibrated
structure was originally selected using validation RMSE. The test result remains
an untouched evaluation of the subsequently train-fitted coefficients.

The calibration implementation, fitted predictions, and complete metrics are in
`calibrate_unified_cloud.py` and
`outputs/symbolic_features_unified_t30/calibration/`.

## 2026-08-04: Expanded SPEEDY-Recovery Search Space

`fmask`, the native-T30 fractional land mask, is now exported as an SR input.
The next search will also permit binary `min` and `max`, and unary `clip01` and
`sqrt_pos`. The latter two reproduce SPEEDY's interval clipping and its
square-root precipitation floor directly rather than approximating them through
several `relu` operations.

Use complexity costs `min=2`, `max=2`, `clip01=3`, and `sqrt_pos=3`, alongside
the existing richer-search costs. An exact SPEEDY raw-sum recovery tree cannot
share intermediate expressions, so its `cloudc` diagnosis is expanded each
time it appears in the land-sea stratiform branch. Under these costs, a compact
algebraic form of the full native-T30 baseline has complexity 140 and depth 16.
The previous limits, complexity 20 and depth 5, therefore cannot contain that
tree. A recovery search needs at least complexity 140 and maximum depth 16;
some additional slack is appropriate for evolutionary search.

## 2026-08-04: Expanded 80-Iteration Recovery Search

An unrestricted 80-iteration PySR search used the five inputs, the expanded
bounded primitives, `maxsize=160`, and `maxdepth=18`. It used the same
deterministic 2,000-row training subsample and random seed as the preceding
searches. The frontier reached complexity 117, demonstrating that the expanded
limits were active, but it did not recover the nested SPEEDY cloud-diagnosis
structure.

Validation RMSE selected this complexity-33 equation:

```text
min(cube(max(min(0.911816,
                 sqrt_abs(precip_mm_day) * 0.848312 / (precip_mm_day + 0.029452)),
             relu(max(max(rh_lowest, rh_cloudc_max), 0.558736)))),
    0.932292)
```

It has validation RMSE 0.3086, MAE 0.2327, r 0.6747, and R-squared 0.4543.
The held-out test metrics are RMSE 0.3203, MAE 0.2427, r 0.6335, and R-squared
0.4004. It is an empirical alternative, not a SPEEDY recovery: it omits both
`gse` and `fmask` despite those inputs being available. The complete frontier
and split-safe evaluation are in
`outputs/symbolic_features_unified_t30/agentsr_pysr_speedy_recovery_80/`.

## 2026-08-04: Expanded 160-Iteration Recovery Search

Repeating the same unrestricted expanded search for 160 iterations selected a
complexity-63 candidate by validation RMSE. It has validation RMSE 0.3070,
MAE 0.2326, r 0.6801, and R-squared 0.4600. The held-out test metrics are
RMSE 0.3191, MAE 0.2419, r 0.6371, and R-squared 0.4051. This is a small
improvement over the 80-iteration result (0.3086 validation and 0.3203 test
RMSE), at almost twice the selected expression complexity.

The candidate uses `gse`, precipitation, and RH inputs with nested bounded
operators, but not `fmask`; it remains an empirical fit rather than a recovery
of the SPEEDY cloud-diagnosis structure. The complete frontier and evaluation
are in `outputs/symbolic_features_unified_t30/agentsr_pysr_speedy_recovery_160/`.

## 2026-08-04: Expanded 400-Iteration Recovery Search

The 400-iteration version completed in 2 minutes 10 seconds, consistent with
near-linear scaling from the earlier 80- and 160-iteration runs. Validation
selected a complexity-70 candidate with RMSE 0.3056, MAE 0.2317, r 0.6826, and
R-squared 0.4650. Its held-out test RMSE is 0.3182, with MAE 0.2421, r 0.6393,
and R-squared 0.4084.

The marginal improvement over 160 iterations (0.3070 validation and 0.3191
test RMSE) came with a less interpretable nested expression containing
`log_abs(precip_mm_day)`. It is not a SPEEDY recovery. Repeatedly selecting
among searches using validation also means the validation metrics are now
exploratory, rather than an independent model-selection result. The complete
frontier and evaluation are in
`outputs/symbolic_features_unified_t30/agentsr_pysr_speedy_recovery_400/`.

## Deferred Data Source: ARM MICROBASE

Do not download or integrate this product yet. ARM's MICROBASE value-added
product retrieves vertical liquid water content, ice water content, cloud
fraction, and liquid/ice effective radii from cloud radar, lidar/ceilometer,
microwave-radiometer, and sonde measurements. It has production records with
uncertainties at SGP, NSA, and GUC, plus evaluation or legacy records at ANX,
ENA, HOU, MAO, and TWP sites relevant to this experiment. See
https://www.arm.gov/capabilities/vaps/microbase.

MICROBASE would enable a future reduced ECHAM cloud-closure experiment after
careful time/vertical co-location with raw ARMBE profiles. Its liquid/ice water
content must first be checked for grid-mean versus in-cloud convention before
conversion to ECHAM mass mixing ratios. It does not provide the full ECHAM 2M
particle-number and precipitating-hydrometeor state.

## 2026-08-05: MICROBASE Archive Inventory and SGP Probe

ARM Live Data queries are authenticated and working. Candidate datastreams were
queried over the overlapping ARMBE date ranges. The available daily-file counts
are 183 (ANX), 1,888 (ENA), 365 (EPC), 640 (GUC), 365 (HOU), 602 (MAO), 2,257
(NSA production), 4,677 (SGP production), 3,291 (TWP C1 PI2), 1,308 (TWP C2
PI2), and 1,606 (TWP C3 PI2). Historical PI2 coverage adds 3,196 NSA and
4,696 SGP daily files. This is 25,074 files before exact timestamp-level
collocation, or roughly 10 TB if downloaded unfiltered: one representative
production SGP daily file is 670,485,004 bytes and one legacy SGP PI2 file is
159,539,004 bytes. File size varies by product/version, so this is a planning
estimate rather than a storage commitment.

The production SGP probe, `sgpmicrobaseC1.c1.20180601.000000.nc`, has 21,600
four-second profiles on 596 30-m height cells. It provides liquid and ice water
concentration (`g m-3`), effective radii, random-uncertainty fields, per-field
QC flags, retrieval flags, cloud/clear and precipitation flags, and the MWR
scale factor. It does not contain an explicit layer cloud-fraction variable.
The legacy PI2 SGP probe has 8,640 profiles on 512 heights (10-second cadence),
the same liquid/ice concentration and QC/retrieval fields, and no uncertainty
fields.

The existing 2018 SGP ARMBECLDRAD record supplies the missing target:
`cld_frac` is an hourly percent cloud fraction with `qc_cld_frac`, dimensions
`(time, height)`, and exactly the same 596-height coordinate as the production
MICROBASE probe. This supports direct height alignment for the modern SGP path.
MICROBASE concentration must still be converted to mixing ratio using collocated
air density, and its grid-mean versus in-cloud convention remains to be verified
from product documentation before it is used as `qc`/`qi`.

## 2026-08-07: Layer-Wise Non-Condensate Symbolic Regression Prototype

The high-resolution ARMBEATM/ARMBECLDRAD pairs for ENA, NSA, and SGP in 2023--24
make a separate layer-cloud experiment possible without MICROBASE. The target is
QC-valid `cld_frac / 100` on the CLDRAD 30-m height grid. ATM height-grid fields
are interpolated to that grid, then cubic-spline first and second vertical
derivatives are evaluated. Only five of Grundner's seven base state features are
available: wind speed, specific humidity derived from dew point, temperature,
pressure, and relative humidity. Their ten derivatives plus height MSL, T30
`fmask`, and surface pressure form 18 standardized train-only inputs. The missing
six features are `qc`, `qi`, and their first/second vertical derivatives.

Rows are sampled at the five-minute centers of 00, 06, 12, and 18 UTC averages,
then split in whole site year-month blocks. The cache has 555,072 finite sampled
layer rows (371,392 train, 160,704 validation, 22,976 test); the search uses a
deterministic 5,000-row train draw. This experiment is distinct from the earlier
column-total cache and is limited to the three high-resolution sites/two years.

An exploratory five-minute PySR run used the Grundner operator-cost groups, a
maximum complexity of 90, population size 20, prediction-clipped squared loss,
and a nominal one-million-iteration limit stopped by a 300-second timeout. Raw
Julia `gamma` throws on negative standardized values, so `gamma_safe` returns
gamma only for positive inputs and NaN otherwise. The selected complexity-78
equation has clipped validation RMSE 0.2149 and test RMSE 0.2433. It is not a
candidate parameterization: the short search, high complexity, and validation-to-
test gap make it exploratory. The cache, configuration, and split-safe evaluation
are in `outputs/symbolic_features_layerwise_armbe_18/`.

## 2026-08-11: Five-Feature MLP Capacity Baseline

A small `MLPRegressor` with two 64-unit ReLU hidden layers was fit to the same
five native-T30 SPEEDY predictors and pooled `tot_cld` target as the calibrated
SPEEDY/SR comparison. Inputs use zero-mean, unit-standard-deviation scaling fit
on the 54,229 training rows only. Adam training stopped after 68 epochs using an
internal 15% split of training rows; predictions are clipped to `[0, 1]` for
physical evaluation. The MLP has validation RMSE 0.3098 and test RMSE 0.3190.

This is a small nonlinear capacity baseline, not an estimate of irreducible
predictability: it is one architecture/seed with the same limited five inputs,
and the randomized test split has already been used for exploratory comparisons.
Its test RMSE is lower than the train-fitted capped SR (0.3265) and calibrated
SPEEDY (0.3438), while close to the earlier expanded PySR result (0.3182).

## 2026-08-11: ERA5-Initialized T31L8 Closure Smoke Test

`evaluate_speedy_era5_smoke.py` compares the default SPEEDY closure with the
one-component `sr_total_cloudc` closure from the same ERA5 initial state. It
horizontally remaps WeatherBench2 ERA5 to the T31 grid, then vertically maps
each field to local `sigma * surface_pressure` before creating the `PhysicsState`.
The local store begins at 50 hPa while SPEEDY's top full level is approximately
25 hPa, so only target pressures outside ERA5's available range are clamped to
the nearest source pressure level. This is a finite-run smoke-test convention,
not a balanced analysis or a scientific upper-stratosphere initialization.

For the 2020-01-01 00 UTC initialization and a 6-hour lead, both closures were
finite. Unweighted nodal RMSE against the similarly mapped 06 UTC ERA5 state was
4.9828 K, 0.6331 g kg-1, 10.8068 m s-1, 10.9472 m s-1, and 16584.0 m2 s-2 for
default SPEEDY (temperature, humidity, zonal wind, meridional wind, and
geopotential respectively). The SR closure gave 4.9821 K, 0.6328 g kg-1,
10.8059 m s-1, 10.9463 m s-1, and 16583.3 m2 s-2. This single, short and
unbalanced window is a computational diagnostic only, not evidence of forecast
skill. The output is `outputs/era5_speedy_smoke.json`.

Repeating the same 6-hour diagnostic for 2020-01-01, 2020-04-01, 2020-07-01,
and 2020-10-01 at 00 UTC gave mean default/SR RMSEs of 5.05435/5.05382 K for
temperature, 0.645376/0.645218 g kg-1 for humidity, 10.44848/10.44765 m s-1 for
zonal wind, 10.75925/10.75849 m s-1 for meridional wind, and
16381.31/16380.72 m2 s-2 for geopotential. The SR closure was lower in every
window and metric, but these tiny differences are not statistically assessed and
cannot establish forecast skill under the unbalanced, edge-clamped
initialization. Per-window values and aggregate means are in
`outputs/era5_speedy_multiseason_6h.json`.

## 2026-08-14: Nested Atmospheric-State Feature Search

`export_nested_symbolic_features.py` reuses the archived five-feature table and
derives additional model-compatible summaries from the same eight-level cache.
The three search tables are strictly nested: the five-feature baseline; 14
features after adding vertical RH structure and stability; and 18 after adding
column moisture and winds. Vertical definitions use physical sigma values rather
than level indices. Added RH summaries are clipped to `[0, 1.2]`: a small number
of finite, extrapolated cold/low-pressure cache levels otherwise produce
diagnostic RH values of order 100 to 10,000. This bound does not alter any of the
five archived baseline columns.

Each group used the same deterministic 2,000-row training draw, seed `20260731`,
80 iterations, 12 populations, maximum complexity 20, maximum depth 5, operator
set, and complexity costs as the original richer search. Every Pareto frontier
was evaluated on all validation rows, minimum validation RMSE selected one
equation, and test was evaluated once for that selected equation. The complete
effective search configuration and provenance of each setting are recorded in
`PYSR_SEARCH_CONFIGURATION.md`.

| Group | Features | Validation RMSE | Test RMSE |
| --- | ---: | ---: | ---: |
| Baseline | 5 | 0.31721 | 0.32974 |
| Humidity and stability | 14 | **0.29895** | 0.31109 |
| Moisture and wind | 18 | 0.29901 | **0.31103** |

Validation selected this complexity-13 equation from the 14-feature group:

```text
tanh((rh_low_mean + rh_mid_mean)
     * (rh_vertical_range^3 + rh_high_mean))
```

Its validation MAE is 0.22076, bias 0.00585, correlation 0.69889, and R-squared
0.48797. Test MAE is 0.22865, bias 0.00038, correlation 0.66042, and R-squared
0.43457. The 18-feature group is indistinguishable at this precision and its
selected expression uses column saturation deficit but neither precipitable
water nor wind. Thus the robust gain comes from retaining vertical RH structure;
this search gives no evidence that the four column-moisture/wind additions improve
generalization. The generated tables, manifests, complete frontiers, and
split-safe evaluations are in
`outputs/symbolic_features_unified_t30_nested/`.

### Expanded Complexity 50 / Depth 8 Search

The same three searches were repeated with only `maxsize` increased from 20 to
50 and `maxdepth` from 5 to 8. The same 2,000 training rows, seed, operators,
costs, population settings, and validation-selection procedure were retained.

| Group | Selected complexity | Validation RMSE | Test RMSE |
| --- | ---: | ---: | ---: |
| Baseline | 37 | 0.31440 | 0.32613 |
| Humidity and stability | 23 | **0.29537** | 0.31056 |
| Moisture and wind | 49 | 0.29547 | **0.30925** |

The 14-feature group remains the validation winner. Its selected expression is:

```text
tanh(0.97507286
     * (rh_cloudc_max^9
        + rh_high_mean
        + rh_lowest^3 * rh_vertical_range)
     - 0.076786265)
```

Increasing the limits improves validation RMSE by about 0.0036 relative to the
complexity-13 result, but the added nesting introduces ninth powers and gives
only about 0.0005 lower test RMSE. The 18-feature test score is lower, but test
is not a selection metric and its complexity-49 equation is too opaque to treat
as a physical parameterization. These are exploratory results because the same
validation split has now selected among multiple searches. Complete artifacts
are in each feature group's `pysr_richer_80_c50_d8/` directory.

### Five-Thousand-Row Repeat

Both size/depth configurations were repeated with `sample_size` increased from
2,000 to 5,000 and every other setting unchanged. The seed remained `20260731`;
with the installed NumPy implementation, the fixed 5,000-row draw contains all
2,000 original training rows plus 3,000 additional rows. Full validation again
selected one equation from each frontier before its single test evaluation.

| Limits | Group | Selected complexity | Validation RMSE | Test RMSE |
| --- | --- | ---: | ---: | ---: |
| 20 / 5 | Baseline | 19 | 0.31609 | 0.32656 |
| 20 / 5 | Humidity and stability | 13 | **0.29895** | **0.31109** |
| 20 / 5 | Moisture and wind | 15 | 0.30047 | 0.31200 |
| 50 / 8 | Baseline | 46 | 0.31317 | 0.32441 |
| 50 / 8 | Humidity and stability | 37 | 0.29630 | 0.30744 |
| 50 / 8 | Moisture and wind | 36 | **0.29407** | **0.30716** |

At limits 20/5, validation selects the same complexity-13 14-feature equation as
the 2,000-row search:

```text
tanh((rh_low_mean + rh_mid_mean)
     * (rh_vertical_range^3 + rh_high_mean))
```

At limits 50/8, validation instead selects this 18-feature equation:

```text
cube(tanh(
  rh_low_mean^3
  + rh_mid_mean^3
  + 0.20305507
  + relu(rh_vertical_range^3
         + rh_high_mean
         - rh_low_mean / precipitable_water^2)
))
```

Increasing the sample did not uniformly improve validation performance: results
moved by about 0.0000--0.0015 RMSE relative to their 2,000-row counterparts. The
expanded 18-feature result has the lowest validation RMSE in this repeat and a
lower test RMSE than the previous expanded searches, but its nested powers,
division by precipitable water, and repeated use of the same validation split
make it exploratory rather than a preferred physical closure. Artifacts are in
each feature group's `pysr_richer_80_n5000/` and
`pysr_richer_80_c50_d8_n5000/` directories.

### Full-Training Calibration of the Simple Nested-RH Equation

The complexity-13 equation was held fixed while three identifiable coefficients
were fit by dense JAX gradients and L-BFGS-B on all 54,229 training rows. This is
the same fast offline calibration approach used for the earlier pooled SPEEDY
and symbolic comparison, not the more general JEM-Cal minibatch driver. Expanding
the selected product into two basis terms avoids a redundant product of scaling
coefficients. The fitted equation is:

```text
tanh(
  (rh_low_mean + rh_mid_mean)
  * (1.080393 * rh_vertical_range^3 + 0.979851 * rh_high_mean)
  - 0.000886
)
```

| Split | Uncalibrated RMSE | Calibrated RMSE | Calibrated SPEEDY RMSE |
| --- | ---: | ---: | ---: |
| Train | 0.30812 | 0.30791 | 0.33874 |
| Validation | 0.29895 | 0.29875 | 0.33003 |
| Test | 0.31109 | 0.31123 | 0.34379 |

The coefficient fit makes a negligible validation improvement and slightly
worsens test RMSE, so it does not provide evidence that calibration improves this
already compact structure. The calibrated nested-RH equation nevertheless
remains substantially better than calibrated SPEEDY on both held-out splits.
The implementation is in `calibrate_unified_cloud.py`; complete metrics and
predictions are in the 14-feature group's `fast_calibration_all_train/`
directory.

### Online Nested-RH Closures and ERA5 Smoke Test

The uncalibrated and calibrated complexity-13 equations are now selectable in
SPEEDY as `sr_nested_rh` and `sr_nested_rh_calibrated`. Their online JAX feature
calculation reproduces the offline definition: RH is clipped to `[0, 1.2]`,
interpolated to sigma 0.20, 0.34, 0.51, 0.685, 0.835, and 0.95, then reduced to
the three layer means and vertical range. Both remain one-component closures
with `cloudstr=0`.

`evaluate_speedy_era5_smoke.py` now compares calibrated SPEEDY and both nested-RH
closures, uses Gaussian quadrature weights for global metrics, and initializes
the model calendar from each ERA5 window so seasonal solar forcing is correct.
The 240x121 six-hour WeatherBench2 store used for atmospheric states has no
finite cloud-cover values; direct cloud evaluation therefore uses the finite
`total_cloud_cover` field in the local 64x32 six-hour store.

For one independently initialized 2020-01-01 00 UTC to 06 UTC window:

| Closure | Cloud RMSE | Cloud bias | Temperature RMSE | Humidity RMSE |
| --- | ---: | ---: | ---: | ---: |
| Calibrated SPEEDY | 0.27539 | -0.00843 | 4.28321 K | 0.68800 g kg-1 |
| Nested RH | **0.26010** | **0.00534** | **4.28289 K** | 0.68758 g kg-1 |
| Calibrated nested RH | 0.26062 | 0.01193 | 4.28291 K | **0.68758 g kg-1** |

This verifies full-model execution and direct cloud diagnostics only. One short,
unbalanced ERA5-initialized window cannot establish forecast skill, and the two
nested equations are effectively indistinguishable at this horizon. The result
is `outputs/era5_nested_rh_smoke_6h.json`. A broader run can repeat `--time` for
independent dates and repeat the script for each desired `--lead-hours` value.

### Same-Time ERA5 Cloud and RSUT Diagnostics

`evaluate_speedy_era5_diagnostic.py` evaluates physics on prescribed ERA5 states
without a forecast rollout. Each independent daily window contains the 00, 06,
12, and 18 UTC analyses. Cloud RMSE uses all four same-time ERA5
`total_cloud_cover` fields. SPEEDY radiation is daily mean, so model RSUT is
averaged across the four prescribed states and compared with the matching
four-synoptic-time ERA5 mean. ERA5 RSUT is derived using its downward-positive
TOA convention:

```text
RSUT = mean_top_downward_short_wave_radiation_flux
       - mean_top_net_short_wave_radiation_flux
```

The complete upstream data, feature, split, PySR, calibration, implementation,
and evaluation choice inventory is in
`AUTORESEARCH_DECISION_LEDGER_120_WINDOW.md`.

The diagnostic uses native T30 terrain and packaged climatological albedo, snow,
sea ice, SST, soil, and land-temperature forcing. Metrics use Gaussian
quadrature weights. The benchmark samples the seventh and twenty-first day of
every month from 2016 through 2020, giving 120 independent daily windows and 480
prescribed atmospheric states. Periodic longitude extension includes every model
grid cell in the cloud interpolation. Each closure therefore has 2,211,840 cloud
and 552,960 daily RSUT grid comparisons. Confidence intervals are paired
bootstrap intervals over daily windows.

| Closure | Cloud RMSE | Cloud bias | RSUT RMSE | RSUT bias |
| --- | ---: | ---: | ---: | ---: |
| Calibrated SPEEDY | 0.21112 | 0.03132 | **40.74 W m-2** | 10.23 W m-2 |
| Nested RH | 0.19035 | **0.02856** | 43.07 W m-2 | **8.72 W m-2** |
| Calibrated nested RH | **0.19023** | 0.03480 | 43.13 W m-2 | 9.37 W m-2 |

The area-weighted ERA5 reference distributions are cloud cover 0.62981 +/-
0.25563 and RSUT 97.14 +/- 61.01 W m-2. Dividing mean window RMSE by the ERA5
mean gives cloud/RSUT relative RMSE of 33.52%/41.94% for calibrated SPEEDY,
30.22%/44.33% for nested RH, and 30.20%/44.39% for calibrated nested RH. Using
the ERA5 standard deviation instead gives cloud/RSUT normalized RMSE of
82.59%/66.77%, 74.46%/70.59%, and 74.42%/70.69%, respectively. These are
normalizations of RMSE, not mean absolute percentage errors, which are unstable
for cloud and RSUT values near zero.

Relative to calibrated SPEEDY, uncalibrated nested RH changes cloud RMSE by
-0.02077 (95% CI -0.02228 to -0.01928) and RSUT RMSE by +2.33 W m-2 (95% CI
+2.14 to +2.52). Calibrated nested RH changes cloud RMSE by -0.02089 (95% CI
-0.02233 to -0.01952) and RSUT RMSE by +2.39 W m-2 (95% CI +2.20 to +2.58).
Thus both nested closures robustly improve total-cloud amount while degrading
reflected-shortwave spatial accuracy. Better total cloud cover alone does not
guarantee better radiation: SPEEDY's separate stratiform component, diagnosed
cloud-top placement, and fixed cloud optical properties also control RSUT.

An expanded 2011-2020 run uses the same two dates per month, giving 240 daily
windows and 960 prescribed states. It confirms the result: calibrated SPEEDY
has cloud/RSUT RMSE 0.21074/40.56 W m-2, while nested RH has
0.19077/43.10 W m-2. The paired nested-minus-baseline differences are -0.01997
for cloud (95% CI -0.02107 to -0.01888) and +2.540 W m-2 for RSUT (95% CI
+2.406 to +2.669). The report and processed RSUT cache are
`outputs/era5_nested_rh_diagnostic_240window.json` and
`outputs/era5_rsut_t30_240window_2011_2020.nc`.

The full report is
`outputs/era5_nested_rh_diagnostic_120window.json`. Processed RSUT targets are
cached in `outputs/era5_rsut_t30_120window_2016_2020.nc`; rebuilding that cache
from the public hourly source requires `gcsfs`. Dates and schemes can be extended
with repeated `--date` and `--scheme` arguments, or the two-per-month stratified
range can be changed with `--start-year` and `--end-year`. Reports include
`timings_seconds` for initialization, atmospheric-state preparation, cloud and
RSUT target preparation, each closure's physics execution, metric/bootstrap
evaluation, and total pre-serialization runtime.

### Fixed-Total Cloud-Partition Counterfactual

`evaluate_speedy_partition_counterfactual.py` tests whether the nested closure's
one-component treatment explains its RSUT degradation. On days 7 and 21 of each
month in 2011-2020, it holds nested total cloud and cloud top fixed, applies
calibrated SPEEDY's local `cloudstr/(cloudc+cloudstr)` ratio to that same total,
and reruns only shortwave radiation.

| Configuration | Cloud RMSE | RSUT RMSE | RSUT bias |
| --- | ---: | ---: | ---: |
| Calibrated SPEEDY | 0.21074 | **40.56 W m-2** | 9.97 W m-2 |
| Nested RH, one component | **0.19077** | 43.10 W m-2 | **8.65 W m-2** |
| Nested RH, SPEEDY-like partition | **0.19077** | 41.57 W m-2 | 9.00 W m-2 |

Repartitioning improves RSUT in all 240 windows and removes 60.1% of the nested
RSUT degradation relative to calibrated SPEEDY. The counterfactual remains 1.01
W m-2 worse than the baseline, so partitioning is a major but incomplete cause.
The SPEEDY ratio is not an observed component target, and the test is post-hoc.

Two weighting checks reject the hypothesis that nested RH improves cloud only
where sunlight or shortwave sensitivity is weak. Incoming-solar-weighted cloud
RMSE improves from 0.21187 to 0.19000 (10.32%) and improves in 239 windows. A
stricter proxy passes candidate and ERA5 total cloud through the same
nested-state cloud top and one-component SPEEDY shortwave operator; its RMSE
improves from 20.99 to 18.93 W m-2 (9.79%), in 235 windows. This proxy
includes insolation, atmospheric transmission, surface albedo, and the
shortwave operator's nonlinear response while intentionally holding component
placement fixed.

Replacing nested RH's cloud-top index with calibrated SPEEDY's index changes
RSUT RMSE by only +0.081 W m-2 before repartitioning and -0.0018 W m-2 after
repartitioning. Thus cloud-top diagnosis, including the nested closure's use of
the default `rhcl1=0.30` rather than calibrated SPEEDY's fitted
`rhcl1=0.32163`, is an implementation confound but not an important cause of
this 240-window RSUT gap. The evidence instead identifies the one-component
treatment (`cloudstr=0`) as the dominant tested mechanism. The residual 1.01
W m-2 may involve fixed optical properties, interactions with non-cloud
radiative errors, or favorable error cancellation in calibrated SPEEDY; these
tests do not distinguish those possibilities.
Results are in `outputs/era5_partition_counterfactual_240window.json`.

## 2026-08-18: Expanded 18-Feature MLP Capacity Baseline

`fit_unified_mlp.py` now accepts `--hidden-layers` and records its exact
architecture, trainable parameter count, and a split-aligned comparison with the
uncalibrated compact nested-RH equation. One predeclared run used all 18
moisture/wind features and a `256/256/128` ReLU architecture. Training retained
the earlier protocol: training-only standardization, Adam, batch size 256,
internal 15% early stopping, and seed `20260808`. It stopped after 44 epochs.

| Model | Inputs | Trainable parameters | Validation RMSE | Test RMSE |
| --- | ---: | ---: | ---: | ---: |
| Compact nested RH | 4 of 18 | 0 fitted | 0.29895 | 0.31109 |
| Expanded MLP, raw | 18 | 103,681 | 0.28320 | 0.29678 |
| Expanded MLP, clipped to `[0,1]` | 18 | 103,681 | **0.28283** | **0.29488** |

The clipped MLP improves validation RMSE by 5.39% and test RMSE by 5.21%
relative to the compact equation. On test it has MAE 0.21148, correlation
0.70392, bias -0.01594, and clips 3.33% of predictions. This is a capacity
baseline, not an architecture-only comparison: relative to the compact equation
it changes both feature count and function class, and it has more than 20 times
the parameters of the earlier five-feature MLP. It demonstrates predictive
headroom available to future feature/target searches, but it is not yet an
online parameterization and has not been evaluated for ERA5 transfer, radiation,
or forecast stability.

Artifacts are in
`outputs/symbolic_features_unified_t30_nested/group_18_moisture_wind/mlp_256_256_128_seed20260808/`.
