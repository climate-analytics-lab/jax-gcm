# Sequential Checklist for ECHAM Cloud Autoresearch

## Objective

Build a reproducible, condensate-aware symbolic-regression experiment for
ECHAM layer cloud fraction, then promote successful equations through online
ECHAM 1-moment and RRTMGP evaluation.

The first campaign is a bounded SGP pilot. Do not begin with the full MICROBASE
archive or claim a globally valid ECHAM closure.

## Stage 0: Freeze The Pilot Question

- [ ] Define the primary target as layer cloud fraction, not column-total cloud.
- [ ] Define one sample as an aggregated `(site, time window, ECHAM layer)` row.
- [ ] Select SGP production MICROBASE and overlapping ARMBEATM/ARMBECLDRAD.
- [ ] Select a limited initial period, preferably one month before expanding to
      one year.
- [ ] Declare ECHAM 1-moment physics as the online host.
- [ ] Declare the candidate insertion point: replace Sundqvist cloud fraction
      before RRTMGP, using only variables available at that point.
- [ ] State that this is Grundner-inspired feature exploration, not an exact
      DYAMOND reproduction.
- [ ] Reserve a fresh outer holdout before inspecting candidate performance.

**Artifact:** `echam_pilot_protocol.md`

**Exit criterion:** target, site, period, split strategy, permitted inputs, and
primary metrics are fixed before downloading or fitting.

## Stage 1: Audit MICROBASE Semantics

- [ ] Record datastream name, product generation, file version, cadence, height
      grid, units, and retrieval documentation.
- [ ] Identify liquid-water concentration, ice-water concentration, uncertainty,
      QC, retrieval, cloud/clear, and precipitation fields.
- [ ] Determine whether liquid/ice concentrations are in-cloud or grid/time mean.
- [ ] Determine how clear sky, below-detection condensate, retrieval failure, and
      missing data are encoded.
- [ ] Determine how mixed-phase and precipitating profiles are treated.
- [ ] Verify whether uncertainty fields are random-only or include systematic
      retrieval uncertainty.
- [ ] Confirm that MICROBASE and ARMBE `qc_*` names are not confused: ARMBE
      `qc_*` fields are quality flags, not cloud-liquid `qc`.

**Artifact:** `microbase_data_dictionary.json`

**Exit criterion:** an explicit rule exists for converting every MICROBASE
sample to valid liquid/ice condensate, clear, missing, or excluded status.

## Stage 2: Acquire A Bounded Collocated Sample

- [ ] Query only the selected SGP period; do not bulk-download the archive.
- [ ] Download matching MICROBASE, ARMBEATM, and ARMBECLDRAD daily files.
- [ ] Record ARM order/query details and source filenames.
- [ ] Compute checksums and retain product/version metadata.
- [ ] Verify UTC timestamps are monotonic and duplicate-free.
- [ ] Verify height coordinates and identify days with exact 596-level alignment.
- [ ] Quantify missing, invalid, precipitating, and retrieval-failure fractions.
- [ ] Produce quick time-height plots for several clear, liquid, ice, mixed-phase,
      and precipitating cases.

**Artifact:** `microbase_sgp_pilot_manifest.json`

**Exit criterion:** at least several representative days pass manual and
automated QC, with no unresolved unit or coordinate ambiguity.

## Stage 3: Define The Observational Operator

- [ ] Derive cloud occurrence from valid high-frequency radar/lidar masks.
- [ ] Calculate layer occurrence over 15-, 30-, 60-, 120-, and 360-minute windows.
- [ ] Aggregate thermodynamic and condensate inputs over the same windows.
- [ ] Keep valid-sample counts and target standard errors for every layer/window.
- [ ] Compare derived occurrence with archived hourly ARMBE `cld_frac`.
- [ ] Test sensitivity to averaging duration, minimum valid-profile count, and
      precipitation exclusion.
- [ ] Estimate sampled horizontal distance as wind speed times averaging time.
- [ ] Choose one primary averaging window and retain alternatives as robustness
      tests.
- [ ] Document the assumption that temporal occurrence represents horizontal
      ECHAM grid-cell fraction.

**Artifact:** `cloud_fraction_observation_operator.md`

**Exit criterion:** the target definition is reproducible and its sensitivity to
averaging window is smaller than, or explicitly included in, the expected model
comparison uncertainty.

## Stage 4: Convert To ECHAM-Compatible Profiles

- [ ] Compute collocated air density from pressure, temperature, and moisture.
- [ ] Convert MICROBASE concentration from `g m-3` to mixing ratio `kg kg-1`:

```text
qc = 1e-3 * liquid_water_concentration / air_density
qi = 1e-3 * ice_water_concentration / air_density
```

- [ ] If MICROBASE is in-cloud, apply the documented cloud-occurrence conversion
      required to obtain grid/time-mean condensate.
- [ ] Construct ECHAM L47 pressure half-levels from surface pressure.
- [ ] Map temperature and humidity using documented interpolation or layer means.
- [ ] Map `qc` and `qi` with a mass-conserving vertical operator.
- [ ] Map cloud occurrence with a valid-sample-weighted layer operator, not a
      vertical sum.
- [ ] Preserve both native-height and ECHAM-layer representations for audits.
- [ ] Verify column liquid/ice mass before and after remapping.
- [ ] Exclude levels outside reliable instrument coverage rather than labeling
      them clear.

**Artifact:** `echam_layer_adapter_validation.json`

**Exit criterion:** condensate mass closes within a predeclared tolerance and
several remapped profiles pass visual and numerical review.

## Stage 5: Freeze Features And Runtime Availability

- [ ] Start with the Grundner-style core: RH, temperature, vertical RH gradient,
      `qc`, and `qi`.
- [ ] Define derivative coordinate and units: height, pressure, log-pressure, or
      hybrid coordinate.
- [ ] Add first/second condensate derivatives only as a declared feature group.
- [ ] Define broader groups separately: pressure, height, specific humidity,
      wind, stability, land fraction, and surface pressure.
- [ ] Verify every online feature exists before ECHAM cloud fraction executes.
- [ ] Use incoming-state `qc/qi`; do not silently use condensate generated later
      in the same 1M microphysics call.
- [ ] Record clipping, smoothing, scaling, missingness, and derivative edge rules.
- [ ] Fit any standardization using training rows only.
- [ ] Implement one shared feature function for offline export and online parity.

**Artifact:** `echam_cloud_feature_contract.md`

**Exit criterion:** every feature has a definition, unit, valid range, runtime
source, and offline/online parity strategy.

## Stage 6: Build The Versioned Dataset And Splits

- [ ] Store chunked labelled arrays in NetCDF or Zarr; avoid a full raw CSV cache.
- [ ] Assign immutable IDs for site, source files, time window, profile, and layer.
- [ ] Keep all levels and averaging variants from one profile in the same split.
- [ ] Split by whole site-month blocks for development.
- [ ] Reserve a chronological or later-year SGP outer holdout.
- [ ] Reserve a future leave-site-out test before adding NSA or ENA.
- [ ] Record target validity, retrieval uncertainty, sample counts, and exclusions.
- [ ] Report counts by split, height/pressure, phase, cloud regime, and month.
- [ ] Check feature distributions and exact-zero frequencies for `qc/qi`.

**Artifact:** versioned `echam_layer_cloud_pilot/` cache and split manifest

**Exit criterion:** leakage tests pass and no profile, source window, or derived
variant crosses split boundaries.

## Stage 7: Establish Baselines Before Symbolic Search

- [ ] Evaluate clear-sky/climatology and persistence baselines.
- [ ] Evaluate native Sundqvist cloud fraction offline where inputs permit.
- [ ] Fit a train-only calibrated Sundqvist baseline.
- [ ] Evaluate an RH/temperature-only compact baseline.
- [ ] Evaluate `qc/qi` without derivatives.
- [ ] Evaluate the published Grundner equation with verified units and gate.
- [ ] Train a declared MLP capacity baseline on the richest permitted features.
- [ ] Report raw and physically bounded predictions separately.

**Artifact:** `baseline_comparison.json`

**Exit criterion:** target predictability, condensate value, and flexible-model
headroom are quantified before launching PySR.

## Stage 8: Predeclare The Symbolic-Regression Campaign

- [ ] Lock train/validation/outer-test roles.
- [ ] Lock feature-group ablations.
- [ ] Lock operator vocabulary, costs, nesting constraints, and safe operators.
- [ ] Lock maximum complexity/depth and search budget.
- [ ] Lock sample weighting and loss definition.
- [ ] Decide whether bounds and the zero-condensate gate are structural or part
      of a fixed output transform; do not hide them only inside the loss.
- [ ] Predeclare a cross-run selection rule combining validation error,
      complexity, numerical safety, and recurrence.
- [ ] Require multiple seeds or folds per important search configuration.
- [ ] Keep full Pareto frontiers and failure records.

**Artifact:** `echam_pysr_campaign.yaml`

**Exit criterion:** selection can run mechanically without inspecting the outer
holdout or choosing a preferred equation by eye afterward.

## Stage 9: Run Offline Search And Diagnostics

- [ ] Run thermodynamic-only, condensate, and condensate-derivative ablations.
- [ ] Evaluate all frontier equations on full validation blocks.
- [ ] Reject nonfinite, unsafe, unavailable-feature, or shape-invalid equations.
- [ ] Inspect response derivatives and off-manifold perturbations.
- [ ] Check monotonicity and behavior near zero condensate and phase transitions.
- [ ] Check recurrence across seeds/folds and averaging windows.
- [ ] Compare against Sundqvist, Grundner, and MLP baselines.
- [ ] Freeze finalists before touching the outer holdout.

**Primary offline metrics:** layer RMSE/Brier score, bias, calibration,
correlation, low/mid/high-cloud performance, cloudy/clear regimes, phase regimes,
equal-profile weighting, and equal-site weighting when available.

**Artifact:** complete search frontier and `offline_finalists.json`

**Exit criterion:** at least one compact finalist improves a declared baseline
across folds without physical-gate failures.

## Stage 10: Implement Finalists In ECHAM

- [ ] Implement the equation as a pure bounded JAX function.
- [ ] Add the condensate gate or fixed output transform exactly as declared.
- [ ] Wrap it as a replaceable ECHAM cloud-fraction `PhysicsTerm`.
- [ ] Preserve ECHAM process ordering unless a separate experiment changes it.
- [ ] Test offline/online feature parity on archived profiles.
- [ ] Test L47 shapes, gradients, JIT, vmap, NaN handling, and edge cases.
- [ ] Run identical-input comparisons against the offline evaluator.
- [ ] Record candidate equation, coefficients, feature contract, and source hash.

**Artifact:** tested online candidate implementation and parity report

**Exit criterion:** offline and online predictions agree within numerical
tolerance, and forward/gradient tests are finite.

## Stage 11: Prescribed-State ECHAM Evaluation

- [ ] Evaluate cloud-fraction profiles without evolving the atmospheric state.
- [ ] Compare native Sundqvist, Grundner, SR finalists, and MLP if implementable.
- [ ] Hold RRTMGP random seeds common across candidates.
- [ ] Report layer, column, low/mid/high, and phase-conditioned cloud metrics.
- [ ] Report RSUT, RLUT, surface fluxes, and radiative-heating profiles.
- [ ] Diagnose whether cloud improvements result from amount, placement, phase,
      or optical-depth changes.
- [ ] Use ECHAM-grey for broad screening and RRTMGP only for finalists.

**Artifact:** `echam_prescribed_state_evaluation.json`

**Exit criterion:** a candidate improves the declared cloud target without an
unacceptable radiation or numerical regression.

## Stage 12: Short Continuous Evolution

- [ ] Run single-step tendency and conservation checks.
- [ ] Run two-hour cycles containing one RRTMGP call and nine cached steps.
- [ ] Run one-day and multi-day trajectories from common initial states.
- [ ] Monitor cloud fraction, `qc`, `qi`, precipitation, RH, temperature, and
      surface/TOA energy budgets.
- [ ] Check water and energy conservation.
- [ ] Check cloud persistence, exact-zero condensate behavior, and drift.
- [ ] Reject candidates with finite forward output but nonfinite gradients.

**Artifact:** `echam_short_evolution_evaluation.json`

**Exit criterion:** the candidate remains finite and physically bounded, passes
conservation gates, and retains its prescribed-state advantage.

## Stage 13: Expand Beyond SGP

- [ ] Repeat the product/version audit separately for NSA and ENA.
- [ ] Rebuild site-specific collocation and vertical mapping; do not assume the
      SGP height grid or cadence.
- [ ] Evaluate the frozen SGP equation before retraining.
- [ ] Quantify feature and target distribution shift.
- [ ] Run leave-site-out evaluation.
- [ ] Use GOES or another spatial product to test temporal-to-spatial cloud
      fraction representativeness where possible.
- [ ] Retrain only under a new versioned multi-site campaign.

**Artifact:** `echam_multisite_transfer_report.md`

**Exit criterion:** the result generalizes across at least one contrasting site
or its limitations are explicitly regime-bounded.

## Stage 14: Final Reproducibility Package

- [ ] Save immutable source manifests and checksums.
- [ ] Save environment, package versions, hardware, and source revision.
- [ ] Save all configuration files, seeds, sampled rows, and split manifests.
- [ ] Save complete frontiers, not only selected equations.
- [ ] Save QC/exclusion counts and failed candidates.
- [ ] Save online parity, prescribed-state, evolution, and compute-cost reports.
- [ ] Document all human decisions and deviations from the predeclared protocol.

**Artifact:** versioned ECHAM autoresearch release

**Exit criterion:** another researcher can reconstruct the dataset, rerun model
selection, and reproduce the reported online evaluations without undocumented
choices.

## Immediate Next Actions

- [ ] Write and approve `echam_pilot_protocol.md`.
- [ ] Select the first SGP month.
- [ ] Complete the MICROBASE in-cloud versus grid-mean audit.
- [ ] Download only matching pilot files.
- [ ] Build and validate one day of collocated native-height profiles.
- [ ] Stop and review before scaling to a month or implementing PySR.
