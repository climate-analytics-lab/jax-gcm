# Grundner Cloud Equation Reproduction: Current Gaps

This note concerns Grundner et al. (2024), *Data-Driven Equation Discovery of a
Cloud Cover Parameterization*, DOI
[10.1029/2023MS003763](https://doi.org/10.1029/2023MS003763).

## Published Equation

The discovered cloud-cover function uses relative humidity, temperature,
vertical relative-humidity gradient, cloud liquid water `qc`, and cloud ice
`qi`. Its output is zero in condensate-free cells and otherwise clipped to the
physical cloud-fraction interval. The published coefficients were fitted on
coarse-grained DYAMOND storm-resolving-model data.

## Released Search Workflow

The released `1.1` source repository documents a broader offline search than
the current SGP test. Its constrained PySR notebook used standardized,
coarse-grained DYAMOND inputs and searched a 5,000-sample random training draw
for up to 7.7 hours. The candidate feature pool included humidity, liquid and
ice cloud water, temperature, pressure, geopotential height, wind speed, land
fraction, surface pressure, and first and second vertical derivatives.

The main five-feature searches used relative humidity, temperature, cloud
liquid water, cloud ice, and the vertical relative-humidity gradient. The
release also contains separate cloud-regime searches based on condensate and
pressure. PySR was allowed arithmetic, division, powers, and a broad set of
unary functions; operator-specific complexities, a maximum tree depth of four,
and a maximum expression complexity of 90 discouraged opaque expressions.

The search loss clipped only the *prediction* to the cloud-fraction interval
before comparing it to the target. The authors subsequently inspected candidate
equations, checked physical behavior through derivatives, and re-optimized
selected equation coefficients. Thus the loss clip is not an implementation
substitute for the published equation's condensate gate and final output bounds.

The source uses three temporally ordered folds for train/validation selection;
it selects its second fold as the preferred model. This differs from the fixed
whole-year SGP train/validation/test partition, which remains appropriate for
the current observational comparison.

## Current SGP Diagnostic Setup

The current ARMBE/SPEEDY experiment provides observed temperature, humidity,
cloud diagnostics independently at each timestamp. It does not provide the
published equation's full input state:

- ARMBE provides total cloud fraction and liquid-water path, not vertically
  resolved cloud liquid-water and cloud-ice mixing-ratio profiles matching `qc`
  and `qi`.
- The independent diagnostic intentionally resets tracer and physics carry for
  every sample. It therefore has no meaningful pre-existing model `qc` or `qi`.
- SPEEDY's current cloud closure is not a direct condensate-tracer formulation,
  so inserting the published equation requires a new definition of its `qc` and
  `qi` inputs.
- The current PySR table has only `rh_cloudc_max`, precipitation, `gse`, and
  `rh_lowest`. It is a deliberately limited observational baseline, not a
  reproduction of the Grundner feature space, search budget, or selection path.

## ECHAM Status

JCM can run ECHAM 1-moment physics, which has `qc` and `qi` tracers. The June
2018 SGP pilot now provides observed MICROBASE condensate, hydrostatically
reconstructed atmospheric profiles, and a mass-conserving ECHAM L47 adapter.
The remap closes liquid and ice column mass to about `3e-14` relative error.
Site forcing and land-boundary configuration remain separate online-stage work.

Initializing ECHAM condensate at zero and integrating a step would test a
model-generated-condensate online experiment. It would not directly reproduce
the published offline DYAMOND evaluation.

## Observational Evaluation

The published EQ4 physical form and its RH-monotonicity modification are
implemented in `evaluate_grundner_eq4.py`. Inputs follow the paper contract:
liquid-water RH, temperature in K, `qc/qi` in kg/kg, and `dRH/dz` in m-1. The
exact zero-condensate gate and `[0, 1]` output bounds are applied.

On the June validation split, constrained EQ4 obtains RMSE `0.08825` over 475
valid rows. On the 437 rows shared with the five-feature pilot baselines, its
RMSE is `0.09152`, compared with `0.16710` for Sundqvist and `0.08408` for
histogram gradient boosting. The June 22-30 outer holdout remains untouched.

## Physical-Contract Symbolic Search

A three-seed, 60-iteration PySR search fit the nonzero-condensate training
regime using the same physical input contract as EQ4. All three seeds converged
on nearly identical RH-temperature equations. The selected complexity-10 form
is

```text
0.05771679 * ((RH - 0.5339194013) / 0.2284489987
              + 2.019136
              - (T - 246.9929944) / 25.84616394)^2
```

followed by the exact zero-condensate gate and `[0, 1]` clipping. It obtains
validation RMSE `0.09088` on all 475 rows and `0.09450` on the 437 common-core
rows. EQ4 therefore remains the strongest compact symbolic candidate on this
split. The local equation is a useful simple finalist, but its discovered
structure uses condensate only through the gate and needs regime, perturbation,
and averaging-window diagnostics before candidate freeze.

## Reproduction Path

1. Freeze EQ4 and symbolic-search finalists without reading the outer holdout.
2. Implement frozen equations as pure JAX functions with the verified units,
   vertical-gradient convention, bounds, and condensate gate.
3. Validate offline/online parity on archived profiles.
4. Evaluate prescribed-state radiation and short continuous ECHAM evolution.

## Source Record

- Released code, tag `1.1`:
  <https://github.com/EyringMLClimateGroup/grundner23james_EquationDiscovery_CloudCover/tree/1.1>
- Constrained PySR notebook:
  <https://github.com/EyringMLClimateGroup/grundner23james_EquationDiscovery_CloudCover/blob/1.1/sec3_data-driven_modeling/sec33_symbolic_regression_fits/pysr_on_dyamond_regimes_with_physical_constraints.ipynb>
- Optimized candidate equations and derivative checks:
  <https://github.com/EyringMLClimateGroup/grundner23james_EquationDiscovery_CloudCover/blob/1.1/sec3_data-driven_modeling/sec33_symbolic_regression_fits/pysr_results/optimized_eqns.json>
