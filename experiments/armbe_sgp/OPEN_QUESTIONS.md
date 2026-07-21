# Remaining Questions: ARMBE Single-Column Experiment

This document records unresolved scientific and data-quality work for the SGP
ARMBE/SPEEDY SCM experiment. Completed run configurations and results are in
`LONG_INTERVAL_2018_FALL.md` and `UNNUDGED_6H_HINDCASTS_2018_FALL.md`.

## Established Baseline

- The production 2018 SGP ARMBEATM and ARMBECLDRAD records support both a
  prescribed-state diagnostic run and independent unnudged six-hour physics-only
  hindcasts.
- The SCM selects date-aligned forcing at every timestep, including seasonal
  solar geometry and the surface-temperature `TimeSeries`.
- Land fluxes are enabled for SGP (`fmask=1`, `lfluxland=True`), and evaluation
  selects surface-flux tile 0, the land tile.
- ARMBE humidity is normalized to kg/kg at input and converted to the g/kg
  unit used by the SPEEDY physics-facing state. Regression tests cover both
  supported source-unit conventions.
- The 2018 autumn sequence has 119 contiguous six-hour profiles after dropping
  580 incomplete hourly profiles and 10 valid profiles outside its dominant
  cadence phase.
- The independent hindcast resets atmospheric state, tracers, and physics carry
  for every six-hour window. Temperature, humidity, and horizontal winds evolve
  with SPEEDY physics only; there is no nudging, dynamics, advection, or
  large-scale forcing.

## 1. Apply Observation Quality Control

The current scores use mapped ARMBE values without ARM quality-control masks.
Before interpreting model-observation differences as physics errors:

- Apply product quality-control flags to radiative and surface-flux targets.
- Confirm pressure-level orientation, surface-pressure units, and the
  dewpoint-to-specific-humidity conversion against selected soundings.
- Report observational spread within each six-hour window with its mean.
- Compare BAEBBR and QCECOR turbulent-flux products where both are available.

## 2. Replace the Surface-Temperature Proxy

`temp_sfc` is used as `stl_am`, but it may be surface air temperature rather
than land skin temperature. This can suppress the land-air temperature contrast
and bias sensible heat fluxes.

- Identify a true land/skin-temperature field in an available ARM product.
- If none is suitable, assess an estimate from observed upwelling longwave with
  an explicit emissivity assumption.
- Record the selected source, units, and uncertainty in `armbe_io.py`.

## 3. Use Scores Appropriate to Each Experiment

The prescribed-state run is a diagnostic calculation, not forecast skill. The
unnudged run is forecast error over six-hour physics-only windows. Do not pool
their metrics.

- Use daily means for SPEEDY radiation comparisons because its shortwave scheme
  has daily-mean, zonally averaged insolation and no diurnal cycle.
- For interval-mean hindcast data, duration-weight intervals into complete UTC
  days and exclude incomplete boundary days. The existing complete-day plot
  does this; the old-style daily plot remains only for visual comparison with
  `compare.png`.
- Add a six-hour persistence baseline: the initial observed profile held fixed
  through the forecast window. This establishes whether physics-only evolution
  improves on no evolution.
- Retain six-hour surface/radiation values as process diagnostics, not directly
  comparable headline scores against a model without a diurnal cycle.

## 4. Add Missing Physical Constraints Deliberately

Only the surface-temperature proxy varies with the record. Soil moisture,
albedo, snow, sea ice, greenhouse gases, and all large-scale atmospheric
forcings are fixed or absent.

- Assess time-varying soil moisture and albedo only after identifying a source
  and a defensible mapping to SPEEDY forcing fields.
- Decide whether a forcing-constrained SCM experiment is needed for the target
  question. It must be labelled separately from the current physics-only
  hindcast.
- Keep pressure and geopotential treatment explicit: they are held at the
  initial observed state in each current hindcast window.

## 5. Interpret Hydrology and Clouds With Their Limits

The real-data runs show very low modeled latent heat and positive precipitation
biases. These are useful diagnostics but do not yet isolate a parameterization
defect because the surface-temperature proxy, fixed land state, and missing
large-scale forcing all affect them.

- Evaluate precipitation intermittency and its relationship to observed
  thermodynamic instability in the real windows.
- Keep precipitation diagnostic for prescribed-state calculations because
  temperature and humidity are reset at every observation.
- Liquid-water path remains target-only: the current SPEEDY archive has no
  directly comparable liquid-water-path diagnostic.
