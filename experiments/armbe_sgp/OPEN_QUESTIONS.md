# Remaining Questions: ARMBE Single-Column Experiment

This document records unresolved work for the SGP ARMBE/SPEEDY SCM
experiment. It intentionally does not retain resolved implementation history.

## Current baseline

- The SCM selects `ForcingData` at every timestep using the supplied
  `start_date` and calendar. This supplies seasonal solar geometry and slices
  `TimeSeries` forcing leaves.
- `run_scm.py` derives `start_date` from the first retained ARMBE timestamp and
  uses the Gregorian calendar.
- Land fluxes are enabled for SGP (`fmask=1`, `lfluxland=True`), and evaluation
  uses surface-flux tile 0, the land tile.
- Land surface temperature (`stl_am`) follows the retained input record through
  a date-aligned `TimeSeries`. Soil moisture and albedo remain fixed.
- ARMBE humidity is normalized to kg/kg while it is read, then converted to the
  g/kg unit used by the SPEEDY physics-facing state. The adapter has regression
  coverage for both kg/kg and g/kg source-unit handling.
- The synthetic fixture is only a plumbing fixture. Its profiles and evaluation
  targets are generated independently, so its skill metrics are not scientific
  results.
- Focused SCM and ARMBE-adapter tests cover the date-aligned surface-temperature
  forcing path, dropped-profile alignment, and regular-cadence validation.

## 1. Validate against real ARMBE files

The candidate variable mappings in `armbe_io.CANDIDATES` have not been
validated against a downloaded production ARMBE record. Before interpreting any
model-observation comparison:

- Download a contiguous SGP ARMBEATM and ARMBECLDRAD window.
- Run `python armbe_io.py <atm-file>` and verify every required field resolves
  to the expected variable and units.
- Confirm pressure-level orientation, surface-pressure units, and the
  dewpoint-to-specific-humidity conversion against a few soundings.
- Run `run_scm.py` with the real files and retain the loader report alongside
  the output.

The downloader queries the ARM API for availability; do not hardcode a coverage
period from stale metadata pages.

## 2. Establish an appropriate surface-temperature forcing

The current input field is named `temp_sfc` and may represent surface air
temperature rather than land skin temperature. Passing air temperature as
`stl_am` suppresses the land-air temperature contrast and can bias sensible
heat fluxes.

When real data are available, identify a true land/skin-temperature variable.
If none is present, assess whether it can be estimated from observed upwelling
longwave radiation using an explicit emissivity assumption. Record the choice
and its units in `armbe_io.py` rather than silently reusing air temperature.

## 3. Interpret precipitation only with real profiles

On the synthetic fixture, SPEEDY convection rains nearly every timestep at an
almost constant rate. This cannot diagnose a model defect: the fixture is
frequently convectively unstable by construction, and its precipitation target
is independent of its prescribed profiles.

For a real contiguous window, check:

- Whether convection and total precipitation are intermittent.
- Whether precipitation covaries with observed thermodynamic instability.
- Whether diagnostic prescribed-state mode is suitable for the intended score.

If repeatedly prescribing temperature and humidity prevents the convective
feedback needed for the experiment, test prognostic relaxation with
`relaxation_timescales` for those fields. That is an experiment design choice,
not a correction to synthetic-fixture metrics.

## 4. Decide how much time-varying surface forcing is required

Only land surface temperature varies with the record today. Soil moisture,
albedo, snow, sea ice, and greenhouse gases are fixed. For SGP land columns,
the most relevant next candidates are soil moisture and albedo, subject to data
availability and a clear mapping to the SPEEDY forcing fields.

Any added time-varying field must use `make_time_series(..., align_mode=BY_DATE)`
and share the retained timestamp axis. The runner deliberately rejects gaps and
irregular cadence because the SCM currently advances with a single fixed
`dt_seconds`.

## 5. Choose the model and comparison resolution deliberately

SPEEDY shortwave uses fraction-of-year and produces daily-mean,
zonally-averaged insolation. It has no diurnal cycle, so `evaluate.py` compares
daily means. Hourly radiation or surface-flux evaluation requires an ECHAM
radiation configuration or a different experiment design; it cannot be added by
changing the ARMBE input cadence alone.
