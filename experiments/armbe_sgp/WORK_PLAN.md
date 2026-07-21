# ARMBE SGP Experiment Status and Next Work

## Completed

1. ARMBE adapter and validation
   - Production ARMBEATM and ARMBECLDRAD variable mappings are exercised on the
     2018 SGP archive.
   - Input validation rejects malformed dimensions, pressure coordinates,
     timestamps, and implausible required state fields.
   - ARMBE humidity is normalized to kg/kg and converted to SPEEDY g/kg with
     regression coverage for supported source units.
2. SCM forcing and synthetic integration
   - `SingleColumnModel` selects date-aligned forcing and refreshes solar
     geometry inside its scan.
   - The synthetic fixture remains an offline plumbing test for the loader,
     `run_scm.py`, and `evaluate.py`; it is not a scientific benchmark.
3. Real prescribed-state diagnostic experiment
   - The 2018-09-03 through 2018-10-02 SGP record produced 119 contiguous
     six-hour profiles after cadence filtering.
   - Results, manifest, archive, and daily comparison plot are documented in
     `LONG_INTERVAL_2018_FALL.md`.
4. Real independent unnudged hindcasts
   - 118 independent six-hour windows initialize from observed profiles and
     advance with twelve 30-minute SPEEDY physics steps.
   - Temperature, humidity, and horizontal winds are prognostic; state, tracers,
     and physics carry reset at each window.
   - Results and raw, old-style daily, and complete-day daily plots are
     documented in `UNNUDGED_6H_HINDCASTS_2018_FALL.md`.
5. Provenance and regression coverage
   - Run archives have JSON manifests with input paths, cadence filtering, model
     configuration, and git revision.
   - Focused SCM and ARMBE-adapter tests cover date-aligned forcing, prognostic
     state evolution, dropped-profile alignment, cadence handling, and the
     synthetic end-to-end pipeline.

## Next Priorities

1. Apply ARM quality-control masks and quantify observational uncertainty before
   using scores for scientific conclusions.
2. Replace the `temp_sfc` land-temperature proxy with a verified skin-temperature
   input, or document an explicit longwave/emissivity-derived estimate.
3. Add a six-hour persistence baseline for the independent hindcasts and compare
   final-profile errors against it.
4. Keep comparison resolution explicit:
   - Prescribed-state diagnostics use daily means.
   - Six-hour hindcast diagnostics remain process diagnostics because SPEEDY
     shortwave has no diurnal cycle.
   - Complete-day hindcast plots must duration-weight cross-midnight windows and
     exclude incomplete boundary days.
5. Evaluate whether time-varying soil moisture, albedo, or large-scale forcing
   is justified for a separately labelled constrained-SCM experiment.
6. Investigate the low modeled latent heat and precipitation behavior only after
   the surface and observation-quality issues above are addressed.

## Reproducibility

- `LONG_INTERVAL_2018_FALL.md` records the prescribed-state command and outputs.
- `UNNUDGED_6H_HINDCASTS_2018_FALL.md` records the unnudged-hindcast command,
  metrics, and plot variants.
- Local ARM inputs and generated archives remain ignored under `data/` and
  `outputs/`.
