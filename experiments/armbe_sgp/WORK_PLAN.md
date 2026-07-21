# ARMBE SGP Preparation Plan

This is the local backlog for work that can proceed before ARM credentials are
available. It separates completed plumbing from work that still needs real
observations.

## Completed

1. SCM forcing regression: `SingleColumnModel.run` is tested to select
   date-aligned forcing and refresh solar geometry inside its JITted scan.
2. Synthetic integration: fixture generation, `run_scm.py` archive output, and
   `evaluate.py` run together in a regression test.
3. ARMBE humidity is normalized to kg/kg, then converted to SPEEDY's g/kg;
   adapter tests cover both supported source-unit conventions.
4. Input validation: `validate_armbe_input` rejects malformed dimensions,
   pressure coordinates, timestamps, and implausible required state fields.
5. Run manifest: `run_scm.py` writes a JSON sidecar with input paths, resolved
   variables, retained time range, dropped samples, forcing choices, CLI
   options, and git revision beside every `.npz` archive.

## Provisional Evaluation Protocol

6. Use this fixed baseline for the first real-data pass; record deviations in
   the run manifest.

- Select a contiguous window with regular timestamps and set `--dt` to the
  observed cadence. Do not bridge missing profiles.
- Use diagnostic prescribed-state mode without relaxation for the baseline;
  relaxation is a separately labelled sensitivity experiment.
- Do not apply a spin-up exclusion: the atmospheric state is prescribed at each
  step. Evolving tracers begin from the first state and should be reported.
- Score daily means only. SPEEDY shortwave has no diurnal cycle.
- Score surface shortwave, downward longwave, sensible heat, and latent heat
  after their real-data mappings pass validation.
- Report precipitation, cloud fraction, and liquid-water path as diagnostics in
  the initial pass, not headline skill scores. Reconsider precipitation after a
  real-profile and relaxation sensitivity check.
- Archive the manifest, loader resolution report, command line, and output
  archive together for every comparison.

## Requires ARM Data

- Confirm actual ARMBE variable names, dimensions, and units.
- Identify a land skin-temperature forcing rather than assuming `temp_sfc` is
  suitable for `stl_am`.
- Compare the corrected longwave response and precipitation behavior against a
  real contiguous SGP window.
