# Forcing time semantics: alignment and persistence

Every time-resolved forcing input carries two temporal declarations, and jcm
infers neither of them from the file:

1. **Alignment** (`*_align`): whether the samples are a climatology replayed
   every model year (`wrap_year`) or dated samples placed on the model's exact
   clock (`by_date`, piecewise constant; `by_date_interp`, linear between
   samples).
2. **Persistence** (`*_persist`): what a *dated* input does outside its time
   axis. `strict` (the default everywhere) means the input must cover the run
   or the run fails before it is compiled. `hold` declares that holding the
   edge samples past the archive is the intended experiment ("2014 emissions
   after 2014").

Both follow one principle (#884): a temporal semantic that changes the
science is **declared, never inferred**. A year of monthly samples looks
exactly like a monthly climatology, so the file cannot say which it is. In
the same way a run past an archive's end looks the same whether it is a
mistake or a deliberate "hold the last year" experiment. There is therefore
no `auto` persistence. `align: auto` resolves only data-mirror and packaged
products, from the kind the mirror manifest records, and raises for any
other file (`jcm.forcing.resolve_align`).

## The inputs and their knobs

| input | alignment | persistence | Python door |
| --- | --- | --- | --- |
| surface file: SST, sea ice, land T, snow, soil, and the CO₂/CH₄/N₂O series riding in it | `forcing.align` | `forcing.persist` | `ForcingData.from_file` / `from_dataset(align_mode=, persist=)` |
| ozone | `forcing.ozone_align` | `forcing.ozone_persist` | `OzoneClimatology.from_file(align_mode=, persist=)` |
| emissions (`emis_*`, `aero_emis_*`) | `forcing.emissions_align` | `forcing.emissions_persist` | `read_anthropogenic_emissions` / `read_prescribed_aerosol_emissions(ds, align_mode=, persist=)` |
| oxidants | `forcing.oxidants_align` | `forcing.oxidants_persist` | `read_oxidant_vmr(..., align_mode=, persist=)` |
| prescribed surface fluxes | `forcing.prescribed_surface_flux.align` | `forcing.prescribed_surface_flux.persist` | `read_prescribed_surface_fluxes(..., align_mode=, persist=)` |
| MACv2-SP `year_weight` | dated by construction (one sample per year) | `forcing.macv2_persist` | `read_macv2_weights(path, persist=)` |
| nudging target | dated by construction | none: it is fetched for the run window | `NudgingTarget.from_dataset(..., persist=)` |
| DMS, dust | climatology only (`wrap_year`) | not applicable | their readers |

`ForcingData.from_bundles(..., persist=)` sets the policy of every dated
input it composes at once. `emissions_align` and `emissions_persist` also
take a list with one value per `emissions_file` product, because a list can
mix a transient product with a climatology, or a held product with a strict
one. The pySES backend reads the surface file and ozone as climatologies
only, so its dated inputs are emissions and oxidants, and they take the same
two policies. A `wrap_year` leaf covers every date by construction and
ignores its persistence. A `hold` declared on an input that turns out to be a
climatology is harmless, so a preset can declare it while a user swaps in a
climatology file.

The policy rides on every dated leaf (`TimeSeries.persist`, an int code like
`align_mode`), so the checks below need no side table and hold for leaves
nested inside ozone, the emission and oxidant mappings or a nudging target.

## What "covered" means

A `by_date` / `by_date_interp` leaf covers a run window when the window lies
inside its **usable span**. That span is judged by the one function
`jcm.forcing.by_date_coverage_error`:

- **Declared bounds.** When a reader finds CF time bounds (today the
  prescribed-flux reader, from `time_bnds` / the `bounds` attribute), the span
  is those intervals. They are kept per interval, so a gap between two
  disjoint intervals is a declared gap. The run must lie inside one
  contiguous stretch, because inside a gap a neighbouring sample would stand
  in for data the file says it lacks. Gaps are never inferred from the
  stamps.
- **Otherwise, one end interval of slack at each end.** The first spacing
  extends before the first sample and the last spacing after the last. A
  sample may be stamped at the start or the middle of the interval it
  represents, so a Jan-1…Dec-1 and a Jan-15…Dec-15 monthly archive both cover
  their calendar year. Calendar cadences step by calendar months (same day
  of month, or month end); any other cadence repeats its elapsed length
  (`jcm.forcing._repeat_cadence`). An interior gap never widens the slack.
- **A single dated sample** without bounds covers only its own instant.

For `by_date_interp` this slack is also the judgement on the interpolation
bracket. Past the last stamp there is no later sample to interpolate towards,
so selection holds the last value, and the slack says how long that sample
still stands for its own interval. A run ending after the last mid-month
stamp but inside its month is covered on purpose, not by accident. The
comparison is on exact whole seconds, so a run ending exactly on the edge
passes and one second past it fails. The same is true before the first
sample.

## Where it is checked

The rule is applied at two points, both before any compilation.

**Year expansion** (`jcm.data.input_resolution.expand_yearly_files`). A
`{year}` pattern expands over `forcing.years`. When the product's
`available_years` (or its `*_available_years` override) is known, the
expansion pads one year on each side for the interpolation bracket and clips
that pad where coverage ends. Clipping the pad is not a policy question: the
run-start check then judges whether the loaded axis covers the run. A
*requested* year outside coverage has no file at all. Under `strict` it
raises, naming the product, both ranges and the `hold` escape. Under `hold`
the expansion reuses the edge-year file and warns. This happens before
anything is fetched, so a strict failure costs no download.

**Run start** (`validate_run_forcing`, which calls
`jcm.forcing.check_forcing_coverage`). This is the one choke point every run
entry point applies to its concrete forcing. It walks every dated leaf,
judges it against the window, and reports all uncovered inputs in one error
with one line per input. A `hold` leaf warns once per held input instead,
naming the input, its usable span and the run window.

| entry point | window |
| --- | --- |
| `Model.run_from_state_with_carry` (so `run`, `resume`, `run_from_state`) | the run's exact `[start, start + total_time]` |
| `PrescribedStateModel.run` (`run.mode=prescribed`) | the span of the prescribed state times |
| CLI `run.mode=full` and `jcm.configurations.load`, right after forcing assembly | the configured `[run.start_time, start + total_time / end_time]` (`runners.configured_run_window`), which contains every chunk and resumed segment |
| `SingleColumnModel.run` | none: a column has no absolute date |

The check skips a window or leaf that is traced (a run inside a JAX
transformation), so it never forces a host read of a tracer. The
forced-mode surface-flux terms apply the same function to their own four
fields from `validate_forcing` (see {doc}`surface_exchange`).

## What `hold` does and records

Selection (`jcm.forcing._select_time_series`) clamps to the end samples
outside the axis whatever the policy is. The policy decides only whether a
run may ask for that. A held run therefore reads the first or last sample,
and for `by_date_interp` the end value, not an extrapolation. It warns once
per held input and process, so a chunked run or a resume does not repeat
the warning. It also records the policy of every dated input, with any held
interval, as the provenance fact `dated_input_persistence`. The CLI writes
that into the output as the global attribute
`jcm_prov_dated_input_persistence`.

## Choices the shipped configurations make

- Every default is `strict`, including `forcing=amip`, whose products share
  one 1950–2022 span.
- `forcing=era5` declares `ozone_persist: hold`. Its surface files run to
  2024, but the FZJ ozone bundles end in 2022. Holding the 2022 ozone for
  2023–24 run dates is a reasonable choice for a slowly evolving species, and
  the preset states it rather than letting the lookup do it silently.
  Transient emissions stay `strict` in that preset. Aerosol emissions change
  fast enough that "2022 emissions in 2024" is an experimental choice the user
  makes explicitly.
- The MACv2-SP `year_weight` axis ends at the file's last real year (SPv2.1:
  2023). The trailing `_FillValue` years in the file are not data. Keeping
  them, forward-filled, would hold the last amplitude inside an axis that
  looks covered. With the axis cut there, a run past that year follows the
  same rule as every other input, and `strict` matches the reference code,
  which stops out of range.
