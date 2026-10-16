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

## Monthly-climatology anchors (#902)

A third declaration applies only to a 12-record `wrap_year` climatology:
what instant WITHIN its month each record represents. `forcing.
monthly_anchors` (`mid_month`, the default, or `month_start`) answers this
for the surface climatology (SST, sea ice, snow, soil) and for ozone and
the oxidant climatology, which share one reconstruction rule:

- **`mid_month`** follows ECHAM `mo_time_control.f90::time_weights`
  (consumed for SST/ice in `clsst.f90`, for ozone in `mo_o3clim.f90`):
  record `m` is the instant at month `m`'s own mid-point. The value at any
  other instant is a linear blend of the current month's record with
  whichever neighbour is closer — the previous month before mid-month, the
  next month after — weighted by the ACTUAL length (in days, leap February
  included) of the two months being blended, never a fixed 30-day segment.
  `jcm.forcing._echam_time_weights` is the reimplementation (independently
  verified against the Fortran's formula for 28/29/30/31-day months,
  before/after mid-month, and the Dec/Jan wrap, in `forcing_test.py`); the
  surface climatology reconstructs this at the daily-table expansion
  (`jcm.data.bc.interpolate.interpolate_to_daily`, which switches from the
  day-binned `resample().interpolate()` to exact elapsed-time `.interp()`
  because a month's true mid-point generally falls on a fractional day,
  e.g. 15.5 days into a 31-day month); ozone and the oxidant climatology
  select directly from their 12 records every step.
- **`month_start`** is the pre-#902 behaviour: record `m` is held constant
  from the 1st of Gregorian month `m` (a zero-order hold), which moves the
  annual cycle about half a month early relative to the monthly-mean
  meaning of the records and to ECHAM's own reconstruction.

**This changes the climate of every run forced this way.** `mid_month` is
the default because it is the faithful reconstruction; a run that must
reproduce the pre-#902 numbers (e.g. to isolate this change from an
unrelated difference) sets `forcing.monthly_anchors: month_start`
explicitly.

Two product families are declared exceptions and ignore this knob
entirely — their readers never expose an `anchors` parameter, so
`forcing.monthly_anchors` cannot reach them:

- **Emissions** (`emis_*`, `aero_emis_*`) are interval-total / flux-rate
  products, not interpolation knots: mid-month linear interpolation between
  two monthly means does not preserve either month's mean, so a 12-record
  emissions climatology stays held from the 1st (`read_anthropogenic_
  emissions`, `read_prescribed_aerosol_emissions`).
- **The Tegen dust-source and surface-roughness climatologies**
  (`read_dust_source`, `read_dust_roughness`) stay held from the 1st
  because the Fortran reference itself (`bgc_dust_read_monthly`) steps them
  by calendar month — "held" is the faithful behaviour here, not a
  shortcut.
- **The DMS seawater-concentration climatology** (`read_dms_seawater`)
  stays held from the 1st too, and for the same reason as Tegen dust above:
  the ECHAM-HAMMOZ reference itself does not mid-month-interpolate it.
  Unlike SST/ozone/oxidants, `mo_hammoz_emi_ocean.f90::init_emi_ocean`
  defines the DMS boundary condition with `ef_timedef = EF_IGNOREYEAR` and
  leaves `ef_interpolate` at its default, `EF_NOINTER`
  (`mo_boundary_condition.f90`) — the generic path that holds the active
  record and switches only once the model date reaches the next record's
  own file-encoded timestamp (`ef_get_first_timeindex`/
  `ef_get_next_timeindex` in `mo_external_field_processor.f90`), never
  `mo_time_control.f90::time_weights`'s mid-month blend. The shipped
  climatology's own `time` axis encodes every record at the 1st of its
  month, 00:00 (`days since 2000-01-01`, values `0, 31, 60, ..., 335`, the
  leap-year 1st-of-month offsets), and `EF_IGNOREYEAR` replays
  month/day/hour/minute/second unchanged every simulated year — so the
  reference switches DMS exactly at 00:00 on the civil 1st of each
  Gregorian month, the same instant `_monthly_climatology_index` (the
  `month_start` default) already uses. Investigated and confirmed by the
  now-closed issue 1127 rather than assumed; pinned by
  `test_dms_seawater_held_across_month_boundary` in `forcing_test.py`.

Every other 12-record `wrap_year` leaf (the generic `make_time_series`
default) also stays on `month_start`: only the shipped surface/ozone/
oxidant loaders opt into `mid_month` by default, so a caller building a
`TimeSeries` directly keeps today's numbers unless it asks for the new
reconstruction.

## Per-step lookup cost at long dated axes (#904)

Every dated (`by_date` / `by_date_interp`) leaf is looked up once per model
step. The installed `jax_datetime` 0.1.x `searchsorted` compares the query
against every record, so its cost grows linearly with the axis length —
negligible for a ~500-record AMIP bundle, measurable for a multi-decade
hourly archive. `jcm.forcing._by_date_index` instead does a genuine binary
search (`_lexicographic_upper_bound`, a fixed-depth `lax.fori_loop`
comparing `(day, second-of-day)` lexicographically), exact at any date and
with no int64/`jax_enable_x64` dependency (day/second-of-day stay in
int32; collapsing them to one scalar — `day*86400 + second` — would
overflow int32 by ~2038). See `forcing_test.py`'s `TestLexicographicSearch`
/ `TestForcingLookupScalesWithAxisLength` for the correctness and scaling
checks.

Computing this lookup once per DISTINCT axis rather than once per leaf
(many leaves — every field of one surface file, the four oxidant species,
one emissions product's sectors — share the identical axis object) was
tried and dropped: it only works if the shared leaves retain their Python
object identity up to the point of grouping, and in the real run loop they
do not. `Model._run_from_state` passes `forcing` as a dynamic (non-static)
`jax.jit` argument, and jit's own argument flattening substitutes a fresh
tracer for every leaf POSITION in the pytree — even leaves that were the
literal same array object before entering the trace — so an `id()`-based
grouping made after that point always degenerates to one group per leaf
(confirmed with a standalone two-field probe through `jax.jit`, and by
local Codex review of this change). Reliable sharing would need
`ForcingData`/`TimeSeries` restructured so a shared axis is represented
ONCE in the pytree's static STRUCTURE, not detected at runtime — out of
this PR's scope, and not needed: benchmarking confirmed the O(log N) search
alone already keeps 10 leaves on a 350,400-record axis within noise of the
456-record case (`TestForcingLookupScalesWithAxisLength`), so there was no
measured benefit left for the grouping to capture.

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
