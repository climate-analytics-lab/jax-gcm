# v3 scope: one real datetime clock

Status: agreed v3 implementation scope; implementation in progress on one PR.
Tracking issue: [#876](https://github.com/climate-analytics-lab/jax-gcm/issues/876).

Audited JCM checkout `e631ea7bb8f3f3b4774113748c120179297aca51` and
[jax-esm PR #125](https://github.com/climate-analytics-lab/jax-esm/pull/125)
at `d071debe4518f64eaac880f5eb48b7200e73506a`. The local jax-esm checkout is
older; the downstream findings below use files fetched at the PR revision.

## Recommendation

Make v3 use one real-date clock, with no public `calendar` option. Keep
`jax_datetime.Datetime` and `Timedelta`; advance exact time in the model carry,
use that same time for forcing and output, and let xarray handle date decoding,
run scheduling and monthly grouping at the interfaces. Coordinate a focused Coordax evaluation for forcing/output dimensions with #214.
The clock and monthly-output work do not depend on adopting Coordax; add it only
where it demonstrates a simplification of the broader labelled-array boundary.

The fixed convention is proleptic Gregorian dates, 86,400-second days, and
whole-second precision. Document this once; users should not choose a calendar.
Timezone normalization belongs at input, and leap seconds are outside this
model-time convention. CF calendar metadata can still appear in files: xarray
owns that encoding and decoding, not the physics or run API.

This is a good major-release change. It intentionally changes seasonal timing,
climatology selection, output labels and restart semantics. Merely changing the
default to `gregorian` does not fix all of these.

## What the current code actually does

| Area | Verified implementation | Consequence |
| --- | --- | --- |
| Clock | `Model.date_from_sim_time`, `model.py:758`, adds rounded integer days/seconds to a `jax_datetime` start date. `DateData` holds this real date. | There is already a real-date clock; there is no actual no-leap date type. |
| Seasonal default | `Model(..., calendar="365_day")`, `model.py:389`, applies to every physics package unless overridden. `date.py:190` uses `dt.delta.days % 365`, counted from 1970. | The seasonal year drifts against the clock. A direct probe gives phase day 7 on 2000-01-01 and phase day 14 on 2026-01-01. The default is not confined to SPEEDY. |
| Gregorian helpers | `date.py` already computes Gregorian Y/M/D and fraction of the actual 365- or 366-day year. | Reuse this small centralized implementation after checking its supported range; do not introduce another calendar engine. |
| Climatology | `_wrap_year_index`, `forcing.py:970`, uses `floor(year_fraction * n_records)` and ignores sample dates. | Twelve records are twelve equal fractions of a year, not January–December. Under Gregorian, 2005-03-01 still selects February; March begins March 3. |
| Surface climatology | `ForcingData.from_dataset`, `forcing.py:646`, first expands a 12-record wrapping surface file through `data/bc/interpolate.py:10` to daily data on **1981**. | Standard surface forcing often has 365 interpolated records, not 12. A twelve-record special case would leave this path wrong. Source-year and timestamp semantics are discarded. |
| Transient forcing | `_time_axis_seconds_from_ds`, `forcing.py:848`, maps dates to epoch seconds, reinterpreting cftime Y/M/D as Gregorian. `BY_DATE` holds the preceding record; `BY_DATE_INTERP` interpolates and clamps. | Noleap nominal dates are aligned already, but missing days and unsupported source dates need explicit policies. Epoch-second float32 arithmetic loses sub-minute precision. |
| Inferred repetition | `_resolve_align_mode`, `forcing.py:894`, infers climatology from a span no longer than 380 days. | A short transient series can silently become a repeating annual cycle. |
| Durations | `parse_duration_days`, `date.py:230`, turns months/years into average lengths. `_run_from_state`, `model.py:1137`, truncates both inner and outer step counts. | A month is not a named month. Requested labels can exceed the time actually integrated, and tails can disappear. With Gregorian `1 month` and 1800-second steps, the nominal interval is 2,629,746 seconds but integration covers 2,628,000. |
| Trajectory time | `model.py:1143` constructs floating-point epoch days and omits `start_date.delta.seconds`. `predictions.py:590` multiplies by floating nanoseconds per day. | Noon starts are labelled at midnight. Even float64 can produce 128 ns disagreements with exact labels (#862); float32 day coordinates are much coarser. |
| Other output | `dycore/pyses/dycore.py:1007` duplicates the conversion; `observers.py:78` has another. `snapshot_dataset`, `predictions.py:381`, emits days relative to the window. | Fix all streams together. Publishing just one existing float conversion leaves competing meanings of time. |
| Averaging | `_op_split_trajectory`, `model.py:249`, averages post-step native states and physics diagnostics, then converts the averaged native state to physical output. | Daily means need interval bounds. Nonlinear diagnostics of an averaged state are not necessarily means of those diagnostics. |
| Other entry points | `PrescribedStateModel`, `prescribed_state_model.py:191`, duplicates date conversion and calendar selection. SCM exposes a relative time axis and explicit forcing. | Include prescribed runs, observer preparation, snapshots, and relative-axis adapters in the migration; retain useful idealized/SCM relative time as a duration rather than pretending it is a date. |

The substance of [#449](https://github.com/climate-analytics-lab/jax-gcm/issues/449),
[#450](https://github.com/climate-analytics-lab/jax-gcm/issues/450),
[#805](https://github.com/climate-analytics-lab/jax-gcm/issues/805) and
[#862](https://github.com/climate-analytics-lab/jax-gcm/issues/862) is confirmed.
One qualification to #805: the ordinary surface reader expands monthly data
before selection, so not every named surface consumer still has twelve records.
Both monthly and daily repeating products need the new contract.

## The clock and API

Use one small run-state aggregate containing dynamics, physics carry, an exact
`Datetime`, and an integer step counter. The clock advances once per completed
model step by an exact `Timedelta`. Start date initializes this state; a restart
restores it. The step counter is for scheduling and stochastic/cached physics,
not an alternate source of dates.

Keep dycore-native elapsed time private to its adapter. Existing `sim_time`
fields may remain where the backend needs them, but neither forcing nor output
may recover dates from a long-running floating-point accumulator. Check backend
time against the authoritative clock and handle backend roundoff in the adapter.
Do not simply compute `step * dt_seconds` as int32: that overflows on long runs.
Increment normalized day/second values, or use an overflow-safe decomposition.

`DateData` can remain a thin per-step view of this clock, step and timestep.
Remove its calendar arguments and the model's `calendar` property. Preserve
`date_from_sim_time` only as a documented adapter for callers with elapsed
seconds; it must not remain the source of the main integration clock. Low-level
run/resume APIs should exchange the complete run state, so callers cannot lose
the physics carry or time accidentally.

Agreed public shape (target API):

```python
model = Model(..., start_time="2000-01-01", time_step=30)
daily = model.run(
    forcing,
    end_time="2001-01-01",
    save_interval="1D",
    output_averages=True,
)
monthly = daily.monthly_means()  # xarray Dataset of actual month means
```

The absolute endpoint is `end_time`; the alternative `total_time` is a fixed
elapsed duration. Supply exactly one. `start_time` initializes the clock and a
resume continues the saved clock. Month/year strings must not mean average day
counts. Fixed seconds/minutes/hours/days remain duration concepts.

Retain `save_interval` and `output_averages`. Do not introduce an `Output`
configuration class or a global `output="monthly"` setting. Observers evaluate
observation operators at timestep resolution, with track positions/validity
prepared ahead of the scan; their sampling remains independent of gridded save
frequency. Monthly aggregation is a separate operation on interval means.
`monthly_means()` validates bounds and sampling semantics: it cannot turn
snapshots into time means after integration. A later `run_monthly` convenience
may orchestrate fixed daily-mean integration and streaming aggregation, setting
averaging before execution; it must reuse the same implementation.

Observer datasets can reuse the reduction primitives with an explicit sampling
and weighting interpretation. A monthly mean of satellite overpass samples is
an overpass-sampled mean, not the continuous monthly mean of the model field.
Do not silently average every stream with one global setting.

Validate positive durations and exact divisibility by the integration step.
Never truncate silently. Whole-second timesteps are a deliberate v3 contract:
`jax-datetime` stores seconds, and current JCM rounds its clock already. Reject
subsecond steps/dates clearly rather than appearing to support them. If a real
subsecond use case emerges, resolve it before accepting that input; do not
silently quantize physics time.

Predictions carry exact datetime arrays plus bounds where applicable. Convert
with `Datetime.to_datetime64()` after transfer to the host; this already uses
integer arithmetic. Keep second-resolution datetime64 where possible, avoiding
the unnecessary date-range restriction of nanoseconds. All components must
represent the same instants even if xarray chooses a finer encoding. Publish
this exact time/bounds contract for JEM instead of freezing the old erroneous
float-day multiplication as a permanent API.

## Libraries and forcing interfaces

The division of responsibility should be small and explicit:

| Library | Responsibility |
| --- | --- |
| jax-datetime | Exact traced clock, duration arithmetic, datetime comparison/lookup and NumPy conversion. |
| xarray / pandas | Decode files, validate axes, normalize source dates, construct dated windows, group real months and serialize metadata. |
| Coordax | Named dimensions and coordinate identity for forcing/output fields; conversion to and from xarray. |
| JCM interface code | The scientific meaning of a forcing product: repetition, hold versus interpolation, interval means/totals, and missing-data policy. |

[JAX Datetime](https://pypi.org/project/jax-datetime/) already provides the
day/second representation and host conversion needed here. Its installed 0.1.0
also exports `searchsorted` and `interp`. A JIT probe with a device-resident
datetime axis correctly distinguishes records 30 seconds apart and interpolates
the midpoint. Its `searchsorted` implementation compares against every record;
its `interp` accepts one-dimensional values. For spatial fields, find bracketing
indices/weights once per shared time axis and gather whole fields. Benchmark
long transient axes; window input to bounded chunks before considering a custom
search implementation. Compare exact dates first and subtract local bracketing
times before converting durations to floats. Do not convert absolute dates to
float32 epoch seconds.

The earlier #214 comment saying Coordax lacks `.sel()` is now stale:
the [current Field API](https://coordax.readthedocs.io/en/latest/_autosummary/coordax.Field.html)
includes `.sel()` and `.isel()`, marked experimental in source. These are not a
promise of arbitrary traced datetime selection, interpolation, or resampling.
Its [xarray bridge](https://coordax.readthedocs.io/en/latest/xarray.html) is the
right place to reuse labelled field construction. Dynamic timestamps must remain
array data, not changing static coordinate metadata on every step/window.
Untag the scanned axis inside JAX and attach real-date labels at the output
boundary; [Coordax's scan rules](https://coordax.readthedocs.io/en/latest/jax_transformations.html)
require leading positional axes for shape-changing transformations.

Replace the current `TimeSeries(values, time_seconds, align_mode)` with a small
forcing specification holding a labelled field, a dynamic exact time axis when
needed, and explicit sampling semantics. A Coordax Field alone cannot encode
whether to repeat a year, hold a value, or interpolate. Resolve specifications
before physics; physics continues to receive instantaneous values.

Recommended input contract:

- **Dated series:** explicit `time` coordinate, hold or linear interpolation;
  sorted unique dates; validated coverage of the requested run and interpolation
  brackets. Out-of-range data errors by default; deliberate persistence must be
  declared. Preserve the AMIP `tosbcs`/`siconcbcs` interpolation convention.
- **Monthly climatology:** explicit `month=1..12`, with product-specific hold or
  interpolation. Held values switch on the first of the real month. Interpolated
  products retain real month-start/mid-month anchors and periodic Dec/Jan
  padding. Construct dated anchors for each requested window on the host; do
  not impose twelve equal fractional-year knots. Reuse data with an index map
  where possible, rather than copying a global field for every year/timestep.
- **Daily climatology:** month/day identity, not index divided by 365. At input,
  map it onto the actual requested dates. For continuous noleap products use a
  declared Feb-29 interpolation rule; held/categorical products use a declared
  hold rule. Never use annual phase stretching for all products indiscriminately.
- **Constant:** a timeless field. Do not infer repetition from record count or
  time span. Shipped readers know their product and can declare these semantics
  without adding user configuration to the normal path.

Remove the hard-coded 1981 expansion once direct dated interpolation is in
place. Where a source contains monthly *means* rather than interpolation knots,
do not assume linear interpolation preserves those means; retain the product's
established reconstruction or explicit piecewise-constant interpretation.

Handle foreign file conventions once at ingestion using
[xarray conversion](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.convert_calendar.html).
Noleap nominal dates can map to real dates, with the missing-day policy above.
Conversion does not itself supply interpolation or conserve annual totals.
Reject unsupported transient 360-day/Julian conversions with a useful message;
users can explicitly preprocess them with xarray. A declared twelve-month
climatology from a 360-day source can still be represented by its month labels.
For emissions supplied as interval totals, preserve totals through bounds-aware
conversion; adding a day of an unchanged flux rate changes the annual integral.
Record transformations in provenance, not in per-step physics.

`xarray-jax`, mentioned in #214, is a possible separate array-representation
choice. Its [documented static-coordinate recompilation and experimental dynamic
coordinates](https://github.com/google-deepmind/xarray_jax#treatment-of-xarray-coordinates-under-jax)
do not remove the clock/forcing decisions. This scope does not need a second
array-framework migration alongside Coordax.

## Real monthly output from fixed daily scans

Use the existing short physics timestep within a fixed one-day scan. The day is
an output/aggregation unit, not a one-day physics timestep. Stream daily results
through bounded host chunks; build month boundaries using xarray/pandas. A daily
kernel is reusable across February, leap February and 30/31-day months. There is
no need to compile a distinct trajectory for every month length.

For monthly means, compute **daily means from the inner steps**, then reduce by
real month using their durations. A mean of daily snapshots is a different
statistic. Tag interval means with midpoint timestamps, `time_bounds`, and
`cell_methods="time: mean"`; instantaneous samples retain their actual time.
With complete, equal-length midnight-aligned daily means, the reduction is the
ordinary [xarray monthly resample](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.resample.html)
`daily.resample(time="MS").mean()` on midpoint labels. Keep non-time metadata and
bounds out of the numerical reduction and reconstruct the monthly bounds.

Do not resample the existing end-of-day labels unchanged: a daily mean ending
at Feb 1 00:00 describes Jan 31. A probe for Jan–Feb 2000 gives counts
`[30, 29, 1]` when binned by end timestamps, versus `[31, 29]` using daily
interval-start labels. Midpoints have the latter membership too.

Persist sum and valid-duration per variable across host chunks; never average
chunk means without weights. Handle missing values per variable. Flush complete
months, and explicitly mark partial first/last months with actual bounds and
coverage. Carry any unfinished month's accumulator in a restart. Distinguish
means, accumulated quantities, and extrema; do not apply `.mean()` blindly to
every diagnostic or to integer clock metadata.

For starts away from midnight, integrate a leading segment to midnight and a
final partial day only if their boundaries are representable by whole model
steps. Reject a monthly request whose required boundaries cannot be reached
exactly rather than silently spreading an interval over two months. A bounded
edge segment may require an additional compilation; it does not justify
variable-month scans in the core.

There is one scientific averaging issue to resolve in this work: JCM currently
converts a mean native state to physical variables, including nonlinear
transforms/clipping. To promise true monthly means of output quantities, define
which physical fields are accumulated before averaging. Daily aggregation is
associative for those fields and the existing per-step physics diagnostics;
it is not generally equivalent to decoding one month-averaged native state.
Add a targeted diagnostic test and benchmark the extra decoding cost, rather
than declaring the two equivalent.

For differentiable monthly losses, host xarray reduction is insufficient. Keep
the compiled low-level API and reuse JEM's accumulator mechanism with host-built
month membership/weights supplied as dynamic arrays. This avoids implementing
Gregorian month arithmetic inside physics and avoids materializing a year of
daily global fields on the accelerator. The ordinary output path stays simple.

## SPEEDY physics impact

The relevant seasonality is already at an interface:
`ForcingData.select -> _solar_from_date -> SolarGeometry`.
`speedy_shortwave.py:572` takes `tyear` as a continuous annual phase, and
`speedy_shortwave.py:338` uses the empirical ozone phase offset `10 / 365`.
Other SPEEDY terms predominantly consume selected boundary values and a timestep
in seconds; there is no evidence here of a 365-entry requirement throughout the
physics implementation.

Use elapsed fraction of the **actual** year for solar geometry, computed once
from the real clock. Keep SPEEDY's Fourier coefficients and `10 / 365` fitted
phase constant unchanged initially; that literal is part of the parameterization,
not evidence that the model clock must use a noleap calendar. A leap year smoothly
traverses the same annual phase over 366 days. Do not replace radiation with a
new astronomical formulation as part of a datetime cleanup.

Keep monthly climatology sampling separate from this annual phase: months do
not have equal lengths. A reference-date seasonal override for idealized or
perpetual-season experiments should replace solar geometry/forcing at the
interface, while the clock continues normally.

Expect changed climate: removing the epoch-dependent seasonal offset and fixing
forcing phase changes the input to otherwise unchanged equations. Validate
SPEEDY solar/ozone at fixed phase separately from date-to-phase tests, then run
seasonal surface/radiation checks and climate regressions. Do not update all
reference data automatically merely because the date tests pass.

## jax-esm migration against PR #125

The PR's protocol-based component API and authoritative coupling step counter
are a good foundation. Preserve `Component.step(carry, time)`, exchangers,
subcycling, and optional serialization/checkpoint capabilities. The datetime
change belongs in `time` and adapters, not in each component's physics.

| PR #125 surface | Required change |
| --- | --- |
| `jem/base/component.py:185,275` | Replace `seconds_since_new_year`, fixed `days_per_year`, and `year_offset_seconds` with an exact datetime on `CouplingTime`; derive any solar phase from that date. `end_of_step()` advances the same datetime. |
| `jem/base/coupler.py:483,752,809,1168` | Remove calendar construction/bind checks. Keep start-date and timestep compatibility checks, construct exact clocks for nested/subcycled components, and expose exact interval axes. |
| `jem/base/component.py:372` (`TimeAxis`) | Remove the copied float-day output-label algorithm and calendar field. Delegate to the common exact conversion and publish bounds/sample semantics. |
| `jem/components/jcm/component.py:430,530` | Bind to the new JCM run-state/clock API; preserve physics carry and use exact coupling intervals. Remove `model.calendar` access and update drift checks. |
| `jem/components/jcm/component.py:222,378` | Preserve #125's exchanged-field declaration. Collapse only externally supplied forcing fields to instantaneous values; leave unexchanged land/chemical forcing time-varying. A Coordax migration must preserve the carry's shapes, dtypes and pytree structure when exchangers write. |
| `jem/components/slab/base.py:192,373`; slab ocean/land/ice consumers | Preserve monthly coordinates at load time. Replace equal-spacing `evaluate_cyclic_linear(time.year_fraction, ...)` with the same dated climatology sampler used by JCM, including start and end-of-step evaluation and initialization. |
| `jem/accumulate.py:91,913` | Gregorian monthly means currently raise an error. Replace fixed-year month tables and cyclic-year arithmetic with real-date bin schedules; retain the tested scan reducer and nested-component handling. Explicitly distinguish twelve climatological bins from sequential monthly output. |
| `jem/components/veros_component.py:411,826` | Veros uses elapsed seconds; no ocean-physics calendar rewrite is indicated. Update binding, output labels/bounds and clock comparisons. Preserve its float64 requirements and stable mixed-dtype carries. |
| `jem/driver.py:844`, runners/configs, JCM contract tests, checkpoint metadata | Remove average-month/year duration parsing, update documented signatures and JCM version pin, and version clock/accumulator checkpoint changes together. |

Two important details are easy to miss:

1. JEM's 365-day phase starts from the nominal month/day in a reference nonleap
   year. JCM starts from epoch days modulo 365. At 2000-01-01 JEM's phase is zero
   and JCM's is seven days. Passing the same calendar string never guaranteed
   identical seasonality. JEM's Gregorian phase also uses a constant 365.2425-day
   period, unlike JCM's actual-year fraction.
2. JEM's accumulator deliberately bins by the **end timestamp**, while its JCM
   adapter requests **interval averages**. Its Jan-31-to-Feb-1 atmospheric mean
   is therefore assigned to February. Slab outputs can be instantaneous, so a
   blanket one-day label shift is also wrong. Carry sample meaning and bounds per
   stream; apply duration-aware binning to means and instant-based binning to
   point samples. A ten-year monthly interval series should not need an extra
   bin solely to hold the final endpoint.

JEM already has an optional Coordax-aware irregular periodic interpolator in
`jem/utils/cycles.py:35`; evaluate it for reuse rather than adding a third one.
It still consumes fractional-cycle coordinates, so it is not itself a solution
to real dates or month lengths. The active slab loader currently drops time
coordinates and returns raw arrays.

This is a coordinated downstream API migration, not just deleting a keyword.
It simplifies the new API substantially but needs its own JEM companion change,
tested against #125's exchanged-forcing and nested-coupler behavior.

## Delivery scope and release gates

Deliver as four reviewable work packages, with a compatible JEM branch during
the transition rather than releasing mismatched versions:

1. **Exact clock and output contract.** Add the complete run state, remove
   calendar choice, validate fixed durations, centralize all output conversions,
   preserve non-midnight starts, and version restart state. Include prescribed
   runs and observer/snapshot axes. Coordinate JEM's clock/TimeAxis/bind change.
2. **Forcing and optional focused Coordax evaluation.** Replace implicit repetition and
   float epoch axes; add dated/month/day semantics and input-only normalization;
   preserve product interpolation. Migrate JEM slab climatology and #125's
   exchanged-field handling. Prove one surface and one emissions/chemistry path
   before extending to every reader in #805.
3. **Monthly output and aggregation.** Fixed daily kernel, interval bounds,
   weighted streaming resampling, partial months, resumable aggregation, and
   real-date schedules for JEM's differentiable reducer. Define physical-field
   averaging and keep snapshots/observers distinct.
4. **Scientific validation and migration docs.** SPEEDY and ECHAM/pySES checks,
   JEM slab and Veros smoke runs, checkpoint/version changes, examples/CLI,
   dependency pin and a concise v2-to-v3 guide. Remove obsolete calendar helpers
   and tests instead of keeping two operational clocks indefinitely.

Indicative planning estimate: roughly 2–4 engineering weeks for this coordinated
scope, plus available compute time for climate validation. This is an estimate,
not a measured implementation schedule. The largest uncertainties are true
physical-field averaging cost, run-state/checkpoint integration, and Coordax
behavior under existing sharding and transformations. Full #214 physics-array
migration and general CF-calendar simulation are outside this estimate.

Acceptance criteria:

- Exact agreement between clock, forcing and every output stream at leap day,
  year end, noon starts and 600/1200-second cadences; JCM/JEM datasets merge on
  exactly N timestamps, including multi-chunk/resumed and nested runs.
- Feb 1900/2000/2100 handled correctly; representative long runs retain exact
  seconds without requiring global x64. No float epoch conversion inside JIT.
- Every monthly held product switches on the real first of the month; daily
  and interpolated climatologies have documented Feb-29 and Dec/Jan behavior;
  short transient forcing never becomes a climatology by accident.
- Monthly sample durations/counts are correct (31/28 or 29/30/31 days), with
  no Jan-31 mean in February, correct partial months, and restart-independent
  sums/coverage. Constant and time-varying analytic signals check means/totals.
- JIT/scan/vmap and gradients with respect to state, parameters and forcing
  values work; dates remain nondifferentiable metadata. Changing window dates
  alone does not force recompilation. Dtype/sharding and exchange structures
  remain stable, including Veros imports enabling x64.
- Dependency compatibility is measured on supported JAX/Flax, and compilation,
  throughput and memory are benchmarked for both dycores and long forcing axes.
- Old checkpoints require an explicit migration or fail clearly. Changing
  scientific seasonality cannot provide bitwise continuation of a v2 run;
  importing its state as a new initial condition is a separate operation.

## Validation performed for this scope

Read current source and the linked issue discussions, plus the PR revision's
clock, JCM/Veros adapters, slab loader/consumers, reducers and driver. Consulted
upstream library documentation and Coordax selection source.

Executed small CPU probes using local JAX 0.10.0 and jax-datetime 0.1.0:

- Confirmed the 7/14-day January seasonal offsets and March-3 monthly switch.
- Reproduced 32 mismatched labels out of 144 ten-minute float64 labels for one
  day in 2000, with differences of plus/minus 128 ns against exact dates.
- Confirmed non-midnight output construction loses 12 hours for a noon start.
- Confirmed JIT float32 epoch seconds cannot distinguish 2026-01-01 00:00:00
  and 00:00:30 (ULP 128 seconds); exact `jdt.searchsorted`/`interp` do distinguish
  them when the axis is device-resident.
- Confirmed the truncation and daily/monthly interval-membership examples above.
- A float32 arithmetic probe at ten elapsed years increments a nominal
  1800-second clock by 1792 seconds. This demonstrates the representation risk,
  not a measured full-model ten-year trajectory.
- Ran `python -m pytest jcm/date_test.py -q --disable-warnings`: 10 passed;
  5 failed at Model construction because installed Dinosaur lacks the required
  semi-Lagrangian classes. These were environment failures, not evidence of
  failing date assertions. No model/climate validation or Coordax runtime
  benchmark was performed; Coordax is not installed in this environment.

The audit above predates implementation. Current implementation progress, test
results and release gates are tracked in the linked issue and consolidated PR.


## Initial implementation and remaining release gates

The initial v3 PR implements the exact run clock and checkpoint schema,
``start_time`` / ``total_time`` / ``end_time``, explicit dated/climatology
forcing, exact trajectory and auxiliary timestamps, and bounded monthly
aggregation with a resumable streaming accumulator. It keeps the existing
Observer sampling phase. Integer/boolean categorical diagnostics belong to
snapshot output, not interval means; averaged output records their omission.

The monthly helper accepts already bounded interval means. It rejects an
interval crossing a month boundary instead of inventing its two contributions.
Automatic leading/trailing segment scheduling for a non-midnight run is not
part of this first implementation. Streaming accumulator persistence is exposed
through ``state_dict`` / ``from_state_dict``; the chunked CLI does not yet
persist or publish a separate monthly stream automatically.

Dated lookup currently preserves endpoint holding outside the supplied axis.
The stricter input contract above (validate the requested window and require
explicit persistence) remains a release gate, as does replacing the existing
monthly-to-daily reconstruction with direct dated interpolation anchors.

Before a v3 release, finish the coordinated JEM API migration against PR #125,
run representative multi-year SPEEDY/ECHAM climate comparisons, and benchmark
the extra per-step physical-field decoding used by interval means. The installed
``jax_datetime`` 0.1 search implementation compares every forcing timestamp;
long high-frequency forcing axes need a scalability benchmark and potentially
an upstream search improvement. These are explicit release gates, not claims
that unit tests establish climate or performance equivalence.
