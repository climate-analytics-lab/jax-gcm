# The cloud-cover gate: definition and provenance

*Issue #782, "re-derive the `cloud_cover` gate" half. The physics question that
issue also raised — whether the post-#690 cloud state is right — is #682's.*

Total cloud cover is not a property of a cloud-fraction profile on its own: it
is a profile, an **overlap assumption** and a **time treatment**, and the
choices in common use differ by more than the model changes being measured
against them. This document records which one the release-validation gate
scores, why, and what the resulting numbers may and may not be compared with.

## What the gate scores

`tools/release_validation/health.py` scores `cloud_cover` on
`clouds.total_cloud_cover`: **ECHAM's own `aclcov`, accumulated in the model.**
Every step the model applies maximum-random overlap to the step's final cloud
fraction — cloud in vertically contiguous layers is treated as one cloud
(maximum overlap), cloud separated by clear air combines randomly — and the
saved frame is the mean of that per-step cover over the output interval
(`run.output_averages`, which the release-validation launcher sets
unconditionally, as do the `longrun` and `pyses_year` run configs). That is
ECHAM's accumulation. The clear-sky fraction accumulates down the column as

```
c(k)    = clip(paclc(k), 0, 1)                     ! guard added here, not ECHAM's
zclcov  = 1 - c(1)
DO k = 2, klev                                     ! DO 923; ICON: jks+1, klev
  zclcov = zclcov * (1 - max(c(k), c(k-1))) / (1 - min(c(k-1), zxsec))
END DO
aclcov  = 1 - zclcov                               ! aclcov_na: the instantaneous value
paclcov = paclcov + zdtime*aclcov                  ! ECHAM6's accumulated output
```

a transcription of `mo_cloud.f90` section "10.2 Total cloud cover": ICON
`atm_phy_echam/mo_cloud.f90` lines 1165-1182, identically ECHAM6
`mo_cloud.f90` lines 1359-1383 (ICON's `paclcov` is the instantaneous
value). `zxsec = 1 - 1e-12` is the Fortran's own guard against dividing by
`1 - paclc` at an overcast layer; the matching numerator is zero there, so
such a layer gives cover 1 rather than an infinity.

**Where it is computed.** One recurrence, `max_random_cover` in
`jcm/physics/clouds/cloud_overlap.py`, serves the jitted physics step and the
offline scorer alike (the latter hands it `numpy` and a lazy per-level
accessor, so a dask-backed window stays lazy), so the two cannot drift apart.
The model keeps the result in `CloudData.total_cloud_cover`, and
`CloudData.copy` recomputes it whenever `cloud_fraction` is replaced: the
field is always the cover of the fraction beside it, and no term that writes a
fraction can leave it stale. The microphysics' write-back of ECHAM's `paclc`
(`FSEL(-(zxlp1_d*zxip1_d), paclc, 0)`, section 8.4) is the last write to the
fraction before section 10.2 in ECHAM, and the last in the step in jcm, so the
saved cover is that of the **final** fraction, as ECHAM's is, and not of the
RH-diagnosed fraction the cover term starts the step with.

The Fortran's denominator `1 - min(c, zxsec)` is written as the equal
`max(1 - c, zepsec)`. `zxsec` is not representable in float32 (it rounds to
exactly 1.0), where the Fortran form turns an overcast layer's `0 / 1e-12` into
`0 / 0`: a NaN in the value (the saved field, and the NaN gate), and a NaN
gradient wherever a gradient path reaches the cover, since even a zero
cotangent into the Fortran form gives `0 * NaN`. The floor form is finite at
either precision, and every local derivative of the recurrence is finite.

The `clip` is the one deliberate addition. `paclc` inside the model is
constructed in `[0, 1]`, but *saved* output can carry small out-of-range
excursions, and a `c > 1` would make the numerator negative and the
"clear-sky fraction" meaningless. With it, no clip is needed on the way out:
each factor's numerator `1 - max(c_k, c_{k-1})` is at most its denominator
`max(1 - c_{k-1}, zepsec)`, so every factor, and hence the product and the
cover, lies in `[0, 1]`.

Three properties make this the right quantity for a gate:

* **It is the reference model's definition.** The number is the same
  construction, on the same instantaneous field and with the same
  accumulation, as ECHAM6's `aclcov` ([Stevens et al.
  2013](https://doi.org/10.1002/jame.20015)), and it is a total cover, which
  is the basis the satellite climatologies are quoted on — rather than a
  reduction peculiar to one model's post-processing.
* **It is deterministic and uses the full fraction.** There is no sub-column
  sampling and no optical-depth threshold, so two runs of the same state score
  the same number, and the cover does not move with how thin a cloud is.
* **It is orientation-independent.** Cancelling the denominators leaves the
  clear-sky product as the adjacent-pair factors `1 - max(c_k, c_{k-1})`
  divided by the *interior* levels' `1 - c_k`, and both of those sets survive
  reversing the axis unchanged. The physics-internal frame is top-first, the
  saved output surface-first, and files written before #710 carry TOA-first
  interfaces (see
  [output_vertical_conventions](output_vertical_conventions.md)); all score the
  same, so neither the in-model nor the offline function carries an
  orientation guard. The cancellation is exact, but the two orders agree only
  *to rounding*: the `max(1 - c_{k-1}, zepsec)` guard is applied in loop
  order, so it caps a different denominator in the reversed column, and a
  profile holding cover within `zepsec` of 1 (`[0.2, 1 - 5e-13]`, say) differs
  by `O(zepsec)`. Over 200k random 47-level profiles (uniform, sparsified, and
  with overcast layers injected) the worst forward-versus-reversed difference
  was 1.1e-16.

### Output without the online field

`jcm.analysis.total_cloud_cover` applies the same recurrence to a *saved*
`clouds.cloud_fraction`, and `health.py` falls back to it, with a NOTE, when
the window carries no `clouds.total_cloud_cover` (output written before the
field existed). That keeps archived output scoreable, but it is a **different
and lower number**. Under `run.output_averages` the saved fraction is already a
time mean over the output interval, and the overlap product is non-linear in
it. Smoothing over the interval moves each layer toward its mean fraction, and
where cloud moved between layers during the interval that breaks the maximum
overlap chain the instantaneous cloud had: a 0.5 cloud that alternates between
the top and the bottom layer of a three-layer column is 0.5 cover on every step
and so 0.5 on average, while its mean profile (0.25 in each outer layer, clear
between) overlaps to 0.4375. The inequality is not general — clouds that fill
several layers together, at the same times, push it the other way, and a column
that is overcast half the time and clear the other half gives the same answer
either way — but it is the usual direction. The fallback is an estimate whose
bias is usually low and can have either sign; it is not a bound.

### The covers side by side

All covers are area-weighted, and time-averaged **after** the overlap product
(it is non-linear). Both rows are archived year runs scored with `--last-n 40`
(the settled ~200 days of 5-day means). They predate the online field, so the
gate read them through the fallback, and both FAIL the 0.5 floor at **0.46**.

| run | saved means | offline overlap of the saved profile | column max | `radiation.total_cloud_cover` | **online** `clouds.total_cloud_cover` |
|---|---|---|---|---|---|
| JAM control year (2M + JAM), jcm `519f18e8`, days ~170-365 | 5-day | 0.462 | 0.402 | 0.582 | not saved |
| 2M control year (2M + MACv2-SP), jcm `49c0724c`, days ~170-365 | 5-day | 0.459 | 0.395 | 0.584 | not saved |

Provenance, because it bounds what the table can be used for:

* Both rows are T63 L47 `ECHAM+RRTMGP` years from the January end state of the
  host's warm-state set (`echam-jam-t63-l47_jan_fixed_49c0724c`,
  `echam-2m-t63_jan_fixed_49c0724c`), `run.start_time=2000-12-31`, 12-minute
  time step, 5-day means, scored by the `health.py` of the tree that ran them.
  The JAM year ran on Nautilus, the 2M year on the dev workstation.

Reading it: the offline overlap of the saved profile is the only cover these
files give the gate, and the 0.12 by which it sits below the McICA 0.58 is
context, not a measurement of the averaging bias, because that cover differs
from the online one in its fraction and its thresholds too (see below).

### Printed and not gated

| reported | what it is | why it is not the gate |
|---|---|---|
| `cloud_cover_colmax` | `clouds.cloud_fraction.max("level")` of the saved (interval-mean) fraction — the column-maximum cover of the #638 and #782 tables | It assumes *every* layer overlaps maximally, so it is only a **lower bound**: two half-covered decks in different parts of the column read 0.5 where the sky is 0.75 covered. Retained so the #638 and #782 tables remain readable. |
| `cloud_cover_radiation` | mean of `radiation.total_cloud_cover` — the cover the flux solve integrates: under RRTMGP the fraction of McICA sub-columns with at least one cloudy layer, under the overlap rule the solve actually uses (default: maximum-random, ECHAM6.3's rule; the exponential option adds a decorrelation length); under grey two-stream the beam-split weight between its clear and cloudy calls (`column_total_cover`, the column maximum for the maximum-random and exponential rules) | The radiation view, with its own inputs and its own overlap treatment (sampled under RRTMGP, whose maximum-random sampler has `aclcov`'s adjacent-layer product as its expectation but is a finite draw of it; a column-maximum approximation under grey), and absent from output written before the diagnostic existed (`b772ffec`, 2026-07-31). An all-zero field is dropped rather than reported, so output that does not carry it cannot look like a cloudless run; the printed NOTE says which of the two cases held. |

Random overlap, `1 - prod(1 - c_k)`, is the opposite bound: it ignores that a
physically continuous cloud spans several model layers, and so double-counts
its edges. It is neither scored nor reported.

The observed total cover is printed beside the gate as `cloud_cover_obs`
(**0.63**, see "The observed reference" below). It is a reference for reading
the number, not a band.

### `cloud_cover_radiation` is a different measurement, not a cross-check

Both the online cover and `radiation.total_cloud_cover` are time means of an
*instantaneous* cover, so unlike the offline overlap they are comparable in
kind, but they are not the same field, and they are not expected to agree:

* **Different fraction.** ECHAM calls `cover`, then radiation, and only then
  `cloud` (`physc.f90` l.543, 566 and 1067): radiation integrates the
  RH-diagnosed fraction, masked to cells with step-start condensate, before the
  microphysics has written back its post-microphysics `paclc`. The online cover
  is of the fraction the step *leaves*, after cells below `ccwmin` in both
  phases have been cleared. jcm keeps that order, so the two are the cover of
  different fractions even in the same step.
* **Different treatment of thin cloud, and a sampled estimate.**
  `radiation.total_cloud_cover` is built from
  `effective_cloud_fraction(cloud_fraction, eps=cld_frac_min)`
  (`jcm/physics/radiation/mcica.py`), which zeroes every cell with
  `cloud_fraction <= 2*cld_frac_min` so the sampler and the optics agree about
  which cells are empty, and under RRTMGP it counts a finite set of sampled
  sub-columns. The online cover has neither.

Treat a gap between them as expected, and read neither as a check on the other.

### Measured magnitudes

The three offline definitions of cover (column maximum, maximum-random and
random overlap of the *saved mean profile*), scored with
`jcm.analysis.total_cloud_cover` on the archived T63 L47 ECHAM+RRTMGP output on
the shared dev workstation. Area-weighted, and the reduction is taken
**after** the overlap product in every row (the product is non-linear). The
runs save 5-day means, so "last 40 saved frames" is the window
`--last-n 40` picks out of a 5-day-chunked run — the settled ~200 days. The
online cover is not in this table because these runs predate it; it sits above
the max-random column by the averaging bias set out above.

| run | code point | window | column max | **max-random** | random | offset |
|---|---|---|---|---|---|---|
| 2M year<br>`clim_fixed_260703/v10_2m_prefix_year/echam-rrtmgp-2m_day*.nc` | dev @ 2026-07-04 | full year (73 frames, days 6-365) | 0.544 | **0.691** | 0.843 | +0.147 |
| " | " | settled (last 40 frames, days 171-365) | 0.553 | **0.701** | 0.847 | +0.148 |
| 1M year<br>`clim_fixed_260703/v7_1m_fullyear/echam-rrtmgp_day*.nc` | dev @ 2026-07-04 | full year | 0.555 | **0.679** | 0.815 | +0.124 |
| " | " | settled (last 40 frames) | 0.563 | **0.691** | 0.823 | +0.128 |
| 2M+JAM 90-day arm<br>`jam_scav_ab/abbase_260823_0100_day{30,60,90}.nc` | jcm `0ee92eaa`, **post-#707** | days 1-90 | 0.425 | **0.535** | 0.699 | +0.111 |
| " | " | days 61-90 | 0.417 | **0.527** | 0.703 | +0.110 |
| JAM control year (2M + JAM) | jcm `519f18e8`, **post-#707** | settled (last 40 frames) | 0.402 | **0.462** | not scored | +0.060 |
| 2M control year (2M + MACv2-SP) | jcm `49c0724c`, **post-#707** | settled (last 40 frames) | 0.395 | **0.459** | not scored | +0.064 |

Provenance and caveats, because they bound what the table can be used for:

* The two year runs are **pre-#690, pre-#707 and pre-#710**. Their directories
  carry no surviving `.hydra` snapshot, so the code point is fixed only by the
  run logs' date (2026-07-04) and the full resolved config they echo; #690
  merged 2026-08-21 and #707 2026-08-22.
* The 2M year additionally carries a known defect — its own `README.txt`
  records it as the pre-orientation-fix arm, with inverted MACv2-SP shortwave
  aerosol and ozone. Its *absolute* cover is therefore not a climatology; its
  offset between definitions is what this table uses it for.
* The 90-day arm is **post-#707** (verified:
  `git merge-base --is-ancestor 5ba96f7f 0ee92eaa`). It is a 90-day spin-up from
  a dry Jablonowski-Williamson start, so its absolute values are spin-up
  values, not climate.
* The two control years are the settled, post-#707 year runs (provenance in
  the side-by-side table above); only the column-max and max-random columns
  were recorded for them, so the random-overlap column is empty.
* `radiation.total_cloud_cover` is **absent** from both year runs (they
  predate `b772ffec`). The 90-day arm saves it: 0.757 over days 1-90 and 0.776
  over days 61-90 — against a max-random 0.535/0.527 on the same files, which
  is the gap the section above explains.

Two readings come out of this. First, the spread across definitions is
**~0.27-0.30**, larger than any model change the gate has ever been asked to
judge — which is the whole reason the definition has to be pinned down.
Second, the column max is low against max-random by an offset that is not a
constant: **0.11 to 0.15** on the pre-#690 year runs and on the 90-day
post-#707 spin-up arm, but **0.06** on the two settled post-#707 control years.
It is the artefact a column-maximum score carries, and its size depends on the
cloud state, so a band placed by this offset has to be read with that spread in mind.

### Reconciling with the #638 and #782 tables

The numbers recorded in the #638 baseline sweep (2026-08-16) and the #782
comparison are **column maxima** of the saved mean profile, on different code
points, at T63 *and* T106, scored with `--last-n 40`. They are not directly comparable with the table
above, which is a different (earlier) code point at T63 only — the column max
there reads 0.54-0.56 against 0.60-0.70 in the matrix, and that difference is
model change plus resolution, not definition. What *is* transferable is the
definitional offset, so the matrix maps onto the new definition as:

| member | column max (#638 → post-#690/#707) | max-random (mapped) |
|---|---|---|
| `echam-1m-t63` | 0.68 → 0.63 | ~0.81 → ~0.74 |
| `echam-1m-t106` | 0.70 → 0.66 | ~0.83 → ~0.77 |
| `echam-2m-t63` | 0.60 → 0.48 | ~0.75 → ~0.59 |
| `echam-2m-t106` | 0.61 → 0.49 | ~0.76 → ~0.60 |
| `echam-jam-t63-l47` | 0.61 → (scrapped) | ~0.76 → — |
| `echam-jam-t63-l95` | 0.61 → (scrapped) | ~0.76 → — |
| `speedy-t31` | 0.57 → 0.58 | **n/a — different quantity** |

using the offset measured on the matching scheme for the #638 column (1M
+0.128, 2M +0.148, the JAM members being 2M) and the post-#707 offset (+0.110)
for the current one. The two JAM members were scrapped in the #782 sweep
pending the dust/sea-salt emissions investigation, so they have a #638 column
only. These are *mapped* values on the offline max-random definition, not
measurements, and the online cover of the same member reads higher than them
(see the side-by-side table). The post-#707 column is also an upper estimate:
the settled control years measure the offset directly at +0.06, not +0.11, which
would map the post-#707 members to ~0.69 and ~0.72 (1M, T63 and T106) and ~0.54
and ~0.55 (2M). The next validation sweep saves the online cover and replaces
this mapping with measurements.

`speedy-t31` is deliberately outside the mapping. SPEEDY scores
`shortwave_rad.cloudc`, its own RH-based column cover
(`jcm/physics/radiation/speedy_shortwave.py`), which has no profile to overlap
and which nothing in this work touched. There is no offset to apply to it, and
it keeps the band it was calibrated with — see below.

### The observed reference

`health.py` prints `cloud_cover_obs = 0.63` beside the gate: the area-weighted
global mean (0.632) of the ESA-CCI CLOUD v3.0 AVHRR-AMPM total cloud cover
(`clt`), 1997-2016, conservatively remapped to the T63 Gaussian grid
(`obs/t63/clt.nc` of the jcm-monitor repository, variable `annual`, the mean of
the twelve monthly climatologies). It is the `clt` product the monitor's maps
and the calibration compare the model against. It is a reference for reading
the gated number, not a band: satellite products differ with their detection
threshold by more than the model changes the gate is asked to judge (the
GEWEX figures in the next section), so no single product is a pass line.

### The band

The gate band is **0.5-0.9**: the same width the gate has always had, placed
on the max-random definition.

Placement is the whole question, because a band is calibrated against a
quantity. The previous band, 0.4-0.8, was calibrated by experience with column
maxima. Carrying it unchanged onto a quantity that reads 0.06-0.15 higher
would loosen the floor and tighten the ceiling by that offset, and the mapped
matrix above shows that is not academic: `echam-1m-t106`'s #638 baseline maps
to ~0.83 and `echam-1m-t63`'s to ~0.81, so members that passed would fail on
the ceiling for a reason that has nothing to do with the model. Shifting by
the measured offset rounded to 0.1 — rather than the full 0.13, which would
eat into the floor where the 2M member sits — keeps the gate's character
(loose, spin-up tolerant, a "did the model produce a climate" test rather than
a tuning target) and brackets both anchors:

* **Observations.** The GEWEX Cloud Assessment database gives a global cloud
  amount of **0.68 ± 0.03** for clouds of optical depth > 0.1, rising to 0.74
  when clouds down to COD > 0.01 are counted (CALIPSO) and falling to 0.56 for
  COD > 2 (POLDER) — the detection threshold matters more than the
  inter-dataset spread ([Stubenrauch et al. 2013, BAMS 94,
  1031-1049](https://doi.org/10.1175/BAMS-D-12-00117.1); values as summarised
  by the [NCAR Climate Data
  Guide](https://climatedataguide.ucar.edu/climate-data/cloud-dataset-overview)).
  The whole 0.56-0.74 range sits inside the band. Individual products can sit
  a long way below that when their definition excludes thin or broken cloud:
  under the COSP simulator definitions, MODIS reads 0.49 and CloudSat 0.50
  against higher CALIOP/MISR/ISCCP values ([Kay et al. 2012, J. Climate 25,
  5190-5207](https://doi.org/10.1175/JCLI-D-11-00469.1), figure caption on
  p. 5196) — which is why the anchor here is a range and not one satellite
  number.
* **The model.** Every **ECHAM** member of the #638/#782 matrix maps into
  **0.59-0.83** on the offline definition, and the directly measured year runs
  sit at 0.68-0.70. The tightest margin is 0.07, at the ceiling, and it is held
  by a *superseded* baseline; the current code point maps to 0.59-0.77 with the
  spin-up arm's offset and to 0.54-0.72 with the control years', so 0.04-0.09
  above the floor for the 2M members. The 2M control year itself reads 0.459 on
  the offline definition, 0.04 below the floor, while its McICA cover reads
  0.58.

The mapped values are offline overlaps of a saved mean profile; the online
cover the gate scores reads above them, which moves a member toward the ceiling and away from
the floor. The band stays at 0.5-0.9 by the maintainer's decision; the members
to watch against the 0.9 ceiling are the 1M ones, whose mapped values are the
highest.

### SPEEDY keeps the old band

`speedy-t31` is gated on `RANGES["cloud_cover_speedy"]`, which stays at
**0.4-0.8**.

This is the same argument applied honestly in the other direction. A band is
calibrated against a quantity; SPEEDY's quantity did not change, so its band
must not move either. Its recorded values are 0.57 (#638) and 0.58 (#782) —
inside 0.4-0.8 with 0.17 of floor headroom, and inside 0.5-0.9 too, but with
only 0.07. Carrying the ECHAM shift onto it would have tightened the floor of
the member that sits closest to it, for a definitional reason that does not
apply to it: exactly the failure this document argues against, pointed the
other way.

An ECHAM6 figure would be the natural third anchor, and the earlier draft of
this document quoted one (~0.62-0.65). That number could not be verified from
an accessible source and has been withdrawn rather than repeated;
[Stevens et al. 2013](https://doi.org/10.1002/jame.20015) is cited above for
the *formulation* of `aclcov`, not for a global mean.

## The #707 discontinuity: cover numbers do not cross it

PR #707 gave the 1M scheme ECHAM's post-microphysics cover write-back, which
the 2M scheme already had (#687). `mo_cloud.f90` section 8.4, "Corrections: avoid
negative cloud water/ice" (ICON lines 1129-1138), reads

```
zxlp1_d      = ccwmin - zxlp1                      ! from the RAW liquid
zxip1_d      = ccwmin - zxip1                      ! from the RAW ice
zxlp1        = FSEL(-zxlp1_d, zxlp1, 0)            ! zero a phase below ccwmin
zxip1        = FSEL(-zxip1_d, zxip1, 0)
zxlp1_d      = MAX(zxlp1_d, 0)                     ! liquid only
paclc(jl,jk) = FSEL(-(zxlp1_d*zxip1_d), paclc(jl,jk), 0)
```

`FSEL(a,b,c)` is `a >= 0 ? b : c`, so the last line clears the cover exactly
when `zxlp1_d * zxip1_d > 0`. Both `_d` terms are computed from the **raw,
pre-clamp** condensates and are not recomputed after the two `FSEL` clamps, so
`zxip1_d = ccwmin - qi_raw` can take any value — including anywhere in
`(0, ccwmin)` when the ice sits between zero and the threshold. The `MAX` line
is what makes the test well defined: it applies to the **liquid** term only,
leaving `zxlp1_d = max(ccwmin - qc_raw, 0) >= 0`, so a positive product
requires `zxlp1_d > 0` *and* `zxip1_d > 0` — that is, `qc < ccwmin` **and**
`qi < ccwmin`. Without that `MAX`, a cell with *both* phases above the
threshold would have two negative `_d` terms, a positive product, and would
have its cover wrongly cleared. `jcm/physics/clouds/echam_1m.py` implements
the resulting test directly as `(qc_end < ccwmin) & (qi_end < ccwmin)`.

The consequence is a change in what `clouds.cloud_fraction` *means*: it is now
the cover the step leaves behind, consistently under `cloud_scheme='1m'` and
`'2m'`. The #782 bisect measured the merge carrying it at **−0.066 of low
cloud for +0.15 W/m² at TOA** — cells holding under `ccwmin` (1e-7) of
condensate were radiatively negligible and simply stopped being counted. It is
not *purely* diagnostic (radiation, COSP, AeroCom and the JAM cloud-borne,
aqueous and wet-deposition terms all read that field), but for the radiative
balance it is a no-op. Later merges in the same segment hand back +0.011, which
is how the segment row in the table below nets to −0.055 for +1.2 W/m².

Because every cover definition is a reduction of that same field, **cover
numbers from before #707 are not comparable with numbers after it.** Part of
each member's apparent cloud loss against the #638 baseline is that boundary
rather than a change in the cloud.

The **size** of the shift, though, is known only for the column max. The
−0.066 above was measured on the area-weighted column max of the >680 hPa
band; nothing has measured what the same merge did to the max-random cover,
and that figure must not be carried across to it. `cloud_cover_radiation` is
further removed still: it reduces `effective_cloud_fraction`, which
independently zeroes every cell with `cloud_fraction <= 2*cld_frac_min`, so
the #707 write-back and that threshold bite on overlapping but different sets
of cells — a cell with, say, `cf = 0.5` and no condensate is cleared by the
write-back and not by the threshold. Its offset across #707 is unmeasured too.
The correct statement is that the boundary applies to all three and its
magnitude is measured for one.

The bisect below is 1M-only, which is where #707 added the write-back outright.
The 2M members moved at the same merge for a related reason — it also
consolidated `column_processes`, the 2M path already having the write-back —
and they moved further: against the #638 baseline the year runs record
1M −0.05/−0.04 against 2M −0.12/−0.12. So `echam-2m-t63`'s 0.60 → 0.48 is not
0.12 of lost cloud. The alarming reading it produced at the time — "only 0.08
of headroom above the 0.4 floor" — was also compounded by the gate scoring a
lower bound: on the definition and band this document settles on, that member
maps to ~0.54-0.59 against a 0.5 floor, so the headroom is ~0.04-0.09 and the
picture is unchanged in substance. The point is not that the member is safer than it
looked; it is that neither reading should have been taken from a quantity
whose definition was not pinned down.

## The #782 decomposition

The bisect ran 30-day T63 L47 `physics=echam` (1M) arms, one A100 each,
identical override sets and the same deterministic `init=jw init.rh=0.0` state.
Low cloud is the area-weighted column max within the >680 hPa band, selected by
pressure value on `pressure_full`. Every number in this section is a **column
max**, the definition in use when the bisect ran. A 30-day dry start is still
spinning up, so the absolute values sit far below a settled year (0.336 total
cover against 0.63 annual, both column max) and only the between-arm
differences at identical model time carry signal. Full tables and provenance:
issue #782.

| segment | low cloud | Δ low | Δ TOA | character |
|---|---|---|---|---|
| `1b7e7078` (pre-#690) | 0.437 | — | — | — |
| → `66b2d758` (#690) | 0.331 | **−0.106** | **+14.5 W/m²** | **physical** — the #661 cloud-base cluster |
| → `f5c8b354` (dev) | 0.276 | **−0.055** | +1.2 W/m² | **mostly bookkeeping** — dominated by the #707 write-back |

The TOA column is what separates them. The first segment is a faithful port of
ECHAM's `cubase` (the `klab` walk, `zlift = min(clip(thvsig·cbfac, cminbuoy,
cmaxbuoy), 1)`, stopping at the LCL, the moist buoyancy test) moving where
convection starts consuming the boundary layer: real cloud, and a real
radiative change. The second is dominated by the write-back above: cover that
was already radiatively absent ceasing to be counted. (Two radiation-side
merges inside that segment move TOA with essentially no change in cloud amount
— #719 by +4.6 W/m², #730 by −3.5 W/m² — which is the rest of its +1.2. "No
change" here means ≤0.005: #730 takes low cloud 0.271 → 0.276 and total cover
0.332 → 0.336.)

So there is no defect in either piece of physics, and nothing here argues for
reverting either. What it argues is that the closure constants were
compensating the old cloud-base error.

## What is not settled here

**No post-#707 year run has been scored on the online definition**: the two
control years in the side-by-side table predate the field and read through the
offline fallback, and the offset that maps the #638/#782 matrix onto the band
comes from two pre-#690 year runs and one 90-day post-#707 arm. The next
validation sweep saves the online cover on every member and should replace that
mapping with measurements; if it does, the band is worth revisiting with real
numbers rather than a rounded offset.

Whether the new cloud state is *right* is #682: retune the convective
trigger and closure against the corrected, no-longer-inflated CAPE, then
re-measure low cloud, LWP and SW CRE against CERES (CERES SW CRE ≈ −47 W/m²;
the #638 sweep recorded jcm 1M ≈ −98 and 2M ≈ −58). The open question is
whether −0.11 of low cloud moves 1M toward CERES and 2M away from it, which
would mean the real defect is the 1M microphysics. This document fixes the
measuring stick; it does not answer that.
