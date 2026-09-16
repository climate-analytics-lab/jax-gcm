# The cloud-cover gate: definition and provenance

*Issue #782, "re-derive the `cloud_cover` gate" half. The physics question that
issue also raised — whether the post-#690 cloud state is right — is #682's.*

Total cloud cover is not a property of a cloud-fraction profile on its own: it
is a profile plus an **overlap assumption**, and the three assumptions in
common use differ by more than the model changes being measured against them.
This document records which one the release-validation gate scores, why, and
what the resulting numbers may and may not be compared with.

## What the gate scores

`tools/release_validation/health.py` scores `cloud_cover` on
`jcm.analysis.total_cloud_cover` applied to `clouds.cloud_fraction`:
**ECHAM's own `aclcov`**, maximum-random overlap. Cloud in vertically
contiguous layers is treated as one cloud (maximum overlap); cloud separated by
clear air combines randomly. The clear-sky fraction accumulates down the column
as

```
c(k)   = clip(paclc(k), 0, 1)                      ! guard added here, not ECHAM's
zclcov = 1 - c(1)
DO k = 2, klev                                     ! DO 923; ICON: jks+1, klev
  zclcov = zclcov * (1 - max(c(k), c(k-1))) / (1 - min(c(k-1), zxsec))
END DO
aclcov = 1 - zclcov
```

which is a transcription of `mo_cloud.f90` section "10.2 Total cloud cover" —
ICON `atm_phy_echam/mo_cloud.f90` lines 1165-1182, identically ECHAM6
`mo_cloud.f90` lines 1359-1383. (ECHAM6's `paclcov` there is a *time
accumulation*, `paclcov + zdtime*zclcov`; its `aclcov_na` and ICON's
`paclcov` are the instantaneous value this function computes.)
`zxsec = 1 - 1e-12` is the Fortran's own guard against dividing by
`1 - paclc` at an overcast layer; the matching numerator is zero there, so
such a layer gives cover 1 rather than an infinity.

The `clip` is the one deliberate addition. `paclc` inside the model is
constructed in `[0, 1]`, but *saved* output can carry small out-of-range
excursions, and a `c > 1` would make the numerator negative and the
"clear-sky fraction" meaningless. It is applied out of place, so the caller's
Dataset is never modified. With it, no clip is needed on the way out: each
factor's numerator `1 - max(c_k, c_{k-1})` is at most its denominator
`1 - min(c_{k-1}, zxsec)` under either branch of the `min`, so every factor —
and hence the product, and hence the cover — lies in `[0, 1]` under IEEE
division.

Three properties make this the right quantity for a gate:

* **It is the reference model's definition.** The number is the same
  construction as ECHAM6's `aclcov` ([Stevens et al.
  2013](https://doi.org/10.1002/jame.20015)), and it is a total cover, which
  is the basis the satellite climatologies are quoted on — rather than a
  reduction peculiar to one model's post-processing.
* **It is deterministic and needs only `cloud_fraction`.** Every saved output,
  at every resolution and under every radiation scheme, scores identically —
  including re-scoring an archived run years later.
* **It is orientation-independent.** Cancelling the denominators leaves the
  clear-sky product as the adjacent-pair factors `1 - max(c_k, c_{k-1})`
  divided by the *interior* levels' `1 - c_k`, and both of those sets survive
  reversing the axis unchanged. Files written before #710 carry TOA-first
  interfaces (see
  [output_vertical_conventions](output_vertical_conventions.md)) and still
  score the same, so the function carries no orientation guard and needs none.
  The cancellation is exact, but the two orders agree only *to rounding*: the
  `min(c_{k-1}, zxsec)` guard is applied in loop order, so it caps a different
  denominator in the reversed column, and a profile holding cover within
  `zepsec` of 1 (`[0.2, 1 - 5e-13]`, say) differs by `O(zepsec)`. Over 200k
  random 47-level profiles (uniform, sparsified, and with overcast layers
  injected) the worst forward-versus-reversed difference was 1.1e-16.

Two further covers are **printed and not gated**:

| reported | what it is | why it is not the gate |
|---|---|---|
| `cloud_cover_colmax` | `clouds.cloud_fraction.max("level")` — the quantity the gate scored before | It assumes *every* layer overlaps maximally, so it is only a **lower bound**: two half-covered decks in different parts of the column read 0.5 where the sky is 0.75 covered. Retained so the #638 and #782 tables remain readable across this change. |
| `cloud_cover_radiation` | mean of `radiation.total_cloud_cover` — the fraction of McICA sub-columns with at least one cloudy layer, under the overlap rule and decorrelation length the flux solve actually uses (default: exponential, 2 km) | The radiation view. It is not the gate because it is a *sampled* quantity under a different overlap rule, absent from output written before the diagnostic existed (`b772ffec`, 2026-07-31) and identically zero under grey two-stream, which samples no sub-columns. An all-zero field is dropped rather than reported, so a scheme that does not produce it cannot look like a cloudless run; the printed NOTE says which of the two cases held. |

Random overlap, `1 - prod(1 - c_k)`, is the opposite bound: it ignores that a
physically continuous cloud spans several model layers, and so double-counts
its edges. It is neither scored nor reported.

### `cloud_cover_radiation` is a different measurement, not a cross-check

It is tempting to read the two printed covers against each other. They are not
comparable, for two independent reasons, and the measured gap is large:

* **Different field.** `radiation.total_cloud_cover` is built from
  `effective_cloud_fraction(cloud_fraction, eps=cld_frac_min)`
  (`jcm/physics/radiation/mcica.py`), which zeroes every cell with
  `cloud_fraction <= 2*cld_frac_min` so the sampler and the optics agree about
  which cells are empty. `cloud_cover` reduces the cloud fraction as saved.
* **Different time treatment.** Under `run.output_averages` the saved frame is
  the running mean over the output interval (`jcm/model.py`), and the
  release-validation launcher sets it unconditionally
  (`tools/release_validation/launch.py`: `run.output_averages=true`), as do
  the `longrun` and `pyses_year` run configs. So
  `radiation.total_cloud_cover` is a time mean of an *instantaneous* cover,
  while `cloud_cover` and `cloud_cover_colmax` are
  overlaps of a *time-mean* profile. The overlap product is non-linear, so
  those are different numbers: smoothing over the output interval moves each
  layer toward its time-mean fraction, and where cloud moved between layers
  during the interval that lowers the overlap-derived cover relative to the
  mean of the instantaneous covers. (The inequality is not general — a column
  that is overcast half the time and clear the other half gives the same
  answer either way — but it is the usual direction.)

On the 90-day 2M arm in the table below the two read **0.53 and 0.78**. A gap
of that size is expected; it is not evidence of a defect in either. It also
means the gated number is not the same quantity as the time-mean of the
instantaneous total cover that ECHAM6 and the satellite products report, and
on the one archived arm where both are available it is the smaller of the two
— worth remembering before reading a small offset against an anchor as a
model bias.

### Measured magnitudes

All three definitions, scored with this branch's
`jcm.analysis.total_cloud_cover` on the archived T63 L47 ECHAM+RRTMGP output
on the shared dev workstation. Area-weighted, and the reduction is taken
**after** the overlap product in every row (the product is non-linear). The
runs save 5-day means, so "last 40 saved frames" is the window
`--last-n 40` picks out of a 5-day-chunked run — the settled ~200 days.

| run | code point | window | column max | **max-random** | random | offset |
|---|---|---|---|---|---|---|
| 2M year<br>`clim_fixed_260703/v10_2m_prefix_year/echam-rrtmgp-2m_day*.nc` | dev @ 2026-07-04 | full year (73 frames, days 6-365) | 0.544 | **0.691** | 0.843 | +0.147 |
| " | " | settled (last 40 frames, days 171-365) | 0.553 | **0.701** | 0.847 | +0.148 |
| 1M year<br>`clim_fixed_260703/v7_1m_fullyear/echam-rrtmgp_day*.nc` | dev @ 2026-07-04 | full year | 0.555 | **0.679** | 0.815 | +0.124 |
| " | " | settled (last 40 frames) | 0.563 | **0.691** | 0.823 | +0.128 |
| 2M+JAM 90-day arm<br>`jam_scav_ab/abbase_260823_0100_day{30,60,90}.nc` | jcm `0ee92eaa`, **post-#707** | days 1-90 | 0.425 | **0.535** | 0.699 | +0.111 |
| " | " | days 61-90 | 0.417 | **0.527** | 0.703 | +0.110 |

Provenance and caveats, because they bound what the table can be used for:

* The two year runs are **pre-#690, pre-#707 and pre-#710**. Their directories
  carry no surviving `.hydra` snapshot, so the code point is fixed only by the
  run logs' date (2026-07-04) and the full resolved config they echo; #690
  merged 2026-08-21 and #707 2026-08-22. They are still the only archived
  T63 L47 ECHAM year runs on the box.
* The 2M year additionally carries a known defect — its own `README.txt`
  records it as the pre-orientation-fix arm, with inverted MACv2-SP shortwave
  aerosol and ozone. Its *absolute* cover is therefore not a climatology; its
  offset between definitions is what this table uses it for.
* The 90-day arm is the only archived **post-#707** ECHAM T63 L47 output here
  (verified: `git merge-base --is-ancestor 5ba96f7f 0ee92eaa`). It is a 90-day
  spin-up from a dry Jablonowski-Williamson start, so its absolute values are
  spin-up values, not climate. It is in the table for one purpose: to show the
  definitional offset survives the #690/#707 cloud-state change.
* `radiation.total_cloud_cover` is **absent** from both year runs (they
  predate `b772ffec`). The 90-day arm saves it: 0.757 over days 1-90 and 0.776
  over days 61-90 — against a max-random 0.535/0.527 on the same files, which
  is the gap the section above explains.

Two readings come out of this. First, the spread across definitions is
**~0.27-0.30**, larger than any model change the gate has ever been asked to
judge — which is the whole reason the definition has to be pinned down.
Second, the column max is **0.11 to 0.15 low** against max-random, and that
offset is stable across two microphysics schemes, two code points and
the #690/#707 boundary. It is the artefact the previous gate carried.

### Reconciling with the #638 and #782 tables

The numbers recorded in the #638 baseline sweep (2026-08-16) and the #782
comparison are **column maxima**, on different code points, at T63 *and* T106,
scored with `--last-n 40`. They are not directly comparable with the table
above, which is a different (earlier) code point at T63 only — the column max
there reads 0.54-0.56 against 0.60-0.70 in the matrix, and that difference is
model change plus resolution, not definition. What *is* transferable is the
definitional offset, so the matrix maps onto the new definition as:

| member | column max (#638 → post-#690/#707) | max-random (mapped) |
|---|---|---|
| `echam-1m-t63` | 0.68 → 0.63 | ~0.81 → ~0.74 |
| `echam-1m-t106` | 0.70 → 0.66 | ~0.83 → ~0.77 |
| `echam-2m-t63` | 0.60 → 0.48 | ~0.75 → ~0.59 |

using the offset measured on the matching scheme for the #638 column (1M
+0.128, 2M +0.148) and the post-#707 offset (+0.110) for the current one.
These are *mapped* values, not measurements: no post-#707 year run has been
scored on this definition, because none is archived. The next validation sweep
prints all three covers and replaces this mapping with measurements.

### The band

The gate band is **0.5-0.9**: the same width the gate has always had, placed
on this definition.

Placement is the whole question, because a band is calibrated against a
quantity. The previous band, 0.4-0.8, was calibrated by experience with column
maxima. Carrying it unchanged onto a quantity that reads 0.11-0.15 higher
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
* **The model.** Every member of the #638/#782 matrix maps into **0.59-0.83**,
  and the directly measured year runs sit at 0.68-0.70. The tightest margin is
  0.07, at the ceiling, and it is held by a *superseded* baseline; the current
  code point maps to 0.59-0.77, with 0.09 at the floor.

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
maps to ~0.59 against a 0.5 floor, so the headroom is ~0.09 and the picture is
unchanged in substance. The point is not that the member is safer than it
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

**No post-#707 year run has been scored on this definition**, because none is
archived — the offset that maps the #638/#782 matrix onto the new band comes
from two pre-#690 year runs and one 90-day post-#707 arm. The next validation
sweep prints all three covers on every member and should replace that mapping
with measurements; if it does, the band is worth revisiting with real numbers
rather than a rounded offset.

Whether the new cloud state is *right* is #682: retune the convective
trigger and closure against the corrected, no-longer-inflated CAPE, then
re-measure low cloud, LWP and SW CRE against CERES (CERES SW CRE ≈ −47 W/m²;
the #638 sweep recorded jcm 1M ≈ −98 and 2M ≈ −58). The open question is
whether −0.11 of low cloud moves 1M toward CERES and 2M away from it, which
would mean the real defect is the 1M microphysics. This document fixes the
measuring stick; it does not answer that.
