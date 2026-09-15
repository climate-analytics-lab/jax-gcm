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
zclcov = 1 - paclc(1)
zclcov = zclcov * (1 - max(paclc(k), paclc(k-1))) / (1 - min(paclc(k-1), zxsec))
aclcov = 1 - zclcov
```

which is a transcription of `mo_cloud.f90` section "10.2 Total cloud cover" —
ICON `atm_phy_echam/mo_cloud.f90` lines 1165-1182, identically ECHAM6
`mo_cloud.f90` lines 1359-1381. `zxsec = 1 - 1e-12` is the Fortran's own guard
against dividing by `1 - paclc` at an overcast layer; the matching numerator is
zero there, so such a layer gives cover 1 rather than an infinity.

Three properties make this the right quantity for a gate:

* **It is the reference model's definition.** The number is directly
  comparable with ECHAM6's `aclcov` and with the satellite products, which are
  quoted on a total-cover basis rather than on any single-model reduction.
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

Two further covers are **printed and not gated**:

| reported | what it is | why it is not the gate |
|---|---|---|
| `cloud_cover_colmax` | `clouds.cloud_fraction.max("level")` — the quantity the gate scored before | It assumes *every* layer overlaps maximally, so it is only a **lower bound**: two half-covered decks in different parts of the column read 0.5 where the sky is 0.75 covered. Retained so the #638 and #782 tables remain readable across this change. |
| `cloud_cover_radiation` | mean of `radiation.total_cloud_cover` — the fraction of McICA sub-columns with at least one cloudy layer, under the overlap rule and decorrelation length the flux solve actually uses (default: exponential, 2 km) | The radiation view, and the cross-check that matters for a CRE argument. It is not the gate because it is a *sampled* quantity under a different overlap rule, absent from output written before the diagnostic existed (`b772ffec`, 2026-07-31) and identically zero under grey two-stream, which samples no sub-columns. An all-zero field is dropped rather than reported, so a scheme that does not produce it cannot look like a cloudless run. |

Random overlap, `1 - prod(1 - c_k)`, is the opposite bound: it ignores that a
physically continuous cloud spans several model layers, and so double-counts
its edges. It is neither scored nor reported.

### Indicative magnitudes

Measured on two July-2026 T63 L47 ECHAM+RRTMGP year runs on the shared dev
workstation — **last chunk only, area-weighted, pre-#690 code**, so these are
magnitudes for calibrating the definitions against each other and **not**
validation numbers for any current code point:

| definition | 2M member | 1M member |
|---|---|---|
| column max (`cloud_cover_colmax`) | 0.546 | 0.559 |
| **max-random (`cloud_cover`, ECHAM `aclcov`)** | **0.682** | **0.665** |
| random overlap | 0.835 | 0.798 |

The spread across definitions is ~0.29 — larger than any model change the gate
has ever been asked to judge. The max-random row is the one that lands on
ECHAM6's climatological total cover (~0.62-0.65) and near the satellite
estimates (ISCCP/MODIS ~0.66-0.67, CALIPSO-GOCCP ~0.70); the column max sits
~0.12 low, which is the size of the artefact the previous gate carried.

The gate band stays at **0.4-0.8**. It brackets those anchors with spin-up
slack, and it is deliberately not tightened around them: no post-#707 year has
yet been scored on this definition, so there is nothing to tighten it against.
The next validation sweep prints all three covers and supplies the values.

## The #707 discontinuity: cover numbers do not cross it

#707 gave the 1M scheme ECHAM's post-microphysics cover write-back, which the
2M scheme already had (#687). `mo_cloud.f90` section 8.4, "Corrections: avoid
negative cloud water/ice" (ICON lines 1129-1138), reads

```
zxlp1_d      = ccwmin - zxlp1                      ! end-of-step liquid
zxip1_d      = ccwmin - zxip1                      ! end-of-step ice
zxlp1        = FSEL(-zxlp1_d, zxlp1, 0)            ! zero a phase below ccwmin
zxip1        = FSEL(-zxip1_d, zxip1, 0)
zxlp1_d      = MAX(zxlp1_d, 0)
paclc(jl,jk) = FSEL(-(zxlp1_d*zxip1_d), paclc(jl,jk), 0)
```

`FSEL(a,b,c)` is `a >= 0 ? b : c`. After the two phase clamps each condensate
is either exactly 0 or at least `ccwmin`, so each `_d` term is either `ccwmin`
or non-positive, and their product is positive **iff both phases ended the step
below the threshold** — which is when, and only when, the last line clears the
cover. `jcm/physics/clouds/echam_1m.py` implements that as
`(qc_end < ccwmin) & (qi_end < ccwmin)`.

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
numbers from before #707 are not comparable with numbers after it — on the
column max, on max-random, or on the McICA cover alike.** Part of each member's
apparent cloud loss against the #638 baseline is that boundary rather than a
change in the cloud.

The bisect below is 1M-only, which is where #707 added the write-back outright.
The 2M members moved at the same merge for a related reason — it also
consolidated `column_processes`, the 2M path already having the write-back —
and they moved further: against the #638 baseline the year runs record
1M −0.05/−0.04 against 2M −0.12/−0.12. So `echam-2m-t63`'s 0.60 → 0.48 is not
0.12 of lost cloud, and the "only 0.08 of headroom above the 0.4 floor" reading
of it is compounded by the gate having scored a lower bound.

## The #782 decomposition

The bisect ran 30-day T63 L47 `physics=echam` (1M) arms, one A100 each,
identical override sets and the same deterministic `init=jw init.rh=0.0` state.
Low cloud is the area-weighted column max within the >680 hPa band, selected by
pressure value on `pressure_full`. A 30-day dry start is still spinning up, so
the absolute values sit far below a settled year (0.336 total cover against
0.63 annual) and only the between-arm differences at identical model time carry
signal. Full tables and provenance: issue #782.

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
merges inside that segment move TOA without touching cloud amount at all —
#719 by +4.6 W/m², #730 by −3.5 W/m² — which is the rest of its +1.2.)

So there is no defect in either piece of physics, and nothing here argues for
reverting either. What it argues is that the closure constants were
compensating the old cloud-base error.

## What is not settled here

Whether the new cloud state is *right* is #682: retune the convective
trigger and closure against the corrected, no-longer-inflated CAPE, then
re-measure low cloud, LWP and SW CRE against CERES (CERES SW CRE ≈ −47 W/m²;
the #638 sweep recorded jcm 1M ≈ −98 and 2M ≈ −58). The open question is
whether −0.11 of low cloud moves 1M toward CERES and 2M away from it, which
would mean the real defect is the 1M microphysics. This document fixes the
measuring stick; it does not answer that.
