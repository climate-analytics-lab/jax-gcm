# The JAM aerosol retune: what was tuned, against what, and what is left

The v3 release was calibrated against observations once, on the ECHAM hosts
(`echam-1m-t63`, `echam-2m-t63`, `echam-jam-t63-l47`; SPEEDY is out of scope
and T106 and L95 were not swept). The outcome is **two changed aerosol defaults,
both on the JAM host** (the dust threshold scale and the Gong sea-salt scale) and
**one changed set of cloud-cover defaults at T63, read by all three hosts** (the
five parameters of the Sundqvist cover, Stage 2b), calibrated on the 2M host: it
passes every gate there, but it does not transfer cleanly to the JAM-2M host. Every other default,
including every convection, microphysics, DMS and wet-removal field, stays at
its ECHAM / ECHAM-HAM value. This page records the targets, the levers, the
stages, the evidence for the values, why the 1M and 2M convection and
microphysics defaults were not touched, and what the calibration cannot reach.

| default | value | where | basis |
|---|---|---|---|
| dust threshold scale | **0.379095663** | `NDUSCALE_JCM_T63_SCALE` in `jcm/physics/aerosol/jam/emissions/dust.py`; `physics.jam_dust_nduscale_scale` | best observed arm of Stage 2 (the surrogate's optimum is 0.3779); acts through the dust AOD |
| sea-salt emission scale | **2** | `SEASALT_SCALE_DEFAULT` in `jcm/physics/aerosol/jam/emissions/seasalt.py`; `physics.seasalt.scale` | upper edge of the swept range in both stages |
| Sundqvist cover at T63: `crt`, `crs`, `nex`, `csatsc`, `cinv` | **0.679016061, 0.9, 1.84856084, 0.948216414, 0.213005383** (ECHAM's T63 row: 0.75, 0.975, 2, 0.7, 0.25) | `JCM_CALIBRATED_COVER_T63` in `jcm/physics/clouds/echam_cloud_defaults.py`; all three hosts | interior arm of the Stage-2b sweep on the 2M host (below); acts through the cloud fraction; over-brightens and strips aerosol on JAM-2M |
| DMS flux scale, wet-removal scales, every convection and microphysics field | unchanged | | a weak lever (DMS); a burden guard (wet removal); structural, not parametric (convection, microphysics) |

The science statement of each aerosol value is in {doc}`../science/aerosol` and
of the cover values in {doc}`../science/clouds_microphysics`.

## Targets and the loss

The compared quantities are the monthly climatologies of the jcm-monitor
observation set (`obs/t63`), compared with the model's window mean on the same
grid:

| target | observation |
|---|---|
| SW and LW cloud radiative effect | CERES-EBAF Ed4.2.1, 2001-2020 |
| total cloud cover | ESA-CCI CLOUD v3.0 AVHRR-AMPM, 1997-2016 (the model side is `radiation.total_cloud_cover`, the cover the flux solve sees) |
| liquid water path | the same ESA-CCI product, ocean bands only |
| precipitation | GPCP v3.3, 2001-2020, as a zonal-mean pattern |
| total aerosol optical depth, 550 nm | ESA-CCI AEROSOL SU v4.21 (ATSR-2/AATSR), 1997-2011 |
| dust optical depth, 550 nm | ESA-CCI AEROSOL SLSTR SU v1.12, 2017-2022 |
| clear-sky OLR | CERES-EBAF; a monitor only |

Seven are scored and the clear-sky OLR is recorded with zero weight, which
makes eight compared fields and 26 loss terms for two windows. Each window is a
model integration from a warm state at a calendar position (January from
2000-12-31, July from 2001-06-29) and is evaluated after skipping its first
three days. A scored field contributes a **bias** term (the global mean) and a
**zonal** term (area-weighted 10-degree bands); precipitation and the liquid
water path contribute the zonal term only, and the precipitation bias is
reported, not penalised, because the global-mean precipitation bias is tied to
an evaporation deficit that no lever reaches.

**The error model.** The loss is half the sum of squared residuals, each
divided by the error of its block. A block's error is the larger of the
inter-annual standard deviation of that block's window-mean observation (from
per-year block series, so the spatial correlation inside a block is respected)
and a floor of a fraction of the field's global-mean magnitude: 5 % by default,
10 % for the LW CRE and 30 % for the liquid water path, which have no block
series. The floor keeps a block whose observed variability is nearly zero (a
dust-free band, polar night) from dividing a small misfit by a smaller error.
Every field carries equal total weight, split equally between its bias group
and its zonal group.

**The TOA gate.** The net top-of-atmosphere flux is not a target, because the
surface simplifications of the ECHAM hosts contaminate it. It enters as a
quadratic penalty on the amount by which the window-mean net flux departs from
the observed window-mean net flux by more than 10 W/m² (CERES is +8 W/m² in
January and -9 in July, so a gate on the flux itself would bind in one season).
A gate of that form binds only on a configuration that is far off, which is the
point: it stops a cloud lever from buying score with energy imbalance without
rewarding the compensation.

**Monitors.** Reported for every arm and never in the loss: the sulphate,
sea-salt and dust burdens, the sea-salt and dust lifetimes, the tropical 99.9th
percentile of precipitation (daily in the sweeps, 5-day means in the years)
relative to the control's, the global-mean
precipitation bias (land and ocean separately) and the clear-sky OLR bias.
Sulphate is read against a sanity band of 1.95-5.85 mg SO4/m² (the AeroCom
phase-I experiment-A mean, 3.90, plus or minus two standard deviations of the
model diversity; Textor et al. 2006). The band is a sanity check chosen for
this retune, not a range AeroCom publishes, and sulphate is deliberately not a
target.

**AeroCom reference values** (phase I, Textor et al. 2006, Table 10; the mean
and the median differ markedly for sea-salt emission):

| | dust | sea salt |
|---|---|---|
| emission, Tg/yr (mean / median) | 1840 / 1640 | 16 600 / 6280 |
| burden, mg/m² (mean / median) | 37.6 / 40.2 | 14.7 / 12.5 |
| residence time, d (mean / median) | 4.14 / 4.04 | 0.48 / 0.41 |

## Levers and stages

| lever | hosts | range (times the default) |
|---|---|---|
| Tiedtke `entrpen`, `cprcon`, `tau` | all | 0.5-3, 0.5-2, 0.5-2 |
| `ccraut`, `ccsaut` (`ccsacl` on the 1M) | 1M, 2M, JAM (`ccsaut` only) | 0.5-2 |
| dust threshold scale | JAM | 0.5-2 of 0.5 (absolute 0.25-1) |
| sea-salt scale | JAM | 0.5-2 |
| DMS flux scale | JAM | 0.5-1.25 |
| wet-removal scale: one lever over the stratiform in-cloud scale, the below-cloud scale and the convective in-plume scavenging scale together | JAM | 0.5-2 |

`cevapcu` is not a lever: it was inert in the trees of the early sweeps, and
once made live it was left out of the lever sets. Radiation, the land tile and
the surface emissivity are not levers of this retune, and neither are the
Sundqvist cloud-fraction parameters in Stages 1 to 3: ECHAM's per-truncation
values are fidelity, not tuning. The Sundqvist parameters are the levers of
Stage 2b, which has its own section below.

1. **Stage 1.** A 40-arm Sobol design over every lever of a host, 14-day arms
   scored on days 4-14, January and July. Output: a regression of each loss term
   and each physical response on the levers (standardised effects), the dropped
   levers, and a first optimum.
2. **Stage 2.** A 25-arm Gaussian-process expected-improvement search, 30-day
   arms, over the surviving levers of a host. On the JAM host the four aerosol
   levers only, with the convection and cloud levers frozen at their defaults:
   the loss on that host is dominated by the AOD terms, so the cloud levers are
   weakly constrained there (standardised effect on the loss 0.05-0.54 against
   3.21 for the dust scale in Stage 1), and the same cloud scheme is swept
   without aerosol on the 2M host.
3. **Stage 3.** A 365-day year, from the fixed-tree January state, of the
   selected configuration and of the control, scored on the Stage-2 windows of
   the year output and against the release gates.

Stage 2b is Stage 2 on the 2M host with the five Sundqvist cover parameters as
the only levers, and its own Stage-3 years.

The sweeps and years reported here ran on dev 519f18e8, which carries the
semi-Lagrangian specific-humidity conservation fix (cubic vertical
interpolation and a mass fixer: without it the dynamics created about
0.23 mm/day of water and lost about 11 W/m² of energy), the wet-removal door
and the live `cevapcu`. Sweeps made on the earlier tree were not carried over:
the control year's net TOA flux is +6.3 W/m² on the fixed tree against +11.6 on
the earlier one, and the controls' losses changed by -123 (JAM), -23 (2M) and
+38 (1M). Two kinds of evidence below come from the earlier tree and are marked
as such: the 1M's Stage 2, and the structural review of the sweep arms
(regional dust AOD, the liquid fraction against temperature, the sulphur
budget), which was made on earlier-tree sweeps and not repeated on the fixed
tree.

## The JAM host

**Stage 1** (8 levers, 40 arms; control loss 481.8, best arm 280.4, -41.8 %).
Standardised effect on the loss across each lever's range:

| dust scale | sea-salt scale | `entrpen` | wet removal | `cprcon` | `tau` | `ccsaut` | DMS |
|---|---|---|---|---|---|---|---|
| +3.21 | -1.36 | +0.54 | +0.32 | +0.12 | +0.10 | +0.05 | +0.02 |

The dust scale is the only lever that moves the dust AOD (global standardised
effect -3.2 in both windows, at most 0.3 for any other) and the sea-salt scale
is the largest lever on the total AOD (+2.9 / +2.6 in January / July, against
-1.1 / -2.0 for dust and -1.0 / -0.9 for wet removal). The Stage-1 best
arm is not adopted: it moves `tau` onto its lower bound and `entrpen` to 2.4
times its default, and raises the tropical 99.9th percentile of precipitation to
1.44 times the control's, for a gain the AOD terms (not the cloud terms) supply.

**Stage 2** (4 aerosol levers, 25 arms; control loss 476.8, best observed arm
298.0, -37.5 %). The Gaussian process puts the dust scale sharply (unit-cube
length scale 0.29) and the sea-salt scale loosely (2.3), and the DMS and
wet-removal scales flat (3.0, the bound). Five of the 25 arms are within 3 % of
the best loss, and the surrogate's optimum (predicted loss 302.9, 1.6 % above
the best observed) lies at the same corner of the box as the best observed arm
(dust 0.3779 against 0.3791, sea salt 2, wet removal 0.5, DMS 0.74 against
0.77). That arm has the dust scale
at 0.379095663 (0.758 times the old default), the sea-salt scale on its upper
edge (2), the DMS scale at 0.77 and the wet-removal scale on its lower edge
(0.5).

**Stage 3** years, 365 days, T63 L47, scored on the Stage-2 windows. The loss
is read from the year's 5-day means on those windows; AOD and dust AOD are
12-month means; burdens, lifetimes, sulphate and net TOA flux are the health
statistics over the last 195 days of the record (the settled part); the dust
emission is the mean over all 73 saves.

| year | levers | loss on the Stage-2 windows | AOD 550 (obs 0.145) | dust AOD (0.0213) | SO4, ion basis (band 1.95-5.85) | sea-salt burden mg/m² (14.7) | dust burden mg/m² (37.6) | dust emission Tg/yr | net TOA W/m² |
|---|---|---|---|---|---|---|---|---|---|
| `control_prefix` | defaults, earlier tree | 543.6 | 0.057 | 0.0032 | 5.0 | 6.2 | 5.6 | | +11.6 (gate fails) |
| `control_fixed` | defaults | 469.6 | 0.059 | 0.0032 | 5.26 | 6.6 | 5.3 | 563 | +6.3 |
| `jam_b` | dust 0.379, sea salt x2, DMS x0.77, wet removal x0.5 | 304.8 (-35 %) | 0.110 | 0.0100 | 6.4 (out) | 20.8 (42 % over the mean) | 18.7 | 1602 | +5.6 |
| `jam_bw1` | dust 0.379, sea salt x2, DMS x0.77 | 353.0 (-25 %) | 0.086 | 0.0098 | 4.9 | 12.5 | 16.3 | 1667 | +6.1 |
| `jam_amean` | dust 0.45, sea salt x1.96, DMS x0.95, wet removal x1.88 | 407.2 (-13 %) | 0.075 | 0.0050 | 4.6 | 12.0 | 8.5 | 858 | +6.3 |
| **`jam_rc`** | **release aerosol defaults: dust 0.379095663, sea salt x2, every other default** | **377.6 (-20 %)** | **0.082** | **0.0092** | **5.16** | **13.3** | **15.7** | **1629** | **+6.2** |
| `jam_rc_cloud` | `jam_rc` with the Stage-2b cover set (see Stage 2b) | 461.1 (-2 %) | 0.052 | 0.0082 | 4.15 | 9.2 | 14.6 | not scored (200-day window) | +2.1 (annual +3.2) |

`jam_amean` is the best arm whose two-window mean sulphate lies in the band: an
informational alternative that reaches the band with wet removal at 1.88 times
its default, near the upper bound of its range. `jam_bw1` is the DMS x0.77
neighbour of the release configuration `jam_rc`: the two differ only in DMS.

**The release configuration year.** `jam_rc` lowers the loss by 92 against the
control (377.6 against 469.6), 79 % of the 117 that `jam_bw1` gains (353.0),
because DMS at its default gives back a little of the AOD; the annual mean net
TOA flux is
+7.14 W/m² against the control's 7.41 (CERES +0.99), and the 40-save gate value
+6.22. Sulphate stays inside the sanity band, at 5.16 mg SO4/m² against 4.93
with DMS x0.77 and 5.26 for the control (6.175 on the tracer basis, which is
ammonium bisulphate; the band is on the ion basis). The window monitors read
sea salt 14.7 mg/m², dust 16.0 mg/m², sulphate 5.37 (ion basis) and a tropical
99.9th percentile of precipitation at 0.954 of the control's. The cloud and
radiation fields do not move from the control (total cloud cover 0.461 offline,
SW CRE -42.3, LW CRE 25.3, outgoing SW 92.9 W/m², precipitation 2.40 mm/day).
Against the drift limit of 0.002 /day that the health recipe applies, dust
(+0.00018), POA (-0.00184) and sea salt (+0.0002) pass, and BC (+0.00267) and
sulphate (+0.00250) fail; both of those are inside the 0.003 /day limit for
whole-year records, and the budget residual is 0.0005 against 5 %. The
`cloud_cover` gate fails at 0.46 against a band of 0.5-0.9, as it does for the
control.

**Why not `jam_b`.** Its extra AOD is bought with the wet-removal scale on its
lower bound. That raises sulphate above the sanity band (6.4 against 5.85 mg
SO4/m²), the sea-salt burden to 42 % above the AeroCom mean (14.7) and the sea-salt
lifetime from 0.6 to 1.0 d, which is the burden guard working as intended: the
loss does not see burdens, so a lever that moves a burden the loss ignores is
accepted only where the burden stays plausible. `jam_bw1` returns the
wet-removal scale to 1 and keeps the other three aerosol levers at their
Stage-2 optimum; `jam_rc` then returns DMS to its default as well.

**Why DMS stays at its default.** DMS at 0.77 is a weak lever on the loss
(standardised effect +0.02 in Stage 1 and +0.23 in Stage 2, flat in the
surrogate), and
the decision rule for this release is fewer changes, each with a strong
reason. `jam_rc` is `jam_bw1` with DMS returned to its default, and its sulphate
stays in the band.

**The sea-salt value is the edge of the range.** The optimum sat on the upper
bound of [0.5, 2] in Stage 1 (best arm 1.94) and in Stage 2, so the data
constrain the scale from below only. At 2 the burden is 13.3 mg/m², between the
AeroCom median (12.5) and mean (14.7), and the emission, 4148 Tg/yr, is below
the AeroCom median (6280). The lifetime does not change, so the burden is linear in the scale. The
range was not extended for this release, and nothing above 2 was run.

## The 1M and 2M defaults

Both were swept over the convection and microphysics levers and both are left
unchanged at those levers. (The Sundqvist cover that both read changes in
Stage 2b, below.)

**2M** (`echam-2m-t63`; Tiedtke `entrpen`, `cprcon`, `tau`, `ccraut`,
`ccsaut`). Stage 1: control 135.1, best 126.1 (-6.7 %). Stage 2: control 131.4,
best 116.6 (-11.2 %), with `entrpen`, `tau`, `ccraut` and `ccsaut` all on the
lower bound of 0.5 times the default and `cprcon` at 1.14. In the 30-day windows
of that arm the SW CRE goes from -43.2 / -41.4 to -46.8 / -45.0 W/m² (observed
-50.3 / -44.0), the liquid water path from 33.7 / 37.5 to 41.9 / 46.9 g/m², and
the total cover from 0.562 / 0.566 to 0.565 / 0.571 (observed 0.64 / 0.63):
brighter clouds from more water, not more cloud. Four of five levers on a bound
is a gradient that has not been bracketed, not an optimum. The cloud fraction is
set by the Sundqvist cover, which none of these levers touches, and that is the
case for Stage 2b.

**1M** (`echam-1m-t63`; the same Tiedtke levers, `ccraut`, `ccsaut`,
`ccsacl`). Stage 1 on the fixed tree: control 195.2, best 134.3 (-31.2 %) with
`cprcon` at 0.52 (the bound is 0.5) and `ccraut`, `ccsaut` at 0.61 and 0.64. The gain
is a global brightening: SW CRE from -37.6 / -34.9 to -44.3 / -41.7 W/m²,
cover from 0.503 / 0.489 to 0.524 / 0.517, liquid water path from 49.7 / 57.5
to 60.1 / 71.0 g/m². Every microphysics lever moves toward holding water
longer, which only helps because the control has too few clouds for its water.
Stage 2 ran on the earlier tree and, in the review of its best arm, gave the
same result at larger amplitude: liquid water path 62 / 80 g/m² against an
ESA-CCI 41 / 46 and 160-210 against about 55 in the Northern Hemisphere summer
midlatitudes, with land precipitation 15 % lower. Those figures are emulated
(a regression on the full-output Stage-1 arms, because the Stage-2 output
carries no liquid water path), good to about 30 %.

The loss carries a liquid-water-path term (ocean zonal) as a guard against this
compensation, and the optima still raise the global mean; the defaults stay
ECHAM's.

## Stage 2b: cloud fraction

Stages 1 and 2 showed that the cloud problem of the ECHAM hosts is the cloud
**fraction**, not its brightness: the 2M's radiation cover is 0.58 against 0.63
observed while its liquid water path is already above ESA-CCI, and the
convection and microphysics levers can only trade fraction against brightness
(above). The scheme that sets the fraction is the Sundqvist cover
(`SundqvistCloudFraction`), whose five parameters are ECHAM's own cover-tuning
knobs. They are read by the cover of the 1M, the 2M and the JAM-2M hosts alike.

**Levers.** Absolute ranges, because they are humidity thresholds and an
exponent, not scale factors. ECHAM's T63 row is the control.

| parameter | meaning | ECHAM T63 | swept range | adopted |
|---|---|---|---|---|
| `crt` | critical relative humidity aloft | 0.75 | 0.60-0.90 | **0.679016061** |
| `crs` | critical relative humidity at the surface | 0.975 | 0.90-0.995 | **0.9** (lower bound) |
| `nex` | exponent of the critical-RH profile | 2 | 1-4 | **1.84856084** |
| `csatsc` | stratocumulus saturation factor at an inversion | 0.7 | 0.50-1.00 | **0.948216414** |
| `cinv` | inversion stability threshold, fraction of g/cp | 0.25 | 0.10-0.50 | **0.213005383** |

`csecfrl` and `ccwmin` are not levers: they switch the ice phase and the
condensate threshold, they do not set the cover. ECHAM declares `nex` an
integer; the closure needs none (the profile
`crt + (crs - crt)·exp(1 - (p_s/p)^nex)` has a base of at least 1, so a real
exponent is continuous and differentiable, equals `crs` at the surface and
tends to `crt` aloft), and the sweep treated it as real.

**Sweep.** The 2M host (`echam-2m-t63`), Stage-2 recipe: 25 arms chosen by
Gaussian-process expected improvement, 30-day windows from 2000-12-31 and
2001-06-29 with the first three days skipped, the loss of the other stages
restricted to the fields a 2M host has (cloud cover, SW and LW CRE, liquid water
path, precipitation: 9 terms per window including the TOA gate; no AOD terms;
the clear-sky OLR is a zero-weight monitor). Control loss 130.9 (January 62.1, July 68.9); no arm failed. The
standardised effect on the loss across each range is +2.90 for `crs`, -0.98 for
`cinv`, +0.37 for `nex`, -0.18 for `csatsc` and +0.13 for `crt`; on the global
cover the two critical humidities are the levers (`crt` -3.0 / -2.6 and `crs`
-1.8 / -2.2 in January / July: lower is more cloud).

**The unconstrained optimum is on the edge of the box and is not adopted.** The
best of the 25 arms (loss 93.6, -28.5 %) and the surrogate's posterior-mean
optimum (predicted 93.3) sit at the same corner: `crs` 0.9 on its lower bound,
`nex` 3.96 (surrogate: 4) against an upper bound of 4, `cinv` 0.5 on its upper
bound (with `crt` 0.80 / 0.81 and `csatsc` 0.60 / 0.55 for the arm / the
surrogate). Four arms, the best included, lie within 3 % of that loss, all near
the same corner, and an optimum on a bound is where the search ran out of room,
not a value the data have located. The release rule is the one that
left the convection and microphysics defaults alone: a value on the edge of its
box is not adopted as a default. The adopted set is arm 9, the best of the
sweep's first 13 arms. Every one of the 25 arms with a lower loss has at least
two parameters within 5 % of their range of a bound; arm 9 has one (`crs`), so it
is the best arm that is interior in `crt`, `nex`, `csatsc` and `cinv` (arm 4,
loss 102.0, is interior in all five and within 0.2 of arm 9's loss; the
confirmation year was run on arm 9 and arm 4 has none):

| window mean (30 days) | observed | control | adopted | corner optimum |
|---|---|---|---|---|
| loss (January / July) | | 62.1 / 68.9 | 41.6 / 60.2 (101.8, -22.2 %) | 40.9 / 52.7 (93.6, -28.5 %) |
| total cover, radiation definition, January / July | 0.64 / 0.63 | 0.56 / 0.57 | 0.62 / 0.64 | 0.60 / 0.63 |
| SW CRE, W/m², January / July | -50.3 / -44.0 | -43.5 / -41.6 | -49.2 / -47.7 | -46.9 / -45.6 |
| LW CRE, W/m², January / July | 27.8 / 27.7 | 24.4 / 27.1 | 26.1 / 29.1 | 23.9 / 26.9 |
| liquid water path, g/m², January / July | | 34.1 / 37.7 | 41.4 / 47.1 | 39.1 / 44.4 |
| net TOA flux minus CERES, W/m², January / July | | 9.0 / 9.4 | 5.0 / 5.2 | 5.3 / 5.3 |

The sweep locates `crs`, `crt` and `nex` best (surrogate length scales 0.88,
0.77 and 0.98 of the unit cube) and `csatsc` and `cinv` weakly (length scales at
their bound of 3, standardised effects -0.18 and -0.98), so the adopted values of
those two are an arm's, not located optima; the adopted `csatsc` of 0.948 sits
close to 1, where the inversion enhancement vanishes.

The monitors do not move: the precipitation bias is -0.72 / -0.65 mm/day
against -0.71 / -0.67 for the control, the clear-sky OLR bias -7.6 / -9.2 W/m²
against -7.6 / -9.2, and the tropical 99.9th percentile of precipitation is
1.03 times the control's.

**One edge remains in the adopted set.** `crs` sits on its lower
bound of 0.90, below every value ECHAM6.3 uses (0.95 at T31 to 0.994 at T127),
so the data say it should be at least this low and do not locate it. Ranges below
0.90 were not run.

**Stage 3 years** (365 days, 2M host, T63 L47, from the fixed-tree January
state; annual means against the jcm-monitor climatologies; the loss is read on
the Stage-2 windows of the year output):

| year | cover parameters | gates | loss (control 131.9) | cover: gate / radiation (obs 0.63) | net TOA W/m² (CERES +1.0) | SW CRE (-45.7) | LW CRE (27.9) | reflected SW (99.0) | LWP g/m² (36.4) | precipitation mm/day (3.07) | p99.9 / control |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `2m_control` | ECHAM T63 | `cloud_cover` fails | 131.9 | 0.46 / 0.58 | +9.8 | -42.3 | 26.1 | 90.8 | 42.1 | 2.38 | 1.00 |
| **`2m_cloudi`** | **adopted** | all pass | **100.3 (-24 %)** | **0.51 / 0.65** | **+5.6** | **-48.3** | **28.2** | **96.9** | **49.6** | **2.40** | **0.97** |
| `2m_cloud` | corner optimum | all pass | 93.0 (-29 %) | 0.50 / 0.64 | +6.1 | -45.9 | 26.0 | 94.5 | 47.3 | 2.42 | 0.97 |
| `2m_a` | ECHAM T63 cover; the Stage-2 convection and microphysics corner, not adopted | `cloud_cover` fails | 113.0 | 0.46 / 0.59 | +8.1 | -45.6 | 27.8 | 94.1 | 51.3 | 2.38 | 0.88 |
| `jam_rc_cloud` | release aerosol defaults plus the adopted cover set, **JAM host** (own loss: control 469.6, `jam_rc` 377.6) | `cloud_cover` passes; BC and SO4 drift fail | 461.1 | 0.52 / 0.65 | +3.2 (`jam_rc` +7.1) | -49.0 | 27.3 | 98.6 | 57.1 (`jam_rc` 48.4) | 2.42 | 0.98 |

The adopted set raises the radiation cover by 0.07 and the cover the gate
measures from 0.46 to 0.51 (the gate's floor is 0.5, so it passes by 0.01; see
{doc}`cloud_cover_gate` for the two definitions), halves the TOA bias (+9.8 to
+5.6 W/m², CERES +1.0), brings the reflected SW to within 2.1 W/m² of CERES and
the LW CRE to within 0.3 W/m². What it costs: the SW CRE is 2.6 W/m² too strong
(-48.3 against -45.7, from 3.5 too weak), the liquid water path rises by 7.5 g/m²
over a control that is already above ESA-CCI, the near-surface temperature falls
by 0.2 K, and the all-sky OLR bias grows from -6.7 to -8.7 W/m² because the LW
CRE rises onto its observation while the clear-sky OLR, which is 7.5 to 9.5 W/m²
low (January / July) on every host, does not move. The corner optimum is the better fit by the
loss (93.0 against 100.3) and has the better SW CRE (-45.9) but a LW CRE 1.9
W/m² low, and its cover passes the gate by 0.003; it is not adopted because its
parameters sit on three bounds.

**What did not change:** precipitation (-0.67 mm/day against -0.69 for the
control), the clear-sky OLR bias (-7.5 / -9.2 W/m² against -7.6 / -9.2 in the
January / July windows), the tropical precipitation extremes (the 99.9th
percentile of 5-day means is 0.97 times the control's), and every convection
and microphysics default, which stay ECHAM's. The `jam_rc_cloud` row is the
JAM host and is discussed below.

**The set does not transfer cleanly to the JAM-2M host.** `jam_rc_cloud` is the
JAM-2M host with the release aerosol defaults and the adopted cover set (365
days, from the fixed January state), against `jam_rc`, the same host with the
aerosol defaults and ECHAM's cover parameters. On the 2M host the set passes
every gate and takes 4 W/m² off the TOA bias at the price of 7.5 g/m² of liquid
water path. On the JAM-2M host, whose droplet number comes from the interactive
aerosol, it over-brightens, and the extra cloud strips the aerosol the other two
defaults were fitted to deliver:

| JAM host | window loss (control 469.6) | cover: gate / radiation (obs 0.63) | net TOA W/m² (CERES +1.0) | SW CRE (-45.7) | LW CRE (27.9) | reflected SW (99.0) | LWP g/m² (36.4) | AOD 550 (0.145) | sea-salt burden mg/m² | sea-salt lifetime d | SO4 AeroCom basis |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `control_fixed` | 469.6 (January 164.4, July 305.2) | 0.46 / 0.58 | +7.4 | -42.8 | 25.5 | 92.6 | 48.8 | 0.059 | 6.6 | 0.67 | 5.26 |
| `jam_rc` | 377.6 (136.5, 241.1) | 0.46 / 0.59 | +7.1 | -42.3 | 25.3 | 92.9 | 48.4 | 0.082 | 13.3 | 0.65 | 5.16 |
| `jam_rc_cloud` | **461.1** (130.4, 330.7) | 0.52 / 0.65 | +3.2 | -49.0 | 27.3 | 98.6 | 57.1 | 0.052 | 9.2 | 0.40 | 4.15 |

The cover gate passes (0.52 on the offline overlap, 0.65 as the radiation sees
it), the annual TOA flux falls by 4 W/m² and the reflected SW and the LW CRE sit
on CERES, but the SW CRE is 3.3 W/m² too strong, the liquid water path is 23 g/m²
above ESA-CCI (8.7 above `jam_rc`), and the larger cloud fraction increases wet
removal: the sea-salt burden falls from 13.3 to 9.2 mg/m² (lifetime 0.65 to
0.40 d), the total AOD from 0.082 to 0.052 (below the control's 0.059) and the
dust burden from 15.7 to 14.6 mg/m²; sulphate stays in the band. The loss of the
eight-target recipe is therefore worse than the aerosol-only configuration's:
461.1 against 377.6, with January better (130.4 against 136.5) and July much
worse (330.7 against 241.1; the July AOD bias term alone rises by 40). The drifts
of the record are +0.0032 /day for black carbon and +0.0026 /day for sulphate:
both fail the 0.002 /day limit of the documented recipe, and black carbon is also
above the 0.003 /day limit for whole-year records, while dust and POA pass; the
annual dust emission is not scored on this record (its window spans 200 days).

The cover set is tuned on a host whose droplet number is prescribed, and the
aerosol scales (dust threshold, sea-salt) were fitted on ECHAM's cover, so the
combination is a different configuration from either calibration. Adopting the
set on the JAM host would need a joint cloud-and-aerosol calibration, which this
change does not contain.

## What the calibration cannot reach

The remainder is structural: no lever of this retune closes it, and several of
the structural findings stand in the way of reading any further parametric gain
as real. Items marked "earlier tree" come from the structural review of the
earlier-tree sweeps.

- **Clear-sky OLR is 8-10 W/m² below CERES on every host** (January -7.9,
  July -9.6 W/m² in `jam_rc`; -7.6 and -9.3 on the 2M control, -8.6 and -9.6
  on the 1M control) and no lever moves it by more than 1 W/m². Part of the gap
  is CERES's clear-sky sampling; the split between sampling and model was not
  made.
- **Cloud cover.** With ECHAM's cover parameters, 0.58-0.59 as the radiation
  sees it (the mean of `radiation.total_cloud_cover`) and 0.46 on the offline
  maximum-random overlap of the daily-mean profile (the `cloud_cover` release
  gate, whose band is 0.5-0.9: it fails on the control year as on the JAM
  aerosol-only year), against 0.63 observed; Stage 2b's set closes this on the
  2M host (0.65 and 0.51) and passes the gate on the JAM-2M host (0.65 and 0.52),
  where it also over-brightens (above). The Southern Ocean (45-65S) is too clear on every host
  (0.58-0.69 against 0.85-0.88; earlier tree, ECHAM's cover parameters; not
  re-measured with the Stage-2b set). See {doc}`cloud_cover_gate` for the
  definitions.
- **Liquid water path compensates for the missing cover** on the 1M and 2M, as
  above, and is still high with Stage 2b's set (49.6 g/m² against 42.1 for the
  2M control and 36.4 observed); the liquid fraction reaches one half at -6 to
  -11 °C on every host, where CALIPSO-type estimates put it near -20 °C, and no
  lever moves that by more than 1-2 K (earlier tree).
- **Dust.** The lifetime is 1.8 d against AeroCom's 4.1, which is deposition and
  size, and the regional source balance is wrong: in the best arm of the
  earlier-tree sweeps the modelled-to-observed regional dust AOD was 0.84
  (January) and 0.51 (July) over the Sahara and Sahel and 1.65 / 1.88 over
  Arabia, 3.9 / 1.1 over Asia and 4.1 / 0.38 over Australia. The shipped dust
  AOD is 0.43 of the observed; the threshold scale cannot repair a lifetime or a
  regional ratio.
- **Sea salt.** The burden (13.3 mg/m²) lies between the AeroCom median (12.5)
  and mean (14.7), though the emission (4148 Tg/yr) is below the AeroCom median
  (6280); the coarse-mode extinction per unit mass is at the low end (about
  1.7 m²/g), so the AOD deficit (0.082 against 0.145) is not sea salt's alone.
- **Sulphate** is set by its removal and its DMS source: the standardised
  effects on the January / July burden are -2.3 / -2.6 for the wet-removal
  scale and +1.8 / +0.9 for DMS in Stage 2, against -0.95 / -0.65 for dust and
  +0.4 / +0.3 for sea salt, and -2.8 / -2.6 for `entrpen` in Stage 1. The
  chemistry has no SO2 deposition, so essentially all emitted sulphur becomes
  sulphate (a conversion of 0.96-1.04 against about 0.6-0.7 in comparable
  models; earlier tree). There is no volcanic or SOA source.
- **Precipitation** is 0.67 mm/day below GPCP in the year (2.40 against 3.07),
  0.9-1.1 mm/day over the ocean and within 0.4 over land in the windows, with the
  latent heat flux about 16 W/m² below MERRA-2. The deficit tracks evaporation,
  not convection, and the levers can only change it by drying the atmosphere.

## Scope

- The dust scale applies at T63 with the `ndust = 4` preset only; every other
  resolution keeps HAM's 0.86, so T106 and the cubed sphere do not inherit it.
- The sea-salt scale has no resolution switch and applies at every resolution
  the JAM chain is composed at. It was calibrated at T63 L47 only. The L95 JAM
  member inherits both L47 values: the dust scale because it is a T63 value, and
  the sea-salt scale because it has no switch.
- The cover set applies at T63. It was swept on the 2M host at L47 and
  confirmed there in a 365-day year; the JAM-2M host's year, `jam_rc_cloud`,
  passes the cover gate but is worse than the aerosol-only configuration (above),
  and the 1M host reads the set with no calibration or year of its own (#1014). T106 interpolates linearly in the truncation number between the T63 row
  and ECHAM's T127 row (an untuned blend: `crs` 0.963, `crt` 0.727, `csatsc`
  0.781, `cinv` 0.238, and `nex` 2, the nearer truncation's integer), and a
  grid with no spectral truncation (the cubed sphere) takes the T63 row; neither
  was swept (#1014). See {doc}`resolution_defaults`.
- The release-matrix regression bands of the ECHAM members describe the cloud and
  aerosol climate of the previous defaults and are regenerated against the release
  candidate, together with their init states.
- The release gate on the annual dust emission, `DUST_EMISSION_TG_PER_YR`, is
  400-2600 Tg/yr: the calibrated year emits 1629, 2.5 times the converted
  parent budget (642) and of the order of the AeroCom medians (1640; 1123 in the
  15-model dust intercomparison). The parent model's last-glacial-maximum run
  (5159 Tg/yr in HAM's size window) is about 2700 in this one, only 4 % above
  the upper edge. The gate is not scored by the documented
  `health.py --last-n 40` recipe, whose 200-day window is shorter than the
  300 days it needs (#1010).

## Reproducing the numbers

Each number above is tied to a run label. The sweeps are the 40-arm Stage-1
and 25-arm Stage-2 ledgers of each host on dev 519f18e8 (JEM-Cal recipe v3.1;
observations from jcm-monitor); the years are the five 365-day T63 L47 runs
named in the JAM table, scored with `tools/release_validation/health.py` and the
monitor's release recipe. Stage 2b is the 25-arm ledger of the 2M host on the
same tree with the five `cloud.*` levers, and the years `2m_control`,
`2m_cloudi`, `2m_cloud`, `2m_a` and `jam_rc_cloud`, on the same tree and recipe.
Shipped values are pinned by `dust_test.py`, `seasalt_test.py`,
`echam_terms_test.py` and `runners_test.py`, and the cover set by
`echam_cloud_defaults_test.py`, `parameters_test.py` and `runners_test.py`.
