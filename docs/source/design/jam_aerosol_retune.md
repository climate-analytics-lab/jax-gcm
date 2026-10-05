# The JAM aerosol retune: what was tuned, against what, and what is left

The v3 release was calibrated against observations once, on the ECHAM hosts
(`echam-1m-t63`, `echam-2m-t63`, `echam-jam-t63-l47`; SPEEDY is out of scope
and T106 and L95 were not swept). The outcome is **two changed defaults, both
on the JAM host**: the dust threshold scale and the Gong sea-salt scale. Every other
default, including every cloud, convection, DMS and wet-removal field, stays at
its ECHAM / ECHAM-HAM value. This page records the targets, the levers, the
stages, the evidence for the two values, why the 1M and 2M defaults were not
touched, and what the calibration cannot reach.

| default | value | where | basis |
|---|---|---|---|
| dust threshold scale | **0.379095663** | `NDUSCALE_JCM_T63_SCALE` in `jcm/physics/aerosol/jam/emissions/dust.py`; `physics.jam_dust_nduscale_scale` | best observed arm of Stage 2 (the surrogate's optimum is 0.3779); acts through the dust AOD |
| sea-salt emission scale | **2** | `SEASALT_SCALE_DEFAULT` in `jcm/physics/aerosol/jam/emissions/seasalt.py`; `physics.seasalt.scale` | upper edge of the swept range in both stages |
| DMS flux scale, wet-removal scales, every cloud and convection field | unchanged | | a weak lever (DMS); a burden guard (wet removal); structural, not parametric (cloud, convection) |

The science statement of each value is in {doc}`../science/aerosol`.

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
once made live it was left out of the lever sets. The cloud-cover critical
relative humidity and the other Sundqvist cloud-fraction parameters, radiation,
the land tile and the surface emissivity are not levers of this retune: ECHAM's
per-truncation values are fidelity, not tuning. The Sundqvist parameters are the
subject of a separate study (Stage 2b) that this page does not report.

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
| **`jam_rc`** | **release configuration: dust 0.379095663, sea salt x2, every other default** | **PENDING: filled by the release coordinator from the finished year** | | | | | | | |

`jam_amean` is the best arm whose two-window mean sulphate lies in the band: an
informational alternative that reaches the band with wet removal at 1.88 times
its default, near the upper bound of its range.

**Why not `jam_b`.** Its extra AOD is bought with the wet-removal scale on its
lower bound. That raises sulphate above the sanity band (6.4 against 5.85 mg
SO4/m²), the sea-salt burden to 42 % above the AeroCom mean (14.7) and the sea-salt
lifetime from 0.6 to 1.0 d, which is the burden guard working as intended: the
loss does not see burdens, so a lever that moves a burden the loss ignores is
accepted only where the burden stays plausible. `jam_bw1` returns the
wet-removal scale to 1 and keeps the other three aerosol levers at their
Stage-2 optimum.

**Why DMS stays at its default.** DMS at 0.77 is a weak lever on the loss
(standardised effect +0.02 in Stage 1 and +0.23 in Stage 2, flat in the
surrogate), and
the decision rule for this release is fewer changes, each with a strong
reason. `jam_rc` is `jam_bw1` with DMS returned to its default.

**The sea-salt value is the edge of the range.** The optimum sat on the upper
bound of [0.5, 2] in Stage 1 (best arm 1.94) and in Stage 2, so the data
constrain the scale from below only. At 2 the burden is 12.5 mg/m², the AeroCom
median (mean 14.7), and the emission, 4042 Tg/yr, is below the AeroCom median
(6280). The lifetime does not change, so the burden is linear in the scale. The
range was not extended for this release, and nothing above 2 was run.

## The 1M and 2M defaults

Both were swept and both are left unchanged.

**2M** (`echam-2m-t63`; Tiedtke `entrpen`, `cprcon`, `tau`, `ccraut`,
`ccsaut`). Stage 1: control 135.1, best 126.1 (-6.7 %). Stage 2: control 131.4,
best 116.6 (-11.2 %), with `entrpen`, `tau`, `ccraut` and `ccsaut` all on the
lower bound of 0.5 times the default and `cprcon` at 1.14. In the 30-day windows
of that arm the SW CRE goes from -43.2 / -41.4 to -46.8 / -45.0 W/m² (observed
-50.3 / -44.0), the liquid water path from 33.7 / 37.5 to 41.9 / 46.9 g/m², and
the total cover from 0.562 / 0.566 to 0.565 / 0.571 (observed 0.64 / 0.63):
brighter clouds from more water, not more cloud. Four of five levers on a bound
is a gradient that has not been bracketed, not an optimum.

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

## What the calibration cannot reach

The remainder is structural: no lever of this retune closes it, and several of
the structural findings stand in the way of reading any further parametric gain
as real. Items marked "earlier tree" come from the structural review of the
earlier-tree sweeps.

- **Clear-sky OLR is 8-10 W/m² below CERES on every host** (January -8.0,
  July -9.7 W/m² in `jam_bw1`; -7.6 and -9.3 on the 2M control, -8.6 and -9.6
  on the 1M control) and no lever moves it by more than 1 W/m². Part of the gap
  is CERES's clear-sky sampling; the split between sampling and model was not
  made.
- **Cloud cover.** 0.58-0.59 as the radiation sees it (the mean of
  `radiation.total_cloud_cover`) and 0.46 on the offline maximum-random overlap
  of the daily-mean profile (the `cloud_cover` release gate, whose band is
  0.5-0.9: it fails on the control year as on the calibrated one), against 0.63
  observed. The Southern Ocean (45-65S) is too clear on every host (0.58-0.69
  against 0.85-0.88; earlier tree). See {doc}`cloud_cover_gate` for the definitions.
- **Liquid water path compensates for the missing cover** on the 1M and 2M, as
  above; the liquid fraction reaches one half at -6 to -11 °C on every host,
  where CALIPSO-type estimates put it near -20 °C, and no lever moves that by
  more than 1-2 K (earlier tree).
- **Dust.** The lifetime is 1.8 d against AeroCom's 4.1, which is deposition and
  size, and the regional source balance is wrong: in the best arm of the
  earlier-tree sweeps the modelled-to-observed regional dust AOD was 0.84
  (January) and 0.51 (July) over the Sahara and Sahel and 1.65 / 1.88 over
  Arabia, 3.9 / 1.1 over Asia and 4.1 / 0.38 over Australia. The shipped dust
  AOD is 0.46 of the observed; the threshold scale cannot repair a lifetime or a
  regional ratio.
- **Sea salt.** The burden (12.5 mg/m²) is the AeroCom median, though the
  emission (4042 Tg/yr) is below the AeroCom median (6280); the coarse-mode
  extinction per unit mass is at the low end (about 1.7 m²/g), so the AOD
  deficit (0.086 against 0.145) is not sea salt's alone.
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
- The release-matrix regression bands of the JAM members describe the aerosol
  climate of the previous defaults and are regenerated against the release
  candidate, together with their init states.
- The release gate on the annual dust emission, `DUST_EMISSION_TG_PER_YR`, is
  400-2600 Tg/yr: the calibrated year emits 1667, 2.6 times the converted
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
named in the table, scored with `tools/release_validation/health.py` and the
monitor's release recipe. Shipped values are pinned by `dust_test.py`,
`seasalt_test.py`, `echam_terms_test.py` and `runners_test.py`.
