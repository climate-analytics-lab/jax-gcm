# The whole-model RCE testbed

`jcm.rce` configures a single column in radiative-convective equilibrium (RCE)
on the same `compute_tendencies` path the full model uses. Two columns run on
it. The minimal one, `rce_physics` (radiation and Betts-Miller convection),
has cheap grey-radiation tests of its machinery. The whole-model one runs
`echam_physics()` complete, and
`jcm/rce_test.py::TestRceWholeModelTiedtke` pins its equilibrium. This page
records that column's configuration, the reasons for its choices, and the
measurements its bounds come from.

## Configuration

| | |
|---|---|
| Stack | `echam_physics()`: surface exchange, TTE-TKE vertical diffusion, Sundqvist cover, Tiedtke-Nordeng convection, ECHAM 1M microphysics, RRTMGP; `AerosolFree` in place of MACv2-SP |
| Boundary | fixed SST 300 K; 0°N/0°E, ocean |
| Sun | perpetual (`steady_insolation`), solar constant 420 W/m² |
| Grid and step | L47 (`get_echam_levels`), `dt` 900 s, radiation every 7200 s (the `RadiationParameters` default) |
| Wind | prescribed uniform 5 m/s, not evolved (surface evaporation needs a wind) |
| Humidity | prognostic; the surface evaporation is the only source |
| Forcing | none: an RCE column has no mean vertical motion |
| Run | 80 days from `rce_initial_state` (mixed sub-cloud layer); averages over days 40-80 |

The solar constant of 420 W/m² delivers 431 W/m² at the top of the atmosphere
at this fixed sun (the measured `toa_sw_down`), 5 % above RCEMIP's 409.6 W/m²
(Wing et al. 2018). It is the value the testbed was built with and is left as
it is.

The aerosol is removed because MACv2-SP is a geographic climatology, and 0°N/0°E
sits in the Central African biomass-burning plume (AOD 0.33 at 550 nm, SSA
0.87), which alone absorbs about 100 W/m² of shortwave in the lower
troposphere. An RCE column means the clean, clear-air case.

## Why RRTMGP

The same stack with the idealized grey two-stream radiation
(`idealized_echam_physics`, solar constant 420 W/m²) does not reach a column
that rains what it evaporates. Over days 40-80:

| | grey | RRTMGP |
|---|---|---|
| net atmospheric radiative heating | −6.6 W/m² | −29.8 W/m² |
| surface latent heat flux | 14 W/m² | 29 W/m² |
| precipitation / evaporation | 0.64 | 0.99 |
| Tiedtke steps | 15 % | 100 % |
| lowest-level cloud cover | 1.0 | 0.0 |
| column water drift | +0.18 mm/d | +0.011 mm/d |

The grey atmosphere cools radiatively by too little (#883) to drive the
evaporation and the convection that remove its boundary-layer moisture, and
its lowest level fogs. ECHAM6.3's compiled `cover`, `cloud` and `cumastr`, fed
the grey column's states, make the same fog, so it is a property of that
column and not of the port. The grey scheme is an idealized scheme with no ECHAM counterpart
(`jcm/physics/echam/testing.py`); a test of ECHAM physics behaviour uses
`echam_physics()`.

Under RRTMGP the column is finite for the 200 days run. jax-rrtmgp 0.5.0
extends its temperature tables linearly outside their range (ECHAM's RRTMG
does the same) instead of mirroring them, so the 1 Pa top layer, which cools
from its 200 K start with nothing heating it, holds 160.2-160.4 K from about
day 40, at the tables' lower edge of 160 K. That temperature is a property of
a model top without dynamical or sponge heating, and the test pins only that it
stays above 155 K. ECHAM's upper sponge damps the non-zonal-mean part of the
temperature and does nothing in a column.

The CRE diagnostic (`radiation_compute_cre`) adds a clear-sky solve and
nothing else; every measurement below is identical with it on.

## The pins and where they come from

Seven trajectories were run for 120 days: the test's own and six that differ
from it only by 1e-4 K of random initial temperature noise (seeds 1-6). The
test's trajectory was also run for 200 days. Each bound is the extreme over the
trajectories and over the 40-day windows days 40-80 and 80-120 (and, for the
200-day run, 120-160 and 160-200), plus a margin of at least three times the
largest across-trajectory range of a window mean, rounded outward.

| quantity | measured extreme | across-trajectory range, days 40-80 | pinned |
|---|---|---|---|
| P / E | 0.989 .. 1.001 | 0.012 | > 0.95 |
| Tiedtke steps (`ktype > 0`) | 1.000 | 0 | > 0.90 |
| time-mean convective precipitation | 0.59 .. 0.67 mm/d | 0.02 | > 0 |
| column water drift | −0.0006 .. 0.0115 mm/d | 0.012 | abs < 0.05 mm/d |
| water budget residual / E | < 6.4e-5 | 1e-5 | < 1e-3 |
| rms of the time-mean heating | 0.008 .. 0.049 K/day | 0.002 | < 0.1 K/day |
| largest per-level std of the heating | 8.1 .. 8.8 K/day | 0.18 | < 10 K/day |
| model-top temperature, window minimum | 160.2 .. 160.4 K | 0.02 | > 155 K |
| TOA net (SW down − SW up − OLR) | 47.8 .. 52.3 W/m² | 2.5 | 40 .. 60 W/m² |
| TOA shortwave albedo | 0.463 .. 0.473 | 0.005 | 0.44 .. 0.50 |
| lowest-level cover, time mean | 0 .. 0.001 | 0 | < 0.01 |
| near-surface air temperature, window mean | 298.8 .. 298.9 K | 0.01 | 297.5 .. 300 K |
| near-surface specific humidity, window mean | 20.06 .. 20.28 g/kg | 0.06 | 18 .. 22 g/kg |

Provenance: `dev` at ee1e4e63, jax 0.10.2, jax-rrtmgp 0.5.0, float32, CPU. The 40-day window
means of the trajectories agree across the run-to-run noise; the slow part of
the evolution is what the second window measures. The bounds are meant to tell
the equilibrium from a different regime and not to track every re-tuning.
Seven of the pins fail on the grey column of the previous section (P/E
0.64, 15 % of steps, drift 0.18 mm/d, heating rms 0.11 K/day, TOA net
33 W/m², albedo 0.56, lowest-level cover 1.0); the budget residual, the
flicker, the model-top and the near-surface state do not separate the two.

The two lowest-level and near-surface bounds are wide because their
across-trajectory ranges are small; they separate this column from a fogged or
decoupled boundary layer and not from small changes of it. Three properties
are measured and not pinned.

- *The column is overcast (#920).* The maximum-random total cloud cover
  (`jcm.analysis.total_cloud_cover`) is 1.0 in every step, from a deck between
  270 and 630 hPa whose mean layer cover is 0.99-1.0 at 337-375 hPa, 0.66 at
  413 hPa, 0.92-1.0 at 453-581 hPa and 0.61 at 626 hPa (days 40-80; a thin
  layer at 208-237 hPa appears later, with mean cover 0.30 at 237 hPa over days
  160-200). The lowest model level stays clear.
- *Its hydrological cycle is weak.* P = E = 1.0 mm/d (a latent heat flux of
  29 W/m²) and the net atmospheric radiative cooling is 30 W/m², with a TOA
  shortwave albedo of 0.47. Why this single column goes overcast and is
  radiatively weak is the open question of #920.
- *The column is in balance but drifts slowly.* P/E, the water drift and the
  lowest-level cover are the same in all four windows of the 200-day run, while
  P and E fall together from 1.01 to 0.92 mm/d and the TOA net rises by about
  1.5 W/m² per window.

The flicker bound is a real change from the 8.0 K/day the grey column was
held to: under RRTMGP the mixed-phase deck at 495 hPa (256 K) scatters by
8.1-8.8 K/day in its heating, where the grey column's largest scatter was
5.4 K/day.

The reference states of ECHAM6.3's compiled `cucall`
(`jcm/data/test/echam_cumastr_reference`) were captured from the grey
configuration of this testbed. They are stored arguments and do not depend on
the column that runs now.

## Prescribed large-scale subsidence, tried and not adopted

A large-scale subsidence profile was tried as a sink for the boundary layer.
The profile is a half-sine in pressure, ω(p) = ω_max sin(π (p_s − p) / (p_s −
p_top)) for p_top ≤ p ≤ p_s and 0 above, with p_top = 150 hPa and a peak at
575 hPa (the pressure analogue of the half-sine in height of Warren, Singh &
Jakob 2020, *J. Adv. Model. Earth Syst.* 12, e2019MS001734, who impose it as
an ascent of 0-5 cm/s on a column at 300 K and advect potential temperature
and the water mixing ratio, not condensate). It acts as the standard SCM
vertical advection, first-order upwind: ∂θ/∂t = −ω ∂θ/∂p, which carries the
adiabatic compression of T exactly, and ∂q/∂t = −ω ∂q/∂p.

With no horizontal moisture convergence that form is a net column sink of
water, ∫ q ∂ω/∂p dp / g, and precipitation falls with it. The same column
with RRTMGP, S0 = 420 W/m², days 40-80, one trajectory each:

| ω_max (Pa/s) | P / E | P (mm/d) | E (mm/d) | Tiedtke steps |
|---|---|---|---|---|
| 0 | 0.989 | 0.998 | 1.010 | 1.00 |
| 0.0025 | 0.766 | 0.884 | 1.154 | 0.99 |
| 0.005 | 0.609 | 0.811 | 1.331 | 0.97 |
| 0.0075 | 0.477 | 0.702 | 1.472 | 0.98 |
| 0.01 | 0.347 | 0.539 | 1.552 | 1.00 |
| 0.02 | 0.103 | 0.188 | 1.828 | 0.79 |
| 0.03 | 0.015 | 0.033 | 2.122 | 0.98 |
| 0.05 | 0.000 | 0.000 | 1.189 | 0.20 |

At ω_max = 0.005 Pa/s the sink is 0.52 mm/d, 39 % of E, and the budget
E − P + S = dW/dt closes to 1.4e-4 mm/d. The column water falls with ω_max
(27 mm at day 80 for 0.03 Pa/s, against 54 mm). The grey stack responds the
same way (P/E 0.40 at 0.01 Pa/s, 0.16 at 0.03).

The testbed does not carry it. An RCE column has zero mean vertical motion by
definition, so a prescribed ω is a different configuration (a non-precipitating
subsiding regime) with an amplitude that sets its P/E; an amplitude
of 0.03 Pa/s leaves the column 0.015 of its evaporation as rain; and Warren et
al. impose the profile as an ascent, so no published amplitude exists for a
subsiding single-column RCE whose lateral moisture supply is absent. The prototype is not retained in the
repository.
