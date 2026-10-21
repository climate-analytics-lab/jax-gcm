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
that rains what it evaporates. Over days 40-80, both measured with the
simplified vertical-diffusion closure jcm had before ECHAM6.3's (#996; the
grey column has not been re-run with it):

| | grey | RRTMGP |
|---|---|---|
| net atmospheric radiative heating | −10.8 W/m² | −33.2 W/m² |
| surface latent heat flux | 16 W/m² | 35 W/m² |
| precipitation / evaporation | 0.65 | 0.96 |
| Tiedtke steps | 28 % | 98 % |
| lowest-level cloud cover | 1.0 | 0.0 |
| column water drift | +0.19 mm/d | +0.048 mm/d |

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
from its 200 K start with nothing heating it, holds 160.1 K from about
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
| P / E | 0.946 .. 0.999 | 0.011 | > 0.91 |
| Tiedtke steps (`ktype > 0`) | 0.990 .. 0.997 | 0.001 | > 0.90 |
| time-mean convective precipitation | 0.43 .. 0.49 mm/d | 0.011 | > 0 |
| column water drift | 0.0006 .. 0.043 mm/d | 0.009 | abs < 0.1 mm/d |
| water budget residual / E | < 1.4e-4 | 6e-6 | < 1e-3 |
| rms of the time-mean heating | 0.0014 .. 0.054 K/day | 0.001 | < 0.1 K/day |
| largest per-level std of the heating | 3.3 .. 3.7 K/day | 0.06 | < 10 K/day |
| model-top temperature, window minimum | 159.8 .. 159.9 K | 0.02 | > 155 K |
| TOA net (SW down − SW up − OLR) | 95.5 .. 100.1 W/m² | 1.3 | 90 .. 105 W/m² |
| TOA shortwave albedo | 0.370 .. 0.375 | 0.003 | 0.36 .. 0.39 |
| lowest-level cover, time mean | 0 | 0 | < 0.01 |
| near-surface air temperature, window mean | 299.69 .. 299.71 K | 0.009 | 297.5 .. 300 K |
| near-surface specific humidity, window mean | 20.33 .. 20.50 g/kg | 0.02 | 18 .. 22 g/kg |

The column's vertical diffusion is ECHAM6.3's TKE closure on ECHAM's moist,
cloud-weighted buoyancy ({doc}`../science/vertical_diffusion`,
{doc}`tte_tke_closure`); that sets the evaporation (0.72-0.79 mm/d), the
near-surface air 0.3 K below the SST, and the TOA net the pins are measured at.

The bounds are measured with ECHAM's own cover constants at T63 (`crt` 0.75,
`crs` 0.975, `nex` 2, `csatsc` 0.7, `cinv` 0.25), which the test states
explicitly. The shipped T63 defaults are jcm's calibrated cover set
({doc}`jam_aerosol_retune`, "Stage 2b: cloud fraction"); with it, measured
under the earlier closure, this column held cloud at its lowest level
(time-mean cover 0.275) and rained 0.926 of what it evaporated, because the
surface critical humidity of the calibrated set (0.9) is below the relative
humidity the column's lowest layer reaches (ECHAM's 0.975 is above it).

Provenance: the branch of #996 (off `dev` 43632f49), jax 0.10.2, jax-rrtmgp
0.5.0, float32, CPU. The 40-day window means of the trajectories agree across
the run-to-run noise; the slow part of the evolution is what the later windows
measure, and the first window is the furthest from stationary (P / E 0.96,
drift 0.034 mm/d in days 40-80 of the test's trajectory; 0.998 and 0.002 in
days 80-120). The bounds are meant to tell the equilibrium from a different
regime and not to track every re-tuning.

The two lowest-level and near-surface bounds are wide because their
across-trajectory ranges are small; they separate this column from a fogged or
decoupled boundary layer and not from small changes of it. Three properties
are measured and not pinned.

- *The column is overcast.* The time-mean layer cover is 1.0 at 302-495 hPa,
  0.88 at 538, 0.77 at 581, 0.55 at 626 and 0.27 at 671 hPa (days 40-80); the
  layers at 237 and 269 hPa and the lowest model level stay clear.
- *Its hydrological cycle is weak.* E = 0.78 mm/d in days 40-80 (a latent
  heat flux of 23 W/m²), falling to 0.72 mm/d in days 160-200, with a TOA
  shortwave albedo of 0.37.
- *The column is in balance but still adjusting.* P / E is 0.96, 1.00, 1.00,
  1.00 and the water drift 0.034, 0.002, 0.002, 0.001 mm/d over the four
  windows of the 200-day run, and the TOA net is 98.8, 97.3, 95.5,
  96.1 W/m². The lowest-level cover is 0 in all four.

The reference states of ECHAM6.3's compiled `cucall`
(`jcm/data/test/echam_cumastr_reference`) were captured from the grey
configuration of this testbed. They are stored arguments and do not depend on
the column that runs now.

## The Tiedtke downdraft rain-uptake conservation default

`ConvectionParameters.conservative_downdraft_uptake` (default `True`) bounds
the Tiedtke downdraft's rain uptake by what the final ascent generates,
instead of ECHAM's own non-conservative floor, which can create water where
a (stale, first-ascent-scaled) downdraft outtakes a demoted, re-run ascent's
smaller rain (`docs/source/science/convection.md`'s "Why we differ"). On
this testbed's present (RRTMGP, days 40-80) trajectory, `False` (ECHAM's own
floor) and `True` (the default) both measure `precip_floor_source` at
exactly 0 over the window — this column's current deep/shallow mix never
hits the deficit the deviation guards against — so the two configurations'
time-mean diagnostics agree within this testbed's own across-trajectory
noise band (the table above): `P/E` 0.957 vs. 0.953, time-mean convective
precipitation 0.786 vs. 0.786 mm/d, column water drift 0.051 vs. 0.057 mm/d,
water-budget residual 9.8e-5 vs. 9.4e-5 of E. The deviation's effect is
demonstrated instead on a synthetic column built to reproduce it exactly
(`flux_tendencies_test.py::TestConservativeDowndraftUptake`): there, the
non-conservative floor creates water equal to the constructed shortfall and
the conservative default closes the water ledger to round-off. A 3D A/B
(the global water budget and convective precipitation climatology) is the
validation this single column cannot give (jax-gcm#912).

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
with RRTMGP, S0 = 420 W/m², days 40-80, one trajectory each, measured on `dev`
at ee1e4e63 (with the moist interior stability of the pins above, the
ω_max = 0 row is P / E 0.960, P 1.149, E 1.198 mm/d):

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
