# SPEEDY at T31: land, clear sky and calibration

The ``speedy-t31`` configuration is the cheap host coupled experiments reach for
(about a sim-year per A100-minute), so its top-of-atmosphere (TOA) budget and the
fluxes it hands a coupled ocean or land have to be credible, not merely those of a
smoke test. This page records how the configuration was brought there: the
structural changes made first, each measured, and then the calibration of
SPEEDY's own parameters against observations, with the fit's provenance.

## Grid and observations

SPEEDY is calibrated at its own resolution, T31 with the Fortran model's hand-set
8-level sigma table. On any other grid it is a different model
({doc}`speedy_variable_levels`), so a fit at T63, where most jcm observations
were first used, would tune the wrong model. The observations are jcm-monitor's
T31 build (``obs/t31``): the same ``obs/build.py`` pipeline and sources as its T63
build, conservatively remapped to the 96×48 Gaussian grid. Sources: CERES-EBAF
Ed4.2.1 for the TOA fluxes and cloud radiative effects (2001–2020), GPCP v3.3
precipitation, ERA5 precipitable water and 2 m temperature, ESA-CCI AVHRR cloud
cover, MERRA-2 surface turbulent fluxes.

Two definitional caveats shape the targets. The CERES clear sky is the
cloud-free-scene flux, about 2 W m⁻² from a model's total-region clear sky, and
the cloud radiative effects inherit that offset, so they carry a wider error
than the TOA fluxes. The near-surface temperature is compared as SPEEDY's own
near-surface air temperature ``surface_flux.t0`` (σ = 0.99, the reference of its
bulk formulae), not the lowest model level (σ = 0.95, ~400 m up and 2 K colder
over land); the dev monitor still shows the lowest level, its gate quantity.

## Structural changes, before any parameter moves

Each was measured on two-year means after a one-year spin-up from rest
(``speedy-t31``, 2001–2002, prescribed SSTs), before calibration.

### The land is SPEEDY's slab land

At ``dev`` the SPEEDY land temperature was the prescribed climatology. Near-surface
temperature over land is then close to observed by construction (−0.7 K), so the
land bias does not show where it was expected to. It shows in the energy budget:
the land surface absorbs **+9.3 W m⁻²** in the annual mean over land cells (cell
values −41 to +86), **2.4 W m⁻² of the global mean**, into a reservoir that cannot
hold it. A coupled ocean would see that as TOA imbalance it has to absorb, and the
land cannot warm under forcing.

SPEEDY's Fortran runs a slab land by default (``land_model.f90``,
``land_coupling_flag = 1``); jcm had removed it. It is restored as the equation
that file states, with its 40-day relaxation, integrated every physics step
(speedy.f90 calls it every 40-minute step but keeps the per-call factor, so there
the "40 days" act in 27 hours and the land is nearly prescribed), as
``SpeedySurfaceFlux``'s default ``land_model="slab"``
({doc}`../science/surface`, *The SPEEDY land*). The choice is made in that one
place: the ``speedy`` Hydra presets carry no ``land_model`` key and
``speedy_physics()`` passes none, so the CLI and the Python API cannot run
different lands, and ``runners_test`` pins the two compositions equal.
``land_model="prescribed"`` holds the land at the climatology.

| 2001–2002 | prescribed | slab, 40-day relaxation | free slab |
|---|---|---|---|
| net heat flux into land (land > 95 %) | +9.3 W m⁻² | +1.4 W m⁻² | 0.0 W m⁻² |
| net TOA, global | 3.90 | 3.22 | 3.06 |
| latent heat, land (MERRA-2 44.9) | 39.1 | 44.1 | 44.6 |
| sensible heat, land (MERRA-2 33.1) | 23.9 | 27.5 | 28.3 |
| precipitation, land (GPCP 2.16 mm/day) | 2.24 | 2.60 | 2.62 |
| ``t0`` over land (ERA5 2 m 282.7 K) | 284.0 | 284.6 | 284.8 |

The relaxation stays at SPEEDY's 40 days. A free slab (``tdland = inf``) closes the
land budget exactly but its uncapped ice-sheet slabs drift 18–20 K warm over
northern Greenland and Severnaya Zemlya, against 12–16 K with the relaxation;
coupled climate-change runs that need the land to conserve energy can set
``physics.terms.speedy_surface_flux.land_params.tdland`` and should watch the ice
sheets. SLIM, the ECHAM hosts' land, was not ported to SPEEDY: it is built on the
ECHAM tile interface, and SPEEDY's own slab is the faithful choice for a SPEEDY
host.

### Clear-sky fluxes and the published radiation fields

SPEEDY computed no clear-sky fluxes, so its cloud radiative effects could not be
scored. A diagnostic clear-sky pass (the shortwave without cloud on each radiation
step, which also yields the cloud-free longwave transmissivities, and the upward
longwave with those every step) now gives them, and SPEEDY publishes the
``radiation.*`` and ``pressure_full``/``pressure_thickness`` fields under the names
the ECHAM hosts use ({doc}`../science/radiation`, *SPEEDY radiation*).

### Insolation at the shared solar constant

SPEEDY's hard-coded 1368 W m⁻² put the global insolation 1.6 W m⁻² above CERES;
reading ``jcm.constants.solc`` (1361) leaves −0.2 W m⁻² and lowers the net TOA
flux by 1.1 W m⁻².

### Clear-sky backscatter

With clear-sky fluxes in hand, the dominant shortwave error is structural:

| 2001–2002, prescribed land | model | CERES | bias |
|---|---|---|---|
| clear-sky reflected SW, global | 32.1 | 53.3 | −21.2 |
| clear-sky reflected SW, ocean | 22.4 | 44.3 | −22.0 |
| surface downward SW (CERES EBAF-surface) | 199.3 | 187.3 | +12.0 |
| SW cloud radiative effect | −72.5 | −45.7 | −26.8 |
| reflected SW, all sky | 104.6 | 99.0 | +5.5 |

SPEEDY's clear sky only absorbs. Its reference tuning reaches a plausible total
reflection through clouds about 60 % too bright. The ``refclr`` reflectance
(molecular and aerosol backscatter, proportional to the slant air mass) lets the
reflection sit where it is observed; at 0.05, the first guess from the ocean
clear-sky deficit, the clear-sky reflection deficit falls to −6 W m⁻² and the
surface insolation bias to +0.5 W m⁻², leaving the clouds, now too bright, to the
calibration.

## Calibration

### Recipe

JEM-Cal's derivative-free Gauss-Newton driver (``examples/retune_echam_v3.py``
with host ``speedy-t31`` in ``examples/retune_levers.yaml``; JEM-Cal's
``DESIGN.md`` §5.4). Each arm is one ``python -m jcm.main
+configuration=speedy-t31`` run with the levers as Hydra overrides: 395 days
from a 2001-12-31 warm state (two years from rest at the pre-fit
configuration), the first 30 days discarded, so the compared mean is a full
year and the observed target its day-weighted monthly climatology. Every
target is a global/land/ocean bias group plus a 10° zonal group, the groups
weighted equally, with absolute error floors (no per-year block series exists
at T31):

| target | floor | |
|---|---|---|
| net TOA, OLR, reflected SW | 2 W m⁻² | the coupling-critical budget |
| SW and LW cloud radiative effect | 4 W m⁻² | wider for the CERES clear-sky offset |
| precipitation | 0.15 mm/day | |
| precipitable water | 1 kg m⁻² | |
| total cloud cover | 0.03 | ``radiation.total_cloud_cover`` |
| near-surface temperature | 1 K | ``t0`` against ERA5 2 m |
| sensible / latent heat flux | 4 / 5 W m⁻² | reanalysis, not observation |

The year-mean weather noise, measured with a twin whose gravity is raised by
one part per million, is at most 0.06 W m⁻² in a global TOA flux, 0.002 mm/day
in precipitation and 0.4 W m⁻² in a land-mean flux: an order below every floor,
and far below the finite-difference step's response, so single-year arms
suffice.

Levers, each a SPEEDY parameter that sets one of those fields in its own
formulation: the cloud albedos ``albcl``/``albcls`` and the backscatter
``refclr`` (reflected SW and its clear/cloud split), the cover thresholds
``rhcl1``/``clsmax`` (cover), the cloud and water-vapour longwave
absorptivities ``ablcl1``/``ablwv1`` (OLR and LW CRE), the convective and
large-scale condensation humidity thresholds ``rhbl``/``rhlsc`` (precipitation
against precipitable water), the sea and land exchange coefficients
``chs``/``chl`` (turbulent fluxes) and the shallow-convection time ``trshc``
(boundary-layer moisture export). A first stage on these twelve (two accepted
iterations, loss 222.5 → 82.4) bought the OLR and the SW CRE by removing cloud,
because none of them moves the clear-sky OLR cheaply; the second stage, from
its centre, added the window-band absorptivity ``ablwin`` (two iterations, 83.4
→ 67.9). The shipped values are its best arm, rounded, each with its reason in
``configuration/speedy-t31.yaml``.

| parameter | SPEEDY | calibrated | |
|---|---|---|---|
| ``refclr`` | 0 | 0.0564 | jcm's addition |
| ``albcl`` | 0.43 | 0.500 | |
| ``albcls`` | 0.50 | 0.200 | lower bound of the fit |
| ``rhcl1`` | 0.30 | 0.55 | upper bound of the fit |
| ``clsmax`` | 0.60 | 0.744 | |
| ``ablcl1`` | 12 | 9.97 | |
| ``ablwv1`` | 0.7 | 1.07 | |
| ``ablwin`` | 0.3 | 0.150 | lower bound of the fit |
| ``rhbl`` | 0.9 | 0.870 | |
| ``rhlsc`` | 0.9 | 0.920 | |
| ``chs`` | 9.0·10⁻⁴ | 8.75·10⁻⁴ | |
| ``chl`` | 1.2·10⁻³ | 1.77·10⁻³ | |
| ``trshc`` | 6 h | 17.9 h | |

The Python ``speedy_physics()`` keeps SPEEDY's reference values; only the
``speedy-t31`` configuration carries the fit.

### Result

Biases against the observations, one model year (2002; prescribed SSTs) from a
two-year spin-up, global mean with the land / ocean means in brackets; fluxes
in W m⁻². *dev* is ``speedy-t31`` at ``dev`` (reference SPEEDY, prescribed
land), *structural* the same after the structural changes above with
``refclr`` = 0.05 and SPEEDY's parameters (the fit's starting point),
*calibrated* the shipped configuration. *dev* has no clear-sky fluxes; at
SPEEDY's parameters and the structural code without the backscatter, the SW CRE
is −72.5 W m⁻² (−26.8) and the LW CRE 26.2 (−1.7).

| field | obs | dev | structural | calibrated |
|---|---|---|---|---|
| net TOA | 1.01 | +4.07 (+0.5 / +5.5) | -8.70 (-10.1 / -8.1) | +2.19 (+1.0 / +2.7) |
| OLR | 240.46 | -8.42 (-2.8 / -10.7) | -8.92 (-3.8 / -11.0) | -5.61 (-3.6 / -6.4) |
| reflected SW | 99.03 | +5.94 (+3.7 / +6.8) | +17.46 (+13.7 / +19.0) | +3.25 (+2.4 / +3.6) |
| SW CRE | -45.73 | — | -23.53 (-20.6 / -24.7) | -7.14 (-7.5 / -7.0) |
| LW CRE | 27.88 | — | -1.55 (-2.9 / -1.0) | -7.26 (-6.3 / -7.7) |
| clear-sky OLR (not fitted) | 268.34 | — | -10.47 (-6.7 / -12.0) | -12.87 (-9.9 / -14.1) |
| clear-sky reflected SW (not fitted) | 53.31 | — | -6.07 (-6.9 / -5.7) | -3.89 (-5.1 / -3.4) |
| precipitation (mm/day) | 3.07 | -0.27 (+0.1 / -0.4) | -0.20 (+0.1 / -0.3) | -0.18 (+0.8 / -0.6) |
| precipitable water (kg m⁻²) | 24.48 | +4.33 (+1.8 / +5.3) | +4.41 (+1.8 / +5.5) | +1.89 (+0.8 / +2.3) |
| total cover | 0.63 | -0.00 (+0.0 / -0.0) | +0.00 (+0.0 / -0.0) | -0.09 (-0.1 / -0.1) |
| free-troposphere cover ``cloudc`` | 0.63 | -0.04 (-0.0 / -0.0) | -0.03 (-0.0 / -0.0) | -0.25 (-0.2 / -0.3) |
| near-surface T ``t0`` (K) | 287.65 | +0.37 (+1.4 / -0.0) | +0.22 (+1.0 / -0.1) | +0.35 (+1.2 / +0.0) |
| sensible heat | 19.18 | +3.74 (-9.2 / +8.9) | +4.98 (-7.5 / +10.0) | +6.05 (-2.3 / +9.4) |
| latent heat | 85.72 | -4.61 (-5.7 / -4.1) | -2.73 (-3.6 / -2.4) | -2.10 (+4.2 / -4.7) |
| surface downward SW | 187.34 | +13.16 (+16.5 / +11.8) | +0.49 (+3.7 / -0.8) | +18.71 (+19.2 / +18.5) |
| surface downward LW | 346.39 | -12.69 (-9.8 / -13.9) | -12.64 (-10.2 / -13.6) | -16.44 (-8.3 / -19.7) |

Root-mean-square error of the annual-mean map against the observations:

| field | dev | calibrated |
|---|---|---|
| net TOA | 15.58 | 11.91 |
| OLR | 14.08 | 11.36 |
| reflected SW | 21.08 | 16.42 |
| precipitation (mm/day) | 2.12 | 2.24 |
| precipitable water (kg m⁻²) | 5.92 | 3.68 |
| total cover | 0.13 | 0.16 |
| near-surface T ``t0`` (K) | 2.56 | 2.43 |
| sensible heat | 17.26 | 18.73 |
| latent heat | 21.80 | 23.91 |

The net land heat uptake of the calibrated year is +0.6 W m⁻² over land cells
(+9.3 with the prescribed land).

## Remaining biases

The calibration brings the TOA budget, the reflected shortwave and its
clear/cloud split, and the precipitable water much closer to observed, but it
does so partly by compensation, and three parameters sit on a bound of the fit,
which says where SPEEDY's structure rather than its tuning is the limit.

- **Clear-sky OLR, and the clouds that pay for it.** SPEEDY's clear-sky OLR is
  10 W m⁻² below CERES at its reference values and 13 below after the fit, an
  order more than CERES' clear-sky definition offset; no lever moves it without
  side effects (the window absorptivity reached its bound). The fit raises the
  all-sky OLR instead by removing free-troposphere cloud: ``cloudc`` falls from
  0.59 to 0.38, the stratocumulus deck (dimmer, ``albcls`` 0.2, but wider,
  ``clsmax`` 0.74) carries the cover the shortwave sees to 0.55 against
  ESA-CCI's 0.63, and the LW CRE is 7 W m⁻² weak. The surface downward
  shortwave is therefore 19 W m⁻² high and the downward longwave 16 low: a
  coupled ocean receives a surface energy budget that is right in sum only
  because the net TOA flux is. A clear-sky longwave closer to observed — the
  band absorptivities fitted as one physical set against line-by-line or
  RRTMGP clear-sky fluxes, not as three tuned constants — is the next
  structural step, and would let the cloud cover come back (jax-gcm issue
  1200). {doc}`speedy_longwave_fit` measures the scheme against RRTMGP on this
  configuration's columns (6.6 of the 13 W m⁻² are the scheme's) and records
  the fit and what adopting it needs.
- **Ocean surface fluxes.** The sensible heat flux stays ~9 W m⁻² above MERRA-2
  over the ocean and the latent heat ~5 below: one exchange coefficient serves
  both, so the ocean Bowen ratio is set by SPEEDY's near-surface temperature
  and humidity extrapolation, which is not a lever.
- **Land precipitation** is 0.8 mm/day high (ocean 0.6 low), and the
  near-surface air over land 1.2 K warm, after the slab land replaced the
  prescribed land temperature.
- **Net TOA** is +2.2 W m⁻² above CERES' +1.0 in this configuration (prescribed
  SSTs): within one of its error floors, and the price the fit paid for the
  precipitation and latent heat; an intermediate fit centre had it at −0.8.

## Provenance

- jax-gcm branch ``speedy-tune`` (structural changes ``1172dc69``, interim
  centre ``9d50019a``, calibrated values ``d43e3b65``); JEM-Cal branch
  ``speedy-tune`` (on the Gauss-Newton driver of JEM-Cal pull request 17).
- Runs on one shared A100, ``/scr/dwatsonparris/speedy-tune/``: ``runs/ctl0``
  (dev control, six years), ``runs/land_{p,s,f}`` (land A/B), ``runs/refclr05``,
  ``runs/noise_{ctl,twin}``, ``gn1``/``gn2`` (the fits: ledgers, Jacobians,
  every arm's result), ``runs/spin_tuned`` + ``runs/confirm_tuned`` (the
  confirmation year), ``runs/ctlyear_dev`` (the dev control year).
- Observations: jcm-monitor ``obs/t31`` (built 2026-09-24, CDO remapcon).
- The dev monitor (experiment ``speedy-retune``) holds the dev control year and
  the calibrated year.

