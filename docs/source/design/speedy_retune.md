# SPEEDY at T31: land, clear sky and calibration

The ``speedy-t31`` configuration is the cheap host coupled experiments reach for
(about a sim-year per A100-minute), so its top-of-atmosphere (TOA) budget and the
fluxes it hands a coupled ocean or land have to be credible, not merely those of a
smoke test. This page records how the configuration was brought there: the
structural changes made first, each measured, then the calibration of SPEEDY's
own parameters against observations around a clear-sky longwave fitted to
RRTMGP ({doc}`speedy_longwave_fit`), with the fit's provenance.

## Grid and observations

SPEEDY is calibrated at its own resolution, T31 with the Fortran model's hand-set
8-level sigma table. On any other grid it is a different model
({doc}`speedy_variable_levels`), so a fit at T63, where most jcm observations
were first used, would tune the wrong model. The observations are jcm-monitor's
T31 build (``obs/t31``): the same ``obs/build.py`` pipeline and sources as its T63
build, conservatively remapped to the 96×48 Gaussian grid. Sources: CERES-EBAF
Ed4.2.1 for the TOA fluxes, cloud radiative effects and surface downward
fluxes (2001–2020), GPCP v3.3 precipitation, ERA5 precipitable water, 2 m
temperature and layer temperatures and humidities, ESA-CCI AVHRR cloud cover,
MERRA-2 surface turbulent fluxes.

Two definitional caveats shape the targets. The CERES clear sky is the
cloud-free-scene flux, about 2 W m⁻² from a model's total-region clear sky by
its sampling alone and 10 W m⁻² above RRTMGP on reanalysis states in the
longwave ({doc}`speedy_longwave_fit`); the cloud radiative effects inherit
that offset, so they carry a wider error than the TOA fluxes. The near-surface temperature is compared as SPEEDY's own
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
+configuration=speedy-t31`` run with the levers as Hydra overrides, at the
configuration's 30-minute step: 395 days from a 2001-12-31 warm state (two
years from rest), the first 30 days discarded, so the compared mean is a full
year and the observed target its day-weighted monthly climatology. An
iteration is one finite-difference arm per lever and three damped trial steps.

The clear-sky longwave is not part of this fit. Its fifteen values are fitted
as one set to RRTMGP on the model's own columns and held
({doc}`speedy_longwave_fit`), so the levers below answer for cloud, shortwave
and the hydrological cycle against a clear sky that is already right, instead
of compensating for it.

**Targets.** The budget and surface fields are a global/land/ocean bias group
plus a 10° zonal group each (the CREs and the surface radiation: global bias
plus zonal), the layer fields a zonal group only, the groups weighted equally,
with absolute error floors (no per-year block series exists at T31):

| target | floor | |
|---|---|---|
| net TOA, OLR, reflected SW | 2 W m⁻² | the coupling-critical budget |
| SW and LW cloud radiative effect | 4 W m⁻² | wider for the CERES clear-sky offset |
| surface downward SW and LW | 4 W m⁻² | CERES EBAF-surface, a radiative-transfer product |
| precipitation | 0.15 mm/day | |
| precipitable water | 1 kg m⁻² | |
| total cloud cover | 0.03 | ``radiation.total_cloud_cover`` |
| near-surface temperature | 1 K | ``t0`` against ERA5 2 m |
| sensible / latent heat flux | 4 / 5 W m⁻² | reanalysis, not observation |
| layer temperature, σ = 0.2 to 0.835 | 1.5 K | ERA5, mass mean over the SPEEDY layer |
| layer humidity, σ = 0.34 to 0.95 | 20 % of the layer's global mean | the same |

The clear-sky OLR is not a target: RRTMGP on reanalysis states is itself 10
W m⁻² below the CERES clear-scene product ({doc}`speedy_longwave_fit`), so that
product cannot say whether a clear sky is right to better than that. The LW
CRE, which inherits the offset, stays in at its wide floor. The layer at
σ = 0.095 (50–140 hPa) is scored but carries no weight: no lever here reaches
the lower stratosphere, which the longwave fit and the isothermal top set.

**Noise.** A twin of the shipped configuration with gravity raised by one part
per million differs from it, in the year mean, by 0.24 W m⁻² in the net TOA
flux, up to 0.2 W m⁻² in the other global radiative fluxes, 0.003 mm/day in
precipitation, 0.04 K in ``t0``, about 0.5 W m⁻² in a land-mean flux, and by
0.9 in the loss (79.2 against 80.1). That is an order below every floor and
far below the finite-difference step's response, so single-year arms suffice;
it is also the resolution of the fit, and iterations stop when one gains less
than it.

**Levers**, 29, each a SPEEDY parameter that sets one of those fields in its
own formulation:

- clouds and shortwave: the backscatter ``refclr`` and the cloud albedos
  ``albcl``/``albcls`` (reflected SW and its clear/cloud split); the cover
  thresholds ``rhcl1``/``rhcl2`` and the stratocumulus maximum ``clsmax``
  (cover); the cloud longwave absorptivities ``ablcl1``/``ablcl2`` (LW CRE per
  unit cover); the shortwave absorptivities ``absdry``, ``absaer``,
  ``abswv1``, ``abswv2`` (the atmosphere's share of the absorbed shortwave, so
  the surface insolation);
- convection and condensation: ``rhbl``, ``trcnv``, ``rhil``, ``smf``,
  ``entmax``, ``psmin`` and ``rhlsc``, ``drhlsc``, ``trlsc`` (precipitation
  against precipitable water, and each layer's humidity), with the
  shallow-convection time ``trshc`` (boundary-layer export);
- surface fluxes: the exchange coefficients ``chs``/``chl``, the near-surface
  extrapolation weights ``fhum0``/``ftemp0``, the stability correction
  ``fstab``/``dtheta`` and the gust speed ``vgust``.

Starting from the fitted longwave with the other levers at an earlier
thirteen-lever calibration made against SPEEDY's own longwave, six iterations
were accepted (three at a 15-minute step, loss 162.4 → 86.7; three at the
30-minute step from a fresh spin-up at that centre, 90.3 → 82.7, the last
gaining 0.3). The shipped values are the final centre, each with what it sets
in ``configuration/speedy-t31.yaml``:

| parameter | SPEEDY | calibrated | fit range | |
|---|---|---|---|---|
| ``refclr`` | 0 | 0.0759 | 0.028–0.113 | jcm's addition |
| ``albcl`` | 0.43 | 0.447 | 0.2–0.6 | |
| ``albcls`` | 0.50 | 0.200 | 0.2–0.7 | lower bound |
| ``rhcl1`` | 0.30 | 0.550 | 0.15–0.55 | upper bound |
| ``rhcl2`` | 1.0 | 0.860 | 0.75–1.0 | |
| ``clsmax`` | 0.60 | 0.647 | 0.3–0.85 | |
| ``ablcl1`` | 12 | 5.30 | 5.0–19.9 | |
| ``ablcl2`` | 0.6 | 0.300 | 0.3–1.2 | lower bound |
| ``absdry`` | 0.033 | 0.0280 | 0.0165–0.066 | |
| ``absaer`` | 0.033 | 0.0170 | 0.0165–0.066 | |
| ``abswv1`` | 0.022 | 0.0588 | 0.011–0.088 | |
| ``abswv2`` | 15 | 27.9 | 7.5–30 | |
| ``rhbl`` | 0.9 | 0.858 | 0.7–0.95 | |
| ``trcnv`` | 6 h | 5.64 h | 3–12 | |
| ``rhil`` | 0.7 | 0.685 | 0.5–0.9 | |
| ``smf`` | 0.8 | 1.17 | 0.4–1.2 | |
| ``entmax`` | 0.5 | 0.861 | 0.2–1.0 | |
| ``psmin`` | 0.8 | 0.650 | 0.65–0.9 | lower bound |
| ``rhlsc`` | 0.9 | 0.867 | 0.8–0.98 | |
| ``drhlsc`` | 0.1 | 0.0206 | 0–0.3 | |
| ``trlsc`` | 4 h | 2.39 h | 2–8 | |
| ``trshc`` | 6 h | 20.6 h | 9.0–35.8 | |
| ``chs`` | 9.0·10⁻⁴ | 9.56·10⁻⁴ | 5.3·10⁻⁴–1.4·10⁻³ | |
| ``chl`` | 1.2·10⁻³ | 2.83·10⁻³ | 1.06·10⁻³–2.83·10⁻³ | upper bound |
| ``fhum0`` | 0 | 0 | 0–1 | lower bound, SPEEDY's value |
| ``ftemp0`` | 1 | 0.915 | 0–1 | |
| ``fstab`` | 0.67 | 0.636 | 0.3–1.0 | |
| ``dtheta`` | 3 K | 1.5 K | 1.5–6 | lower bound |
| ``vgust`` | 5 m/s | 4.33 m/s | 2.5–10 | |

The Python ``speedy_physics()`` keeps SPEEDY's reference values; only the
``speedy-t31`` configuration carries the fit.

### Result

Biases against the observations, one model year (2002; prescribed SSTs) from a
two-year spin-up, global mean with the land / ocean means in brackets; fluxes
in W m⁻². *dev* is ``speedy-t31`` at ``dev`` (reference SPEEDY, prescribed
land), *structural* the same after the structural changes above with
``refclr`` = 0.05 and SPEEDY's parameters, *calibrated* the shipped
configuration. *dev* and *structural* ran at the configuration's former
15-minute step. *dev* has no clear-sky fluxes; at SPEEDY's parameters and the
structural code without the backscatter, the SW CRE is −72.5 W m⁻² (−26.8) and
the LW CRE 26.2 (−1.7). The loss of the calibrated year is 79.2 (twin 80.1),
against 160.1 for the fit's starting values run the same way; both are below
the fit's own arms because those start from a state spun up at an earlier
centre, these from their own two-year spin-up.

| field | obs | dev | structural | calibrated |
|---|---|---|---|---|
| net TOA | 1.01 | +4.07 (+0.5 / +5.5) | -8.70 (-10.1 / -8.1) | -0.61 (-1.9 / -0.1) |
| OLR | 240.46 | -8.42 (-2.8 / -10.7) | -8.92 (-3.8 / -11.0) | -1.40 (+0.5 / -2.2) |
| reflected SW | 99.03 | +5.94 (+3.7 / +6.8) | +17.46 (+13.7 / +19.0) | +1.81 (+1.3 / +2.0) |
| SW CRE | -45.73 | — | -23.53 (-20.6 / -24.7) | -3.19 (-6.1 / -2.0) |
| LW CRE | 27.88 | — | -1.55 (-2.9 / -1.0) | -5.14 (-4.0 / -5.6) |
| clear-sky OLR (not fitted) | 268.34 | — | -10.47 (-6.7 / -12.0) | -6.54 (-3.5 / -7.8) |
| clear-sky reflected SW (not fitted) | 53.31 | — | -6.07 (-6.9 / -5.7) | -1.38 (-4.7 / -0.0) |
| precipitation (mm/day) | 3.07 | -0.27 (+0.1 / -0.4) | -0.20 (+0.1 / -0.3) | -0.17 (+0.6 / -0.5) |
| precipitable water (kg m⁻²) | 24.48 | +4.33 (+1.8 / +5.3) | +4.41 (+1.8 / +5.5) | +0.32 (-0.6 / +0.7) |
| total cover | 0.63 | -0.00 (+0.0 / -0.0) | +0.00 (+0.0 / -0.0) | -0.06 (-0.0 / -0.1) |
| free-troposphere cover ``cloudc`` | 0.63 | -0.04 (-0.0 / -0.0) | -0.03 (-0.0 / -0.0) | -0.20 (-0.1 / -0.2) |
| near-surface T ``t0`` (K) | 287.65 | +0.37 (+1.4 / -0.0) | +0.22 (+1.0 / -0.1) | +1.05 (+1.7 / +0.8) |
| sensible heat | 19.18 | +3.74 (-9.2 / +8.9) | +4.98 (-7.5 / +10.0) | -0.18 (-10.7 / +4.0) |
| latent heat | 85.72 | -4.61 (-5.7 / -4.1) | -2.73 (-3.6 / -2.4) | -1.63 (+7.3 / -5.2) |
| surface downward SW | 187.34 | +13.16 (+16.5 / +11.8) | +0.49 (+3.7 / -0.8) | -0.48 (+3.6 / -2.1) |
| surface downward LW | 346.39 | -12.69 (-9.8 / -13.9) | -12.64 (-10.2 / -13.6) | -6.80 (-2.0 / -8.8) |

The calibrated year's layers against ERA5 (global-mean bias, and the RMSE of
the zonal mean):

| layer (σ) | 0.095 | 0.2 | 0.34 | 0.51 | 0.685 | 0.835 | 0.95 |
|---|---|---|---|---|---|---|---|
| temperature bias (K) | -3.6 | +1.6 | +2.9 | -1.7 | -2.1 | +0.3 | +1.4 |
| temperature, zonal RMSE (K) | 8.2 | 5.8 | 4.1 | 1.9 | 2.2 | 0.9 | 1.9 |
| humidity bias (% of the layer mean) | — | — | -24 | -4 | -7 | +1 | +9 |

Root-mean-square error of the annual-mean map against the observations:

| field | dev | calibrated |
|---|---|---|
| net TOA | 15.58 | 12.18 |
| OLR | 14.08 | 10.31 |
| reflected SW | 21.08 | 14.63 |
| precipitation (mm/day) | 2.12 | 1.89 |
| precipitable water (kg m⁻²) | 5.92 | 3.25 |
| total cover | 0.13 | 0.14 |
| near-surface T ``t0`` (K) | 2.56 | 2.55 |
| sensible heat | 17.26 | 19.78 |
| latent heat | 21.80 | 26.70 |

The net land heat uptake of the calibrated year is +0.7 W m⁻² over land cells
(+9.3 with the prescribed land).

## Remaining biases

The TOA budget, the reflected shortwave and its clear/cloud split, the surface
insolation and the precipitable water are within their floors in the global
mean. What is left is where SPEEDY's structure rather than its tuning is the
limit: seven levers sit on an end of their range, several others are far from
SPEEDY's values (``trshc`` 21 h against 6, ``chl`` 2.4 times the reference,
``entmax`` 0.86 against 0.5), and the turbulent-flux maps are further from
MERRA-2 than at ``dev``.

- **The upper troposphere and above.** The layers at σ = 0.2 and 0.34 are 1.6
  and 2.9 K warm with a zonal-mean RMSE of 6 and 4 K, the layer at 0.095 is
  3.6 K cold, and the mid-troposphere 2 K cold. The temperature at σ = 0.2 is
  the largest single term of the loss (11.6 of 79.2) and the fit barely moved
  it (13.1 at its start): eight layers, an isothermal top and no stratosphere
  set it, not a parameter.
- **Longwave cloud effect and the clear-sky reference.** The clear-sky OLR is
  6.5 W m⁻² below the CERES clear-scene product, where a scheme matching
  RRTMGP on a perfect state would read 7–10 below
  ({doc}`speedy_longwave_fit`). With the all-sky OLR on target (−1.4), the LW
  CRE therefore reads 5 W m⁻² weak against the same product, and the two
  cannot both be closed. Cloud is still part of it: the free-troposphere cover
  ``cloudc`` is 0.43 and the cover the shortwave sees 0.58 against ESA-CCI's
  0.63, with the stratocumulus albedo and the upper-cloud absorptivity on
  their lower bounds.
- **Surface downward longwave** is 6.8 W m⁻² below CERES (ocean 8.8). The
  clear-sky flux matches RRTMGP, and with the cloud values the fit started
  from the all-sky flux was 2.5 low; the fit took 4 W m⁻² from it for the TOA
  budget (cloud about half as absorbing in the window, 0.6 kg m⁻² less water
  vapour).
- **Land surface fluxes.** Global sensible and latent heat are within 2 W m⁻²
  of MERRA-2, but the land Bowen ratio is not: over land the sensible heat
  flux is 11 W m⁻² low and the latent heat 7 high, with the land exchange
  coefficient on its upper bound; over the ocean they are 4 high and 5 low.
  Land precipitation is 0.6 mm/day high (ocean 0.5 low).
- **Near-surface temperature** is 1.0 K warm (land 1.7, ocean 0.8, the latter
  over prescribed SSTs: it is the air-sea contrast the extrapolation weight
  ``ftemp0`` sets).
- **Net TOA** is +0.4 W m⁻² against CERES' +1.0 in this configuration
  (prescribed SSTs), well inside its floor. Its zonal structure is the larger
  error: 7.5 W m⁻² RMS in the zonal mean.

## Provenance

- jax-gcm branches ``speedy-tune`` (structural changes ``1172dc69``),
  ``speedy-lw-fit`` (the longwave parameters and the surface longwave fields)
  and ``speedy-refit`` (the calibrated values); JEM-Cal branch
  ``speedy-refit`` (host table, layer and surface-radiation targets, linear
  levers).
- Runs on one shared A100. Structural measurements,
  ``/scr/dwatsonparris/speedy-tune/runs``: ``ctl0`` (dev control, six years),
  ``land_{p,s,f}`` (land A/B), ``refclr05``, ``ctlyear_dev`` (the dev control
  year). The fit, ``/scr/dwatsonparris/speedy-tune/refit``: ``gn1`` (15-minute
  step) and ``gn2`` (30-minute step) hold the ledgers, Jacobians and every
  arm's result; ``confirm/`` holds the two-year spin-ups, the confirmation
  year (``year_final``), its twin (``year_final_twin``), the starting values'
  year (``year_base``), their scores and the table script.
- Observations: jcm-monitor ``obs/t31`` (built 2026-09-24, CDO remapcon). The
  layer temperatures and humidities are ERA5's 1990–2019 monthly climatology on
  13 pressure levels (WeatherBench2, 64×32), averaged in mass over each SPEEDY
  layer with ERA5's surface pressure and interpolated to the T31 grid, built
  for this fit (``refit/scripts/build_era5_layers.py``).
- The dev monitor (experiment ``speedy-retune``) holds the dev control year.
