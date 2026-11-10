# SPEEDY's clear-sky longwave against RRTMGP

SPEEDY's longwave is four bands with one absorptivity each and a tabulated
split of the blackbody spectrum between them
({doc}`../science/radiation`, *SPEEDY radiation*). The calibrated
``speedy-t31`` configuration ({doc}`speedy_retune`) leaves a clear-sky OLR
13 W m⁻² below CERES, which no single constant could move without side
effects. This page records what that deficit is made of, measured on the
model's own columns against RRTMGP, and the fit of the whole clear-sky
parameter set to RRTMGP that follows from it. The fitted values are **not** in
the ``speedy-t31`` configuration: they change the model's energy balance
enough that the cloud and shortwave levers have to be refitted with them
(*What adopting the fit needs*, below).

## The reference

**Columns.** 73 instantaneous snapshots, one every five days through 2002, of
the calibrated ``speedy-t31`` started from its two-year spin-up: all 4608
columns of each, 336 384 in total, with temperature, humidity, surface
pressure, the surface emission and the near-surface air temperature. SPEEDY's
side is the model's own routines run offline on those columns with the cloud
removed; they reproduce the model's ``radiation.toa_lw_up_clear`` to the size of
the surface-reflection term (next section).

**RRTMGP on a refined column.** The reference is jax-rrtmgp through jcm's own
plumbing (``jcm.physics.radiation.rrtmgp``: g128 tables, two-stream solver),
which on these inputs agrees with the Fortran RTE+RRTMGP 1.9 with 256 g-points
and three quadrature angles to −0.2 W m⁻² at the top and +1.2 W m⁻² at the
surface (RMS 0.2 and 1.7). It is not run on SPEEDY's eight layers as they are:
with layers 100–180 hPa thick, RRTMGP's answer then depends on how the library
reconstructs interface temperatures and layer masses, and comes out 4.4 W m⁻²
lower at the top. Each column is instead refined to 100 equal-mass sublayers
(10 hPa at 1000 hPa; every SPEEDY interface is a multiple of 0.01 in σ, so
they nest), with

- temperature piecewise linear in ln σ through the full-level values — the
  profile SPEEDY's own longwave assumes for its interface temperatures —
  isothermal above the top full level, as SPEEDY assumes, and continued below
  the lowest full level to SPEEDY's near-surface air temperature at σ = 1;
- humidity piecewise linear in (ln q, ln σ), then rescaled within each SPEEDY
  layer so the layer's vapour mass is exactly SPEEDY's;
- the gases a jcm ECHAM/RRTMGP run would see: CO₂ 420 ppmv (what the SPEEDY run
  is told), CH₄ 1.9 ppmv, N₂O 0.327 ppmv, the packaged present-day ozone
  climatology;
- SPEEDY's surface: emissivity 0.98 and the temperature whose emission is
  SPEEDY's ``rlus``, so both schemes start from the same surface emission.

Doubling the sublayers changes the fluxes by 0.01 W m⁻². The refinement itself
was checked on ERA5: instantaneous ERA5 states averaged onto SPEEDY's eight
layers and refined this way give the clear-sky OLR of the same states on
ERA5's own 13 levels to −0.07 W m⁻² in the global mean and 0.6 W m⁻² in the
worst latitude band, and the surface downward flux to 0.01 W m⁻².

One assumption is SPEEDY's and not the atmosphere's: the isothermal top. On the
ERA5 columns a standard-atmosphere temperature increase above 50 hPa raises the
clear-sky OLR by 2.0 W m⁻²; an 8-layer model cannot carry that structure, and
the reference does not ask it to.

## What the deficit is made of

Clear-sky OLR, model minus CERES-EBAF Ed4.2.1 (2001–2020 climatology), W m⁻².
*Scheme* is SPEEDY minus RRTMGP on the same instantaneous columns; *state* is
RRTMGP on SPEEDY's columns minus RRTMGP on instantaneous ERA5 states (2010, 73
snapshots, WeatherBench2's 64×32 ERA5) carried on SPEEDY's layers; *reference*
is RRTMGP on ERA5 minus CERES; *reflection* is the model as it ran minus the
scheme as it is now (the upward pass used to reflect a stale array off the
surface instead of the downward band fluxes).

| | 90S–60S | 60S–30S | 30S–10S | 10S–10N | 10N–30N | 30N–60N | 60N–90N | 60S–60N | global |
|---|---|---|---|---|---|---|---|---|---|
| model − CERES | −0.8 | −15.8 | −17.3 | −17.5 | −15.6 | −10.4 | +0.4 | −15.2 | **−13.1** |
| scheme | −0.2 | −4.1 | −9.1 | −12.4 | −8.7 | −4.0 | −0.6 | −7.6 | **−6.6** |
| state | +6.9 | −2.0 | +2.7 | +6.3 | +2.2 | +2.1 | +4.7 | +3.6 | **+2.5** |
| 8 layers | 0.0 | +0.2 | −0.2 | −0.6 | −0.3 | +0.2 | +0.3 | −0.1 | −0.1 |
| reference − CERES | −8.7 | −10.6 | −11.7 | −11.5 | −9.8 | −9.6 | −4.8 | −11.9 | **−9.8** |
| reflection | +1.1 | +0.7 | +0.9 | +0.8 | +1.0 | +0.9 | +0.8 | +0.8 | +0.9 |

The scheme is 6.6 W m⁻² low in the global mean and 12 in the deep tropics,
vanishing toward the poles: the signature of a window that is too transparent
where the boundary layer is moist. SPEEDY's *state* does not explain the
deficit; it hides part of it. By RRTMGP the calibrated state is 2.5 W m⁻²
brighter than ERA5's (a drier upper troposphere outweighs the humid layer
above it and the warm bias).

The largest term is not SPEEDY's. RRTMGP on ERA5's own states is 9.8 W m⁻²
below the CERES product the calibration compares against. Of that, about 2
W m⁻² is CERES' clear-scene sampling (Loeb et al. 2020; the same sampling of
SPEEDY's own clear-sky flux by its cloud-free fraction raises it by 3.3), 2.0
the isothermal top, 0.9 the surface emissivity of 0.98 where the ocean's is
nearer 1, and 4–5 remain unexplained here, as about 2 W m⁻² did in the same
exercise for jcm's ECHAM/RRTMGP configuration, whose clear-sky OLR is 7.5
W m⁻² below the same product. A SPEEDY whose longwave matched RRTMGP exactly on
a perfect state would therefore still read 7–10 W m⁻² below this target.

At the surface the clear-sky downward flux is 18.6 W m⁻² too weak in the
scheme (33 in the deep tropics) against a state that is 7.8 too strong, and the
clear-sky column loses 24.8 W m⁻² less longwave than RRTMGP's (156 against
181): SPEEDY's atmosphere is under-cooled by a seventh.

## The fit

**Parameters.** The four band absorptivities (``ablwin``, ``ablco2_ref``,
``ablwv1``, ``ablwv2``), ``epslw``, the nine band-fraction coefficients and
the window continuum ``ablwvc``, all at once.

**Loss.** Per column, the squared errors of the clear-sky OLR and of the
surface downward flux against the reference, plus two terms that keep the fit
from buying those two numbers with the rest of the scheme:

- *the layer flux convergence*, each SPEEDY layer's absorbed flux against
  RRTMGP's over the same pressure interval, at a tenth of the weight. Without
  it the column totals are met with the cooling in the wrong layers;
- *the response to doubled CO₂*, the change of both fluxes when 420 ppmv goes
  to 840 (SPEEDY: ``ablco2`` doubles), at ten times the weight, since a forcing
  matters at a tenth of the size a flux bias does. Without it the minimiser
  uses the CO₂ band as a spare band: the unconstrained fit has a clear-sky
  response of −10.8 W m⁻² where RRTMGP has −3.9.

A barrier keeps every band's share of the spectrum above 2 % between 200 and
320 K. The gradients are reverse-mode AD through the model's own routines; the
minimiser is L-BFGS-B from the calibrated values, on 20 000 columns drawn with
probability proportional to area, with 20 000 more held out.

**Result**, on all 336 384 columns (area-weighted; W m⁻²):

| | calibrated | four bands refitted | with the continuum |
|---|---|---|---|
| clear-sky OLR, bias / RMS | −6.6 / 8.9 | 0.0 / 4.8 | −0.2 / 3.0 |
| surface downward, bias / RMS | −18.6 / 22.0 | −0.7 / 4.9 | −0.3 / 3.2 |
| largest band-mean OLR bias | −12.4 | −2.7 | −2.8 |
| largest band-mean surface bias | −32.9 | −2.4 | −1.3 |
| column flux convergence, bias | +24.8 | +0.8 | +0.5 |
| layer flux convergence RMS, worst layer | 14.3 | 12.2 | 6.8 |
| doubled CO₂: ΔOLR / Δsurface (RRTMGP −3.9 / +1.9) | −3.1 / +3.3 | −4.0 / +2.0 | −3.8 / +2.0 |

The four-band form cannot hold the top and the surface together to better than
about 5 W m⁻² RMS: with one humidity-independent window, either the moist
tropics leak or the dry columns are closed. The continuum term buys 1.8 W m⁻²
of RMS at both ends and nearly halves the worst layer's error, and with it the
window's dry absorptivity returns to SPEEDY's original value.

| parameter | SPEEDY | calibrated | fitted |
|---|---|---|---|
| ``ablwin`` | 0.3 | 0.15 | 0.3397 |
| ``ablwvc`` | — | — | 0.02509 |
| ``ablco2_ref`` | 6.0 | 6.0 | 4.181 |
| ``ablwv1`` | 0.7 | 1.07 | 2.465 |
| ``ablwv2`` | 50 | 50 | 57.83 |
| ``epslw`` | 0.05 | 0.05 | 0.04947 |
| ``fbco2_a``, ``_b``, ``_t0`` | 0.148, −3.0·10⁻⁶, 247 | same | 0.2863, −5.688·10⁻⁶, 119.2 |
| ``fbwv1_a``, ``_b``, ``_t0`` | 0.356, −5.2·10⁻⁶, 282 | same | 0.3080, −3.395·10⁻⁵, 247.3 |
| ``fbwv2_a``, ``_b``, ``_t0`` | 0.314, 1.0·10⁻⁵, 315 | same | 0.2941, 4.322·10⁻⁵, 268.9 |

The fitted bands are effective ones: the "CO₂" band carries 25 % of the
spectrum at 200 K and 10 % at 300 K, and the window 2 % (the barrier) at 200 K
and 35 % at 300 K. Responses the fit did not see stay close to RRTMGP's on a
separate sample (clear-sky ΔOLR / Δsurface downward, W m⁻²):

| perturbation | RRTMGP | calibrated | fitted |
|---|---|---|---|
| +1 K, humidity fixed | +3.92 / +3.91 | +4.01 / +3.97 | +4.12 / +4.10 |
| +1 K, relative humidity fixed | +2.37 / +7.47 | +2.11 / +5.83 | +2.24 / +7.92 |
| humidity × 1.1 | −1.83 / +5.18 | −2.08 / +2.47 | −2.22 / +5.55 |

## One year with the fitted longwave

One model year (2002, prescribed SSTs, slab land) from the calibrated
configuration's spun-up state, with nothing but the fifteen longwave values
changed. *Control* is the calibrated ``speedy-t31`` as calibrated; *twin* the
same with gravity raised by one part per million (the weather noise of a
one-year mean); *reflection* the calibrated values on the scheme as it is now,
with the upward pass reflecting the downward band fluxes; *fitted* adds the
fitted longwave. Global-mean bias against the observations of
{doc}`speedy_retune` (land / ocean in brackets); fluxes in W m⁻². The surface
downward longwave is the flux reaching the surface in every column.

| field | obs | control | twin | reflection | fitted |
|---|---|---|---|---|---|
| clear-sky OLR | 268.34 | −12.75 | −12.74 | −13.47 | −7.33 (−3.1 / −9.0) |
| OLR | 240.46 | −5.36 | −5.38 | −5.85 | −9.56 (−6.6 / −10.8) |
| surface downward LW | 346.39 | −16.27 | −16.45 | −15.72 | −2.52 (+4.4 / −5.3) |
| net TOA | 1.01 | +2.13 | +2.02 | +3.21 | +7.45 (+5.5 / +8.3) |
| total cloud cover | 0.63 | −0.09 | −0.09 | −0.09 | −0.09 |
| precipitation (mm/day) | 3.07 | −0.17 | −0.18 | −0.20 | +0.07 (+1.06 / −0.33) |
| LW CRE | 27.88 | −7.39 | −7.36 | −7.62 | +2.23 |
| SW CRE | −45.73 | −6.92 | −7.04 | −6.34 | −5.66 |
| surface downward SW | 187.34 | +18.89 | +18.79 | +19.45 | +21.01 |
| precipitable water (kg m⁻²) | 24.48 | +1.92 | +1.90 | +2.00 | +0.91 |
| near-surface T ``t0`` (K) | 287.65 | +0.40 | +0.37 | +0.55 | +0.63 (+1.9 / +0.1) |
| sensible heat | 19.18 | +6.06 | +5.99 | +5.09 | +6.94 |
| latent heat | 85.72 | −1.88 | −1.94 | −2.54 | +5.08 (+11.2 / +2.6) |

Clear-sky OLR bias by band, control → fitted: 60S–30S −15.1 → −11.9, 30S–10S
−17.1 → −9.8, 10S–10N −17.2 → −9.5, 10N–30N −15.5 → −7.0, 30N–60N −9.7 → −5.0,
poleward of 60° −0.3 / +0.9 → −1.4 / +3.4. The map RMSE falls from 15.0 to
10.0 (clear-sky OLR) and from 21.8 to 15.7 (surface downward longwave), and
rises from 11.2 to 15.4 (OLR) and from 11.8 to 16.1 (net TOA).

What the fitted longwave does on its own, then: the clear-sky OLR rises by 5.4
W m⁻² to within the distance the reference itself keeps from the CERES product;
the surface receives 13.8 W m⁻² more longwave, which closes most of that
deficit; the column cools more, so the latent heat flux rises by 7 W m⁻² and
the precipitation by 0.24 mm/day (toward the observations over the ocean, away
from them over land) while the precipitable water falls by 1 kg m⁻². But the
cloud levers were calibrated against the old clear sky. With a window that
carries a third of the warm-surface emission instead of a fifth, the same
clouds trap 9.6 W m⁻² more: the LW CRE goes from 7 too weak to 2 too strong,
the all-sky OLR *falls* by 4 W m⁻², and the net TOA flux ends 7.5 W m⁻² above
CERES. The layer centred on 95 hPa cools by 3.0 K and the top layer by 0.8 K,
on top of an existing cold bias there. The one-year differences are tens to
hundreds of times the twin's.

The reflection term alone is not noise either: clear-sky OLR −0.7, OLR −0.5,
net TOA +1.1 W m⁻² against the control (twin: 0.0, 0.0, −0.1).

## What adopting the fit needs

The fitted values are a statement about the clear sky; a configuration needs
the rest refitted around them. Concretely:

- **Targets.** The net TOA flux and the all-sky OLR stay the binding ones. The
  clear-sky OLR and LW CRE targets should not be the CERES clear-scene product
  at face value: RRTMGP on reanalysis states is 10 W m⁻² below it, so a correct
  clear sky reads 7–10 low and a "correct" LW CRE against it would need the
  all-sky OLR to be as wrong. Either build the total-region clear-sky product
  into ``obs/t31`` or widen those two floors to the closure.
- **Levers.** The fifteen longwave values held fixed; then the cloud longwave
  absorptivities ``ablcl1``/``ablcl2`` (the LW CRE is now 2 W m⁻² strong and is
  what lowered the OLR), the cover thresholds ``rhcl1``/``rhcl2``/``clsmax``
  and albedos ``albcl``/``albcls`` (cover is 0.09 low with the SW CRE 6 too
  strong: brighter-per-cloud is still compensating), ``refclr``, and the
  moist-physics and exchange levers of the first calibration. The shortwave
  water-vapour and dry-air absorptivities (``abswv1``, ``abswv2``, ``absdry``,
  ``absaer``) belong in the set this time: the surface receives 21 W m⁻² too
  much shortwave, and the longwave no longer under-cools the column to hide
  how little shortwave the atmosphere absorbs.
- **The upper layers.** Add the layer temperatures at 95 and 25 hPa (ERA5) as
  targets, or give the top two layers more weight in the longwave loss: their
  flux convergence is the part of the fit that did not improve.
- **Spin-up.** Each arm needs the slab land and the hydrological cycle
  re-equilibrated (land ``t0`` +0.5 K and land latent heat +6.6 W m⁻² within
  the year); the first calibration's 395-day arms from a common state are the
  minimum.

## Provenance

- Runs (CPU; every arm on the same platform):
  ``/scr/dwatsonparris/speedy-tune/lw-fit/runs/{sample,ctl,twin,fix,fit}``.
- jax-gcm branch ``speedy-lw-fit`` (from ``speedy-tune``); the fitting and
  reference scripts are deliberately outside the repository, in
  ``/scr/dwatsonparris/speedy-tune/lw-fit/scripts`` (calibration tooling lives
  in JEM-Cal), with the report ``/scr/dwatsonparris/speedy-tune/lw-fit/REPORT.md``.
- Reference inputs: jax-rrtmgp 0.7.0; Fortran cross-check with pyrte_rrtmgp
  0.2.5 (RTE+RRTMGP 1.9); ERA5 from WeatherBench2
  (``1959-2022-6h-64x32_equiangular_conservative``); CERES-EBAF Ed4.2.1 as
  built for jcm-monitor's ``obs/t31``.
- Loeb, N. G., et al. (2020), Toward a consistent definition between satellite
  and model clear-sky radiative fluxes, *J. Climate*, 33, 61–75,
  doi:10.1175/JCLI-D-19-0381.1.
