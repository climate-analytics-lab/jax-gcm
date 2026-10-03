# Convection

**What we do.** Three interchangeable convection schemes:

- **Tiedtke-Nordeng mass-flux**
  (``jcm/physics/convection/tiedtke_nordeng/tiedtke_nordeng.py::TiedtkeConvection``)
  — the ECHAM/ICON scheme: deep, shallow and mid-level convection, convective
  momentum transport, and downdrafts (``updraft.py``, ``downdraft.py``,
  ``flux_tendencies.py``), a finite-volume scheme on the model's half levels
  (see the section below). Organized (Nordeng) entrainment and detrainment are
  the **metre-based fractional rates** of ``mo_cuascent.f90`` — organized
  entrainment carries the ``zbuoyz·0.5/(1+∫buoyancy) + zdrodz`` density-lapse
  term and organized detrainment is the ``tan``-profile in height that scales
  as 1/(cloud depth), acting from ``khmin`` up to the cloud-top bound — each
  clamped to ECHAM's hard cap ``centrmax = 3.0e-4 m⁻¹``, and per-layer
  detrained mass is capped at 0.75 of the plume (``cu_asc`` line 500) so the
  detrained-condensate ledger can never exceed the plume mass. Momentum transport is the ``cududv`` deviation-flux divergence
  with SEPARATE updraft and downdraft fluxes, each carrying its own prognostic
  plume wind (mass-weighted entrainment of the environment), the upstream
  ``jk−1`` environment offset, the sub-cloud pressure-ratio taper, and the
  explicit surface-layer closure. The ``cudtdq`` ledger keys its
  condensate-flux latent heat to the phase (``alhs`` below the melting point,
  ``alhc`` above) and writes a tendency to the surface layer (ECHAM's
  ``jk == klev`` branch). Moisture convergence *classifies* deep vs shallow
  (ECHAM's ``mo_cumastr.f90`` test), and the moisture budget of the sub-cloud
  layer both *gates* a surface plume and sets its first-guess cloud-base flux
  ``zmfub = zdqpbl/(g·max(zqumqe, zdqmin))``: the moisture the layers below
  the cloud base gain per step, exported by the cloud-base parcel's water
  excess (see the decision chain below). The deep amplitude is then set by
  the Nordeng ``zmfub1 = zcape·zmfub/(zheat·tau)`` rescale, which is where the
  CAPE-consumption timescale ``tau`` lives, and the shallow one by the
  moisture re-closure after the downdraft. Mid-level (``cubasmc``) plumes take
  neither: their base flux is the resolved ascent that triggered them. The saturation
  adjustment ``cuadjtq`` (``cuadjtq.py``) is ``mo_cuadjust.f90::cuadjtq``: one
  damped Newton step clipped by ``kcall`` (``0`` both signs for ``cuini``,
  ``1`` condensation only for ``cubase``/``cuasc``, ``2`` evaporation only for
  ``cudlfs``/``cuddraf``), then one unclipped refinement step where the first
  was non-zero. It reproduces ECHAM's compiled routine to rounding where the
  lookup tables are replaced by the Sonntag fit they hold, and to the tables'
  interpolation error (2.4e-11 K) where they are not. The precipitation budget
  (rain/snow partition, snow melt, sub-cloud Kessler evaporation, proportional
  depletion) transcribes ECHAM ``cuflx`` (``flux_tendencies.py``,
  ``mo_cufluxdts.f90``). The fractional precipitation cover the sub-cloud
  evaporation acts over follows ECHAM's submodel dependence
  (``mo_cufluxdts.f90:414-420``): plain ECHAM uses the constant
  ``zcucov = 0.05``, but with the JAM aerosol chain composed
  (``aerosol_module='jam'``, jcm's analogue of ECHAM's ``lham``) it uses the
  **updraft area** ``pmfu/(zwu·ρ_u)`` — ``ρ_u = p/(R_d·T_u)`` the updraft
  density, ``zwu = 2`` m/s the assumed in-cloud updraft speed — evaluated
  over the plume profile and ECHAM's sub-cloud taper. This is the identical
  footprint the JAM convective wet deposition uses for below-cloud washout
  (``updraft_area_cover`` is shared between the two), so the rain the
  aerosol scavenging sees and the rain the evaporation depletes agree by
  construction.
- **SPEEDY convection** (``jcm/physics/convection/speedy_convection.py::diagnose_convection``)
  — SPEEDY's simplified Tiedtke (1993) mass-flux scheme with a
  conditional-instability trigger on saturation moist static energy.
- **Betts-Miller**
  (``jcm/physics/convection/betts_miller/betts_miller_terms.py::BettsMillerConvection``;
  core ``betts_miller.py``) — a faithful port of Isca's ``betts_miller.f90``
  (Frierson 2007 Simplified Betts-Miller), relaxing T and q toward a moist-adiabatic
  reference at RH ``rhbm`` over ``tau_bm``, with ``do_shallower`` / ``do_changeqref``
  siblings. Written broadcasting-native (vertical on axis 0).

**Tiedtke's decisions are ECHAM's.** Whether a column convects, which plume it
carries, where the plume stops and where it rains are hard comparisons in
``cumastr``/``cuasc``, and the port makes each exactly as ECHAM6.3 does:

1. ``cubase`` finds a buoyant cloud base (the section below), or ``cubasmc`` a
   mid-level one where no surface plume is possible.
2. ``zlo1`` (``mo_cumastr.f90:560-579``): a ``cubase`` column convects only
   if its sub-cloud layer gains moisture, ``zdqpbl = Σ_{jk ≥ kcbot} pqte·Δp >
   0``, and the cloud-base parcel carries more water than the environment
   there, ``zqumqe = pqu + plu − pqenh > zdqmin = max(0.01·pqenh, 1e-10)``.
   ``pqte`` is the whole pre-convection moisture tendency, the same-step
   vertical diffusion (which contains the surface evaporation it delivered)
   plus the one-step-lagged dynamics; a standalone caller that gives only the
   surface evaporation has it delivered to the lowest layer. A column that
   fails the gate is not convective, and ``cubasmc`` may seed a mid-level
   plume in it instead.
3. The type: deep exactly where ``zdqcv = Σ pqte·Δp`` exceeds
   ``zhelp = max(0, 1.1·E·g)`` (``ktype = FSEL(zhelp − zdqcv, 2, 1)``,
   lines 571-574), else shallow; a deep plume thinner than 200 hPa after the
   first ascent is demoted to shallow.
4. The ascent test at each interface (``mo_cuascent.f90:442-465``): the plume
   continues only if it condensed there, its buoyancy ``zbuo`` (plus
   ``zlift`` on a mid-level plume's first step) is positive, it carries at
   least 1 % of the cloud-base flux and the interface lies at or below the
   cloud-top bound. The first interface that fails ends the ascent: no level
   above it is visited (the ``klab = 0`` latch, line 294), and a fraction
   ``cmfctop`` of the flux overshoots into it. A plume based at ``klevm1``
   that passes no interface above it leaves ``kctop = klevm1`` and the column
   non-convective (line 541) — a cloud base higher up counts as passed, its
   test in ``cuasc`` repeating ``cubase``'s; a column that the first ascent
   leaves non-convective runs no surface plume in the second.
5. Precipitation starts where the interface lies ``zdnoprc`` or more above
   cloud base (1.5e4 Pa over sea, 3e4 Pa over land: a land fraction of 0.5
   or more, as ECHAM's default binary land-sea mask ``slm`` classifies it).

ECHAM has no CAPE trigger, and neither does the port.
``jcm/data/test/echam_cumastr_reference`` holds what ECHAM6.3's compiled
``cucall`` returns for 758 columns — states of the whole-model RCE column in its
earlier grey-radiation configuration, where the
first ascent test above a cloud base at ``klevm1`` decides whether the column
convects, and the same states under a resolved ascent (mid-level plumes), a
moisture convergence (deep plumes) or a sub-cloud divergence and a humid
sub-cloud layer (both ``zlo1`` conditions): in float64 with ECHAM's physical constants the port takes the same
decision (convective or not, type, cloud base, cloud top) on every column,
and its cloud-base mass flux, surface precipitation and per-level tendencies
agree to 2.1e-12 or better (``cumastr_reference_test.py``).

Each decision's derivative is a named surrogate's (``switches.py``; see
{doc}`../design/surrogate_gradients`): the value is ECHAM's, and the
derivative is that of a logistic of the switching quantity, of width
``ascent_buoyancy_width`` (0.01 K), ``ascent_mass_flux_width`` (2e-3 of the
cloud-base flux), ``ascent_condensate_width`` (1e-8 kg/kg, a rescaled
logistic that is exactly zero without condensation),
``precip_onset_width`` (2000 Pa), ``deep_convergence_width`` and
``sub_cloud_supply_width`` (2e-7 kg m⁻² s⁻¹) or
``cloud_base_excess_width`` (0.1 of ``zdqmin``). They are static fields of
``ConvectionParameters``; zero selects the reference derivative. The
per-level ascent test is one surrogate, used for the continuing flux, the
overshoot and, at the first interface above a cloud base at ``klevm1``, the
column's ``ldcum``; the chain of decisions that makes ``ldcum`` weights the
whole ledger and the published mass fluxes, each link only where the links
before it passed, so a column that is off carries the derivative of the
switch that turned it off, applied to the ledger of the plume the scheme ran
for it (at ECHAM's first-guess flux for a plume the ``zlo1`` gate rejects).
Tiedtke-Nordeng's saturation is ECHAM's
``ua`` table, Sonntag (1990) over ice at and below the melting point and over
water above (``jcm/physics/convection/tiedtke_nordeng/cuadjtq.py`` on
``jcm/physics/thermodynamics.py``; see {doc}`constants`), with the ``cuadjtq``
latent heat switching at the same point (``mo_cuadjust.f90``,
``mo_echam_convect_tables.f90::lookup_ubc``). Betts-Miller follows Isca and
saturates over water with the Tetens form of
``jcm/physics/convection/saturation.py``.

**What ECHAM/CAM does.** ECHAM6-HAM2.3 uses the **Tiedtke (1989) bulk mass-flux
scheme with Nordeng (1994) CAPE closure** (``mo_cumastr.f90`` master driver,
``mo_cuasc.f90`` / ``mo_cuascn.f90`` updraft ascent, ``mo_cudlfs`` / ``mo_cuddraf``
downdrafts, ``mo_cuadjust.f90`` saturation adjustment,
``mo_cufluxdts.f90`` fluxes). References: Tiedtke, M. (1989), *Mon. Wea. Rev.* 117,
1779-1800; Nordeng, T.E. (1994), ECMWF Tech. Memo. 206. Betts-Miller's reference
is Betts & Miller (1986) as simplified by Frierson, D.M.W. (2007), *J. Atmos. Sci.*
64, 1959-1976 (Isca ``betts_miller.f90``).

**Why we differ.**
- `differentiability` — the decisions are ECHAM's in the value and carry the
  derivatives of logistic surrogates (above). **The level choices stay
  discrete**: the ``cubase`` cloud base, the mid-level seed level and its
  trigger conditions, the cloud-top bound and the 200 hPa demotion are level
  indices or trigger identities and carry no derivative, so gradients do not
  flow across the onset of a cloud base or of a mid-level plume.
- `science` — jcm's physical constants are unified across its schemes. Its
  vapour gas constant is ECHAM's (``rv = 461.51`` J/(kg K)); its ``grav``
  (9.81 against 9.80665 m/s²) and latent heats (2.501e6 and 2.834e6 against
  2.5008e6 and 2.8345e6 J/kg) differ from ECHAM's by 0.008-0.035 %. That
  still moves marginal decisions: the ascent test at the first interface
  above a cloud base at ``klevm1`` sits within hundredths of a kelvin of zero
  in that grey-radiation RCE column. On the 758 reference columns jcm's
  constants change 2 cloud tops, and ECHAM's latent heats bring both back. On
  that column's own days 40-80 states the port, run in float32 as the
  model runs it, convects in 15.1 % of the steps and ECHAM6.3 in 15.0 %, the
  two differing in 16 of the 3840 steps; in float64 with ECHAM's constants
  the port takes ECHAM's decision on every step.
- `science` — a downdraft ``cuflx`` cancels is removed whole. ECHAM zeroes
  it only from ``kctop − 1`` down and keeps the levels above as ``cuddraf``
  left them, absolute (not deviation) heat and moisture fluxes, which
  ``cudtdq`` then applies (``mo_cufluxdts.f90:189-212``, ``mo_cudescent.f90``
  l.164, 312): where the level of free sinking is two or more interfaces
  above the final top (4 of 1.1 million column-steps over half a day of
  ``t63-echam-1m``) that is a heating dipole estimated at order 10 K/hr,
  which jcm does not reproduce. Wherever it is at most one interface above,
  the two agree.
- `science` / `compute` (stopgap) — ECHAM bounds the mass flux, not the
  heating: the cloud-base flux and the per-level entrainment are held to the
  layer's air mass per step (``zmfmax = layer_mass/dt``, ``mo_cumastr.f90``
  and ``mo_cuascent.f90::cuasc``, both ported). jcm adds a safety net ECHAM
  does not have (#961): an explicitly-labelled stopgap caps the convective
  T-tendency at 5 K/hr
  (``_DTDT_MAX``) and rescales the **whole** ledger homogeneously — T, q,
  qc/qi, precipitation, the mass fluxes with the tracer transport they drive,
  **and the momentum tendencies** ``dudt``/``dvdt``, which share the same mass
  flux — preserving column conservation by linearity, as ECHAM's ``zmfub1``
  amplitude scaling does. This cap is the documented cause of a cap-pinned
  single-layer heating artifact in pathological columns.

**Status & known limitations.**
- The 5 K/hr tendency cap is a **safety net, not physics**; it fires only where
  the parcel-vs-environment balance has gone pathological (healthy tropical deep
  convection is ~1 K/hr). It has no ECHAM counterpart; whether it can go is
  a measurement of where it fires and of stability without it (#961).
- A ``cubase`` column whose sub-cloud layer loses moisture (vertical
  diffusion carrying more out through the cloud base than the surface and
  the dynamics bring in), or whose cloud-base parcel is within 1 % of the
  environment's humidity, is not convective, as in ECHAM; only ``cubasmc``
  can then give it a plume.
- SPEEDY and Betts-Miller are idealized alternatives; Betts-Miller is
  specific-humidity-formulated (Isca's mixing-ratio form differs at second order).
- **Tiedtke creates water where the downdraft out-takes the plume's rain
  (#912).** As in ECHAM, ``cumastr`` scales the first ascent's downdraft to the
  closed flux and re-runs ``cuasc``, and ``cuflx`` floors the rain and snow
  fluxes at zero while the vapour ledger keeps the unfloored
  ``pdmfup + pdmfdp``. A second ascent that rains less than the scaled
  downdraft takes up (a deep-to-shallow demotion) therefore creates the
  difference as water. The amount is published as
  ``convection.precip_floor_source``, so a column budget closes as
  ``E - P + precip_floor_source``; in the whole-model RCE column, with grey
  or RRTMGP radiation, it is below 0.001 mm/d over days 40-80
  ({doc}`../design/rce_testbed`).

**Code pointers.**
- ``jcm/physics/convection/tiedtke_nordeng/`` — ``tiedtke_nordeng.py``
  (``TiedtkeConvection``, the CFL cap, ``_DTDT_MAX``), ``switches.py`` (the
  decisions' exact values and surrogates: ``ascent_test``,
  ``threshold_switch``, ``relative_threshold_switch``), ``cuadjtq.py`` (``cuadjtq``,
  ``saturation_mixing_ratio``, ``cuadjtq_newton``, ``cuadjtq_newton_evap``), ``flux_tendencies.py``
  (``convective_precip_fluxes`` [ECHAM cuflx]),
  ``updraft.py``, ``downdraft.py``.
- ``jcm/physics/convection/speedy_convection.py`` — ``diagnose_convection``.
- ``jcm/physics/convection/betts_miller/`` — ``betts_miller.py``,
  ``betts_miller_terms.py`` (``BettsMillerConvection``).
- Convective *tracer* transport and in-plume scavenging live with the aerosol
  chain — see {doc}`aerosol`.

**Validation evidence.** ``jcm/physics/convection/tiedtke_nordeng/`` test suite
(``tiedtke_nordeng_test.py``, ``half_level_ledger_test.py``,
``cuadjtq_test.py`` (ECHAM's compiled ``cuadjtq``, all three ``kcall``
modes), ``updraft_test.py``,
``downdraft_test.py``, ``deep_shallow_test.py``, ``midlevel_trigger_test.py``,
``rce_integration_test.py``, ``convection_units_test.py``,
``cumastr_reference_test.py`` (ECHAM6.3's compiled convection on 758
columns), ``switches_test.py`` and ``surrogate_gradients_test.py`` (the
decisions' surrogate derivatives), ``cuasc_port_test.py``,
``ledger_entrainment_test.py``);
``betts_miller/betts_miller_test.py``; ``speedy_convection_test.py``.

## The Tiedtke ledger on half levels

**What we do.** The Tiedtke scheme is ECHAM's finite-volume scheme on the
model's own layers. ``half_levels.py::half_level_environment`` is ``cuini``:
it builds, from the host's interface pressures (``pressure_half``), the
hydrostatic half-level geopotential ``pgeoh`` (virtual temperature loaded with
the cloud condensate), the half-level heat capacity ``pcpcu`` (mean of the two
adjacent ``pcpen``), and the half-level environment ``ptenh``/``pqenh`` — the
larger adjacent dry static energy carried to the interface, saturation
adjusted (``cuadjtq`` ``kcall = 0``) with the saturation humidity of the level
above, then monotonised so the interface dry static energy never decreases
upward; the humidity is the level above's plus the change of saturation
humidity to the interface. Every plume property and mass flux lives on those
interfaces — in the physics-internal top-first frame, entry ``k`` is the value
at the TOP interface of layer ``k``, exactly ECHAM's ``klev``-long half-level
arrays, and the surface interface carries no flux:

- ``cubase`` walks the parcel up the interfaces from the lowest one;
  ``cubasmc`` seeds a mid-level plume at the bottom interface of its layer.
- ``cuasc`` (``updraft.py``) carries the plume from interface ``k+1`` to ``k``
  across layer ``k``: the flux entering from below plus the entrained
  half-level environment of interface ``k+1`` minus the detrained air,
  condensation-only ``cuadjtq`` at the interface pressure with the plume's
  condensate carried separately from its vapour, buoyancy against the
  half-level environment, precipitation over the layer's geopotential depth,
  and the ``zmfmax`` entrainment limiter (the flux leaving an interface cannot
  exceed the air mass of the layer above per step). The prognostic plume wind
  is ECHAM's running momentum flux with the ``zz`` detrainment enhancement,
  and the Nordeng integrated buoyancy starts from the cloud-base buoyancy plus
  the sub-cloud parcel's, as cuasc's level loop accumulates it.
- ``cuentr`` sets the rates. Turbulent detrainment ``pentr·pmfu·Δz`` acts in
  every layer above cloud base and leaves at the properties of the plume that
  entered the layer; turbulent entrainment at the same rate acts only in each
  plume type's band — deep plumes at and below the level of maximum resolved
  ascent ``klwmin`` (``cuini``, not above ``kctop0 + 2``) or in the lower half
  of the cloud (below the pressure midway between cloud base and ``kctop0``),
  shallow plumes within 200 hPa of cloud base or in the lower half, mid-level
  plumes at and below ``klwmin``. A mid-level plume there also entrains the
  pre-convection moisture convergence (``zentest``: the layer's positive
  ``pqte`` over the half-level humidity, as a fractional rate capped at
  ``centrmax``, where that humidity exceeds 1e-5 kg/kg). Organized
  detrainment of deep plumes acts from ``khmin`` — the level where
  ``cumastr``'s height-weighted moist-static-energy lapse first exceeds the
  environment's saturation deficit, at or above ``ictop0`` — up to ``kctop0``;
  below ``khmin`` it is limited by ``zodmax``, the detrainment that would bring
  the plume's moist static energy back to the cloud-base value at ``kctop0``.
  Organized detrainment removes air with the static energy and humidity that
  are neutral against the environment of the layer's bottom interface
  (``zscod``/``zqcod``).
- The ascent test ends the plume at the first interface where it does not
  condense, is not buoyant (condensate-loaded virtual temperature against the
  half-level environment, with ``zlift`` on a mid-level plume's first step),
  carries less than 1 % of the cloud-base flux, or lies above the cloud-top
  bound ``kctop0``; no interface above it is visited. The last interface that
  passed is the cloud top ``kctop``. At the interface above it a fraction ``cmfctop`` of the flux
  there overshoots with the properties the ascent gave it and no
  precipitation; the rest detrains in that layer with the plume's condensate,
  and the overshoot's own condensate detrains in the layer above. A plume
  based at ``klevm1`` that passes no interface above it leaves ``kctop`` at
  ``klevm1``, which makes the column non-convective (``ldcum`` false) — a
  ``cubase`` plume's seed interface counts, since the test there repeats
  ``cubase``'s.
- ``kctop0`` is ``cumastr``'s first-pass estimate: ``ictop0``, the highest
  interface at least two above cloud base where the cloud-base parcel's moist
  static energy exceeds the environment's reduced saturation value
  (``zhhatt``), else the interface just above cloud base; for a column without
  a surface plume, where a mid-level plume may start, the lowest interface
  above 400 hPa. As in ``cumastr``, a first ascent at the first-guess flux
  feeds the closure and the downdraft, and a second ascent at the closed flux
  — bounded by the first ascent's realized top, and with the entrainment of
  the demoted type where a thin deep cloud is relabelled shallow — produces
  the final plume; the downdraft is scaled, not re-run. A column whose first
  ascent passes no interface is non-convective; a surface-plume column whose
  first ascent fails in this way may still take a mid-level plume in the
  second, as ``cuasc``'s ``cubasmc`` would seed it. The deep ``zmfub1`` is
  floored at 0.001 kg m⁻² s⁻¹ (``mo_cumastr.f90:902``) before the CFL cap.
- ``cudlfs``/``cuddraf`` (``downdraft.py``) search the interfaces strictly
  inside the realized cloud for the level of free sinking and descend
  interface to interface, charging the rain the downdraft evaporates to the
  layer it crossed; below ``itopde`` the downdraft detrains linearly in
  pressure to zero at the surface.
- ``cuflx`` forms deviation fluxes against the half-level environment
  (``pcpcu·(T_plume − ptenh)·M`` etc.) and, below the cloud-base interface,
  replaces the updraft fluxes by the cloud-base values scaled by
  ``(p_s − p_half)/(p_s − p_half(kcbot))`` (squared for mid-level plumes), so
  the plume draws its air from, and deposits the cloud-base flux divergence
  through, the whole sub-cloud layer. The updraft mass flux the term
  publishes (``ConvectionData.mass_flux_up``) carries the same taper, so the
  convective tracer transport draws the cloud-base supply from the same
  layers, as ECHAM's ``pmfuxt`` does.
- ``cudtdq``/``cududv`` give each layer the difference of the fluxes through
  its two interfaces plus its per-layer sources, divided by its true air mass
  ``Δp/g`` — the same mass the host applies the tendencies with. The flux
  differences telescope, so the column water changes by exactly minus the
  surface convective precipitation and the column enthalpy by the latent heat
  of the vapour removed.
- The closure quantities are the reference's half-level ones: the moisture
  budget's ``q_u − q_e`` is ``pqu + plu − pqenh`` at the cloud-base interface,
  the CFL cap is the air mass of the layer above it per step, the Nordeng
  ``zheat``/``zcape`` integrals run over the interfaces
  ``kctop < jk ≤ kcbot``, and the depth demotion uses interface pressures.

**What ECHAM does.** ``mo_cuinitialize.f90::cuini``/``cubase``,
``mo_cuascent.f90::cuasc``/``cubasmc``/``cuentr``,
``mo_cudescent.f90::cudlfs``/``cuddraf``,
``mo_cufluxdts.f90::cuflx``/``cudtdq``/``cududv``, driven by
``mo_cumastr.f90::cumastr``.

**Why we differ.**
- `science` — the ``cubase`` sub-cloud plume wind is the pressure-weighted
  mean over ALL sub-cloud layers; ECHAM's loop accumulates only the layers it
  visits after the cloud base is set, so for a base above the lowest two
  interfaces its weights do not sum to one.
- `differentiability` — the ascent test and the precipitation onset are
  ECHAM's switches in the value with surrogate derivatives (the decision
  chain above). The level choices (``klwmin``, ``ictop0``, ``khmin``,
  ``kctop0``, the entrainment bands) stay discrete, as level indices.
- `science` — ``cuflx`` cancels a downdraft whose level of free sinking, found
  in the first ascent's cloud, lies above the final plume's top
  (``mo_cufluxdts.f90:189``), and then zeroes it only from ``kctop − 1``
  down, leaving the absolute (not deviation) fluxes of any downdraft level
  above that in its ledger. jcm removes the whole downdraft, which is the same
  wherever the level of free sinking is at most one interface above the top
  (every case in the reference data).
- `science` — the sub-cloud rain evaporation's ``cevapcu`` profile uses the
  column's ``p/p_s`` for ECHAM's ``ceta``, the grid's full-level hybrid
  coordinate at the reference surface pressure: the same on a sigma grid,
  and within the ``p_s/101325`` ratio of the ``a`` term on a hybrid one.
- `differentiability` — the profile's leading coefficient, ECHAM's hard-coded
  ``1.93E-6`` (``iniphy.f90:87-89``), is the parameter
  ``ConvectionParameters.cevapcu``, defaulting to ECHAM's value, so that
  calibration and gradients reach the sub-cloud evaporation; it scales the
  whole level-dependent profile. As in ECHAM (``mo_cufluxdts.f90:428-432``) a
  layer evaporates the smaller of the Kessler chain and the amount that
  moistens it to 80 % of saturation in one step, so where that cap binds
  (warm, dry sub-cloud air under a long step) the coefficient does not
  change the result.
- `science` — the overshoot never reaches the top interface of the model:
  jcm's ascent test fails above interface 2 (0-based), where a flux through
  the model top would leave the column. ECHAM's level loop could place the
  overshoot there only for ``kctop0 ≤ 1``, which neither of its bounds gives
  on a grid with more than two interfaces above 400 hPa.

**Status & known limitations.** The ``cuasc``/``cuentr`` ascent, its
``cumastr`` bounds, and the ``cuini`` levels it uses are ported in full; the
deviations are the ones listed above.

Operational notes:
- In a prescribed (re-imposed) column the deep/shallow split can only see
  large-scale convergence one step late (the lagged ``pqte``), and the
  strongly entraining shallow plume (``entrscv``, detrainment at the incoming
  plume's properties) does not precipitate in the warm tropical check column,
  so such a column stays shallow unless it starts under convergence —
  ``jcm/rce.py::convergent_initial_physics_data`` supplies that for the JAM
  aerosol-pathway checks.
- In the top few levels (below ~150 Pa) ``cuini``'s saturation adjustment works with
  ECHAM's capped ``x = MIN(es·rd/rv/p, 0.5)``, so its saturation humidity sits
  at ``0.5/(1 − 0.5·vtmpc1) ≈ 0.72`` and its interface values are not
  physical, as in the reference; no plume reaches them.

## Heat capacity of the Tiedtke plume and ledger

**What we do.** Every static-energy and latent-heat conversion the Tiedtke
port shares with ECHAM ``cumastr`` uses the **moist** isobaric heat capacity
``cp = cpd·(1 + vtmpc2·max(q, 0))``
(``jcm/physics/thermodynamics.py::moist_isobaric_heat_capacity``), evaluated
per level from the **step-start** humidity: the cloud-base and mid-level parcel
lifts (``find_cloud_base``, ``find_midlevel_cloud_base``, the ``calculate_updraft``
seed), the updraft and downdraft dry-static-energy mixing
(``calculate_updraft``, ``calculate_downdraft``), the Nordeng ``zheat`` lapse term,
and the ``cudtdq`` ledger — the DSE deviation fluxes ``cp·(T_plume − T)·M`` and
the conversion of the whole heat ledger to a temperature tendency. The column
enthalpy the ledger deposits is therefore ``Σ cp·dT·Δp/g``. Three sites keep
dry ``cpd`` because the reference does: the ``cuadjtq`` Newton step and the
wet-bulb adjustment (``cuadjtq.py``), and the ``cuflx``
melting constant, which applies its own ``(1 + vtmpc2·q)`` factor with the
provisional humidity. jcm's own trigger diagnostic ``calculate_cape_cin`` has no
ECHAM counterpart and uses the textbook dry-``cpd`` parcel.

**What ECHAM does.** ``mo_cumastr.f90:229`` builds ``zcpq = cpd·(1 +
vtmpc2·MAX(pqm1, 0))`` from the step-start humidity ``pqm1`` (not the
provisional ``zqp1`` the plume sees) and passes it as ``pcpen``; ``cuini``
averages it to half levels (``pcpcu``). ``cubase``/``cuasc``/``cubasmc``/
``cuddraf`` carry plume heat as ``pcpcu·T + pgeoh``; ``cuflx`` subtracts the
environment's ``pcpcu·ptenh + pgeoh`` (``mo_cufluxdts.f90:198-204``);
``cudtdq`` divides by ``pcpen`` (``zrcpm``, ``mo_cufluxdts.f90:648``); ``zheat``
uses ``zcpcui = 1/zcpcu`` (``mo_cumastr.f90:598/849``). ``cuadjtq`` reads
``L/cp`` from tables built with ``alv/cpd`` (``mo_echam_convect_tables.f90:214``).

**Why we differ.** We do not: the port matches the reference's choice at every
site. Because ``cp`` is the **environment's** at each level, a parcel lifted
through an environment that dries with height gains ``≈ T·vtmpc2·Δq_env``
relative to a dry-``cpd`` lift (the heat content is carried with the lower
level's larger ``cp`` and divided by the upper level's smaller one). That is
the reference's thermodynamics and ECHAM's convective parameters were tuned
with it, so it is kept rather than replaced by the parcel's own heat capacity.

**Status & known limitations.** As in the reference, ``pcpen`` is the
full-level value and ``pcpcu`` its half-level mean (``cuini``); the plume and
the deviation fluxes use ``pcpcu``, the tendency conversion ``pcpen``.

## Cloud-base trigger and the sub-cloud layer

**What we do.** Tiedtke's cloud base is ECHAM's ``cubase`` ``klab`` walk
(``jcm/physics/convection/tiedtke_nordeng/tiedtke_nordeng.py::find_cloud_base``):
a parcel starts at the lowest half level (the top of the bottom layer) with the
half-level environment's temperature and humidity and is lifted up the
interfaces conserving dry static energy. At each interface a dry
buoyancy test ``zbuo = Tv_u - Tv_e + zlift`` decides whether the walk continues;
the first level at which the parcel condenses is the LCL, where a second test —
the same buoyancy with condensate loading — decides whether a cloud base exists.
``zlift`` is the sub-grid thermal excess of the warmest boundary-layer plumes,
``min(clip(thvsig·cbfac, cminbuoy, cmaxbuoy), 1)`` K, taken from vdiff's
prognostic θ_v variance where one is available and from
``ConvectionParameters.cu_thvsig`` otherwise. A column whose parcel loses its
buoyancy before reaching its own LCL gets **no surface-based convection at all**,
whatever its CAPE: the walk stops, and CAPE never enters. The second,
independent entry is the mid-level ``cubasmc`` trigger
(``find_midlevel_cloud_base``), which starts a plume with no surface connection
where the environment is nearly saturated and resolved ascent is lifting it.

**What ECHAM does.** ``mo_cuinitialize.f90::cubase`` (the ``klab`` walk, ``zlift``
from ``pthvsig``) and ``mo_cuascent.f90::cubasmc`` (the mid-level trigger).
ECHAM evaluates the walk on half levels whose environment temperature is the
**dry-static-energy upper envelope** of the two adjacent full levels
(``mo_cuinitialize.f90::cuini``: ``ptenh = (MAX(s(jk-1), s(jk)) - geoh)/cpm``,
saturation adjusted with the humidity of the level above, then monotonized
upward), which flattens any dry-neutral or dry-unstable layer before the
parcel is compared against it.

**Why we differ.** We do not: the walk runs on the reference's half levels
(see the half-level section above), and the parcel starts with ECHAM's seed
energy ``pcpcu(klev)·ptenh(klev) + pgeoh(klev)``, the lowest interface's
environment with the half-level heat capacity — about 0.13 K colder, per
g/kg of humidity drop between the lowest two levels, than the bottom full
level's ``pcpen·pten + pgeo``. The ``cubasmc`` seed likewise carries ECHAM's
``pcpen(kk+1)·ptu(kk+1) + pgeoh(kk+1)``. ``calculate_cape_cin``, a
surface-parcel CAPE diagnostic, is not part of the scheme.

**Status & known limitations.**
- The trigger is **strict by construction, and this is the reference's
  behaviour, not an approximation of it**: because ``zlift`` is capped at 1 K, a
  sounding whose sub-cloud layer is stably stratified loses more parcel
  buoyancy per level than the excess can cover and never reaches its LCL. The
  walk's moist heat capacity (see the heat-capacity section above) credits the parcel
  ``≈ T·vtmpc2·Δq_env`` per level where the environment dries with height, so
  a moist 6.5 K/km surface layer can still reach its LCL; a genuinely stable
  (e.g. 4 K/km or inverted) one cannot. Convection in such
  a column is the job of ``cubasmc``, which needs resolved ascent and a nearly
  saturated environment.
- It follows that **any prescribed or idealised column handed to Tiedtke assumes
  a well-mixed (dry-adiabatic) sub-cloud layer**, the state a prognostic run's
  vertical diffusion maintains and real tropical soundings have. The
  release-validation single-column state
  (``jcm/rce.py::rce_initial_state``, ``jcm/rce.py::jam_scavenging_column``)
  supplies one explicitly via ``mixed_layer_top_m``, since a prescribed column
  re-imposed every step never lets vdiff build it. Failure of this assumption is
  silent: the plume simply never exists, while turbulent mixing keeps the
  profiles plausible. See {doc}`../design/convective_trigger_soundings`.
- A column carries **one plume per step**, as in the reference: ``cuasc``
  calls ``cubasmc`` only while ``ldcum`` is false in the current ascent, and a
  ``cubase`` plume sets it at its own cloud base, whose ascent test repeats
  ``cubase``'s — so no mid-level plume starts above a surface plume that dies
  partway up. Where a mid-level seed's first step fails, the seed moves to
  the next qualifying level up, as ``cubasmc``'s does.
