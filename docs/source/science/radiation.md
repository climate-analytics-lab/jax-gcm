# Radiation

**What we do.** The ECHAM stack runs one of two radiation backends, RRTMGP or
its neural-network emulator; SPEEDY carries its own; and an idealized grey
two-stream scheme exists alongside them for idealized studies and cheap tests.
Each is a composable ``PhysicsTerm`` (SPEEDY's is a function pair in the SPEEDY
term list):

- **RRTMGP correlated-k** (``jcm/physics/radiation/rrtmgp.py::RRTMGPRadiation``) —
  the production backend. Wraps the external ``jax-rrtmgp`` library behind the
  ICON radiation interface. The per-column entry point (``radiation_scheme_rrtmgp``)
  is vmapped over columns; it handles TOA-first ↔ surface-first reordering, a
  pressure halo that encodes the model's exact boundary-layer Δp, and a
  float32-end-to-end library boundary with a scoped ``jax.enable_x64(False)`` so
  it is bit-consistent on x64 hosts (pySES / MAM4-JAX).
- **SPEEDY SW/LW** (``jcm/physics/radiation/speedy_shortwave.py``,
  ``speedy_longwave.py`` — 4-band LW).
- **NN emulator** (``jcm/physics/radiation/nn_emulator_scheme.py::NNEmulatorRadiation``,
  network in ``nn_emulator.py``) — a bidirectional-GRU emulator of RTE+RRTMGP.
  See {doc}`../design/radiation_nn_emulator`.
- **Grey two-stream (idealized)**
  (``jcm/physics/radiation/grey_two_stream/radiation_scheme.py::GreyTwoStreamRadiation``)
  — an idealized scheme in the same class as Betts-Miller convection: broadband
  (two SW, three LW bands) with its own gas optics and Planck bands, solved with
  the delta-Eddington two-stream layer solution (Joseph, Wiscombe & Weinman
  1976; Meador & Weaver 1980), the Toon et al. (1989) direct-beam source and
  the Shonk & Hogan (2008) adding solve. It is **not ECHAM physics**: it has no
  ECHAM reference formulation, is not calibrated against RRTMGP, and has never
  been validated in an ECHAM composition, so ``echam_physics()`` does not offer
  it (``radiation_scheme="grey"`` raises). It exists as a cheap, differentiable,
  energy-conserving radiative driver for idealized work — the RCE columns of
  ``jcm.rce`` — and for tests whose subject is not radiation, which compose it
  explicitly through ``jcm.physics.echam.testing.idealized_echam_physics``. See
  *The grey two-stream (idealized)* below.

**Partial-cloud / overlap** differs by backend. **RRTMGP** uses full **McICA**
(``jcm/physics/radiation/mcica.py``): one stochastic binary cloud profile per
g-point, seeded deterministically per column and model step, with three overlap
rules — random, maximum-random (Geleyn-Hollingsworth: maximum overlap of
adjacent cloudy layers, random across a clear layer), and
generalised-exponential with a decorrelation length
(``RadiationParameters.cloud_overlap``, ``cloud_decorrelation_km``). The
default is **maximum-random**, ECHAM6.3's default (``i_overlap = 1``,
``mo_radiation_parameters.f90`` l.71), with the rank rule of ECHAM's
``mo_cld_sampling.f90::sample_cld_state`` (l.66-83). ECHAM runs that chain
from the top down on the surface-first column its ``psrad_interface`` hands
the radiation (``mo_psrad_interface.f90`` l.221-227): a sub-column keeps the
rank of the level above where it is cloudy there and otherwise draws a new
rank in that level's clear part. jcm runs the same rule from the bottom up on
its top-first column; the two directions give every sub-column cloud pattern
the same probability, so the expected total cover is ECHAM's ``cld_cvr``,
the adjacent-layer Geleyn-Hollingsworth product (``mo_radiation.f90``
l.436-442). The rank
comparisons are piecewise constant in the cover, as the ``r < cf`` test of
every rule is, so the sampled masks carry no cover gradient and need no
surrogate. ECHAM's sampler also offers random overlap (and maximum, which jcm
does not); exponential is a jcm option with no ECHAM counterpart. The **grey**
backend instead combines one clear and one cloudy beam weighted by
``column_total_cover``, which is the column's largest cover under both
maximum-random and exponential overlap (a closed-form approximation, not
ECHAM's adjacent-layer product above) and ``1 - ∏(1 - f)`` under random; the
**NN emulator's** fluxes carry whatever overlap its RRTMGP training labels
embedded — the network sees only layer cloud fractions and paths, so the
runtime ``cloud_overlap`` / ``cloud_decorrelation_km`` knobs change its
*reported total-cover diagnostic* (a post-hoc ``expected_total_cover``) and
not its heating or fluxes; **SPEEDY** carries its own cloud formulation.
Swapping backends therefore changes the cloud-overlap treatment, not just the
gas optics. The AeroCom total-cloud-cover diagnostic uses the maximum-random
closure.

The total cloud cover jcm reports is also maximum-random: ECHAM's own ``aclcov``
(``mo_cloud.f90`` §10.2), **computed in the model every step** from the final
cloud fraction and saved as ``clouds.total_cloud_cover`` (time-averaged under
``run.output_averages``, which is ECHAM's accumulation); it is what the
release-validation ``cloud_cover`` gate scores. For output that carries no such
field, :func:`jcm.analysis.total_cloud_cover` applies the same recurrence to a
saved profile, which reads lower when that profile is a time mean. That choice
defers to ECHAM and is deliberate: overlap is a definition, the three in common
use differ by ~0.3 in the global mean, and a total cover is the basis the
satellite climatologies are quoted on. It is a different number from the McICA
``radiation.total_cloud_cover`` above — a sampled quantity from a
differently-preprocessed fraction — and {doc}`../design/cloud_cover_gate` sets
out the provenance, the measured magnitudes and how far apart the three run.

Radiation **sub-steps** on the RRTMGP, NN emulator and grey backends: a gate
(``radiation_should_compute``) skips the expensive solve and rescales cached
heating on intermediate steps. SPEEDY has its own, different
cadence — ``SpeedyFlags`` gates shortwave every ``nstrad`` calls, and the
skipped calls re-apply the heating rate cached from the last solve
(``SWRadiationData.heating_rate``, SPEEDY's ``tt_rsw``) rather than rescaling
it, which is what the Fortran does. The two families therefore differ in how
they *reuse* the cached solve, not in whether they reuse it.

**Aerosol-radiation coupling** is per-band: MACv2-SP simple plumes and JAM online
optics both feed per-band aerosol optical depth / SSA / asymmetry into RRTMGP. For
the AeroCom ERFari diagnostic an **aerosol-free companion solve** reruns radiation
with the aerosol optics zeroed but the cloud field bit-identical; its cadence is
one integer knob (``jcm/physics/radiation/aerosol_free.py``). See
{doc}`../design/aerocom_erfari_sampling`.

**Cloud effective radii** (RRTMGP and the NN emulator) are formed inside the
radiation call from the current step's state, as ECHAM6.3-HAM2.3 forms them
inside ``mo_cloud_optics.f90::cloud_optics`` (lines 339-374). The one
implementation is ``cloud_optics.echam_cloud_effective_radii``, fed by
``cloud_optics.radiation_effective_radii``, which both backends call:

- **Droplet radius** — the Martin et al. (1994) law
  ``re_droplets = zfact·zkap·(zlwc/zcdnc)^(1/3)``, ``zfact =
  1e6·(3e-9/(4π·rhoh2o))^(1/3)``, with the in-cloud liquid water content
  ``zlwc`` [g/m³] and droplet number ``zcdnc`` [cm⁻³] that
  ``mo_psrad_interface.f90::psrad_interface`` builds (lines 220-271:
  ``xm_liq·1000/cf · p/(rd·T)``, zeroed where ``cf ≤ 2ε``). It is
  ``cloud_utils.eff_liquid_droplet_radius``, the same function the Lohmann 2M
  scheme evaluates for its own ``preffl``.
- **Crystal radius** — with prescribed droplet number (ECHAM's 1M setup,
  ``nic_cirrus = 0``) the Moss/Foot in-cloud-IWC law
  ``zrieff = 83.8·ziwc^0.216`` (``effective_radius_ice``); with the 2-moment
  scheme (``nic_cirrus > 0``) the Lohmann et al. (2008) plate law
  ``eff_ice_crystal_radius(ziwc, zicnc)`` of ``mo_cloud_micro_2m.f90``, at
  every temperature (``cloud_optics`` applies it below ``cthomi`` too, a
  change the ECHAM-HAM source labels SF 176). The Pruppacher & Klett
  constants are the 2M scheme's
  defaults (fixed parameters in ECHAM's ``mo_cloud_utils``).
- **Droplet number and breadth by cloud scheme.**
  - *1M* (``physics=echam``, ECHAM ``cloud`` with ``ncd_activ = 0``): ECHAM's
    prescribed ``acdnc`` (``physc.f90`` §3.12, ICON-A
    ``mo_echam_phy_diag.f90::droplet_number``: 80 / 180 cm⁻³ maritime /
    continental up to 800 hPa, ``20 + (zn2 − 20)·exp(1 − (80000/p)²)`` cm⁻³
    above), ``cloud_utils.prescribed_cdnc_profile``, times the MACv2-SP Twomey
    factor ``aerosol.cdnc_factor`` (the simple plumes' ``x_cdnc`` "scale factor
    for cloud droplet number concentration",
    ``mo_bc_aeropt_splumes.f90::add_bc_aeropt_splumes``); breadth ``zkap`` =
    1.143 continental / 1.077 maritime. The number is
    ``cloud_utils.prescribed_droplet_number``, the one call the 1M
    microphysics also makes, as ECHAM hands one ``acdnc`` to both (see
    {doc}`clouds_microphysics`).
  - *2M* (``cloud_scheme="2m"`` with SPA, and the JAM composition where ARG
    activation feeds the droplet number): the prognostic ``qnc``/``qni``
    tracers times the air density, which is what ECHAM-HAM's radiation reads
    (``acdnc`` and ``icnc_instantan``, both left by the previous step's
    ``cloud_micro_2m``); breadth ``zkap = breadth_factor(zcdnc)`` (Peng &
    Lohmann 2003). The droplet number already carries the aerosol through
    activation, so the Twomey factor is not applied again.
  - *Continental* is ECHAM's ``laland .AND. .NOT. laglac``: land fraction
    ``fmask ≥ 0.5`` (ECHAM6's binary land-sea mask, ``lfractional_mask =
    .FALSE.``) without glacier cover (``forcing.glacier_fraction``).
- **Clamp and clear layers** — ECHAM clamps each radius to the size range of
  its optics table (``MAX(relmin, MIN(relmax, …))``); jcm radiates through
  jax-rrtmgp's tables and clamps to theirs (2.5-21.5 µm droplets, 5-90 µm
  crystals), so a cell holding condensate but a floored number radiates with
  the largest tabulated size, as in ECHAM. A layer without the phase gets 0,
  ECHAM's ``re_droplets2d``/``re_crystals2d`` for a clear layer; it has no
  condensate path, so the value weights nothing.
- **Published diagnostic** — ``clouds.r_eff_liq`` / ``clouds.r_eff_ice`` are
  the radii the radiation used, written by the radiation term (held from the
  last solve on a cached sub-step). No term reads them back into the
  radiation.
- **Radii of the satellite simulators and AeroCom** — ECHAM's COSP pairs
  the radii and cover of the last radiation call (``cosp_reffl``/
  ``cosp_reffi``/``cosp_f3d``, set in ``mo_psrad_interface.f90``) with the
  step-start condensate (``xlm1``/``xim1``), which match on radiation
  steps. jcm's COSP simulators and AeroCom cloud diagnostics
  read the post-microphysics condensate and cover instead, so they can
  describe the state saved at the end of the step. Paired with the
  radiation's radii, a layer the microphysics filled after the solve would
  carry condensate with a radius of 0. They therefore form the radii of the
  condensate they read with the same law
  (``cloud_optics.post_physics_effective_radii``: the post-physics
  temperature and number tracers, and a cover floor of ~10⁻¹² rather than
  the radiation's thin-cloud optical-depth guard). AeroCom publishes them as
  ``aerocom_cdr3d``/``aerocom_icr3d``, which the CMOR writer maps to
  ``cdr3d``/``icr3d``.

Time level of each input, ECHAM (``physc``: ``cover``, then radiation, then
the cloud scheme) against jcm (Sundqvist cover, then radiation, then the
microphysics):

| input | ECHAM6.3-HAM2.3 | jcm |
|---|---|---|
| cloud fraction | ``aclc`` from ``cover``, this step, before radiation, zeroed where ``xlm1`` and ``xim1`` are both ≤ 0 (``mo_radiation.f90`` l.433-434) | ``clouds.cloud_fraction`` from the cover, this step, before radiation, zeroed where ``qc`` and ``qi`` are both ≤ 0 (``cloud_data.radiation_cloud_fields``) |
| cloud water / ice | ``xlm1``/``xim1``, the step-start state, clipped at 0 | the step-start ``qc``/``qi`` tracers, clipped at 0 |
| droplet number, 1M | ``acdnc`` prescribed profile, set at the first step from that step's pressure; × ``x_cdnc`` of the step's plumes | profile at this step's pressure; × this step's ``aerosol.cdnc_factor`` |
| droplet number, 2M | ``acdnc`` = ``zcdnc`` at the end of the previous step's ``cloud_micro_2m`` | step-start ``qnc`` tracer (the previous step's microphysics output, after that step's transport) × air density |
| crystal number, 2M | ``icnc_instantan`` = ``picnc`` at the end of the previous step's ``cloud_micro_2m`` | step-start ``qni`` tracer × air density |
| density, temperature, pressure | step-start ``p/(rd·T)`` | step-start ``p/(rd·T)`` |

The **grey** scheme's cloud optics (``cloud_optics.cloud_optics``) are its own
and do not use these radii: a column-constant liquid radius
(``effective_radius_liquid``, 11 µm scaled by the Twomey factor, the land/ocean
average of CAM4's ``reltab``, not given CAM4's land/ocean contrast because
``cdnc_factor`` already carries the aerosol/CCN effect) and the Moss/Foot ice
radius.

**Cloud optical depth.**
- **Sub-grid inhomogeneity factors** — the cloud **optical depth** is multiplied
  by fixed per-phase factors, correcting the plane-parallel albedo bias of
  homogeneous-cloud radiative transfer. This is ECHAM's
  ``mo_cloud_optics.f90::cloud_optics`` treatment with ``l_variable_inhoml =
  .FALSE.`` (``ztau = ztol*zinhoml + ztoi*zinhomi``), at ECHAM's T63 values
  (``setup_cloud_optics``, nn = 63): ice ``zinhomi = 0.8`` — ``0.7`` in the JAM
  composition, ECHAM-HAM's value for its 2M + ARG setup (``lcdnc_progn``,
  ``ncd_activ = 2``), which is the pairing JAM is — and a liquid factor
  chosen per column by the convective type ``ktype`` — ``zinhoml1 = 0.8`` with
  no convection, ``zinhoml3 = 0.8`` for deep, shallow and mid-level convection,
  and ``zinhoml2 = 0.4`` for ``ktype = 4``, a shallow-convective column whose
  liquid water path at and below the convective cloud top exceeds
  ``clwprat = 4`` times the path above it. As in ECHAM the re-typing to 4 is
  done by the 1M cloud scheme after convection (``mo_cloud.f90``;
  ``Echam1MMicrophysics`` amends ``convection.ktype`` from the step-start
  liquid and the Tiedtke cloud top) and read by the *next* step's radiation,
  which runs before convection (ECHAM's lagged ``rtype``). ECHAM's 2M
  ``cloud_micro_interface`` never re-types, so under ``cloud_scheme = "2m"``
  every column keeps the 0.8 liquid factor, as in ECHAM-HAM. The
  factors scale the optical depth only: the effective radii come from the
  physical (unscaled) condensate, so the inhomogeneity does not shrink the
  crystal radius the IWC laws give. On the grey backend the τ-weighted ssa/asymmetry are
  likewise taken from the unscaled per-phase optical depths, exactly ECHAM's
  ``zomg``/``zasy``. The RRTMGP backend scales the per-phase condensate paths
  it hands to jax-rrtmgp, which weights the combined ssa/asymmetry by the τ
  those paths produce: identical to ECHAM wherever the liquid and ice factors
  are equal (every column except ``ktype = 4`` ones at the defaults), while in
  a ``ktype = 4`` layer holding both phases the total τ is exact but the
  ssa/asymmetry weighting uses the scaled τ. jax-rrtmgp 0.5.0 accepts per-phase
  optical-depth scales that weight by the unscaled τ; wiring them is #958. The
  four factors are
  ``RadiationParameters.cloud_inhomogeneity_{liquid, liquid_convective,
  liquid_shallow, ice}``, differentiable leaves.

**What ECHAM/CAM does.** ECHAM6-HAM2.3 runs the **PSrad/RRTMG** two-stream
correlated-k scheme (``mo_psrad_interface.f90``; Pincus & Stevens 2013; RRTMG:
Mlawer et al. 1997, Iacono et al. 2008) with **McICA** sub-column sampling (Pincus,
Barker & Morcrette 2003) and maximum-random overlap by default (``i_overlap = 1``,
``mo_radiation_parameters.f90`` l.71): ``mo_cld_sampling.f90::sample_cld_state``
offers maximum-random, maximum and random, and no exponential rule. Its
maximum-random sampler (l.66-83) keeps a sub-column's rank below a cloudy cell
of that sub-column and redraws it in the clear part otherwise, and the total
cover it reports is the adjacent-layer Geleyn-Hollingsworth product
(``mo_radiation.f90`` l.436-442); jcm's McICA samples the same rule. Cloud optics use ECHAM's ``mo_cloud_optics.f90`` LUTs. CAM6 runs
**RRTMGP** (Pincus, Mlawer & Delamere 2019) with liquid effective radius from
``cloud_optical_properties.F90`` (``reltab``). MACv2-SP is Stevens et al. (2017),
``mo_bc_aeropt_splumes.f90``. The NN emulator architecture is Ukkonen (2024),
``rte-rrtmgp-nn``.

**Why we differ.**
- `science` — the effective radii follow ECHAM's ``cloud_optics`` with four
  stated departures. (1) On the 2-moment path ECHAM-HAM evaluates
  ``breadth_factor`` on the droplet number after converting it to cm⁻³,
  although the function takes 1/m³ (``0.00045e-6·pcdnc + 1.18``), which pins
  its radiative ``zkap`` at 1.18; jcm evaluates Peng & Lohmann in the units the
  function documents, as the 2M microphysics' own ``preffl`` does (``zkap`` =
  1.225 at 100 cm⁻³, 1.41 at 500 cm⁻³, a radius 4-20 % above what ECHAM-HAM's
  radiation would form). (2) The clamp is to jax-rrtmgp's table range, the
  tables jcm radiates with, not ECHAM6's ``ECHAM6_CldOptProps.nc``. (3) The
  prescribed 1M droplet profile is re-evaluated at every radiation call from
  the current pressure (ICON-A's ``droplet_number``) rather than frozen at the
  initial step (ECHAM6), which differs only through surface-pressure change,
  and jcm carries no lake map, so no lake cell takes the continental profile.
  (4) The 2M droplet and crystal number are the step-start tracers, which have
  been transported since the previous step's microphysics wrote them; ECHAM-HAM's
  radiation reads the microphysics' own copies (``acdnc``, ``icnc_instantan``)
  from before transport, while the condensate it pairs them with is
  transported. jcm pairs transported number with transported condensate.
- `compute` — the SW spectrum is collapsed to a single broadband albedo
  (``0.46·vis + 0.54·nir``) at the RRTMGP surface BC. jax-rrtmgp 0.5.0 accepts
  per-band direct and diffuse albedos; passing them is #959.
  The same single value serves the direct beam and diffuse light, so the
  open-water direct and diffuse albedos are merged before they reach it (see
  {doc}`surface`, *Surface albedo*).
  Radiation sub-stepping and the frozen-McICA-step option are compute affordances
  with no ECHAM analogue.
- `differentiability` — cloud-optics SSA/asymmetry combination uses double-
  ``where`` safe-denominator guards so backward-mode cloud-parameter gradients do
  not form ``0·inf`` on clear columns.
- `differentiability` — the effective radius is a function of the current
  state only (condensate, cloud fraction, droplet/crystal number, temperature,
  pressure); no radius is carried between steps and no float value doubles as
  a "not provided" flag. Wherever a phase is present the radius is continuous
  in the state up to the table clamp; where its condensate is exactly 0 the
  phase is absent, the radius is reported as 0, and the layer carries no
  optical depth of that phase. The cube-root and ``IWC^0.216`` /
  ``IWC^(1/2.475)`` powers are evaluated on a positive base behind a double
  ``where``, so the reverse pass is finite in clear layers and as the
  condensate goes to zero, where the radius sits at the table minimum and the
  clamp passes no gradient (``cloud_optics_test.py``,
  ``TestEchamCloudEffectiveRadii``).
- `differentiability` — the gas optics interpolate the k-distribution tables
  linearly in temperature, log-pressure and the binary-species fraction η, so
  the fluxes are continuous in temperature and humidity, with slope changes
  only at table cell boundaries. On the per-term gradient harness's single
  columns (``term_gradients_test.py``), with the library promoted to float64,
  a joint temperature-plus-humidity perturbation has a central difference
  that agrees with AD to 1.5 % or better at every step from 1.25·10⁻⁴ down to
  6·10⁻⁸. Two other things stop a finite-difference reference for the whole
  RRTMGP term there. One is inputs that sit exactly at zero. Where a
  condensate tracer is 0 the term has a kink (tracked in #843, where the grey
  scheme's identical clip is catalogued): a negative step is clipped in
  ``prepare_radiation_state`` and changes nothing, a positive step radiates,
  and AD returns the cloud-free side's derivative, 0. The cloudy side is
  steeper at small steps than its limit, because the Moss/Foot ice radius
  shrinks with the ice content until it reaches the table floor. The
  identically-zero MACv2-SP longwave aerosol inputs also keep the one-sided
  secants apart. The other is the float32 primal: the library runs in float32
  by construction (see ``radiation_scheme_rrtmgp``), and the heating of the
  top few-Pa layers is a difference of two large fluxes, whose float32
  rounding (6·10⁻⁸ to 3·10⁻⁷ K/s at the 1 and 4 Pa levels) does not shrink
  with the step as the response does. AD itself is unaffected: jvp and vjp
  agree to float32 reduction order, which is the check the harness applies.

**Status & known limitations.**
- **The packaged NN emulator was trained on exponential overlap.** Its training
  labels are RRTMGP fluxes under the exponential rule at 2 km, the default when
  they were generated, so under the maximum-random default its fluxes and its
  published ``total_cloud_cover`` describe different overlaps until it is
  retrained (#881).
- **Cloud inhomogeneity carries ECHAM's T63 values, not its per-resolution
  table.** ECHAM raises the ice factor at higher truncation (``zinhomi = 0.85``
  at T127+) and uses ``zinhoml3 = 0.4`` at T31; jcm takes the T63 values at
  every resolution and exposes them as parameters (#974). The cover's and the
  microphysics' constants, by contrast, take ECHAM's per-truncation defaults
  ({doc}`../design/resolution_defaults`, which also gives why these two are
  held: ``RadiationParameters`` cannot record the truncation its defaults were
  built for, and ECHAM-HAM's JAM value of ``zinhomi`` is defined at T63
  only). The 2M + SPA composition (no ECHAM-HAM
  counterpart: HAM activates with Lin-Leaitch or ARG) keeps ECHAM6's
  ``zinhomi = 0.8``. On RRTMGP, a mixed-phase layer in a ``ktype = 4``
  column weights its combined ssa/asymmetry by the scaled rather than the
  physical per-phase τ until the backend passes jax-rrtmgp's per-phase
  optical-depth scales (#958). The separate in-cloud-condensate cap
  (``_MAX_IN_CLOUD_CONDENSATE``) is only a NaN guard against thin-cloud
  optical-depth blow-up; it binds in ~0.003 % of cloudy cells and is *not* an
  inhomogeneity term.
- **The NN emulator does not reflect the inhomogeneity factor yet.** The emulated
  path predicts fluxes directly, so the inhomogeneity effect is implicit in its
  training labels rather than a runtime knob; it deliberately does not read the
  ``cloud_inhomogeneity_*`` factors or the convective type. The packaged
  checkpoint predates the 0.8 factor, so ``echam-emulated-2m`` diverges from the
  RRTMGP backend by a few W/m² for cloudy columns until the emulator is
  retrained against the corrected radiation (#881, folded into the #743
  retrain). The convective-type dependence does not widen that gap: the emulated
  configuration is 2M, where no column is re-typed and RRTMGP also applies the
  uniform 0.8 liquid factor.
- **Out-of-range RRTMGP inputs.** Water vapour: jax-rrtmgp clips the
  humidity it receives at zero and treats an absent absorber as absent (the
  zero relative abundance is guarded), so a dry or slightly negative layer
  gives finite fluxes. ECHAM instead bounds the input below at machine epsilon
  (``mo_radiation.f90``: ``xq_vap = MAX(qm_vap, EPSILON(1.0_wp))``). The two
  give the same radiation: a model-top humidity anywhere from 1e-12 kg/kg
  down to zero, or negative, gives the same heating rate, so jcm adds no bound
  of its own. Temperature and pressure: RRTMGP's gas-optics and Planck
  tables cover 160–355 K and 1096 hPa–1.005 Pa, and a model layer can lie
  outside both. A 1 Pa top layer is below the pressure range, and radiative
  cooling can take a thin top layer to the 160 K edge. jax-rrtmgp extends the
  absorption coefficients and the Planck source linearly along the end
  interval of each axis and floors them at zero. That is the index-and-fraction
  rule of RRTMGP's own kernels (``mo_gas_optics_rrtmgp_kernels.F90``), whose
  frontend rejects out-of-range input before it reaches them
  (``mo_gas_optics_rrtmgp.F90``), and of the RRTMG that ECHAM6 runs
  (``mo_lrtm_driver.f90::planckFunction``, ``mo_rrtm_coeffs.f90``). The Planck
  fractions, which partition a band's source among its g-points, are held at
  their end values instead, so they stay non-negative and each band's
  fractions still sum to one. A layer colder than 160 K therefore emits less
  than one at 160 K, and its longwave cooling weakens as it cools, on both
  sides of the table edge (``rrtmgp_test.py::TestRRTMGPColdLayerEmission``).
  In the single-column RRTMGP RCE of the full ECHAM stack the 1 Pa layer
  settles at 160.2 K.
- **The NN emulator's radius features keep a trained fill in phase-free
  layers.** The emulator is fed the radius RRTMGP radiates the layer with
  wherever the phase is present, but where it is absent the feature is
  ``11 µm·cdnc_factor^(-1/3)`` (liquid) / 83.8 µm (ice) rather than 0
  (``nn_emulator.emulator_radius_features``), because the packaged checkpoint
  was trained with those values there. That checkpoint was also trained where
  the 2M scheme published no radius (supercooled liquid below its droplet
  floor, cold ice) on the 11 µm / Moss-Foot fallback, which is not what
  RRTMGP now radiates those layers with. Offline, on 1024 columns of a
  ``t63-echam-2m`` state, the packaged u64 emulator's error against RRTMGP is
  the same under both feature definitions (cloudy-column TOA SW-up RMSE
  27.7 W/m² fed the current radii and labelled with them, 28.2 W/m² fed and
  labelled with the trained ones; OLR 40.4 vs 39.9 W/m²), so moving the
  features costs nothing measurable. Retraining on data from the current
  generator (``tools/radiation_emulator/generate_training_data.py`` forms the
  features with the same functions) removes the fill (#881).
- **Thin-lid aerosol-radiation cutoff.** Online aerosol optics are zeroed above
  ``_AER_RAD_PMIN`` (``jcm/physics/aerosol/jam/optics/optics_term.py``) and the
  per-layer band τ is capped, to bound heating over ~1 Pa lid layers; aerosol mass
  above ~2 hPa is radiatively negligible.
- **ERFari double-solve cost** — the exact aerosol-free companion is a large
  wall-clock addition at T63L47; subsampling every Nth step trades fidelity for
  runtime (see {doc}`../design/aerocom_erfari_sampling`).
- The reverse pass holds per-g-point profiles of all columns live;
  ``JCM_RRTMGP_COL_CHUNKS`` rematerializes in column blocks.

**Code pointers.**
- ``jcm/physics/radiation/rrtmgp.py`` — ``RRTMGPRadiation``,
  ``radiation_scheme_rrtmgp``, ``prepare_rrtmgp_data``, the aerosol-free
  companion.
- ``jcm/physics/radiation/cloud_optics.py`` — ``radiation_effective_radii``,
  ``echam_cloud_effective_radii``, ``continental_columns``,
  ``RRTMGP_LIQUID_RADIUS_RANGE_UM`` / ``RRTMGP_ICE_RADIUS_RANGE_UM``;
  ``effective_radius_ice``, ``effective_radius_liquid`` (grey).
- ``jcm/physics/clouds/cloud_utils.py`` — ``eff_liquid_droplet_radius``,
  ``eff_ice_crystal_radius``, ``breadth_factor``, ``prescribed_cdnc_profile``.
- ``jcm/physics/radiation/mcica.py`` — ``generate_subcolumns``,
  ``column_total_cover``, ``in_cloud_condensate``,
  ``_MAX_IN_CLOUD_CONDENSATE``; ``band_config.py`` (``RadiationBandConfig``).
- ``jcm/physics/radiation/nn_emulator.py`` (``emulator_radius_features``),
  ``nn_emulator_scheme.py``.
- ``jcm/physics/radiation/speedy_shortwave.py``, ``speedy_longwave.py``.
- ``jcm/physics/radiation/grey_two_stream/radiation_scheme.py``.
- ``jcm/physics/radiation/aerosol_free.py``;
  ``jcm/physics/aerosol/jam/optics/optics_term.py``.

**Validation evidence.** ``jcm/physics/radiation/rrtmgp_test.py`` (compute +
sub-step caching + carry wiring), ``cloud_optics_test.py``, ``mcica_test.py``
(overlap-rule cloud-cover checks), ``aerosol_radiation_test.py``,
``nn_emulator_test.py`` / ``nn_emulator_scheme_test.py`` (pinned against a real
``rte-rrtmgp-nn`` checkpoint). Design references:
{doc}`../design/aerocom_erfari_sampling`, {doc}`../design/radiation_nn_emulator`,
{doc}`../design/aerosol_optics_diagnostics`.

## The grey two-stream (idealized)

**What it is.** ``GreyTwoStreamRadiation`` is an idealized radiation scheme:
two shortwave bands (0.20–0.69 µm, 0.69–2.5 µm) and three longwave bands
(10–350, 350–500 and 500–2500 cm⁻¹) with hand-tuned gas absorption, Planck
emission per band, the cloud optics of ``cloud_optics.py`` and the same
sub-step cache as the other backends. It is energy-conserving by construction
(below) and differentiable everywhere on its operating range, which is what it
is for: a cheap radiative driver for idealized configurations (the ``jcm.rce``
radiative-convective columns) and for tests that need a full, realistic term
chain but assert nothing about radiation
(``jcm.physics.echam.testing.idealized_echam_physics``). It is not calibrated:
on a tropical column against RRTMGP its mid-tropospheric longwave cooling is
roughly 150× too weak and its OLR ~70 % too high. It is not ECHAM physics — ECHAM
runs PSrad/RRTMG — so the ECHAM factory does not offer it; it is composed
explicitly, by passing the term to the factory,
``echam_physics(radiation_scheme=GreyTwoStreamRadiation())``, which derives the
composition's band structure and the JAM optics cadence from that term.

**Grey two-stream layer solution.** Each homogeneous layer's diffuse
reflectance and transmittance are the exact two-stream result of Meador &
Weaver (1980, eq. 14-15; equivalently Toon et al. 1989) under the Eddington
closure (``two_stream_coefficients``: ``gamma1 = (7 - ssa(4 + 3g))/4``,
``gamma2 = -(1 - ssa(4 - 3g))/4``). With eigenvalue
``lambda = sqrt(gamma1**2 - gamma2**2)``, ``e = exp(-lambda*tau)`` and
``Gamma = gamma2/(gamma1 + lambda)`` the solution is
``R = Gamma (1 - e^2)/(1 - Gamma^2 e^2)`` and
``T = (1 - Gamma^2) e/(1 - Gamma^2 e^2)``,
so a semi-infinite layer has albedo ``Gamma`` and a conservative
(``ssa = 1``) layer reflects ``R = gamma1*tau/(1 + gamma1*tau)`` with
``R + T = 1``. ``layer_reflectance_transmittance`` evaluates the algebraically
identical rearrangement ``R = gamma2 S/(gamma1 S + 1 + e^2)``,
``T = 2 e/(gamma1 S + 1 + e^2)`` with ``S = (1 - e^2)/lambda``, chosen because
its denominator cannot cancel to zero, it never divides by ``gamma1`` or
``gamma1 + lambda`` (both zero at ``ssa = g = 1``), and it contains no growing
exponential — one expression is valid and differentiable at every optical
depth. For ``lambda*tau < 0.1`` it is evaluated as an even Taylor series in
``(lambda*tau)^2``: R and T are even functions of ``lambda``, hence smooth in
``lambda^2 = 3(1-ssa)(1-ssa*g)``, and the series carries the true endpoint
derivative through autodiff at the conservative limit (where the closed value
is ``R = gamma1*tau/(1+gamma1*tau)``). Longwave gas layers (``ssa = 0``) keep
the Eddington coefficients ``gamma1 = 7/4``, ``gamma2 = -1/4``,
``lambda = sqrt(3)``: the closure yields a small *negative* reflectance
(``Gamma ~ -0.07``), clipped to ``R = 0`` as an approximation artefact, and a
diffuse transmittance ``T = (1 - Gamma^2) e/(1 - Gamma^2 e^2)`` — within 0.5%
of, but not exactly, ``exp(-sqrt(3)*tau)``.

**Grey shortwave: delta-Eddington layers joined by adding.** The shortwave
optical properties are first delta-scaled (``delta_eddington_scaling``; Joseph,
Wiscombe & Weinman 1976): the forward diffraction peak ``f = g^2`` is counted as
unscattered, ``tau' = (1 - ssa f) tau``, ``ssa' = (1 - f) ssa/(1 - ssa f)``,
``g' = g/(1 + g)``; the truncation removes a *forward* peak, so it applies only to
``g > 0`` (``f = max(g, 0)^2``) and backward-scattering layers pass through
unscaled. This is the adjustment Toon et al. (1989) prescribe with the
Eddington coefficients for solar radiation; unscaled, a cloud's ``g = 0.85``
makes the direct-beam backscatter coefficient ``gamma3 = (2 - 3 g mu0)/4``
negative at high sun, while scaled ``gamma3 >= 1/8``. Each layer then scatters
the collimated beam into the diffuse streams by the exact two-stream solution
with a direct source (``_direct_beam_layer``; Meador & Weaver 1980, Toon et al.
1989 eq. 23-24): the particular solution ``C+ = A exp(-tau/mu0)``,
``C- = B exp(-tau/mu0)`` with the resonance factor
``1/(lambda^2 - 1/mu0^2)``, plus the homogeneous solution that cancels its
diffuse flux at the faces, which is the layer's own diffuse response, so
``R_dir = A (1 - T E) - R B`` and ``T_dir = B (E - T) - R A E`` with
``E = exp(-tau/mu0)``. For ``ssa = 1`` this gives ``R_dir + T_dir + E = 1``
exactly, and a thick conservative cloud reflects the closed-form two-stream
albedo ``[gamma1 tau + (gamma3 - gamma1 mu0)(1 - E)]/(1 + gamma1 tau)`` (0.88 for
``tau = 82``, ``g = 0.85``, overhead sun). The layers are combined with the
adding method (Shonk & Hogan 2008, eqs. 9-13, the RTE solver jax-rrtmgp uses),
so every multiple reflection between layers and with the surface is counted and
the column closes its energy budget: with no absorption, what enters at the top
leaves through the top or is absorbed at the surface (float32 closure 3e-7).
`differentiability` — the direct-beam solution is written so that it has no
removable singularity on the differentiated path: below ``lambda^2 = 1/4`` it
depends on the eigenvalue only through ``lambda^2`` and the diffuse ``R``,
``T`` (smooth at the conservative limit ``lambda = 0``, needing no floor); above
it, where the resonance ``lambda = 1/mu0`` can occur, the resonance is factored
out analytically into ``(E - e)/(lambda - 1/mu0)``, evaluated as a scaled
``(1 - exp(-x))/x`` that is carried as a series near ``x = 0`` so its
derivative is exact through the resonance. No clip is applied to ``R_dir`` or
``T_dir``: both are non-negative on the delta-scaled coefficients, and a clip
would break the conservation identity. Domain: for a strictly non-negative diffuse
component the Eddington closure needs ``g mu0 >= -2/3`` (below it
``gamma4 = (2 + 3 g mu0)/4 < 0``); energy still closes there. No scatterer in
the model has ``g < 0``.

**Overlap.** The grey scheme combines one clear and one cloudy beam weighted by
the overlap-derived total cover (``column_total_cover``) rather than sampling
sub-columns.

**Aerosol.** The grey scheme reads a single *broadband* aerosol
profile (``aerosol.aod_profile``/``ssa_profile``/``asy_profile`` plus a column
``angstrom`` it band-scales itself) rather than the per-band arrays only
``rrtmgp.py`` consumes. ``JamOpticsTerm`` writes those broadband fields from
the SW band centred nearest 550 nm, so an idealized grey + JAM composition keeps
its aerosol direct effect — at band-centre rather than exact-550 nm accuracy, and
only with ``jam_optics=True`` (the default; ``False`` leaves the carry slot
radiatively passive).

**Band wavelengths.** Every wavelength-dependent grey optical
property (the Mie/heuristic cloud optics in
``jcm/physics/radiation/cloud_optics.py``, the Ångström scaling of the
broadband aerosol AOD, and the near-IR/visible band classifier that assigns
surface albedo and gas absorbers) reads one representative wavelength per band
from ``get_band_wavelength``, derived from the band limits in
``jcm/physics/radiation/constants.py``. For a **shortwave** band it is the
solar-flux-weighted mean wavelength, ``λ_eff = ∫λ B_λ(5772 K) dλ / ∫B_λ dλ``
over the band: 0.489 µm for UV/visible (0.20–0.69 µm) and 1.136 µm for the
near-IR (0.69–2.5 µm). **Longwave** bands use the mid-wavenumber wavelength
(their cloud absorption is tabulated per band and their aerosol AOD is
negligible, so the value only has to lie inside the band). The grey scheme's
TOA flux is split equally between the two SW bands, which the 5772 K
blackbody supports (50.6 % / 49.4 %).

`science` — the grey scheme has no ECHAM counterpart (ECHAM's radiation is
PSrad/RRTMG with narrow bands), so its broadband representative wavelength is
our choice. It follows the standard broadband-effective-wavelength convention
(weight by the incident solar spectrum) rather than the mid-wavenumber value,
which for the broad UV/visible band sits at 0.31 µm, where little of the
band's solar energy lies: with an Ångström exponent of 2 it inflates that
band's aerosol optical depth ~2.5× relative to the solar-weighted value
(MACv2-SP plumes carry exponents up to 2). Cloud SW optics barely move: for
liquid at ``r_eff`` = 10 µm the optical depth and asymmetry are unchanged
(geometric-optics limit, size parameter ≫ 1) and the near-IR single-scattering
albedo shifts by 3e-5; for ice at 30 µm the optical depth changes by ≤ 2 % and
the near-IR single-scattering albedo by −0.005.

**Known limitations.**
- **Grey shortwave keeps the Eddington closure's negative diffuse
  reflectance clipped.** For strongly absorbing layers
  (``ssa' < 1/(4 - 3 g')`` after delta scaling, e.g. ``ssa = 0.5``,
  ``g = 0.85``) Eddington's ``gamma2`` is negative, and so is the exact
  solution's diffuse reflectance; the diffuse field clips it to 0 (see the layer
  solution above), which raises the layer's reflection slightly and makes a
  homogeneous slab differ from its subdivision by up to ~1e-3 of the incident
  flux. The direct-beam solution keeps the exact value, so over such a layer the
  reflectance first rises with optical depth and then settles ~0.3 % lower onto
  its semi-infinite value. RRTMGP avoids this by using the practical improved
  flux method (Zdunkowski et al. 1980), whose ``gamma2 >= 0``; the grey scheme
  keeps the Eddington coefficients its layer solution is built and tested on.
- **Grey longwave is non-scattering.** Every grey longwave optical property
  (gas, cloud, aerosol) has ``ssa = 0``, so the layer reflectance is zero and
  the longwave recurrence carries transmission and emission only, with an
  isothermal-layer source ``B (1 - T)`` at the layer-mean temperature rather
  than a linear-in-tau Planck profile.

**Code pointers.** ``jcm/physics/radiation/grey_two_stream/`` —
``radiation_scheme.py`` (``GreyTwoStreamRadiation``, ``radiation_scheme``),
``two_stream.py``, ``gas_optics.py``, ``planck.py``;
``jcm/physics/echam/testing.py`` (``idealized_echam_physics``).

**Validation evidence.** In ``jcm/physics/radiation/grey_two_stream/``:
``two_stream_test.py`` (layer solution, direct beam, adding solve and energy
closure, with gradient checks), ``radiation_scheme_test.py`` and
``jcm/physics/radiation/grey_two_stream/rce_test.py``; the term's derivatives are
checked on the idealized composition in
``jcm/physics/echam/term_gradients_test.py``.

## Seasonal timing in v3

The forcing interface derives the annual phase from the elapsed fraction of
**the current Gregorian year**, including the fractional day. January 1 is
phase zero; a leap year has 366 days. SPEEDY's solar Fourier coefficients and
its empirical ozone offset `10 / 365` remain unchanged: that denominator is a
local empirical constant, not the model's date arithmetic. Monthly forcing
uses civil months rather than twelve equal fractions of a nominal year.

This changes the seasonal forcing relative to the former epoch-based 365-day
clock, so old climate baselines are not numerically interchangeable. The
calendar-boundary tests establish date alignment; multi-year SPEEDY and ECHAM
climate comparisons are part of the v3.0 release validation tracked in #831.
The clock, forcing-selection and output-labelling contracts, and how to migrate
from the v2 clock, are described in {ref}`v3-datetime`.
