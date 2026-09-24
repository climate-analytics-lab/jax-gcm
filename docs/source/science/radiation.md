# Radiation

**What we do.** jax-gcm carries four interchangeable radiation backends, each a
composable ``PhysicsTerm`` (SPEEDY's is a function pair in the SPEEDY term list),
selected by config:

- **RRTMGP correlated-k** (``jcm/physics/radiation/rrtmgp.py::RRTMGPRadiation``) —
  the production backend. Wraps the external ``jax-rrtmgp`` library behind the
  ICON radiation interface. The per-column entry point (``radiation_scheme_rrtmgp``)
  is vmapped over columns; it handles TOA-first ↔ surface-first reordering, a
  pressure halo that encodes the model's exact boundary-layer Δp, and a
  float32-end-to-end library boundary with a scoped ``jax.enable_x64(False)`` so
  it is bit-consistent on x64 hosts (pySES / MAM4-JAX).
- **Grey two-stream**
  (``jcm/physics/radiation/grey_two_stream/radiation_scheme.py``) — a cheap
  idealized backend with its own gas optics and Planck bands.
- **SPEEDY SW/LW** (``jcm/physics/radiation/speedy_shortwave.py``,
  ``speedy_longwave.py`` — 4-band LW).
- **NN emulator** (``jcm/physics/radiation/nn_emulator_scheme.py::NNEmulatorRadiation``,
  network in ``nn_emulator.py``) — a bidirectional-GRU emulator of RTE+RRTMGP.
  See {doc}`../design/radiation_nn_emulator`.

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

**Partial-cloud / overlap** differs by backend. **RRTMGP** uses full **McICA**
(``jcm/physics/radiation/mcica.py``): one stochastic binary cloud profile per
g-point, seeded deterministically per column and model step, with three overlap
rules — random, maximum-random (Geleyn-Hollingsworth), and
generalised-exponential with a decorrelation length. The **grey** backend
instead combines one clear and one cloudy beam weighted by the overlap-derived
total cover (``column_total_cover``); the **NN emulator's** fluxes carry
whatever overlap its RRTMGP training labels embedded — the network sees only
layer cloud fractions and paths, so the runtime ``cloud_overlap`` /
``cloud_decorrelation_km`` knobs change its *reported total-cover diagnostic*
(a post-hoc ``expected_total_cover``) and not its heating or fluxes; **SPEEDY**
carries its own cloud formulation. Swapping backends therefore changes the
cloud-overlap treatment, not just the gas optics. The AeroCom
total-cloud-cover diagnostic uses the maximum-random closure.

**Offline**, the total cloud cover jcm reports from saved output is also
maximum-random — ECHAM's own ``aclcov`` (``mo_cloud.f90`` §10.2), as
:func:`jcm.analysis.total_cloud_cover`, and it is what the release-validation
``cloud_cover`` gate scores. That choice defers to ECHAM and is deliberate:
overlap is a definition, the three in common use differ by ~0.3 in the global
mean, and a total cover is the basis the satellite climatologies are quoted
on. It is a different number from the McICA ``radiation.total_cloud_cover``
above — sampled quantity, different preprocessing, different time treatment —
and {doc}`../design/cloud_cover_gate` sets out the provenance, the measured
magnitudes and how far apart the two run.
Radiation **sub-steps** on the ECHAM-family backends (grey, RRTMGP, NN
emulator): a gate (``radiation_should_compute``) skips the expensive solve and
rescales cached heating on intermediate steps. SPEEDY has its own, different
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

**Cloud optics** (``jcm/physics/radiation/cloud_optics.py``) has a genuinely
**mixed provenance**, resolved through ``resolve_effective_radii`` (shared by
RRTMGP and the NN emulator so a feature and its label describe the same cloud):
- **Ice effective radius** — ECHAM's Moss/Foot in-cloud-IWC power law,
  ``effective_radius_ice``, citing ``mo_cloud_optics.f90`` (Moss et al. 1996).
- **Liquid effective-radius fallback** — a column constant scaled by the Twomey
  factor (``effective_radius_liquid``). This constant is the land/ocean average of
  **CAM4's ``reltab``** (``cloud_optical_properties.F90``); both origins are
  documented in-code.
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
  physical (unscaled) condensate, so the diagnostic ice radius still follows the
  Moss/Foot IWC law. On the grey backend the τ-weighted ssa/asymmetry are
  likewise taken from the unscaled per-phase optical depths, exactly ECHAM's
  ``zomg``/``zasy``. The RRTMGP backend can only pass per-phase condensate paths
  to jax-rrtmgp, which weights the combined ssa/asymmetry by the τ those paths
  produce: identical to ECHAM wherever the liquid and ice factors are equal
  (every column except ``ktype = 4`` ones at the defaults), while in a
  ``ktype = 4`` layer holding both phases the total τ is exact but the
  ssa/asymmetry weighting uses the scaled τ (jax-rrtmgp#37). The four factors are
  ``RadiationParameters.cloud_inhomogeneity_{liquid, liquid_convective,
  liquid_shallow, ice}``, differentiable leaves.
- **Grey backend band wavelengths** — every wavelength-dependent grey optical
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

**What ECHAM/CAM does.** ECHAM6-HAM2.3 runs the **PSrad/RRTMG** two-stream
correlated-k scheme (``mo_psrad_interface.f90``; Pincus & Stevens 2013; RRTMG:
Mlawer et al. 1997, Iacono et al. 2008) with **McICA** sub-column sampling (Pincus,
Barker & Morcrette 2003) and generalised exponential-random overlap (Räisänen et
al. 2004). Cloud optics use ECHAM's ``mo_cloud_optics.f90`` LUTs. CAM6 runs
**RRTMGP** (Pincus, Mlawer & Delamere 2019) with liquid effective radius from
``cloud_optical_properties.F90`` (``reltab``). MACv2-SP is Stevens et al. (2017),
``mo_bc_aeropt_splumes.f90``. The NN emulator architecture is Ukkonen (2024),
``rte-rrtmgp-nn``.

**Why we differ.**
- `science` — the liquid effective-radius fallback deliberately does *not* apply
  CAM4's land/ocean contrast, because ``cdnc_factor`` already carries the
  aerosol/CCN effect (applying both double-counts it). Both the 1M and 2M
  microphysics now publish an LWC-dependent ``clouds.r_eff_liq`` from the ECHAM
  Martin/Bower law; radiation reads it from the carried ``clouds`` state, which
  — because the ECHAM term order runs radiation *before* microphysics — is the
  previous step's value (a one-step lag). The constant fallback therefore
  survives only where that carried radius is still zero, resolved **cell by
  cell** (``resolve_effective_radii`` selects on ``r_eff > 0`` per level and
  column): the cold-start first step, and thereafter any cloudy cell that was
  clear the previous step — so a level that newly turns cloudy falls back for
  that step even in a column already cloudy elsewhere
  (``eff_liquid_droplet_radius`` returns exactly 0 in a clear cell). A
  composition that runs radiation with no droplet-radius-publishing microphysics
  uses the fallback throughout.
- `science` — the grey two-stream backend reads a single *broadband* aerosol
  profile (``aerosol.aod_profile``/``ssa_profile``/``asy_profile`` plus a column
  ``angstrom`` it band-scales itself) rather than the per-band arrays only
  ``rrtmgp.py`` consumes. ``JamOpticsTerm`` writes those broadband fields from
  the SW band centred nearest 550 nm, so a grey + JAM configuration keeps its
  aerosol direct effect — at band-centre rather than exact-550 nm accuracy, and
  only with ``jam_optics=True`` (the default; ``False`` leaves the carry slot
  radiatively passive).
- `science` — the grey scheme has no ECHAM counterpart (ECHAM's radiation is
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
- `compute` — the SW spectrum is collapsed to a single broadband albedo
  (``0.46·vis + 0.54·nir``) at the RRTMGP surface BC; a true per-band /
  direct-diffuse albedo needs a g-point→band map in the library (deferred).
  Radiation sub-stepping and the frozen-McICA-step option are compute affordances
  with no ECHAM analogue.
- `differentiability` — cloud-optics SSA/asymmetry combination uses double-
  ``where`` safe-denominator guards so backward-mode cloud-parameter gradients do
  not form ``0·inf`` on clear columns.

**Status & known limitations.**
- **Grey shortwave direct-beam source is not energy-conserving (#855).** The
  direct-to-diffuse reflectance ``R_dir`` uses a single-scattering source whose
  ``gamma3`` goes negative for forward-scattering clouds at high sun, so a thick
  conservative cloud reflects ~0 at TOA and its scattered energy is dropped. The
  diffuse layer solution above is exact; this is a separate defect in the
  direct-beam source, awaiting the Toon et al. (1989) source functions.
- **Cloud inhomogeneity carries ECHAM's T63 values, not its per-resolution
  table.** ECHAM raises the ice factor at higher truncation (``zinhomi = 0.85``
  at T127+) and uses ``zinhoml3 = 0.4`` at T31; jcm takes the T63 values at
  every resolution and exposes them as parameters, the same T63 convention as
  the other ECHAM cloud constants. The 2M + SPA composition (no ECHAM-HAM
  counterpart: HAM activates with Lin-Leaitch or ARG) keeps ECHAM6's
  ``zinhomi = 0.8``. On RRTMGP, a mixed-phase layer in a ``ktype = 4``
  column weights its combined ssa/asymmetry by the scaled rather than the
  physical per-phase τ until jax-rrtmgp takes a per-phase optical-depth scale
  (jax-rrtmgp#37). The separate in-cloud-condensate cap
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
  ``radiation_scheme_rrtmgp``, ``prepare_rrtmgp_data``, the aerosol-free companion,
  ``_MAX_IN_CLOUD_CONDENSATE``.
- ``jcm/physics/radiation/cloud_optics.py`` — ``effective_radius_ice``,
  ``effective_radius_liquid``, ``resolve_effective_radii``.
- ``jcm/physics/radiation/mcica.py`` — ``generate_subcolumns``,
  ``column_total_cover``; ``band_config.py`` (``RadiationBandConfig``).
- ``jcm/physics/radiation/nn_emulator.py``, ``nn_emulator_scheme.py``.
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
