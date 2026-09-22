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
- **Sub-grid inhomogeneity factor** — the in-cloud liquid/ice condensate path is
  multiplied by a fixed factor (default 0.8/0.8) before the optics, correcting the
  plane-parallel albedo bias of homogeneous-cloud radiative transfer. This is
  ECHAM's ``mo_cloud_optics.f90`` treatment with ``l_variable_inhoml = .FALSE.``
  (``ztau = ztol*zinhoml + ztoi*zinhomi``); τ is linear in path at fixed effective
  radius, so scaling the path scales τ identically. Applied on both the RRTMGP and
  grey backends via the ``cloud_inhomogeneity_liquid`` / ``cloud_inhomogeneity_ice``
  fields of ``RadiationParameters``.
- **Grey backend cloud-optics bands** — the grey Mie/heuristic cloud optics
  (``jcm/physics/radiation/cloud_optics.py``, grey backend only) key their
  representative wavelength to each band's own wavenumber limits in
  ``jcm/physics/radiation/constants.py`` (``λ = 1e4/wn_mid``), so band *b* is
  always evaluated inside band *b*.

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
  aerosol/CCN effect (applying both double-counts it). The fallback — a constant
  radius with no LWC dependence — is live on every 1M composition, including the
  release-validated ``t63-echam-1m`` / ``t106-echam-1m`` configurations (#717);
  2M configurations use microphysical effective radii.
- `science` — the grey two-stream backend reads a single *broadband* aerosol
  profile (``aerosol.aod_profile``/``ssa_profile``/``asy_profile`` plus a column
  ``angstrom`` it band-scales itself) rather than the per-band arrays only
  ``rrtmgp.py`` consumes. ``JamOpticsTerm`` writes those broadband fields from
  the SW band centred nearest 550 nm, so a grey + JAM configuration keeps its
  aerosol direct effect — at band-centre rather than exact-550 nm accuracy, and
  only with ``jam_optics=True`` (the default; ``False`` leaves the carry slot
  radiatively passive).
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
- **Cloud inhomogeneity uses a single factor per phase, not ECHAM's
  convection-type switch.** The fixed 0.8/0.8 factors above are ECHAM's nn=63
  no-/deep-convection values (``zinhoml1``/``zinhomi``); ECHAM additionally drops
  the liquid factor to ``zinhoml2 = 0.4`` in shallow-convective columns
  (``ktype = 4``). jcm applies the uniform value because the radiation glue does
  not carry the convective type — correct for every column except shallow-
  convective ones (#870). The separate in-cloud-condensate cap
  (``_MAX_IN_CLOUD_CONDENSATE``) is only a NaN guard against thin-cloud
  optical-depth blow-up; it binds in ~0.003 % of cloudy cells and is *not* an
  inhomogeneity term.
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
