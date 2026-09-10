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

**Partial-cloud / overlap** differs by backend. **RRTMGP** uses full **McICA**
(``jcm/physics/radiation/mcica.py``): one stochastic binary cloud profile per
g-point, seeded deterministically per column and model step, with three overlap
rules — random, maximum-random (Geleyn-Hollingsworth), and
generalised-exponential with a decorrelation length. The **grey** backend
instead combines one clear and one cloudy beam weighted by the overlap-derived
total cover (``column_total_cover``); the **NN emulator** consumes a
deterministic overlap-derived expectation (no stochastic sampling); **SPEEDY**
carries its own cloud formulation. Swapping backends therefore changes the
cloud-overlap treatment, not just the gas optics. The AeroCom
total-cloud-cover diagnostic uses the maximum-random closure.
Radiation **sub-steps** on the ECHAM-family backends (grey, RRTMGP, NN
emulator): a gate (``radiation_should_compute``) skips the expensive solve and
rescales cached heating on intermediate steps. SPEEDY has its own, different
cadence — ``SpeedyFlags`` gates shortwave every ``nstrad`` calls and the skipped
calls contribute a *zero* shortwave tendency rather than replaying cached
heating (#752).

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
- **No sub-grid cloud inhomogeneity scaling.** ECHAM's continuous
  ``zinhoml = LWP^{-p}`` rescaling is not implemented; instead a one-sided
  in-cloud-condensate cap (``_MAX_IN_CLOUD_CONDENSATE``) prevents thin-cloud
  optical-depth blow-up. It binds in a negligible fraction of cloudy cells — a NaN
  guard, *not* inhomogeneity coverage (#678).
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
