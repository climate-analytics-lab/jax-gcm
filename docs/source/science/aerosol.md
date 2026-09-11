# Aerosol

Two aerosol paths exist: the prescribed MACv2-SP simple plumes (the default
optics source) and the prognostic JAM modal aerosol package. The provenance here
is deliberately hybrid, and several sites carried an inherited reference label
that does not match the code; this section records the *true* reference and the
deviations, which is the whole reason the living document exists.

## JAM prognostic modal aerosol

**What we do.** JAM is an online, prognostic modal aerosol package built on the
**MAM4-MOM 4-mode** population (accumulation, Aitken, coarse, primary-carbon),
with the interstitial mass/number (``m_``/``n_``) and gas precursors (``g_``)
carried as flat ``state.tracers`` keys, and the cloud-borne mirrors
(``mc_``/``nc_``) living in the cross-step physics carry rather than the tracer
set (see the cloud-borne store below). It is assembled as an ordered
HAMMOZ-style ``PhysicsTerm`` chain by
``jcm/physics/aerosol/jam/jam_terms.py::jam_aerosol_physics`` and spliced into
``echam_physics``. The chain is: aerosol carry-slot seeder → cloud-borne carry
store → reset emission accumulators → natural emissions
(sea-salt, DMS, dust) + optional anthropogenic / pre-speciated → physics-side
vertical transport (turbulent diffusion of all tracers + convective transport of
interstitial/gas tracers) → prescribed oxidants + gas-phase sulfur chemistry
(producing the H₂SO₄/SOAG the core condenses the same step) →
**microphysics core** → optional online optics → ARG
activation → heterogeneous ice nucleation → sedimentation → dry deposition →
cloud-borne exchange → aqueous sulfur chemistry → wet scavenging. Diagnostics
thread in this call order, so custom compositions should preserve it.

The microphysics core is a swap point: ``"placeholder"`` (the bare-factory
default, keeping the Apache-2.0 core importable with no GPL dependency) is a
**κ-Köhler equilibrium** core with zero tendency that still exposes the real MAM4
mode/species geometry so the full harness runs end-to-end; ``"mam4_jax"`` is the
real MAM4-JAX box model (GPL-3.0, optional ``jcm[mam4]`` extra, imported lazily)
doing ``calcsize → wateruptake → amicphys`` (gas-aerosol exchange, rename, binary
H₂SO₄ nucleation, coagulation, carbonaceous ageing). Every process term reads
modal properties from the core's diagnostic and shares one population spec, and
all process parameters are differentiable ``@tree_math.struct`` leaves.

**What ECHAM/CAM/MAM4 does.** The population geometry is **E3SM/MAM4**, not
ESCOMP/CAM: mode names, ``nspec_amode``, ``lspectype_amode`` from
``modal_aero_data.F90``; σ_g and ``dgnum`` from ``rad_constituents.F90``
(MODAL_AERO_4MODE_MOM branch), cross-checked against the MAM4-JAX port
(``reflective-org/MAM4-JAX`` @ ``ccd872b``). Species density/κ from
``rad_constituents.F90``. The ``amicphys`` microphysics is E3SM
``modal_aero_amicphys.F90``; carbonaceous ageing is MAM4's
``mam_pcarbon_aging_1subarea``. The surrounding process chain (emissions,
deposition, scavenging, aqueous chemistry) follows the **ECHAM-HAMMOZ** lineage.
The JAM harness is therefore an **E3SM/MAM4 microphysics core inside a
HAMMOZ-lineage process harness**, with specific CAM borrowings where HAMMOZ has
structural gaps (below). See {doc}`../design/jam_carbon_aging`.

**Why we differ.**
- `science` — where MAM4 (E3SM) and HAM disagree on a coefficient (e.g. the
  carbon-ageing monolayer threshold: MAM4 ``amicphys`` = 3, HAM ``m7_coat`` = 1),
  the MAM4 value is the default and the parameter is exposed and differentiable.
- `compute` / `differentiability` — the MAM4-JAX core is float64 by default
  because its float32 reverse pass is non-finite upstream; forward-only production
  drivers opt into a *scoped* float32 core while the host model keeps its
  precision. Gas tracers into the core are ``h2so4`` / ``soag`` only (SO₂/DMS
  oxidation is jcm-side).

**Status & known limitations.** Every shipped ``echam-jam*`` configuration pins
``jam_microphysics: mam4_jax`` (requiring the ``jcm[mam4]`` extra); the
zero-tendency κ-Köhler placeholder is the bare-factory default and the
documented fallback when the GPL extra is unavailable. The core's cloudy ``amicphys``
sub-area is not ported upstream, so cloud-borne activation is the harness's job
(``ArgActivation`` / ``CloudBorneExchange``) and the core runs clear-sky. Aerosol
lifetimes vs observations (``tools/jam_burden_report.py``): BC roughly matches
observations, SO4 is somewhat long (wet scavenging too weak), and the sea-salt
source under-emits (see {doc}`../design/dinosaur_sl_jam_configuration`).

### Cloud-droplet activation (ARG)

**What we do.** Abdul-Razzak & Ghan (2000) closed-form maximum-supersaturation
activation for the log-normal modes, using the **κ-Köhler** critical
supersaturation with κ read from the MAM4 core's per-mode volume-weighted
hygroscopicity, and a single characteristic updraft ``w = √(2·TKE/3)`` from the
previous step's TTE-TKE. Two shape-coefficient variants are selectable: every shipped ``echam-jam*``
configuration pins ``ghosh2025`` (the revised coefficients); ``arg2000`` (the
original paper's) is the bare-factory default.
(``jcm/physics/aerosol/jam/activation/arg.py``, ``arg_term.py``.)

**What ECHAM/CAM does.** This is CAM's ``ndrop.F90`` structure
(``activate_modal`` / ``maxsat``, f1/f2 shape factors, volume-weighted κ-mixing,
``√(2/3·TKE)`` single updraft), fed genuine MAM4 modal properties. ECHAM-HAM's
``mo_ham_activ::ham_activ_abdulrazzak_ghan`` implements the same ARG closed form
but with a **van 't Hoff electrolyte "B" (soluble-ion) Köhler** hygroscopicity and
a ``0.7·√TKE`` updraft.

**Why we differ.**
- `science` (provenance correction) — the true reference is **CAM ``ndrop.F90``**,
  not HAM. Because HAM uses κ-free electrolyte hygroscopicity, "fixing" jcm toward
  the historically-cited HAM would replace κ with ``B`` for the MAM4 modes and
  materially change the hygroscopicity of any dust- or BC-rich mode. The coupling
  is correct as written; the historical HAM citation was a mislabel corrected as
  part of this document.

**Status & known limitations.** The ``ghosh2025`` variant's coefficients are
fitted to the paper's tables (flagged in code) and off by default. A negative
floor is applied before the number-weighted fraction to survive spectral ringing
on the cold-start aerosol field.

### Cloud-borne aerosol store

**What we do.** When cloud-borne aerosol is enabled, the in-droplet ``mc_*`` /
``nc_*`` phase lives in the **cross-step physics carry**, never in dycore tracers.
``jcm/physics/aerosol/jam/cloud_borne_store.py::CloudBorneCarryStore`` owns the
carry slot, guarantees a structurally stable pytree every step, and applies
turbulent vertical mixing of the carry with the same TTE-TKE coefficients the
interstitial tracers get. ``CloudBorneExchange`` cycles activation-transfer +
resuspension against the current cloud field.

**What ECHAM/CAM does.** This is CAM's ``qqcw``-in-``pbuf`` pattern (cloud-borne
aerosol in the physics buffer, not advected). ECHAM-HAM (M7/TOMAS-style, implicit)
has no explicit prognostic cloud-borne phase — it scavenges interstitial aerosol
by activated fraction.

**Why we differ.**
- `compute` / `differentiability` / stability — a controlled 30-day A/B showed
  dycore-advected ``mc_*`` / ``nc_*`` mirrors went ~90%-of-cells negative under
  spectral advection at ~2.2× the carry cost, and even against a fair
  semi-Lagrangian baseline the carry agreed to a few percent at lower cost. The
  advected-tracer store was removed by decision (pySES is CAM-SE; CAM itself uses
  ``pbuf``; pySES is the most tracer-count-sensitive backend). The trade given up
  — resolved-scale advection of in-droplet aerosol — is one CAM accepts too. See
  {doc}`../design/dinosaur_sl_jam_configuration`.

### Convective tracer transport + in-plume scavenging

**What we do.** ``jcm/physics/convection/tracer_transport.py::ConvectiveTracerTransport``
is a bulk entraining/detraining plume with compensating subsidence plus a mirrored
downdraft leg, driven by the mass-flux/entrainment profiles the Tiedtke term
publishes (carrying the scheme's rescale+cap ledger) — **one step lagged**: the
term sits before convection in the chain, so it reads the *previous* step's
plume profiles (and is a no-op on the first step, before any exist). Transient
aerosol–convection coupling therefore trails the driving convection by one
``dt``; the same lag applies to the in-plume scavenging below. Detrainment from plume
continuity; updraft concentration from an upward convex-mix scan; downdraft from
the mirror continuity + downward scan. In-plume scavenging follows CAM
``aero_convproc`` (mirage2 form): a first-order removal from the plume's
condensate-to-precip conversion, applied inside the ascent scan so aerosol
scavenged low never detrains aloft. Only interstitial + gas tracers are
transported. **This module's header is the gold-standard provenance-comment
example** the rest of the tree is measured against.

**What ECHAM/CAM does.** ECHAM transports every tracer through Tiedtke
(``cuxtte`` / ``mo_cuascn`` xt budgeting); CAM's ``convtran`` does the same;
in-plume scavenging is CAM ``aero_convproc`` (mirage2). The downdraft is ECHAM
``cudlfs`` / ``cuddraf`` / CAM ``convtran``'s ``cond`` loop.

**Why we differ.**
- `science` (documented deviation) — the downdraft **seeds the level of free
  sinking by entraining environment air** (exact column telescoping, matching
  CAM's "environment entrainment only, no transformation in the downdraft"),
  rather than ECHAM ``cudlfs``'s 50/50 updraft/wet-bulb-environment mix. Aerosol
  resuspension by evaporating convective precip (CAM ``dcondt_prevap``) is not
  modelled — the removed flux goes straight to the surface, matching the existing
  wet-deposition treatment. When active, ``WetScavenging`` retires its
  own environment-profile convective in-cloud pathway to avoid double-counting.

### Emissions, deposition, sedimentation, wet scavenging, ice nucleation

**What we do.** Natural emissions are faithful ports of the HAMMOZ schemes — Gong
(2003) sea-salt, Nightingale (2000) DMS, Tegen et al. (2002) dust — collapsed to
differentiable jittable forms. Gong sea salt is computed **online** from the
lowest-level wind and open-water fraction, so it needs no input file; DMS and
dust read prescribed seawater-concentration / erodibility fields from
``ForcingData`` and are inert until those are supplied. Anthropogenic emissions are either bulk
super-sectors with in-model differentiable speciation or CAM6/MAM4-faithful
already-speciated per-tracer fields. Dry deposition
(``jcm/physics/aerosol/jam/drydep/``) is a resistance-in-series scheme with a
Slinn & Slinn (1980) sub-layer resistance; sedimentation
(``sedimentation/``) is per-moment Stokes settling with Cunningham slip; wet
scavenging (``wetdep/``) is in-cloud nucleation + below-cloud impaction + a
**re-evaporation re-injection ledger** that returns carried aerosol to the
interstitial phase where precip evaporates, and it deliberately excludes the
sedimenting cloud-ice flux from the in-cloud carrier flux. Ice nucleation
(``ice_nucleation/``) writes an ``ice_nuclei`` field for the 2M cloud scheme, with
two schemes: ``niemand`` (default; Niemand et al. 2012) and ``lohmann_diehl``
(Lohmann & Diehl 2006 + Meyers 1992 deposition).

**What ECHAM/CAM does.** Deposition mirrors ``mo_hammoz_drydep`` /
Ganzeveld (Slinn & Slinn 1980); sedimentation ``mo_ham_sedimentation``; wet
scavenging ``mo_hammoz_wetdep`` / ``mo_ham_wetdep`` (``peffwat`` / ``peffice``
re-evaporation ledger); ice nucleation Lohmann & Diehl (2006), Niemand et al.
(2012), Meyers et al. (1992).

**Why we differ.**
- `science` (provenance correction) — dry deposition is a **HAMMOZ/Ganzeveld
  (Slinn & Slinn) resistance-in-series** scheme specialised to the aquaplanet ocean
  surface, *not* a CAM ``aero_model_drydep`` port; ``r_a`` uses a neutral log-law
  (a Monin-Obukhov stability correction is a noted future refinement). Convectively
  scavenged aerosol is deposited directly (convection exposes no evaporation
  profile). Ice-nucleation active-site densities are calibratable rather than
  claiming exact published coefficients.

**Code pointers (JAM).**
- ``jcm/physics/aerosol/jam/jam_terms.py`` — ``jam_aerosol_physics`` (the ordered
  chain and its options).
- ``jcm/physics/aerosol/jam/microphysics/`` — ``mam4_data.py`` (population),
  ``mam4_jax.py`` (real core), ``placeholder.py`` (κ-Köhler core); ``species.py``,
  ``gas_species.py``.
- ``jcm/physics/aerosol/jam/activation/`` — ``arg.py``, ``arg_term.py``.
- ``jcm/physics/aerosol/jam/cloud_borne_store.py``, ``cloud_borne.py``.
- ``jcm/physics/convection/tracer_transport.py`` — ``ConvectiveTracerTransport``.
- ``jcm/physics/aerosol/jam/emissions/`` (seasalt, dms, dust, anthropogenic,
  prescribed); ``drydep/``; ``sedimentation/``; ``wetdep/``; ``ice_nucleation/``;
  ``optics/optics_term.py``.

### Aerosol removal: below-cloud scavenging, settling, and the removal chain

**What we do.** Below-cloud impaction is CAM's Slinn coefficient, not a
size-power law: ``Λ = sol_factb · Λ₁(D_wet) · R`` with ``R`` the precipitation
flux (kg m⁻² s⁻¹ ≡ mm s⁻¹) and ``Λ₁`` in 1/mm the collection-efficiency integral
``E = min(E_brown + E_intercept + E_impact, 1)`` over a raindrop spectrum and the
mode's lognormal, evaluated separately against number and volume weights so the
two moments carry their own coefficient. ``Λ₁`` is tabulated once per mode at
construction over the wet/dry diameter growth ratio and read back by
differentiable log-linear interpolation. ``sol_factb`` is a differentiable
parameter at CAM's namelist default 0.1 for interstitial aerosol and structurally
zero for the cloud-borne phase, which is in-droplet by definition; no cloud
weighting is applied, because the swept precipitating volume cancels against the
in-precip-area rain rate and the stratiform in-cloud pathway acts on the
cloud-borne tracers rather than the interstitial ones.

Stokes settling and the Slinn quasi-laminar resistance are evaluated at the
**wet** particle's density, the mass-weighted mixture of dry material and
condensed water ``ρ_wet = (ρ_dry + (g³ − 1)·ρ_w)/g³`` with ``g`` the κ-Köhler
growth factor, so the density and the radius describe the same particle.

The three removal terms are **operator-split**: sedimentation, then dry
deposition, then wet scavenging, each acting on the working copy its predecessors
left, reconstructed from the running tendency the driver publishes on both its
whole-grid and column-vectorized hosts. The reconstruction folds in every term
already run in the step, not only the removal chain — emissions, convective
tracer transport, sulfur chemistry, the microphysics core and activation all
precede sedimentation in the chain — so aerosol emitted or formed this step is
present to be removed and aerosol convection has exported is not. Each term
removes at most what it sees, so the chain cannot remove more than the cell
holds.

``dry_<species>`` is the column-integrated settling **plus** turbulent/Brownian
surface removal, ``wet_<species>`` the scavenging net of re-evaporation; each term
integrates its own tendencies, so the ledger is the mass the terms actually took
rather than a separately-derived flux. Behind the chain, the physics interface
floors every aerosol and gas tendency at the rate that empties the tracer and no
further, leaving sources untouched.

**What ECHAM/CAM does.** The coefficient and its tabulation are CAM
``aero_model.F90::calc_1_impact_rate`` (Slinn collection efficiency; Marshall-Palmer-like
drop spectrum) with ``modal_aero_bcscavcoef_init`` / ``modal_aero_bcscavcoef_get``
(20 growth-ratio nodes spaced ``log(1.25)``, evaluated at 273.16 K / 750 hPa and
the mode's first-species material density, clamped below the grid and linearly
extrapolated above); the grid-mean rate is ``wetdep.F90::wetdepa_v2``'s
below-cloud term with ``sol_factb`` = ``sol_factb_interstitial``. Wet density is
CAM ``modal_aero_wateruptake``'s ``wetdens``, the density
``modal_aero_depvel_part`` pairs with the wet radius. Sequential application of
each removal process to an updated working copy is the ECHAM/CAM operator split.

**Why we differ.**
- `compute` — ``Λ₁`` is tabulated per mode at construction rather than evaluated
  per cell, exactly as CAM tabulates it: the integral is a 50×51 double sum whose
  per-cell evaluation would dominate the physics cost. The runtime lookup is the
  differentiable part.
- `science` (documented deviation) — two harmless departures from
  ``modal_aero_bcscavcoef_get``: the lookup always interpolates rather than
  short-circuiting a growth ratio within 1 % of unity to the base node, and there
  is no ``isprx`` precipitation mask because ``Λ ∝ R`` already vanishes without
  precipitation.
- `differentiability` — the operator split reconstructs the working copy from the
  driver's running tendency instead of mutating state mid-step, so the chain stays
  a pure function of the step-start state and its tendencies.

**Status & known limitations.** CAM's fallback when ``sol_factb_interstitial`` is
left unset — the mode's mass-weighted hygroscopicity — is not ported; the scalar
namelist default is used, which is the path every supported CAM configuration
takes. That fallback would give sea salt a solubility factor above one. The
operator split is order-dependent by construction; the order is the composed one.
The deposition ledger equals the chain's mass change up to the interface
non-negativity guard, which each term records its contribution before. Ice
sedimenting to the surface as snow carries no aerosol removal, matching CAM,
which has no ice-phase aerosol scavenging.

**Code pointers.**
- ``jcm/physics/aerosol/jam/wetdep/impaction.py`` — ``impaction_scavenging_rates``
  (the Slinn integral), ``build_impaction_table``, ``bcscavcoef`` (the lookup).
- ``jcm/physics/aerosol/jam/wetdep/wetdep_term.py`` — ``below_cloud_rate``,
  ``WetDepParameters`` (``sol_factb``).
- ``jcm/physics/aerosol/jam/sedimentation/sedi_term.py`` — ``stokes_velocity``,
  ``moment_radius``.
- ``jcm/physics/aerosol/jam/drydep/resistances.py`` — ``deposition_velocity``.
- ``jcm/physics/aerosol/jam/microphysics/placeholder.py`` —
  ``equilibrium_modal_state`` (wet density from the κ-Köhler growth factor).
- ``jcm/physics/aerosol/jam/removal_split.py`` — ``split_view``.
- ``jcm/physics/composable_physics.py`` —
  ``ComposablePhysics._compute_tendencies_3d``,
  ``ComposablePhysics._compute_tendencies_columns`` (both publish the running
  tendency).
- ``jcm/physics/aerosol/jam/emissions/flux_diagnostic.py`` —
  ``accumulate_deposition_fluxes``.
- ``jcm/physics_interface.py`` — ``verify_tendencies``,
  ``has_non_negative_tendency``.

**Validation evidence.** ``impaction_scavenging_rates`` reproduces CAM's own
compiled ``calc_1_impact_rate`` to eight significant figures at eight
(diameter, σ, density) points, pinned in
``jcm/physics/aerosol/jam/wetdep/impaction_test.py`` (``CAM_REFERENCE``); the same
file pins the Greenfield minimum in 0.05–0.5 µm, saturation with size, and
``check_vjp``/``check_jvp`` on a size scale.
``jcm/physics/aerosol/jam/removal_split_test.py`` asserts the composed chain never
removes more than the cell holds and that ``dry_*`` + ``wet_*`` equals the chain's
interstitial + cloud-borne mass change.
``jcm/physics/aerosol/jam/microphysics/placeholder_test.py`` pins the wet-density
mixture against the κ-Köhler growth factor. A 30-day T63L47 A/B against the
un-fixed removal chain gives a sea-salt lifetime of 0.50 d (observed 0.4–1 d) and
a 32 % wet / 68 % dry+sedimentation pathway split against HAM's published ~30/70;
accumulation-mode sulfate and black carbon are unchanged, as expected for a mode
sitting in the Greenfield gap.

## MACv2-SP simple plumes

**What we do.** A faithful port of MACv2-SP (Stevens et al. 2017): nine
anthropogenic plumes with rotated Gaussians and time weights, dz-weighted β
vertical profiles truncated at the orography, per-plume AOD-weighted optics per SW
band, and the Twomey factor from column plume AOD against a natural background.
This is the default aerosol-optics path (overwritten by JAM online optics when JAM
is composed). The companion **SPA** rule (``jcm/physics/aerosol/spa.py``, Lin et
al. 2025) turns MACv2-SP CCN into a 2M cloud-droplet floor — a sublinear
prescribed-aerosol activation source interchangeable with ARG.
(``jcm/physics/aerosol/macv2_sp.py::get_simple_aerosol``,
``macv2_sp_params.py::AerosolParameters``.)

**What ECHAM/CAM does.** MACv2-SP ``sp_aop_profile``
(``mo_simple_plumes_v1.f90``, Stevens et al. 2017), the standard CMIP6 simple-plume
anthropogenic aerosol forcing. SPA: Lin et al. (2025, ACP 25, 15105).

**Why we differ.** Faithful — no material deviation. SPA's CCN input is sourced
from the MACv2-SP plumes rather than a standalone climatology, and its fit
parameters are differentiable leaves for calibration.

**Status & known limitations.** Time weights default to perpetual-2005 amplitude
(no seasonal cycle) until the forcing supplies them; a smooth-cap option replaces
SPA's hard ``min`` for gradient calibration.

**Validation evidence (aerosol).** JAM: ``jam_integration_test.py``,
``jam_phase0_test.py``, ``microphysics/mam4_jax_test.py``, and per-process
``*_test.py`` under each ``jam/`` sub-package. MACv2-SP / SPA:
``macv2_sp_test.py``, ``spa_test.py``, ``per_band_optics_test.py``. Design
references: {doc}`../design/jam_carbon_aging`,
{doc}`../design/dinosaur_sl_jam_configuration`,
{doc}`../design/aerosol_optics_diagnostics`,
{doc}`../design/aerocom_erfari_sampling`.
