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
differentiable jittable forms. Gong sea salt is computed **online** from the 10 m
wind and open-water fraction, so it needs no input file; DMS reads a prescribed
seawater-concentration field and dust the five prescribed soil/source fields
below, and both are inert until those are supplied. Anthropogenic emissions are either bulk
super-sectors with in-model differentiable speciation or CAM6/MAM4-faithful
already-speciated per-tracer fields. Dry deposition
(``jcm/physics/aerosol/jam/drydep/``) is a resistance-in-series scheme with a
Slinn & Slinn (1980) sub-layer resistance; sedimentation
(``sedimentation/``) is per-mode Stokes settling with Cunningham slip; wet
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

#### Sea-salt and DMS emission wind

**What we do.** Both schemes read a diagnosed **10 m** wind, not the lowest
model level: ``SeaSaltEmissions`` and ``DmsEmissions`` take
``VerticalDiffusionData.wind_10m`` (see
[vertical diffusion](vertical_diffusion.md)) through
``jcm/physics/aerosol/jam/emissions/surface_wind.py::wind_10m``. Emission terms
run before vertical diffusion in the ECHAM ordering, so the value is the
previous step's — ``vertical_diffusion`` is a declared cross-step carry slot,
the same one-step lag the dust term's ``u*`` takes.

On the first step of a cold start no surface layer has been diagnosed yet (the
carry slot is zero-filled), and both schemes fall back to the lowest level for
that step, publishing a per-column flag ``wind_10m_model_level`` — 1 where the
model level was used, 0 where the diagnosed wind was — which
``ResetEmissionFluxes`` zeroes every step alongside the emission fluxes. A
resumed run carries a real 10 m wind and never takes the fallback.

**What ECHAM/CAM does.** HAMMOZ passes ``vphysc%velo10m`` to both schemes
(``mo_vphysc.f90``); Gong (2003) is fitted to ``u10**3.41`` and Nightingale
et al. (2000) to a piston velocity in ``u10``.

**Why we differ.** We do not, for the wind. The fallback is a `compute`
consequence of the operator-split ordering — the surface layer is diagnosed
later in the step than the emissions that consume it — rather than a scheme
choice, and it is reported rather than hidden.

**Status & known limitations.** The three surface-flux terms behave
differently on step 1 of a cold start, deliberately: dust emits nothing (its
carried ``u*`` is zero and a saltation threshold has no defensible value
without a surface layer), sea salt and DMS emit from the lowest level and flag
it, and the surface term has no step-1 case because it runs after vertical
diffusion. ``check_health`` reports the chunk-mean flag and fails a chunk only
on a *persistent* fallback (``chunk_idx > 0 and frac > 0.5``), because under
interval averaging the legitimate bootstrap step and a single defective step
mid-chunk give the same value; the per-step invariant is asserted in the unit
tests instead. A single-chunk run therefore degrades to report-only.

**Code pointers.**
- ``jcm/physics/aerosol/jam/emissions/surface_wind.py`` — ``wind_10m``,
  ``MODEL_LEVEL_WIND_KEY``.
- ``jcm/physics/aerosol/jam/emissions/seasalt.py`` — ``SeaSaltEmissions``.
- ``jcm/physics/aerosol/jam/emissions/dms.py`` — ``DmsEmissions``.
- ``jcm/physics/aerosol/jam/emissions/flux_diagnostic.py`` —
  ``ResetEmissionFluxes``, ``all_flux_keys``.
- ``jcm/diagnostics.py`` — ``check_health``.

**Validation evidence.** ``seasalt_test.py`` and ``dms_test.py`` pin the
reduction against the carry state per step and the flag per column;
``jcm/diagnostics_test.py`` pins the chunk-level rule. A 30-day T63L47
integration reduces sea-salt emission by 32 % and the DMS flux by 17 % against
the lowest-level wind, with the flag reading 1/480 on the first chunk and
exactly zero on every chunk thereafter.

#### Dust emission (Tegen / HAMMOZ)

**What we do.** A port of the MPI-BGC dust scheme as HAM2 configures it
(``ndust = 4``). A 191-class soil size grid spanning 0.2-1262 µm carries a
Marticorena & Bergametti (1995) threshold friction velocity ``u*t(D)``; ``u*``
comes from the diagnosed **10 m wind** through the fixed log law
``u* = vk·U10/ln(1000 cm/z0)`` with ``z0 = ndurough = 0.001 cm``, not from the
vertical-diffusion scheme. Per soil texture, four lognormal populations give the
relative surface ``srel`` (which drives the flux) and the relative mass
``srelV`` (which drives the emitted spectrum); the saltation flux is
``srel·(1+R)²·(1−R)·cd·u*³·α`` with ``R = u*t·nduscale·utsc/(feff·u*)``, and every
class ``k > 1`` sandblasts over classes ``1…k`` weighted by ``srelV``. A cell is a
mixture: ``1 − psrc`` of it keeps its mapped Zobler/East-Asian textures with the
unmapped remainder as type 1 (coarse), and ``psrc`` becomes soil type 10
(100 % silt, α = 1e-5). Emission needs ``u* ≥ 21·nduscale/feff`` cm/s and
``pot_source > r_dust_lai``; the surviving flux is multiplied by ``pot_source``
again, by ``1 − snow_cover``, and zeroed where the relative soil wetness exceeds
0.99.

The emitted spectrum is integrated onto MAM4's emission windows — accumulation
``0.1 ≤ D < 1 µm``, coarse ``1 ≤ D < 10 µm``, everything coarser discarded — with
the **number** taken from the flux-weighted effective diameter *within* each
window, ``D_eff = (Σ F / Σ F/D³)^(1/3)``, not from the mode's equilibrium
``dgnum``.

**What ECHAM/HAM does.** ``mo_ham_dust.f90`` (``set_dust_data``,
``bgc_dust_initialize``, ``bgc_dust_calc_emis``, ``bgc_read_annual_fields``,
``comp_nduscale_reg``) — Tegen et al. (2002), Marticorena & Bergametti (1995)
eqs. (5)-(7), (15), (17), (28)-(33), with Cheng et al. (2008)'s East-Asian
textures and the HAM2 selection of Zhang et al. (2012) §4.1.5. The module hands
out an 8-bin size-resolved flux; the bin-to-mode step lives outside it.

**Why we differ.**
- `science` (mode mapping) — HAM emits into M7's insoluble accumulation and
  coarse modes via an offline three-lognormal fit to the *global multi-annual
  mean* spectrum (Stier et al. 2005 §2.3.4). MAM4's modes are different
  (σ 1.80 vs 1.59/2.00), and the fit throws away the soil and wind dependence the
  scheme actually produces — the sub-micron mass fraction spans 0.00004 to 0.151
  across textures. We integrate the **online** spectrum over MAM4's windows
  instead. The window edges (0.1 / 1 / 10 µm) are MAM4's convention, so HAM's
  8-bin structure is deliberately *not* reproduced — only the underlying
  191-class spectrum is. The ≥ 10 µm remainder is discarded, as HAM discards its
  super-coarse mode, and nothing is renormalised; it is published per column as
  ``dust_supercoarse_flux`` because it is large (0.11-0.75 of the total).
- `data` (snow) — HAM multiplies by ``1 − cvs`` with ECHAM's ``physc`` snow-cover
  formula (a ``tanh`` in snow depth with orographic-σ damping, a canopy-snow
  substitute and ``cvs = 1`` on glaciers). jcm has no prognostic snow depth
  (``land.py`` still zeroes its tendency), so ``1 − snowc_am`` stands in.
  ``snowc_am`` is zero on permanent ice where ECHAM sets ``cvs = 1``, which is
  harmless because ``pot_source`` is zero there.
- `data` (soil moisture) — the ``ws/wsmx > 0.99`` cut-off is unconditional in
  every HAM preset, but jcm carries SPEEDY's vegetation-weighted availability
  index ``soilw_am``, not ECHAM's ``ws/wsmx``. It is wired so the term is
  structurally complete, and is close to inert because ``soilw_am`` is capped at
  field capacity by construction (#787).
- `data` (resolution) — the HAMMOZ inputs exist only at T63. The T106 products
  are derived from them by nearest neighbour (conservative regridding cannot
  refine a grid, and the region mask is categorical), and the ``ndust = 3``
  resolution polynomial carries an explicit source warning that
  ``nduscale_reg`` must be re-tuned above T63 — which applies to jcm's T106 and
  ne30 configurations too (#802).

**Status & known limitations.**
- The **``U10 = 10 m/s`` texture switch is a hard step**: above it the
  preferential source keeps type 10's flux magnitude but emits type 11's clay
  spectrum, a ~3400x jump in sub-micron mass. It is ported as the step it is, so
  ``jax.grad`` sees zero through it and it will show up in any optimisation
  (#664). ``dust_test.py`` asserts the zero gradient deliberately.
- The sandblasting weights sum to slightly more than 1 (the Fortran's numerator
  includes class 1 while its denominator excludes it, ~3e-5): reproduced, not
  fixed.
- The soil-type file holds two **overlapping** partitions — global Zobler
  ``type2/3/4/6`` and Cheng's Chinese ``type13..17``, summing to 1.85 over the
  Gobi. Summing all nine drives the type-1 residual to −0.85 and the flux
  negative; ``k_dust_easo = 2`` (the default) replaces such cells with the
  East-Asian textures outright, ``= 1`` zeroes them, and ``= 0`` is the branch
  the source itself labels buggy.
- The Fécan et al. (1999) moisture correction and the satellite roughness map
  are both implemented but **off by default**, matching HAM2. With
  ``ndurough = 0.001 cm = z0s`` the drag partition ``feff`` is identically 1, so
  the roughness map is read and immediately overwritten exactly as the Fortran
  does. Fécan additionally needs a gravimetric water content jcm does not carry.
- ``nduscale_reg`` is HAM's only global tuning knob and it scales the
  *threshold*, so a larger value emits **less**. The default is the T63
  free-running vector ``(1.05, 1.45, 1.45, 1.05, 1.05, 1.05, 1.45, 1.05)``;
  ``DustParameters.preset(3)`` selects Stier et al. (2005) instead.

**Code pointers.**
- ``jcm/physics/aerosol/jam/emissions/dust.py`` — ``DustEmissions``,
  ``DustParameters``, ``threshold_friction_velocity``,
  ``soil_size_distributions``, ``emission_weight_matrix``, ``SOIL_TABLE``,
  ``MIXTURE_ROWS``, ``DUST_SUPERCOARSE_KEY``.
- ``jcm/forcing.py`` — ``read_dust_source``, ``read_dust_preferential``,
  ``read_dust_soil_types``, ``read_dust_regions``, ``read_dust_roughness``.
- ``jcm/forcing_assembly.py`` — ``_attach_dust``.
- ``jcm/data/mirror/dust.py`` — ``build_dust_product``.

**Validation evidence.** ``dust_test.py`` pins ``u*t`` at eight diameters
against MB95 (1091.08 cm/s at 0.2 µm to 66.30 at 1262 µm, minimum 20.4502 at
76.04 µm), the emission onset at ``U10 = 6.2377 m/s``, the single-class flux
chain (soil type 2, ``u* = 40 cm/s``, ``D = 20 µm`` → 4.4758e-10 g cm⁻² s⁻¹),
the matrix form of the sandblasting redistribution against a direct
transcription of the Fortran loops for all twelve mixture rows, the East-Asia
overlap guards, the region vector, the snow and saturation cut-offs, the
emitted ``D_eff`` (0.5698 / 1.9125 µm for a medium soil, against MAM4's own
0.310 / 5.64 µm) and the gradients through α and ``nduscale_reg``.

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
