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
observations and sea salt is in range, while SO4 is somewhat long (wet
scavenging too weak). See {doc}`../design/dinosaur_sl_jam_configuration`.

### Online aerosol optics

**What we do.** ``JamOpticsTerm`` (``jcm/physics/aerosol/jam/optics/optics_term.py``)
gives the modal population a direct radiative effect: per mode and radiation
band it forms a **volume-mixed complex refractive index** over the mode's dry
species plus its **hygroscopic water**, looks up Mie efficiencies at the wet
size parameter, integrates them over the mode's lognormal with an 8-node
Gauss–Hermite quadrature in ``ln r`` (σ_g preserved under growth), and sums
extinction across modes; single-scattering albedo and asymmetry are
extinction-/scattering-weighted. The water volume is
``V_w = V_dry·(g³ − 1)`` with the hygroscopic growth factor
``g = r_wet/r_dry``: each core grows a mode by applying one ratio to the whole
of it, so every radius scales by the same ``g`` and the wet third moment is
exactly ``g³`` times the dry one, whatever ``σ_g``. ``V_dry`` is the species
mass over density summed within the mode. Water's share of a mode's volume —
hence of its apportioned extinction, ``od550aerh2o`` — is therefore
``(g³ − 1)/g³``. Only the *ratio* of the two radii enters, so this holds
whether or not the core clips ``dg`` to a per-mode bound.

Where ``dg`` is **unclipped** there is a stronger property: ``V_dry + V_w`` is
then the third moment of the very lognormal the Gauss–Hermite quadrature
integrates over, so the mixing rule and the size integral describe one
particle population. Clipping breaks that second statement (not the first):
the size integral follows the clipped radius while ``V_dry`` follows the mass,
and the two part company by ``(dg_clip/dg_true)³``.

**What ECHAM-HAM/MAM does.** HAM carries aerosol water as a per-mode tracer
and volume-mixes it with the mode's dry species before the optics lookup
(``mo_ham_rad.f90::ham_rad_refrac_volume``, the "Add aerosol water" block at
lines 277-298 summing ``zv = mass/density`` into the same ``znrsum``/
``znisum``/``zvsum`` as the dry species), then reports it as an optics
component of its own — ``zvcomp`` at ``ham_rad_diag`` lines 1898-1906,
apportioned by volume fraction at 1927-1933 and written to the
``TAU_COMP_WAT`` stream (``mo_ham_streams.f90:652``), which ``od550aerh2o``
mirrors.

CAM computes the same quantity per particle:
``modal_aero_wateruptake.F90::modal_aero_wateruptake_sub`` takes
``wetvol = (4/3)π·wetrad³`` and ``wtrvol = wetvol − dryvol`` (lines 596-598),
and ``qaerwat = ρ_w·naer·wtrvol`` (line 456). That is consistent because its
``dryrad`` is the dry **volume-mean** radius, defined by
``dryrad = (dryvol/((4/3)π))^(1/3)`` from the single-particle-mean ``dryvol``
(``modal_aero_calcsize.F90::modal_aero_calcdry``, lines 1565-1567), so
``naer·wtrvol`` telescopes to exactly ``V_dry·(g³ − 1)``. CAM's *number-median*
diameter is a separate quantity, ``dgncur_a = (drv/(dumfac·num))^(1/3)`` with
``dumfac = exp(4.5 ln²σ_g)·π/6`` (``modal_aero_calcsize.F90:549, 685``), and
the mode is grown by one ratio applied to it,
``dgncur_awet = dgncur_a·(wetrad/dryrad)`` (``modal_aero_wateruptake.F90:455``).

**Why we differ.** Faithful in the mixing rule and the lognormal integration;
the LUT-and-quadrature evaluation is a `compute` choice (Mie paid once at
construction, differentiable table interpolation per step). The mode-volume
form is stated explicitly because CAM's **per-particle** shape,
``N·(4/3)π·(r_wet³ − r_dry³)``, is *not* transferable to this code's radii:
CAM applies it to the volume-mean ``dryrad``, whereas ``r_dry`` here (and in
MAM4-JAX) is the **number-median** radius, defined through the third moment
``V = N·(π/6)·Dg³·exp(4.5 ln²σ_g)``. On those radii ``N·(4/3)π·r_dry³`` is not
``V_dry`` but ``V_dry·(dg_clip/dg_true)³/exp(4.5 ln²σ_g)``. Off a clip bound
that understates the water by ``exp(4.5 ln²σ_g)`` — 2.70 for σ_g = 1.6
(Aitken, primary carbon), 4.73 for σ_g = 1.8 (accumulation, coarse). On a clip
bound the factor moves either way: clipping down to ``dgnum_hi`` understates
further, while clipping up to ``dgnum_lo`` — more number than the mass
supports — can *overstate* the water instead. Either way the mixed index would
be wrong on these radii, which is why the volume form is the one used. It is
also identically zero where a mode holds no dry material, whatever ringing the
number field carries.

**Status & known limitations.** Spherical, homogeneously mixed particles: the
in-repo pathway has no core–shell treatment of black carbon, so BC's imaginary
index is smeared over the whole particle by the volume mixing rule above. An
alternative bulk-optics pathway — a core–shell treatment, or a neural emulator
of the mode integral — is supplied out-of-tree against the per-mode seam in
{doc}`../design/jam_optics_mode_seam`, which is why no second Mie pathway is
carried here. Per-species optics are an apportionment of the mixed mode's
extinction, not a decomposition — see
{doc}`../design/aerosol_optics_diagnostics`.

Because the water volume is number-free, the mixed refractive index is
scale-free in the mode's masses, so **the core's ``dg`` diagnosis is the only
channel from aerosol burden to radiation**. That channel saturates on a mode
whose ``dg`` sits on a ``dgnum_lo``/``dgnum_hi`` bound: its radii, and hence
its cross-section, stop responding to mass entirely. Neither core adjusts
number to bring a clipped mode back inside its bounds the way MAM4's
``calcsize`` does, so a clipped mode stays clipped — tracked in issue #823.

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

The in-plume removal rate is keyed to the plume's own condensate,
``ConvectionData.qc_conv + qi_conv``, which the Tiedtke ledger publishes as the
updraft liquid water where the mass flux is active. Its liquid/ice split is
taken at the **updraft** temperature, not the environment's.

Convective **below-cloud** (impaction) scavenging in
``jcm/physics/aerosol/jam/wetdep/wetdep_term.py::WetScavenging`` is driven by the
level-wise convective precipitation flux ``ConvectionData.precip_flux`` — the
total rain + snow entering each layer from above, i.e. generation above less the
sub-cloud evaporation already charged above, from the same ``cuflx`` budget that
produces the surface precipitation, and carrying the convective ledger's
per-column cap scaling. Because that profile is zero above the level where
convective precip first forms, it is itself the cloud-top confinement of the
convective washout; no separately diagnosed convective cloud top is needed.

That washout acts only in the fraction of the grid box the convective rain
falls through, HAMMOZ's **updraft area** ``f_cu = M_u / (ρ·w_u)``
(``wetdep_term.py::conv_precip_cover``): the updraft mass flux
``ConvectionData.mass_flux_up`` over the air density and an assumed in-cloud
updraft velocity ``w_u = 2 m/s`` (``WetDepParameters.conv_updraft_velocity``,
differentiable). Below the cloud base the mass flux is tapered as ECHAM
``cuflx`` tapers it — linearly in the air mass below the interface to zero at
the surface, squared for mid-level convection — so the shaft keeps its
footprint under the base; the taper is rebuilt from the layer masses because
``mass_flux_up`` publishes the plume profile alone. Each layer loses
``f_cu · (1 − exp(−Λ·Δt))`` of its interstitial aerosol to the convective
carrier (``conv_below_cloud_rate``), with ``Λ`` evaluated at the grid-mean
convective flux, so a step can take at most the covered fraction.

**What ECHAM/CAM does.** ECHAM transports every tracer through Tiedtke
(``cuxtte`` / ``mo_cuascn`` xt budgeting); CAM's ``convtran`` does the same;
in-plume scavenging is CAM ``aero_convproc`` (mirage2). The downdraft is ECHAM
``cudlfs`` / ``cuddraf`` / CAM ``convtran``'s ``cond`` loop. The in-cloud and
below-cloud pathways are distinct sinks driven by different quantities in both
references — ECHAM ``xtwetdep`` and CAM ``wetdepa`` take the in-cloud rate from
the local condensate-to-precip conversion and the below-cloud rate from the
precipitation flux falling through the layer, never from the surface value. The
convective precipitation profile is ECHAM ``mo_cufluxdts.f90::cuflx``. The
liquid/ice phase of an in-plume quantity follows the plume: ``mo_cuascent.f90``
keys the updraft's own latent heat to ``ptu`` (``zalvs = MERGE(alv, als, ptu >
tmelt)``) and reserves the environment ``ptenh`` for environment quantities.
For the convective wet-deposition call (``cuflx_subm`` →
``mo_hammoz_wetdep.f90::wetdep_interface``), HAMMOZ's ``prep_wetdep_hydro``
sets the precipitating fraction to the updraft area
``pclceff = pmfu/(zwu·prhou)`` with ``zwu = 2`` m/s, on the ``pmfu`` that
``mo_cufluxdts.f90::cuflx`` has already filled below the cloud base
(``pmfu(jk) = pmfu(kcbot)·zzp``), and ``mo_ham_wetdep.f90::ham_wetdep``
removes ``pxtp10·pclc·(1 − exp(−Λ·Δt))`` from the ambient air with ``Λ`` from
``bc_rain`` at the grid-mean rain flux. CAM's ``wetdepa_v2`` partitions
differently: it rescales the rain rate to the precipitating area, so the area
cancels and its below-cloud term acts on the grid mean.

**Why we differ.**
- `science` (documented deviation) — the downdraft **seeds the level of free
  sinking by entraining environment air** (exact column telescoping, matching
  CAM's "environment entrainment only, no transformation in the downdraft"),
  rather than ECHAM ``cudlfs``'s 50/50 updraft/wet-bulb-environment mix. Aerosol
  resuspension by evaporating convective precip (CAM ``dcondt_prevap``) is not
  modelled — the removed flux goes straight to the surface, matching the existing
  wet-deposition treatment. When active, ``WetScavenging`` retires its
  own environment-profile convective in-cloud pathway to avoid double-counting.
- `science` (reference disagreement) — the convective carrier follows HAMMOZ's
  updraft-area footprint with ``Λ`` at the grid-mean flux; the stratiform
  carrier follows CAM's cancellation (see [aerosol removal](#aerosol-removal-below-cloud-scavenging-settling-and-the-removal-chain)).
  For the convective carrier the two references differ by the factor ``f_cu``
  (a few per cent). HAMMOZ's form is used because the updraft area is the
  footprint of the shaft and the in-plume sink already removes what is inside
  it — the double counting of #781. The ambient tracer scavenged is the
  grid-mean working copy, which stands in for HAMMOZ's environment value
  ``pxtenh`` to O(``f_cu``).
- `science` (documented deviation) — HAMMOZ zeroes below-cloud scavenging in
  any layer with stratiform cloud (``paclc ≥ 1e-10`` in ``ham_wetdep``). jcm
  does not gate the convective carrier on the stratiform cover: its stratiform
  carrier carries no cover at all, and gating one carrier on the other's cover
  is the cross-carrier coupling #781 removed.
- `compute` (documented deviation) — HAMMOZ's updraft area divides by the
  **updraft** density ``zrhou = p/(rd·ptu)``; ``ConvectionData`` publishes no
  updraft temperature, so the environment density stands in and ``f_cu`` is
  low by ``(T_u − T_env)/T_env``, under 2 % in the plume core.
- Faithful otherwise — the convective carrier flux, the in-cloud/below-cloud
  separation and the updraft-temperature phase split all follow the references
  above. Detrained condensate keeps the **environment** split instead, which is
  what ECHAM ``cudtdq`` uses for it.

**Status & known limitations.**
- Under HAMMOZ's form the convective washout is ``f_cu`` times the CAM form
  for ``Λ·Δt ≪ 1``. The CAM-consistent alternative — the in-shaft intensity
  ``R/f_cu`` inside the exponential, which reduces to CAM in that limit and
  still caps the per-step removal at ``f_cu`` — is a one-line change in
  ``conv_below_cloud_rate`` should validation call for it.
- The in-plume and transport pathways read the previous step's plume profiles
  (the one-``dt`` lag above). The below-cloud pathway does not: ``WetScavenging``
  runs after ``TiedtkeConvection`` in the ECHAM chain, so its convective
  precipitation flux and updraft-area footprint are the current step's.

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
differently on step 1 of a cold start, deliberately: dust emits nothing (a
saltation threshold and a log law both calibrated to 10 m have no defensible
value on a ~33 m model-level wind, so ``DustEmissions`` zeroes exactly the
columns ``wind_10m`` flags), sea salt and DMS emit from the lowest level and
flag it, and the surface term has no step-1 case because it runs after vertical
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
again, by ``1 − snow_cover``, and zeroed where the relative soil wetness
``forcing.soilw_rel`` (soil water as a fraction of that soil's field capacity,
ECHAM's ``ws/wsmx``) exceeds 0.99.

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
  191-class spectrum is. The ≥ 10 µm remainder is discarded and nothing is
  renormalised; it is published per column as ``dust_supercoarse_flux``
  because it is large (0.11-0.75 of the total). HAM discards a remainder too,
  but at a **different edge**: ``ham_m7_dust_emissions`` puts tracer 1 into
  the insoluble accumulation mode and tracers 2-4 into the insoluble coarse
  mode, so everything below its tracer-4 edge of 15.887 µm is emitted and
  tracers 5-8 (to 1300 µm) never reach the aerosol. The 10-15.887 µm slice
  between the two conventions is what a budget comparison has to account for;
  it is measured under **The one scalar jcm calibrates** below.
- `data` (snow) — HAM multiplies by ``1 − cvs`` with ECHAM's ``physc`` snow-cover
  formula (a ``tanh`` in snow depth with orographic-σ damping, a canopy-snow
  substitute and ``cvs = 1`` on glaciers). jcm has no prognostic snow depth
  (``land.py`` still zeroes its tendency), so ``1 − snowc_am`` stands in.
  ``snowc_am`` is zero on permanent ice where ECHAM sets ``cvs = 1``, which is
  harmless because ``pot_source`` is zero there.
- `data` (soil moisture) — the ``ws/wsmx > 0.99`` cut-off is unconditional in
  every HAM preset, and jcm feeds it ``forcing.soilw_rel``: ERA5's 0-7 cm
  volumetric water content divided by the HTESSEL field capacity of that
  cell's own soil type (Balsamo et al. 2009), i.e. soil water as a fraction of
  field capacity — what ECHAM's ``ws/wsmx`` means. Normalising each texture by
  its own capacity is what makes 1 mean "saturated" on sand as well as on
  clay; one global constant would read a saturated desert sand
  (θ_cap = 0.244) as 0.70 and never fire the cut-off in the cells that emit.
  Two differences from ECHAM remain, stated rather than hidden: the layer is
  the 0-7 cm one that governs saltation rather than ECHAM's whole root-zone
  bucket, and the field is a **monthly climatology** where ECHAM's is
  prognostic, which averages away individual saturation events — over the T63
  cells that pass the vegetation gate the cut-off fires on 0.77 % of
  cell-months on the climatology against 2.19 % on individual ERA5 samples of
  the same decade. A forcing file that carries no such channel leaves the
  cut-off inert and logs that it has. ECHAM's own ``wsmx`` and an ECHAM ``ws``
  exist at T63 (``T63GR15_jan_surf.nc``, ``ic_land_soil_T63GR15_*.nc``) and
  agree with this field on magnitude over the source cells (mean 0.238 against
  0.245 in January, spatial correlation 0.46), but that ``ws`` is a single
  initial condition with no time axis, and both files exist only at T63. ERA5
  is used instead because it carries the seasonal cycle and derives on every
  published grid.
- `data` (resolution) — the HAMMOZ inputs exist only at T63. The T106 products
  are derived from them by nearest neighbour (conservative regridding cannot
  refine a grid, and the region mask is categorical), and the ``ndust = 3``
  resolution polynomial carries an explicit source warning that
  ``nduscale_reg`` must be re-tuned above T63 — which applies to jcm's T106 and
  ne30 configurations too. The
  regional ``ndust = 4`` vector is likewise set only at T63; every other
  resolution, the cubed sphere included, takes the Fortran's uniform
  ``CASE DEFAULT`` 0.86. Every shipped JAM configuration is T63, so this
  bounds what an unsupported composition would do rather than describing
  one the model ships. There are no ne30 dust products on the data mirror at
  all, so a shipped ne30 configuration runs with the dust emission term composed
  but inert — see {doc}`boundary_conditions` for what that means for the inputs
  and for the column sampling a hand-supplied file gets.

**Status & known limitations.**
- The **``U10 = 10 m/s`` texture switch is a hard step**: above it the
  preferential source keeps type 10's flux magnitude but emits type 11's clay
  spectrum, a ~3400x jump in sub-micron mass. It is ported as the step it is, so
  ``jax.grad`` sees zero through it and it will show up in any optimisation
  (#664). ``dust_test.py`` asserts the zero gradient deliberately.
- The `u*` pre-gate (`u* ≥ 21·nduscale/feff`) omits the East-Asian threshold
  multiplier `utsc`, exactly as the Fortran does, so a preset-4 Taklamakan cell
  (`r_dust_sf13 = 0.6`) is withheld until `U10 ≈ 7.6 m/s` even though its
  per-class threshold is cleared near 4.6 m/s. The gate is meant to be a cheap
  *necessary* condition and `utsc < 1` is where it stops being one — a latent
  inconsistency in HAM, reproduced rather than silently corrected.
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
  ``DustParameters.preset(3)`` selects Stier et al. (2005) instead. HAM tunes
  that vector twice — ``(0.95, 1.25)`` for nudged simulations against
  ``(1.05, 1.45)`` free-running — and jcm carries both, selected by
  ``physics.jam_dust_nudged``. ``null`` (the shipped value) means "follow the
  run": ``runners.py::_resolve_nudging_dependent_physics`` fills it from
  ``nudging.enabled``, because the nudging term is appended after physics is
  composed and the dust term cannot otherwise see it. An explicit ``true`` or
  ``false`` wins.
- **The one scalar jcm calibrates.** ``NDUSCALE_JCM_T63_SCALE`` multiplies the
  whole ``ndust = 4`` T63 vector, and is exposed per run as
  ``physics.jam_dust_nduscale_scale``. It is one number rather than eight
  because HAM's eight regional parameters cannot be identified against a
  single global budget: the regional *ratios* stay HAM's and only the level
  moves. It applies at T63, which is every resolution the model ships JAM
  at; a composition built at another resolution keeps HAM's untuned
  ``0.86``, since its source fields are interpolated from T63 anyway.

  The **target** is the parent model's own budget, converted to this port's
  size window. ECHAM6.3-HAM2.3 emits 1221 Tg/yr present-day and 923
  pre-industrial at T63 (Krätschmer et al. 2022, Clim. Past 18, 67, §3.1 and
  Table 2). That number is the mass that reaches the aerosol — the paper's
  emissions go "either into the insoluble accumulation mode (mmr 0.37 µm) or
  the insoluble coarse mode (mmr 1.75 µm)" and "emissions into the
  super-coarse mode are neglected" — which in the code is
  ``ham_m7_dust_emissions`` summing BGC-dust tracer 1 and tracers 2-4, i.e.
  every class below **15.887 µm**. (HAM's own ``flux_a10`` budget diagnostic
  selects the same four tracers: it keeps those whose ``dpk`` is under 10, and
  ``dpk`` is the geometric-mean *radius* of the tracer's bin, so its "< 10 µm"
  label is the one place HAM's published figure is loosely named.) The
  conversion to this port's D < 10 µm window is therefore a single slice: over
  the full T63 year, re-running the emission with the coarse window widened to
  15.887 µm puts **47.4 %** of HAM's window in 10-15.887 µm (49.7 % over
  N Africa, 48.1 % Middle East, 37.2 % Asia, 35.7 % N America). The
  HAM-equivalent D < 10 µm target is thus **1221 × 0.526 = 642 Tg/yr**
  present-day, and 485 pre-industrial.

  Two further points of reference, neither of them a like-for-like target.
  AeroCom phase I gives a median of 1123 Tg/yr across 15 models with a spread
  of roughly 500-4400 (Huneeus et al. 2011, ACP 11, 7781), but its members use
  different upper size cut-offs, so it measures inter-model spread rather than
  a value to hit. Kok et al. (2021, ACP 21, 8169, Table 1, "All source
  regions") put the AeroCom ensemble at 1.7 (1.2-3.1) × 10³ Tg/yr and their
  own observationally-constrained inverse model at **4.7 (3.4-9.1) × 10³
  Tg/yr** for dust with *geometric* diameter ≤ 20 µm. Their window is twice
  this one's and their central value is seven times HAM's converted budget,
  which is their point: models, this one included, carry far less coarse dust
  than the observations support. It is a statement about the size range being
  compared, not a bound this port can be scored against.

  The release gate ``DUST_EMISSION_TG_PER_YR`` is **400-1300 Tg/yr**, roughly
  a factor of 1.6 below and 2 above that 642. It spans the parent model's own
  pre-industrial value (485) and a T63 year 29 % above its present-day point
  (the 829 Tg/yr measured here), and is far wider than the 6 % run-to-run
  spread, so it cannot function as a tuning target — which is also why it is
  exempt from the regression tier. It is the check that dust has neither
  vanished (HAM's untuned threshold gives 5.7 Tg/yr here) nor run away (the
  same parent model's last-glacial-maximum run emits 5159 Tg/yr).

  The value is **0.5**, calibrated on 30-day T63L47 April members started from
  an ERA5 state and driven by the model's own instantaneous 10 m winds:

  | ``nduscale_scale`` | 1.00 | 0.65 | **0.50** | 0.45 |
  |---|---|---|---|---|
  | April D < 10 µm, Tg/yr | 5.7 | 294.6 | **1158.4** | 1838.6 |
  | discarded ≥ 10 µm | 77.1 % | 75.2 % | 72.6 % | 71.1 % |

  A factor 2 in the threshold is a factor ~300 in emission here, because
  saltation samples the far tail of the wind distribution and jcm's tail is
  thin. That steepness is the reason the scalar is fitted to a run rather than
  inherited, and the reason it is one scalar and not eight.

  A full ``echam-jam-t63-l47`` year at 0.5 confirms it: **829 Tg/yr** of
  D < 10 µm dust, with a further 73.1 % of the emitted spectrum discarded
  above 10 µm, a dust burden of 6.0 mg/m² and a dust lifetime of 1.36 days.
  That is the number the release band is scored against.

  The same scheme driven offline with ERA5 6-hourly 10 m winds regridded to
  T63 — every other input the model's own — gives 125 Tg/yr at 1.00, 421 at
  0.80, 1024 at 0.65 and 2540 at 0.50 annually, with April/annual running 1.05
  to 1.33. Two things follow from the pair of curves. HAM's published
  threshold is ~5x short of the 642 Tg/yr that is the parent model's own
  budget in this size window **even on a perfect wind field**, so the retune
  is required by the threshold and not only by a host-model wind bias. And jcm needs a lower multiplier than ERA5's
  winds would, because its wind tail is thinner: over the same T63 April
  source cells the two agree on the mean (3.98 m/s against 4.00) and diverge
  in the tail, 0.69 % of cell-samples above 7.62 m/s against ERA5's 4.41 %,
  and 0.003 % against 0.332 % above 10.52 m/s (the saltation onsets at
  ``nduscale`` 1.05 and 1.45). Both tables are instantaneous samples over the
  cells that pass the vegetation gate; a time-mean wind cannot resolve an
  exceedance frequency at all, which is why the calibration run saves
  ``vertical_diffusion.wind_10m`` through ``run.snapshot_variables`` rather
  than as a chunk mean.

- HAM's nudged/free-running split of ``nduscale_reg`` earns its keep in jcm
  too, which is why both vectors are carried. Holding the threshold fixed and
  changing only the nudging, a 30-day member relaxed toward ERA5 emits
  549 Tg/yr against 1158 free-running — while two free members differing only
  in start date give 1158 and 1085, a 6 % spread. The nudging signal is eight
  times that spread, and it acts through the wind the emission reads: the
  nudged member's 10 m wind over the source cells is weaker throughout
  (mean 3.60 m/s against 3.97, 18.1 % of samples above 4.93 m/s against
  27.5 %), even though ``nudging/era5.yaml`` relaxes winds only and excludes
  the two lowest levels. HAM's nudged vector is 0.905x its free-running one,
  i.e. a lower threshold and more emission — the same sign as this deficit,
  and of comparable size on the curve above.

**Code pointers.**
- ``jcm/physics/aerosol/jam/emissions/dust.py`` — ``DustEmissions``,
  ``DustParameters``, ``threshold_friction_velocity``,
  ``soil_size_distributions``, ``emission_weight_matrix``, ``SOIL_TABLE``,
  ``MIXTURE_ROWS``, ``DUST_SUPERCOARSE_KEY``.
- ``jcm/forcing.py`` — ``read_dust_source``, ``read_dust_preferential``,
  ``read_dust_soil_types``, ``read_dust_regions``, ``read_dust_roughness``,
  and the ``soilw_rel`` channel.
- ``jcm/forcing_assembly.py`` — ``_attach_dust``.
- ``jcm/data/mirror/dust.py`` — ``build_dust_product``;
  ``jcm/data/mirror/bundles.py`` — ``translate_land``, ``HTESSEL_THETA_CAP``.
- ``tools/release_validation/aerosol_stats.py`` —
  ``DUST_EMISSION_TG_PER_YR``, the release band on the annual budget.

**Validation evidence.** ``dust_test.py`` pins ``u*t`` at eight diameters
against MB95 (1091.08 cm/s at 0.2 µm to 66.30 at 1262 µm, minimum 20.4502 at
76.04 µm), the emission onset at ``U10 = 6.2377 m/s``, the single-class flux
chain (soil type 2, ``u* = 40 cm/s``, ``D = 20 µm`` → 4.4758e-10 g cm⁻² s⁻¹),
the matrix form of the sandblasting redistribution against a direct
transcription of the Fortran loops for all twelve mixture rows, the East-Asia
overlap guards, the region vector, the snow and saturation cut-offs, the
emitted ``D_eff`` (0.5698 / 1.9125 µm for a medium soil, against MAM4's own
0.310 / 5.64 µm) and the gradients through α and ``nduscale_reg``.

### Aerosol removal: below-cloud scavenging, settling, and the removal chain

**What we do.** Below-cloud impaction is CAM's Slinn coefficient, not a
size-power law: ``Λ = sol_factb · Λ₁(D_wet) · R`` with ``R`` the precipitation
flux (kg m⁻² s⁻¹ ≡ mm s⁻¹) and ``Λ₁`` in 1/mm the collection-efficiency integral
``E = min(E_brown + E_intercept + E_impact, 1)`` over a raindrop spectrum and the
mode's lognormal, evaluated separately against number and volume weights so the
two moments carry their own coefficient. ``Λ₁`` is tabulated per mode over the wet/dry
diameter growth ratio and read back by differentiable log-linear
interpolation; the knob-independent part of the kernel is built once in
float64 and the knob-dependent completion is evaluated in the traced step,
so the two collection knobs below stay differentiable. ``sol_factb`` is a differentiable
parameter at CAM's namelist default 0.1 for interstitial aerosol and structurally
zero for the cloud-borne phase, which is in-droplet by definition. Two further
differentiable knobs sit on the collection integral itself — ``mu_water_air``
(Slinn's water/air viscosity ratio, 60) and ``impact_scale`` (a multiplier on
the inertial-impaction efficiency, 1) — because impaction is the least
constrained part of the scheme; both default to CAM as written. The stratiform
carrier is not cloud-weighted, because the swept precipitating volume cancels
against the in-precip-area rain rate and the stratiform in-cloud pathway acts on
the cloud-borne tracers rather than the interstitial ones; the convective
carrier acts in HAMMOZ's updraft-area footprint (see [convective tracer
transport](#convective-tracer-transport--in-plume-scavenging)).

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
rather than a separately-derived flux. The physics interface applies no
positivity cap to aerosol or gas tendencies: those carry conservative
redistributions and paired transfers, and a per-cell cap would clip one side of a
conserved pair and create column mass. Bounding removal where it is produced is
what makes a cap unnecessary.

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
Tracer vertical diffusion and convective transport still read the step-start
state in parallel, so their summed redistribution can leave a donor cell
negative. Nothing in physics removes that: the negative persists through the
dynamics step and is cleaned on the way back INTO physics by the dycore-side
``filters.MassConservingPositivity`` — a column-mass-conserving hole-filler, on
by default for JAM runs (``diffusion.tracer_positivity: auto``) and wired into
the dinosaur backend only. Under pySES, which floors water vapour alone, and
with the filter switched off, the negative simply persists and stays visible to
the mass-budget gauge. The filter is a guard at the boundary, not
positivity-preserving tracer transport. Ice
sedimenting to the surface as snow carries no aerosol removal, matching CAM,
which has no ice-phase aerosol scavenging.

**Code pointers.**
- ``jcm/physics/aerosol/jam/wetdep/impaction.py`` — ``impaction_scavenging_rates``
  (the Slinn integral), ``build_impaction_table``, ``bcscavcoef`` (the lookup).
- ``jcm/physics/aerosol/jam/wetdep/wetdep_term.py`` — ``below_cloud_rate``,
  ``conv_precip_cover``, ``conv_below_cloud_rate``, ``WetDepParameters``
  (``sol_factb``, ``conv_updraft_velocity``).
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
- ``jcm/physics_interface.py`` — ``verify_tendencies``.

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
