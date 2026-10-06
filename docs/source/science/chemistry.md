# Chemistry

Two chemistry layers exist: a lightweight ECHAM-physics chemistry (ozone and
methane) that runs in every ECHAM configuration, and the JAM sulfur chain that
runs only when the prognostic aerosol is composed. The aqueous scheme lives with
the aerosol code (``jcm/physics/aerosol/jam/chemistry/``), not under
``jcm/physics/chemistry/``.

## SimpleChemistry — ozone and methane

**What we do.** Radiation's ozone and methane are both supplied by
``EchamBoundaryConditions``, which seeds ``chemistry.ozone_vmr`` (the forcing
climatology, or an **analytic fixed distribution** — a stratospheric-max profile
parameterised by scale height, max VMR and tropopause height — when no file is
given) and
``chemistry.methane_vmr`` every step.
Both public fields are in **ppmv**, matching ``ForcingData.co2_vmr``,
``ForcingData.ch4_vmr``, ``ForcingData.n2o_vmr`` and
``OzoneClimatology.o3_ppmv``. Their production/loss diagnostics are therefore
in ppmv s⁻¹. Radiation converts each gas exactly once at its boundary to the
dimensionless mol/mol values required by gas optics; use
``ChemistryData.ozone_mole_fraction()`` and
``ChemistryData.methane_mole_fraction()`` when another consumer needs that
representation.
``jcm/physics/chemistry/simple_chemistry.py::SimpleChemistry`` runs alongside it
as a **diagnostic** relaxation: it recomputes ozone production/loss and the
methane sink from that seeded state, but what it returns for the VMRs does not
survive the next boundary-conditions seed. Methane is
**prescribed, not prognostic**: ``EchamBoundaryConditions`` overwrites the
chemistry carry's CH₄ with ``forcing.ch4_vmr`` every step, so the linear
OH-scaled decay ``SimpleChemistry`` computes survives only as the
``methane_loss`` diagnostic (a sink-rate readout) — radiation sees the
prescribed VMR and there is no evolving CH₄ budget — the same pattern as
ozone. CO₂ is deliberately *not* here — it is a prescribed forcing (``forcing.co2_vmr``) read directly by
radiation.

**What ECHAM/CAM does.** A stand-in for prescribed CMIP ozone/GHG chemistry; the
analytic ozone profile is a jcm interim, not a port.

**Why we differ.**
- `science` — the analytic ozone profile is an interim proxy with a tropospheric
  ozone column several times too large, biasing clear-sky OLR low. Production
  configurations replace it with the packaged CMIP6 climatology (see
  {doc}`boundary_conditions`), so ``SimpleChemistry``'s ozone is a fallback.

**Status & known limitations.** Analytic ozone is a documented low-fidelity
fallback. A prognostic CH₄ budget would need the per-step prescribed reseed
lifted; today ``methane_loss`` is diagnostic-only and nothing downstream
consumes it.

**Code pointers.** ``jcm/physics/chemistry/simple_chemistry.py`` —
``ChemistryParameters`` (ozone + methane, explicit no-CO₂ note),
``fixed_ozone_distribution``.

**Validation evidence.** ``jcm/physics/chemistry/simple_chemistry_test.py``.

## JAM sulfur chemistry — oxidants, gas-phase, aqueous

**What we do.**
- **Prescribed oxidants** (``jcm/physics/aerosol/jam/chemistry/oxidants.py::PrescribedOxidants``)
  supply OH, NO₃, O₃, H₂O₂ in molec cm⁻³. The preferred source is a file
  climatology (``forcing.oxidant_vmr``, e.g. a HAMMOZ/MACC monthly field, converted
  VMR→molec cm⁻³ with instantaneous T, p); interim analytic proxies (OH ∝ cos
  zenith·[O₃], etc.) are the fallback.
- **Gas-phase sulfur** (``sulfur_gas.py::SulfurGasChemistry``) is a port of
  ECHAM-HAM ``mo_ham_chemistry.f90::ham_gas_chemistry`` (prescribed-oxidant), rate
  constants verbatim (DMS+OH abstraction/addition, DMS+NO₃, SO₂+OH+M Troe), sulfur
  conserved atom-for-atom, integrated with stable exponential decay.
- **Aqueous sulfur** (``aqueous.py::AqueousSulfur``) is a full port of ECHAM-HAM
  ``mo_ham_chemistry.f90::ham_wet_chemistry`` (Feichter et al. 1996), with **no
  kinetics simplification** — H₂O₂ and O₃ pathways over sub-steps, cloud-droplet
  pH solved each sub-step from the sulfate/SO₂ charge balance, Henry/rate constants
  verbatim from HAM. **This module header is a gold-standard provenance example.**

**What ECHAM/CAM does.** HAM ``ham_gas_chemistry`` / ``ham_wet_chemistry``
(Feichter 1996). CAM's aqueous analogue is ``mo_setsox`` (a longer-iteration
electroneutrality pH including NH₃/HNO₃/CO₂).

**Why we differ.**
- `science` (adaptations) — gas-phase SO₂+OH and the DMS branch route to
  **gas-phase H₂SO₄** (not directly to particulate SO₄) so the MAM4 core does the
  gas→particle step. Aqueous product sulfate goes to the **cloud-borne**
  accumulation/coarse split by cloud-borne number fraction; H₂O₂ is a prescribed
  oxidant depleted within the sub-stepping and reset each step (matching HAM's
  offline-oxidant discard).
- `science` (explicit CAM rejection) — the aqueous header records that CAM
  ``mo_setsox`` is **not** simpler than the full HAM port here, so the lightweight
  ``scheme="simple"`` option (H₂O₂-limited stoichiometric, no Henry/pH/O₃) is a
  **reduced scheme, not a CAM port**.

**Status & known limitations.** Oxidants default to interim cos-zenith proxies
until a climatology is supplied; H₂O₂ is not persisted across steps (a coupled
prognostic budget is out of scope). SOAG production in sulfur chemistry
defaults to zero; its source is supplied separately through prescribed emissions.

**Code pointers.** ``jcm/physics/aerosol/jam/chemistry/`` — ``oxidants.py``
(``PrescribedOxidants``), ``sulfur_gas.py`` (``SulfurGasChemistry``),
``aqueous.py`` (``AqueousSulfur``; the ``mo_setsox`` rejection is recorded in the
header).

**Validation evidence.** ``chemistry/oxidants_test.py``,
``chemistry/sulfur_gas_test.py``, ``chemistry/aqueous_test.py``.

## Prescribed CAM6 secondary organic aerosol

**What we do.** The present-day `t63-echam-jam-soa` configuration supplies
anthropogenic, biogenic and biomass-burning SOAG from the official CAM6
2014 historical inventory (matching the present-day emissions bundle). The fixed VOC yields and 1.5 source multiplier are
already included in the inventories. Carbon-equivalent fluxes are converted
with the CAM destination-tracer molecular weight, 12.011 g/mol, conservatively
remapped, and injected through the existing surface-emissions term.
The MAM4 core provides reversible exchange with ΔHvap = 156 kJ/mol and
reference vapour pressure 10⁻¹⁰ atm at 298 K. Aerosol SOA has the existing
wet and dry removal; SOAG has no gas deposition or photolysis in this scheme.
SOAG condenses into fine modes and transient primary-carbon coating, which
ageing transfers to accumulation. The core's per-call uptake mask excludes
coarse-mode SOA, matching CAM6 rather than the MOM box-model default.

**Reference and motivation.** `science` — original CAM6 single-bin SOA,
[Jo et al. (2023), section 2.2](https://gmd.copernicus.org/articles/16/3893/2023/),
CAM `mo_srf_emissions::srf_emissions` and
`modal_aero_amicphys::mam_soaexch_1subarea`. The prescribed source is a published
alternative to explicitly evolving VOC chemistry. It implements this original
CAM6 formulation, not the newer CAM6.3 SOAE oxidation-delay scheme or VBS.

**Status and validation.** The inventory is explicitly period-specific; PI
and transient simulations need matching emissions. Preparation tests verify
carbon-equivalent conversion, conservative remapping and calendar preservation.
The adapter tests close the exchange on the core's common molecular basis
(SOAG and SOA conversion factors differ by 12.011/12), including the cloud-borne
store. A five-day response from zero SOA does not establish equilibrium.
See [SOA and dust design](../design/jam_soa_dust.md).
