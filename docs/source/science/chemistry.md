# Chemistry

Two chemistry layers exist: a lightweight ECHAM-physics chemistry (ozone and
methane) that runs in every ECHAM configuration, and the JAM sulfur chain that
runs only when the prognostic aerosol is composed. The aqueous scheme lives with
the aerosol code (``jcm/physics/aerosol/jam/chemistry/``), not under
``jcm/physics/chemistry/``.

## SimpleChemistry — ozone and methane

**What we do.** A lightweight ECHAM-physics chemistry
(``jcm/physics/chemistry/simple_chemistry.py``) providing an **analytic fixed
ozone distribution** (a stratospheric-max profile parameterised by scale height,
max VMR, tropopause height and a stratosphere coefficient) plus a basic methane
oxidation (relaxation to climatology / linear OH-scaled decay). CO₂ is
deliberately *not* here — it is a prescribed forcing (``forcing.co2_vmr``) read
directly by radiation.

**What ECHAM/CAM does.** A stand-in for prescribed CMIP ozone/GHG chemistry; the
analytic ozone profile is a jcm interim, not a port.

**Why we differ.**
- `science` — the analytic ozone profile is an interim proxy with a tropospheric
  ozone column several times too large, biasing clear-sky OLR low. Production
  configurations replace it with the packaged CMIP6 climatology (see
  {doc}`boundary_conditions`), so ``SimpleChemistry``'s ozone is a fallback.

**Status & known limitations.** Analytic ozone is a documented low-fidelity
fallback; methane oxidation is a simple relaxation, not a mechanism.

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
prognostic budget is out of scope); the SOAG source is interim (#496).

**Code pointers.** ``jcm/physics/aerosol/jam/chemistry/`` — ``oxidants.py``
(``PrescribedOxidants``), ``sulfur_gas.py`` (``SulfurGasChemistry``),
``aqueous.py`` (``AqueousSulfur``; the ``mo_setsox`` rejection is recorded in the
header).

**Validation evidence.** ``chemistry/oxidants_test.py``,
``chemistry/sulfur_gas_test.py``, ``chemistry/aqueous_test.py``.
