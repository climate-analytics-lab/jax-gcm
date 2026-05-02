# ECHAM Physics Implementation Roadmap

This document outlines the path from the current JAX ECHAM physics implementation to a fully-functioning replica of the original Fortran code in `../icon_plumeworld/src/atm_phy_echam/`.

**Last Updated:** 2025-12-09

---

## Executive Summary

| Metric | Current | Target |
|--------|---------|--------|
| Lines of code | ~22K | ~27K |
| Feature coverage | 65-75% | 90% |
| Radiation | Simplified 5-band | External scheme (out of scope) |
| Aerosol schemes | 0 | 1 (Simple Plumes) |
| Chemistry | Placeholder | Prescribed climatology fields |
| Cloud microphysics | Single-moment | Two-moment (with ACI) |

**Estimated Total Effort:** 4-6 months

### Scope Decisions
- **Radiation:** Will use external implementation - not covered in this roadmap
- **Chemistry:** Prescribed climatology fields only (no prognostic chemistry)
- **Aerosol:** Simple Plumes scheme only (SPLUMES)
- **Clouds:** Two-moment microphysics with aerosol-cloud interactions (ACI)

---

## Phase 1: Bug Fixes & Stabilization (1-2 weeks)

**Goal:** Fix critical issues blocking production use.

### 1.1 Hybrid Coordinate NaNs (HIGH)
- **Issue:** Model produces NaNs with `use_hybrid_coords=True`
- **Location:** Pressure level calculations in hybrid coordinates
- **Action:** Debug pressure computation, add numerical guards
- **Files:** `echam_physics.py`, `vertical_diffusion/`

### 1.2 Surface Fraction Setup (HIGH)
- **Issue:** Ice fraction always zero, no lake handling
- **Location:** `echam_physics.py:1032-1035`
- **Action:**
  ```python
  # Compute ice fraction from sea ice forcing
  ice_fraction = forcing.sice_am * (1.0 - land_fraction)
  water_fraction = (1.0 - land_fraction) - ice_fraction
  ```
- **Files:** `echam_physics.py`, `surface/`

### 1.3 Exchange Coefficient Coupling (HIGH)
- **Issue:** Hardcoded dummy values instead of turbulence-derived
- **Location:** `echam_physics.py:1058-1060`
- **Action:** Extract from vertical diffusion output
- **Files:** `echam_physics.py`, `vertical_diffusion/turbulence_coefficients.py`

---

## Phase 2: Convection Improvements (2-4 weeks)

**Goal:** Match Fortran convection scheme accuracy.

### 2.1 Fix Convection Timestep (MEDIUM)
- **Issue:** Hardcoded `dt_conv=3600s` instead of model timestep
- **Fortran:** `mo_cumastr.f90` uses `pdtime` for mass flux limits
- **Action:** Pass model timestep through, remove `dt_conv` parameter
- **Impact:** Prevents unrealistic mass flux magnitudes

### 2.2 Updraft Properties (MEDIUM)
- **Issue:** Simplified Gaussian detrainment vs Fortran's tan() profile
- **Fortran:** `mo_cuascent.f90:684-802` (cuentr subroutine)
- **Action:**
  1. Implement organized detrainment profile
  2. Derive updraft properties from flux conservation
  3. Validate against Fortran output
- **Files:** `convection/updraft.py`, `convection/tiedtke_nordeng.py`

### 2.3 Downdraft LFS Criteria (MEDIUM)
- **Issue:** Wrong mass flux threshold for level of free sinking
- **Current:** `min_flux = config.cmfcmin * updraft_mf[kbase]`
- **Fortran:** `zmftop = -cmfdeps*pmfub(jl)` with cmfdeps ≈ 0.33
- **Files:** `convection/downdraft.py:136-139`

### 2.4 Cloud Work Function (LOW)
- **Issue:** Simplified CAPE approach vs full work function integration
- **Fortran:** `mo_cuinitialize.f90`
- **Files:** `convection/cape.py`

---

## Phase 3: Radiation Interface (OUT OF SCOPE)

**Status:** External implementation will be used.

The current simplified 5-band radiation scheme will be replaced with an external radiation implementation. This roadmap does not cover radiation scheme development.

**Integration requirements:**
- Define clean interface for external radiation scheme
- Ensure inputs/outputs match `PhysicsState`/`PhysicsTendency` format
- Pass cloud properties and aerosol optical depth to external scheme

---

## Phase 4: Surface & Land Model (1-2 months)

**Goal:** Complete surface physics with tile-aware calculations.

### 4.1 Tile-Aware Turbulence (MEDIUM)
- **Issue:** Single exchange coefficients for all surface types
- **Fortran:** Loops over surface types with different roughness
- **Action:** Modify turbulence scheme to output per-tile coefficients
- **Files:** `vertical_diffusion/turbulence_coefficients.py`, `surface/`

### 4.2 Wind-Dependent Roughness (LOW)
- **Issue:** Static roughness lengths over water
- **Fortran:** Charnock relation for wave-dependent roughness
- **Action:** Implement `z0 = alpha * u_star^2 / g`
- **Files:** `surface/roughness.py`

### 4.3 Land Surface Model (NEW)
- **Current:** Basic heat capacity and albedo only
- **Target:** Soil hydrology, infiltration, runoff
- **Fortran:** Coupling to JSBACH land model
- **Minimum viable:** 2-layer soil temperature + bucket hydrology
- **Files to create:**
  - `surface/land_model.py`
  - `surface/soil_hydrology.py`

### 4.4 Lake Model (NEW)
- **Current:** Not implemented
- **Fortran:** FLake model integration
- **Action:** Implement simplified lake temperature model
- **Files to create:** `surface/lake_model.py`

---

## Phase 5: Chemistry & Aerosol (3-4 weeks)

**Goal:** Implement prescribed chemistry fields and Simple Plumes aerosol scheme.

### 5.1 Prescribed Chemistry Climatology (MEDIUM)
- **Current:** Fixed O3/CH4/CO2 concentrations
- **Target:** Monthly/zonal climatology fields from files
- **Approach:** Read prescribed fields, no prognostic chemistry
- **Features:**
  - Ozone climatology (latitude × height × month)
  - Greenhouse gas concentrations (time-varying)
  - Optional: stratospheric water vapor climatology
- **Files to create:**
  - `chemistry/prescribed_fields.py` - Climatology reader
  - `data/chemistry/` - Climatology data files (netCDF)
- **Fortran reference:** `mo_bc_ozone.f90`, `mo_bc_greenhouse_gases.f90`

### 5.2 Simple Plumes Aerosol Scheme (HIGH)
- **Current:** Framework only
- **Fortran:** `mo_bc_aeropt_splumes.f90` (727 lines)
- **Features:**
  - Anthropogenic aerosol plumes (9 plume regions)
  - Time-varying emissions (1850-2100)
  - Aerosol optical depth by species (sulfate, BC, OC)
  - Wavelength-dependent optical properties
- **Action:** Port SPLUMES scheme with lookup tables
- **Files to create:**
  - `aerosol/simple_plumes.py` - Main scheme
  - `data/aerosol/MACv2-SP/` - Plume parameter files
- **Effort:** 2-3 weeks

### 5.3 Aerosol-Radiation Coupling
- **Action:** Pass aerosol optical depth to external radiation scheme
- **Interface:**
  ```python
  def compute_aerosol_optical_properties(
      date: DateData,
      latitude: jnp.ndarray,
      longitude: jnp.ndarray,
      pressure: jnp.ndarray,
  ) -> AerosolOpticalProperties:
      # Returns AOD, SSA, asymmetry parameter per band
  ```

### 5.4 Aerosol-Cloud Coupling
- **Action:** Connect Simple Plumes aerosol to cloud activation (Phase 6)
- **Interface:** Pass aerosol concentrations to activation scheme

---

## Phase 6: Two-Moment Cloud Microphysics & ACI (1-2 months)

**Goal:** Implement two-moment cloud scheme with aerosol-cloud interactions.

### 6.1 Overview
- **Current:** Single-moment scheme with prescribed CDNC
- **Target:** Two-moment scheme with prognostic cloud droplet and ice crystal number
- **Fortran Reference:**
  - `mo_2mom_mcrph_driver.f90` (1151 lines)
  - `mo_2mom_mcrph_processes.f90` (4987 lines) - Seifert-Beheng scheme

### 6.2 Prognostic Variables (NEW)
Add new tracers to `PhysicsState`:
```python
tracers = {
    'qc': ...,   # Cloud water mass (existing)
    'qi': ...,   # Cloud ice mass (existing)
    'qnc': ...,  # Cloud droplet number concentration (NEW)
    'qni': ...,  # Ice crystal number concentration (NEW)
}
```

### 6.3 Aerosol Activation Scheme (HIGH)
- **Purpose:** Convert aerosol to cloud droplets at cloud base
- **Approach:** Abdul-Razzak & Ghan (2000) or similar parameterization
- **Inputs:** Aerosol concentration (from Simple Plumes), updraft velocity, temperature
- **Outputs:** Activated droplet number concentration
- **Files to create:**
  - `clouds/activation.py` - Aerosol activation parameterization
- **Effort:** 2-3 weeks

### 6.4 Two-Moment Warm Microphysics (HIGH)
- **Processes:**
  - Autoconversion (cloud → rain) with CDNC dependence
  - Accretion (cloud + rain → rain)
  - Self-collection of cloud droplets
  - Evaporation of cloud water
- **Key physics:** Autoconversion rate ~ 1/CDNC (Twomey effect)
- **Fortran Reference:** `mo_2mom_mcrph_processes.f90` (rain_selfcollection, cloud_selfcollection, autoconversion)
- **Files to create:**
  - `clouds/two_moment_warm.py` - Warm rain processes
- **Effort:** 2-3 weeks

### 6.5 Two-Moment Ice Microphysics (MEDIUM)
- **Processes:**
  - Ice nucleation (heterogeneous)
  - Depositional growth
  - Aggregation
  - Riming
  - Bergeron-Findeisen process
- **Fortran Reference:** `mo_2mom_mcrph_processes.f90` (ice_nucleation_homhet, ice_selfcollection)
- **Files to create:**
  - `clouds/two_moment_ice.py` - Ice phase processes
  - `clouds/ice_nucleation.py` - IN parameterization
- **Effort:** 2-3 weeks

### 6.6 Cloud-Radiation Coupling
- **Current:** Cloud optical properties use fixed effective radius
- **Target:** Effective radius from CDNC and LWC
- **Formula:** `r_eff = (3 * LWC / (4 * pi * rho_w * CDNC))^(1/3)`
- **Impact:** First indirect effect (Twomey) on radiation
- **Files:** Update `clouds/cloud_optics.py`

### 6.7 Validation Targets
- CDNC range: 20-500 cm⁻³ (marine-continental)
- Effective radius: 5-15 μm (polluted-clean)
- Autoconversion sensitivity to CDNC
- Compare cloud optical depth vs Fortran

### 6.8 Implementation Order
1. Add prognostic number tracers (qnc, qni)
2. Implement activation scheme
3. Implement CDNC-dependent autoconversion
4. Update cloud optics for variable r_eff
5. Add ice nucleation
6. Full two-moment ice processes

---

## Phase 7: Gravity Wave Drag — DONE (PR #350/#351)

### 7.1 Hines (1997) non-orographic GWD — COMPLETE
- **Fortran source:** `mo_gw_hines.f90` (atm_phy_echam, 2326 lines)
- **JAX port:** `jcm/physics/gravity_waves/hines/hines.py`
- **Status:** Bit-exact against Fortran reference on 8 column scenarios.
  Targets the production control flow (8 azimuths, slope=1, lheatcal=true,
  no front/precip/lat-dependent sources, no orographic-wave coupling).
- **Wired as:** ``EchamHines`` term in default ``echam_physics()`` factory.

### 7.2 Lott & Miller (1997) sub-grid orographic drag — COMPLETE
- **Fortran source:** `mo_ssodrag.f90` (atm_phy_echam, 1564 lines)
- **JAX port:** `jcm/physics/gravity_waves/sso/lott_miller.py`
- **Status:** Bit-exact against Fortran reference on 4 column scenarios.
  Includes orodrag (wave drag) + orosetup + gwstress + gwprofil (saturation).
  Mountain-lift branch (``orolift``, gklift=0 in production) NOT ported.
- **Wired as:** ``EchamSSO`` term in default ``echam_physics()`` factory.
- **Known gap:** SSO descriptor data (orostd, orosig, etc.) not yet
  plumbed through ``TerrainData``; current wiring uses placeholders
  derived from terrain.orog and terrain.fmask. Ocean is auto-disabled
  by the activation gate.

---

## Phase 7: Boundary Conditions & Forcing (1-2 weeks)

**Goal:** Complete forcing data infrastructure.

### 7.1 Greenhouse Gas & Ozone Climatology
- **Covered in Phase 5** (prescribed chemistry fields)

### 7.2 Solar Irradiance (LOW)
- **Current:** Fixed solar constant via jax-solar
- **Action:** Optional - add solar cycle variability if needed
- **Status:** Low priority, current implementation sufficient

### 7.3 Mixed-Layer Ocean (OPTIONAL)
- **Current:** Fixed SST from forcing
- **Fortran:** `mo_ml_ocean.f90` (269 lines)
- **Action:** Implement slab ocean for coupled runs (if needed)
- **Files to create:** `surface/mixed_layer_ocean.py`
- **Status:** Optional for prescribed-SST experiments

---

## Phase 8: Diagnostics & Validation (Ongoing)

### 8.1 Missing Diagnostics
Add diagnostic outputs matching Fortran:
- Spectral fluxes at all levels
- Radiative heating by band
- Cloud diagnostics (droplet/ice concentration)
- Convective mass flux profiles
- Boundary layer depth and Richardson number
- Energy conservation metrics

### 8.2 Validation Framework
- **Unit tests:** Compare individual routines vs Fortran output
- **Column tests:** Single-column validation against Fortran SCM
- **Global tests:** Multi-year climate statistics comparison

### 8.3 Reference Data
Generate Fortran reference data for:
- Tropical convection profiles
- Mid-latitude radiation balance
- Polar boundary layer
- Stratospheric chemistry

---

## Implementation Priority Matrix

| Phase | Effort | Impact | Priority |
|-------|--------|--------|----------|
| 1. Bug Fixes | 1-2 weeks | Critical | **IMMEDIATE** |
| 2. Convection | 2-4 weeks | High | **HIGH** |
| 3. Radiation | - | - | **OUT OF SCOPE** |
| 4. Surface | 1-2 months | Medium | **MEDIUM** |
| 5. Chemistry/Aerosol | 3-4 weeks | High | **HIGH** |
| 6. Two-Moment Clouds & ACI | 1-2 months | Very High | **HIGH** |
| 7. Gravity Waves | 1 month | Low | **LOW** |
| 8. Forcing | 1-2 weeks | Low | **LOW** |
| 9. Diagnostics | Ongoing | Medium | **ONGOING** |

**Revised Total Estimate:** 4-6 months (excluding external radiation integration)

---

## Fortran Files Reference

### Core Physics (atm_phy_echam/)

| Module | Lines | JAX Status | Priority |
|--------|-------|------------|----------|
| `mo_cumastr.f90` | 771 | Substantially ported | - |
| `mo_cuascent.f90` | 804 | Partially ported | MEDIUM |
| `mo_cudescent.f90` | 727 | Partially ported | MEDIUM |
| `mo_cufluxdts.f90` | 489 | Partially ported | LOW |
| `mo_cuinitialize.f90` | 615 | Partially ported | LOW |
| `mo_vdiff_solver.f90` | 939 | Substantially ported | - |
| `mo_turbulence_diag.f90` | 984 | Partially ported | MEDIUM |
| `mo_surface.f90` | 938 | Partially ported | MEDIUM |
| `mo_cloud.f90` | 1215 | Partially ported | MEDIUM |
| `mo_cover.f90` | 897 | Partially ported | LOW |
| `mo_gw_hines.f90` | 2326 | Framework only | LOW |
| `mo_ssodrag.f90` | 1564 | Framework only | LOW |
| ~~`mo_methox.f90`~~ | ~~5600~~ | ~~Not ported~~ | ~~OUT OF SCOPE~~ |

### Two-Moment Microphysics (atm_phy_schemes/)
| Module | Lines | JAX Status | Priority |
|--------|-------|------------|----------|
| `mo_2mom_mcrph_driver.f90` | 1151 | Not ported | **HIGH** |
| `mo_2mom_mcrph_processes.f90` | 4987 | Not ported | **HIGH** |

### Radiation (atm_phy_psrad/) - OUT OF SCOPE
External radiation scheme will be used.

### Boundary Conditions
| Module | Lines | JAX Status | Priority |
|--------|-------|------------|----------|
| ~~`mo_bc_aeropt_kinne.f90`~~ | ~~591~~ | ~~Not ported~~ | ~~OUT OF SCOPE~~ |
| ~~`mo_bc_aeropt_stenchikov.f90`~~ | ~~625~~ | ~~Not ported~~ | ~~OUT OF SCOPE~~ |
| `mo_bc_aeropt_splumes.f90` | 727 | Not ported | **HIGH** |
| `mo_bc_ozone.f90` | ~400 | Not ported | MEDIUM |
| `mo_bc_greenhouse_gases.f90` | ~300 | Not ported | MEDIUM |

---

## Success Criteria

### V1 Release (Current Target)
- [x] Basic radiation (simplified)
- [x] Tiedtke-Nordeng convection (core)
- [x] Vertical diffusion (fixed)
- [x] Basic surface physics
- [ ] Hybrid coordinate stability
- [ ] Surface fraction fix

### V2 Release (Production Ready)
- [ ] External radiation scheme integrated
- [ ] Complete convection scheme (updraft/downdraft fixes)
- [ ] Simple Plumes aerosol scheme
- [ ] Prescribed ozone/GHG climatology
- [ ] Tile-aware surface physics
- [ ] Two-moment cloud microphysics
- [ ] Aerosol activation (ACI)

### V3 Release (Full Feature)
- [ ] Full two-moment ice microphysics
- [ ] Hines GWD improvements
- [ ] Lake model (optional)
- [ ] Mixed-layer ocean (optional)
- [ ] Full diagnostics
- [ ] Validation against Fortran

---

## Related Documents

- `REMAINING_ISSUES.md` - Detailed bug tracker
- `JAX_CONVERSION_PATTERNS.md` - Fortran-to-JAX patterns
- `JAX_gotchas.md` - Common JAX pitfalls
- `UNIT_CONVERSIONS.md` - Physics interface units
- `../icon_plumeworld/` - Fortran reference code
