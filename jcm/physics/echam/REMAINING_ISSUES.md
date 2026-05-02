# ECHAM Physics Implementation - Remaining Issues

This document catalogs all remaining issues, FIXMEs, and TODOs in the JAX ECHAM physics implementation, compared against the Fortran reference in `../icon_plumeworld/src/atm_phy_echam/`.

**Last Updated:** 2025-12-08

---

## Table of Contents

1. [Critical Issues](#critical-issues)
2. [Vertical Diffusion](#vertical-diffusion)
3. [Radiation](#radiation)
4. [Convection](#convection)
5. [Surface Physics](#surface-physics)
6. [Chemistry & Aerosol](#chemistry--aerosol)
7. [Forcing & Boundary Conditions](#forcing--boundary-conditions)
8. [Testing Recommendations](#testing-recommendations)

---

## Critical Issues

### 1. Hybrid Coordinate System NaNs
**Status:** OPEN
**Location:** Model integration with `use_hybrid_coords=True`
**Symptom:** `test_echam_model_hybrid` fails with NaNs after a few timesteps
**Root Cause:** Unknown - may be related to pressure level calculations in hybrid coordinates
**Priority:** HIGH

### 2. ~~Vertical Diffusion Temperature Clamping (Workaround)~~ **FIXED**
**Status:** RESOLVED
**Location:** `echam_physics.py`
**Issue:** Temperature tendencies from vertical diffusion were excessively large (~0.05 K/s = 180 K/hour)
**Root Cause:** Multiple issues in vertical diffusion scheme:
1. tpfac parameters were incorrect (tpfac1=1.0 instead of 1.5, tpfac2/tpfac3=0 instead of 0.667/0.333)
2. Missing prefactor (rho/dz) in matrix coefficient calculation
3. Tendency calculation didn't apply the semi-implicit tpfac3 correction

**Fix Applied:**
- Updated `vertical_diffusion_types.py` with correct tpfac defaults
- Added proper prefactor calculation in `matrix_solver.py:setup_matrix_system`
- Created new `setup_momentum_matrix_with_prefactor` function
- Updated `setup_rhs_vectors` to apply tpfac2 scaling
- Updated `compute_tendencies_from_solution` to apply tpfac3 correction
- Removed temperature clamping workaround from `echam_physics.py`

---

## Vertical Diffusion

### ~~Issue VD-1: Incorrect Default tpfac Parameters~~ **FIXED**
**Status:** RESOLVED
**Location:** `vertical_diffusion/vertical_diffusion_types.py:37`

**Previous:**
```python
def default(cls, tpfac1=1.0, tpfac2=0.0, tpfac3=0.0, ...)
```

**Now Matches Fortran (mo_echam_vdiff_params.f90:62-66):**
```python
def default(cls, tpfac1=1.5, tpfac2=0.667, tpfac3=0.333, ...)
```

### Issue VD-2: TKE Source Terms Not Applied
**Location:** `vertical_diffusion/tke_budget.py:218`
**FIXME:** `# Will be handled by matrix solver FIXME: check that is is being handled`

**Issue:** TKE source terms (shear production, buoyancy production, dissipation) are computed but never added to the solution.

**Fortran Reference:** TKE budget includes explicit source terms in mo_vdiff_solver.f90

**Priority:** LOW (TKE used mainly for diagnostics)

### Issue VD-3: Unused vertical_diffusion_scheme Import
**Location:** `echam_physics.py:28`
**FIXME:** `# FIXME: would be good to use this`

```python
from jcm.physics.echam.vertical_diffusion import vertical_diffusion_scheme
```

**Issue:** The imported `vertical_diffusion_scheme` is not used; instead a custom `apply_vdiff_to_column` is defined inline.

**Recommendation:** Refactor to use the proper scheme function.
**Priority:** LOW (code quality)

### ~~Issue VD-4: Height Level Offset~~ **FIXED**
**Status:** RESOLVED
**Location:** `echam_physics.py:523-541`

**Previous:**
```python
height_half = jnp.concatenate((
    height_levels[:1] + 1000.0, # FIXME: validate choice of offset
    ...
```

**Now Fixed:** Height interfaces are computed by extrapolating using consistent layer thicknesses:
```python
# Top interface: extrapolate using the same spacing as the top layer
top_layer_thickness = height_levels[0] - height_half_internal[0]
height_top = height_levels[0] + top_layer_thickness
```

**Priority:** ~~LOW~~ N/A

---

## Radiation

### ~~Issue RAD-0: Incorrect Pressure Interfaces~~ **FIXED**
**Status:** RESOLVED
**Location:** `radiation/radiation_scheme.py:168-176`

**Previous:**
```python
# Interface pressures computed locally with arbitrary scaling
pressure_interfaces = pressure_interfaces.at[0].set(pressure_levels[0] * 0.1)  # Much lower for TOA
pressure_interfaces = pressure_interfaces.at[-1].set(pressure_levels[-1] * 1.1)  # Slight increase for surface
```

**Issue:** Radiation scheme was computing its own pressure interfaces with arbitrary factors (0.1x at TOA, 1.1x at surface) instead of using the model's half-level pressures. This caused incorrect heating rate calculations because the dp used in `dT/dt = -(g/cp) * dF/dp` was wrong.

**Fix Applied:** The `radiation_scheme` and `prepare_radiation_state` functions now accept `pressure_interfaces` as a parameter, which is passed from the model's `pressure_half` computed from sigma/hybrid coordinates.

**Impact:** Should improve radiation heating rate profiles, especially in top layers where the arbitrary 0.1x factor created very small dp values leading to extreme heating rates.

### Issue RAD-1: Unused Band Variables
**Location:** `radiation/radiation_scheme.py:245-247`
**FIXME:** `# FIXME - this isn't used`

```python
n_sw_bands = parameters.n_sw_bands
n_lw_bands = parameters.n_lw_bands
```

**Issue:** Variables extracted but immediately overridden with hardcoded defaults.

**Recommendation:** Remove unused lines.
**Priority:** LOW (code cleanup)

### Issue RAD-2: Solar Flux Shape Comment
**Location:** `radiation/radiation_scheme.py:294`
**FIXME:** `#FIXME: I think this is causing an issue...`

**Status:** NOT AN ISSUE - Functions return scalars as expected.

**Recommendation:** Remove FIXME comment.
**Priority:** LOW

### Issue RAD-3: Unused tau Parameter in Two-Stream
**Location:** `radiation/two_stream.py:23`
**FIXME:** `# FIXME: remove this if unused`

```python
def two_stream_coefficients(
    tau: jnp.ndarray, # FIXME: remove this if unused
    ...
)
```

**Issue:** `tau` parameter defined but never used in function body.

**Recommendation:** Remove parameter from signature and update call sites.
**Priority:** LOW

---

## Convection

### ~~Issue CONV-1: Timestep Parameter Validation~~ **FIXED**
**Status:** RESOLVED
**Location:** `convection/tiedtke_nordeng.py:74`, `parameters.py`, `echam_physics.py`, `model.py`

**Previous:**
```python
def default(cls, dt_conv=3600.0, ...)  # FIXME: validate dt_conv
```

**Issue:** Default `dt_conv=3600s` (1 hour) didn't match typical model timesteps (720-1800s).

**Fix Applied:**
1. Added `Parameters.with_timestep(dt_seconds)` method that updates all physics timesteps (dt_conv, dt_rad, dt_sedi)
2. Added `dt_physics` parameter to `EchamPhysics.__init__` for explicit timestep setting
3. Model class now automatically calls `parameters.with_timestep(dt_si)` when using EchamPhysics

**Usage:** Timesteps are now automatically synchronized. Users can also explicitly set:
```python
physics = EchamPhysics(dt_physics=1800.0)  # 30 minutes in seconds
# Or the Model class handles it automatically based on time_step parameter
```

**Priority:** ~~MEDIUM~~ N/A

### Issue CONV-2: Updraft States Not Investigated
**Location:** `echam_physics.py:719`
**FIXME:** `# FIXME: investigate updraft states (conv_states_all.tu and .mfu)`

**Issues:**
1. Missing organized detrainment calculation (uses simplified Gaussian instead of Fortran's tan() profile)
2. Updraft properties not derived from flux conservation equations
3. May cause incorrect vertical structure of convective heating

**Fortran Reference:** `mo_cuascent.f90:684-802` (cuentr subroutine)

**Priority:** MEDIUM

### Issue CONV-3: Downdraft LFS Criteria
**Location:** `convection/downdraft.py:136-139`

**Issue:** Uses wrong mass flux threshold calculation.

**Current:**
```python
min_flux = config.cmfcmin * updraft_mf[kbase]
```

**Fortran:**
```fortran
zmftop = -cmfdeps*pmfub(jl)  ! cmfdeps ≈ 0.33
```

**Priority:** MEDIUM

---

## Surface Physics

### Issue SFC-1: Surface Fraction Setup
**Location:** `echam_physics.py:1032-1035`
**FIXME:** `# FIXME: verify/improve this setup`

```python
surface_fractions = surface_fractions.at[:, 0].set(1.0 - land_fraction)
surface_fractions = surface_fractions.at[:, 2].set(land_fraction)
```

**Issues:**
1. Ice fraction (index 1) always zero
2. No lake handling
3. Fractions may not sum to 1.0

**Fortran Reference:** `mo_surface.f90:137` - uses pre-computed fractions for all surface types

**Recommendation:**
```python
# Add ice fraction based on latitude/season
ice_fraction = compute_ice_fraction(latitude, sea_ice_concentration)
water_fraction = (1.0 - land_fraction) - ice_fraction
surface_fractions = surface_fractions.at[:, 0].set(water_fraction)
surface_fractions = surface_fractions.at[:, 1].set(ice_fraction)
surface_fractions = surface_fractions.at[:, 2].set(land_fraction)
```

**Priority:** HIGH

### Issue SFC-2: Dummy Exchange Coefficients
**Location:** `echam_physics.py:1058-1060`
**FIXME:** `# FIXME: replace with real values`

```python
dummy_exchange = jnp.ones((ncols, nsfc_type)) * 0.001
```

**Issue:** Exchange coefficients hardcoded instead of computed from vertical diffusion.

**Impact:** Inconsistent surface-atmosphere coupling; incorrect surface fluxes.

**Fortran Reference:** `mo_turbulence_diag.f90:794-796` computes tile-specific coefficients.

**Recommendation:** Extract exchange coefficients from vertical diffusion diagnostics:
```python
exchange_coeff_heat = vdiff_diagnostics.exchange_coeff_heat[:, -1]  # Surface level
exchange_coeff_momentum = vdiff_diagnostics.exchange_coeff_momentum[:, -1]
```

**Priority:** HIGH

### Issue SFC-3: Missing Tile-Aware Turbulence
**Location:** `vertical_diffusion/turbulence_coefficients.py`

**Issue:** Vertical diffusion computes single exchange coefficients, not per-tile.

**Fortran:** Loops over surface types with different roughness lengths.

**Impact:** Incorrect fluxes over heterogeneous surfaces.

**Priority:** MEDIUM

### Issue SFC-4: Static Roughness Lengths
**Location:** `surface/surface_physics.py:80-85`

**Issues:**
1. Water roughness should depend on wind speed (Charnock relation)
2. Heat roughness z0h ≠ momentum roughness z0m over water
3. Land roughness should come from land model

**Priority:** LOW

---

## Chemistry & Aerosol

### Issue CHEM-1: Constant Gas Concentrations
**Location:** `forcing.py:65-68`

```python
co2_concentration = 420.0  # ppmv
ch4_concentration = 1900.0  # ppbv
o3_concentration = 300.0  # ppbv
```

**Issue:** Greenhouse gas concentrations are hardcoded constants.

**Recommendation:** Make configurable via Parameters or ForcingData.

**Priority:** LOW

---

## Forcing & Boundary Conditions

### Issue FORC-1: Unused Sea Ice Variables
**Location:** `forcing.py:72`
**TODO:** `#TODO: use these somewhere`

```python
sea_ice_fraction = forcing.sice_am[..., 0] if forcing.sice_am.ndim == 3 else forcing.sice_am
sea_ice_thickness = jnp.where(sea_ice_fraction > 0.1, 1.0, 0.0)
```

**Issue:** Sea ice fraction and thickness computed but not used.

**Recommendation:** Connect to surface fraction initialization (Issue SFC-1).

**Priority:** MEDIUM

### Issue FORC-2: Hardcoded Surface Properties
**Location:** `forcing.py:110`
**TODO:** `# TODO: Pull these out into parameters`

**Issue:** Surface albedo values for different surface types are hardcoded.

**Recommendation:** Add to Parameters class or ForcingData.

**Priority:** LOW

---

## Testing Recommendations

### Unit Tests Needed

1. **Vertical Diffusion Matrix Solver**
   - Test with known analytical solutions
   - Verify temperature tendency magnitudes
   - Test stability with different tpfac values

2. **Surface Fraction Logic**
   - Test fractions sum to 1.0
   - Test polar vs tropical ice distribution
   - Test land/water/ice combinations

3. **Exchange Coefficient Coupling**
   - Verify magnitude matches Fortran output
   - Test stability function ranges

### Integration Tests Needed

1. **Compare JAX vs Fortran outputs for:**
   - Clear-sky OLR (should be within 10%)
   - Convective heating profiles
   - Surface fluxes over different surface types

2. **Stability tests:**
   - Run with different timesteps (180s, 720s, 1800s)
   - Check for numerical blow-up
   - Verify energy conservation

3. **Hybrid coordinate validation:**
   - Compare with sigma coordinate results
   - Check pressure level calculations
   - Verify vertical interpolation

---

## Summary by Priority

### RESOLVED Issues
- ~~VD-1: tpfac parameters~~ - Fixed with correct defaults (1.5, 0.667, 0.333)
- ~~Vertical diffusion temperature instability~~ - Fixed with proper prefactor and time-stepping
- ~~VD-4: Height level offset~~ - Fixed with proper extrapolation instead of arbitrary 1000m
- ~~RAD-0: Incorrect pressure interfaces~~ - Fixed by passing model's pressure_half to radiation
- ~~CONV-1: Timestep parameter~~ - Fixed with automatic timestep synchronization

### HIGH Priority (3 issues)
- Hybrid coordinate NaNs
- Surface fraction setup (SFC-1)
- Exchange coefficient coupling (SFC-2)

### MEDIUM Priority (4 issues)
- Updraft states (CONV-2)
- Downdraft LFS criteria (CONV-3)
- Sea ice integration (FORC-1)
- Tile-aware turbulence (SFC-3)

### LOW Priority (6 issues)
- TKE source terms (VD-2)
- Unused imports (VD-3)
- Radiation cleanup (RAD-1, RAD-2, RAD-3)
- Roughness lengths (SFC-4)
- Hardcoded parameters (CHEM-1, FORC-2)

---

## References

### Fortran Source Files
- `mo_vdiff_solver.f90` - Vertical diffusion matrix solver
- `mo_vdiff_downward_sweep.f90` - Matrix coefficient setup
- `mo_echam_vdiff_params.f90` - Vertical diffusion parameters
- `mo_turbulence_diag.f90` - Turbulence diagnostics and exchange coefficients
- `mo_cumastr.f90` - Convection master routine
- `mo_cuascent.f90` - Convective updraft
- `mo_cudescent.f90` - Convective downdraft
- `mo_surface.f90` - Surface physics interface
- `mo_surface_diag.f90` - Surface diagnostics

### Key Commits
- `1cf6cd2` - Fix critical T=0K bug in vertical diffusion matrix solver
- `a520cb6` - Fix shortwave radiation surface reflection bug
- `e815bc0` - Fix critical bugs in longwave radiation Planck function
- (pending) Fix vertical diffusion tpfac parameters and prefactor
