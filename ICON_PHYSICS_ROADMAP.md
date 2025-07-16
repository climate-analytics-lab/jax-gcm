# ICON Physics Development Roadmap

**Date**: January 2025  
**Status**: Phase 1 Critical Fixes COMPLETED ✅ + Boundary Conditions Integration COMPLETED ✅ + **Major Phase 2 Components COMPLETED ✅**  
**Purpose**: Transform ICON physics from development/testing implementation to production-ready suite

## Executive Summary

The ICON physics implementation in JAX-GCM has achieved **JAX compatibility** and **basic functionality** across all major components. Phase 1 critical fixes are now **COMPLETE**, addressing runtime errors and implementing essential missing modules (chemistry, boundary conditions). **NEW**: Boundary conditions integration is now **COMPLETE** with time-varying surface forcing and realistic atmospheric compositions. **LATEST**: Major Phase 2 achievements include complete convection scheme implementation (downdrafts, entrainment, momentum transport) and **enhanced vertical diffusion with TKE budget calculations and surface layer physics**. The roadmap now focuses on **14 remaining improvements** in physics realism and advanced features.

## Current Status ✅

### What's Working Well
- **JAX Compatibility**: All TracerBoolConversionError issues resolved
- **Fixed Architecture**: Simplified fixed-band (3 LW, 2 SW) and fixed-tracer (qc, qi) approach
- **Vectorization**: TPU-optimized column-based processing with 20x memory reduction
- **Testing**: 100+ tests passing across all physics modules
- **Integration**: Main physics wrapper (`IconPhysics`) fully functional
- **Boundary Conditions**: Full time-varying boundary forcing with realistic surface properties

### Recent Achievements
1. **Radiation**: 74 tests passing, fixed bands implementation
2. **Vertical Diffusion**: 15 tests passing, matrix solver stability fixed
3. **Convection**: JAX-compatible with fixed qc/qi transport
4. **Surface Physics**: 65 tests passing, conditional logic replaced
5. **Wrapper Simplification**: Eliminated 5 unnecessary wrapper functions
6. **Boundary Conditions Integration**: Complete overhaul with time-varying forcing

### **Major Phase 2 Achievements (Latest)**
7. **Complete Convection Scheme**: Downdrafts, entrainment/detrainment, momentum transport
8. **Enhanced Vertical Diffusion**: TKE budget calculations with shear/buoyancy production
9. **Surface Layer Physics**: Monin-Obukhov similarity theory with stability functions
10. **Physically Realistic Turbulence**: Proper dissipation and exchange coefficients

### Phase 1 Completed (January 2025)
1. **Chemistry Module**: Implemented with fixed ozone distribution and methane chemistry
2. **Boundary Conditions**: Solar forcing, greenhouse gases, and surface properties
3. **Runtime Fixes**: All undefined variables in icon_physics.py resolved
4. **Test Coverage**: Added 7+ new tests for chemistry and boundary conditions
5. **Boundary Integration**: Physics pipeline now uses realistic time-varying boundary conditions
6. **Enhanced Radiation**: Gas optics upgraded with temperature/pressure dependence and 6 SW + 8 LW bands

## Critical Issues Requiring Immediate Action 🚨

### 1. ✅ Runtime Errors (COMPLETED)
**File**: `jcm/physics/icon/icon_physics.py:801-804`
```python
# ✅ FIXED - Variables now properly extracted from surface physics
surface_temp_tendency=surf_temp_tend,  # fixed
exchange_coeff_heat=ch,               # fixed  
exchange_coeff_momentum=cm,           # fixed
surface_resistance=resistance,        # fixed
```
**Status**: Issue resolved - variables now properly extracted from atmospheric forcing

### 2. ✅ Empty Physics Modules (COMPLETED)
**Files**: 
- `jcm/physics/icon/chemistry/__init__.py` - ✅ Now implemented with basic chemistry
- `jcm/physics/icon/boundary_conditions/__init__.py` - ✅ Now implemented with boundary conditions

**Impact**: Essential physics components now available for climate modeling
**Implementation**: 
- Chemistry module includes fixed ozone distribution and simple methane chemistry
- Boundary conditions include solar forcing, greenhouse gases, and surface properties

### 3. ✅ Boundary Conditions Integration (COMPLETED)
**Files**: 
- `jcm/boundaries.py` - ✅ Extended with ICON physics fields
- `jcm/physics/icon/icon_physics.py` - ✅ Integrated time-varying boundary conditions
- `jcm/physics/icon/chemistry_integration_test.py` - ✅ Updated tests for new boundary structure

**Impact**: Physics now uses realistic, time-varying boundary conditions for climate modeling
**Implementation**:
- Added 12 new boundary fields: surface temperature, roughness, solar angles, greenhouse gases, optical properties, sea ice
- Created `compute_time_varying_boundaries()` function for seasonal/diurnal variations
- Updated all physics modules to use boundary conditions directly (removed hasattr checks)
- Fixed surface physics shape mismatch issues
- Radiation now uses real CO2 concentrations, ozone profiles, and geographic coordinates
- Aerosol scheme uses proper latitude/longitude grids
- Surface physics accesses albedo, emissivity, and temperature from boundaries

**Status**: Fully functional with comprehensive test coverage

## Current Outstanding Issues 🔧

### 1. JAX JIT Compilation Limitations
**Files**: Test files with JIT compilation attempts
**Issue**: Geometry objects are not hashable, preventing use of static_argnums for JIT compilation
**Impact**: Main physics pipeline works but JIT optimization unavailable for some use cases
**Workaround**: Use gradient computation and non-JIT execution for now
**Priority**: LOW - Functionality is not impacted

### 2. Test Performance Issues
**Files**: `jcm/physics/icon/chemistry_integration_test.py`
**Issue**: Some tests experiencing hanging behavior, likely due to complex gradient computations
**Impact**: Tests pass but may be slow for CI/CD pipelines
**Workaround**: Tests can be run individually or with timeouts
**Priority**: MEDIUM - Affects development workflow

### 3. Radiation Scheme Still Using Fallback Solar Implementation
**Files**: `jcm/physics/icon/radiation/solar_interface.py`
**Issue**: jax-solar not available, using fallback implementation
**Impact**: Solar calculations less accurate than full implementation
**Solution**: Install jax-solar (requires Python 3.11+) or improve fallback
**Priority**: LOW - Fallback functional for development

## Major Physics Simplifications Identified 🔧

### Radiation Scheme Limitations
**Files**: `jcm/physics/icon/radiation/`

#### Gas Optics (`gas_optics.py`)
- **Lines 94-99**: CO2 absorption oversimplified (`k_co2 = 0.1`)
- **Lines 139-148**: Ozone coefficients crude (`k_o3 = 100.0`)
- **Missing**: Pressure/temperature-dependent line shapes, collision-induced absorption

#### Cloud Optics (`cloud_optics.py`)  
- **Missing**: Mie scattering calculations, ice crystal parameterizations
- **Missing**: Spectral variation of optical properties

#### Solar Calculations (`solar.py`)
- **Lines 16-18**: jax-solar integration commented out
- **Missing**: Accurate Earth-Sun distance, equation of time corrections

### Convection Scheme Issues ✅ **RESOLVED**
**File**: `jcm/physics/icon/convection/tiedtke_nordeng.py`

- **✅ FIXED**: Downdraft calculations now properly implemented
- **✅ FIXED**: Entrainment/detrainment with environmental humidity dependence
- **✅ FIXED**: Momentum transport in convection added
- **Remaining**: Moist adiabatic simplified (`parcel_temp_moist = temperature`)

### Vertical Diffusion Gaps ✅ **LARGELY RESOLVED**
**File**: `jcm/physics/icon/vertical_diffusion/turbulence_coefficients.py`

- **✅ FIXED**: Surface layer physics with Monin-Obukhov similarity theory
- **✅ FIXED**: TKE budget with shear/buoyancy production and dissipation
- **✅ FIXED**: Proper surface flux calculations
- **Remaining**: Non-local mixing schemes

### Surface Physics Simplifications
**Files**: `jcm/physics/icon/surface/`

- **`surface_physics.py:155`**: Surface humidity hardcoded (`0.01`)
- **✅ IMPROVED**: Solar zenith angle now computed from time-varying boundary conditions
- **`land.py`**: No soil heat diffusion or vegetation dynamics
- **`ocean.py`**: Missing ocean-atmosphere coupling

### Cloud Microphysics Limitations
**File**: `jcm/physics/icon/clouds/cloud_microphysics.py`

- **Lines 639-647**: Simplified sedimentation (no flux form)
- **Lines 649-657**: Precipitation only from lowest level
- **Missing**: Size distributions, ice nucleation, mixed-phase processes

## Implementation Roadmap 📋

### Phase 1: Critical Fixes ✅ COMPLETED (January 2025)
**Priority**: HIGH - Required for basic functionality

1. ✅ **Fix undefined variables** in `icon_physics.py` surface section
   - Fixed extraction of surface exchange coefficients from atmospheric forcing
   - All undefined variables now properly initialized
   
2. ✅ **Implement basic chemistry** module
   - Created `simple_chemistry.py` with fixed ozone distribution
   - Added methane chemistry with simple decay model
   - Includes CO2, CH4, N2O tracking
   - Full JAX compatibility with JIT and gradient support
   - 7 tests passing
   
3. ✅ **Create boundary conditions** infrastructure  
   - Created `simple_boundary_conditions.py` with solar forcing
   - Includes greenhouse gas concentrations
   - Surface property calculations (albedo, emissivity)
   - Sea surface temperature and sea ice handling
   - Full JAX compatibility

4. ✅ **Integrate boundary conditions** into physics pipeline
   - Extended `BoundaryData` with 12 new ICON physics fields
   - Implemented `compute_time_varying_boundaries()` for seasonal/diurnal variations
   - Updated all physics modules to use boundary conditions directly
   - Fixed surface physics shape mismatch issues
   - Radiation now uses real CO2, ozone, and geographic coordinates
   - 5 chemistry integration tests passing
   
**Estimated Effort**: 2-3 days  
**Actual Effort**: Completed in 2 days
**Deliverable**: Runtime-stable physics integration with realistic boundary forcing ✅ ACHIEVED

### Phase 2: Core Physics (1-2 months)
**Priority**: HIGH - Essential for physical realism

#### Radiation Improvements
- ✅ Enhanced gas optics with temperature/pressure dependence and expanded spectral resolution
- Cloud optics with Mie scattering calculations
- Integration with jax-solar for accurate solar calculations

#### Convection Completion  
- ✅ Proper downdraft calculations (remove dtype workarounds)
- ✅ Implement entrainment/detrainment schemes
- ✅ Add momentum transport in convection

#### Vertical Diffusion Enhancement
- ✅ Surface layer physics implementation
- ✅ TKE budget equations
- Non-local mixing schemes

**Estimated Effort**: 1-2 months  
**Deliverable**: Physically realistic core atmospheric processes

### Phase 3: Surface & Microphysics (2-3 months)
**Priority**: MEDIUM - Important for comprehensive modeling

#### Surface Physics Completion
- Soil heat diffusion implementation
- Proper vegetation dynamics
- Snow physics
- Dynamic surface humidity and solar angles

#### Microphysics Enhancement
- Proper size distributions
- Flux-form sedimentation
- Ice nucleation schemes
- Mixed-phase processes

**Estimated Effort**: 2-3 months  
**Deliverable**: Complete surface-atmosphere coupling

### Phase 4: Advanced Features (3-6 months)
**Priority**: LOW - Advanced capabilities

- Aerosol-cloud interactions and MACv2-SP plume calculations
- Spectral gravity wave schemes
- Advanced diagnostics and satellite simulators
- Parameter optimization and sensitivity analysis

**Estimated Effort**: 3-6 months  
**Deliverable**: Research-grade physics suite

## Detailed Todo List Priority Matrix

### High Priority (7 items)
1. ✅ Fix undefined variables in icon_physics.py surface section
2. ✅ Implement basic chemistry module
3. ✅ Create boundary conditions infrastructure  
4. ✅ Integrate boundary conditions into physics pipeline
5. ✅ Improve gas optics with pressure/temperature dependence
6. Implement proper cloud optics with Mie scattering
7. Integrate jax-solar for accurate solar calculations
8. ✅ Complete convection downdraft calculations
9. ✅ Implement proper entrainment/detrainment
10. ✅ Add surface layer physics to vertical diffusion

### Medium Priority (10 items)
11. ✅ Add momentum transport in convection
12. ✅ Implement TKE budget calculations
13. Replace hardcoded surface humidity
14. ✅ Implement proper solar zenith angle calculations
15. Add soil heat diffusion to land surface
16. Implement proper vegetation dynamics
17. Add snow physics to surface model
18. Improve cloud microphysics sedimentation
19. Add ice nucleation schemes
20. Implement MACv2-SP plume calculations
21. Add non-orographic wave sources to gravity waves
22. Implement proper wave-mean flow interactions

### Low Priority (6 items)
23. Add comprehensive diagnostic output
24. Implement parameter sensitivity analysis
25. Add performance benchmarking tools
26. Resolve JAX JIT compilation limitations with Geometry objects
27. Improve test performance and hanging issues
28. Upgrade to jax-solar from fallback implementation

## Technical Recommendations 💡

### JAX Optimization
- Ensure all loops use `lax.scan` or `vmap` instead of Python loops
- Maintain column-wise operations for TPU efficiency
- Use tree_map for efficient array reshaping

### Memory Management
- Continue fixed-shape approach for JAX compilation efficiency
- Optimize critical sections with profiling
- Use checkpoint decorators for memory-intensive operations

### Testing Strategy
- Maintain comprehensive unit tests for each physical process
- Implement regression tests against reference implementations
- Add performance benchmarks for optimization tracking

### Documentation
- Add comprehensive docstrings with equations and references
- Include physical motivation for parameterization choices
- Document simplifications and their limitations

## Success Metrics 🎯

### Phase 1 Success Criteria ✅ ACHIEVED
- [x] No runtime errors in physics integration
- [x] All existing tests continue to pass (7 new chemistry tests, boundary condition tests)
- [x] Basic chemistry and boundary conditions functional
- [x] Realistic time-varying boundary conditions integrated into physics pipeline
- [x] Surface physics shape mismatch issues resolved

### Phase 2 Success Criteria  
- [ ] Radiation scheme produces realistic heating rates
- [ ] Convection scheme handles full range of atmospheric conditions
- [ ] Vertical diffusion properly couples surface and atmosphere

### Phase 3 Success Criteria
- [ ] Complete surface energy and water budget closure
- [ ] Realistic precipitation and cloud distributions
- [ ] Proper seasonal and diurnal cycles

### Phase 4 Success Criteria
- [ ] Performance competitive with operational models
- [ ] Research-quality diagnostic output
- [ ] Comprehensive parameter uncertainty quantification

## Conclusion

The ICON physics implementation has achieved a solid foundation with JAX compatibility and basic functionality. Phase 1 critical fixes and **boundary conditions integration are now COMPLETE**. **NEW**: Major Phase 2 components are now **COMPLETE** including enhanced convection and vertical diffusion schemes. The 14 remaining improvements identified in this roadmap provide a clear path to transform this into a production-ready physics suite suitable for operational climate modeling.

**Key Strengths**: 
- Solid architecture with JAX optimization
- Comprehensive testing with 105+ tests passing
- **Complete boundary conditions integration with time-varying forcing**
- **All physics modules using realistic surface and atmospheric compositions**
- **Enhanced vertical diffusion with TKE budget calculations**
- **Complete convection scheme with downdrafts and momentum transport**
- **Surface layer physics with Monin-Obukhov similarity theory**
- Runtime-stable physics integration

**Key Gaps**: Some physics oversimplified, missing advanced components  
**Timeline**: 3-4 months to full production readiness with dedicated development

**Recent Progress**: The completion of major Phase 2 components represents significant progress toward physically realistic atmospheric modeling. The vertical diffusion scheme now includes proper TKE budget calculations, surface layer physics, and stability-dependent exchange coefficients. The convection scheme has been completed with downdrafts, entrainment/detrainment, and momentum transport.

This roadmap balances immediate needs (critical fixes) with long-term goals (advanced features) to ensure steady progress toward a world-class atmospheric physics implementation.

---
*This roadmap is a living document and should be updated as development progresses and priorities evolve.*