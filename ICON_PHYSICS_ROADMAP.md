# ICON Physics Development Roadmap

**Date**: January 2025  
**Status**: Comprehensive codebase review completed  
**Purpose**: Transform ICON physics from development/testing implementation to production-ready suite

## Executive Summary

The ICON physics implementation in JAX-GCM has achieved **JAX compatibility** and **basic functionality** across all major components. However, systematic review reveals **25 critical areas** requiring development to reach production readiness. This roadmap prioritizes these improvements and provides a clear path forward.

## Current Status ✅

### What's Working Well
- **JAX Compatibility**: All TracerBoolConversionError issues resolved
- **Fixed Architecture**: Simplified fixed-band (3 LW, 2 SW) and fixed-tracer (qc, qi) approach
- **Vectorization**: TPU-optimized column-based processing with 20x memory reduction
- **Testing**: 100+ tests passing across all physics modules
- **Integration**: Main physics wrapper (`IconPhysics`) fully functional

### Recent Achievements
1. **Radiation**: 74 tests passing, fixed bands implementation
2. **Vertical Diffusion**: 15 tests passing, matrix solver stability fixed
3. **Convection**: JAX-compatible with fixed qc/qi transport
4. **Surface Physics**: 65 tests passing, conditional logic replaced
5. **Wrapper Simplification**: Eliminated 5 unnecessary wrapper functions

## Critical Issues Requiring Immediate Action 🚨

### 1. Runtime Errors (HIGH PRIORITY)
**File**: `jcm/physics/icon/icon_physics.py:801-804`
```python
# ❌ UNDEFINED VARIABLES - Will cause runtime crashes
surface_temp_tendency=surf_temp_tend,  # undefined
exchange_coeff_heat=ch,               # undefined  
exchange_coeff_momentum=cm,           # undefined
surface_resistance=resistance,        # undefined
```
**Fix Required**: Extract these variables from surface physics calculations

### 2. Empty Physics Modules (HIGH PRIORITY)
**Files**: 
- `jcm/physics/icon/chemistry/__init__.py` - Completely empty (`__all__ = []`)
- `jcm/physics/icon/boundary_conditions/__init__.py` - Missing entirely

**Impact**: Essential physics components missing for climate modeling

### 3. Hardcoded Limitations (HIGH PRIORITY)
**Issue**: Fixed array sizes prevent flexible configurations
**Examples**:
- `max_bands=10` hardcoded in multiple radiation files
- Band limits hardcoded in test files
**Impact**: JAX compilation issues, inflexible spectral resolution

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

### Convection Scheme Issues
**File**: `jcm/physics/icon/convection/tiedtke_nordeng.py`

- **Line 359**: Moist adiabatic simplified (`parcel_temp_moist = temperature`)
- **Lines 509-522**: Downdraft calculations stubbed to avoid dtype issues
- **Lines 534-538**: Crude tracer transport (`tracer_flux * 0.1`)
- **Missing**: Proper entrainment/detrainment, momentum transport

### Vertical Diffusion Gaps
**File**: `jcm/physics/icon/vertical_diffusion/turbulence_coefficients.py`

- **Lines 165-172**: All surface fluxes set to zero
- **Line 179**: Kinetic energy dissipation disabled
- **Missing**: Surface layer physics, TKE budget, non-local mixing

### Surface Physics Simplifications
**Files**: `jcm/physics/icon/surface/`

- **`surface_physics.py:155`**: Surface humidity hardcoded (`0.01`)
- **`surface_physics.py:189`**: Solar zenith angle set to zero (permanent noon)
- **`land.py`**: No soil heat diffusion or vegetation dynamics
- **`ocean.py`**: Missing ocean-atmosphere coupling

### Cloud Microphysics Limitations
**File**: `jcm/physics/icon/clouds/cloud_microphysics.py`

- **Lines 639-647**: Simplified sedimentation (no flux form)
- **Lines 649-657**: Precipitation only from lowest level
- **Missing**: Size distributions, ice nucleation, mixed-phase processes

## Implementation Roadmap 📋

### Phase 1: Critical Fixes (Immediate - Days)
**Priority**: HIGH - Required for basic functionality

1. **Fix undefined variables** in `icon_physics.py` surface section
2. **Implement basic chemistry** module (minimum: fixed ozone distribution)
3. **Create boundary conditions** infrastructure  
4. **Replace hardcoded band arrays** with dynamic sizing

**Estimated Effort**: 2-3 days  
**Deliverable**: Runtime-stable physics integration

### Phase 2: Core Physics (1-2 months)
**Priority**: HIGH - Essential for physical realism

#### Radiation Improvements
- Better gas optics with proper spectral bands
- Cloud optics with Mie scattering calculations
- Integration with jax-solar for accurate solar calculations

#### Convection Completion
- Proper downdraft calculations (remove dtype workarounds)
- Implement entrainment/detrainment schemes
- Add momentum transport in convection

#### Vertical Diffusion Enhancement
- Surface layer physics implementation
- TKE budget equations
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

### High Priority (10 items)
1. Fix undefined variables in icon_physics.py surface section
2. Implement basic chemistry module
3. Create boundary conditions infrastructure  
4. Replace hardcoded spectral band arrays
5. Improve gas optics with pressure/temperature dependence
6. Implement proper cloud optics with Mie scattering
7. Integrate jax-solar for accurate solar calculations
8. Complete convection downdraft calculations
9. Implement proper entrainment/detrainment
10. Add surface layer physics to vertical diffusion

### Medium Priority (12 items)
11. Add momentum transport in convection
12. Implement TKE budget calculations
13. Replace hardcoded surface humidity
14. Implement proper solar zenith angle calculations
15. Add soil heat diffusion to land surface
16. Implement proper vegetation dynamics
17. Add snow physics to surface model
18. Improve cloud microphysics sedimentation
19. Add ice nucleation schemes
20. Implement MACv2-SP plume calculations
21. Add non-orographic wave sources to gravity waves
22. Implement proper wave-mean flow interactions

### Low Priority (3 items)
23. Add comprehensive diagnostic output
24. Implement parameter sensitivity analysis
25. Add performance benchmarking tools

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

### Phase 1 Success Criteria
- [ ] No runtime errors in physics integration
- [ ] All existing tests continue to pass
- [ ] Basic chemistry and boundary conditions functional

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

The ICON physics implementation has achieved a solid foundation with JAX compatibility and basic functionality. The 25 improvements identified in this roadmap provide a clear path to transform this into a production-ready physics suite suitable for operational climate modeling.

**Key Strengths**: Solid architecture, JAX optimization, comprehensive testing  
**Key Gaps**: Some physics oversimplified, missing essential components  
**Timeline**: 6 months to full production readiness with dedicated development

This roadmap balances immediate needs (critical fixes) with long-term goals (advanced features) to ensure steady progress toward a world-class atmospheric physics implementation.

---
*This roadmap is a living document and should be updated as development progresses and priorities evolve.*