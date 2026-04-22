# ICON Full Radiation Scheme Implementation Roadmap

## Executive Summary

This document outlines the roadmap for implementing the complete ICON PSRad radiation scheme in JAX-GCM. Our current implementation provides a solid foundation with all core physics components working and comprehensive test coverage. The full implementation would require 2-3 months of focused development, with the primary challenge being the integration of RRTMG gas optics tables.

## Current Implementation Status

### What We Have ✅
- **Working radiation scheme** with realistic heating rates (-25 to +15 K/day)
- **JAX-compatible implementation** (jit, vmap, grad support)
- **Comprehensive test suite** (48/48 tests passing)
- **Modular architecture** with clean separation of concerns
- **Basic spectral resolution**: 2 SW bands, 3 LW bands
- **Core physics components**:
  - Solar geometry and top-of-atmosphere flux
  - Gas optics (H2O, CO2, O3 absorption)
  - Cloud optics (liquid and ice)
  - Planck functions for thermal emission
  - Two-stream radiative transfer solver
  - Flux-to-heating-rate conversion

### Current Limitations
- Simplified spectral resolution (5 bands vs 30 bands)
- Basic gas absorption parameterizations vs RRTMG lookup tables
- Simple cloud overlap vs Monte Carlo sampling
- No aerosol effects
- Fixed trace gas concentrations

## Full ICON PSRad Comparison

### Spectral Resolution
- **Current**: 2 SW bands (vis, NIR) + 3 LW bands (window, CO2, H2O)
- **ICON**: 14 SW bands + 16 LW bands
- **Impact**: Higher accuracy for gas absorption and surface albedo

### Gas Optics
- **Current**: Parameterized absorption (temperature/pressure dependent)
- **ICON**: RRTMG k-distribution tables with 280+ g-points
- **Impact**: Much more accurate absorption coefficients

### Cloud Treatment
- **Current**: Simple cloud overlap assumption
- **ICON**: Monte Carlo sampling with maximum-random overlap
- **Impact**: Better representation of cloud radiative effects

### Additional Features
- **Current**: Basic surface properties
- **ICON**: Aerosol effects, advanced cloud optics, trace gases

## Implementation Roadmap

### Phase 1: Core RRTMG Integration (4 weeks)
**Objective**: Replace parameterized gas optics with RRTMG lookup tables

**Tasks**:
1. **Data Conversion** (1 week)
   - Convert RRTMG NetCDF tables to JAX arrays
   - Implement efficient data structures for k-distributions
   - Handle pressure/temperature grids

2. **Gas Optics Implementation** (2 weeks)
   - Port RRTMG interpolation algorithms to JAX
   - Implement k-distribution mixing
   - Add trace gas handling (N2O, CH4, CFCs)

3. **Integration & Testing** (1 week)
   - Replace current gas optics with RRTMG
   - Validate against reference calculations
   - Ensure energy conservation

**Deliverables**:
- `rrtmg_gas_optics.py` module
- Updated test suite
- Performance benchmarks

**Challenges**:
- Large data files (~50MB) need efficient JAX representation
- Complex interpolation schemes must be JAX-compatible
- Memory management for large lookup tables

### Phase 2: Full Spectral Resolution (2 weeks)
**Objective**: Expand to complete 14 SW + 16 LW band structure

**Tasks**:
1. **Band Structure Update** (1 week)
   - Expand band definitions and wavelength ranges
   - Update surface albedo handling for all bands
   - Modify solar flux calculations

2. **Component Updates** (1 week)
   - Update cloud optics for all bands
   - Extend Planck function calculations
   - Test energy conservation across all bands

**Deliverables**:
- Updated band definitions
- Extended optical property calculations
- Comprehensive spectral validation

**Challenges**:
- 6x increase in memory usage
- Need band-specific surface properties
- Computational cost scaling

### Phase 3: Monte Carlo Cloud Sampling (3 weeks)
**Objective**: Implement realistic cloud overlap schemes

**Tasks**:
1. **Random Number Generation** (1 week)
   - Implement JAX-compatible random number generation
   - Design reproducible sampling schemes
   - Handle vectorization over columns

2. **Cloud Sampling Algorithm** (1 week)
   - Port `sample_cld_state` functionality
   - Implement maximum-random overlap
   - Add decorrelation length handling

3. **Integration & Validation** (1 week)
   - Integrate with existing radiation solver
   - Test against reference cloud calculations
   - Validate statistical properties

**Deliverables**:
- `cloud_sampling.py` module
- Random number generation utilities
- Cloud overlap validation tests

**Challenges**:
- Random number generation in JAX
- Vectorization of sampling algorithms
- Statistical validation requirements

### Phase 4: Advanced Features (2-3 weeks)
**Objective**: Add aerosols and enhanced cloud optics

**Tasks**:
1. **Aerosol Integration** (1 week)
   - Add aerosol optical depth inputs
   - Implement wavelength-dependent properties
   - Test aerosol-radiation interactions

2. **Enhanced Cloud Optics** (1 week)
   - Advanced ice crystal parameterizations
   - Precipitation particle optics
   - Mixed-phase cloud handling

3. **Diagnostics & Validation** (1 week)
   - Add spectral flux diagnostics
   - Implement heating rate decomposition
   - Comprehensive validation suite

**Deliverables**:
- Aerosol optics module
- Enhanced cloud parameterizations
- Advanced diagnostic capabilities

## Technical Considerations

### Memory Management
- **Current**: ~10MB per atmospheric column
- **Full ICON**: ~60MB per atmospheric column
- **Solutions**: Checkpointing, band-by-band processing, efficient data structures

### Performance Optimization
- **Target**: <10% performance degradation vs current scheme
- **Strategies**: 
  - JIT compilation of lookup table access
  - Vectorization over spectral dimensions
  - Memory-efficient data layouts

### Validation Strategy
- **Reference Data**: Use ICON PSRad calculations as ground truth
- **Test Cases**: 
  - Clear-sky profiles (various atmospheres)
  - Cloudy scenarios (different cloud types)
  - Extreme conditions (tropical, arctic)
- **Metrics**: 
  - Flux accuracy (<1 W/m²)
  - Heating rate accuracy (<0.1 K/day)
  - Energy conservation (<0.01%)

## Data Requirements

### RRTMG Tables
- **Size**: ~50MB of k-distribution data
- **Format**: NetCDF converted to JAX arrays
- **Content**: 
  - Absorption coefficients vs pressure/temperature
  - Planck function derivatives
  - Solar irradiance data

### Validation Data
- **ICON Reference**: Output from standard ICON radiation calls
- **Benchmark Cases**: Standard atmospheric profiles
- **Intercomparison**: Results from other radiation schemes

## Risk Assessment

### High Risk
- **RRTMG Data Conversion**: Complex data structures, potential accuracy loss
- **Memory Scaling**: 6x memory increase may require architectural changes

### Medium Risk
- **Performance**: Monte Carlo sampling adds computational overhead
- **Validation**: Need extensive comparison with reference calculations

### Low Risk
- **JAX Compatibility**: Existing architecture handles transformations well
- **Modular Design**: Current structure supports incremental development

## Success Metrics

### Technical Milestones
- [ ] Phase 1: <1% difference in clear-sky heating rates vs RRTMG
- [ ] Phase 2: Energy conservation <0.01% across all bands
- [ ] Phase 3: Cloud radiative forcing within 2% of reference
- [ ] Phase 4: Complete feature parity with ICON PSRad

### Performance Targets
- [ ] Memory usage <100MB per atmospheric column
- [ ] Runtime <2x current implementation
- [ ] GPU acceleration maintains efficiency

## Conclusion

The current JAX-based radiation implementation provides an excellent foundation for a complete ICON PSRad scheme. The modular architecture, comprehensive testing, and JAX compatibility make this a feasible 2-3 month development effort. The primary technical challenge lies in efficiently representing and accessing the large RRTMG lookup tables in JAX, but this is a well-understood problem with established solutions.

The phased approach allows for incremental validation and ensures that each component works correctly before proceeding to the next phase. The resulting implementation would provide research-quality radiation calculations with full differentiability and GPU acceleration capabilities.

---

**Document Version**: 1.0  
**Date**: 2025-01-10  
**Author**: JAX-GCM Development Team  
**Next Review**: After Phase 1 completion