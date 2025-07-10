# ICON Radiation Scheme Implementation Plan

## Overview
Implement a radiation scheme for ICON physics in JAX, inspired by ICON's PSRad and ECHAM radiation schemes.

## Components to Implement

### Phase 1: Basic Infrastructure
1. **Module Structure** ✓
   - Create radiation module directory
   - Base classes for radiation calculations
   - Parameter dataclasses

2. **Solar Radiation**
   - Integrate jax-solar for TOA incident radiation
   - Solar zenith angle calculations
   - Diurnal cycle handling

3. **Gas Optics**
   - Absorption coefficients for key gases (H2O, CO2, O3)
   - Simplified band models
   - Pressure/temperature interpolation

### Phase 2: Cloud Optics
1. **Cloud Optical Properties**
   - Cloud water/ice optical thickness
   - Single scattering albedo
   - Asymmetry parameter
   
2. **Cloud Overlap**
   - Maximum-random overlap assumption
   - Cloud fraction handling

### Phase 3: Radiative Transfer
1. **Longwave Solver**
   - Two-stream approximation
   - Planck function calculations
   - Surface emission

2. **Shortwave Solver**
   - Two-stream approximation
   - Multiple scattering
   - Surface albedo

### Phase 4: Integration
1. **Heating Rate Calculations**
   - Flux divergence
   - Temperature tendencies

2. **Integration with IconPhysics**
   - Add radiation calls
   - Handle time stepping

## Key Design Decisions

### 1. Spectral Bands
Start with simplified band model:
- **Shortwave**: 2 bands (visible, near-IR)
- **Longwave**: 2-3 bands (window, H2O, CO2)

### 2. JAX Patterns
- Use `vmap` for column-wise calculations
- `@jax.jit` for performance
- Avoid Python loops - use JAX array operations

### 3. Simplifications for V1
- Fixed gas concentrations (CO2, O3)
- Simple surface albedo
- No aerosols initially
- Simplified cloud optics

## Implementation Order

1. **Solar calculations** (using jax-solar)
2. **Gas absorption** (simple parameterization)
3. **Planck functions** (longwave emission)
4. **Two-stream solver** (core radiative transfer)
5. **Cloud optics** (from microphysics)
6. **Full radiation scheme**

## Testing Strategy

### Unit Tests
- Solar angle calculations
- Gas optical properties
- Planck functions
- Two-stream solver
- Cloud optics

### Integration Tests
- Clear-sky radiation
- Cloudy-sky radiation
- Heating rate profiles
- Energy conservation

## Key Functions to Implement

```python
# Solar radiation
def solar_zenith_angle(time, lon, lat)
def top_of_atmosphere_flux(time, lon, lat)

# Gas optics
def gas_optical_depth(pressure, temperature, mixing_ratio)
def planck_function(temperature, wavenumber)

# Cloud optics
def cloud_optical_properties(cloud_water, cloud_ice, effective_radius)

# Radiative transfer
def two_stream_solver(optical_depth, single_scatter_albedo, asymmetry_factor)
def longwave_fluxes(temperature, optical_depth, surface_emissivity)
def shortwave_fluxes(solar_flux, optical_depth, surface_albedo)

# Main interface
def radiation_tendencies(state, cloud_state, surface_properties, time)
```

## References
- ICON PSRad documentation
- ECHAM6 radiation scheme (Iacono et al. 2008)
- Two-stream approximation (Meador & Weaver 1980)
- Cloud overlap (Räisänen et al. 2004)

## Notes for Continuation
When we resume:
1. Start with solar calculations and basic infrastructure
2. Implement gas optics with simple band model
3. Add two-stream solver
4. Integrate cloud optics from microphysics
5. Test each component thoroughly