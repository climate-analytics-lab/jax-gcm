# Radiation Implementation Progress Notes

## Completed Components (2025-01-10)

### 1. Module Structure ✓
- Created `radiation/` module directory
- Defined type system in `radiation_types.py`:
  - `RadiationParameters`: Configuration parameters
  - `RadiationState`: Input state variables
  - `RadiationFluxes`: Output flux profiles
  - `RadiationTendencies`: Temperature tendencies
  - `OpticalProperties`: For radiative transfer

### 2. Solar Radiation (`solar.py`) ✓
- Solar geometry calculations:
  - Solar declination
  - Hour angle
  - Cosine solar zenith angle
- Top-of-atmosphere flux
- Earth-Sun distance correction
- Daylight fraction calculation
- Note: Production version should use `jax-solar` package

### 3. Gas Optics (`gas_optics.py`) ✓
- Absorption parameterizations:
  - Water vapor continuum (temperature/pressure dependent)
  - CO2 absorption (15 μm band)
  - Ozone absorption (UV/vis and 9.6 μm)
- Rayleigh scattering
- Spectral band integration
- Separate SW and LW calculations

### 4. Planck Functions (`planck.py`) ✓
- Planck radiance calculations
- Band-integrated Planck functions
- Temperature derivatives
- Stefan-Boltzmann emission
- Layer and interface calculations

### 5. Cloud Optics (`cloud_optics.py`) ✓
- Effective radius parameterizations
- Liquid cloud optics (Slingo 1989)
- Ice cloud optics
- Combined liquid/ice properties
- SW: optical depth, SSA, asymmetry
- LW: absorption only
- Cloud overlap considerations

## Next Steps

### 1. Two-Stream Radiative Transfer Solver
Need to implement:
- Two-stream equations (Eddington approximation)
- Adding method for multiple layers
- Boundary conditions (surface, TOA)
- Clear-sky and all-sky calculations

### 2. Longwave Radiative Transfer
- Combine gas optics + cloud optics
- Apply two-stream solver
- Surface emission (Planck)
- Calculate upward/downward fluxes

### 3. Shortwave Radiative Transfer  
- Direct and diffuse components
- Multiple scattering
- Surface reflection
- Solar zenith angle effects

### 4. Heating Rate Calculations
- Flux divergence
- Convert to temperature tendency (K/s)
- Separate SW and LW contributions

### 5. Integration
- Main radiation interface function
- Integration with IconPhysics
- Handle radiation timestep

## Design Patterns Used

### JAX Patterns
- `@jax.jit` for all functions
- `vmap` for vectorization over columns
- `partial` for static arguments
- Array operations instead of loops
- Functional style (no mutations)

### Simplifications
- 2 SW bands (vis, NIR) instead of many
- 3 LW bands (window, CO2, H2O) 
- Fixed gas concentrations (except H2O)
- Simple cloud overlap
- No aerosols yet

## Testing Strategy
Each module has basic tests. Need comprehensive tests for:
- Energy conservation
- Clear-sky comparison with benchmark
- Cloud radiative forcing
- Heating rate profiles

## Key Implementation Decisions

1. **Spectral Resolution**: Started simple (2-3 bands) for easier debugging
2. **Gas Optics**: Parameterized rather than line-by-line
3. **Cloud Optics**: Following established parameterizations
4. **JAX Compatibility**: Pure functions, no side effects

## References Used
- Slingo (1989) - Cloud optics
- RRTM documentation - Gas absorption
- Fu-Liou papers - Two-stream methods
- ICON documentation - Overall structure

## Performance Considerations
- Minimize recomputation of Planck functions
- Vectorize over columns
- Consider checkpointing for memory
- Precompute band-averaged quantities

## Known Limitations
1. No scattering in LW (reasonable approximation)
2. Simple gas absorption (could use RRTM tables)
3. Fixed trace gases
4. No aerosols
5. Simple surface properties

## When Resuming
1. Start with two-stream solver implementation
2. Test with clear-sky case first
3. Add clouds incrementally
4. Verify energy conservation
5. Compare with reference calculations