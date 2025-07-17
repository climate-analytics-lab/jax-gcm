# JAX-GCM Validation Framework

This directory contains the validation framework for verifying the scientific accuracy of the JAX-GCM ICON physics implementation.

## Overview

The validation framework provides two complementary approaches:

1. **Direct Comparison**: Compare physics tendencies against ICON reference data
2. **Emergent Properties**: Test key atmospheric phenomena that emerge from the physics

## Directory Structure

```
validation/
├── README.md                          # This file
├── emergent_climate_validation.py     # Emergent properties validation
├── test_emergent_climate.py          # Test suite for emergent validation
├── single_column_validation.py       # Single column validation (TODO)
├── icon_hpc_configs/                 # HPC configuration files
│   ├── tropical_convection.nml       # Tropical convection test case
│   ├── midlatitude_winter.nml        # Mid-latitude winter test case
│   ├── arctic_polar.nml              # Arctic polar test case
│   ├── subtropical_clear.nml         # Subtropical clear sky test case
│   ├── run_tropical_convection.sh    # HPC run script
│   ├── run_midlatitude_winter.sh     # HPC run script
│   ├── run_arctic_polar.sh           # HPC run script
│   └── run_subtropical_clear.sh      # HPC run script
└── results/                          # Validation results (created when run)
```

## Quick Start

### 1. Test the Validation Framework

```bash
# Run quick validation test
python validation/test_emergent_climate.py --quick

# Run full test suite
pytest validation/test_emergent_climate.py -v
```

### 2. Run Emergent Properties Validation

```bash
# Run comprehensive emergent climate validation
python validation/emergent_climate_validation.py --duration 30 --output results/emergent_validation.json

# View results
cat results/emergent_validation.json
```

### 3. Generate ICON Reference Data (HPC)

```bash
# On HPC system, set environment variables
export ICON_BASE_PATH="/path/to/icon"

# Run test cases
cd validation/icon_hpc_configs/
sbatch run_tropical_convection.sh
sbatch run_midlatitude_winter.sh
sbatch run_arctic_polar.sh
sbatch run_subtropical_clear.sh
```

## Validation Tests

### Emergent Properties Validation

Tests fundamental atmospheric phenomena that emerge from the physics:

#### 1. Inter-Tropical Convergence Zone (ITCZ)
- **Expected**: 5°S to 10°N latitude
- **Intensity**: 8-20 mm/day precipitation
- **Width**: 2-8 degrees latitude
- **Reference**: Schneider et al. (2014)

#### 2. Hadley Cell Circulation
- **Strength**: 1-3 × 10¹¹ kg/s
- **Extent**: 25-35°N/S
- **Reference**: Dima & Wallace (2003)

#### 3. Global Energy Balance
- **TOA Balance**: ±2 W/m² globally
- **Tropical Excess**: 50-150 W/m²
- **Reference**: Trenberth et al. (2009)

#### 4. Precipitation Patterns
- **Global Mean**: 2.5-3.5 mm/day
- **Tropical Max**: 8-15 mm/day
- **Subtropical Min**: 0.5-2.0 mm/day
- **Reference**: GPCP observations

#### 5. Temperature Distribution
- **Global Mean**: 285-290 K
- **Equatorial**: 298-302 K
- **Polar**: 240-260 K
- **Reference**: ERA5 reanalysis

#### 6. Jet Stream Structure
- **Subtropical Jet**: 20-40 m/s at 25-35°N
- **Polar Jet**: 50-65°N
- **Reference**: Archer & Caldeira (2008)

### Single Column Validation

Direct comparison of physics tendencies against ICON reference data:

#### Test Cases
1. **Tropical Convection** (0°N, 180°E)
2. **Mid-latitude Winter** (50°N, 0°E)
3. **Arctic Polar** (85°N, 0°E)
4. **Subtropical Clear** (30°N, 30°W)

#### Validation Metrics
- **Temperature Tendencies**: RMSE < 20%
- **Moisture Tendencies**: RMSE < 30%
- **Momentum Tendencies**: RMSE < 40%
- **Surface Fluxes**: RMSE < 25%

## Usage Examples

### Example 1: Quick Validation Check

```python
from validation.emergent_climate_validation import EmergentClimateValidator

# Create validator
validator = EmergentClimateValidator()

# Run short validation
results = validator.run_comprehensive_validation(duration_days=10)

# Check results
for test_name, result in results.items():
    print(f"{test_name}: {'PASS' if result.passed else 'FAIL'}")
    print(f"  Measured: {result.measured_value:.2f} {result.units}")
    print(f"  Expected: {result.expected_range}")
```

### Example 2: ITCZ Analysis

```python
# Run aquaplanet simulation
state_history = validator.run_aquaplanet_simulation(duration_days=100)

# Test ITCZ formation
itcz_result = validator.test_itcz_formation(state_history)

print(f"ITCZ latitude: {itcz_result.measured_value:.1f}°N")
print(f"Expected range: {itcz_result.expected_range}")
print(f"Test passed: {itcz_result.passed}")
```

### Example 3: Energy Balance Check

```python
# Test global energy balance
energy_result = validator.test_global_energy_balance(state_history)

print(f"Global energy balance: {energy_result.measured_value:.1f} W/m²")
print(f"Expected range: {energy_result.expected_range}")
```

## Configuration

### Model Configuration

```python
model_config = {
    'grid_resolution': 'R2B4',      # 160 km resolution
    'vertical_levels': 47,          # ICON standard
    'time_step': 600,               # 10 minutes
}

validator = EmergentClimateValidator(model_config)
```

### Literature Values

The validation framework uses literature values for expected ranges:

```python
literature_values = {
    'itcz': {
        'latitude_range': (-5.0, 10.0),
        'intensity_range': (8.0, 20.0),
        'width_range': (2.0, 8.0),
        'reference': 'Schneider et al. (2014)'
    },
    # ... more values
}
```

## Interpretation Guidelines

### Success Criteria

- **Level 1**: Basic physics correctness (single column validation)
- **Level 2**: Emergent phenomena formation (ITCZ, Hadley cells)
- **Level 3**: Quantitative accuracy (energy balance, precipitation)
- **Level 4**: Long-term stability (multi-year simulations)

### Pass/Fail Thresholds

- **ITCZ**: Latitude within 5°, intensity within 50%
- **Energy Balance**: Global within 5 W/m², tropical within 30%
- **Circulation**: Hadley cell strength within 50%
- **Overall**: 80% of tests must pass

### Common Issues

1. **ITCZ too weak**: Check convection scheme parameters
2. **Energy imbalance**: Verify radiation calculations
3. **Weak circulation**: Check pressure gradient computation
4. **Unstable simulation**: Reduce time step or check numerics

## Expected Output

### Successful Validation
```
============================================================
EMERGENT CLIMATE VALIDATION SUMMARY
============================================================
Total tests: 6
Passed tests: 5
Success rate: 83.3%
Overall status: PASS
============================================================
ITCZ Formation           [PASS] ITCZ at 6.2°N, intensity 12.4 mm/day, width 4.1°
Hadley Cell Circulation  [PASS] Hadley cell strength: 2.1e+11 kg/s
Global Energy Balance    [PASS] Global: 0.8 W/m², Tropical: 89.2 W/m², Polar: -91.4 W/m²
Precipitation Patterns   [PASS] Global: 2.9, Tropical max: 11.8, Subtropical min: 1.2 mm/day
Temperature Distribution [PASS] Global: 287.3 K, Equatorial: 299.8 K, Polar: 248.2 K
Jet Stream Structure     [FAIL] Subtropical jet: 18.2 m/s at 28.3°N, Polar jet: 58.4°N
============================================================
```

## Development

### Adding New Tests

1. Add expected values to `literature_values` dictionary
2. Implement test method in `EmergentClimateValidator`
3. Add test to `run_comprehensive_validation()`
4. Update documentation

### Extending Validation

- Add seasonal cycle tests
- Include extreme weather validation
- Add regional climate validation
- Include coupled ocean-atmosphere tests

## References

- Schneider, T., et al. (2014). Migrations and dynamics of the intertropical convergence zone. *Nature*, 513(7516), 45-53.
- Dima, I. M., & Wallace, J. M. (2003). On the seasonality of the Hadley cell. *Journal of Climate*, 16(18), 3047-3060.
- Trenberth, K. E., et al. (2009). Earth's global energy budget. *BAMS*, 90(3), 311-323.
- Archer, C. L., & Caldeira, K. (2008). Historical trends in the jet streams. *GRL*, 35(8).

## Support

For questions or issues with the validation framework:

1. Check the test suite: `pytest validation/test_emergent_climate.py -v`
2. Review the validation plan: `ICON_PHYSICS_VALIDATION_PLAN.md`
3. Examine the physics implementation: `jcm/physics/icon/`