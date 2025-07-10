# Unit Conversions in ICON Physics

This document describes the unit conversions between the JAX-GCM physics interface and ICON physics parameterizations.

## Physics Interface Units (Input)

The physics interface provides state variables in the following units:

| Variable | Units | Notes |
|----------|-------|-------|
| `u_wind`, `v_wind` | m/s | Dimensional wind components |
| `temperature` | K | Dimensional temperature |
| `specific_humidity` | kg/kg | Mass mixing ratio |
| `geopotential` | m²/s² | Geopotential (Φ = gz) |
| `surface_pressure` | dimensionless | Normalized by p0 (100000 Pa) |
| `tracers` | kg/kg | Mass mixing ratios |

## ICON Physics Expected Units

ICON physics modules expect all inputs in SI units:

| Variable | Units | Notes |
|----------|-------|-------|
| Wind components | m/s | No conversion needed |
| Temperature | K | No conversion needed |
| Specific humidity | kg/kg | No conversion needed |
| Pressure | Pa | Must convert from normalized |
| Height | m | Must convert from geopotential |
| Air density | kg/m³ | Must calculate |
| Time step | s | Seconds |

## Required Conversions

### 1. Surface Pressure
```python
surface_pressure_pa = surface_pressure_normalized * p0
```
where `p0 = 100000 Pa`

### 2. Pressure Levels
```python
pressure_levels = sigma_levels * surface_pressure_pa
```
where sigma levels go from top (small values) to bottom (1.0)

### 3. Geopotential to Height
```python
height = geopotential / g
```
where `g = 9.80665 m/s²`

### 4. Air Density
```python
rho = pressure / (Rd * temperature)
```
where `Rd = 287.04 J/(kg·K)` is the gas constant for dry air

### 5. Layer Thickness
Using hydrostatic approximation:
```python
dz = dp / (rho * g)
```
where dp is the pressure difference between levels

## Implementation

All unit conversions are handled within the `IconPhysics` class:

1. **Convection** (`_apply_convection`): Converts surface pressure to Pa and calculates pressure levels
2. **Clouds** (`_apply_clouds`): Same pressure conversions
3. **Microphysics** (`_apply_microphysics`): Additionally calculates air density and layer thickness
4. **Gravity Waves** (`_apply_gravity_waves`): Converts geopotential to height and pressure to Pa

## Verification

The `unit_conversions.py` module provides:
- Individual conversion functions
- `prepare_physics_state_2d()` for batch conversions
- `verify_physics_units()` to check conversions produce reasonable values

## Important Notes

1. **Vertical Coordinate Convention**: 
   - Index 0 = top of atmosphere
   - Index nlev-1 = surface
   - Pressure increases with index
   - Height decreases with index

2. **No Unit Conversions Needed For**:
   - Wind components (already in m/s)
   - Temperature (already in K)
   - Specific humidity (already in kg/kg)
   - Tracers (already in kg/kg)

3. **Time Step**: 
   - Physics modules expect dt in seconds
   - Currently hardcoded to 1800s (30 minutes) in IconPhysics

## Testing

Run unit conversion tests:
```bash
python -m jcm.physics.icon.test_unit_conversions
```

This verifies:
- Surface pressure conversion
- Pressure level calculation
- Height conversion
- Air density calculation
- Layer thickness calculation
- Full state preparation