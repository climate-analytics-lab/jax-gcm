# Solar Radiation Implementation

## Overview

The ICON radiation scheme supports two implementations for solar radiation calculations:

1. **jax-solar** (preferred): The official JAX-based solar radiation package
2. **Fallback implementation**: A compatible pure-JAX implementation

## jax-solar Integration

When available (Python 3.11+), the radiation scheme uses the `jax-solar` package which provides:
- Accurate solar geometry calculations
- Earth-Sun distance corrections
- Optimized JAX implementations
- Full compatibility with JAX transformations

### Installation

```bash
pip install jax-solar  # Requires Python 3.11+
```

## Implementation Details

### Module Structure

- `solar_jax.py`: Direct jax-solar integration
- `solar_interface.py`: Compatibility layer providing fallback
- `solar.py`: Original fallback implementation
- `__init__.py`: Automatic selection of available implementation

### Key Functions

All implementations provide these core functions:

```python
# GCM-oriented interface
calculate_solar_radiation_gcm(
    day_of_year: float,
    seconds_since_midnight: float,
    longitude: jnp.ndarray,
    latitude: jnp.ndarray,
    solar_constant: float = 1361.0
) -> Tuple[flux, cos_zenith]

# Basic calculations
cosine_solar_zenith_angle(latitude, longitude, day_of_year, hour_utc)
top_of_atmosphere_flux(cos_zenith, day_of_year, solar_constant)
daylight_fraction(latitude, day_of_year, timestep_hours)
```

### Checking Implementation

To check which implementation is being used:

```python
from jcm.physics.icon.radiation import get_solar_implementation
print(get_solar_implementation())
# Output: "jax-solar" or "Using fallback JAX implementation"
```

## Compatibility

Both implementations provide identical interfaces and comparable accuracy:
- Solar geometry calculations match within 0.5%
- TOA flux calculations are consistent
- All functions are JAX-compatible (jit, vmap, grad)

## Testing

The test suite (`test_radiation.py`) automatically tests whichever implementation is available:

```bash
python -m jcm.physics.icon.radiation.test_radiation
```

## Performance

- **jax-solar**: Optimized for performance, especially on GPU/TPU
- **Fallback**: Adequate performance for most use cases

Both implementations are JIT-compiled for optimal performance.

## Future Work

- Add support for orbital variations beyond eccentricity
- Include atmospheric refraction effects
- Support for other planets/orbital configurations