# ICON Physics Integration Guide

This guide explains how to use the ICON atmospheric physics package with JAX-GCM.

## Overview

The ICON physics package provides a JAX-compatible implementation of ICON atmospheric physics parameterizations, designed to work seamlessly with the JAX-GCM framework.

## Quick Start

### Basic Usage

```python
from jcm.physics.icon.icon_physics import IconPhysics
from jcm.physics.icon.constants import physical_constants

# Create ICON physics instance
icon_physics = IconPhysics(
    enable_radiation=True,
    enable_convection=True,
    enable_clouds=True,
    enable_vertical_diffusion=True,
    enable_surface=True,
    enable_gravity_waves=True,
    enable_chemistry=False
)

# Use with JAX-GCM Model (once integration is complete)
# model = Model(physics=icon_physics)
```

### Using ICON Diagnostics

```python
from jcm.physics.icon.diagnostics import wmo_tropopause
import jax.numpy as jnp

# Create atmospheric profile
pressure = jnp.logspace(jnp.log10(100000), jnp.log10(1000), 20)
temperature = jnp.array([...])  # Your temperature profile
surface_pressure = jnp.array([100000.0])

# Calculate tropopause pressure
tropopause_pressure = wmo_tropopause(
    temperature[None, :], 
    pressure[None, :], 
    surface_pressure
)
```

## Physics Modules

### Currently Implemented

- ✅ **Physical Constants**: Complete set of ICON-compatible physical constants
- ✅ **WMO Tropopause Diagnostic**: Fully functional tropopause calculation
- ✅ **Infrastructure**: Main physics class, data containers, test framework

### In Development

- ⏳ **Radiation**: Shortwave and longwave radiation schemes
- ⏳ **Convection**: Tiedtke-Nordeng convection parameterization
- ⏳ **Clouds**: Large-scale cloud microphysics
- ⏳ **Vertical Diffusion**: Boundary layer and turbulent mixing
- ⏳ **Surface**: Land-atmosphere exchange
- ⏳ **Gravity Waves**: Atmospheric gravity wave drag
- ⏳ **Chemistry**: Simple chemistry schemes (ozone, methane)

## Configuration Options

### IconPhysics Parameters

- `enable_radiation`: Enable radiation calculations
- `enable_convection`: Enable convection parameterization
- `enable_clouds`: Enable cloud microphysics
- `enable_vertical_diffusion`: Enable vertical diffusion
- `enable_surface`: Enable surface flux calculations
- `enable_gravity_waves`: Enable gravity wave drag
- `enable_chemistry`: Enable chemistry schemes
- `write_output`: Write physics output to predictions
- `checkpoint_terms`: Checkpoint physics terms for memory efficiency

### Physical Constants

Access physical constants through:

```python
from jcm.physics.icon.constants import physical_constants

# Examples
g = physical_constants.grav          # 9.81 m/s²
cp = physical_constants.cp           # 1004.0 J/K/kg
R = physical_constants.rgas          # 287.0 J/K/kg
```

## Integration Status

### ✅ Completed

1. **Package Structure**: Complete modular organization
2. **JAX Compatibility**: All code works with JAX transformations
3. **Test Framework**: Comprehensive testing setup
4. **Physical Constants**: Full set of ICON constants
5. **WMO Tropopause**: Working diagnostic implementation
6. **Documentation**: Complete API documentation

### 🔄 In Progress

1. **Full Model Integration**: Pending dinosaur package compatibility
2. **Physics Module Implementation**: Individual physics processes
3. **Validation**: Comparison with ICON Fortran reference

### 📋 TODO

1. **Performance Optimization**: JIT compilation and vectorization
2. **Memory Management**: Efficient handling of large atmospheric states
3. **Parallel Processing**: Multi-device support
4. **Benchmarking**: Performance comparison with SPEEDY physics

## Architecture

### Design Principles

1. **Modular**: Each physics process is an independent module
2. **JAX-Compatible**: Full support for autodiff, JIT, and vectorization
3. **Differentiable**: End-to-end differentiability for ML applications
4. **Extensible**: Easy to add new physics parameterizations
5. **Testable**: Comprehensive test coverage

### Code Organization

```
jcm/physics/icon/
├── __init__.py                    # Main package exports
├── constants/                     # Physical constants
├── boundary_conditions/          # External forcings
├── radiation/                    # Radiation schemes
├── convection/                   # Convection parameterizations
├── clouds/                       # Cloud microphysics
├── vertical_diffusion/           # Boundary layer mixing
├── surface/                      # Land-atmosphere exchange
├── gravity_waves/               # Gravity wave drag
├── chemistry/                   # Chemistry schemes
├── diagnostics/                 # Physics diagnostics
├── icon_physics.py             # Main physics class
└── icon_physics_test.py        # Integration tests
```

## Testing

### Running Tests

```bash
# Run all ICON physics tests
python -m pytest jcm/physics/icon/ -v

# Run specific test modules
python -m pytest jcm/physics/icon/icon_physics_test.py -v
python -m pytest jcm/physics/icon/diagnostics/wmo_tropopause_test.py -v
```

### Test Coverage

- ✅ Physical constants validation
- ✅ Data container functionality
- ✅ Physics class initialization
- ✅ WMO tropopause diagnostic
- ✅ JAX compatibility (autodiff, JIT, vmap)

## Performance

### JAX Optimizations

- **JIT Compilation**: All physics functions are JIT-compatible
- **Vectorization**: Support for batch processing with `jax.vmap`
- **Autodifferentiation**: Full gradient support for ML applications
- **Memory Efficiency**: Checkpoint-based memory management

### Benchmarks

Performance benchmarks will be added as physics modules are implemented.

## Contributing

### Adding New Physics Modules

1. Create module directory under appropriate category
2. Implement JAX-compatible physics function
3. Add comprehensive tests
4. Update module `__init__.py` exports
5. Add integration to main `IconPhysics` class

### Code Style

- Follow JAX best practices (pure functions, immutable data)
- Use type hints for all function signatures
- Include comprehensive docstrings
- Maintain test coverage > 90%

## Examples

See `example_icon_physics.py` for a complete example of using ICON physics with JAX-GCM.

## Support

For questions or issues:
1. Check the test files for usage examples
2. Review the ICON Fortran reference implementation
3. Open GitHub issues for bugs or feature requests

## License

Same as JAX-GCM main package.