# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

JAX-GCM is a JAX-based General Circulation Model that combines the Dinosaur dynamical core with JAX implementations of atmospheric physics parameterizations. It provides a fully differentiable climate model suitable for ML-enhanced weather and climate modeling.

## Key Components

### Architecture
- **Dynamical Core**: Uses [Dinosaur](https://github.com/neuralgcm/dinosaur) for atmospheric dynamics
- **Physics Packages**:
  - **SPEEDY Physics**: Complete JAX implementation of SPEEDY atmospheric physics
  - **ICON Physics**: New JAX implementation of ICON parameterizations (active development)
  - **Held-Suarez**: Simplified physics for testing
- **Configuration**: Hydra-based configuration system with YAML files

### Project Structure
```
jcm/                      # Main package (JAX Climate Model)
├── physics/              # Physics implementations
│   ├── speedy/          # SPEEDY physics modules
│   ├── icon/            # ICON physics (radiation, convection, clouds, etc.)
│   └── held_suarez/     # Simplified physics
├── model.py             # Main model class
├── physics_interface.py # Physics abstraction layer
├── geometry.py          # Grid and coordinate handling
├── date.py              # Time/date management
└── boundaries.py        # Surface boundary conditions
```

## Development Commands

### Installation
```bash
pip install -e .  # Development mode
```

### Testing
```bash
# Run all tests
pytest

# Run specific module tests
pytest jcm/physics/icon/ -v

# Run with coverage (90% minimum required)
pytest --cov=jcm --cov-fail-under=90

# Skip slow tests
pytest -m "not slow"
```

### Linting
```bash
# Uses Ruff (configured for Python 3.11)
ruff --format=github --target-version=py311 .
```

### Documentation
```bash
cd docs && make html
```

## JAX Development Patterns

### Critical JAX Rules
1. **No Python control flow on JAX arrays** - Use JAX alternatives:
   - `if/else` → `jnp.where` or `lax.cond`
   - `for` loops → `lax.scan` or vectorized operations
   - `while` loops → `lax.while_loop`

2. **Static shapes required** - All array shapes must be known at compile time

3. **Pure functions only** - No side effects or stateful operations

4. **Vectorization pattern** - Use `vmap` for spatial operations:
   ```python
   # Centralized vectorization in physics modules
   tendency_fn = jax.vmap(compute_tendency, in_axes=(0, None))
   ```

### Common Conversions (from JAX_CONVERSION_PATTERNS.md)
- Replace `np.maximum(0, x)` with `jax.nn.relu(x)`
- Replace boolean indexing with `jnp.where`
- Use `lax.scan` for accumulation over sequences
- Implement custom gradients with `jax.custom_vjp`

## Testing Best Practices

1. **Use real objects instead of mocks**:
   - Create `DateData` with: `jcm.date.DateData(...)`
   - Create geometry with: `jcm.geometry.Geometry.from_grid_shape(...)`
   - These objects are JAX-compatible and easy to instantiate

2. **Test JAX transformations**:
   - Test `jax.jit` compilation
   - Test `jax.grad` for differentiability
   - Test `jax.vmap` for vectorization

3. **Coverage requirements**: Maintain >90% test coverage

## Physics Implementation Guidelines

### Physics Interface
All physics modules implement this protocol:
```python
class ExamplePhysics(Physics):

    def compute_tendencies(
        self,
        state: PhysicsState,
        boundaries: BoundaryData,
        geometry: Geometry,
        date: DateData,
    ) -> Tuple[PhysicsTendency, PhysicsData]:
       ...
```

Each term in the physics scheme has a top-level method that the physics interface can call, e.g.:
```python

@jit
def apply_radiation(state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> tuple[PhysicsTendency, PhysicsData]:
   ...
```

### ICON Physics Structure
The ICON physics package (`jcm/physics/icon/`) includes:
- **Radiation**: Shortwave and longwave schemes
- **Convection**: Deep and shallow convection
- **Clouds**: Cloud microphysics and cover
- **Vertical Diffusion**: Turbulent mixing
- **Surface**: Land-atmosphere interactions
- **Gravity Waves**: Orographic and non-orographic

Each module follows:
1. Modular design with clear interfaces
2. Full JAX compatibility (autodiff, JIT, vmap)
3. Comprehensive test coverage
4. Detailed documentation

### State Management
- `PhysicsState`: Immutable atmospheric state (u, v, T, q, φ, ps)
- `PhysicsTendency`: Time derivatives of state variables
- Tree-structured data using `tree_math` for arithmetic operations

## Dependencies and Requirements

- **Python**: 3.11+ (required for jax-solar)
- **Core**: JAX, Dinosaur, tree-math
- **Configuration**: hydra-core
- **I/O**: xarray, netCDF4
- **Testing**: pytest, pytest-cov
- **Documentation**: Sphinx with Furo theme

## Common Development Tasks

### Adding New Physics
1. Create module in appropriate physics package directory
2. Implement `Physics` calls
3. Ensure JAX compatibility (no Python control flow)
4. Add comprehensive tests
5. Document with docstrings and update integration guide

### Running Single Tests
```bash
pytest path/to/test_file.py::test_function_name -v
```

### Debugging JAX Issues
- Check for Python control flow on JAX arrays
- Verify static shapes
- Use `jax.debug.print` for debugging inside JIT
- Consult JAX_gotchas.md for common pitfalls
