# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

jax-gcm is a fully differentiable General Circulation Model (GCM) written in JAX for climate modeling and machine learning applications. It combines the Dinosaur dynamical core with JAX implementations of atmospheric physics packages (SPEEDY and ICON), enabling GPU/TPU acceleration and automatic differentiation through the entire model.

## Installation and Dependencies

```bash
# Install in development mode
pip install -e .

# Core dependencies (from requirements.txt)
# - dinosaur (dynamical core)
# - tree-math (PyTree arithmetic)
# - hydra-core (configuration)
# - xarray[io, parallel] (I/O and data handling)
# - jax (core computation framework)
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest jcm/model_test.py

# Run single test function
pytest jcm/physics/speedy/convection_test.py::test_function_name

# Run tests with verbose output
pytest -v

# Test file naming convention: *_test.py (configured in setup.cfg)
# Slow tests can be marked with @pytest.mark.slow
```

## Code Architecture

### High-Level Model Structure

The model follows a three-layer architecture:

1. **Dynamical Core (Dinosaur)**: Solves primitive equations in spectral space
2. **Physics Interface (`jcm/physics_interface.py`)**: Converts between spectral (dynamics) and nodal (physics) representations
3. **Physics Packages**: SPEEDY (`jcm/physics/speedy/`) and ICON (`jcm/physics/icon/`) parameterizations

### Key Data Structures

- **`PhysicsState`**: Nodal space atmospheric state (u_wind, v_wind, temperature, specific_humidity, geopotential, normalized_surface_pressure)
- **`PhysicsTendency`**: Rates of change for physics variables (u_wind, v_wind, temperature, specific_humidity)
- **`BoundaryData`**: Surface boundary conditions (land-sea mask, orography, SST, albedo, etc.)
- **`DateData`**: JAX-compatible time/date representation (use `jcm.date.DateData`, not datetime objects, for JAX compatibility)
- **`Geometry`**: Grid geometry and sigma coordinates (create with `Geometry.from_coords()` or `Geometry.from_grid_shape()`)

### Model Execution Flow

1. **Initialization** (`Model.__init__`):
   - Creates coordinate system via `get_coords(layers, horizontal_resolution)`
   - Initializes geometry, physics package, and dynamical core
   - Sets up diffusion filters and reference temperature profile

2. **Time Integration** (`Model.run` / `Model.resume`):
   - `run()`: Initializes model state and starts simulation from beginning
   - `resume()`: Continues from previous `_final_modal_state`
   - Uses IMEX-RK time stepping with physics tendencies added as explicit forcing
   - Applies filters (global mean pressure conservation, horizontal diffusion) after each step

3. **Physics-Dynamics Coupling** (`get_physical_tendencies` in `physics_interface.py`):
   - Converts spectral dynamics state to nodal physics state
   - Calls physics package to compute tendencies
   - Verifies state validity (clamps negative humidity)
   - Converts physics tendencies back to spectral space
   - Applies diffusion filter to tendencies

### Physics Packages

#### SPEEDY Physics (`jcm/physics/speedy/`)
Default physics package with modules for:
- Convection (Tiedtke-like scheme)
- Large-scale condensation
- Shortwave and longwave radiation
- Surface fluxes
- Vertical diffusion
- Orographic correction (drag)

Physics terms are executed sequentially in `SpeedyPhysics.compute_tendencies()`, accumulating tendencies and updating `PhysicsData` at each step.

#### ICON Physics (`jcm/physics/icon/`)
Modular atmospheric physics from ICON model:
- Subdirectories: `radiation/`, `convection/`, `clouds/`, `vertical_diffusion/`, `surface/`, `gravity_waves/`, `chemistry/`, `aerosol/`, `diagnostics/`, `boundary_conditions/`
- Each module is JAX-compatible (autodiff, JIT, vmap support)
- Test files: `test_*.py` or `*_test.py` patterns

### Coordinate Systems and Grids

- **Horizontal**: Spherical harmonic spectral representation
  - Available resolutions: T21, T31, T42, T85, T106, T119, T170, T213, T340, T425
  - Resolution = total_wavenumbers - 2
- **Vertical**: Sigma coordinates (pressure-normalized)
  - Available layer configurations: 8 layers (default), others defined in `jcm/geometry.py::sigma_layer_boundaries`

### State Conversions

- `dynamics_state_to_physics_state()`: Spectral → Nodal (vor/div → u/v, dimensionalize humidity)
- `physics_state_to_dynamics_state()`: Nodal → Spectral (inverse of above)
- `physics_tendency_to_dynamics_tendency()`: Converts physics tendencies to dynamics format
- Conversions use `coords.horizontal.to_modal()` / `to_nodal()` for spectral transforms

## JAX-Specific Considerations

### Common Patterns

1. **Conditional Logic**: Use `jax.lax.cond()` instead of Python if/else for JAX types (see `JAX_gotchas.md`)
2. **Vectorization**: When using `vmap`, be careful with scalar comparisons; use `jnp.where()` or `jnp.less_equal()`
3. **Checkpointing**: Physics terms can be checkpointed with `jax.checkpoint()` to reduce memory usage
4. **JIT Compilation**: Main integration loop is JIT-compiled for performance

### Testing Best Practices

- Use `jcm.date.DateData` for time/date objects (JAX-compatible)
- Use `jcm.geometry.Geometry.from_grid_shape()` to create geometry objects
- Avoid mocking `DateData` or `Geometry` - they're easy to instantiate and work with JAX
- The `conftest.py` fixture automatically cleans up jcm imports between tests

## Common Development Tasks

### Running a Simulation

```python
from jcm.model import Model

# Create model with default aquaplanet configuration
model = Model(
    time_step=30.0,  # minutes
    layers=8,
    horizontal_resolution=31,  # T31 grid
)

# Run simulation
predictions = model.run(
    save_interval=10.0,  # days
    total_time=120.0,    # days
)

# Convert to xarray for analysis
ds = model.predictions_to_xarray(predictions)
```

### Using Custom Physics or Boundaries

```python
from jcm.physics.speedy.speedy_physics import SpeedyPhysics
from jcm.boundaries import boundaries_from_file

# Custom physics with modified parameters
physics = SpeedyPhysics(write_output=True, checkpoint_terms=True)

# Load realistic boundaries from netCDF
boundaries = boundaries_from_file("boundaries.nc", model.coords.horizontal)

# Run with custom configuration
predictions = model.run(boundaries=boundaries)
```

### Computing Gradients

```python
import jax

# Define loss function
def loss_fn(params):
    model = Model(...)
    predictions = model.run(...)
    return jnp.mean(predictions.dynamics.temperature)

# Compute gradient
grad_fn = jax.grad(loss_fn)
gradients = grad_fn(params)
```

## Documentation

- **Build docs**: `cd docs && make clean && make html`
- **View docs**: Open `docs/build/html/index.html`
- **Format**: reStructuredText (.rst) with Sphinx
- **API reference**: Auto-generated from docstrings
- **Key docs**:
  - `docs/source/getting_started.rst`: Installation and first steps
  - `docs/source/design.rst`: Architecture overview
  - `docs/source/developer.rst`: Profiling and development guides
  - `docs/source/speedy_translation.rst`: SPEEDY implementation notes

## Profiling

Use JAX profiler for performance analysis:

```python
import jax.profiler

jax.profiler.start_trace("./tensorboard_logs", create_perfetto_trace=True)
predictions = model.run(...)
jax.tree_util.tree_map(
    lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
    predictions
)
jax.profiler.stop_trace()
```

View traces at https://ui.perfetto.dev/

## Project Structure

```
jcm/
├── model.py              # Main Model class
├── physics_interface.py  # Physics-dynamics coupling
├── boundaries.py         # Boundary condition handling
├── date.py              # JAX-compatible date/time
├── geometry.py          # Grid geometry and sigma coordinates
├── diffusion.py         # Horizontal diffusion filter
├── physics/
│   ├── speedy/         # SPEEDY physics package
│   ├── icon/           # ICON physics package (modular)
│   └── held_suarez/    # Held-Suarez idealized physics
├── config/             # Hydra configuration files
└── vertical/           # Vertical coordinate utilities

notebooks/              # Example Jupyter notebooks
docs/                  # Sphinx documentation
validation/            # Validation scripts
```

## Version Control

- Main branch: `main`
- Python version: >=3.11
- Package name: `jcm` (JAX Climate Model)
- Versioning: setuptools_scm with fallback version "999"
