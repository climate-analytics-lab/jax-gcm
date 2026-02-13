# CLAUDE.md

## Project Overview

JAX-GCM (`jcm`) is a fully differentiable General Circulation Model (GCM) for atmospheric simulation, written entirely in JAX. It combines the Dinosaur spectral dynamical core with JAX implementations of SPEEDY atmospheric physics parameterizations. The model supports gradient-based optimization, data assimilation, and hybrid physics-ML workflows.

- **Package name:** `jcm`
- **Python:** >= 3.11 (strict requirement)
- **License:** Apache 2.0
- **Status:** Alpha (v1.0.0)

## Repository Structure

```
jcm/                          # Main package
├── model.py                  # Core Model class - main entry point
├── main.py                   # CLI entry point (Hydra config)
├── constants.py              # Global physical constants
├── utils.py                  # Utilities and lookup tables
├── geometry.py               # Grid geometry and terrain
├── forcing.py                # Boundary conditions and I/O
├── date.py                   # Date handling
├── physics_interface.py      # Physics-dynamics coupling
├── diffusion.py              # Diffusion filter
├── config/                   # Hydra configuration files
├── physics/
│   ├── speedy/               # SPEEDY physics parameterizations
│   │   ├── speedy_physics.py # Main SPEEDY orchestrator
│   │   ├── params.py         # Tunable parameter structs
│   │   ├── convection.py     # Convection scheme
│   │   ├── humidity.py       # Moisture processes
│   │   ├── large_scale_condensation.py
│   │   ├── shortwave_radiation.py
│   │   ├── longwave_radiation.py
│   │   ├── surface_flux.py   # Surface exchange
│   │   ├── vertical_diffusion.py
│   │   └── *_test.py         # Co-located unit tests
│   └── held_suarez/          # Simplified Held-Suarez physics
├── data/
│   ├── bc/                   # Boundary condition data (T30 climatology)
│   └── test/                 # Test reference data
└── *_test.py                 # Co-located unit tests
docs/                         # Sphinx documentation (RST + Furo theme)
notebooks/                    # Example Jupyter notebooks
```

## Build & Install

```bash
pip install -e .
```

Dependencies are in `requirements.txt`: dinosaur, flax, jax-datetime, tree-math, hydra-core, xarray.

## Running Tests

```bash
# All tests
pytest

# Fast tests only (skip slow integration tests >1 min)
pytest -m "not slow"

# Specific test file
pytest jcm/model_test.py

# With coverage
pytest --cov=jcm --cov-fail-under=90
```

Test files use the `*_test.py` naming convention and are co-located with their source modules. Tests use `unittest.TestCase` classes run via pytest. The `conftest.py` at root cleans `jcm` module imports between tests to prevent state leakage.

**CI thresholds:**
- Push: fast tests only, 90% coverage required
- Pull request: includes slow tests, 80% coverage required

## Linting

```bash
ruff check .
```

Ruff is the only linter. Configuration is in `pyproject.toml`. Docstring checks (D rules) are enabled but most missing-docstring rules are suppressed. No formatter (Black), no type checker (mypy), no pre-commit hooks.

## Key Coding Conventions

### Functional programming with JAX
- All functions must be **pure** (no side effects) to work with JAX transformations (`jit`, `grad`, `vmap`)
- Use **immutable data structures** via `@tree_math.struct` decorator
- No Python `if/else` on JAX-traced values — use `jax.lax.cond()` or `jnp.where()` instead
- Array shapes must be **statically known** where possible
- See `JAX_gotchas.md` for common pitfalls

### Data structures
```python
@tree_math.struct
class PhysicsState:
    u_wind: jnp.ndarray
    v_wind: jnp.ndarray
    temperature: jnp.ndarray
    ...
```

### Import conventions
```python
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
import xarray as xr
import tree_math
from dinosaur import primitive_equations
```

### Naming
- **snake_case** for functions and variables
- **PascalCase** for classes
- Descriptive names for physics variables: `u_wind`, `specific_humidity`, `surface_pressure`
- Abbreviated names acceptable in performance-critical inner functions

### Function patterns
- `get_*` — computation functions (e.g., `get_convection_tendencies`)
- `diagnose_*` — diagnostic calculations
- `compute_*` — derived quantity computation
- `set_*` — parameter/state modification

### Type hints and docstrings
- Type hints in function signatures (not strictly enforced)
- NumPy-style docstrings for public functions

### Testing
- Test files: `module_name_test.py` in the same directory as the module
- Mark slow tests (>1 min) with `@pytest.mark.slow`
- Include gradient checks (`check_vjp`, `check_jvp`) for JAX functions
- PRs should include tests for new functionality and bug fixes

## Documentation

Built with Sphinx + Furo theme:

```bash
cd docs && make html
```

Auto-generated physics variable translation docs come from `jcm/physics/speedy/units_table.csv` via `docs/generate_docs.py`.

## Architecture Notes

- **Dynamics** are handled by the external `dinosaur` package (spectral dynamical core)
- **Physics** parameterizations are modular — SPEEDY is the main implementation, Held-Suarez is a simpler alternative
- **physics_interface.py** bridges dynamics (spectral space) and physics (gridpoint space) with `PhysicsState` and `PhysicsTendency` structs
- **model.py** orchestrates time-stepping, combining dynamics and physics
- Configuration is managed via **Hydra** (see `jcm/config/`)
- Supports multiple resolutions: T21 to T425 spectral truncations
- SPMD sharding support for multi-device execution
