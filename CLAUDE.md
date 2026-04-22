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
├── utils.py                  # Utilities, lookup tables, and coordinate creation
├── terrain.py                # Terrain boundary conditions (orography, land-sea mask)
├── forcing.py                # Forcing boundary conditions and I/O
├── date.py                   # Date handling
├── physics_interface.py      # Physics-dynamics coupling
├── diffusion.py              # Diffusion filter
├── config/                   # Hydra configuration files
├── physics/
│   ├── physics_term.py          # PhysicsTerm base class
│   ├── composable_physics.py    # ComposablePhysics container
│   ├── speedy/                  # SPEEDY infrastructure (params, coords)
│   │   ├── speedy_terms.py      # Composable terms + speedy_physics() factory
│   │   ├── speedy_coords.py
│   │   ├── params.py
│   │   ├── physics_data.py
│   │   └── physical_constants.py
│   ├── icon/                    # ICON infrastructure (params, coords)
│   │   ├── icon_terms.py        # Composable terms + icon_physics() factory
│   │   ├── icon_physics.py      # Standalone apply_* term functions used by icon_terms
│   │   ├── icon_coords.py, icon_levels.py, icon_physics_data.py, parameters.py
│   │   ├── unit_conversions.py, forcing.py
│   │   └── constants/           # ICON physical constants
│   ├── radiation/
│   │   ├── grey_two_stream/     # ICON-style grey two-stream package
│   │   ├── rrtmgp.py
│   │   ├── nn_emulator.py + nn_emulator_scheme.py
│   │   ├── radiation_types.py, cloud_optics.py, constants.py   # shared
│   │   └── speedy_shortwave.py, speedy_longwave.py
│   ├── convection/
│   │   ├── tiedtke_nordeng/     # Tiedtke-Nordeng mass flux scheme
│   │   └── speedy_convection.py
│   ├── clouds/
│   │   ├── sundqvist.py         # Sundqvist diagnostic cloud fraction
│   │   ├── echam_1m.py          # ECHAM 1-moment microphysics
│   │   ├── speedy_humidity.py, speedy_condensation.py
│   ├── vertical_diffusion/
│   │   ├── tte_tke/             # TTE-TKE closure
│   │   └── speedy_vdiff.py
│   ├── gravity_waves/hines/     # Hines (1997) gravity wave drag
│   ├── aerosol/macv2_sp.py      # Stevens et al. (2017) MACv2-SP simple plumes
│   ├── chemistry/simple_chemistry.py
│   ├── diagnostics/wmo_tropopause.py
│   ├── surface/                 # Speedy bulk + ICON multi-tile (in surface/icon/)
│   ├── forcing/speedy_forcing.py
│   ├── orographic_correction/speedy_orographic.py
│   └── held_suarez/             # Simplified Held-Suarez forcing
│       ├── held_suarez_physics.py
│       └── utils.py             # Coordinate helpers for Held-Suarez
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
- **Physics** parameterizations are modular — SPEEDY and ICON ports are the main implementations, Held-Suarez is a simpler alternative
- **Composable physics is the only physics API.** `PhysicsTerm` (flax.nnx.Module) base class wraps each parameterization. `ComposablePhysics` (and `ComposableIconPhysics`) aggregates terms with `replace()`, `remove()`, and `__add__()` operators. Build pre-configured packages via the `speedy_physics()` and `icon_physics()` factories.
- **physics_interface.py** bridges dynamics (spectral space) and physics (gridpoint space) with `PhysicsState` and `PhysicsTendency` structs
- **model.py** orchestrates time-stepping, combining dynamics and physics
- **Physics directory** is organized by physical process. Files are named after the **scheme** (e.g. `convection/tiedtke_nordeng/`, `clouds/sundqvist.py`, `aerosol/macv2_sp.py`), not the model they were ported from. Model-specific *infrastructure* (parameter containers, coords, data structs) stays under `speedy/` and `icon/`.
- Configuration is managed via **Hydra** (see `jcm/config/`)
- Supports multiple resolutions: T21 to T425 spectral truncations
- SPMD sharding support for multi-device execution

### Coordinate and terrain system

The grid/geometry system is split into three layers with clear separation of concerns:

1. **CoordinateSystem** (from `dinosaur`) — horizontal and vertical discretization, created via `utils.get_coords(sigma_boundaries, ...)`. This is physics-agnostic.

2. **TerrainData** (`terrain.py`) — runtime boundary conditions (orography, land-sea mask). Immutable and physics-agnostic. Has factory classmethods: `from_coords()`, `from_file()`, `aquaplanet()`, `single_column()`.

3. **SpeedyCoords** (`physics/speedy/speedy_coords.py`) — SPEEDY-specific precomputed coordinate transforms (sigma layers, trig functions). Cached on the physics object at init time via `cache_coords()`.

**Model initialization pattern:**
```python
from jcm.model import Model
from jcm.terrain import TerrainData
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.speedy.speedy_terms import speedy_physics

# 1. Create coordinate system (includes sigma boundaries)
coords = get_speedy_coords(layers=8, spectral_truncation=31)

# 2. Create terrain boundary conditions
terrain = TerrainData.from_coords(coords)  # or .aquaplanet()

# 3. Create model (physics caches coords internally)
model = Model(coords=coords, terrain=terrain, physics=speedy_physics())
```

**Key design principles:**
- **Static config** (coordinates, physics transforms) is set once at init and cached
- **Dynamic config** (terrain, forcing) can vary per simulation
- Physics classes implement `cache_coords(coords)` to precompute coordinate-dependent data
- Data structs use `@classmethod` factories (`.from_coords()`, `.from_file()`, `.aquaplanet()`) for clear construction intent
- `TerrainData` replaced the old monolithic `Geometry` class — terrain is now separate from coordinate configuration
