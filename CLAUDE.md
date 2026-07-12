# CLAUDE.md

## Think Before Coding
This is a complex codebase with many interdependencies and intricate scientific formulations. Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

 - State your assumptions explicitly. If uncertain, ask.
 - If multiple interpretations exist, present them - don't pick silently.
 - If a simpler approach exists, say so. Push back when warranted.
 - If something is unclear, stop. Name what's confusing. Ask.

Always document these decisions in the comments, and if appropriate in the documentation (and possibly in the high-level design documentation)

Comments should always reference the current state of the code, and explain *why* it is doing what it is doing, not how it is different to some previous version of the code (Which can get out of date and confusing)

## Inspecting model output — use the labelled xarray Dataset, never blind indexing
When analysing a run's netCDF output, **always** work through the xarray
``Dataset`` with its coordinates attached, and select by coordinate *value* —
never by a bare positional index whose meaning you have assumed.

 - **Select levels by their coordinate/pressure, not by index.** Use
   ``ds.sel(level=..., method="nearest")`` or first read ``ds.pressure_full`` to
   identify which index is the surface. Do **not** write ``.isel(level=-1)`` (or
   ``[-1]``/``[0]``) to mean "surface" — that bakes in a vertical-ordering
   assumption. The model's saved output is **surface-first** (level index 0 is
   the surface: ``level`` coordinate ≈ 0.996, ``pressure_full`` ≈ surface
   pressure at index 0; the top is the *last* index, ``level`` ≈ 1e-5, 1 Pa).
   This differs from the physics-**internal** frame (top-first: the radiation
   code's ``needs_reversal``) and from the HAMMOZ/ECHAM input **files**
   (top-first: ``hybm[0]=0``). All three conventions coexist, so never carry a
   "surface = index −1" habit between them — confirm from ``pressure_full``.
 - Within a single output file every level-dimensioned variable shares the
   **same** ``level`` coordinate and ordering (verified: temperature, pressure,
   tracers, oxidants all peak at index 0 = surface together). The Dataset is
   self-consistent; the risk is not a mixed-ordering file but *your* blind
   indexing of it. A 2026-07-05 "oxidant flip" investigation was a wasted effort
   caused entirely by reading ``.isel(level=-1)`` as the surface when it was the
   model top.

## Finish the Job — No Half Implementations
When asked to fix or implement something, deliver the **complete, faithful** solution by
default — do not ship a partial fix, a band-aid, or a "good enough for now" workaround and
present it as done. In this codebase "faithful" specifically means matching the reference
formulation (e.g. the ECHAM/ICON/SPEEDY Fortran), including the parts that are subtle or
tedious (implicit balances, conservation, edge cases) — not just the easy 80%.

 - If the correct fix turns out to be deeper than expected, do the deeper fix. Don't
   silently descope to the shallow version.
 - A workaround/cap/guard is acceptable **only** as an explicitly-labelled stopgap that the
   user has agreed to — never as a substitute for the real fix.
 - Validate that the full fix actually works (tests + a representative run where relevant),
   and verify conservation / physical correctness, before calling it done.
 - The only time to stop short is a genuine blocking decision that is the user's to make
   (per "Think Before Coding" above) — surface it and ask. Effort or tedium is not such a
   reason.

## Documentation lives with the change — no doc debt
Documentation updates are part of the change, not a follow-up. A PR that alters
user-facing behaviour is incomplete until the docs say so:

 - **Where things go.** General design decisions and analyses belong in
   ``docs/source/design/*.md`` (added to the toctree in ``docs/source/design.rst``);
   implementation-specific details and gotchas belong in the PR description.
   Do **not** create ad-hoc top-level ``*.md`` files in the repo root.
 - **User-facing behaviour changes** (new/changed defaults, new mechanisms like
   timestep resolution, new CLI/config knobs) must be reflected in
   ``README.md`` and/or ``docs/source/getting_started.rst`` in the same PR.
 - Keep code cross-references (docstrings/comments pointing at design docs)
   updated when a doc moves.

## Project Overview

JAX-GCM (`jcm`) is a fully differentiable General Circulation Model (GCM) for atmospheric simulation, written entirely in JAX. It combines the Dinosaur spectral dynamical core with JAX implementations of ICON /ECHAM and SPEEDY atmospheric physics parameterizations. The model supports gradient-based optimization, data assimilation, and hybrid physics-ML workflows.

- **Package name:** `jcm`
- **Python:** >= 3.11 (strict requirement)
- **License:** Apache 2.0
- **Status:** Alpha (v1.0.0)

Note, the latest development work should target the `dev` branch. Clean, working releases are periodically merged into `main` and tagged. 

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
├── dycore/                   # DynamicalCore protocol + implementations
│   ├── base.py, registry.py     # Protocol and registry
│   └── dinosaur/                # Dinosaur wrapper (dycore.py) + state_bridge.py
├── physics/
│   ├── physics_term.py          # PhysicsTerm base class
│   ├── composable_physics.py    # ComposablePhysics container
│   ├── speedy/                  # SPEEDY infrastructure (params, coords)
│   │   ├── speedy_terms.py      # Composable terms + speedy_physics() factory
│   │   ├── speedy_coords.py
│   │   ├── params.py
│   │   ├── physics_data.py
│   │   └── physical_constants.py
│   ├── echam/                   # ECHAM infrastructure (terms, coords)
│   │   ├── echam_terms.py       # Composable terms + echam_physics() factory
│   │   ├── echam_coords.py      # ECHAM-specific coordinate transforms
│   │   └── echam_levels.py      # Hybrid vertical level definitions
│   │   # (per-scheme Parameters live with each scheme; boundary
│   │   # conditions live in jcm/physics/forcing/echam_boundary_conditions.py)
│   ├── radiation/
│   │   ├── grey_two_stream/     # Grey two-stream package
│   │   ├── rrtmgp.py            # RRTMGP correlated-k wrapper (jax-rrtmgp)
│   │   ├── mcica.py + band_config.py  # McICA cloud sampling + band setup
│   │   ├── nn_emulator.py + nn_emulator_scheme.py
│   │   ├── radiation_types.py, cloud_optics.py, constants.py   # shared
│   │   └── speedy_shortwave.py, speedy_longwave.py
│   ├── convection/
│   │   ├── tiedtke_nordeng/     # Tiedtke-Nordeng mass flux scheme
│   │   ├── betts_miller/        # Betts-Miller convective adjustment (Isca/Frierson)
│   │   ├── saturation.py        # Shared Tetens saturation thermodynamics
│   │   └── speedy_convection.py
│   ├── clouds/
│   │   ├── sundqvist.py         # Sundqvist diagnostic cloud fraction
│   │   ├── echam_1m.py          # ECHAM 1-moment microphysics
│   │   ├── lohmann_2m.py        # Lohmann 2-moment microphysics (+ _params, cloud_utils)
│   │   ├── speedy_humidity.py, speedy_condensation.py
│   ├── vertical_diffusion/
│   │   ├── tte_tke/             # TTE-TKE closure
│   │   └── speedy_vdiff.py
│   ├── gravity_waves/
│   │   ├── hines/               # Hines (1997) non-orographic GWD
│   │   ├── sso/                 # Lott & Miller subgrid-orography drag
│   │   └── simple/              # Simple GWD fallback
│   ├── aerosol/
│   │   ├── macv2_sp.py          # Stevens et al. (2017) MACv2-SP simple plumes
│   │   ├── spa.py               # Simple Plumes Activation (CDNC/ICNC for 2M)
│   │   └── jam/                 # JAM modal aerosol (MAM4-style modes, tracers)
│   ├── chemistry/simple_chemistry.py
│   ├── diagnostics/             # wmo_tropopause.py, moist_air_state.py
│   ├── dissipation/upper_sponge.py  # Upper-level sponge dissipation
│   ├── surface/                 # Speedy bulk + ECHAM multi-tile (in surface/echam/)
│   ├── forcing/                 # speedy_forcing.py, echam_boundary_conditions.py
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
# Default — run in parallel across ~12 workers (pytest-xdist).
# Cuts a full sweep from ~15 min to a couple of minutes locally.
JAX_PLATFORMS=cpu pytest -n 12

# Single-process if you need ordered output or are debugging a flake
pytest

# Fast tests only (skip slow integration tests >1 min)
JAX_PLATFORMS=cpu pytest -n 12 -m "not slow"

# Specific test file
pytest jcm/model_test.py

# With coverage (xdist works with --cov)
JAX_PLATFORMS=cpu pytest -n 12 --cov=jcm --cov-fail-under=90
```

`-n auto` will pick the number of workers from the visible CPU count;
`-n 12` is the recommended local default on the dev workstation. Use
`-n 0` (or just omit `-n`) to fall back to a single process when you
need deterministic ordering.

**``JAX_PLATFORMS=cpu`` is required for parallel runs on GPU hosts.**
Without it, every xdist worker tries to grab the same GPU and you
get ``CUDA_ERROR_OUT_OF_MEMORY`` / ``dnn_support != nullptr``
``RET_CHECK`` failures from XLA. The unit tests don't need a GPU —
they're small column-mode integrations that compile and run faster
on CPU than they would round-trip through the device anyway.

Test files use the `*_test.py` naming convention and are co-located with their source modules. Tests use `unittest.TestCase` classes run via pytest. The `conftest.py` at root cleans `jcm` module imports between tests to prevent state leakage.

**CI thresholds:**
- Push: fast tests only, 90% coverage required
- Pull request: includes slow tests, 80% coverage required

## Linting

```bash
ruff check .
```

**Always run `ruff check .` locally and get it clean BEFORE every push.**
A push with lint errors burns a full CI cycle on a failure ruff reports in
seconds locally. Treat it like the test suite: part of the definition of
done for any commit that will be pushed.

Ruff is the only linter. Configuration is in `pyproject.toml`. Docstring checks (D rules) are enabled but most missing-docstring rules are suppressed. No formatter (Black), no type checker (mypy), no pre-commit hooks.

## Key Coding Conventions

### Functional programming with JAX
- All functions must be **pure** (no side effects) to work with JAX transformations (`jit`, `grad`, `vmap`)
- Use **immutable data structures** via `@tree_math.struct` decorator
- No Python `if/else` on JAX-traced values — use `jax.lax.cond()` or `jnp.where()` instead
- Array shapes must be **statically known** where possible
- See `JAX_gotchas.md` for common pitfalls

### Column physics: broadcasting-native, vertical on axis 0
New column-physics schemes (e.g. `convection/betts_miller/`) must be written
**broadcasting-native**: put the vertical level on **axis 0** and let any trailing
axes be horizontal and broadcast numpy-style. The *identical* code then runs
unchanged on a single `(kx,)` column, a `(kx, ncols)` vectorized block, or a whole
`(kx, ix, il)` grid — with **no `vmap`, no `reshape`, and no awareness of how the
host lays out the horizontal dimension.**

```python
# Vertical reductions/scans are over axis 0; per-column scalars are (*horiz).
dp     = phalf[1:] - phalf[:-1]            # (kx, *horiz)
precip = jnp.sum(qdel * dp, axis=0) / c.grav   # (*horiz)
# Reshape a level index to broadcast against a (kx, *horiz) field:
levels = jnp.arange(kx).reshape((kx,) + (1,) * (field.ndim - 1))
```

- A term **must not** branch on the horizontal rank (`if temperature.ndim == 3`)
  or unpack horizontal shape (`kx, ix, il = temperature.shape`). Index axis 0 for
  the vertical and reduce with `axis=0`; everything else broadcasts.
- The `PhysicsTerm` wrapper builds `(kx+1, *horiz)` pressures with
  `phalf = a_half.reshape((-1,) + (1,)*ps.ndim) + b_half[...] * ps[None]`, so it too
  is agnostic to whole-grid vs column-vectorized hosts.
- This is what lets `ComposablePhysics(vectorize_columns=...)` run a scheme either
  way without the scheme knowing. Cover both shapes in tests (a `(kx,)` column and
  a `(kx, ncols)` / `(kx, ix, il)` broadcast must agree per column).

### Data structures
State and tendencies use `@tree_math.struct` (vector-math semantics for the
time-stepper):
```python
@tree_math.struct
class PhysicsState:
    u_wind: jnp.ndarray
    v_wind: jnp.ndarray
    temperature: jnp.ndarray
    ...
```

**Differentiable physics parameters** use `flax.struct.dataclass` so the numeric
tunables are pytree leaves you can take gradients with respect to, while genuinely
*static* configuration (enums/flags that select code paths at trace time) is marked
`pytree_node=False` and stays as Python aux data usable in ordinary `if` branches:
```python
@struct.dataclass
class BettsMillerParameters:
    tau_bm: jnp.ndarray = 7200.0                                    # differentiable leaf
    rhbm:   jnp.ndarray = 0.8                                       # differentiable leaf
    shallow: ShallowScheme = struct.field(pytree_node=False,
                                           default=ShallowScheme.SIMP)  # static aux
```
Inside a `PhysicsTerm`, hold such a parameter object in an `nnx.Param(...)` and read
it back with `.get_value()` so the leaves are visible to `jax.grad` / optimizers.
Do **not** make a numeric tunable static — that hides it from gradients (this was a
review requirement: physics parameters must be differentiable like all the others).

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

- **Dynamics** are handled by the external `dinosaur` package (spectral dynamical core), wrapped behind the `DynamicalCore` protocol in `jcm/dycore/` (dinosaur wrapper + state_bridge)
- **Physics** parameterizations are modular — SPEEDY and ECHAM ports are the main implementations, Held-Suarez is a simpler alternative
- **Composable physics is the only physics API.** `PhysicsTerm` (flax.nnx.Module) base class wraps each parameterization. `ComposablePhysics` aggregates terms with `replace()`, `remove()`, and `__add__()` operators. Build pre-configured packages via the `speedy_physics()` and `echam_physics()` factories.
- **physics_interface.py** bridges dynamics (spectral space) and physics (gridpoint space) with `PhysicsState` and `PhysicsTendency` structs
- **model.py** orchestrates time-stepping, combining dynamics and physics
- **Physics directory** is organized by physical process. Files are named after the **scheme** (e.g. `convection/tiedtke_nordeng/`, `clouds/sundqvist.py`, `aerosol/macv2_sp.py`), not the model they were ported from. Model-specific *infrastructure* (terms, coords, levels) stays under `speedy/` and `echam/`.
- Configuration is managed via **Hydra** (see `jcm/config/`)
- Supports multiple resolutions: T21 to T425 spectral truncations
- SPMD sharding support for multi-device execution

### Physical constants (`jcm.constants`)

`jcm/constants.py` is the **single source of truth** for general physical
constants shared across all physics packages. The v2 API:

- **One canonical name per independent quantity** — no aliases. Dry-air specific
  heat is `cpd`, the dry-air gas constant `rd`, the melting point `tmelt`, etc.
- **Derived quantities are `@property`** on `PhysicalConstants` (a `NamedTuple`):
  `rd = akap·cpd`, `cvd = cpd - rd`, `rgrav = 1/grav`, the `vtmpc*` moisture
  coefficients. They recompute on access, so they can never drift out of sync
  with the bases — even after an override.
- **Process-global override** via `set_constants(...)`:
  ```python
  import jcm.constants as c
  c.set_constants(grav=9.80665)            # tweak one base value (kwargs)
  c.set_constants(PhysicalConstants(...))  # or replace the whole set
  ```
  Call it *before* constructing the model (the dynamical core reads the live
  singleton at construction). Only *base* fields may be overridden by keyword;
  derived quantities recompute automatically.
- **Read constants by attribute access**, not `from`-import. `import jcm.constants
  as c; ... c.grav` (a module-level `__getattr__` forwards to the live singleton)
  honours overrides. `from jcm.constants import grav` binds the value at import
  time and will **not** track `set_constants`. New code must use `c.<name>`.

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
