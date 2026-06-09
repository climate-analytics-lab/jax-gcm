# Writing a Physics Scheme

How to add your own parameterisation to JAX-GCM. The composable physics
infrastructure is meant to make this a one-file drop-in: you write a
`PhysicsTerm` subclass, declare what diagnostics you read and write, and
plug it into an existing factory. No edits to the model orchestrator, no
edits to the package's `Parameters` aggregator, no edits to a monolithic
`PhysicsData` struct.

This document walks through the contract and shows a complete minimal
example. For the *why* behind the design, see
[`composable_physics.md`](composable_physics.md).

## The contract

A scheme is a `PhysicsTerm` subclass. The base class lives at
`jcm/physics/physics_term.py` and is just a thin
`flax.nnx.Module` wrapper:

```python
from typing import ClassVar
from flax import nnx
from jcm.physics.physics_term import PhysicsTerm

class MyScheme(PhysicsTerm):
    name: ClassVar[str] = "my_scheme"
    category: ClassVar[str] = "convection"
    requires: ClassVar[tuple[str, ...]] = ("pressure_full",)
    provides: ClassVar[tuple[str, ...]] = ("my_scheme",)

    def __init__(self, params: MyParameters | None = None):
        self.params = nnx.Param(params or MyParameters.default())

    def __call__(self, state, diagnostics, forcing, terrain):
        ...
        return tendency, {**diagnostics, "my_scheme": MyData(...)}
```

The four `ClassVar`s drive composition-time validation. The two methods
do the real work.

### Static metadata

| Field | Meaning |
|---|---|
| `name` | Human-readable identifier. Shown in error messages and used as a fallback diagnostic key. Keep it unique within a composition. |
| `category` | The slot this term occupies (`radiation`, `convection`, `clouds`, `surface`, `vertical_diffusion`, `gravity_waves`, `aerosol`, `chemistry`, …). Used by `replace()` and `remove()` to find the term to swap out. Two terms with the same category are allowed; `replace()` collapses them. |
| `requires` | Diagnostic keys that some upstream term must already have written. Validated at construction time — see [Validation](#validation) below. |
| `provides` | Diagnostic keys this term writes into the dict it returns. Validation flags two terms claiming the same key. |

The keys in `requires` / `provides` are the public-facing keys in the
diagnostics dict (`pressure_full`, `air_density`, `convection`,
`radiation`, …). Anything starting with `_` is internal state (radiation
caching, the date struct) and should not appear in either field.

### `__init__`

Hold tunable parameters as `nnx.Param`. That is what makes them
gradient-trainable through `flax.nnx`:

```python
def __init__(self, params: MyParameters | None = None):
    self.params = nnx.Param(params or MyParameters.default())
```

`MyParameters` is your own scheme's parameters struct — typically a
`@tree_math.struct` dataclass. It is *not* the package-level `Parameters`
aggregator; each term owns only the parameters its scheme cares about.

If the term needs precomputed coordinate-dependent caches (e.g.
sigma layers, basis functions), put them in `nnx.Variable` attributes
inside an overridden `cache_coords(coords)`. That hook is called once
at `Model` construction time and is the right place for non-trainable
arrays:

```python
def cache_coords(self, coords):
    self.sigma = nnx.Variable(jnp.asarray(coords.vertical.layer_centers))
```

### `__call__`

```python
def __call__(self, state, diagnostics, forcing, terrain):
    return tendency, updated_diagnostics
```

Inputs:

- `state`: a `PhysicsState` (winds, temperature, humidity, geopotential,
  surface pressure, tracers).
- `diagnostics`: a `dict[str, jnp.ndarray | pytree]` containing
  everything upstream terms have written. Read by key; **never**
  mutate — always return a new dict.
- `forcing`: a `ForcingData` already sliced to the current step
  (`Model._get_step_fn_factory` calls `ForcingData.select(date)`).
- `terrain`: a `TerrainData` with `orog`, `fmask`, `gsl`, `vlt`, …

Outputs:

- A `PhysicsTendency` (winds, temperature, humidity, tracers). Most
  schemes use `PhysicsTendency.zero(...)` as a base and `.copy(...)`
  to fill in just the fields they touch.
- A new diagnostics dict — typically `{**diagnostics, "my_key": data}`.

The function must be pure: no side effects, no Python `if/else` on
JAX-traced values, no dict mutation. Use `jax.lax.cond` /
`jnp.where` for conditional logic.

### Tracers

If your scheme reads or writes a non-default tracer (anything beyond
`specific_humidity`), declare it via `required_tracers()`. Model
collects specs from every term at build time and seeds the initial
state's tracer dict accordingly:

```python
from jcm.physics.physics_term import TracerSpec

class MyScheme(PhysicsTerm):
    @classmethod
    def required_tracers(cls):
        return (
            TracerSpec(name="qc", units="kg/kg", initial_value=0.0),
            TracerSpec(
                name="qnc", units="1/kg", initial_value=0.0,
                nondimensionalize=False,
            ),
        )
```

`nondimensionalize=False` is for tracers that aren't mass mixing ratios
(e.g. number concentrations).

### Column vectorisation

If the scheme is a single-column algorithm and the host package uses
`vectorize_columns=True` (as `echam_physics()` does), `ComposablePhysics`
wraps each term in a `jax.vmap` over the horizontal axes. Inside the
term you can write the calculation as if it acted on one column.

Most ECHAM terms are written this way; `MoistAirColumnState` is a
counter-example that runs on the full 3-D grid because it is upstream
of the per-column vmap.

## Validation

`ComposablePhysics(terms=[...])` runs `_validate_ordering` at
construction time:

1. **Single-writer per key.** If two terms list the same key in their
   `provides`, the second would overwrite the first and is almost
   always a misconfiguration. Validation raises with the offending
   categories.
2. **All `requires` resolved upstream.** Each term's `requires` must
   appear in the union of upstream terms' `provides`. Errors point at
   the first unsatisfied dependency.

Most plugin authors hit this when they forget that the moist-air
diagnostics (`pressure_full`, `pressure_half`, `air_density`,
`layer_thickness`, …) are produced by the `MoistAirColumnState` term —
that has to be the first term in any column-physics composition.

## Minimal complete example

A 30-line `RayleighDamping` term that adds a height-dependent linear
drag on horizontal winds, with a tunable timescale. Single file,
no other repo edits required.

```python
# my_package/rayleigh_damping.py
from typing import ClassVar

import jax.numpy as jnp
import tree_math
from flax import nnx

from jcm.physics.physics_term import PhysicsTerm
from jcm.physics_interface import PhysicsTendency


@tree_math.struct
class RayleighDampingParameters:
    timescale_seconds: float
    sigma_top: float

    @classmethod
    def default(cls):
        return cls(timescale_seconds=1.0 * 86400.0, sigma_top=0.1)


class RayleighDamping(PhysicsTerm):
    name: ClassVar[str] = "rayleigh_damping"
    category: ClassVar[str] = "gravity_waves"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(self, params: RayleighDampingParameters | None = None):
        self.params = nnx.Param(
            params or RayleighDampingParameters.default(),
        )

    def cache_coords(self, coords):
        sigma = jnp.asarray(coords.vertical.layer_centers)
        self.sigma = nnx.Variable(sigma)

    def __call__(self, state, diagnostics, forcing, terrain):
        p = self.params.value
        # Smooth ramp from 0 below sigma_top to 1/timescale at the top.
        weight = jnp.clip(
            (p.sigma_top - self.sigma.value) / p.sigma_top, 0.0, 1.0,
        )
        rate = (weight / p.timescale_seconds)[:, None, None]
        u_t = -rate * state.u_wind
        v_t = -rate * state.v_wind
        tend = PhysicsTendency.zero(state).copy(u_wind=u_t, v_wind=v_t)
        return tend, diagnostics
```

The `requires` / `provides` are empty because the term reads only
the prognostic state and writes only tendencies — it does not consume
or produce any diagnostic keys.

## Composing your term in

Once the term exists, the standard composition operators on
`ComposablePhysics` let you splice it in without editing anyone else's
code:

```python
from jcm.physics.echam.echam_terms import echam_physics
from my_package.rayleigh_damping import RayleighDamping

# Append: add the term at the end of the default ECHAM stack.
physics = echam_physics() + RayleighDamping()

# Replace: swap out the existing non-orographic Hines drag term.
physics = echam_physics().replace("hines", RayleighDamping())

# Remove: drop a category, then build whatever you want on top.
physics = echam_physics().remove("clouds") + RayleighDamping()
```

`replace(category, new_term)` collapses *all* existing terms in that
category into the single replacement, inserted at the position of the
first one. `remove(category)` drops every term with that category.
`+` concatenates two compositions. Each operation runs the validation
checks again on the resulting term list.

For SPEEDY:

```python
from jcm.physics.speedy.speedy_terms import speedy_physics

physics = speedy_physics() + RayleighDamping()
```

The same drop-in works because the term contract is package-agnostic.

## Where the term lives in the tree

The repo organises files by *physical process*, named after the
*scheme*. Add your new file under the appropriate process directory:

```
jcm/physics/
├── convection/<scheme>.py
├── radiation/<scheme>.py
├── clouds/<scheme>.py
├── surface/<scheme>/
├── vertical_diffusion/<scheme>/
├── gravity_waves/<scheme>/
├── aerosol/<scheme>.py
├── chemistry/<scheme>.py
```

Third-party plugins can live anywhere on the import path. They do not
need to live inside `jcm/`.

## Verifying

Two cheap checks before you commit:

1. `JAX_PLATFORMS=cpu pytest -n 12 -m "not slow"` — the regular fast
   suite. Composing your term into `echam_physics()` or
   `speedy_physics()` and running an existing smoke test is a good
   way to make sure it survives a step.
2. A short gradient check via `jax.value_and_grad` to confirm
   `flax.nnx.Param` is wired up correctly. If the gradient w.r.t.
   your parameters is `None` or all-zero, the `nnx.Param` decoration
   is missing somewhere.

## Anti-patterns

- **Don't** subclass `ComposablePhysics`. The container is final;
  composition happens through the `+`, `replace`, and `remove`
  operators.
- **Don't** mutate `diagnostics` in place. Always return
  `{**diagnostics, "my_key": ...}`.
- **Don't** read the model timestep from a parameter struct. Use
  `diagnostics["_date"].dt_seconds`. The model dt may change across
  configurations and should never be smuggled through `Parameters`.
- **Don't** add a leading underscore to a `provides` key unless the
  data is genuinely internal (caches, transient state). Public keys
  flatten directly into `model.run().to_xarray()` output.
- **Don't** import from `jcm/physics/echam/` to make a SPEEDY-side
  term work. If you need shared infrastructure, put it under a
  scheme-neutral location like `jcm/physics/diagnostics/`.
