# Composable physics

## Overview

A JCM physics package is a `ComposablePhysics` container holding an
ordered list of `PhysicsTerm` instances. Each term is a self-contained
parameterisation that reads the prognostic state and a shared
`diagnostics` dict, returns its tendency, and writes its outputs back
into the dict for downstream terms to consume.

```text
Model._get_op_split_step_fn
   └─ ComposablePhysics.compute_tendencies(state, forcing, terrain, prev_carry)
        diagnostics = {**prev_carry}    ← cross-step physics carry seed
        for term in terms:
            tend, diagnostics = term(state, diagnostics, forcing, terrain)
            tendencies += tend
        return tendencies, diagnostics  ← diagnostics is the next step's carry
```

`ComposablePhysics` is `flax.nnx.Module`, every `PhysicsTerm` is
`flax.nnx.Module`, and per-term parameters are `nnx.Param`. Composition
is differentiable end-to-end via either `nnx.grad` directly on the
container or `jax.grad` over a `nnx.split`-flattened state.

The container is final — it is not subclassed. Composition happens at
construction time via `+`, `replace(category, term)`, and
`remove(category)`. `_validate_ordering` runs each time a composition
is built.

## Components

### `PhysicsTerm` (`jcm/physics/physics_term.py`)

Base class for one parameterisation. Each subclass declares four
`ClassVar`s of static metadata and implements two methods.

```python
class PhysicsTerm(nnx.Module):
    name:     ClassVar[str]                  # unique identifier
    category: ClassVar[str]                  # "radiation", "convection", …
    requires: ClassVar[tuple[str, ...]] = () # diagnostics keys read from upstream
    provides: ClassVar[tuple[str, ...]] = () # diagnostics keys written

    def cache_coords(self, coords) -> None:
        """Populate coordinate-dependent caches as nnx.Variable. In-place.
        Called once at Model construction time, outside any traced region."""

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict[str, Any],
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict[str, Any]]:
        """Return (tendency, updated_diagnostics)."""
```

Storage convention:

- **`nnx.Param`** — tunable parameters. `nnx.grad` differentiates through them.
- **`nnx.Variable`** — coordinate caches and other traced-but-frozen
  state. Read inside `__call__` as live pytree leaves; not differentiated by default.
- **Plain Python attributes** — static configuration (flags, integer
  knobs that should not change post-construction).

### `ComposablePhysics` (`jcm/physics/composable_physics.py`)

The single container class. Iterates terms in order, threads the
`diagnostics` dict through, sums tendencies, and exposes the
composition operators.

```python
class ComposablePhysics(nnx.Module, Physics):
    def __init__(self, terms: list[PhysicsTerm], *, vectorize_columns: bool = False):
        self.terms = terms
        self._validate_ordering()
        self.vectorize_columns = vectorize_columns

    def cache_coords(self, coords):
        for term in self.terms:
            term.cache_coords(coords)

    def compute_tendencies(self, state, forcing, terrain, *, prev_physics_data=None):
        diagnostics = dict(prev_physics_data) if prev_physics_data else {}
        tendencies = PhysicsTendency.zeros(state.temperature.shape)
        for term in self.terms:
            tend, diagnostics = term(state, diagnostics, forcing, terrain)
            tendencies += tend
        return tendencies, diagnostics

    # Composition operators (each returns a fresh container, runs validation):
    def __add__(self, other):           ...   # concatenate term lists
    def replace(self, category, new):   ...   # swap all terms of category
    def remove(self, category):         ...   # drop all terms of category
```

`vectorize_columns=True` (used by `echam_physics()`) wraps each term in
`jax.vmap` over the horizontal axes so single-column algorithms can be
written as if they acted on one column. The outer container handles the
reshape/un-reshape; terms see `(nlev,)` or `(nlev, ncols)` as appropriate.

### Pre-built factories

Each physics package is a factory that returns a `ComposablePhysics`
with a validated ordering. The factories live next to their term files:

- `jcm/physics/speedy/speedy_terms.py::speedy_physics()`
- `jcm/physics/echam/echam_terms.py::echam_physics()`
- `jcm/physics/held_suarez/held_suarez_physics.py::held_suarez_physics()`

Held-Suarez stays a hand-written package — composable machinery is
optional, not forced.

## The `diagnostics` dict

Terms communicate through `dict[str, Any]` (typically
`dict[str, jnp.ndarray]` or typed sub-structs of arrays). The dict
flows forward through the term list, every term reads the keys it
needs and returns a *new* dict with any keys it produces — never
mutate in place.

Two key conventions:

- **Public keys** (no leading underscore): exposed as user-facing
  diagnostic output. Flatten directly into `model.run().to_xarray()`
  via `Physics.data_struct_to_dict`. Example: `radiation`, `convection`,
  `cloud_fraction`.
- **Internal keys** (leading underscore): cross-step or transient
  state used by terms internally; filtered out of user-facing output.
  Examples: `_radiation` (sub-cycle cache), `_date` (jax_datetime
  struct + dt_seconds carried for terms that need it),
  `_dt_seconds` (model timestep injected by `ComposablePhysics` so
  terms read a single source of truth instead of plumbing it through
  parameters or date).

The diagnostics dict that comes out of `compute_tendencies` *is* the
cross-step physics carry — operator-split integration threads it as a
JAX pytree through `lax.scan` (see [`operator_split_physics.md`](operator_split_physics.md)).

## Validation

`ComposablePhysics(terms=[...])` runs `_validate_ordering` at
construction time:

1. **Single-writer per key.** Two terms cannot list the same key in
   their `provides`. Catches misconfigurations where one term would
   silently overwrite another's output.
2. **All `requires` resolved upstream.** Each term's `requires` must
   appear in the union of upstream terms' `provides`. Catches
   ordering bugs at `Model` construction rather than at the first
   `model.run()`.

Empty `requires` / `provides` are fine — terms that read only the
prognostic state and write only tendencies (e.g. a Rayleigh damping
term) declare nothing.

`required_tracers()` is a separate hook each term can implement to
declare non-default tracers (anything beyond `specific_humidity`).
`Model` collects specs from every term at build time and seeds the
initial state's tracer dict.

## Composition operators

Three operators on the container, each returning a fresh
`ComposablePhysics` after re-running validation:

```python
physics_a + physics_b                 # concatenate term lists
physics.replace("convection", new)    # swap all 'convection' terms for `new`
physics.remove("clouds")              # drop every term whose category is 'clouds'
```

`replace(category, new_term)` collapses *all* existing terms in that
category into the single replacement, inserted at the position of the
first one — so you can swap an entire process category in one call:

```python
# ECHAM with a custom convection scheme
physics = echam_physics().replace("convection", MyConvection())

# Strip clouds entirely, then add Rayleigh damping
physics = echam_physics().remove("clouds") + RayleighDamping()
```

Replacing a wavelength-dependent radiation backend also requires updating the
enclosing composition's `band_config` to match that backend. The RRTMGP Hydra
configurations perform this setup automatically.

When users compose from scratch (`term_a + term_b + term_c`), they
accept responsibility for ordering correctness — `_validate_ordering`
catches dependency bugs but cannot enforce semantic order beyond the
`requires` / `provides` graph.

## Cross-package compatibility

SPEEDY and ECHAM terms historically used different intermediate data
representations. The diagnostics dict is the bridge:

- SPEEDY wrappers store SPEEDY sub-structs under keys like
  `"_shortwave_rad"`, `"_convection"`.
- ECHAM wrappers store ECHAM sub-structs under keys like
  `"radiation"`, `"convection"`.

Mixing terms across packages works cleanly when the replacement
covers an entire process category — the new term reads only public
diagnostics and prognostic state, and produces what downstream terms
need from scratch. Sharing internal sub-structs across packages
requires either a translation term or independent recomputation; in
practice the entire-category replacement pattern is what most use
cases want.

## Differentiability

End-to-end gradient flow is preserved through the diagnostics dict
(all values are JAX arrays or pytrees of arrays), per-term
`nnx.Param` storage, and the composability operators (which produce
fresh `ComposablePhysics` modules with the same nnx graph
properties).

**Pattern 1 — direct `nnx.grad` (most ergonomic):**

```python
physics = speedy_physics()
physics.cache_coords(coords)   # run ONCE, outside the traced region

def loss_fn(physics):
    model = Model(coords=coords, terrain=terrain, physics=physics)
    return compute_loss(model.run(total_time=...))

grads = nnx.grad(loss_fn)(physics)
# grads.terms[i].params is the gradient w.r.t. term i's parameters
```

**Pattern 2 — pure JAX via split/merge (interop with existing `jax.grad` code):**

```python
graphdef, state = nnx.split(physics)

def loss_fn(state):
    physics = nnx.merge(graphdef, state)
    model = Model(coords=coords, terrain=terrain, physics=physics)
    return compute_loss(model.run(total_time=...))

grads = jax.grad(loss_fn)(state)
```

**Pattern 3 — per-scheme optimisation:**

```python
# Optimise only convection parameters, freeze the rest
convection_filter = nnx.PathContains("convection")
grads = nnx.grad(loss_fn, wrt=convection_filter)(physics)

# Or address a single term's params directly
opt = optax.adam(1e-3)
opt_state = opt.init(physics.terms[3].params)
```

Path-based filtering is the largest ergonomic improvement over a
monolithic `Parameters` struct: optimising one scheme in isolation
needs no surgery.

## `cache_coords` lifecycle

`cache_coords(coords)` is called once at `Model.__init__` time, before
any traced region:

```python
self.physics = physics
self.physics.cache_coords(self.coords)
```

Inside `cache_coords`, a term stores precomputed data (sigma layer
midpoints, basis functions, lookup tables) as `nnx.Variable`
attributes. During `model.run()` those variables are read as traced
values but never written.

The default behaviour of `nnx.grad` is to ignore `nnx.Variable` and
only differentiate `nnx.Param` — so coordinate caches do not appear in
the gradient. Callers that *do* want to differentiate w.r.t. the
vertical-level placement (rare, but supported for learnable level
schemes) can broaden the path filter.

## Process-parallel coupling

Within a `ComposablePhysics` step, terms run **process-parallel**:
every term reads the same input prognostic `state`, and tendencies
are summed. Terms may read each other's *diagnostic* outputs
(through the dict), but they do not see each other's tendency
contribution applied to the prognostic state until the next dynamics
step.

This is order-independent at the prognostic-state level —
`A + B + C` and `B + A + C` produce the same total tendency from the
same state. It differs from ECHAM6's sequential coupling, where each
scheme reads the state with prior schemes' tendencies already added
(via the `tte += ...` pattern on shared accumulators in
`mo_scan_buffer.f90`). Sequential coupling is more accurate for
tightly-coupled process pairs but gives up the order-independence
that makes `replace()` / `remove()` semantically clean.

For terms that genuinely need sequential coupling (e.g. a tightly
coupled CLUBB+MG2 pair as in E3SM), the recommended pattern is a
*process group* term that runs the inner sub-cycle internally and
presents a single tendency externally — keeping the outer container
process-parallel.

## Plugin contract

A third-party scheme is a single-file drop-in. See
[`writing_a_physics_scheme.md`](writing_a_physics_scheme.md) for a
walkthrough; the contract is:

```python
# my_package/my_scheme.py
from typing import ClassVar
from flax import nnx
from jcm.physics.physics_term import PhysicsTerm

class MyScheme(PhysicsTerm):
    name:     ClassVar[str] = "my_scheme"
    category: ClassVar[str] = "convection"
    requires: ClassVar[tuple[str, ...]] = ("pressure_full",)
    provides: ClassVar[tuple[str, ...]] = ("my_scheme",)

    def __init__(self, params=None):
        self.params = nnx.Param(params or MySchemeParameters.default())

    def __call__(self, state, diagnostics, forcing, terrain):
        ...
        return tendency, {**diagnostics, "my_scheme": MyData(...)}
```

```python
# user code
from jcm.physics.echam.echam_terms import echam_physics
from my_package.my_scheme import MyScheme

physics = echam_physics().replace("convection", MyScheme())
```

`ComposablePhysics` validates the new term list at construction time:
every `requires` has an upstream `provides`; no key is provided
twice; `required_tracers()` are seeded into the initial state by
`Model`. No edits to the model orchestrator, no edits to the
package's parameters aggregator, no edits to a monolithic data
struct.

## Directory layout

Physics is organised by **physical process**, with files named after
the **scheme** rather than the model they were ported from. New ports
of the same scheme drop in beside the existing one without an extra
per-model subfolder:

```text
jcm/physics/
├── physics_term.py             # PhysicsTerm base class
├── composable_physics.py       # ComposablePhysics container
├── radiation/
│   ├── grey_two_stream/        # fast grey two-stream package
│   ├── rrtmgp.py               # RRTMGP wrapper
│   ├── nn_emulator.py          # NN radiation emulator
│   ├── speedy_shortwave.py
│   └── speedy_longwave.py
├── convection/
│   ├── tiedtke_nordeng/        # Tiedtke-Nordeng mass flux
│   └── speedy_convection.py
├── clouds/
│   ├── sundqvist.py            # Sundqvist diagnostic cloud fraction
│   ├── echam_1m.py             # ECHAM 1-moment microphysics
│   ├── speedy_humidity.py
│   └── speedy_condensation.py
├── vertical_diffusion/
│   ├── tte_tke/                # TTE-TKE closure
│   └── speedy_vdiff.py
├── gravity_waves/{hines,sso,simple}/
├── aerosol/macv2_sp.py         # Stevens MACv2-SP simple plumes
├── chemistry/simple_chemistry.py
├── surface/                    # SPEEDY and ECHAM surface schemes
├── speedy/                     # SPEEDY infrastructure (params, coords)
└── echam/                      # ECHAM infrastructure (params, coords)
```

Model-specific *infrastructure* (parameter containers, coordinate
caches, data structs) lives under `speedy/` and `echam/`. Everything
else is named after the scheme so an ECHAM port and a CAM port of
the same parameterisation sit side-by-side.
