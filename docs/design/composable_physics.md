# Composable Physics Design

**Status:** Implemented (Phases 1–4 complete). Per PR #429 review the legacy
`SpeedyPhysics` / `IconPhysics` orchestrator classes were removed and the
process directories were flattened to scheme-named files (e.g.
`convection/tiedtke_nordeng/`, `clouds/sundqvist.py`); see "Post-PR-#429
state" at the bottom of this document. The historical narrative below
describes the design as it landed in PR #429 itself.

**Issue:** [#206 — Make physics composable](https://github.com/climate-analytics-lab/jax-gcm/issues/206)
**Branch:** `feature/composable-physics-206`
**Related issues:** #10, #230, #293, #315, #355

## Motivation

Issue #206 proposes making physics modules composable so that users can mix and
match parameterizations — for example, running SPEEDY's convection and surface
scheme with a more comprehensive radiation scheme like RRTMGP (analogous to the
MiMA model). This capability is a strong motivator for ongoing ICON physics
work, and was raised most sharply in the context of wanting to pull individual
ICON parameterizations into SPEEDY incrementally rather than merging all of
ICON at once.

The discussion on #206 converged on **Option 1** from duncanwp's enumeration:
> Make the list of terms a fundamental part of the Physics class such that you
> can literally add schemes together: `RRTMGP+Convection+ShallowClouds` and get
> a Physics object with each of those components in one list of terms.

This document describes a concrete design for Option 1 that:

1. Preserves numerical equivalence with the existing `SpeedyPhysics` and
   `IconPhysics` classes on `feature/icon-physics-v1`.
2. Preserves end-to-end differentiability of physics data outputs with respect
   to physics parameters at the `Model` level — a non-negotiable requirement
   for gradient-based calibration and hybrid physics-ML workflows.
3. Respects the ordering dependencies between parameterizations that duncanwp
   highlighted (e.g., SPEEDY's
   `shortwave → downward_longwave → surface_flux → upward_longwave` chain).
4. Enables per-scheme addressing for partial optimization, freezing, and
   neural-network replacement of individual terms.

## Current State (on `feature/icon-physics-v1`)

Both `SpeedyPhysics` and `IconPhysics` already share the same core pattern:

- A `Physics` base class with a `compute_tendencies` method.
- An ordered list of callable **terms** (`self.terms`) iterated in
  `compute_tendencies`.
- Every term has the same signature:
  `(PhysicsState, PhysicsData, Parameters, ForcingData, TerrainData)` →
  `(PhysicsTendency, PhysicsData)`.
- Tendencies are summed; `PhysicsData` is threaded through the term list as
  mutable intermediate state so that downstream terms can read upstream
  results (e.g., surface fluxes consume radiation fluxes computed earlier in
  the list).

The blocker for composability is that `PhysicsData` and `Parameters` are
**physics-package-specific** typed structs. SPEEDY's terms expect SPEEDY's
`PhysicsData` / `Parameters`; ICON's terms expect ICON's. Dropping an ICON
radiation term into SPEEDY's term list is impossible today because the
intermediate data struct types are incompatible.

## Proposed Architecture

The refactor replaces two things:

| Today | Proposed |
|---|---|
| Typed, physics-specific `PhysicsData` struct | `diagnostics: dict[str, jnp.ndarray]` — a flexible pytree keyed by well-known strings |
| Monolithic per-package `Parameters` struct held as a `self.parameters` attribute | Per-term parameters stored as `nnx.Param` attributes on each `PhysicsTerm` |

Everything else — `PhysicsState`, `PhysicsTendency`, the `Physics` base class
interface consumed by `Model` and `get_physical_tendencies` — stays the same.

### The `diagnostics` dict

Terms communicate through a `dict[str, jnp.ndarray]` (or pytree-of-arrays) that
flows forward through the term list. This replaces the physics-specific
`PhysicsData` struct. Each term reads the keys it needs and returns a new dict
with any new keys it produces (functional update, no in-place mutation).

Values must be JAX-compatible pytree leaves so that gradients propagate
through them. This is enforced by convention and exercised by Phase 2b's
gradient tests.

A `jit`-compile-cache consequence: two physics configurations that produce
different diagnostic-key sets will compile separately. This matches current
behavior (swapping `SpeedyPhysics` for `IconPhysics` already recompiles) and is
fine in practice — teams running a fixed physics stack get one compilation per
stack.

### Differentiability via `flax.nnx`

This design must preserve the existing ability to compute gradients of model
outputs with respect to physics parameters at the `Model` level, both across
all terms simultaneously and for individual schemes. We achieve this by
building `PhysicsTerm` and `ComposablePhysics` as `flax.nnx.Module` subclasses.

`flax.nnx` is the right fit because:

- `flax` is already in `requirements.txt` and the codebase imports `nnx`.
- It is explicit about trainable vs. non-trainable state — `nnx.Param` marks
  parameters that `nnx.grad` will differentiate through, `nnx.Variable` marks
  traced-but-frozen state (such as cached coordinate precomputes), and plain
  Python attributes remain static.
- It embraces in-place mutation, so the existing `cache_coords(self, coords)`
  pattern — which mutates `self.cached_coords` — continues to work.
- Nested `nnx.Module` instances stored in a plain list attribute are traversed
  automatically during `nnx.split` / `nnx.merge`, so `ComposablePhysics.terms`
  is naturally a pytree without explicit registration.
- `nnx.split(module) → (graphdef, state)` gives a pure-JAX interop path for
  code that uses `jax.grad` / `jax.jit` directly rather than the `nnx.*`
  transform variants.

### Phase 1 — `PhysicsTerm` protocol

```python
# jcm/physics/physics_term.py
from typing import ClassVar
from flax import nnx
import jax.numpy as jnp
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData

class PhysicsTerm(nnx.Module):
    """Base class for a composable physics parameterization.

    Subclasses must:
      - Declare ``name``, ``category``, ``requires``, ``provides`` as
        ``ClassVar`` — static metadata, not pytree leaves.
      - Store tunable parameters as ``nnx.Param`` attributes so that
        gradients flow through them.
      - Store coordinate-dependent caches as ``nnx.Variable`` attributes
        (traced but not trainable by default).
      - Implement ``__call__`` and ``cache_coords``.
    """

    name: ClassVar[str]
    category: ClassVar[str]           # "radiation", "convection", "surface", ...
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def cache_coords(self, coords) -> None:
        """Populate coordinate-dependent cached state. In-place."""

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict[str, jnp.ndarray],
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict[str, jnp.ndarray]]:
        raise NotImplementedError
```

A concrete term looks like:

```python
# jcm/physics/radiation/rrtmgp_radiation.py
class RRTMGPRadiation(PhysicsTerm):
    name: ClassVar[str] = "rrtmgp_radiation"
    category: ClassVar[str] = "radiation"
    requires: ClassVar[tuple[str, ...]] = ("pressure_full", "cloud_fraction")
    provides: ClassVar[tuple[str, ...]] = (
        "sw_flux_down", "lw_flux_down", "sw_heating_rate", "lw_heating_rate",
    )

    def __init__(self, params: RadiationParams, *, rngs: nnx.Rngs | None = None):
        # nnx.Param → differentiable leaf
        self.params = nnx.Param(params)
        # nnx.Variable → traced-but-frozen; populated by cache_coords
        self.cached_coords: nnx.Variable | None = None

    def cache_coords(self, coords):
        self.cached_coords = nnx.Variable(
            RRTMGPCoords.from_coords(coords)
        )

    def __call__(self, state, diagnostics, forcing, terrain):
        p = self.params.value
        cc = self.cached_coords.value
        sw_flux, lw_flux, heating = _rrtmgp_kernel(state, p, cc, ...)
        new_diagnostics = {
            **diagnostics,
            "sw_flux_down": sw_flux,
            "lw_flux_down": lw_flux,
            "sw_heating_rate": heating,
        }
        tendency = PhysicsTendency.zeros(state.temperature.shape).copy(
            temperature=heating,
        )
        return tendency, new_diagnostics
```

### Phase 2 — `ComposablePhysics`

```python
# jcm/physics/composable_physics.py
from flax import nnx
from jcm.physics_interface import Physics, PhysicsTendency

class ComposablePhysics(nnx.Module, Physics):
    def __init__(self, terms: list[PhysicsTerm]):
        self.terms = terms              # nnx traverses nested modules in lists
        self._validate_ordering()       # requires/provides DAG check — static

    def cache_coords(self, coords):
        for term in self.terms:
            term.cache_coords(coords)

    def compute_tendencies(
        self, state, forcing, terrain, date, prev_physics_data=None,
    ):
        diagnostics: dict[str, jnp.ndarray] = {}
        if prev_physics_data is not None:
            diagnostics = {**prev_physics_data}  # carry forward cached fields

        tendencies = PhysicsTendency.zeros(state.temperature.shape)
        for term in self.terms:
            tend, diagnostics = term(state, diagnostics, forcing, terrain)
            tendencies += tend
        return tendencies, diagnostics

    # Composition operators — each returns a fresh ComposablePhysics
    def __add__(self, other: "ComposablePhysics") -> "ComposablePhysics":
        return ComposablePhysics(terms=list(self.terms) + list(other.terms))

    def replace(self, category: str, new_term: PhysicsTerm) -> "ComposablePhysics":
        return ComposablePhysics(terms=[
            new_term if t.category == category else t for t in self.terms
        ])

    def remove(self, category: str) -> "ComposablePhysics":
        return ComposablePhysics(terms=[
            t for t in self.terms if t.category != category
        ])
```

The motivating use case — SPEEDY plus RRTMGP — becomes:

```python
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.physics.radiation import RRTMGPRadiation

physics = speedy_physics().replace("radiation", RRTMGPRadiation())
```

And literal addition matches duncanwp's Option 1 phrasing directly:

```python
physics = RRTMGPRadiation() + SpeedyConvection() + ShallowClouds()
```

### Differentiation patterns enabled by this design

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

**Pattern 3 — per-scheme optimization (strictly better than today):**

```python
# Optimize only convection parameters
convection_filter = nnx.PathContains("convection")
grads = nnx.grad(loss_fn, wrt=convection_filter)(physics)

# Address a single term's params directly
opt = optax.adam(1e-3)
opt_state = opt.init(physics.terms[3].params)
```

This is the largest ergonomic improvement over the current SPEEDY pattern: no
surgery on a monolithic `Parameters` struct is needed to optimize one scheme in
isolation.

### `cache_coords` lifecycle

`cache_coords` is called **at `Model` construction time**, outside any jitted
or traced region. Inside `cache_coords`, a term stores its precomputed data as
`nnx.Variable` (traced-but-frozen). During `model.run()` those variables are
read as traced values but never change. This matches the current SPEEDY/ICON
pattern exactly — we are formalizing it under `nnx`'s state system rather than
relying on untracked Python attributes.

Users who want to differentiate w.r.t. the coordinate system (rare, but
possible for learnable vertical-level placement) can do so because cached
coords are live pytree leaves. The default is that `nnx.grad` ignores
`nnx.Variable` and only differentiates `nnx.Param`.

### Phase 3 — wrap existing terms

Wrap the existing SPEEDY and ICON term functions as `PhysicsTerm` subclasses.
This is mostly mechanical: each existing function keeps its numerical
implementation, with a thin wrapper that translates between the old typed
`PhysicsData` struct fields and the new `diagnostics` dict keys.

Priority order for wrapping, driven by the primary use cases raised on #206:

1. **Radiation.** SPEEDY SW/LW, ICON grey, and RRTMGP. This unlocks the
   "SPEEDY + RRTMGP" configuration.
2. **Convection.** SPEEDY's simplified Tiedtke and ICON's Tiedtke-Nordeng, plus
   Betts-Miller from #315.
3. **Surface fluxes.** Shared fundamentals between SPEEDY and ICON.
4. **Vertical diffusion, clouds, microphysics.** Fill in the remaining ICON
   schemes.

Existing monolithic classes `SpeedyPhysics` and `IconPhysics` remain functional
during the migration. Once all terms are wrapped, they become factory functions
that return a `ComposablePhysics`:

```python
# jcm/physics/packages/speedy.py
def speedy_physics(parameters=None) -> ComposablePhysics:
    p = parameters or SpeedyParameters.default()
    return ComposablePhysics(terms=[
        SpeedyFlags(p.flags),
        SpeedyForcing(p.forcing),
        SpecificToRelativeHumidity(),
        SpeedyConvection(p.convection),
        SpeedyLargeScaleCondensation(p.condensation),
        SpeedyClouds(p.clouds),
        SpeedyShortwaveRadiation(p.sw_radiation),
        SpeedyDownwardLongwaveRadiation(p.lw_radiation),
        SpeedySurfaceFlux(p.surface),
        SpeedyUpwardLongwaveRadiation(p.lw_radiation),
        SpeedyVerticalDiffusion(p.vertical_diffusion),
    ])
```

The ordering constraint duncanwp raised on #206 — that surface fluxes must
fall between downward and upward longwave — is preserved by encoding it in
`speedy_physics`'s default list. Users can still reorder or swap terms, but
they accept responsibility for compatibility. The `requires`/`provides`
metadata lets `_validate_ordering` catch many invalid configurations at
construction time (e.g., a term that requires a key no upstream term
provides).

### Phase 4 — reorganize directories (implemented)

The `jcm/physics/` directory has been reorganized from organization-by-package
to organization-by-process. SPEEDY and ICON implementations of the same physical
process now live side-by-side:

```
jcm/physics/
├── physics_term.py              # PhysicsTerm base class
├── composable_physics.py        # ComposablePhysics container
├── speedy/                      # SPEEDY infrastructure only
│   ├── speedy_physics.py        # Legacy orchestrator
│   ├── speedy_terms.py          # Composable wrappers + speedy_physics() factory
│   ├── params.py, speedy_coords.py, physics_data.py, physical_constants.py
├── icon/                        # ICON infrastructure only
│   ├── icon_physics.py          # Legacy orchestrator
│   ├── icon_terms.py            # Composable wrappers + icon_physics() factory
│   ├── parameters.py, icon_coords.py, icon_physics_data.py, constants/
├── radiation/
│   ├── speedy_shortwave.py      # SPEEDY shortwave radiation
│   ├── speedy_longwave.py       # SPEEDY longwave radiation
│   └── icon/                    # ICON radiation sub-package (grey, RRTMGP, NN emulator)
├── convection/
│   ├── speedy_convection.py
│   └── icon/                    # ICON Tiedtke-Nordeng convection
├── clouds/
│   ├── speedy_humidity.py       # SPEEDY moisture conversion
│   ├── speedy_condensation.py   # SPEEDY large-scale condensation
│   └── icon/                    # ICON cloud diagnostics + microphysics
├── surface/
│   ├── speedy_surface_flux.py
│   └── icon/                    # ICON multi-surface tile scheme
├── vertical_diffusion/
│   ├── speedy_vdiff.py
│   └── icon/                    # ICON TKE-based diffusion
├── gravity_waves/icon/          # ICON gravity wave drag
├── aerosol/icon/                # ICON MACv2-SP aerosol
├── chemistry/icon/              # ICON simple chemistry
├── diagnostics/icon/            # ICON diagnostics (WMO tropopause)
├── forcing/speedy_forcing.py    # SPEEDY forcing/boundary conditions
├── orographic_correction/speedy_orographic.py
├── packages/                    # Pre-built physics factories
│   ├── speedy.py                # Re-exports speedy_physics()
│   └── icon.py                  # Re-exports icon_physics()
└── held_suarez/                 # Stays as-is (intentionally simple)
```

**Key design decisions for the reorganization:**

1. **Infrastructure stays in place.** `speedy/` and `icon/` retain params, coords,
   data structs, constants, and orchestrators. This preserves ~60 external references
   to `get_speedy_coords`, `SpeedyPhysics`, `Parameters`, etc.
2. **ICON sub-packages move as directories** (e.g., `icon/radiation/` → `radiation/icon/`),
   preserving internal same-directory relative imports.
3. **SPEEDY modules move as individual files** with renaming
   (e.g., `speedy/convection.py` → `convection/speedy_convection.py`).
4. **All ICON `from ..` relative imports converted to absolute** to avoid breakage
   when parent directories change.
5. **`icon/__init__.py` uses lazy `__getattr__`** to break circular import chains
   that arise from the reorganization.

## Phase Table

| Phase | Status | Notes |
|---|---|---|
| 1. `PhysicsTerm` as `nnx.Module`, `diagnostics` dict convention | **Complete** | `physics_term.py` |
| 2. `ComposablePhysics` with `__add__` / `replace` / `remove` | **Complete** | `composable_physics.py` |
| 2b. **Differentiability gate** — bit-identical gradient check | **Complete** | `composable_physics_test.py` |
| 3. Wrap existing SPEEDY and ICON terms | **Complete** | `speedy_terms.py`, `icon_terms.py` |
| 4. Reorganize directories by process | **Complete** | 18 commits, 603 tests pass |

## What stays the same

- `PhysicsState` and `PhysicsTendency` — unchanged. Still the primary
  input/output for every term.
- The `Physics` base class interface (`compute_tendencies`, `cache_coords`,
  `get_empty_data`) — `ComposablePhysics` implements it, so `Model` and
  `get_physical_tendencies` need no changes.
- `SpeedyPhysics` and `IconPhysics` remain valid entry points during the
  migration and are re-implemented as factory functions once Phase 3 lands.
- The ordering-dependency problem duncanwp raised is unchanged but unblocked:
  users specify order, pre-built packages encode sensible defaults, and
  `requires`/`provides` metadata catches invalid orderings at construction
  time rather than at runtime.
- `held_suarez/` stays as a hand-written simple physics package — the
  composable machinery is not forced on physics that do not need it.

## Composability Considerations

### Ordering dependencies

Physics parameterizations have implicit ordering dependencies — downstream
terms read diagnostic fields produced by upstream terms. For example, in SPEEDY
the surface flux scheme reads radiation fluxes, so radiation must run first.

The composable design encodes these dependencies in two ways:

1. **Pre-built factories** (`speedy_physics()`, `icon_physics()`) encode
   validated orderings as their default term lists. Users who don't need
   custom configurations get correct orderings for free.

2. **`requires` / `provides` metadata** on `PhysicsTerm` allows
   `ComposablePhysics._validate_ordering()` to catch many invalid
   configurations at construction time (e.g., a term that requires a key
   no upstream term provides).

When users `replace()` a term, the category-based replacement preserves the
original position in the list, maintaining ordering. When users compose from
scratch (e.g., `term_a + term_b + term_c`), they accept responsibility for
ordering correctness.

### Cross-package compatibility

SPEEDY and ICON terms use different intermediate data representations
(`PhysicsData` structs). The composable design handles this through the
`diagnostics` dict — a flexible `dict[str, jnp.ndarray]` that replaces
the typed structs for inter-term communication. Each term's wrapper
translates between its package's typed structs and the dict:

- SPEEDY wrappers (`speedy_terms.py`) store SPEEDY `PhysicsData` sub-structs
  under keys like `"_shortwave_rad"`, `"_convection"`, etc.
- ICON wrappers (`icon_terms.py`) store ICON `PhysicsData` sub-structs
  under keys like `"_radiation"`, `"_convection"`, etc.

This means mixing a SPEEDY and ICON term that need to share data
(e.g., radiation fluxes) requires either:
- A translation layer that maps between the key conventions, or
- The replacement term independently computing what it needs from
  the atmospheric state.

In practice, most cross-package use cases replace an entire process category
(e.g., swapping SPEEDY radiation for ICON RRTMGP), where the replacement
term produces everything downstream terms need from the state alone.

### Differentiability

Each `PhysicsTerm` stores tunable parameters as `nnx.Param` attributes and
coordinate caches as `nnx.Variable`. This enables:

- **Per-scheme gradients**: Optimize one scheme's parameters while freezing
  others, using `nnx.grad` with path-based filtering.
- **Neural network replacement**: An `nnx.Module`-based ML term slots into
  the composable list and automatically participates in gradient computation.
- **End-to-end differentiability**: Gradients flow through the `diagnostics`
  dict since all values are JAX arrays.

The gradient path through `diagnostics` is exercised by the differentiability
tests in `composable_physics_test.py`, which verify bit-identical gradients
between the composable and legacy physics implementations.

### ICON column vectorization

ICON terms operate in column-vectorized format `(nlev, ncols)` rather than
3D grid format `(nlev, nlon, nlat)`. The `ComposableIconPhysics` subclass
handles this reshaping transparently — it reshapes the state to columns before
iterating terms and reshapes accumulated tendencies back to 3D afterward.
This matches the optimized pattern from the original `IconPhysics` class.

### Backward compatibility

The legacy `SpeedyPhysics` and `IconPhysics` classes remain functional and
unchanged. Existing code that uses `SpeedyPhysics(parameters=...)` or
`IconPhysics(parameters=...)` continues to work. The composable API is
opt-in via `speedy_physics()` and `icon_physics()` factory functions.

The directory reorganization moved process-specific modules but kept all
infrastructure (params, coords, orchestrators) in their original locations.
All external imports from `jcm.physics.speedy.*` and `jcm.physics.icon.*`
infrastructure modules are unchanged.

## Resolved questions

1. **Parameter grouping.** Terms hold multiple named `nnx.Param` attributes
   (e.g., `SpeedyShortwaveRadiation` has `sw_params` and `mod_radcon_params`).
   This gives users fine-grained control over which parameters to optimize.

2. **Validation strictness.** `_validate_ordering` currently performs a
   soft check. Terms with empty `requires`/`provides` are always accepted.
   Strict validation is opt-in and may be tightened in future.

3. **Back-compat shim.** Both `SpeedyPhysics` and `IconPhysics` remain
   as standalone classes (not shims). The composable path is a parallel API,
   not a replacement.


---

## Post-PR-#429 state

After review on PR #429 (`composable-physics-206`), the design above was
tightened in three ways:

1. **Removed legacy orchestrator classes** (`SpeedyPhysics`, `IconPhysics`,
   `HeldSuarezPhysics`). The composable path is no longer "parallel" — it
   is the only physics API. Users must instantiate via the
   `speedy_physics()`, `icon_physics()`, or `held_suarez_physics()`
   factories. This is a breaking change appropriate to a major version.

2. **Removed `jcm/physics/packages/`.** Factory functions live alongside
   their schemes in `speedy_terms.py` / `icon_terms.py`; the extra
   re-export layer added no value.

3. **Flattened process directories to scheme-named files.** Files are
   named for what they *are* (the scheme) rather than where they were
   ported from. Examples: `convection/tiedtke_nordeng/`,
   `clouds/sundqvist.py`, `clouds/echam_1m.py`, `aerosol/macv2_sp.py`,
   `radiation/grey_two_stream/`, `radiation/rrtmgp.py`,
   `vertical_diffusion/tte_tke/`, `gravity_waves/hines/`. New ports of the
   same scheme from a different model (e.g. CAM Tiedtke) drop in beside the
   existing one without an extra `cam/` subfolder. Model-specific
   *infrastructure* (parameter containers, coordinate caches, data
   structs) stays under `speedy/` and `icon/`.

The "diagnostics" dict threaded through the term list serves a dual role:
keys without a leading underscore are exposed as user-facing diagnostic
output (written to xarray); keys prefixed with `_` (e.g. `_radiation`,
`_convection`, `_chemistry`) are internal inter-term state and are
filtered out of the user-facing output by `data_struct_to_dict`.

