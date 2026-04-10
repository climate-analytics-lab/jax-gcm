# Composable Physics Design

**Status:** Proposal
**Issue:** [#206 — Make physics composable](https://github.com/climate-analytics-lab/jax-gcm/issues/206)
**Target branch:** `feature/icon-physics-v1`
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
from jcm.physics.packages import speedy_physics
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

### Phase 4 — reorganize directories

After Phases 1–3 are proven with real mixed-package runs, reorganize the
`jcm/physics/` directory from organization-by-package to organization-by-process,
as proposed by AndrewILWilliams on #206:

```
jcm/physics/
├── physics_term.py              # PhysicsTerm protocol
├── composable_physics.py        # ComposablePhysics class
├── radiation/
│   ├── speedy_radiation.py
│   ├── grey_radiation.py
│   └── rrtmgp_radiation.py
├── convection/
│   ├── speedy_convection.py
│   ├── tiedtke_nordeng.py
│   └── betts_miller.py
├── clouds/
│   ├── speedy_clouds.py
│   └── cloud_microphysics.py
├── surface/
│   ├── speedy_surface.py
│   └── icon_surface/
├── vertical_diffusion/
│   ├── speedy_vdiff.py
│   └── icon_vdiff/
├── packages/                    # Pre-built sensible defaults
│   ├── speedy.py
│   ├── icon.py
│   └── speedy_rrtmgp.py
└── held_suarez/                 # Stays as-is (intentionally simple)
```

This phase is a large diff with many import changes and should land after the
ICON branch has merged to `main`.

## Phase Table

| Phase | Target branch | Depends on | Risk |
|---|---|---|---|
| 1. `PhysicsTerm` as `nnx.Module`, `diagnostics` dict convention | `feature/icon-physics-v1` | — | Low — additive |
| 2. `ComposablePhysics` with `__add__` / `replace` / `remove` | `feature/icon-physics-v1` | Phase 1 | Low |
| 2b. **Differentiability gate** — verify `nnx.grad` flows through all terms; bit-identical gradient check against current `SpeedyPhysics` on a toy configuration | `feature/icon-physics-v1` | Phase 2 | **Gating** |
| 3. Wrap existing SPEEDY and ICON terms | `feature/icon-physics-v1` | Phase 2b | Medium — must preserve numerical equivalence |
| 4. Reorganize directories by process | After ICON merges to `main` | Phase 3 | High — large diff |

Phase 2b is explicitly gating: no term-wrapping work begins until a toy
`ComposablePhysics` demonstrates bit-identical gradients to an equivalent
monolithic physics object. This catches pytree-registration bugs before they
are baked into a large migration.

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

## Open questions

1. **Parameter grouping.** Should a single `PhysicsTerm` subclass own one
   typed `nnx.Param` block, or should it be free to hold several named
   `nnx.Param` attributes? The examples above use one block; experience with
   real terms may argue for finer granularity so that users can optimize a
   sub-scalar in isolation.
2. **Validation strictness.** How strict should `_validate_ordering` be? A
   hard error on missing `requires` is clear, but tolerating optional keys
   (e.g., a term that can run with or without aerosol optical depth) may be
   worth supporting.
3. **Back-compat shim.** Should `SpeedyPhysics(parameters=...)` remain a class
   for one release after Phase 3, constructing a `ComposablePhysics` under the
   hood and exposing `self.parameters` as a view over the term params? This
   would keep existing notebooks running unmodified.
