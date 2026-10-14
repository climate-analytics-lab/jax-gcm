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
  diagnostic output. Flatten directly into `predictions.to_xarray()`
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

### Declaring a conditional (`.get()`/`in`) read

`requires` only covers a bare `diagnostics["key"]` subscript — an
UNCONDITIONAL read the term cannot run without. A term that reads a key
conditionally, via `diagnostics.get("key")` or an `if "key" in
diagnostics` guard, declares it in `optional_requires` instead. Two
distinct, both legitimate, patterns use this form:

1. **A deliberate cross-step read.** The reading term runs upstream, in
   the composition order, of the term that `provides` the key, so the
   value seen is the previous step's (one-step-lagged) — e.g.
   `SundqvistCloudFraction` reading `"convection"` before
   `TiedtkeConvection` runs, matching ECHAM's own `cover`-before-`cucall`
   sequencing.
2. **An optional cross-composition dependency.** The key is always
   fresh (same-step) when its producer IS composed, but the reading term
   can also run in a composition without that producer (a reduced
   stack, a unit-test harness), where the `.get` default takes over.

`requires_audit_test.py` AST-walks every term's `__call__` and fails if
a `.get()`/`in` read of `diagnostics` is not covered by `requires` |
`provides` | `carry_slots` | `requires_dycore_fields` | the framework
plumbing keys | `optional_requires`. Before this declaration existed
(jax-gcm#673), that convention's only enforcement was for *unconditional*
reads — a `.get()` was unconditionally exempt, so an accidental stale
read (reading a key nobody provides yet this step, with no fallback
reasoning behind it) was indistinguishable from the two legitimate
patterns above and nothing caught it. Declaring the key is cheap; the
comment next to the read explaining WHICH of the two patterns applies
(most terms' module or class docstring already does) is the part that
actually prevents the next one.

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

### Radiation-derived configuration: resolved lazily, once, by `cache_coords()`

Two things in a composition are derived from the active radiation term
rather than set on their own terms at construction:

- `band_config` — the SW/LW band centers wavelength-dependent terms (aerosol
  optics) read, via `RadiationBandConfig.for_terms`.
- The radiation cadence passed to every sibling exposing
  `configure_radiation_gate` (the JAM optics term): the interval between full
  radiation solves, read off the active radiation term's
  `RadiationParameters`, so a gated sibling skips its per-band recompute on
  exactly the steps radiation itself replays its cache.

**History — why this is no longer decided at composition-edit time.** Nine
separate Codex review rounds on jax-gcm#926 each found another composition-
editing path (`replace`, `remove`, `ComposablePhysics + ComposablePhysics`,
`PhysicsTerm + ComposablePhysics`/`ComposablePhysics + PhysicsTerm`,
`PhysicsTerm + PhysicsTerm` with either operand holding the radiation term,
both operand orders, a gated sibling already correctly configured on either
side of a `+`) that disagreed about whether to re-derive or preserve
`band_config`/the gate cadence, depending on operand order or which operator
ran. Each round's fix made the surviving code correct for one more case
while leaving the SHAPE of the problem intact: deciding "did something
radiation-relevant change" at every composition-edit call site, repeated
once per operator, is inherently exposed to exactly this class of bug — a
tenth case was still conceivable. The fix is not a tenth patch: it removes
the decision from composition-edit time ENTIRELY.

**The final design:**

1. `band_config=None` (the `ComposablePhysics` constructor's default) means
   "derive from the final term set" — nothing to inherit, preserve or
   re-derive at edit time, because nothing has been decided yet. A non-`None`
   value passed to the constructor is a genuine, hand-supplied override
   (typically for a custom radiation backend `RadiationBandConfig.for_terms`
   cannot itself recognize) and is kept VERBATIM by every operator — there is
   no "but radiation changed, so re-derive anyway" exception any more.
2. **Provenance travels separately from the value**, via
   `_band_config_explicit` (a plain `bool` attribute next to `band_config`,
   not a pytree leaf). `band_config is not None` is true BOTH for a genuine
   override AND for a value a PRIOR `_finalize` call already derived and
   left on `self.band_config` — a composition that has run `cache_coords()`
   even once always has a non-`None` value, override or not. Conflating the
   two let a composition's own derived value survive a LATER
   `replace`/`remove`/`+` as if it had been pinned: swapping RRTMGP for an
   unrecognized backend (grey) after a prior `cache_coords()` call carried
   RRTMGP's derived 14 SW/16 LW bands into the grey composition completely
   silently, because `_finalize`'s "already set AND recognized" validation
   branch (point 3 below) only fires for a *recognized* new backend — grey
   never reaches it, so the stale value was never touched (GitHub Codex
   review of PR #1096). The constructor infers `_band_config_explicit` from
   `band_config is not None` ONLY when nothing is passed for it explicitly —
   the right default for every ordinary, fresh construction, where there is
   no prior derivation to have left anything behind; `replace`, `remove` and
   both `__add__` implementations instead pass the flag through EXPLICITLY,
   decoupled from whatever the carried-over value happens to be.
3. `replace`, `remove`, `ComposablePhysics.__add__` and
   `PhysicsTerm.__add__` are PURE term-list operations: they build the new
   term list (`replace`/`remove`) or concatenate two (`__add__`), and
   nothing else. The only thing `__add__` itself still decides — because it
   is inherent to combining exactly two operands and belongs nowhere else —
   is which operand's CURRENT `(band_config, _band_config_explicit)` pair
   the result carries forward; see `_add_band_config` below for the rule,
   which (also per GitHub Codex review of PR #1096) does not look at which
   operand owns the active radiation term AT ALL — only at which side(s), if
   either, are explicit. Ownership (never aliasing a term with another live
   composition) is the only other thing these operators enforce — see the
   next section.
4. `ComposablePhysics._finalize` — called by `cache_coords()`, before
   anything else — is the ONE place `band_config` is actually resolved and
   validated, and gated siblings are actually configured. It runs every
   time `cache_coords()` is called, deterministically, from whatever the
   FINAL term set and `(band_config, _band_config_explicit)` pair happen to
   be at that moment — not incrementally, not conditioned on what changed
   since some earlier call. Concretely: raise if more than one
   `"radiation"`-category term is present (ambiguous — nothing says which is
   active); if `not self._band_config_explicit` — covering BOTH `None`
   (pending) and a value some earlier `_finalize` call derived — derive (or
   re-derive) it via `RadiationBandConfig.for_terms`, unconditionally, from
   the term set as it stands NOW; only a genuinely EXPLICIT value, and only
   when the active radiation term is a RECOGNIZED backend (RRTMGP/the NN
   emulator — `RadiationBandConfig.is_recognized`), is validated against
   what that backend actually provides instead of being re-derived, raising
   with a clear message if they disagree (this is what catches a *deliberate*
   override that no longer matches a recognized backend swapped in after it
   was set — silently keeping a wrong value, or silently discarding a
   deliberate override, are both worse than a loud error; an UNRECOGNIZED
   radiation term is never checked this way — `for_terms` has no opinion on
   it, which is the entire reason an explicit override exists); then
   configure every gated sibling's cadence from the active radiation term's
   `RadiationParameters`, raising if one is present with no radiation term
   to read from. The gated-sibling cadence needs no analogous provenance
   flag: it is never stored as a value with a None/non-None sentinel in the
   first place — `_finalize` reconfigures every gated sibling directly from
   whichever radiation term is active, unconditionally, on every call, so
   there is nothing for a stale value to hide behind.
5. `diagnostics["_band_config"]` is still populated from `self.band_config`
   at the top of every `compute_tendencies` call (unchanged — every host
   that steps a `ComposablePhysics` already calls `cache_coords()` first,
   so this per-step, JIT-traced read never has to search `self.terms` for
   the active radiation term itself).

Because neither `echam_physics()` nor `rce_physics()` nor the generic Hydra
runner path pre-compute `band_config` any more (each used to call
`RadiationBandConfig.for_terms(...)` itself and pass the result straight to
`ComposablePhysics(...)`, which is now indistinguishable from a genuine
override and would have survived a later `replace("radiation", ...)`
verbatim), every one of them leaves it `None` and relies on `_finalize` to
derive the identical value — the factory computing it itself bought nothing
once `_finalize` exists, and the one case it got WRONG (a factory-built
composition whose radiation term is later swapped by `replace`) is exactly
what `test_replace_radiation_matches_the_factory_route` in
`jcm/physics/echam/echam_terms_test.py` pins.

### Ownership: the one invariant the operators themselves enforce

`replace`, `remove` and both `__add__` implementations each build a NEW
`ComposablePhysics` while the operand(s) they were called on remain
independently live and usable — nothing about calling `physics.replace(...)`
or `a + b` destroys `physics`, `a` or `b`. A term carried over BY REFERENCE
into the new composition would therefore be shared between it and whichever
operand it came from, and a later `cache_coords()` call on EITHER one could
silently rewrite the gated-sibling cadence or `cache_band_config`-cached
carry dimensions the other still expects — the exact hazard every one of the
nine Codex rounds above was really circling, just expressed through
`band_config` specifically.

The fix is uniform and type-agnostic: the COMPLETE combined/surviving term
collection is `nnx.clone`-d as ONE GRAPH, via the shared helper
`_clone_composition_terms(terms) -> list(nnx.clone(list(terms)))`, before
joining a NEW composition — never term-by-term. `nnx.clone`'s graph mode
(the default) clones every distinct object it reaches exactly once and
rewires every reference to the SAME new object, so two terms that
intentionally share an `nnx.Variable` or submodule (a tied-parameter model)
stay tied in the result; cloning each term SEPARATELY — `[nnx.clone(t) for
t in terms]`, an earlier version of this fix — treats each as its own
independent graph root instead, so it silently UNTIES such a pair even on a
complete no-op edit (`remove("missing")`), a real correctness defect for a
differentiable model (GitHub Codex review of PR #1096). `nnx.clone(node)`'s
default (`variables=True`) also gives each clone its OWN `nnx.Variable`
wrapper objects without copying the underlying JAX array *buffers* — JAX
arrays are immutable, so a fresh wrapper around the same array is
sufficient and cheap. That is enough to isolate every current
`cache_band_config`/`configure_radiation_gate` implementation, since all of
them only ever REASSIGN a plain scalar/array attribute
(`self._n_bnd_sw = ...`, `self._radiation_interval_s = ...`) rather than
mutating a shared container in place — it does **not** deep-copy a plain
Python list/dict attribute `.append()`-ed or mutated in place instead of
reassigned (pinned directly by
`test_clone_does_not_deep_copy_plain_mutable_attributes`), which is a scope
boundary a future term with such an attribute needs to know about, not a gap
in today's coverage.

- `replace(category, new_term)` builds the combined RAW (uncloned) list
  first — `new_term` plus every unchanged survivor — then clones the WHOLE
  thing in ONE `_clone_composition_terms` call, and only THEN runs
  `adopt_runtime_configuration` on the resulting clone of `new_term`: never
  on the caller's own object (which stays untouched and reusable elsewhere),
  and never pre-severing a tie `new_term` might share with a survivor by
  cloning it separately first.
- `remove(category)` clones every surviving term together, for the same
  ownership reason.
- `ComposablePhysics.__add__`/`PhysicsTerm.__add__` clone `self`'s terms AND
  every term from `other` TOGETHER, in one call — `self` remains live
  exactly like `other` does, and a tie between a term of `self`'s and a term
  of `other`'s (not just within one operand's own terms) survives too.

This is what `test_operators_preserve_tied_parameters_across_terms` (two toy
terms sharing one `nnx.Param`, exercised across `replace`, a true no-op
`remove`, and all three `+` shapes) pins directly;
`test_add_never_aliases_a_band_consumer_with_the_operand_it_came_from` and
`test_replace_radiation_does_not_corrupt_a_shared_band_consumers_cache`
(both in `jcm/physics/composable_physics_test.py`) cover the narrower
"never aliased with the SOURCE composition" half of the same invariant, and
`test_matches_from_scratch_across_every_composition_path` (the `#926`
matrix) exercises the full pipeline across all composition-editing shapes
with no per-operator branching left to get wrong.

**Known gap, tracked separately:** this cloning only happens inside
`replace`/`remove`/`__add__` — the raw `ComposablePhysics(terms=[...])`
constructor stores its `terms` argument by reference, so two separate raw
constructions that happen to share the SAME term instance reproduce the
exact aliasing this section describes. No in-tree factory does this (each
builds a disjoint, freshly-instantiated term list), so it is a theoretical
risk for today's code, not a live bug; jax-gcm#1106 tracks closing it (the
fix needs its own audit of the test suite's existing convention of handing
a pre-configured term into raw construction and asserting identity against
it afterward, which cloning in `__init__` would break).

### `_add_band_config`: ONE total merge rule, independent of where radiation sits

`replace`/`remove` always have exactly one composition, so there is nothing
to decide — `self.band_config` AND `self._band_config_explicit`, whatever
they are, carry straight through together (never one without the other —
see point 2 above on why the flag has to travel WITH the value). `+` has two
operands, so both `ComposablePhysics.__add__` and `PhysicsTerm.__add__`
first unpack `other` through a shared helper, `_unpack_add_operand(x) ->
(terms, band_config, band_config_explicit, settings, is_composition)` (a
`ComposablePhysics` or a bare `PhysicsTerm` are the only two valid shapes on
either side of `+`, and both methods need the identical unpacking — a bare
term's `band_config_explicit` is always `False`, since it never held a
genuine override of its own), then `_add_band_config(self_band_config,
self_explicit, other_band_config, other_explicit)` merges the two operands'
`(band_config, explicit)` pairs into the result's pair:

- **An EXPLICIT override on either side is kept.**
- **Both sides explicit and the values disagree:** ambiguous (nothing says
  which override should win) — raise immediately, naming both.
- **Neither side explicit:** the result is `(None, False)` — pending, for
  `_finalize` to derive from whatever radiation term ends up present in the
  FINAL combined term list. A non-explicit value (pending, or a value some
  prior `_finalize` call derived for a term set that may be completely
  different now) carries no information worth comparing or preserving, so
  it never survives an edit; only a genuine override does.

**Which operand (if either) owns the active `"radiation"`-category term
contributes NOTHING to this rule.** It only still decides something
downstream, in `_finalize`: which backend an explicit override gets
VALIDATED against (a *recognized* backend's real bands; an unrecognized one
keeps the override unvalidated, since the override is the only way to
describe such a backend's bands in the first place).

**Why the radiation-term check was removed, not patched a third time.** An
earlier version of this function branched on which operand's OWN terms
included a `"radiation"`-category one FIRST — that operand's pair won
outright — and only fell back to comparing explicit values when NEITHER
did. GitHub Codex review of PR #1096 found this was a THIRD case-by-case
special rule discovered on this exact merge: it silently dropped a GENUINE
explicit override sitting on the side with no radiation term whenever the
OTHER side held even a bare, custom radiation `PhysicsTerm` — a bare term
always reports `(None, False)`, so "whichever side owns radiation wins
outright" preferred that `(None, False)` over a real override on the other
side (`ComposablePhysics([consumer], band_config=custom) +
CustomRadiation()` silently went broadband at the next `cache_coords()`).
Each of the three rounds that found a variant of this bug patched the SHAPE
it found (first: which operand structurally owns radiation; second and
third: whether that owning side is a bare term), not the rule itself.
Removing the radiation-term check from this function ENTIRELY — rather than
special-casing a fourth shape — is what makes a fourth round impossible: no
branch here can ever again disagree about whether a value on one side is
real just because the other side happens to look a certain way.
`test_add_band_config_merge_rule_matrix` in
`jcm/physics/composable_physics_test.py` pins this directly: the SAME
`(band_config, explicit)` outcome for every `(left_state, right_state)` pair,
checked across every placement of the active radiation term (the left
operand's own terms, the right's, a bare term substituted for either
operand, or nowhere at all) — proving placement is a pure confound.

`checkpoint_terms`/`vectorize_columns`/`dt_seconds` go through the same
"operand order cannot matter" discipline via a separate helper,
`_resolve_add_settings`: whichever operand is an actual pre-existing
`ComposablePhysics` contributes these (a bare `PhysicsTerm` has none to
contribute); when BOTH operands are compositions, their settings must
already agree — a `ValueError`, not silently preferring one side, since
nothing says whether the left or right operand's checkpointing,
vectorization or timestep should win; when NEITHER is a composition
(`term_a + term_b`), the constructor's own defaults apply, exactly as
building the same composition any other way would.

Otherwise, users composing from scratch accept responsibility for ordering
correctness — `_validate_ordering` catches dependency bugs but cannot
enforce semantic order beyond the `requires` / `provides` graph.

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

## Coupling between terms

Within a `ComposablePhysics` step every term reads the same input
prognostic `state`, and the tendencies are summed and applied once by the
host. The state is not updated between terms. Terms run in list order and
see their predecessors through three channels:

- **`_tendency_run`**: the host publishes the sum of the tendencies of the
  terms already run (winds, temperature, humidity, every tracer) before
  each term, on both the grid and the column host.
- **`thermo_run`**: a running (T, q, qc, qi), seeded to the step start and
  advanced by the terms that call `advance_thermo_run` (vertical diffusion,
  the prescribed surface flux, Tiedtke, the cloud schemes).
- **the diagnostics dict itself**: published structs (`clouds`, which also
  carries the convective detrainment of the step, `convection`, …),
  step-local keys such as `_surface_optics`, and the cross-step carry
  (`_prev_step`, `_post_physics_state`).

So the total tendency does not depend on the order of two terms that use
none of these channels, but it does for the ECHAM stack, whose convection
reads vertical diffusion's `thermo_run` and whose cloud scheme reads the
running tendency, the detrainment and the carried post-physics state: the
`physc` order `echam_physics()` ships is load-bearing. What each hand-off
holds, who advances it and what the cloud schemes receive is set out in
[operator-split physics](operator_split_physics.md#coupling-within-physics).

A term that needs the previous step's post-physics state declares the
fields in `requires_post_physics_fields`; the container carries the slot
only when some term does, and `Model` fills it from
`DynamicalCore.after_physics_state` after every physics call. The slot is
invalid (its `valid` flag is 0) on a first step, after restoring a
checkpoint that predates it and on hosts without a dynamical core; a
reader must treat it as absent then.

For terms that need a tighter coupling than this (e.g. a CLUBB+MG2 pair as
in E3SM), the pattern is a *process group* term that runs the inner
sub-cycle internally and presents a single tendency externally.

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
