# Operator-split physics — implementation plan

**Status:** Phases 0-4 implemented (PR #472, with P1/P2 carry-threading and Phase 4 follow-up); Phase 5 Strang upgrade still to come.
**Issue:** [#471 — Operator-split physics from the IMEX-RK dynamical core](https://github.com/climate-analytics-lab/jax-gcm/issues/471)
**Related:**
- [#470 — Integration trajectory should not depend on output mode](https://github.com/climate-analytics-lab/jax-gcm/issues/470) (the architectural concern this refactor closes)
- [#469 — Mixing-length, Thomas-pivot, TKE-source, IMEX-substage cache fixes](https://github.com/climate-analytics-lab/jax-gcm/pull/469) (the immediate correctness fixes that surfaced #470/#471)
- [#468 — Grey two-stream day-16 NaN](https://github.com/climate-analytics-lab/jax-gcm/issues/468) (a downstream physics issue this should clarify)
- [#388 — Pluggable dycore (pysces)](https://github.com/climate-analytics-lab/jax-gcm/issues/388) (this refactor is essentially a prerequisite)

## Motivation

JCM currently couples physics to the dynamical core at the *RK-stage level*: `_step_tendencies` is wrapped in an `ExplicitODE` and composed with the primitive equations inside `imex_rk_sil3`. The SIL3 tableau has three explicit stages per timestep, so physics runs **three times per `dt`**, each call on a different intermediate Runge-Kutta state. The `DiagnosticsCollector` then threads a `physics_data_cache` through the dinosaur scan to give cross-step physics state (radiation flux, TKE, droplet number, ...) somewhere to live.

This is what we get in exchange:

- **Formal third-order accuracy on the explicit-physics terms.** This is the textbook reason to put physics inside the RK stages.

And here is what it costs:

- **~3× the physics work per `dt`.** Radiation is partly mitigated by `radiation_should_compute` sub-cycling (full optics fire every `radiation_interval`, cached values flow in between), but TTE-TKE, convection, microphysics, surface and aerosol all re-run on every stage.
- **An asymmetry between snapshot and averaged output modes.** Snapshot mode has no `DiagnosticsCollector`, so `prev_physics_data = None` always. Averaged mode threads it through the scan. The integration *trajectory* is therefore not the same in the two modes — bit-exact in arithmetic, observably different in long-time behaviour. #470 calls this out.
- **A substage-disambiguation problem.** Within a single timestep, multiple RK substages each read and write the cache. PR #469 had to gate the cache *write* on a `physical_step` flag so later substages don't read their own intermediate output back as "previous-step" cache. The fix is correct, but the bug itself shouldn't have been possible.
- **Coupling to dinosaur's IMEX-RK at a deep level.** Swapping in a different dycore (#388: pysces, a spectral-element FE scheme) requires re-routing physics through whatever stage abstraction the new dycore uses, plus rewiring the `DiagnosticsCollector` carry.

The proposed refactor is the standard operational-GCM approach used by ECHAM, CAM, IFS, and E3SM: **operator-split physics from dynamics at the `dt` boundary**.

## Background (papers + current state)

Two reference papers shape this plan:

- **Ullrich & Jablonowski (2012)**, *Operator-Split Runge–Kutta–Rosenbrock Methods for Nonhydrostatic Atmospheric Models* (MWR 140, 1257–1284). The RKR family of schemes — explicit RK for the cheap/non-stiff part, linearly implicit Rosenbrock for the stiff vertical part. Relevant here for two ideas: (a) the **Strang carryover scheme** (their §3d) achieves 3rd-order accuracy in the explicit part with only one implicit solve per step by interleaving an initial implicit operation and then carrying it through; (b) the **Ascher-Ruuth-Spiteri ARS(2,3,3) scheme** (their §3e) which is linearly third-order accurate in both `f` and `g` and nonlinearly third-order in `f`, requiring three explicit and two implicit operations. Either is overkill for what we need at the *physics-dynamics* boundary (where coupling is weak and `dt` is small relative to dynamic adjustment timescales), but the paper's framework is useful for understanding where the splitting error lives.

- **Santos, Caldwell & Bretherton (2021)**, *Cloud Process Coupling and Time Integration in the E3SM Atmosphere Model* (JAMES 13, e2020MS002359). The directly-relevant paper. They show that in E3SM the time-step sensitivity of the model is dominated by **process coupling error** (each process responding to the others' updates at the finite coupling interval) rather than **self-feedback error** (each process responding to its own effect at finite intervals). Their schematic of the four-mode failure pattern (figure 1) is exactly the structure we see in #468 (grey day-16) and #470. They also tabulate E3SM's actual sub-stepping (table 1): deep convection 1800 s, CLUBB+MG2 looped at 300 s, radiation applied at 1800 s but recalculated at 3600 s, dynamics 300 s with advection sub-cycled at 100 s and hyperviscosity at 100 s. **This is what we're moving toward.**

JCM's current state (as of `dev` post-#469):

- `Model._get_step_fn_factory` (`jcm/model.py:554-604`): builds `_step_tendencies` → `physics_forcing_eqn = ExplicitODE.from_functions(...)` → `_composed_eqn(d) = compose_equations([primitive, physics_forcing_eqn(d), ...])` → `imex_rk_sil3(_composed_eqn(d), dt)` → `step_with_filters(..., self.filters)`.
- `DiagnosticsCollector` (`jcm/model.py:166-188`): three carry slots — `data` (running-mean accumulator, averaged-mode-only), `physics_data_cache` (cross-step physics state), `physical_step` (first-substage flag).
- `averaged_trajectory_from_step` (`jcm/model.py:190-264`): runs an `nnx.scan` over inner_steps with the collector as carry, divides by `inner_steps` to get the mean.
- `get_physical_tendencies` (`jcm/physics_interface.py:548-633`): reads `prev_physics_data` from the cache, calls `physics.compute_tendencies(..., prev_physics_data=...)`, writes back the cache (gated on `physical_step`).
- `ComposablePhysics.compute_tendencies` (`jcm/physics/composable_physics.py:103-144`): iterates terms, accumulates tendencies, returns a single tendency + a diagnostics dict. Process-parallel within physics — every term reads the same input state.
- `ComposablePhysics.get_empty_data` (`jcm/physics/composable_physics.py:211`): probes the term loop with a zero-state PhysicsState to discover diagnostic shapes, then `tree_map(jnp.zeros_like, ...)`. This is what currently initialises `physics_data_cache.value` — and was the source of the 0/0 = NaN bugs in #469.

## Design

### 1. Splitting scheme

We have three reasonable choices.

| Scheme | Order in physics tendency | Order overall | Physics calls / dt | Dynamics calls / dt | Notes |
|---|---|---|---|---|---|
| **Lie split (a)** | 1 | min(1, dycore_order) | 1 | 1 | Simplest. `state → physics → dynamics → next`. |
| **Lie split (b)** | 1 | min(1, dycore_order) | 1 | 1 | Same cost. `state → dynamics → physics → next`. |
| **Strang split** | 2 | min(2, dycore_order) | 1 (evaluated once, applied as two half-steps) | 1 | Symmetric. `state → ½ physics → dynamics → ½ physics → next`. |

**Recommendation: start with Lie split (a), upgrade to Strang in a later phase.**

Reasoning:

- Lie (a) is the standard operational-GCM choice (ECHAM, CAM, IFS, E3SM). They've validated it at climate-rate `dt` for decades.
- The splitting error is `O(dt²)` and bounded by the commutator of the physics and dynamics tendency operators. For climate-rate `dt = 12 min` with the magnitudes JCM sees (physics tendencies are O(K/day) for T, O(g/kg/day) for q, etc.), the error is dominated by other discretisation sources.
- Strang is a one-line change once Lie is in place — same physics call, just applied as two half-step adds bracketing dynamics. We can do it as a follow-up after Lie is validated.
- ARS(2,3,3) and Strang carryover (UJ2011) are physics-dynamics overkill. Their value is in the *within-dynamics* horizontal-explicit / vertical-implicit splitting, which is exactly what `imex_rk_sil3` is already doing.

The step function becomes:

```python
def step(state, physics_state, forcing, terrain, date, dt):
    # 1. Compute physics tendency on the current state.
    physics_tend, physics_state_next, diagnostics = physics.compute_tendencies(
        state, physics_state, forcing, terrain, date,
    )
    # 2. Apply physics tendency to the dynamical state.
    state_after_physics = apply_physics_tendency(state, physics_tend, dt)
    # 3. Step pure dynamics over dt (IMEX RK for primitive + sponge + nudging).
    state_next = step_dyn(state_after_physics, dt)
    return state_next, physics_state_next, diagnostics
```

`apply_physics_tendency` does the existing `physics_tendency_to_dynamics_tendency` conversion and applies it: spectral-domain adds for `(vorticity, divergence, temperature_variation, log_surface_pressure, tracers)`. Everything stays differentiable.

For Strang (Phase 4), the same `physics_tend` is applied as two half-steps:

```python
def step_strang(state, physics_state, forcing, terrain, date, dt):
    physics_tend, physics_state_next, diagnostics = physics.compute_tendencies(
        state, physics_state, forcing, terrain, date,
    )
    state_mid = apply_physics_tendency(state, physics_tend, dt * 0.5)
    state_after_dyn = step_dyn(state_mid, dt)
    state_next = apply_physics_tendency(state_after_dyn, physics_tend, dt * 0.5)
    return state_next, physics_state_next, diagnostics
```

### 2. First-class `physics_state` carry

Today's `physics_data_cache.value` is a `dict` of typed sub-structs (`RadiationData`, `VerticalDiffusionData`, `CloudData`, `AerosolData`, `ChemistryData`, `SurfaceData`, …) that flows across timesteps. We promote that dict to a first-class typed carry:

```python
@tree_math.struct
class PhysicsCarryState:
    """Cross-step state carried by composable-physics terms.

    Each PhysicsTerm that has cross-step state (radiation flux for sub-
    cycling, TKE for the post-source ECHAM update, droplet number for
    2M micro, aerosol AOD for next step's radiation, …) reads from /
    writes to its own slot.

    The structure is determined entirely by the term list at compose
    time. ComposablePhysics builds an empty instance via each term's
    initial_carry_state(coords) factory.
    """
    radiation: RadiationData | None
    vertical_diffusion: VerticalDiffusionData | None
    clouds: CloudData | None
    aerosol: AerosolData | None
    chemistry: ChemistryData | None
    surface: SurfaceData | None
    # extensible — additional slots are added per-term
```

Each `PhysicsTerm` exposes:

```python
class PhysicsTerm(nnx.Module):
    ...
    def initial_carry_state(self, coords) -> dict[str, Any]:
        """Return a sensible non-zero initial value for this term's carry slot.

        Default ``{}`` — terms with no cross-step state don't need to
        override.

        IMPORTANT: should NOT use a zero-state probe like ``get_empty_data``
        does today. Zero state has T = 0 K which causes spurious NaNs in
        downstream radiation (this was the bug PR #469 had to work
        around). Use either:
          (a) a sensible-default constructor (e.g. RadiationData with
              albedo = sensible defaults, fluxes = 0 — fluxes get
              overwritten on first compute step anyway);
          (b) a single "warm-up" physics call on the IC state.
        """
        return {}

    def __call__(
        self, state, physics_state, diagnostics, forcing, terrain,
    ) -> tuple[PhysicsTendency, dict, dict]:
        """Return (tendency, updated_physics_state_slot, updated_diagnostics)."""
        ...
```

`ComposablePhysics.initial_carry_state(coords)` aggregates each term's slot. The result replaces today's `get_empty_data` and is built once at `Model` construction time, *not* probed during integration.

### 3. `Model` API changes

The `Model.run / .resume` API today returns `(final_state, ModelPredictions)`. The new API also carries `physics_state`:

```python
class Model:
    def __init__(self, ...):
        ...
        self._physics_carry_state = self.physics.initial_carry_state(self.coords)
        self._final_modal_state = self._prepare_initial_modal_state()

    def step(self, state, physics_state, forcing, ...):
        """Single forward step. Pure function; for use by integration loops."""
        ...

    def run(self, forcing, save_interval, total_time, output_averages):
        """Public entry point. Threads (state, physics_state) through inner scan."""
        ...
```

`physics_state` is held on `self._physics_carry_state` between `.run` / `.resume` calls so the user-facing API stays the same.

### 4. `ComposablePhysics.compute_tendencies` new signature

```python
def compute_tendencies(
    self,
    state: PhysicsState,
    physics_state: PhysicsCarryState,
    forcing: ForcingData,
    terrain: TerrainData,
    date: DateData,
) -> tuple[PhysicsTendency, PhysicsCarryState, dict]:
    """Run all terms once on ``state``. Returns (tendency, new_carry, diagnostics).

    ``physics_state`` is the cross-step carry (radiation flux, prev TKE,
    etc.). It replaces the ``prev_physics_data`` argument and the
    ``DiagnosticsCollector.physics_data_cache`` plumbing.

    ``diagnostics`` is the per-step diagnostics dict (cloud fraction at
    this step, surface fluxes, …) — only used for the saved trajectory,
    *not* for cross-step state. Saving it to the trajectory is the
    averaging path's job; this function just returns it.
    """
```

Each term gets called with `(state, physics_state, diagnostics, forcing, terrain)` and returns `(tendency, physics_state_slot_updated, diagnostics_updated)`.

The term loop is otherwise unchanged. Process-parallel coupling within physics (every term sees the same input `state`) is preserved — see "Coupling within physics" below.

### 5. `DiagnosticsCollector` simplifies

After this refactor, the collector has *one job*: running-mean accumulation over `inner_steps` for the averaged-output path. Specifically:

```python
class DiagnosticsCollector(nnx.Module):
    data: nnx.Variable      # (outer_steps,) + diagnostic_shape, accumulated mean
    i: nnx.Variable         # current outer-step index
    steps_to_average: int   # = inner_steps

    def accumulate(self, new_data):
        self.data.value = tree_map(
            lambda stacked, new_: stacked.at[self.i.value].add(
                new_ / self.steps_to_average
            ),
            self.data.value, new_data,
        )
```

Gone: `physics_data_cache`, `physical_step`, `accumulate_if_physical_step`. The substage-disambiguation problem is gone because physics is called once per outer step, not once per RK stage.

### 6. Dynamics-only step

The IMEX-RK SIL3 step over pure dynamics:

```python
def _get_dynamics_step_fn(self):
    """Build the IMEX-RK step for pure dynamics. No physics."""
    # primitive equations live here (this is what dinosaur was designed for).
    # sponge and nudging stay inside RK if they should respond to intermediate
    # states; otherwise migrate them to op-split alongside physics.
    equations = [self.primitive]
    if self.nudging is not None:
        equations.append(_nudging_eqn(self))
    composed = dinosaur.time_integration.compose_equations(equations)
    return dinosaur.time_integration.step_with_filters(
        dinosaur.time_integration.imex_rk_sil3(composed, self.dt),
        self.filters,
    )
```

This is the function pysces would replace for #388. The interface is `state → state'` over `dt`, no physics knowledge.

**Sponge: stays inside RK.** The upper-sponge is a stiff dissipation acting on vertical-wave modes that the IMEX-RK stages need to see during the implicit solve. Migrating it to op-split would lose stability at the model top. Keep inside.

**Nudging: stays inside RK for now.** It's a fast linear relaxation; treat like sponge. Could be moved later.

### 7. Coupling within physics

Today's `ComposablePhysics` is **process-parallel**: every term sees the same input state, tendencies are summed. Santos 2021 §1.2 describes the alternative — **sequential** — where each parameterisation accepts a state already updated by previous ones.

**Recommendation: keep process-parallel as the default**, exactly as today, and call out sequential as a future option. Reasons:

- Process-parallel preserves the composability story (`A + B + C` doesn't depend on order).
- Sequential is more accurate for fast-coupling processes but introduces order-dependence — every (re-)composition needs sensible ordering and the `replace()` / `remove()` operators get more semantic baggage.
- E3SM's sequential coupling has well-documented coupling-error pathologies (Santos figure 1) at large `dt`. Going sequential is a fix for *those* — but JCM at `dt = 12 min` doesn't have them.

A later option, for terms that genuinely need sequential coupling (e.g. cloud-microphysics ↔ convection ↔ radiation in a tightly-coupled cloud regime), is to introduce explicit *process groups* à la E3SM's CLUBB+MG2 loop, where the group runs sub-cycled internally and presents a single tendency externally.

### 8. Initial physics state

Replaces `get_empty_data`. Each term builds its slot at compose time:

```python
# example: GreyTwoStreamRadiation
def initial_carry_state(self, coords) -> dict[str, Any]:
    nlev, ncols = coords.nodal_shape[0], coords.horizontal.nodal_shape[0] * coords.horizontal.nodal_shape[1]
    return {"radiation": RadiationData.zeros((ncols,), nlev)}
```

`RadiationData.zeros` uses *physical* zeros, not a zero-state probe. The first compute step will overwrite the fluxes with real values before any other term reads them; this is the same behaviour the zero-state probe was *trying* to achieve, but built deterministically instead of probed.

### 9. Differentiability

All four changes are pure-functional. `physics_state` is a pytree leaf in the integration carry. `apply_physics_tendency` is a tree-add. `step_dyn` is whatever the dycore is. `jax.grad` and `jax.checkpoint` continue to work through the new step function as long as it's expressed as a pure function of `(state, physics_state, dt, ...)`.

Memory budget: the IMEX-RK no longer re-traces physics three times per step, so the saved-for-backward memory inside `jax.checkpoint(step)` drops by approximately 3× the physics-data dict size. Backward passes get cheaper.

## Phased delivery

Each phase is a self-contained PR with passing tests. Branches off `dev`.

| Phase | Status | Scope | Files touched (rough) | Acceptance |
|---|---|---|---|---|
| **0. Scaffolding** | ✅ done | Add `PhysicsTerm.initial_carry_state` (default `{}` for all current terms); `ComposablePhysics.initial_carry_state(coords)` that aggregates. No call-site changes. Existing path keeps working. | `jcm/physics/physics_term.py`, `jcm/physics/composable_physics.py`, plus shim `initial_carry_state` on every existing term. | Existing tests pass unchanged. `initial_carry_state(coords)` produces shapes/keys that are a subset of today's `get_empty_data(coords)` for an ECHAM composition; the slots both produce share shape (test: `test_initial_carry_state_echam_keys_subset_of_get_empty_data`). |
| **1. Lie split path, behind a flag** | ✅ done | New `Model._get_op_split_step_fn` + `_get_dynamics_step_fn`. `ComposablePhysics.compute_tendencies` already takes a `prev_physics_data` arg (becomes the cross-step `physics_state` carry). Plumb through `Model.run / .resume` via a new `use_op_split: bool = False` keyword. | `jcm/model.py`, `jcm/physics_interface.py`. | New tests in `model_test.py::TestOperatorSplitPhysics` — SPEEDY and ECHAM-hybrid op-split in both snapshot and averaged modes produce finite atmospheric state. Default (legacy) path unchanged, all existing tests pass. |
| **2. Switch default to op-split** | ✅ done | Make `use_op_split=True` the default. Add a deprecation log message when `use_op_split=False` is requested. | `jcm/model.py`, docs. | Existing tests that don't explicitly set the flag now run via op-split (842 → 849 passing under op-split default). `test_op_split_deprecation_warning_on_legacy` confirms the legacy-path warning fires. |
| **3. First-class `PhysicsCarryState` + `DiagnosticsCollector` cleanup** | ✅ done | Introduce a `PhysicsCarryState` type alias in `physics_interface` (currently a `dict[str, Any]`). Delete `physics_data_cache` / `physical_step` from `DiagnosticsCollector`. Replace `accumulate_if_physical_step` with `accumulate`. Drop the cache read/write in `physics_interface.get_physical_tendencies`. Op-split uses `initial_carry_state(coords)` as the cross-step carry seed (falling back to `get_empty_data` for term-slots that haven't yet been migrated). Added overrides on `EchamBoundaryConditions`, `TTETKEVerticalDiffusion`, `SundqvistCloudFraction`, `Macv2SpAerosol`. | `jcm/model.py`, `jcm/physics_interface.py`, `jcm/physics/forcing/echam_boundary_conditions.py`, `jcm/physics/vertical_diffusion/tte_tke/{vertical_diffusion.py, vertical_diffusion_types.py}`, `jcm/physics/clouds/sundqvist.py`, `jcm/physics/aerosol/macv2_sp.py`. | `test_no_grep_physics_data_cache` (production code is clean of the identifier). `test_collector_has_no_physics_data_cache`, `test_collector_has_no_physical_step`, `test_collector_accumulate_method`. 849 fast tests still pass. |
| **4. Remove the legacy inside-RK path** | ✅ done | Deleted: `_step_tendencies`, `physics_forcing_eqn`, the `compose_equations([primitive, physics_forcing_eqn(d)])` line, the `use_op_split` kwarg on `run` / `resume` / `run_from_state` / `_run_from_state`, `DiagnosticsCollector` (and its tests), `averaged_trajectory_from_step`, `_get_step_fn_factory`, `_get_integrate_fn`, `get_physical_tendencies`, the legacy `_post_process` recompute branch. `_op_split_post_process` becomes the only `_post_process`. **Kept (renamed in role, not name):** `Physics.get_empty_data` — repurposed as an internal *structural template* helper for the `lax.scan` carry pytree, with the zero-state probe replaced by an isothermal 288 K probe to avoid the 0/0 = NaN cascade. Used only at Model construction to discover the post-step `compute_tendencies` output structure; result is zero-filled and only ever used as a shape template, never as live state. | `jcm/model.py`, `jcm/physics_interface.py`, `jcm/physics/composable_physics.py`, plus test updates in `jcm/model_test.py`, `jcm/physics/held_suarez/held_suarez_test.py`. | `TestLegacyPathRemoved` (in `model_test.py`) asserts the removed symbols are gone and no production code references the legacy identifiers (grep-level regression). All 897 fast + slow tests still pass. |
| **5. Strang upgrade (optional)** | ⏳ todo | Replace Lie's single full-step apply with two half-step applies bracketing `step_dyn`. | `jcm/model.py`. | A 30-day comparison shows monotone reduction in splitting error vs Lie at coarser `dt` (e.g. `dt = 30 min`); at `dt = 12 min` differences are negligible. |
| **6. Op-split-enabled dycore swap (separate issue #388)** | ⏳ todo | With physics fully decoupled, write `dycore.pysces.step_fn(state, dt) -> state'` and a `physics_state ↔ pysces_state` conversion. Substitute via `Model.__init__(dycore=...)`. | New `jcm/dycore/pysces.py`, new `jcm/dycore/dinosaur_wrapper.py`. | T63L47 ECHAM-on-pysces runs cleanly for `n` days (`n` per the pysces validation harness). |

Phases 0-2 deliver the immediate value (averaging works on long integrations, #470 closes). Phase 3 collapses the architectural complexity (`DiagnosticsCollector` is just a running-mean accumulator). Phase 4 is housekeeping. Phase 5 is an accuracy upgrade. Phase 6 is the dycore-swap payoff.

### Phase 3 implementation notes (deviations from the original plan)

The original plan called for fully deleting `get_empty_data` and `accumulate_if_physical_step` in Phase 3. Two practical concessions:

- **`get_empty_data` is kept as a deprecated helper.** It is no longer used as the cross-step cache (the bug path that motivated #470); the op-split path uses `initial_carry_state` for that. It is still used as the zero-shape *template* for the legacy averaged path's stacked running-mean accumulator and as a backwards-compat fallback for op-split term slots that haven't yet had their `initial_carry_state` override added. The zero-state probe is fine for shape discovery — the bug was using the probe output as live cross-step state. Phase 4 deletes `get_empty_data` along with the legacy path.

- **The legacy averaged path lost its radiation sub-cycle cache.** With `physics_data_cache` gone from `DiagnosticsCollector`, the legacy `get_physical_tendencies` always passes `prev_physics_data=None` to `compute_tendencies`. Radiation recomputes at every substage in the legacy averaged path — slower than before, but the path is deprecated. Op-split (the default) is the correctness path.

- **Phase 3 added four `initial_carry_state` overrides** — on `EchamBoundaryConditions` (radiation, surface, chemistry slots), `TTETKEVerticalDiffusion` (`vertical_diffusion` slot with TKE seeded at the 0.01 m²/s² ECHAM floor), `SundqvistCloudFraction` (`clouds` slot), `Macv2SpAerosol` (`aerosol` slot). All four use deterministic constructors on the typed sub-struct's `.zeros()` factory rather than the zero-state term-loop probe.

- **`VerticalDiffusionData.km` / `.kh` shape doc was corrected**: the docstring claimed `(nlev+1, ncols)` but the term writes `(nlev, ncols)`; `.zeros()` was producing the documented (wrong) shape, which masked itself in the legacy averaged path because `get_empty_data` overwrote it on the first probe call. The fix aligns docstring + factory + actual term output at `(nlev, ncols)`.

### Phase 3 follow-up: P1/P2 carry-threading fixes

Phase 3 wired the carry through the integration scan but left two
gaps that a code review of #472 surfaced:

- **P1: persist the carry across `run()`/`resume()` API boundaries.**
  The trajectory builder discarded the final `physics_state`, and
  `Model` only persisted `_final_modal_state`. Each call to
  `_run_from_state` therefore rebuilt the cross-step carry from
  scratch, so radiation cache, prior-step TKE, etc. reset at every
  API seam — a continuous 10-day run differed from a 5-day `run()`
  plus 5-day `resume()`. Fixed by adding a `_final_physics_state`
  slot to `Model`, returning the final carry from
  `_op_split_trajectory` / `_run_from_state` / `run_from_state`, and
  threading it back in on `resume()`. `run()` resets the slot so a
  fresh trajectory does not pick up stale carry from a previous run.
  New regression test: `test_op_split_carry_persists_across_resume`.

- **P2: snapshot diagnostics use the integration carry.** In
  snapshot mode the scan saved `post_process_fn(x_final)` per outer
  step, and `_post_process` recomputed physics inside with
  `prev_physics_data=None`. Because the default `radiation_interval`
  is 7200 s the dycore re-uses cached radiation on most outer steps;
  the recompute path however always sees a freshly-seeded cache and
  silently reports zero / IC radiation fields. Fixed by saving the
  carried `physics_state` from the scan and using it directly as
  `predictions.physics` (the averaged path already did this with the
  inner-step running mean). New `_op_split_post_process` does only
  the dynamics-state conversion; no recompute. New regression test:
  `test_op_split_snapshot_physics_uses_integration_carry`.

- **Averaged accumulator switched to post-step states.** Previously
  the averaged accumulator summed pre-step states (matching the
  legacy convention), which gave a one-timestep offset against the
  snapshot path's post-step samples. The offset was tolerable for
  slow fields under the legacy inside-RK scheme, but op-split's
  larger per-step transient amplified the difference enough to break
  `test_speedy_model_averages` at `rtol=1e-2`. Summing `x_next`
  rather than `x` brings the two paths into numerical-roundoff
  agreement (the test now passes at `rtol=1e-4`).

Estimated effort: phases 0-2 ≈ 2-3 days of focused work + validation. Phase 3 ≈ 1 day (+ P1/P2 follow-up ≈ 0.5 day). Phase 4 ≈ 0.5 day. Phase 5 ≈ 0.5 day + validation. Phase 6 is sized by the pysces work, not by this refactor.

## Tests and validation

### Unit-level
- `test_op_split_lie_first_order_in_dt` — RCE column with prescribed `state.tke = 0` everywhere, run with `dt ∈ {60, 120, 240, 480, 960}` s, verify the trajectory error scales as `O(dt)` for Lie / `O(dt²)` for Strang.
- `test_initial_carry_state_no_zero_probe` — assert that `ComposablePhysics.initial_carry_state(coords)` does not invoke `compute_tendencies` with a zero state (defensive against regressing to the `get_empty_data` style).
- `test_physics_state_is_pytree` — `jax.tree_util.tree_flatten(physics_state)` round-trips; `jax.grad` through `step` returns a finite gradient.

### Integration-level
- The existing `test_echam_hybrid_model_output_averages` (hardened in PR #469 to assert non-NaN values) must pass with both `use_op_split=True` and `False` during phases 1-3, and with op-split only from phase 4 on.
- New `test_op_split_30day_averaged` — `t63_l47_hybrid + balanced_isothermal + ECHAM + output_averages=True + run.total_time=30` (the test that surfaced #468 and #470). Asserts finite atmosphere and reasonable global means: surface T 270-290 K, TOA OLR 220-280 W/m², SW down 320-360 W/m², total precip 0.5-5 mm/day, TKE max < 50 m²/s².
- `test_op_split_matches_snapshot_at_dt_zero` — sanity check that Lie / Strang split agree with the inside-RK path in the `dt → 0` limit on a 3-hour integration. Differences should be `O(dt)` for Lie and `O(dt²)` for Strang.

### Stability tests
- `test_op_split_360day_year` — T63L47 ECHAM 2M-snapshot year run (the one that completed cleanly on 2026-05-10 with the old code). Verify it still completes after the refactor. Allow a relative diff < 5% on year-mean fields vs the saved baseline.

### Performance check
- Wall-time per `dt` should drop ≈ 2× (physics goes from 3× per `dt` to 1×). RRTMGP runs in particular should be noticeably faster because the radiation sub-cycle now has one timeline to honour, not three.

## Risks and open questions

### 1. Order-of-accuracy loss on the explicit physics tendency

We go from formal 3rd-order (SIL3 explicit) to 1st-order (Lie) or 2nd-order (Strang). The Santos 2021 analysis applies directly: at climate-rate `dt`, **coupling error dominates self-feedback error**, and Lie/Strang are both fine. NWP-rate `dt` with strong convection might be different; if pursuing wave-resolving regimes later, revisit.

Quantitative test plan: run the same 30-day integration with `dt ∈ {6, 12, 24} min` under each scheme and look at the spread in zonal-mean climatology. If the spread under Strang is < the spread between Strang and inside-RK at any of those `dt`, Strang is "good enough."

### 2. Tightly-coupled regimes

Some physics-dynamics couplings benefit from inside-RK evaluation for accuracy (deep convection in shear). JCM already supports adding tightly-coupled terms via `+ UpperSponge` *inside* the stage loop — we'll preserve that escape hatch. A hybrid scheme ("expensive physics op-split, cheap fast-feedback terms inside-RK") is achievable.

### 3. Tracer interaction

Tracers are advected by dynamics and modified by physics. Today: each RK stage applies a fractional advection + a fractional physics-tendency, intermixed. Op-split: full physics-tendency applied once, then full dynamics-advection over `dt`. For tracers with strong negative tendencies (e.g. precipitating water in heavy rain) this can drive intermediate states negative before clipping. `verify_tendencies` (`jcm/physics_interface.py:509`) already caps negative tendencies at `-state/dt`, which is conservative for the full-step case. Should still work — add a test.

### 4. Backward compatibility for the existing API

`Model.run(forcing, save_interval, total_time, output_averages)` should keep working. Internally it dispatches to either the old or new step function via the `use_op_split` flag. Phase 4 removes the flag; until then, no user-facing breakage.

### 5. RRTMGP cache flow

RRTMGP's `radiation_should_compute` reads `model_step` from `_date` and the `radiation_interval` parameter. With op-split, that pattern is preserved — radiation runs every Nth `dt`, otherwise re-emits the cached heating from `physics_state.radiation.lw_heating_rate + .sw_heating_rate`. The cache lives in `physics_state` (first-class) instead of in `physics_data_cache.value` (hidden). Same semantics, cleaner plumbing. This is what should unblock #470.

### 6. SCM and prescribed-state modes

`_run_prescribed` and `_run_scm` (`jcm/runners.py:540` onward) side-step the dynamical core. They'll need the `physics_state` carry too. Straightforward — both already iterate over saved snapshots; just thread `physics_state` between iterations.

### 7. Differentiability tests

`test_speedy_model_gradients_isnan` and equivalents should keep passing. Adding a `test_op_split_gradients_finite` is cheap and worth doing.

### 8. What if Strang turns out to *not* be enough?

If the climatology shifts more than acceptable between inside-RK and Strang at climate-rate `dt`, the next step up is **Marchuk-Strang** (a 4-call symmetric form) or a sub-cycled physics tendency where physics is applied as N small additions of size `dt/N` bracketed by dynamics sub-steps. This is what E3SM does for CLUBB+MG2. Add this only if needed.

### 9. Coupling to `EchamBoundaryConditions`

This term seeds CO2/CH4/O3 VMRs and surface fields from forcing. It runs at the top of the term loop today. With op-split, the same logic applies — `EchamBoundaryConditions` reads from forcing once per `dt` and writes into the diagnostics dict. No semantic change. The fact that downstream terms see the *just-updated* boundary fields in the same step (process-parallel coupling) is unchanged.

### 10. Multi-physics-step-per-dynamics-step

E3SM sub-cycles CLUBB+MG2 at 300s within the 1800s convection time step. JCM doesn't have this complication today (everything runs at the dynamics `dt`), but the design should allow it — `Model.step` could in principle take an inner `physics_steps_per_dt` parameter and loop. Not in this PR; mentioned to verify the design space.

## Concrete starting point

Phase 0 PR diff sketch (the smallest possible step that scaffolds the rest):

```python
# jcm/physics/physics_term.py
class PhysicsTerm(nnx.Module):
    ...
    def initial_carry_state(self, coords) -> dict[str, Any]:
        """Default: no cross-step state. Override per-term as needed."""
        return {}

# jcm/physics/composable_physics.py
class ComposablePhysics(nnx.Module, Physics):
    ...
    def initial_carry_state(self, coords) -> dict[str, Any]:
        """Aggregate per-term initial carry states. Replaces ``get_empty_data``."""
        state = {}
        for term in self.terms:
            state.update(term.initial_carry_state(coords))
        return state
```

Plus a no-op override on the handful of terms that today rely on the `prev_physics_data` cache. Then a test that `initial_carry_state(coords)` and `get_empty_data(coords)` produce keys with matching shapes for an ECHAM composition. Nothing else changes — full backwards compatibility through phase 0.

Phase 1's PR is the substantial one. Phases 2-4 are mechanical cleanups once Phase 1's tests pass.

## Appendix — ECHAM6 reference

For completeness, the proposed JCM structure mirrors ECHAM6's well-established operator-split design. The relevant code chain in `echam6.3.0-ham2.3-moz1.0.r7492/src`:

```text
stepon.f90:156      integration_loop: DO                 ← main time loop
stepon.f90:271        CALL scan1                         ← gridpoint phase
  scan1.f90:499         CALL gpc(jrow)
    gpc.f90:78           CALL physc(krow)
      physc.f90:543       CALL cover               ← cloud cover
      physc.f90:566       CALL radiation           ← full RT (sub-cycled)
      physc.f90:678       CALL vdiff               ← vertical diffusion
      physc.f90:776       CALL radheat             ← cached-flux heating
      physc.f90:835       CALL gwspectrum          ← Hines GW drag
      physc.f90:877       CALL ssodrag             ← orographic GW drag
      physc.f90:987       CALL cucall              ← Tiedtke convection
      physc.f90:1067      CALL cloud               ← stratiform + micro
stepon.f90:280        CALL sccd                          ← divergence semi-implicit
stepon.f90:286        CALL scctp                         ← T + Ps semi-implicit
stepon.f90:294        CALL uspnge                        ← upper sponge (in dynamics)
stepon.f90:309        CALL hdiff                         ← horizontal diffusion
stepon.f90:476      END DO integration_loop
```

Physics tendencies are accumulated into the gridpoint buffer arrays declared at `mo_scan_buffer.f90:36-47` (`vol`, `vom`, `tte`, `qte`, `xlte`, `xite`, `xtte`) and consumed by `sccd` / `scctp` as the explicit forcing in the semi-implicit Helmholtz solve. **Physics is called exactly once per dynamics timestep.**

Two structural differences from the JCM target:

1. **Dynamics integrator.** ECHAM uses leapfrog + semi-implicit (spectral); JCM uses IMEX-RK SIL3 (also spectral, via dinosaur). The split point is the same — operator-split physics — but the dynamics integrator on each side differs. **Caveat:** leapfrog only ever asks for one RHS evaluation per `dt`, so "operator-split physics" in ECHAM isn't a *choice* — it's the natural shape that leapfrog imposes. JCM's SIL3 has substages and could in principle evaluate physics at each one (and does today), so for JCM operator-splitting is a real design decision, not a forced consequence of the integrator. The directly comparable models — multi-stage or sub-stepped dynamics that *choose* to op-split physics anyway — are CAM (HOMME spectral element with internal sub-stepping), E3SM (HOMME with sub-cycled advection at 100 s and physics applied at 1800 s), and IFS (semi-Lagrangian + semi-implicit). All three op-split despite having more than one dynamics evaluation per physics `dt`, which is exactly the situation JCM-with-SIL3 is in. Treat the ECHAM mapping below as an existence proof of "physics-as-tendency-forcing → dynamics" being a workable pattern, not as a precedent that directly justifies JCM's choice.

2. **Within-physics coupling.** ECHAM is **sequential** — each scheme reads the state with prior schemes' tendencies already applied, via the `tte += ...` pattern on the shared accumulators in `mo_scan_buffer.f90`. JCM's `ComposablePhysics` is **process-parallel** — every term reads the same input state, tendencies are summed. This plan preserves JCM's process-parallel default for composability (the order-independence of `A + B + C`), with sequential coupling available as a follow-up if specific term pairs need it (cf. E3SM's CLUBB+MG2 loop in Santos 2021 §2.1).

The cross-step physics state in ECHAM lives in module-level globals (`mo_radiation_forcing`, etc.) — semantically the same as the proposed `PhysicsCarryState` pytree, just routed via Fortran's module-level mutability rather than as an explicit functional carry.
