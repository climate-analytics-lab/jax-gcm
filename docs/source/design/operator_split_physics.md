# Operator-split physics

## Overview

JCM evaluates physics exactly once per dynamics timestep `dt`, outside
the IMEX-RK stages, and applies the resulting tendency as a Lie split
ahead of the dynamics integral:

```text
state ─┐
       ├── compute_physics_step ─► dyn_tendency, new_physics_state
       │
       ▼
state + dt · dyn_tendency
       │
       ▼
   IMEX-RK SIL3 dynamics over dt   ← primitive equations (+ optional nudging)
       │                              + upper sponge, hyperdiffusion filters
       ▼
   filters (mean-Ps fix, level-dependent hyperdiffusion on
            divergence / vorticity-tracers / temperature)
       │
       ▼
state_next
```

The splitting error is `O(dt)` (Lie split (a), `state → physics →
dynamics → next`). The dynamics IMEX-RK is the only step that advances
`sim_time` — `dyn_tendency` carries `sim_time = 0` so the forward-Euler
add does not perturb it.

This mirrors operational GCM practice (ECHAM, CAM, IFS, E3SM): physics
runs once per `dt` as forcing to the dynamics, rather than at each RK
substage.

### Why Lie and not Strang

The natural next step from Lie is a Strang split (`state → ½dt
physics → dynamics → ½dt physics → next`). Two forms exist:

- **Single-evaluation** ("symmetric Lie"): reuse the same
  `dyn_tendency` for both half-steps. 1× physics cost. But for
  state-dependent physics (the normal case) this is *not* second-order
  accurate — the local truncation analysis shows it misses the
  `(dt²/2)(PD + P²)` term the exact propagator carries. Globally `O(dt)`,
  same as Lie.
- **Classical Strang**: re-evaluate physics on the post-dynamics state
  for the second half-step. 2× physics cost (still cheaper than the
  legacy 3× per-substage scheme). True `O(dt²)` accuracy.

Neither is implemented today. Classical Strang is a worthwhile
follow-up if a coarser-`dt` regime exposes the splitting error; at the
current climate-rate `dt = 12-30 min`, the Santos 2021 analysis
(JAMES 13, e2020MS002359) finds coupling error dominates self-feedback
error and Lie/Strang are both adequate. The single-evaluation symmetric
form was attempted in this PR's history but reverted after Codex review
correctly flagged the order issue (PR #481 review thread).

## Key components

### `Model._get_op_split_step_fn` (`jcm/model.py`)

Builds the per-`dt` step closure `(state, physics_state) -> (state_next,
physics_state_next)`. Internals:

1. Resolve the current step's date and forcing slice from `state.sim_time`.
2. Call `compute_physics_step` for `(dyn_tendency, new_physics_state)`.
3. Lie apply: `state + dt·tend → dynamics_step` (full forward-Euler add
   then full-`dt` IMEX-RK).
4. Run the post-step filters (conserve global-mean surface pressure,
   level-dependent hyperdiffusion on divergence / vorticity+tracers /
   temperature).

The step is a pure function of `(state, physics_state)`. The
`physics_state` carry is a JAX pytree and is the only cross-step state
the integrator threads.

### `Model._get_dynamics_step_fn` (`jcm/model.py`)

Builds the IMEX-RK SIL3 integrator over the dynamics composition. The
dynamics composition is the primitive equations plus, optionally, a
Newtonian nudging tendency. Sponge and hyperdiffusion stay inside the
IMEX-RK stage loop / `step_with_filters` path — they are stiff /
fast-linear couplings that benefit from intermediate-state evaluation
and would lose stability if migrated to op-split.

This is the function a future pysces backend (#388) would replace.
Its interface is `state -> state'` over `dt` with no physics knowledge.

### `compute_physics_step` (`jcm/physics_interface.py`)

Converts the spectral dynamics `State` to a nodal `PhysicsState`, runs
`verify_state` (non-negativity clamp on tracers), calls
`Physics.compute_tendencies(state, forcing, terrain,
prev_physics_data=physics_state)`, applies `verify_tendencies` (caps
negative-going tracer tendencies at `-state/dt`), and converts the
result back to a dinosaur dynamics tendency.

Returns `(dynamics_tendency, new_physics_state)`. The new carry is the
dict the physics call writes to (radiation cache, prior-step TKE,
cloud / aerosol / chemistry diagnostics, …) — exactly the input shape
of the next step.

### `_op_split_trajectory` (`jcm/model.py`)

The trajectory builder. Takes the per-`dt` step function and threads
`(state, physics_state)` through a nested `lax.scan`:

- **Outer scan** — `outer_steps` saved frames.
- **Inner scan** — `inner_steps` `dt` steps between saves.

Two output modes:

- **Snapshot** (`output_averages=False`). The inner scan steps
  silently; the outer step saves the post-inner-scan `(state_final,
  physics_state_final)`. The saved `predictions.physics` is the
  cross-step carry the integration *actually consumed* — radiation
  sub-cycle cache, TKE memory, etc. This is required for correctness:
  with `radiation_interval = 7200 s` the dycore reuses cached fluxes
  on most outer steps, and recomputing physics from a freshly seeded
  carry at save time would silently report zero / IC radiation fields
  (the original #470 bug).
- **Averaged** (`output_averages=True`). The inner scan accumulates a
  running mean of post-step dynamics states (`state_next`) and of the
  per-step physics-state dict. The outer step saves the time-mean.
  Summing *post-step* states gives bit-equivalence with the snapshot
  path's end-of-step samples — `mean(snapshots) == averaged(...)` to
  numerical roundoff. The summands are cast to `float` before the
  running mean to avoid mid-scan dtype promotion (which `lax.scan`
  rejects).

`jax.checkpoint` wraps each inner step, so backward passes
re-materialise the step rather than save it. Memory budget: physics
runs once per `dt` rather than three times, so the autodiff tape for
a checkpointed step is `~3×` smaller than under the deprecated
in-stage scheme.

### Cross-step carry persistence

`Model` holds two slots of cross-step state:

- `_final_modal_state` — dinosaur spectral state at the end of the
  last `run()` / `resume()` call.
- `_final_physics_state` — the cross-step physics carry at the end of
  the last call.

`run()` resets both via `bootstrap_state` (and rebuilds the physics
carry via `_build_initial_physics_carry`). `resume()` threads them
back in. The result: a `run(5d)` + `resume(5d)` chain is numerically
equivalent to a contiguous `run(10d)` — sub-cycled radiation and
prior-step TKE do not reset at the API seam. Regression covered by
`test_op_split_carry_persists_across_resume`.

`run_from_state_with_carry` exposes the carry seed and final carry
directly for callers that need explicit control.

### Initial physics carry

`Model._build_initial_physics_carry` unions:

- `Physics.initial_carry_state(coords)` — per-term deterministic seed.
  Each `PhysicsTerm` that has cross-step state overrides this with a
  sensible non-zero value (e.g. `TTETKEVerticalDiffusion` seeds TKE at
  the 0.01 m²/s² ECHAM floor; radiation slots use `RadiationData.zeros`
  for fluxes — the first compute step writes real values before any
  consumer reads them).
- `Physics.get_empty_data(coords)` — a *structural template* obtained
  by probing the term loop with an isothermal-288 K `PhysicsState`,
  zero-filled. Used only to discover the dict structure that
  `compute_tendencies` produces, so the `lax.scan` carry pytree
  matches the post-step output on iteration 1 for any diagnostic keys
  whose owning term has not overridden `initial_carry_state`.

The isothermal probe (rather than a zero-state probe) avoids the
`0/0 = NaN` cascade that motivated the cache-cleanup in PR #469. The
result of `get_empty_data` is never used as live state — only as a
shape template.

### Coupling within physics

`ComposablePhysics` is **process-parallel**: every term sees the same
input `state`, tendencies are summed. Order-independent
(`A + B + C == B + A + C`), which keeps `replace()` / `remove()` /
`__add__()` composition operators well-defined.

This differs from ECHAM6's sequential coupling (each scheme sees the
state with prior schemes' tendencies already applied via
`tte += ...`). Sequential coupling is more accurate for tightly-coupled
process pairs but introduces order-dependence. Process-parallel is
adequate at JCM's climate-rate `dt`; sequential coupling is available
as a follow-up for terms that need it (cf. E3SM's CLUBB+MG2 loop).

### Filters and dycore composition

The post-step filter chain runs on `(pre_step_state, post_dynamics_state)`
in this order:

1. `conserve_global_mean_surface_pressure` — pins the 0,0,0 spectral
   mode of `log_surface_pressure` to its pre-step value.
2. `diffuse_div` — level-dependent hyperdiffusion on divergence.
3. `diffuse_vor_q` — hyperdiffusion on vorticity and *every* tracer
   (specific humidity + microphysics tracers + GHG VMRs).
4. `diffuse_temp` — hyperdiffusion on temperature variation.

The level-dependent scaling is precomputed once at `Model.__init__`
and inlined as a JIT constant.

A final `verify_state` clamp runs in `_post_process` at the
modal→nodal output boundary to mask any spectral Gibbs-ringing of
negative tracer tendencies before user-visible output.

## Tests

Op-split coverage in `jcm/model_test.py::TestOperatorSplitPhysics`:

- `test_op_split_snapshot_speedy_finite` /
  `test_op_split_averaged_speedy_finite` /
  `test_op_split_averaged_echam_hybrid_finite` — both modes, SPEEDY
  and ECHAM-hybrid; require finite atmospheric state.
- `test_op_split_step_is_jax_pure` — the step function traces cleanly
  under `jit` and round-trips the dynamics pytree structure.
- `test_op_split_carry_threading` — `physics_state` returned by step
  N is the same pytree structure as the input to step N+1 (the
  `lax.scan` carry contract).
- `test_op_split_carry_persists_across_resume` — `run(5d) +
  resume(5d)` matches `run(10d)` to numerical roundoff (the P1 fix).
- `test_op_split_run_resets_carry` — repeated `run()` on the same
  `Model` does not inherit stale carry.
- `test_op_split_snapshot_physics_uses_integration_carry` — the
  snapshot `predictions.physics` is the carry the integration
  consumed, not a freshly-seeded copy (the P2 fix).

The legacy in-stage path was removed in Phase 4 (`TestLegacyPathRemoved`);
no flag remains to switch back. The deleted symbols are
`_step_tendencies`, `physics_forcing_eqn`, `DiagnosticsCollector`,
`averaged_trajectory_from_step`, `get_physical_tendencies`,
`_get_step_fn_factory`, `_get_integrate_fn`, and the `use_op_split`
kwarg on `run()` / `resume()` / `run_from_state()`.

## Performance notes

- Physics runs once per `dt` rather than three times per `dt` (one per
  IMEX-RK explicit substage). For RRTMGP / TTE-TKE / Tiedtke-Nordeng
  this is roughly 3× the wall-time saving on the physics path.
- Backward passes pay `~3×` less memory under `jax.checkpoint` for the
  physics path because each checkpoint re-traces a single physics call.
- Radiation sub-cycling honours one timeline (`radiation_should_compute`
  reads `model_step` from `_date` and `radiation_interval` from the
  term config), not three intermixed substage timelines. Same
  semantics as before, cleaner plumbing.

## Appendix — ECHAM6 reference

The proposed JCM structure mirrors ECHAM6's well-established
operator-split design. The relevant code chain in
`echam6.3.0-ham2.3-moz1.0.r7492/src`:

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

Physics tendencies are accumulated into the gridpoint buffer arrays
declared at `mo_scan_buffer.f90:36-47` (`vol`, `vom`, `tte`, `qte`,
`xlte`, `xite`, `xtte`) and consumed by `sccd` / `scctp` as the
explicit forcing in the semi-implicit Helmholtz solve. **Physics is
called exactly once per dynamics timestep.**

Two structural differences from JCM:

1. **Dynamics integrator.** ECHAM uses leapfrog + semi-implicit
   (spectral); JCM uses IMEX-RK SIL3 (also spectral, via dinosaur).
   The split point is the same — operator-split physics — but the
   dynamics integrator on each side differs. ECHAM's split is forced
   by leapfrog's single RHS evaluation per `dt`; JCM's SIL3 has
   substages and *could* in principle evaluate physics at each one,
   so for JCM operator-splitting is a real design choice. CAM (HOMME
   spectral element), E3SM (HOMME with sub-cycled advection), and IFS
   (semi-Lagrangian + semi-implicit) op-split despite having more
   than one dynamics evaluation per physics `dt` — the same situation
   JCM-with-SIL3 is in.

2. **Within-physics coupling.** ECHAM is **sequential** (each scheme
   reads the state with prior schemes' tendencies already applied via
   the `tte += ...` pattern on the shared accumulators in
   `mo_scan_buffer.f90`). JCM's `ComposablePhysics` is
   **process-parallel** (every term reads the same input state,
   tendencies are summed). The latter preserves the composability
   story (the order-independence of `A + B + C`) at the cost of a
   small accuracy penalty for tightly-coupled term pairs.

The cross-step physics state in ECHAM lives in module-level globals
(`mo_radiation_forcing`, …) — semantically the same as JCM's
`PhysicsCarryState` pytree, just routed via Fortran's module-level
mutability rather than as an explicit functional carry.
