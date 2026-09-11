# Operator splitting

**What we do.** Physics is evaluated exactly once per dynamics timestep ``dt``,
outside the dycore's internal integration stages, and its tendency is applied as
a **Lie split** ahead of the dynamics integral (``state → physics → dynamics →
next``, splitting error ``O(dt)``). The per-``dt`` closure
(``jcm/model.py::Model._get_op_split_step_fn``) resolves the date/forcing slice,
projects the native state with ``dycore.to_physics_state``, calls
``jcm/physics_interface.py::compute_physics_step_gridpoint`` for
``(physics_tendency, new_physics_state)``, then hands the gridpoint tendency to
``dycore.step``, which converts it to the backend-native representation, advances
dynamics and applies backend filters. Trajectories are a nested ``lax.scan`` with
``jax.checkpoint`` on each inner step; both snapshot and time-averaged output
modes save the cross-step physics carry the integration actually consumed. Within
a physics call, ``ComposablePhysics`` is **state-parallel but
diagnostics-sequential**: every term reads the same input *state* and the
tendencies are summed, but terms run in list order and each receives the
``diagnostics`` dict its predecessors produced (validated against declared
``requires``/``provides``). Tightly-coupled pairs use that channel to couple
sequentially anyway — vertical diffusion and convection publish their updated
thermodynamics via ``advance_thermo_run``
(``jcm/physics/diagnostics/moist_air_state.py``), which downstream convection
and microphysics consume — so term order is **not** free:
``echam_physics()`` ships the validated ECHAM ``physc`` sequence, and
reordering coupled terms (e.g. convection before vertical diffusion) is a
known-unstable configuration. The full engineering treatment is
{doc}`../design/operator_split_physics`.

**What ECHAM/CAM does.** ECHAM6 op-splits identically: physics tendencies
accumulate into gridpoint buffers (``mo_scan_buffer.f90``) inside ``physc.f90``
(called once per ``dt``), then ``sccd``/``scctp`` consume them as the explicit
forcing of the semi-implicit solve (``stepon.f90``). ECHAM's *within-physics*
coupling is **sequential** — each scheme reads the state with prior tendencies
already applied (``tte += …``). CAM, E3SM and IFS likewise op-split physics as
forcing to the dynamics despite multiple dynamics sub-evaluations per physics
``dt``.

**Why we differ.**
- `science` — the *state* is not sequentially updated between terms (tendencies
  sum against one input state), unlike ECHAM's ``tte += …`` accumulation. This
  keeps the ``replace()`` / ``remove()`` / ``__add__()`` composition algebra
  simple for loosely-coupled terms, at a small accuracy cost; where the
  sequential coupling *matters* (vdiff → convection → clouds thermodynamics) it
  is restored explicitly through the ``advance_thermo_run`` diagnostics channel,
  which is why the shipped orderings are load-bearing rather than arbitrary.
- `compute` / `differentiability` — one physics call per ``dt`` (rather than one
  per RK substage) keeps the autodiff tape small under ``jax.checkpoint``, and
  the ``physics_state`` carry is threaded as an explicit JAX pytree rather than
  ECHAM's module-level globals, so the step is a pure function.

**Status & known limitations.** Only Lie splitting is implemented; Strang
splitting (``O(dt²)``, 2× physics cost) is a documented follow-up should a
coarser-``dt`` regime expose the coupling error. At the current climate-rate
``dt = 12–30 min`` Lie is adequate. There is no flag to restore an in-stage
scheme.

**Code pointers.**
- ``jcm/model.py`` — ``_get_op_split_step_fn``, ``_op_split_trajectory``,
  ``_build_initial_physics_carry``.
- ``jcm/physics_interface.py`` — ``compute_physics_step_gridpoint``,
  ``verify_state``, ``verify_tendencies``.
- ``jcm/dycore/base.py`` — ``DynamicalCore.step``.

**Validation evidence.** ``jcm/model_test.py`` (operator-split physics: snapshot
and averaged modes for SPEEDY and ECHAM-hybrid, step purity, carry threading,
``run(5d)+resume(5d) == run(10d)``). Design reference:
{doc}`../design/operator_split_physics`.
