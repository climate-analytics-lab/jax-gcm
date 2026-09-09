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
a physics call, ``ComposablePhysics`` is **process-parallel** — every term sees
the same input state and the tendencies are summed (order-independent). The full
engineering treatment is {doc}`../design/operator_split_physics`.

**What ECHAM/CAM does.** ECHAM6 op-splits identically: physics tendencies
accumulate into gridpoint buffers (``mo_scan_buffer.f90``) inside ``physc.f90``
(called once per ``dt``), then ``sccd``/``scctp`` consume them as the explicit
forcing of the semi-implicit solve (``stepon.f90``). ECHAM's *within-physics*
coupling is **sequential** — each scheme reads the state with prior tendencies
already applied (``tte += …``). CAM, E3SM and IFS likewise op-split physics as
forcing to the dynamics despite multiple dynamics sub-evaluations per physics
``dt``.

**Why we differ.**
- `science` — within-physics coupling is process-parallel rather than ECHAM's
  sequential accumulation. This preserves the composability algebra
  (``A+B+C == B+A+C`` under ``replace()`` / ``remove()`` / ``__add__()``) at a
  small accuracy cost for tightly-coupled term pairs. Sequential coupling is a
  documented future option.
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
