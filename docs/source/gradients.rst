Differentiating through jcm
============================

JAX-GCM is fully differentiable: ``jax.jvp``/``jax.vjp``/``jax.jacfwd``/
``jax.jacrev`` all work through a complete ``Model.run`` integration, not
just through an isolated physics term. This page runs three such examples —
each is literally the code in
``docs/gradients_examples.py``/``docs/gradients_examples_test.py`` (the fast
test suite runs it, under 60 s on CPU, so these cannot go stale the way a
hand-copied snippet could) — and names the two gotchas that bite most often.

For the general JAX pitfalls (``vmap`` of a scalar, ``jax.lax.cond`` instead
of a Python ``if``, NaN gradients where the forward pass is finite, ...), see
:doc:`jax_gotchas` — this page does not repeat it, only adds what is specific
to differentiating *jcm*.

jax.jvp and jax.vjp through a run
----------------------------------

The cheapest full integration to differentiate is Held-Suarez
(``jcm.physics.held_suarez``): Newtonian relaxation + Rayleigh friction, no
boundary-condition files, no moisture physics. The function below perturbs
the model's default initial temperature by a scalar ``delta_k`` (Kelvin) and
returns the global-mean temperature after a one-day run — a stand-in for
"any scalar diagnostic of any jcm run":

.. literalinclude:: ../gradients_examples.py
   :pyobject: held_suarez_temperature_response

``jax.jvp`` and ``jax.vjp`` both work on it exactly as they would on any JAX
function:

.. code-block:: python

    import numpy as np

    f = held_suarez_temperature_response()

    # Forward mode: the directional derivative along a tangent of 1.0.
    primal, tangent = jax.jvp(f, (0.0,), (1.0,))

    # Reverse mode: the gradient, via a cotangent of 1.0 on the scalar output.
    primal, vjp_fun = jax.vjp(f, 0.0)
    (grad,) = vjp_fun(1.0)

    # Forward and reverse mode compute the same derivative by construction,
    # but not bit for bit: forward mode accumulates the directional
    # derivative in one tangent pass over the primal's operations, reverse
    # mode in a separate adjoint pass over them in the opposite order, and
    # the two orders of floating-point summation round differently.
    np.testing.assert_allclose(grad, tangent, rtol=1e-4)

Both calls trace and compile the integration independently — forward mode
and reverse mode are different trace types in JAX, so neither call's
executable is reused by the other. That is also why this page's test splits
the jvp/vjp example from the checker examples below rather than running a
finite-difference check against this same function: a finite-difference
secant needs a THIRD, plain-forward compile, and the three together would
no longer fit the 60 s budget. ``jcm/model_test.py``'s
``test_speedy_model_state_gradient_check`` does run that full combination,
as a ``slow`` test.

jacfwd of a diagnostic w.r.t. a physics parameter
---------------------------------------------------

Differentiable physics parameters are pytree leaves held in an ``nnx.Param``
on the owning ``PhysicsTerm`` (see ``CLAUDE.md``, "Differentiable physics
parameters"), so any of them is reachable from ``jax.grad``/``jacfwd`` the
same way as any other JAX array. The example below differentiates a SPEEDY
run's mean temperature with respect to ``trvdi`` — the moisture-diffusion
relaxation time held on the vertical-diffusion term's parameters
(``jcm.physics.speedy.params.VerticalDiffusionParameters``) — by rebuilding
the physics from a changed ``Parameters`` struct *inside* the traced
function:

.. literalinclude:: ../gradients_examples.py
   :pyobject: speedy_mean_temperature_wrt_trvdi

.. code-block:: python

    diagnostic = speedy_mean_temperature_wrt_trvdi()
    d_mean_temperature_d_trvdi = jax.jacfwd(diagnostic)(6.0)

This is forward-mode over a single scalar input, so it is one jvp-shaped
trace — cheap compared to differentiating w.r.t. a whole ``Parameters``
struct (hundreds of leaves), which is what ``jax.vjp``/``jax.jacrev`` are
for: reverse mode's cost is one pass regardless of the number of inputs, so
a whole-parameter sensitivity sweep (as in jcm's own calibration tooling)
uses ``vjp``, not ``jacfwd``, for exactly that reason. See
``jcm/model_test.py``'s ``test_speedy_model_param_gradients_isnan_vjp`` /
``test_speedy_model_param_gradients_isnan_jvp`` for that whole-struct case,
including the "the gradient must be CONNECTED, not just finite" check that
catches a parameter accidentally marked static.

Checking a gradient: check_jvp, check_vjp, check_gradients
-------------------------------------------------------------

Three checkers compare automatic differentiation against a finite-difference
secant; jcm uses all three, on real physics schemes, in its own test suite.
Their *mechanics* do not depend on what ``f`` computes, so the runnable demo
here uses a cheap non-physics function — see the docstring of
``docs/gradients_examples_test.py`` for why a model-backed version of this
section does not also live in the fast suite (budget, not capability: the
combination works fine in a ``slow`` test, see the pointers below).

``jax.test_util.check_jvp`` / ``check_vjp`` compare a single random
direction against a fixed-step central difference:

.. code-block:: python

    import functools
    from jax.test_util import check_jvp, check_vjp

    f = toy_function_for_check_gradients()
    f_jvp = functools.partial(jax.jvp, f)
    f_vjp = functools.partial(jax.vjp, f)
    check_jvp(f, f_jvp, args=(1.3,), rtol=1e-4)
    check_vjp(f, f_vjp, args=(1.3,), rtol=1e-4)

``jcm.testing.check_gradients`` is the house tool for jcm's own
``PhysicsData``/``PhysicsState``-shaped pytree arguments (CLAUDE.md,
Testing): it names each input leaf by its field path rather than its
position, so adding a sibling field to a struct cannot silently re-roll
every other leaf's probe direction, and it walks a ladder of
finite-difference step sizes rather than trusting one fixed step — see its
docstring and :doc:`jax_gotchas` for the two failure modes ("a branch on a
rounding residue" and "gradients that are NaN where the forward pass is
fine") it exists to catch that a naive central difference would miss or
misdiagnose:

.. code-block:: python

    from jcm import testing

    f = toy_function_for_check_gradients()
    testing.check_gradients(f, (1.3,), rtol=1e-3)

For the full-model versions of all three checkers, see
``jcm/model_test.py`` (``test_speedy_model_state_gradient_check`` for
``check_vjp``/``check_jvp`` through a run; ``test_speedy_model_param_gradients_isnan_vjp``/
``_jvp`` for ``check_gradients``-style connectivity over a whole
``Parameters`` struct) and ``jcm/testing_test.py``.

Two gotchas that matter here
------------------------------

**The reference formulation's derivative is sometimes useless, on purpose.**
Several SPEEDY/ECHAM schemes replace a clip, a threshold switch, or a
singular power law with :func:`jcm.physics.surrogate_gradient.with_surrogate_gradient`:
the VALUE is the reference formulation's, bit for bit, but the DERIVATIVE
(in both AD modes, at every order) is that of a separate, named smooth
function. If a gradient through such a quantity looks suspiciously smooth —
or suspiciously like a single fixed slope regardless of the input — that is
the surrogate at work, not a bug. See
:doc:`design/surrogate_gradients` for when this applies and how to read the
per-scheme table of what uses it.

**predictions.times is not (yet) meaningfully differentiated.** Taking
``jax.jvp``/``jax.vjp`` of a function that returns a full
:class:`jcm.predictions.ModelPredictions` rather than a single array — as
``test_speedy_model_state_gradient_check`` above does — currently gets back
a degenerate (all-zero) tangent/cotangent on the ``.times`` field, since
nothing meaningful should be differentiated with respect to ``save_interval``
or ``total_time`` and ``.times`` is derived from them. Work around it with
``jvp_output.replace(times=predictions.times)`` before converting to xarray,
rather than treating the zeroed field as a real sensitivity.
`Issue #306 <https://github.com/climate-analytics-lab/jax-gcm/issues/306>`_
tracks giving ``.times`` a well-defined (rather than merely zeroed)
derivative; do not depend on that landing — the workaround above is correct
either way.
