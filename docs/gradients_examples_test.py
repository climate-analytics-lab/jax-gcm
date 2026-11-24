"""Runs the examples ``docs/source/gradients.rst`` literalincludes (#241).

Keeps the documented examples honest: if a refactor breaks ``jax.jvp``/
``jax.vjp``/``jacfwd`` through jcm, this test (part of the ordinary fast
suite, not ``slow``) fails instead of the doc quietly going stale.

Runtime split (CPU), why it is shaped this way: ``jax.jvp``/``jax.vjp``/
``jacfwd`` through jcm's ``Model.run`` each force their OWN XLA compile of
the (tiny, one-timestep) integration — forward-mode, reverse-mode and plain
forward are different trace types, so none of the three compiles is reused
by the others. Adding a finite-difference check (``check_jvp``/
``check_vjp``/``jcm.testing.check_gradients``) on TOP of the model example
would add a fourth (plain-forward, needed for the secant) and push this
file over the 60 s budget (#241) — measured at ~64 s with it included. So
the finite-difference-checker demos below run on a cheap non-physics
function; ``jcm/model_test.py``'s ``test_speedy_model_state_gradient_check``
/ ``test_speedy_model_param_gradients_isnan_vjp`` are the full-model worked
examples for those same checkers, already covered by ``slow`` tests there —
this file does not duplicate them.
"""

import functools
import unittest

import jax
import numpy as np
from jax.test_util import check_jvp, check_vjp

import gradients_examples as examples
from jcm import testing


class JvpVjpThroughHeldSuarezTest(unittest.TestCase):
    """jax.jvp and jax.vjp through a short jcm run, and that they agree."""

    @classmethod
    def setUpClass(cls):
        # staticmethod: a plain function assigned as a class attribute is
        # bound as a method on instance access (self.f would then be called
        # with an extra leading ``self`` argument), which breaks jax.jvp/
        # vjp's own call of it below. Built once per class so both AD
        # modes' compiles (not reused between each other) are each paid
        # for exactly once.
        cls.f = staticmethod(examples.held_suarez_temperature_response())

    def test_jvp_and_vjp_agree(self):
        # ~25 s total: jax.vjp's and jax.jvp's compiles of the (tiny,
        # 1-timestep) Held-Suarez integration are each their own trace type,
        # so neither call reuses the other's executable.
        primal_vjp, vjp_fun = jax.vjp(self.f, 0.0)
        (grad,) = vjp_fun(1.0)
        primal_jvp, tangent = jax.jvp(self.f, (0.0,), (1.0,))

        self.assertEqual(primal_vjp, primal_jvp)
        self.assertTrue(np.isfinite(primal_vjp))
        # Nonzero and forward/reverse-consistent: a disconnected gradient
        # (e.g. a stray stop_gradient) would pass "finite" but fail this.
        self.assertGreater(abs(float(grad)), 0.0)
        np.testing.assert_allclose(float(grad), float(tangent), rtol=1e-4)


class JacfwdThroughAPhysicsParameterTest(unittest.TestCase):
    """jacfwd of a diagnostic w.r.t. one nnx.Param-held physics parameter."""

    def test_jacfwd_wrt_trvdi_is_finite_and_connected(self):
        # ~20 s: one SPEEDY timestep, forward-mode over the single scalar
        # input ``trvdi`` (so this is one jvp-shaped trace, not a full
        # Jacobian over every parameter the way jcm/model_test.py's
        # whole-Parameters-tree vjp check is).
        diagnostic = examples.speedy_mean_temperature_wrt_trvdi()
        d_dtrvdi = jax.jacfwd(diagnostic)(6.0)
        self.assertTrue(np.isfinite(d_dtrvdi))
        self.assertNotEqual(float(d_dtrvdi), 0.0)


class CheckersApiTest(unittest.TestCase):
    """check_jvp / check_vjp / jcm.testing.check_gradients call patterns.

    On a cheap non-physics function — see the module docstring for why a
    model-backed finite-difference check does not also live in this file.
    Each checker's *mechanics* (AD vs. a finite-difference secant) is the
    same regardless of what ``f`` computes.
    """

    def test_check_jvp_and_check_vjp(self):
        f = examples.toy_function_for_check_gradients()
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f)
        check_jvp(f, f_jvp, args=(1.3,), rtol=1e-4)
        check_vjp(f, f_vjp, args=(1.3,), rtol=1e-4)

    def test_check_gradients(self):
        f = examples.toy_function_for_check_gradients()
        testing.check_gradients(f, (1.3,), rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
