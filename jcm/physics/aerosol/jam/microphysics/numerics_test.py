"""The float32-safe quotient: same value, same derivative, no overflow."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.microphysics import numerics


class DivideTest(unittest.TestCase):

    def test_value_and_derivative_match_the_plain_quotient(self):
        x, y = jnp.asarray(3.0), jnp.asarray(7.0)
        self.assertEqual(float(numerics.divide(x, y)), float(x / y))
        plain = jax.grad(lambda x, y: x / y, argnums=(0, 1))(x, y)
        safe = jax.grad(numerics.divide, argnums=(0, 1))(x, y)
        np.testing.assert_allclose(np.asarray(safe), np.asarray(plain), rtol=1e-6)
        # And forward mode.
        _, plain_t = jax.jvp(lambda x, y: x / y, (x, y), (jnp.asarray(1.0), jnp.asarray(0.5)))
        _, safe_t = jax.jvp(numerics.divide, (x, y), (jnp.asarray(1.0), jnp.asarray(0.5)))
        self.assertAlmostEqual(float(safe_t), float(plain_t), places=6)

    def test_reverse_mode_survives_a_small_denominator_and_a_large_cotangent(self):
        """The measured failure: two 1e-15 limits under a 1e19 cotangent."""
        a, b = jnp.float32(7.972e-16), jnp.float32(6.931e-15)
        cot = 1e19
        plain = jax.grad(lambda a, b: cot * (a * b / (a + b)), argnums=(0, 1))(a, b)
        self.assertFalse(all(bool(jnp.isfinite(g)) for g in plain))   # the trap, pinned
        safe = jax.grad(lambda a, b: cot * numerics.harmonic_mean(a, b), argnums=(0, 1))(a, b)
        self.assertTrue(all(bool(jnp.isfinite(g)) for g in safe))
        # Against the closed form d(ab/(a+b))/da = b²/(a+b)².
        expected = cot * float(b) ** 2 / (float(a) + float(b)) ** 2
        self.assertAlmostEqual(float(safe[0]) / expected, 1.0, places=4)

    def test_harmonic_mean_of_nothing_is_zero_with_finite_gradient(self):
        g = jax.grad(lambda a, b: numerics.harmonic_mean(a, b), argnums=(0, 1))(
            jnp.asarray(0.0), jnp.asarray(0.0))
        self.assertTrue(all(bool(jnp.isfinite(v)) for v in g))
        self.assertEqual(float(numerics.harmonic_mean(jnp.asarray(0.0), jnp.asarray(0.0))), 0.0)


if __name__ == "__main__":
    unittest.main()
