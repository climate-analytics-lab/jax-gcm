"""Tests for the smooth differentiable vertical injection profile (#498)."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.emissions.injection import (
    gaussian_injection_weights,
)

# Column: level 0 = top, level -1 = surface. Heights [m] and thicknesses [m].
_Z = jnp.asarray([4000.0, 2000.0, 1000.0, 300.0, 50.0])[:, None]  # (nlev, 1)
_DZ = jnp.asarray([2000.0, 1200.0, 700.0, 300.0, 100.0])[:, None]


class InjectionProfileTest(unittest.TestCase):
    def test_weights_sum_to_one(self):
        for h in (0.0, 50.0, 1000.0, 3000.0):
            w = gaussian_injection_weights(_Z, _DZ, jnp.asarray(h), jnp.asarray(50.0))
            np.testing.assert_allclose(float(jnp.sum(w)), 1.0, rtol=1e-5)

    def test_surface_injection_loads_lowest_layer(self):
        w = gaussian_injection_weights(_Z, _DZ, jnp.asarray(50.0), jnp.asarray(40.0))
        self.assertEqual(int(jnp.argmax(w[:, 0])), _Z.shape[0] - 1)

    def test_elevated_injection_shifts_upward(self):
        low = gaussian_injection_weights(_Z, _DZ, jnp.asarray(50.0), jnp.asarray(100.0))
        high = gaussian_injection_weights(_Z, _DZ, jnp.asarray(2000.0), jnp.asarray(100.0))
        # Mass-weighted mean injection height rises with injection_height.
        zbar_low = float(jnp.sum(low[:, 0] * _Z[:, 0]))
        zbar_high = float(jnp.sum(high[:, 0] * _Z[:, 0]))
        self.assertGreater(zbar_high, zbar_low)

    def test_grad_wrt_height_and_thickness_finite_and_nonzero(self):
        def mean_height(h, t):
            w = gaussian_injection_weights(_Z, _DZ, h, t)
            return jnp.sum(w[:, 0] * _Z[:, 0])

        gh = jax.grad(mean_height, argnums=0)(jnp.asarray(800.0), jnp.asarray(300.0))
        gt = jax.grad(mean_height, argnums=1)(jnp.asarray(800.0), jnp.asarray(300.0))
        self.assertTrue(np.isfinite(float(gh)) and abs(float(gh)) > 0.0)
        self.assertTrue(np.isfinite(float(gt)))


if __name__ == "__main__":
    unittest.main()
