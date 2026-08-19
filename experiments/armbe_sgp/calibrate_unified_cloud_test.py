import unittest

import jax
import jax.numpy as jnp
import numpy as np

from calibrate_unified_cloud import NESTED_INITIAL, nested_symbolic_cloud


class NestedSymbolicCloudTest(unittest.TestCase):
    def setUp(self):
        self.features = {
            "rh_low_mean": jnp.asarray([0.7]),
            "rh_mid_mean": jnp.asarray([0.5]),
            "rh_vertical_range": jnp.asarray([0.4]),
            "rh_high_mean": jnp.asarray([0.2]),
        }

    def test_initial_parameters_reproduce_selected_equation(self):
        actual = nested_symbolic_cloud(self.features, jnp.asarray(NESTED_INITIAL))
        expected = jnp.tanh((0.7 + 0.5) * (0.4**3 + 0.2))

        np.testing.assert_allclose(actual, expected)

    def test_all_parameters_have_finite_nonzero_gradients(self):
        gradient = jax.grad(
            lambda params: nested_symbolic_cloud(self.features, params)[0]
        )(jnp.asarray(NESTED_INITIAL))

        self.assertTrue(np.isfinite(gradient).all())
        self.assertTrue(np.all(np.asarray(gradient) != 0.0))


if __name__ == "__main__":
    unittest.main()
