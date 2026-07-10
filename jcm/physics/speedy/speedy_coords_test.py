"""Tests for the fixed-sigma evaluation helpers in speedy_coords.py.

SPEEDY's physics schemes evaluate several boundary-layer diagnostics at fixed
sigma surfaces (via :func:`interp_to_sigma` and :data:`PBL_TOP_SIGMA`) so that
their answers are independent of the number of vertical levels. These tests
lock in the two contracts that makes safe:

1. On the 8-level reference grid the fixed sigmas coincide with model layer
   centres, so fixed-sigma evaluation reproduces the validated 8-level level
   indexing (bit-for-bit up to float32 rounding of the sigma table).
2. On any other grid the evaluation is a static, differentiable linear
   combination of the two bracketing levels.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.speedy.speedy_coords import (
    PBL_TOP_SIGMA,
    compute_speedy_vertical_coords,
    interp_to_sigma,
)


class InterpToSigmaTest(unittest.TestCase):
    def test_reproduces_level_indexing_on_8_level_grid(self):
        # The 8-level sigma table has layer centres at exactly the reference
        # sigmas used by the physics (0.835 = second-lowest, 0.95 = lowest), so
        # fixed-sigma evaluation must return those levels' values.
        _, fsg, *_ = compute_speedy_vertical_coords(8)
        field = jnp.arange(8, dtype=jnp.float32) ** 2 + 3.0
        np.testing.assert_allclose(
            interp_to_sigma(field, fsg, PBL_TOP_SIGMA), field[-2], rtol=1e-5)
        np.testing.assert_allclose(
            interp_to_sigma(field, fsg, 0.95), field[-1], rtol=1e-5)
        np.testing.assert_allclose(
            interp_to_sigma(field, fsg, float(fsg[3])), field[3], rtol=1e-5)

    def test_linear_between_levels(self):
        _, fsg, *_ = compute_speedy_vertical_coords(8)
        # On a field linear in sigma, interpolation is exact at any target.
        field = 2.0 * fsg + 1.0
        for sig_t in (0.3, 0.6, PBL_TOP_SIGMA, 0.9):
            np.testing.assert_allclose(
                interp_to_sigma(field, fsg, sig_t), 2.0 * sig_t + 1.0,
                rtol=1e-5)

    def test_leading_axis_only(self):
        # Interpolation acts on the level axis and broadcasts over trailing
        # (horizontal) axes.
        _, fsg, *_ = compute_speedy_vertical_coords(16)
        field = jnp.tile(fsg[:, None, None], (1, 4, 3)) * jnp.arange(3)
        out = interp_to_sigma(field, fsg, 0.7)
        self.assertEqual(out.shape, (4, 3))
        np.testing.assert_allclose(out, 0.7 * jnp.broadcast_to(jnp.arange(3), (4, 3)),
                                   rtol=1e-5)

    def test_differentiable_with_constant_weights(self):
        # The gather indices and weights depend only on the (static) sigma
        # grid, so the derivative w.r.t. the field is a constant two-point
        # stencil with weights summing to one — and never NaN.
        _, fsg, *_ = compute_speedy_vertical_coords(24)
        grad = jax.grad(lambda f: interp_to_sigma(f, fsg, PBL_TOP_SIGMA))(
            jnp.ones(24))
        self.assertFalse(jnp.isnan(grad).any())
        np.testing.assert_allclose(grad.sum(), 1.0, rtol=1e-6)
        self.assertLessEqual(int((grad != 0).sum()), 2)

    def test_surface_flux_lapse_weight_matches_wvi_on_8_levels(self):
        # The surface-flux scheme extrapolates the surface air temperature with
        # the factor (log(0.99) - sigl[-1]) / (sigl[-1] - log(PBL_TOP_SIGMA)).
        # On the 8-level grid, where PBL_TOP_SIGMA is the second-lowest layer
        # centre, this must equal the original interpolation weight wvi[-1, 1],
        # keeping the validated 8-level surface fluxes unchanged.
        _, _, _, sigl, _, _, wvi = compute_speedy_vertical_coords(8)
        fac = (np.log(0.99) - float(sigl[-1])) / (
            float(sigl[-1]) - np.log(PBL_TOP_SIGMA))
        np.testing.assert_allclose(fac, float(wvi[-1, 1]), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
