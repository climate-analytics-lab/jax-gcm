"""κ-Köhler diagnostics: differentiable at an empty mode, in float32.

The forward values are covered by ``placeholder_test`` and
``jam_phase0_test``; what those never checked is the reverse pass at a mode
with no mass or no particles. Measured 2026-10-01 before the fix: in
float32 every ``_jam_state`` field had a NaN gradient there — ``r_dry``
with respect to number, ``rho``/``kappa`` with respect to mass, ``r_wet``
with respect to both — while in float64 all were finite. The cause was a
floored divisor whose square underflows in float32, so a NaN cotangent
reached every upstream parameter through ``where``'s unselected branch.

These tests pin the fix in the dtype the default configuration and CI
actually run.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.microphysics.koehler import equilibrium_modal_state
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name

ACC = MAM4_SPEC.mode("accum")
SHAPE = (2, 1)
FIELDS = ("r_dry", "r_wet", "rho", "kappa", "mass", "number")


def _state(acc_so4_mass, acc_number, saturation=0.8):
    """Every mode populated except as overridden on the accumulation mode."""
    masses = {mass_name(sp, m.short): jnp.full(SHAPE, 1.0e-10)
              for m in MAM4_SPEC.modes for sp in m.species}
    numbers = {number_name(m.short): jnp.full(SHAPE, 1.0e8)
               for m in MAM4_SPEC.modes}
    for sp in ACC.species:
        masses[mass_name(sp, ACC.short)] = jnp.zeros(SHAPE)
    masses[mass_name("so4", ACC.short)] = acc_so4_mass
    numbers[number_name(ACC.short)] = acc_number
    return equilibrium_modal_state(masses, numbers, MAM4_SPEC,
                                   jnp.full(SHAPE, saturation))


class EmptyModeGradientTest(unittest.TestCase):

    def test_gradients_finite_with_no_mass_in_the_mode(self):
        for field in FIELDS:
            grad = jax.grad(lambda m: jnp.sum(
                getattr(_state(m, jnp.full(SHAPE, 1.0e8)), field)))(
                jnp.zeros(SHAPE))
            self.assertTrue(np.all(np.isfinite(np.asarray(grad))),
                            msg=f"d{field}/dmass is not finite at mass=0")

    def test_gradients_finite_with_no_particles_in_the_mode(self):
        for field in FIELDS:
            grad = jax.grad(lambda n: jnp.sum(
                getattr(_state(jnp.full(SHAPE, 1.0e-10), n), field)))(
                jnp.zeros(SHAPE))
            self.assertTrue(np.all(np.isfinite(np.asarray(grad))),
                            msg=f"d{field}/dnumber is not finite at number=0")

    def test_gradients_finite_with_neither(self):
        for field in FIELDS:
            grad = jax.grad(lambda m, n: jnp.sum(
                getattr(_state(m, n), field)), argnums=(0, 1))(
                jnp.zeros(SHAPE), jnp.zeros(SHAPE))
            self.assertTrue(all(np.all(np.isfinite(np.asarray(g))) for g in grad),
                            msg=f"{field} gradient not finite at (0, 0)")


class EmptyModeForwardTest(unittest.TestCase):

    def test_empty_mode_reports_reference_diameter_and_default_density(self):
        state = _state(jnp.zeros(SHAPE), jnp.zeros(SHAPE))
        acc = MAM4_SPEC.mode_index("accum")
        np.testing.assert_allclose(np.asarray(state.r_dry[acc]),
                                   0.5 * ACC.dgnum, rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(state.rho[acc]),
            MAM4_SPEC.species_props(ACC.species[0]).density, rtol=1e-6)
        self.assertTrue(np.all(np.asarray(state.kappa[acc]) == 0.0))

    def test_populated_modes_are_untouched_by_the_empty_one(self):
        # The guards act per mode: an empty accumulation mode must not
        # perturb the Aitken diagnostics beside it.
        empty = _state(jnp.zeros(SHAPE), jnp.zeros(SHAPE))
        full = _state(jnp.full(SHAPE, 1.0e-10), jnp.full(SHAPE, 1.0e8))
        ait = MAM4_SPEC.mode_index("aitken")
        for field in FIELDS:
            np.testing.assert_array_equal(
                np.asarray(getattr(empty, field)[ait]),
                np.asarray(getattr(full, field)[ait]),
                err_msg=f"aitken {field} changed when accum was emptied")


if __name__ == "__main__":
    unittest.main()
