"""AER-02: the log-normal moment algebra, pinned by its own identities.

Property tests rather than golden numbers: the moment relations are exact
mathematics, so the right check is that the functions satisfy the
identities a log-normal distribution must, not that they reproduce a
number somebody once wrote down.
"""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.microphysics import moments
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.population import AerosolMode
from jcm.physics.aerosol.jam.tracer_layout import mass_name
from jcm.testing import check_gradients

ACC = MAM4_SPEC.mode("accum")
AIT = MAM4_SPEC.mode("aitken")


def _lognormal_quadrature(number, dg, sigma_g, k, n=20001):
    """Brute-force M_k by integrating the log-normal density over ln D.

    The independent reference the closed form is checked against.
    """
    ln_sigma = math.log(sigma_g)
    x = np.linspace(-10 * ln_sigma, 10 * ln_sigma, n)          # ln(D/Dg)
    density = np.exp(-0.5 * (x / ln_sigma) ** 2) / (ln_sigma * math.sqrt(2 * math.pi))
    return number * np.trapezoid((dg * np.exp(x)) ** k * density, x)


class MomentIdentityTest(unittest.TestCase):

    def test_zeroth_moment_is_number(self):
        n, dg = jnp.full((3, 2), 2.0e8), jnp.full((3, 2), 1.5e-7)
        np.testing.assert_allclose(moments.moment(n, dg, ACC, 0.0), n, rtol=0)

    def test_closed_form_matches_quadrature(self):
        # Fractional and integer orders, both modes' widths.
        for mode in (ACC, AIT):
            for k in (0.0, 1.0, 1.5, 2.0, 3.0):
                closed = float(moments.moment(
                    jnp.asarray(1.0e8), jnp.asarray(mode.dgnum), mode, k))
                brute = _lognormal_quadrature(1.0e8, mode.dgnum,
                                              mode.geom_std_dev, k)
                self.assertAlmostEqual(closed / brute, 1.0, places=6,
                                       msg=f"{mode.name} k={k}")

    def test_moment_factor_grows_with_order_squared(self):
        # ln(f_k) = k^2 ln^2(sigma)/2, so ln f_2 / ln f_1 == 4 exactly.
        f1 = math.log(moments.moment_factor(ACC, 1.0))
        f2 = math.log(moments.moment_factor(ACC, 2.0))
        self.assertAlmostEqual(f2 / f1, 4.0, places=12)

    def test_monodisperse_limit(self):
        # sigma_g -> 1 collapses every moment to N * Dg^k.
        mono = AerosolMode(name="m", short="m", geom_std_dev=1.0,
                           dgnum=1e-7, dgnum_lo=1e-8, dgnum_hi=1e-6,
                           species=("so4",), soluble=True,
                           can_activate=True, sediments=True)
        self.assertEqual(moments.moment_factor(mono, 3.0), 1.0)
        self.assertAlmostEqual(
            moments.volume_factor(mono), math.pi / 6.0, places=15)


class VolumeDiameterRoundTripTest(unittest.TestCase):

    def test_diameter_from_volume_inverts_volume_from_diameter(self):
        rng = np.random.default_rng(0)
        for mode in (ACC, AIT):
            # Stay strictly inside the clip window so the clip is inert.
            dg = jnp.asarray(rng.uniform(mode.dgnum_lo * 1.05,
                                         mode.dgnum_hi * 0.95, (4, 3)))
            n = jnp.asarray(rng.uniform(1e6, 1e10, (4, 3)))
            vol = moments.volume_from_diameter(n, dg, mode)
            back = moments.diameter_from_volume(vol, n, mode)
            np.testing.assert_allclose(np.asarray(back), np.asarray(dg),
                                       rtol=1e-5)

    def test_volume_is_sigma_consistent_with_third_moment(self):
        n, dg = jnp.asarray(3.0e8), jnp.asarray(2.0e-7)
        via_moment = (math.pi / 6.0) * moments.moment(n, dg, ACC, 3.0)
        direct = moments.volume_from_diameter(n, dg, ACC)
        self.assertAlmostEqual(float(via_moment / direct), 1.0, places=5)

    def test_clip_to_the_mode_window(self):
        n = jnp.asarray(1.0e8)
        huge = moments.diameter_from_volume(jnp.asarray(1.0), n, ACC)
        tiny = moments.diameter_from_volume(jnp.asarray(1e-40), n, ACC)
        np.testing.assert_allclose(float(huge), ACC.dgnum_hi, rtol=1e-6)
        np.testing.assert_allclose(float(tiny), ACC.dgnum_lo, rtol=1e-6)

    def test_empty_mode_is_finite_in_value_and_gradient(self):
        """The sharp edge: zero number and zero volume together."""
        def dg(vol, n):
            return moments.diameter_from_volume(vol, n, ACC)
        value = dg(jnp.asarray(0.0), jnp.asarray(0.0))
        self.assertTrue(np.isfinite(float(value)))
        g = jax.grad(dg, argnums=(0, 1))(jnp.asarray(0.0), jnp.asarray(0.0))
        self.assertTrue(all(np.isfinite(float(x)) for x in g))

    def test_gradient_matches_finite_difference_inside_the_window(self):
        n = jnp.asarray(2.0e8)
        vol_mid = moments.volume_from_diameter(n, jnp.asarray(ACC.dgnum), ACC)
        check_gradients(lambda v: moments.diameter_from_volume(v, n, ACC),
                        (vol_mid,), rtol=1e-3)


class SurfaceAreaTest(unittest.TestCase):

    def test_is_pi_times_second_moment(self):
        n, dg = jnp.asarray(5.0e8), jnp.asarray(1.2e-7)
        self.assertAlmostEqual(
            float(moments.surface_area(n, dg, ACC)
                  / (math.pi * moments.moment(n, dg, ACC, 2.0))),
            1.0, places=5)

    def test_monodisperse_sphere_area(self):
        mono = AerosolMode(name="m", short="m", geom_std_dev=1.0,
                           dgnum=1e-7, dgnum_lo=1e-8, dgnum_hi=1e-6,
                           species=("so4",), soluble=True,
                           can_activate=True, sediments=True)
        n, dg = jnp.asarray(1.0), jnp.asarray(2.0e-7)
        np.testing.assert_allclose(float(moments.surface_area(n, dg, mono)),
                                   math.pi * (2.0e-7) ** 2, rtol=1e-6)


class ModeSumTest(unittest.TestCase):

    def test_mode_volume_sums_species_over_density(self):
        masses = {mass_name(sp, ACC.short): jnp.asarray(float(i + 1) * 1e-10)
                  for i, sp in enumerate(ACC.species)}
        expected = sum(float(masses[mass_name(sp, ACC.short)])
                       / MAM4_SPEC.species_props(sp).density
                       for sp in ACC.species)
        self.assertAlmostEqual(
            float(moments.mode_volume(masses, ACC, MAM4_SPEC)) / expected,
            1.0, places=5)

    def test_mode_mass_sums_species(self):
        masses = {mass_name(sp, AIT.short): jnp.asarray(2.0e-10)
                  for sp in AIT.species}
        np.testing.assert_allclose(float(moments.mode_mass(masses, AIT)),
                                   2.0e-10 * len(AIT.species), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
