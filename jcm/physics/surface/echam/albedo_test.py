"""ECHAM 6.3 surface albedo schemes pinned at hand-evaluated reference points.

Every expected value below is the Fortran formula evaluated by hand with the
constants of the cited routine, so a change to a constant or to the branch
structure shows up as a named failure rather than a drift.
"""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics.surface.echam import albedo as a

P = a.EchamSurfaceAlbedoParameters()
TMELT = c.tmelt


class LandAlbedoTest(unittest.TestCase):
    """JSBACH ``update_land_surface_fast`` with lctlib_nlct21 constants."""

    def test_snow_free_land_is_the_background(self):
        got = a.land_albedo(jnp.array([0.12, 0.35]), 0.0,
                            jnp.array([250.0, 300.0]), P)
        np.testing.assert_allclose(got, [0.12, 0.35], rtol=1e-6)

    def test_open_snow_ramps_from_0p4_at_tmelt_to_0p8_five_kelvin_below(self):
        t = jnp.array([TMELT + 5.0, TMELT, TMELT - 2.5, TMELT - 5.0, 240.0])
        got = a.land_albedo(0.2, 1.0, t, P)
        np.testing.assert_allclose(got, [0.4, 0.4, 0.6, 0.8, 0.8], rtol=1e-6)

    def test_partial_snow_blends_linearly_with_the_background(self):
        # sf·a_s + (1-sf)·bg at hard frost: 0.3·0.8 + 0.7·0.15.
        got = a.land_albedo(0.15, 0.3, 250.0, P)
        np.testing.assert_allclose(got, 0.3 * 0.8 + 0.7 * 0.15, rtol=1e-6)

    def test_forest_masks_the_snow_with_the_lai_floor(self):
        # ff = forest·(1 - exp(-max(lai, 2))) with lai below the floor;
        # snow-free canopy: (1-ff)·a_s + ff·bg.
        ff = 0.6 * (1.0 - np.exp(-2.0))
        expected = (1.0 - ff) * 0.8 + ff * 0.1
        got = a.land_albedo(0.1, 1.0, 250.0, P, forest_fraction=0.6,
                            leaf_area_index=1.0)
        np.testing.assert_allclose(got, expected, rtol=1e-6)
        # A denser canopy (lai above the floor) masks more.
        denser = a.land_albedo(0.1, 1.0, 250.0, P, forest_fraction=0.6,
                               leaf_area_index=4.0)
        self.assertLess(float(denser), float(got))

    def test_snow_covered_canopy_uses_the_canopy_snow_albedo(self):
        ff = 1.0 - np.exp(-2.0)
        expected = (1.0 - ff) * 0.8 + ff * 0.20
        got = a.land_albedo(0.1, 1.0, 250.0, P, forest_fraction=1.0,
                            canopy_snow_fraction=1.0)
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_albedo_never_falls_below_the_background(self):
        # A bright desert under melting snow: the MAX(..., bg) clamp holds
        # it at the background rather than darkening it to 0.4.
        got = a.land_albedo(0.55, 1.0, TMELT + 1.0, P)
        np.testing.assert_allclose(got, 0.55, rtol=1e-6)

    def test_glacier_ramps_between_0p75_and_0p85(self):
        t = jnp.array([TMELT, TMELT - 2.5, TMELT - 10.0])
        got = a.land_albedo(0.1, 0.0, t, P, glacier_fraction=1.0)
        np.testing.assert_allclose(got, [0.75, 0.80, 0.85], rtol=1e-6)

    def test_fractional_glacier_is_the_tile_average(self):
        got = a.land_albedo(0.2, 0.0, 250.0, P, glacier_fraction=0.25)
        np.testing.assert_allclose(got, 0.25 * 0.85 + 0.75 * 0.2, rtol=1e-6)

    def test_column_and_broadcast_shapes_agree(self):
        bg = jnp.array([[0.1, 0.2], [0.3, 0.15]])
        sf = jnp.array([[0.0, 0.5], [1.0, 0.2]])
        t = jnp.array([[260.0, 272.0], [250.0, 280.0]])
        whole = a.land_albedo(bg, sf, t, P, forest_fraction=0.3)
        for i in range(2):
            for j in range(2):
                one = a.land_albedo(bg[i, j], sf[i, j], t[i, j], P,
                                    forest_fraction=0.3)
                np.testing.assert_allclose(whole[i, j], one, rtol=1e-6)


class SeaIceAlbedoTest(unittest.TestCase):
    """``mo_surface_ice.f90::update_albedo_ice`` at T63 (init_albedo_ice)."""

    def test_bare_ice_ramps_over_one_kelvin(self):
        t = jnp.array([TMELT + 1.0, TMELT, TMELT - 0.5, TMELT - 1.0, 240.0])
        got = a.sea_ice_albedo(t, P)
        np.testing.assert_allclose(got, [0.60, 0.60, 0.675, 0.75, 0.75],
                                   rtol=1e-6)

    def test_snow_on_ice_uses_calbmns_calbmxs_above_one_centimetre(self):
        t = jnp.array([TMELT, TMELT - 0.5, 240.0])
        np.testing.assert_allclose(a.sea_ice_albedo(t, P, snow_depth=0.02),
                                   [0.70, 0.775, 0.85], rtol=1e-6)
        # 1 cm is the threshold itself: still bare.
        np.testing.assert_allclose(a.sea_ice_albedo(240.0, P, snow_depth=0.01),
                                   0.75, rtol=1e-6)

    def test_prescribed_ice_at_ctfreez_sits_at_the_cold_maximum(self):
        # The ice tile runs at min(SST, ctfreez) = 271.38 K, below tmelt - 1.
        np.testing.assert_allclose(a.sea_ice_albedo(a.CTFREEZ, P), 0.75,
                                   rtol=1e-6)

    def test_t31_constants(self):
        p31 = a.EchamSurfaceAlbedoParameters.echam_t31()
        np.testing.assert_allclose(a.sea_ice_albedo(TMELT, p31), 0.55)
        np.testing.assert_allclose(a.sea_ice_albedo(240.0, p31), 0.75)
        np.testing.assert_allclose(
            a.sea_ice_albedo(240.0, p31, snow_depth=1.0), 0.80)


class OceanAlbedoTest(unittest.TestCase):
    """``mo_surface_ocean.f90::update_albedo_ocean``."""

    @staticmethod
    def _zalw(mu):
        return (0.026 / (mu ** 1.7 + 0.065)
                + 0.015 * (mu - 0.1) * (mu - 0.5) * (mu - 1.0))

    def test_direct_beam_fit_and_band_offsets(self):
        mu = np.array([0.2, 0.5, 1.0])
        vis, nir, dif = a.ocean_albedo(jnp.asarray(mu), P)
        np.testing.assert_allclose(vis, self._zalw(mu) + 0.0082, rtol=1e-5)
        np.testing.assert_allclose(nir, self._zalw(mu) - 0.007, rtol=1e-5)
        np.testing.assert_allclose(dif, 0.07, rtol=1e-6)
        # Overhead sun: ~0.024 direct, glancing sun far brighter.
        self.assertAlmostEqual(float(self._zalw(1.0)), 0.02441, places=4)
        self.assertGreater(float(vis[0]), 0.2)

    def test_dark_columns_carry_the_diffuse_value(self):
        vis, nir, dif = a.ocean_albedo(jnp.array([-0.3, 0.0]), P)
        np.testing.assert_allclose(vis, 0.07)
        np.testing.assert_allclose(nir, 0.07)

    def test_rce_switch_replaces_the_fit_with_0p07(self):
        prce = a.EchamSurfaceAlbedoParameters(rce=True)
        vis, nir, _ = a.ocean_albedo(jnp.array([0.2, 1.0]), prce)
        np.testing.assert_allclose(vis, 0.07 + 0.0082, rtol=1e-6)
        np.testing.assert_allclose(nir, 0.07 - 0.007, rtol=1e-6)

    def test_per_band_merge_weights_direct_and_diffuse(self):
        mu = jnp.array([0.5])
        vis_d, nir_d, dif = a.ocean_albedo(mu, P)
        vis, nir = a.ocean_albedo_per_band(mu, P)
        w = a.OCEAN_DIRECT_WEIGHT
        np.testing.assert_allclose(vis, w * vis_d + (1 - w) * dif, rtol=1e-6)
        np.testing.assert_allclose(nir, w * nir_d + (1 - w) * dif, rtol=1e-6)


class DifferentiabilityTest(unittest.TestCase):
    def test_constants_are_gradient_leaves(self):
        def total(p):
            land = a.land_albedo(0.2, 0.5, TMELT - 2.5, p,
                                 forest_fraction=0.4, glacier_fraction=0.1)
            ice = a.sea_ice_albedo(TMELT - 0.5, p)
            vis, nir = a.ocean_albedo_per_band(jnp.array(0.6), p)
            return land + ice + vis + nir

        g = jax.grad(total)(P)
        leaves = jax.tree_util.tree_leaves(g)
        self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in leaves))
        self.assertNotEqual(float(g.snow_albedo_max), 0.0)
        self.assertNotEqual(float(g.glacier_albedo_min), 0.0)
        self.assertNotEqual(float(g.seaice_albedo_bare_max), 0.0)
        self.assertNotEqual(float(g.ocean_albedo_diffuse), 0.0)
        self.assertNotEqual(float(g.ocean_direct_a), 0.0)

    def test_ad_matches_a_central_difference(self):
        # Operating point inside every ramp (no kink in reach of the steps).
        from jcm.testing import check_gradients

        def f(p, bg, sf, t_land, t_ice, mu):
            land = a.land_albedo(bg, sf, t_land, p, forest_fraction=0.3,
                                 glacier_fraction=0.2)
            vis, nir = a.ocean_albedo_per_band(mu, p)
            return land, a.sea_ice_albedo(t_ice, p), vis, nir

        args = (P, jnp.array([0.15, 0.3]), jnp.array([0.4, 0.7]),
                jnp.array([TMELT - 2.0, TMELT - 3.5]),
                jnp.array([TMELT - 0.4, TMELT - 0.6]), jnp.array([0.3, 0.8]))
        check_gradients(f, args, rtol=2e-2)

    def test_parameter_gradients_are_finite_on_the_night_side(self):
        # mu0**b has d/db = mu0**b·ln(mu0), 0·(-inf) at mu0 = 0.
        def f(p):
            vis, nir = a.ocean_albedo_per_band(jnp.array([-0.4, 0.0, 0.3]), p)
            return vis.sum() + nir.sum()

        g = jax.grad(f)(P)
        for leaf in jax.tree_util.tree_leaves(g):
            self.assertTrue(np.isfinite(np.asarray(leaf)).all())
        self.assertNotEqual(float(g.ocean_direct_b), 0.0)

    def test_state_gradients_are_finite_at_the_branch_points(self):
        # Temperatures exactly on the ramp ends and a zero zenith.
        def f(t, mu):
            return (a.land_albedo(0.2, 1.0, t, P).sum()
                    + a.sea_ice_albedo(t, P).sum()
                    + a.ocean_albedo_per_band(mu, P)[0].sum())

        t = jnp.array([TMELT, TMELT - 5.0, TMELT - 1.0])
        gt, gmu = jax.grad(f, argnums=(0, 1))(t, jnp.array([0.0, 0.5]))
        self.assertTrue(np.isfinite(np.asarray(gt)).all())
        self.assertTrue(np.isfinite(np.asarray(gmu)).all())


if __name__ == "__main__":
    unittest.main()
