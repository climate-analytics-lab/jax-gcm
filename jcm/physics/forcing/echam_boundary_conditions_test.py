"""Tests for EchamBoundaryConditions' surface optics (#347, #703, #672)."""
import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.forcing import ForcingData, SolarGeometry
from jcm.physics.forcing.echam_boundary_conditions import (
    EchamBoundaryConditions,
    SurfaceOpticsParameters,
    _surface_optical_properties,
    sea_ice_surface_temperature,
)
from jcm.physics.radiation import current_cos_zenith
from jcm.physics.surface.echam import albedo as albedo_scheme
from jcm.physics_interface import PhysicsState

P = SurfaceOpticsParameters()


def _optics(land, ice, **overrides):
    """Surface optics at hard frost, overhead sun, snow-free 0.2 background."""
    n = jnp.broadcast_shapes(jnp.shape(land), jnp.shape(ice))
    kwargs = dict(
        background_albedo=jnp.full(n, 0.2),
        snow_fraction=jnp.zeros(n),
        land_temperature=jnp.full(n, 250.0),
        ice_temperature=jnp.full(n, albedo_scheme.CTFREEZ),
        cos_zenith=jnp.ones(n),
    )
    kwargs.update(overrides)
    p = kwargs.pop("p", P)
    return _surface_optical_properties(land, ice, p, **kwargs)


class SurfaceOpticsTest(unittest.TestCase):
    def test_pure_tiles_return_the_echam_tile_albedos(self):
        vis, nir, emis = _optics(jnp.array([1.0, 0.0, 0.0]),
                                 jnp.array([0.0, 0.0, 1.0]))
        ocean_vis, ocean_nir = albedo_scheme.ocean_albedo_per_band(
            jnp.ones(1), P.albedo)
        # Land: snow-free background, broadband in both bands. Sea ice at
        # ctfreez: calbmxi. Ocean: the zenith-dependent open-water albedo.
        np.testing.assert_allclose(vis, [0.2, float(ocean_vis[0]), 0.75],
                                   rtol=1e-6)
        np.testing.assert_allclose(nir, [0.2, float(ocean_nir[0]), 0.75],
                                   rtol=1e-6)
        np.testing.assert_allclose(emis, [0.95, 0.98, 0.95])

    def test_polar_land_flagged_as_sea_ice_stays_a_convex_blend(self):
        # The bundle carries icec = 1 over Antarctic/Arctic land, where
        # lsm = 1 as well. Unclipped, the tiles summed to 2 and emissivity
        # reached 1.90 -- so surface reflectance 1 - eps went negative and
        # RRTMGP was handed an impossible surface (#703).
        vis, nir, emis = _optics(jnp.array([1.0]), jnp.array([1.0]))
        np.testing.assert_allclose(emis, [0.95])
        np.testing.assert_allclose(vis, [0.2], rtol=1e-6)
        np.testing.assert_allclose(nir, [0.2], rtol=1e-6)

    def test_blend_is_convex_across_the_whole_fraction_plane(self):
        f = jnp.linspace(0.0, 1.0, 11)
        land, ice = (x.ravel() for x in jnp.meshgrid(f, f))
        vis, nir, emis = _optics(land, ice)
        ocean_vis, ocean_nir = albedo_scheme.ocean_albedo_per_band(
            jnp.ones(1), P.albedo)
        for got, tiles in (
            (vis, (0.2, float(ocean_vis[0]), 0.75)),
            (nir, (0.2, float(ocean_nir[0]), 0.75)),
            (emis, (0.95, 0.98, 0.95)),
        ):
            self.assertGreaterEqual(float(jnp.min(got)), min(tiles) - 1e-6)
            self.assertLessEqual(float(jnp.max(got)), max(tiles) + 1e-6)

    def test_background_albedo_and_snow_cover_reach_the_land_albedo(self):
        # Before #672 the land albedo was a constant 0.15/0.25 that ignored
        # both fields; now alb0 is the snow-free value and snowc_am
        # brightens it towards the temperature-dependent snow albedo.
        land, ice = jnp.ones(3), jnp.zeros(3)
        vis, nir, _ = _optics(land, ice,
                              background_albedo=jnp.array([0.1, 0.3, 0.1]),
                              snow_fraction=jnp.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(vis, [0.1, 0.3, 0.8], rtol=1e-6)
        np.testing.assert_allclose(nir, vis, rtol=1e-6)

    def test_glacier_and_forest_maps_reach_the_land_albedo(self):
        land, ice = jnp.ones(2), jnp.zeros(2)
        vis, _, _ = _optics(land, ice, glacier_fraction=jnp.array([1.0, 0.0]))
        np.testing.assert_allclose(vis, [0.85, 0.2], rtol=1e-6)
        snowy, _, _ = _optics(land, ice, snow_fraction=jnp.ones(2),
                              forest_fraction=jnp.array([0.0, 1.0]))
        self.assertAlmostEqual(float(snowy[0]), 0.8, places=6)
        self.assertLess(float(snowy[1]), 0.3)

    def test_ocean_albedo_follows_the_sun(self):
        ocean, ice = jnp.zeros(2), jnp.zeros(2)
        vis, _, _ = _optics(ocean, ice, cos_zenith=jnp.array([1.0, 0.2]))
        self.assertLess(float(vis[0]), float(vis[1]))

    def test_sea_ice_tile_temperature_is_min_sst_ctfreez(self):
        t = sea_ice_surface_temperature(jnp.array([275.0, 270.0, 280.0]),
                                        jnp.array([0.5, 0.5, 0.0]))
        np.testing.assert_allclose(t, [albedo_scheme.CTFREEZ, 270.0, 280.0])

    def test_parameters_are_differentiable(self):
        def total_vis(p):
            vis, _, emis = _optics(jnp.array([0.4]), jnp.array([0.1]), p=p,
                                   snow_fraction=jnp.array([0.5]))
            return vis.sum() + emis.sum()

        g = jax.grad(total_vis)(P)
        # d(vis)/d(snow_albedo_max) = land fraction · snow fraction.
        self.assertAlmostEqual(float(g.albedo.snow_albedo_max), 0.2, places=6)
        # Ice at ctfreez sits at calbmxi: d/d(calbmxi) = ice fraction.
        self.assertAlmostEqual(float(g.albedo.seaice_albedo_bare_max), 0.1,
                               places=6)
        self.assertAlmostEqual(float(g.land_emissivity), 0.4, places=6)
        self.assertAlmostEqual(float(g.ocean_emissivity), 0.5, places=6)


def _term_inputs(ncols=4, nlev=3, **forcing_overrides):
    """Minimal (state, diagnostics, forcing, terrain) for a direct term call."""
    state = PhysicsState.zeros((nlev, ncols),
                               temperature=jnp.full((nlev, ncols), 250.0))
    p_full = jnp.linspace(2.0e4, 9.5e4, nlev)[:, None] * jnp.ones((1, ncols))
    diagnostics = {"pressure_full": p_full,
                   "surface_pressure": jnp.full((ncols,), 1.0e5)}
    fields = dict(
        alb0=jnp.full((ncols,), 0.2),
        snowc_am=jnp.zeros((ncols,)),
        stl_am=jnp.full((ncols,), 250.0),
        sea_surface_temperature=jnp.full((ncols,), 290.0),
        solar=SolarGeometry.zero(),
    )
    fields.update(forcing_overrides)
    forcing = ForcingData.zeros(nodal_shape=(ncols,), **fields)
    terrain = types.SimpleNamespace(fmask=jnp.ones((ncols,)))
    return state, diagnostics, forcing, terrain


def _cached_term(ncols=4):
    term = EchamBoundaryConditions()
    horizontal = types.SimpleNamespace(
        latitudes=jnp.linspace(-1.2, 1.2, ncols),
        longitudes=jnp.zeros(1),
        nodal_shape=(1, ncols),
    )
    term.cache_coords(types.SimpleNamespace(horizontal=horizontal))
    return term


class EchamBoundaryConditionsTermTest(unittest.TestCase):
    def test_alb0_and_snowc_change_the_radiation_input(self):
        term = _cached_term()
        base = term(*_term_inputs())[1]["radiation"]
        np.testing.assert_allclose(base.surface_albedo_vis, 0.2, rtol=1e-6)
        brighter = term(*_term_inputs(alb0=jnp.full((4,), 0.3)))[1]
        np.testing.assert_allclose(brighter["radiation"].surface_albedo_vis,
                                   0.3, rtol=1e-6)
        snowy = term(*_term_inputs(snowc_am=jnp.full((4,), 0.5)))[1]
        np.testing.assert_allclose(snowy["radiation"].surface_albedo_nir,
                                   0.5 * 0.8 + 0.5 * 0.2, rtol=1e-6)

    def test_land_cover_maps_are_read_from_the_forcing(self):
        term = _cached_term()
        rad = term(*_term_inputs(
            glacier_fraction=jnp.array([1.0, 0.0, 0.0, 0.0])))[1]["radiation"]
        np.testing.assert_allclose(rad.surface_albedo_vis,
                                   [0.85, 0.2, 0.2, 0.2], rtol=1e-6)

    def test_snow_cover_above_one_is_treated_as_full_cover(self):
        # Legacy packaged files can carry snowc outside [0, 1].
        term = _cached_term()
        rad = term(*_term_inputs(snowc_am=jnp.full((4,), 3.0)))[1]["radiation"]
        np.testing.assert_allclose(rad.surface_albedo_vis, 0.8, rtol=1e-6)

    def test_call_without_cached_coordinates_is_a_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "cache_coords"):
            EchamBoundaryConditions()(*_term_inputs())

    def test_ocean_albedo_uses_the_columns_solar_zenith(self):
        term = _cached_term()
        state, diagnostics, forcing, _ = _term_inputs()
        ocean = types.SimpleNamespace(fmask=jnp.zeros((4,)))
        rad = term(state, diagnostics, forcing, ocean)[1]["radiation"]
        lats = jnp.linspace(-1.2, 1.2, 4)
        mu = current_cos_zenith(forcing.solar, jnp.zeros(4),
                                lats * 180.0 / jnp.pi)
        expected, _ = albedo_scheme.ocean_albedo_per_band(mu, P.albedo)
        np.testing.assert_allclose(rad.surface_albedo_vis, expected,
                                   rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
