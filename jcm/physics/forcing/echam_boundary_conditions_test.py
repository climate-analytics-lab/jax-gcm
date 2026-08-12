"""Tests for the EchamBoundaryConditions surface-optics parameters (#347)."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.forcing.echam_boundary_conditions import (
    SurfaceOpticsParameters,
    _surface_optical_properties,
)


class SurfaceOpticsTest(unittest.TestCase):
    def test_pure_surface_types_return_per_type_values(self):
        p = SurfaceOpticsParameters()
        # Columns: all-land, all-ocean, all-sea-ice.
        vis, nir, emis = _surface_optical_properties(
            jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0]), p,
        )
        np.testing.assert_allclose(vis, [0.15, 0.05, 0.80])
        np.testing.assert_allclose(nir, [0.25, 0.05, 0.70])
        np.testing.assert_allclose(emis, [0.95, 0.98, 0.95])

    def test_custom_parameters_are_honoured(self):
        p = SurfaceOpticsParameters(land_albedo_vis=0.30)
        vis, _, _ = _surface_optical_properties(
            jnp.array([1.0]), jnp.array([0.0]), p,
        )
        np.testing.assert_allclose(vis, [0.30])

    def test_parameters_are_differentiable(self):
        # The per-type values must be gradient leaves like every other
        # physics parameter: d(albedo_vis)/d(type value) = type fraction.
        def total_vis(p):
            vis, _, _ = _surface_optical_properties(
                jnp.array([0.4]), jnp.array([0.1]), p,
            )
            return vis.sum()

        g = jax.grad(total_vis)(SurfaceOpticsParameters())
        self.assertAlmostEqual(float(g.land_albedo_vis), 0.4, places=6)
        self.assertAlmostEqual(float(g.ocean_albedo_vis), 0.5, places=6)
        self.assertAlmostEqual(float(g.seaice_albedo_vis), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
