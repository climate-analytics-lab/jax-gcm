"""Tests for composable SPEEDY physics (speedy_terms.py).

Smoke tests: build via the speedy_physics() factory, run through Model,
exercise nnx.grad and the replace() composition operator.
"""

import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.speedy.params import Parameters
from jcm.physics_interface import PhysicsState
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.date import DateData


def _make_test_state(coords):
    """Create a test PhysicsState with physically plausible values."""
    nlev = coords.nodal_shape[0]
    nodal_shape = coords.horizontal.nodal_shape
    shape_3d = (nlev,) + nodal_shape

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 6)

    return PhysicsState(
        u_wind=5.0 * jax.random.normal(keys[0], shape_3d),
        v_wind=5.0 * jax.random.normal(keys[1], shape_3d),
        temperature=250.0 + 20.0 * jax.random.normal(keys[2], shape_3d),
        specific_humidity=jnp.abs(3.0 * jax.random.normal(keys[3], shape_3d)),
        geopotential=jnp.broadcast_to(
            jnp.linspace(50000, 0, nlev)[:, None, None], shape_3d
        ),
        normalized_surface_pressure=(
            1.0 + 0.01 * jax.random.normal(keys[4], nodal_shape)
        ),
    )


class TestSpeedyNumericalEquivalence(unittest.TestCase):
    """Smoke tests for composable SPEEDY physics."""

    def setUp(self):
        self.coords = get_speedy_coords(layers=8, spectral_truncation=21)
        self.params = Parameters.default()
        self.state = _make_test_state(self.coords)
        self.forcing = ForcingData.zeros(self.coords.horizontal.nodal_shape)
        self.terrain = TerrainData.aquaplanet(self.coords)
        self.date = DateData.zeros()

    def test_composable_with_model(self):
        """ComposablePhysics can be passed to Model and run."""
        from jcm.model import Model

        composable = speedy_physics(parameters=self.params)
        model = Model(
            coords=self.coords,
            terrain=self.terrain,
            physics=composable,
        )
        # Just verify it doesn't crash during a very short run
        preds = model.run(
            forcing=self.forcing,
            save_interval=1.0,
            total_time=1.0,
        )
        self.assertIsNotNone(preds)

    def test_nnx_grad_through_composable_speedy(self):
        """Gradients flow through the composable SPEEDY physics."""
        composable = speedy_physics(parameters=self.params, checkpoint_terms=False)
        composable.cache_coords(self.coords)

        def loss_fn(physics):
            tend, _ = physics.compute_tendencies(self.state, self.forcing, self.terrain)
            return jnp.sum(tend.temperature ** 2)

        grads = nnx.grad(loss_fn)(composable)

        # Verify at least some parameter gradients are non-zero
        grad_leaves = jax.tree_util.tree_leaves(grads)
        any_nonzero = any(
            jnp.any(leaf != 0.0) for leaf in grad_leaves
            if hasattr(leaf, 'shape')
        )
        self.assertTrue(any_nonzero, "All gradients are zero")

    def test_replace_term(self):
        """Verify we can replace a SPEEDY term category."""
        composable = speedy_physics(parameters=self.params, checkpoint_terms=False)

        # Replace convection with different convection params
        from jcm.physics.speedy.speedy_terms import SpeedyConvection
        from jcm.physics.speedy.params import ConvectionParameters

        new_conv_params = ConvectionParameters(
            psmin=jnp.array(0.8),
            trcnv=jnp.array(12.0),  # doubled relaxation time
            rhil=jnp.array(0.7),
            rhbl=jnp.array(0.9),
            entmax=jnp.array(0.5),
            smf=jnp.array(0.8),
        )
        replaced = composable.replace("convection", SpeedyConvection(new_conv_params))

        replaced.cache_coords(self.coords)
        tend, _ = replaced.compute_tendencies(self.state, self.forcing, self.terrain)

        # Should produce valid (non-NaN) tendencies
        self.assertFalse(jnp.any(jnp.isnan(tend.temperature)))


if __name__ == "__main__":
    unittest.main()
