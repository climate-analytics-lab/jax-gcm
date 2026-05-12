"""Slow integration tests for composable physics.

These tests exercise the composable physics API through the full model
pipeline, verifying that speedy_physics() and echam_physics() factories
produce working physics that can run simulations and compute gradients.

Marked @pytest.mark.slow so they run in PR CI coverage checks.
"""

import unittest

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from jcm.physics_interface import PhysicsState
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.utils import get_coords


class TestComposableSpeedyIntegration(unittest.TestCase):
    """Integration tests for composable SPEEDY physics."""

    def setUp(self):
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        self.coords = get_speedy_coords(layers=8, spectral_truncation=21)
        self.forcing = ForcingData.zeros(self.coords.horizontal.nodal_shape)
        self.terrain = TerrainData.aquaplanet(self.coords)

    @pytest.mark.slow
    def test_speedy_composable_model_run(self):
        """Composable SPEEDY runs a short simulation through the Model."""
        from jcm.model import Model
        from jcm.physics.speedy.speedy_terms import speedy_physics

        physics = speedy_physics()
        model = Model(
            coords=self.coords,
            terrain=self.terrain,
            physics=physics,
        )
        preds = model.run(
            forcing=self.forcing,
            save_interval=1.0,
            total_time=1.0,
        )
        self.assertIsNotNone(preds)

    @pytest.mark.slow
    def test_speedy_composable_replace_and_run(self):
        """Replace a SPEEDY term and run through Model."""
        from jcm.model import Model
        from jcm.physics.speedy.speedy_terms import (
            speedy_physics, SpeedyConvection,
        )

        physics = speedy_physics()
        replaced = physics.replace(
            "convection", SpeedyConvection(),
        )
        model = Model(
            coords=self.coords,
            terrain=self.terrain,
            physics=replaced,
        )
        preds = model.run(
            forcing=self.forcing,
            save_interval=1.0,
            total_time=1.0,
        )
        self.assertIsNotNone(preds)

    @pytest.mark.slow
    def test_speedy_composable_gradient(self):
        """Gradients flow through composable SPEEDY physics."""
        from jcm.physics.speedy.speedy_terms import speedy_physics

        composable = speedy_physics(checkpoint_terms=False)
        composable.cache_coords(self.coords)
        state = self._make_state()

        def loss_fn(physics):
            tend, _ = physics.compute_tendencies(state, self.forcing, self.terrain)
            return jnp.sum(tend.temperature ** 2)

        grads = nnx.grad(loss_fn)(composable)
        grad_leaves = jax.tree_util.tree_leaves(grads)
        any_nonzero = any(
            jnp.any(leaf != 0.0) for leaf in grad_leaves
            if hasattr(leaf, 'shape')
        )
        self.assertTrue(any_nonzero, "All gradients are zero")

    def _make_state(self):
        nlev = self.coords.nodal_shape[0]
        nodal_shape = self.coords.horizontal.nodal_shape
        shape = (nlev,) + nodal_shape
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)
        return PhysicsState(
            u_wind=5.0 * jax.random.normal(keys[0], shape),
            v_wind=5.0 * jax.random.normal(keys[1], shape),
            temperature=250.0 + 20.0 * jax.random.normal(keys[2], shape),
            specific_humidity=jnp.abs(
                3.0 * jax.random.normal(keys[3], shape),
            ),
            geopotential=jnp.broadcast_to(
                jnp.linspace(50000, 0, nlev)[:, None, None], shape,
            ),
            normalized_surface_pressure=(
                1.0 + 0.01 * jax.random.normal(keys[4], nodal_shape)
            ),
        )


class TestComposableEchamIntegration(unittest.TestCase):
    """Integration tests for composable ECHAM physics."""

    def setUp(self):
        sigma_boundaries = np.linspace(0, 1, 9)  # 8 levels
        self.coords = get_coords(
            sigma_boundaries, nodal_shape=(64, 32),
        )
        self.terrain = TerrainData.aquaplanet(self.coords)
        self.forcing = ForcingData.zeros((64, 32))

    @pytest.mark.slow
    def test_echam_composable_model_run(self):
        """Composable ECHAM runs a short simulation through the Model."""
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics

        physics = echam_physics()
        model = Model(
            coords=self.coords,
            terrain=self.terrain,
            physics=physics,
        )
        preds = model.run(
            forcing=self.forcing,
            save_interval=1.0,
            total_time=1.0,
        )
        self.assertIsNotNone(preds)

    @pytest.mark.slow
    def test_echam_composable_remove_and_run(self):
        """Remove a term from ECHAM physics and run."""
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics

        physics = echam_physics().remove("gravity_waves")
        model = Model(
            coords=self.coords,
            terrain=self.terrain,
            physics=physics,
        )
        preds = model.run(
            forcing=self.forcing,
            save_interval=1.0,
            total_time=1.0,
        )
        self.assertIsNotNone(preds)

    @pytest.mark.slow
    def test_echam_2m_composable_model_run(self):
        """Composable ECHAM with 2-moment microphysics runs through Model.

        Exercises ``cloud_microphysics_2m`` (and the cloud_utils helpers)
        end-to-end so the slow-test coverage gate sees them. The ``qc``,
        ``qi``, ``qnc``, ``qni``, ``qr``, ``qs`` tracers are auto-zeroed by
        ``apply_microphysics_2m`` if the initial state doesn't supply them.
        """
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics

        physics = echam_physics(cloud_scheme="2m")
        model = Model(
            coords=self.coords,
            terrain=self.terrain,
            physics=physics,
        )
        preds = model.run(
            forcing=self.forcing,
            save_interval=1.0,
            total_time=1.0,
        )
        self.assertIsNotNone(preds)

    @pytest.mark.slow
    def test_packages_factories(self):
        """Package factory re-exports work end-to-end."""
        from jcm.physics.speedy.speedy_terms import speedy_physics
        from jcm.physics.echam.echam_terms import echam_physics

        sp = speedy_physics()
        self.assertGreater(len(sp.terms), 0)

        ip = echam_physics()
        self.assertGreater(len(ip.terms), 0)

    def _make_state(self):
        nlev = 8
        shape = (nlev, 64, 32)
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 6)
        return PhysicsState(
            temperature=250.0 + 20.0 * jax.random.normal(
                keys[0], shape,
            ),
            specific_humidity=jnp.abs(
                3.0 * jax.random.normal(keys[1], shape),
            ),
            u_wind=5.0 * jax.random.normal(keys[2], shape),
            v_wind=5.0 * jax.random.normal(keys[3], shape),
            geopotential=jnp.broadcast_to(
                jnp.linspace(50000, 0, nlev)[:, None, None], shape,
            ),
            normalized_surface_pressure=(
                1.0 + 0.01 * jax.random.normal(keys[4], (64, 32))
            ),
            tracers={
                "qc": jnp.abs(
                    1e-4 * jax.random.normal(keys[5], shape),
                ),
                "qi": jnp.zeros(shape),
            },
        )


if __name__ == "__main__":
    unittest.main()
