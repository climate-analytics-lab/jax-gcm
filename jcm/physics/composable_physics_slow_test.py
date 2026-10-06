"""Slow integration tests for composable physics.

These tests exercise the composable physics API through the full model
pipeline, verifying that the ``speedy_physics()`` / ``echam_physics()``
factories — and the ``replace``/``remove``/``cloud_scheme="2m"`` container
paths — produce physics that runs a short simulation to a *physical*
atmosphere (no NaNs, plausible mean temperature, moisture present), not
merely one that returns without raising. Each configuration folds a
container operation into its run so three model compiles cover what five
used to.

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


def _assert_physical_atmosphere(testcase, preds, label):
    """Assert the saved trajectory is a physical atmosphere, not just non-None.

    Pins the failure modes ``assertIsNotNone`` cannot see: an all-NaN
    state (the historical blow-up signature), a wildly wrong temperature
    scale (unit errors), and a dried-out or negative moisture field.
    """
    T = np.asarray(preds.dynamics.temperature)
    u = np.asarray(preds.dynamics.u_wind)
    q = np.asarray(preds.dynamics.specific_humidity)
    testcase.assertFalse(np.isnan(T).any(), f"{label}: temperature has NaN")
    testcase.assertFalse(np.isnan(u).any(), f"{label}: u_wind has NaN")
    testcase.assertFalse(np.isnan(q).any(), f"{label}: humidity has NaN")
    testcase.assertGreater(float(T.mean()), 200.0, f"{label}: T mean too cold")
    testcase.assertLess(float(T.mean()), 320.0, f"{label}: T mean too hot")
    testcase.assertGreater(float(q.max()), 0.0, f"{label}: atmosphere is bone dry")


def assert_total_cover_is_aclcov_of_the_saved_fraction(
        testcase, preds, label, require_cloud=False):
    """``clouds.total_cloud_cover`` is saved and is ECHAM's ``aclcov`` of the
    saved ``clouds.cloud_fraction``.

    In a snapshot frame both belong to the same step, and ``cloud_fraction`` is
    the one the microphysics left (its write-back is the last to write it), so
    equality of the cover with the overlap of that fraction says the cover is
    the FINAL fraction's, not the diagnosed fraction's the cover term started
    from. The output carries the field with its CF metadata.

    A cold-started idealised aquaplanet makes no cloud for its first day, and
    there the equality is 0 == 0: it still checks the field is wired into the
    output, with its shape, range and metadata. ``require_cloud`` makes the
    caller's run one that carries cloud, so the equality compares real covers.
    """
    import xarray as xr

    from jcm.analysis import total_cloud_cover

    clouds = preds.physics["clouds"]
    cover = np.asarray(clouds.total_cloud_cover)
    fraction = np.asarray(clouds.cloud_fraction)
    # (time, level, *horizontal) against (time, *horizontal).
    testcase.assertEqual(fraction.shape[2:], cover.shape[1:], label)
    testcase.assertTrue(np.isfinite(cover).all(), f"{label}: cover has NaN")
    testcase.assertGreaterEqual(float(cover.min()), 0.0, label)
    testcase.assertLessEqual(float(cover.max()), 1.0, label)
    offline = total_cloud_cover(xr.DataArray(
        fraction.astype(np.float64),
        dims=("time", "level") + tuple(f"h{i}" for i in range(cover.ndim - 1))))
    np.testing.assert_allclose(cover, np.asarray(offline), atol=1e-5,
                               err_msg=label)
    if require_cloud:
        testcase.assertGreater(
            float(cover.max()), 0.0,
            f"{label}: no cloud anywhere, the comparison is vacuous")

    ds = preds.to_xarray()
    testcase.assertIn("clouds.total_cloud_cover", ds, label)
    testcase.assertEqual(ds["clouds.total_cloud_cover"].attrs["units"], "1")
    testcase.assertEqual(ds["clouds.total_cloud_cover"].attrs["standard_name"],
                         "cloud_area_fraction")


class TestComposableSpeedyIntegration(unittest.TestCase):
    """Integration tests for composable SPEEDY physics."""

    def setUp(self):
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        self.coords = get_speedy_coords(layers=8, spectral_truncation=21)
        self.forcing = ForcingData.zeros(self.coords.horizontal.nodal_shape)
        self.terrain = TerrainData.aquaplanet(self.coords)

    @pytest.mark.slow
    def test_speedy_composable_replace_and_run(self):
        """SPEEDY with a replaced term runs to a physical atmosphere.

        The ``replace`` exercises the container path on top of the plain
        factory run (previously two separate model compiles).
        """
        from jcm.model import Model
        from jcm.physics.speedy.speedy_terms import (
            speedy_physics, SpeedyConvection,
        )

        physics = speedy_physics().replace("convection", SpeedyConvection())
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
        _assert_physical_atmosphere(self, preds, "speedy+replace")

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
    def test_echam_composable_remove_and_run(self):
        """ECHAM with a removed term runs to a physical atmosphere.

        The ``remove`` exercises the container path on top of the plain
        factory run (previously two separate model compiles).
        """
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
        _assert_physical_atmosphere(self, preds, "echam+remove")
        assert_total_cover_is_aclcov_of_the_saved_fraction(
            self, preds, "echam-1m")

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
        _assert_physical_atmosphere(self, preds, "echam-2m")
        assert_total_cover_is_aclcov_of_the_saved_fraction(
            self, preds, "echam-2m")


if __name__ == "__main__":
    unittest.main()
