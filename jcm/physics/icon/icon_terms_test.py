"""Tests for composable ICON physics (icon_terms.py).

Verifies numerical equivalence between composable icon_physics() and the
original IconPhysics class, and tests mixed-package composition.

Date: 2026-04-13
"""

import unittest

import numpy as np
import jax
import jax.numpy as jnp
import numpy.testing as npt
from flax import nnx

from jcm.physics_interface import PhysicsState
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.date import DateData
from jcm.utils import get_coords


def _make_icon_test_setup(nlev=8, nlat=64, nlon=32):
    """Create test setup matching ICON conventions."""
    sigma_boundaries = np.linspace(0, 1, nlev + 1)
    coords = get_coords(sigma_boundaries, nodal_shape=(nlat, nlon))
    terrain = TerrainData.aquaplanet(coords)
    forcing = ForcingData.zeros((nlat, nlon))
    date = DateData.zeros()

    shape_3d = (nlev, nlat, nlon)
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 6)

    state = PhysicsState(
        temperature=250.0 + 20.0 * jax.random.normal(
            keys[0], shape_3d,
        ),
        specific_humidity=jnp.abs(
            3.0 * jax.random.normal(keys[1], shape_3d),
        ),
        u_wind=5.0 * jax.random.normal(keys[2], shape_3d),
        v_wind=5.0 * jax.random.normal(keys[3], shape_3d),
        geopotential=jnp.broadcast_to(
            jnp.linspace(50000, 0, nlev)[:, None, None],
            shape_3d,
        ),
        normalized_surface_pressure=(
            1.0
            + 0.01 * jax.random.normal(keys[4], (nlat, nlon))
        ),
        tracers={
            "qc": jnp.abs(
                1e-4 * jax.random.normal(keys[5], shape_3d),
            ),
            "qi": jnp.zeros(shape_3d),
        },
    )

    return coords, state, forcing, terrain, date


class TestIconComposablePhysics(unittest.TestCase):
    """Test composable ICON physics wrapper."""

    def setUp(self):
        """Set up test fixtures."""
        self.coords, self.state, self.forcing, self.terrain, self.date = (
            _make_icon_test_setup()
        )

    def test_icon_physics_factory(self):
        """icon_physics() creates composable physics with correct terms."""
        from jcm.physics.icon.icon_terms import icon_physics

        physics = icon_physics(checkpoint_terms=False)
        self.assertEqual(len(physics.terms), 10)
        categories = [t.category for t in physics.terms]
        self.assertIn("radiation", categories)
        self.assertIn("convection", categories)
        self.assertIn("surface", categories)

    def test_tendency_equivalence(self):
        """Composable ICON tendencies must match original IconPhysics."""
        from jcm.physics.icon.icon_physics import IconPhysics
        from jcm.physics.icon.icon_terms import icon_physics
        from jcm.physics.icon.parameters import Parameters

        params = Parameters.default()

        # Original IconPhysics
        original = IconPhysics(
            parameters=params, checkpoint_terms=False,
        )
        original.cache_coords(self.coords)
        orig_tend, _ = original.compute_tendencies(
            self.state, self.forcing, self.terrain, self.date,
        )

        # Composable version
        composable = icon_physics(
            parameters=params, checkpoint_terms=False,
        )
        composable.cache_coords(self.coords)
        comp_tend, _ = composable.compute_tendencies(
            self.state, self.forcing, self.terrain, self.date,
        )

        # Compare all tendency fields.
        # ICON physics produces NaN at some grid points for random
        # test states; mask those and verify the rest match.
        for name in ("temperature", "specific_humidity",
                      "u_wind", "v_wind"):
            comp = getattr(comp_tend, name)
            orig = getattr(orig_tend, name)
            # NaN locations must match
            npt.assert_array_equal(
                jnp.isnan(comp), jnp.isnan(orig),
                err_msg=f"NaN locations differ for {name}",
            )
            # Non-NaN values must be close
            mask = ~jnp.isnan(orig)
            if jnp.any(mask):
                npt.assert_allclose(
                    comp[mask], orig[mask],
                    rtol=1e-3, atol=1e-6,
                    err_msg=f"{name} tendencies differ",
                )

    def test_composable_with_model(self):
        """Composable ICON physics works with Model."""
        from jcm.model import Model
        from jcm.physics.icon.icon_terms import icon_physics

        composable = icon_physics()
        model = Model(
            coords=self.coords,
            terrain=self.terrain,
            physics=composable,
        )
        preds = model.run(
            forcing=self.forcing,
            save_interval=1.0,
            total_time=1.0,
        )
        self.assertIsNotNone(preds)

    def test_replace_radiation(self):
        """Can replace ICON radiation with a different scheme."""
        from jcm.physics.icon.icon_terms import (
            icon_physics,
            IconRadiation,
        )

        composable = icon_physics(checkpoint_terms=False)
        composable.cache_coords(self.coords)

        # Replace radiation with a fresh instance
        new_rad = IconRadiation()
        replaced = composable.replace("radiation", new_rad)
        replaced.cache_coords(self.coords)

        tend, _ = replaced.compute_tendencies(
            self.state, self.forcing, self.terrain, self.date,
        )
        # Check shape is correct (NaNs expected with random state)
        self.assertEqual(
            tend.temperature.shape, self.state.temperature.shape,
        )

    def test_nnx_split_merge(self):
        """nnx.split/merge works for ICON composable physics."""
        from jcm.physics.icon.icon_terms import icon_physics

        composable = icon_physics(checkpoint_terms=False)
        composable.cache_coords(self.coords)

        # Verify split/merge roundtrip works
        graphdef, state = nnx.split(composable)
        restored = nnx.merge(graphdef, state)
        self.assertEqual(len(restored.terms), 10)


if __name__ == "__main__":
    unittest.main()
