"""Tests for composable ECHAM physics (echam_terms.py).

Tests mixed-package composition, replacement of individual terms, and
roundtripping through nnx.split / nnx.merge.
"""

import unittest

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from jcm.physics_interface import PhysicsState
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.date import DateData
from jcm.utils import get_coords
from jcm.physics.physics_term import PhysicsTerm


class DummyRadiationTerm(PhysicsTerm):
    """Minimal radiation term used to test ECHAM factory dispatch."""

    name = "dummy_radiation"
    category = "radiation"


class DummyConvectionTerm(PhysicsTerm):
    """Minimal non-radiation term used to test factory validation."""

    name = "dummy_convection"
    category = "convection"


def _make_echam_test_setup(nlev=8, nlat=64, nlon=32):
    """Create test setup matching ECHAM conventions."""
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


class TestEchamComposablePhysics(unittest.TestCase):
    """Test composable ECHAM physics wrapper."""

    def setUp(self):
        """Set up test fixtures."""
        self.coords, self.state, self.forcing, self.terrain, self.date = (
            _make_echam_test_setup()
        )

    def test_echam_physics_factory(self):
        """echam_physics() creates composable physics with correct terms."""
        from jcm.physics.echam.echam_terms import echam_physics

        physics = echam_physics(checkpoint_terms=False)
        # Cloud fraction and microphysics are separate terms; the GWD
        # category split adds Hines + SSO (the simple-GWD scheme is kept
        # available but excluded from the default factory).
        self.assertEqual(len(physics.terms), 12)
        categories = [t.category for t in physics.terms]
        self.assertIn("radiation", categories)
        self.assertIn("convection", categories)
        self.assertIn("surface", categories)
        self.assertIn("cloud_fraction", categories)
        self.assertIn("clouds", categories)
        self.assertIn("hines", categories)
        self.assertIn("sso", categories)
        self.assertNotIn("simple_gwd", categories)
        # Cloud fraction must precede microphysics so the microphysics term
        # can read the post-condensation qc/qi/cloud_fraction diagnostics.
        self.assertLess(
            categories.index("cloud_fraction"),
            categories.index("clouds"),
        )

    def test_echam_physics_accepts_custom_radiation_term(self):
        """A radiation PhysicsTerm can be passed directly."""
        from jcm.physics.echam.echam_terms import echam_physics

        custom_rad = DummyRadiationTerm()
        physics = echam_physics(
            checkpoint_terms=False,
            radiation_scheme=custom_rad,
        )

        self.assertIs(physics.terms[4], custom_rad)
        self.assertEqual(physics.terms[4].category, "radiation")

    def test_echam_physics_rejects_non_radiation_custom_term(self):
        """Custom radiation terms must declare the radiation category."""
        from jcm.physics.echam.echam_terms import echam_physics

        with self.assertRaisesRegex(ValueError, "category 'radiation'"):
            echam_physics(
                checkpoint_terms=False,
                radiation_scheme=DummyConvectionTerm(),
            )

    def test_column_vector_handles_vmap_scalar_shapes(self):
        """Radiation scalar diagnostics are normalized to [ncols]."""
        from jcm.physics.radiation.grey_two_stream.radiation_scheme import (
            _column_vector,
        )

        self.assertEqual(_column_vector(jnp.arange(3), 3).shape, (3,))
        self.assertEqual(
            _column_vector(jnp.arange(3).reshape(3, 1), 3).shape,
            (3,),
        )

    def test_composable_with_model(self):
        """Composable ECHAM physics works with Model."""
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics

        composable = echam_physics()
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
        """Can replace radiation with a different scheme."""
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics.radiation.grey_two_stream import (
            GreyTwoStreamRadiation,
        )

        composable = echam_physics(checkpoint_terms=False)
        composable.cache_coords(self.coords)

        # Replace radiation with a fresh instance
        new_rad = GreyTwoStreamRadiation()
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
        """nnx.split/merge works for ECHAM composable physics."""
        from jcm.physics.echam.echam_terms import echam_physics

        composable = echam_physics(checkpoint_terms=False)
        composable.cache_coords(self.coords)

        # Verify split/merge roundtrip works
        graphdef, state = nnx.split(composable)
        restored = nnx.merge(graphdef, state)
        self.assertEqual(len(restored.terms), 12)


if __name__ == "__main__":
    unittest.main()
