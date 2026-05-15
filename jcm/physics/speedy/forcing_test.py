"""Tests for jcm/physics/speedy/forcing.py"""

import unittest
import jax.numpy as jnp

from jcm.physics.speedy.forcing import set_forcing
from jcm.physics.speedy.speedy_coords import get_speedy_coords, SpeedyCoords
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.physics.speedy.params import Parameters
from jcm.physics_interface import PhysicsState
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.date import DateData


class TestSetForcing(unittest.TestCase):

    def setUp(self):
        ix, il, kx = 96, 48, 8
        self.coords = get_speedy_coords(layers=kx, nodal_shape=(ix, il))
        speedy_coords = SpeedyCoords.from_coordinate_system(self.coords)
        self.terrain = TerrainData.aquaplanet(self.coords)
        self.parameters = Parameters.default()
        self.state = PhysicsState.zeros((kx, ix, il))
        self.physics_data = PhysicsData.zeros(
            (ix, il), kx, date=DateData.zeros(), speedy_coords=speedy_coords
        )

    def test_set_forcing_propagates_ablco2(self):
        """ablco2 from ForcingData should appear in mod_radcon after set_forcing."""
        forcing = ForcingData.zeros(
            self.coords.horizontal.nodal_shape, ablco2=12.0
        )
        _, out_data = set_forcing(
            self.state, self.physics_data, self.parameters,
            forcing=forcing, terrain=self.terrain
        )
        self.assertTrue(jnp.allclose(out_data.mod_radcon.ablco2, 12.0))

    def test_set_forcing_default_ablco2(self):
        """Default ablco2 (6.0) should pass through unchanged."""
        forcing = ForcingData.zeros(self.coords.horizontal.nodal_shape)
        _, out_data = set_forcing(
            self.state, self.physics_data, self.parameters,
            forcing=forcing, terrain=self.terrain
        )
        self.assertTrue(jnp.allclose(out_data.mod_radcon.ablco2, 6.0))


if __name__ == '__main__':
    unittest.main()
