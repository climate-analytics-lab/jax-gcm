import unittest

class TestSpeedyPhysicsUnit(unittest.TestCase):
    def setUp(self):
        global PhysicsState, SpeedyPhysics, ForcingData, Parameters, terrain, DateData, coords, ix, il, kx
        ix, il, kx = 96, 48, 8

        from jcm.physics_interface import PhysicsState
        from jcm.physics.speedy.speedy_physics import SpeedyPhysics
        from jcm.forcing import ForcingData
        from jcm.physics.speedy.params import Parameters
        from jcm.date import DateData
        from jcm.terrain import TerrainData
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        coords = get_speedy_coords(layers=kx, nodal_shape=(ix, il))
        terrain = TerrainData.aquaplanet(coords)

    def test_speedy_forcing(self):
        grid_shape = (kx,ix,il)
        physics = SpeedyPhysics()
        physics.cache_coords(coords)
        tendencies, data = physics.compute_tendencies(
            state=PhysicsState.zeros(grid_shape),
            forcing=ForcingData.ones(grid_shape[1:]),
            terrain=terrain,
            date=DateData.zeros()
        )
        self.assertIsNotNone(tendencies)
        self.assertIsNotNone(data)