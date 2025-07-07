import unittest

class TestSpeedyPhysicsUnit(unittest.TestCase):
    def setUp(self):
        global PhysicsState, SpeedyPhysics, BoundaryData, Parameters, Geometry, DateData
        from jcm.physics_interface import PhysicsState
        from jcm.physics.speedy.speedy_physics import SpeedyPhysics
        from jcm.boundaries import BoundaryData
        from jcm.params import Parameters
        from jcm.geometry import Geometry
        from jcm.date import DateData

    def test_speedy_forcing(self):
        grid_shape = (8,1,2)
        tendencies, data = SpeedyPhysics().compute_tendencies(
            state=PhysicsState.zeros(grid_shape),
            boundaries=BoundaryData.ones(grid_shape[1:]),
            geometry=Geometry.from_grid_shape(grid_shape[1:], grid_shape[0]),
            date=DateData.zeros()
        )
        self.assertIsNotNone(tendencies)
        self.assertIsNotNone(data)