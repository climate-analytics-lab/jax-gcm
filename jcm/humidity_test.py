import unittest
from jcm import humidity
from jcm.physics_data import ConvectionData, HumidityData
import jax.numpy as jnp
from jcm.physics import PhysicsData, PhysicsState
from jcm.params import kx #FIXME: doing this because we need to have the same number of levels as fsg, need to call intialize_geometry() 
# spec_hum_to_rel_hum no longer takes sigma as an argument. it does all sigma levels

class TestHumidityUnit(unittest.TestCase):

    def setUp(self):
        # Initialize test parameters here if needed
        self.temp_standard = jnp.array([[[273] * 96] * 48])
        self.pressure_standard = jnp.array([[[0.5] * 96] * 48])
        self.sigma = 4
        self.qg_standard = jnp.array([[[2] * 96] * 48])

    def test_get_qsat(self):
        temp = self.temp_standard
        pressure = self.pressure_standard
        sigma = self.sigma
        qsat = humidity.get_qsat(temp, pressure, sigma)

        self.assertIsNotNone(qsat)
        self.assertTrue((qsat >= 0).all(), "Found negative qsat values")

        # Edge case: Very low temperature
        temp = jnp.array([[[100] * 96] * 48])
        qsat = humidity.get_qsat(temp, pressure, sigma)
        self.assertIsNotNone(qsat)
        self.assertTrue((qsat >= 0).all(), "Found negative qsat values at low temperature")

        # Edge case: Very high temperature
        temp = jnp.array([[[350] * 96] * 48])
        qsat = humidity.get_qsat(temp, pressure, sigma)
        self.assertIsNotNone(qsat)
        self.assertTrue((qsat >= 0).all(), "Found negative qsat values at high temperature")

    def test_spec_hum_to_rel_hum(self):
        temp = self.temp_standard
        pressure = self.pressure_standard
        qg = self.qg_standard

        humidity_data = HumidityData((96,48), kx)
        convection_data = ConvectionData((96,48), kx,psa=pressure)
        physics_data = PhysicsData((96,48), kx, convection=convection_data, humidity=humidity_data)
        state = PhysicsState(jnp.zeros_like(temp), jnp.zeros_like(temp), temp, qg, jnp.zeros_like(temp),jnp.zeros((96,48)))

        _, physics_data = humidity.spec_hum_to_rel_hum(physics_data=physics_data, state=state)
        self.assertIsNotNone(physics_data.humidity.rh)
        self.assertIsNotNone(physics_data.humidity.qsat)

        # Edge case: Zero Specific Humidity
        qg = jnp.array([[[0] * 96] * 48])
        state = PhysicsState(jnp.zeros_like(temp), jnp.zeros_like(temp), temp, qg)
        _, physics_data = humidity.spec_hum_to_rel_hum(physics_data=physics_data, state=state)
        self.assertTrue((physics_data.humidity.rh == 0).all(), "Relative humidity should be 0 when specific humidity is 0")

        # Edge case: High Specific Humidity (near saturation)
        qg = jnp.array([[[physics_data.humidity.qsat[0, 0, 0] - 1e-6] * 96] * 48])
        state = PhysicsState(jnp.zeros_like(temp), jnp.zeros_like(temp), temp, qg)
        _, physics_data = humidity.spec_hum_to_rel_hum(physics_data=physics_data, state=state)
        self.assertTrue((physics_data.humidity.rh >= 0.99).all() and (physics_data.humidity.rh <= 1).all(), "Relative humidity should be close to 1 when specific humidity is near qsat")

        # Edge case: Very High Temperature
        temp = jnp.array([[[400] * 96] * 48])
        state = PhysicsState(jnp.zeros_like(temp), jnp.zeros_like(temp), temp, qg)
        _, physics_data = humidity.spec_hum_to_rel_hum(physics_data=physics_data, state=state)
        self.assertTrue(((physics_data.humidity.rh >= 0) & (physics_data.humidity.rh <= 1)).all(), "Relative humidity should be between 0 and 1 at very high temperatures")

        # Edge case: Extremely High Pressure
        pressure = jnp.array([[[10] * 96] * 48])
        convection_data = convection_data.copy(psa=pressure)
        physics_data = physics_data.copy(convection=convection_data)
        _, physics_data = humidity.spec_hum_to_rel_hum(physics_data=physics_data, state=state)
        self.assertTrue(((physics_data.humidity.rh >= 0) & (physics_data.humidity.rh <= 1)).all(), "Relative humidity should be between 0 and 1 at very high pressures")

    def test_rel_hum_to_spec_hum(self):
        temp = self.temp_standard
        pressure = self.pressure_standard
        sigma = self.sigma
        qg = self.qg_standard
        
        humidity_data = HumidityData((96,48), kx)
        convection_data = ConvectionData((96,48), kx,psa=pressure)
        physics_data = PhysicsData((96,48), kx, convection=convection_data, humidity=humidity_data)
        state = PhysicsState(jnp.zeros_like(temp), jnp.zeros_like(temp), temp, qg, jnp.zeros_like(temp),jnp.zeros((96,48)))


        _, physics_data = humidity.spec_hum_to_rel_hum(physics_data=physics_data, state=state)
        qa, _ = humidity.rel_hum_to_spec_hum(temp, pressure, sigma, physics_data.humidity.rh)

        # Allow a small tolerance for floating point comparisons
        tolerance = 1e-6
        self.assertTrue(jnp.allclose(qa, qg, atol=tolerance), "QA should be close to the original QG when converted from RH")

if __name__ == '__main__':
    unittest.main()
