import unittest
import jax.numpy as jnp

class TestHumidityUnit(unittest.TestCase):

    def setUp(self):
        global ix, il, kx
        ix, il, kx = 96, 48, 8
        from jcm.model import initialize_modules
        initialize_modules(kx=kx, il=il)

        global ConvectionData, PhysicsData, PhysicsState, get_qsat, spec_hum_to_rel_hum, rel_hum_to_spec_hum, fsg
        from jcm.physics_data import ConvectionData, PhysicsData
        from jcm.physics import PhysicsState
        from jcm.humidity import get_qsat, spec_hum_to_rel_hum, rel_hum_to_spec_hum
        from jcm.geometry import fsg
        
        self.temp_standard = jnp.ones((ix,il,kx))*273
        self.pressure_standard = jnp.ones((ix,il))*0.5
        self.sigma = 4
        self.qg_standard = jnp.ones((ix,il,kx))*2

    def test_get_qsat(self):
        temp = self.temp_standard
        pressure = self.pressure_standard
        sigma = self.sigma
        qsat = get_qsat(temp[:,:,sigma], pressure, sigma)

        self.assertIsNotNone(qsat)
        self.assertTrue((qsat >= 0).all(), "Found negative qsat values")

        # Edge case: Very low temperature
        temp = jnp.ones((ix,il))*100
        qsat = get_qsat(temp, pressure, sigma)
        self.assertIsNotNone(qsat)
        self.assertTrue((qsat >= 0).all(), "Found negative qsat values at low temperature")

        # Edge case: Very high temperature
        temp = jnp.ones((ix,il))*350
        qsat = get_qsat(temp, pressure, sigma)
        self.assertIsNotNone(qsat)
        self.assertTrue((qsat >= 0).all(), "Found negative qsat values at high temperature")

    def test_spec_hum_to_rel_hum(self):
        temp = self.temp_standard
        pressure = self.pressure_standard
        qg = self.qg_standard

        convection_data = ConvectionData((ix,il), kx, psa=pressure)
        physics_data = PhysicsData((ix,il), kx, convection=convection_data)
        state = PhysicsState(jnp.zeros_like(temp), jnp.zeros_like(temp), temp, qg, jnp.zeros_like(temp), jnp.zeros((ix,il)))

        # Edge case: Zero Specific Humidity
        qg = jnp.ones((ix,il,kx))*0
        state = PhysicsState(jnp.zeros_like(temp), jnp.zeros_like(temp), temp, qg, jnp.zeros_like(temp), jnp.zeros((ix,il)))
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state)
        self.assertTrue((physics_data.humidity.rh == 0).all(), "Relative humidity should be 0 when specific humidity is 0")

        # Edge case: Very High Temperature
        temp = jnp.ones((ix,il,kx))*400
        state = PhysicsState(jnp.zeros_like(temp), jnp.zeros_like(temp), temp, qg, jnp.zeros_like(temp), jnp.zeros((ix,il)))
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state)
        self.assertTrue(((physics_data.humidity.rh >= 0) & (physics_data.humidity.rh <= 1)).all(), "Relative humidity should be between 0 and 1 at very high temperatures")

        # Edge case: Extremely High Pressure
        pressure = jnp.ones((ix,il))*10
        convection_data = convection_data.copy(psa=pressure)
        physics_data = physics_data.copy(convection=convection_data)
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state)
        self.assertTrue(((physics_data.humidity.rh >= 0) & (physics_data.humidity.rh <= 1)).all(), "Relative humidity should be between 0 and 1 at very high pressures")

        # Edge case: High Specific Humidity (near saturation)
        qg = jnp.ones((ix,il,kx))*(physics_data.humidity.qsat[0, 0, :] - 1e-6)
        state = PhysicsState(jnp.zeros_like(temp), jnp.zeros_like(temp), temp, qg, jnp.zeros_like(temp), jnp.zeros((ix,il)))
        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state)
        self.assertTrue((physics_data.humidity.rh >= 0.99).all() and (physics_data.humidity.rh <= 1).all(), "Relative humidity should be close to 1 when specific humidity is near qsat")

    def test_rel_hum_to_spec_hum(self):
        temp = self.temp_standard
        pressure = self.pressure_standard
        qg = self.qg_standard

        convection_data = ConvectionData((ix,il), kx, psa=pressure)
        physics_data = PhysicsData((ix,il), kx, convection=convection_data)
        state = PhysicsState(jnp.zeros_like(temp), jnp.zeros_like(temp), temp, qg, jnp.zeros_like(temp),pressure)

        _, physics_data = spec_hum_to_rel_hum(physics_data=physics_data, state=state)
        qa, qsat = rel_hum_to_spec_hum(temp[:,:,0], pressure, fsg[0], physics_data.humidity.rh[:,:,0])
        # Allow a small tolerance for floating point comparisons
        tolerance = 1e-6
        self.assertTrue(jnp.allclose(qa, qg[:,:,0], atol=tolerance), "QA should be close to the original QG when converted from RH")

if __name__ == '__main__':
    unittest.main()