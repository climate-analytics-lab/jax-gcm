import unittest
from jcm import humidity
import jax.numpy as jnp

class TestHumidityUnit(unittest.TestCase):

    def test_get_qsat(self):
        # Standard case
        temp = jnp.array([[[273] * 96] * 48])
        pressure = jnp.array([[[0.5] * 96] * 48])
        sigma = 4
        qsat = humidity.get_qsat(temp, pressure, sigma)
        print("QSAT (standard case):", qsat)

        # Check that qsat is not null and all values are non-negative.
        self.assertIsNotNone(qsat)
        self.assertTrue((qsat >= 0).all(), "Found negative qsat values")

        # Edge case: Very low temperature
        temp = jnp.array([[[100] * 96] * 48])
        qsat = humidity.get_qsat(temp, pressure, sigma)
        print("QSAT (low temperature):", qsat)
        self.assertIsNotNone(qsat)
        self.assertTrue((qsat >= 0).all(), "Found negative qsat values at low temperature")

        # Edge case: Very high temperature
        temp = jnp.array([[[350] * 96] * 48])
        qsat = humidity.get_qsat(temp, pressure, sigma)
        print("QSAT (high temperature):", qsat)
        self.assertIsNotNone(qsat)
        self.assertTrue((qsat >= 0).all(), "Found negative qsat values at high temperature")

    def test_spec_hum_to_rel_hum(self):
        temp = jnp.array([[[273] * 96] * 48])
        pressure = jnp.array([[[0.5] * 96] * 48])
        sigma = 4
        qg = jnp.array([[[2] * 96] * 48])
        rh, qsat = humidity.spec_hum_to_rel_hum(temp, pressure, sigma, qg)
        print("RH (standard case):", rh)
        print("QSAT (standard case):", qsat)

        # Check that rh and qsat are not null.
        self.assertIsNotNone(rh)
        self.assertIsNotNone(qsat)

        # Edge case: Zero Specific Humidity
        qg = jnp.array([[[0] * 96] * 48])
        rh, qsat = humidity.spec_hum_to_rel_hum(temp, pressure, sigma, qg)
        print("RH (zero specific humidity):", rh)
        self.assertTrue((rh == 0).all(), "Relative humidity should be 0 when specific humidity is 0")

        # Edge case: High Specific Humidity (near saturation)
        qg = jnp.array([[[qsat[0, 0, 0] - 1e-6] * 96] * 48])  # Slightly less than qsat
        rh, _ = humidity.spec_hum_to_rel_hum(temp, pressure, sigma, qg)
        print("RH (high specific humidity):", rh)
        self.assertTrue((rh >= 0.99).all() and (rh <= 1).all(), "Relative humidity should be close to 1 when specific humidity is near qsat")

        # Edge case: Very Low Temperature
        temp = jnp.array([[[1] * 96] * 48])  # Near absolute zero
        rh, qsat = humidity.spec_hum_to_rel_hum(temp, pressure, sigma, qg)
        print("RH (low temperature):", rh)
        self.assertTrue((rh >= 0).all() and (rh <= 1).all(), "Relative humidity should be between 0 and 1 at very low temperatures")

        # Edge case: Very High Temperature
        temp = jnp.array([[[400] * 96] * 48])  # Extremely high temperature
        rh, qsat = humidity.spec_hum_to_rel_hum(temp, pressure, sigma, qg)
        print("RH (high temperature):", rh)
        self.assertTrue((rh >= 0).all() and (rh <= 1).all(), "Relative humidity should be between 0 and 1 at very high temperatures")

        # Edge case: Zero Pressure
        pressure = jnp.array([[[0] * 96] * 48])
        rh, qsat = humidity.spec_hum_to_rel_hum(temp, pressure, sigma, qg)
        print("RH (zero pressure):", rh)
        self.assertTrue((rh == 0).all(), "Relative humidity should be 0 when pressure is 0")

        # Edge case: Extremely High Pressure
        pressure = jnp.array([[[10] * 96] * 48])  # Much higher than standard atmospheric pressure
        rh, qsat = humidity.spec_hum_to_rel_hum(temp, pressure, sigma, qg)
        print("RH (high pressure):", rh)
        self.assertTrue((rh >= 0).all() and (rh <= 1).all(), "Relative humidity should be between 0 and 1 at very high pressures")
    
    def test_rel_hum_to_spec_hum(self):
        temp = jnp.array([[[273] * 96] * 48])
        pressure = jnp.array([[[0.5] * 96] * 48])
        sigma = 4
        qg = jnp.array([[[2] * 96] * 48])
        rh, _ = humidity.spec_hum_to_rel_hum(temp, pressure, sigma, qg)
        qa, _ = humidity.rel_hum_to_spec_hum(temp, pressure, sigma, rh)
        print("QA (converted from RH):", qa)
        print("Original QG:", qg)

        # Check that qa is the same when converted to rh then back again.
        self.assertEqual(float(jnp.take(qa, 0)), float(jnp.take(qg, 0)))
        
