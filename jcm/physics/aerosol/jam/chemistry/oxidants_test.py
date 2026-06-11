"""Phase 1 tests: interim prescribed oxidant fields (#496)."""

import types
import unittest

import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.chemistry.oxidants import (
    OxidantParameters,
    PrescribedOxidants,
    air_number_density,
    oxidant_field,
)
from jcm.physics_interface import PhysicsState


class AirNumberDensityTest(unittest.TestCase):
    def test_surface_value(self):
        n = air_number_density(jnp.asarray(288.0), jnp.asarray(1.0e5))
        # ~2.5e19 molec/cm^3 at the surface.
        self.assertAlmostEqual(float(n) / 1.0e19, 2.5, delta=0.2)

    def test_decreases_with_pressure(self):
        n_sfc = air_number_density(jnp.asarray(250.0), jnp.asarray(1.0e5))
        n_top = air_number_density(jnp.asarray(250.0), jnp.asarray(1.0e4))
        self.assertGreater(float(n_sfc), float(n_top))


class OxidantFieldTest(unittest.TestCase):
    def _inputs(self, cosz):
        shape = (3, 2)
        t = jnp.full(shape, 280.0)
        p = jnp.full(shape, 9.0e4)
        o3 = jnp.full(shape, 0.04)  # ppmv-style (40 ppbv)
        return t, p, o3, jnp.full(shape, cosz)

    def test_fields_finite_and_nonnegative(self):
        t, p, o3, cz = self._inputs(0.5)
        f = oxidant_field(t, p, o3, cz, OxidantParameters.default())
        for arr in (f.oh, f.no3, f.o3, f.h2o2):
            a = np.asarray(arr)
            self.assertTrue(np.all(np.isfinite(a)))
            self.assertTrue(np.all(a >= 0.0))

    def test_oh_is_daytime(self):
        params = OxidantParameters.default()
        day = oxidant_field(*self._inputs(1.0), params)
        night = oxidant_field(*self._inputs(0.0), params)
        self.assertGreater(float(day.oh.mean()), float(night.oh.mean()))
        self.assertAlmostEqual(float(night.oh.mean()), 0.0)

    def test_no3_is_nighttime(self):
        params = OxidantParameters.default()
        day = oxidant_field(*self._inputs(1.0), params)
        night = oxidant_field(*self._inputs(0.0), params)
        self.assertGreater(float(night.no3.mean()), float(day.no3.mean()))

    def test_oh_magnitude_reasonable(self):
        # Daytime OH should be O(1e6) molec/cm^3, not orders off.
        f = oxidant_field(*self._inputs(1.0), OxidantParameters.default())
        self.assertTrue(1.0e5 < float(f.oh.mean()) < 1.0e7)


class PrescribedOxidantsTermTest(unittest.TestCase):
    def _state(self, nlev=3, ncols=2):
        state = PhysicsState.zeros((nlev, ncols)).copy(
            temperature=jnp.full((nlev, ncols), 280.0),
        )
        return state

    def test_with_carried_chemistry_and_radiation(self):
        state = self._state()
        diagnostics = {
            "pressure_full": jnp.full((3, 2), 9.0e4),
            "chemistry": types.SimpleNamespace(
                ozone_vmr=jnp.full((3, 2), 0.05)
            ),
            "radiation": types.SimpleNamespace(
                cos_zenith=jnp.full((2,), 0.8)
            ),
        }
        _, diags = PrescribedOxidants()(state, diagnostics, None, None)
        ox = diags["oxidants"]
        self.assertEqual(ox.oh.shape, (3, 2))
        for arr in (ox.oh, ox.no3, ox.o3, ox.h2o2):
            self.assertTrue(np.all(np.isfinite(np.asarray(arr))))
        self.assertGreater(float(ox.oh.mean()), 0.0)

    def test_fallback_when_carry_absent(self):
        state = self._state()
        diagnostics = {"pressure_full": jnp.full((3, 2), 9.0e4)}
        _, diags = PrescribedOxidants()(state, diagnostics, None, None)
        ox = diags["oxidants"]
        for arr in (ox.oh, ox.no3, ox.o3, ox.h2o2):
            a = np.asarray(arr)
            self.assertTrue(np.all(np.isfinite(a)))
            self.assertTrue(np.all(a >= 0.0))
        # Fallback O3 is finite and positive.
        self.assertGreater(float(ox.o3.mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
