"""Phase 2 tests: gas-phase sulfur chemistry (DMS/SO2 -> SO2/H2SO4) (#496)."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.chemistry.oxidants import OxidantField
from jcm.physics.aerosol.jam.chemistry.sulfur_gas import (
    SulfurGasChemistry,
    _k_dms_no3,
    _k_dms_oh,
    _k_so2_oh,
    sulfur_gas_tendencies,
)
from jcm.physics.aerosol.jam.gas_species import GAS_SPECIES
from jcm.physics.aerosol.jam.tracer_layout import gas_name
from jcm.physics_interface import PhysicsState

_M = {sp: GAS_SPECIES[sp].molar_mass for sp in ("dms", "so2", "h2so4")}


class RateConstantTest(unittest.TestCase):
    def test_rates_positive_and_finite(self):
        t = jnp.asarray(280.0)
        n_air = jnp.asarray(2.3e19)
        k1, k2 = _k_dms_oh(t, n_air)
        for k in (k1, k2, _k_so2_oh(t, n_air), _k_dms_no3(t)):
            self.assertTrue(np.isfinite(float(k)))
            self.assertGreater(float(k), 0.0)

    def test_dms_oh_abstraction_matches_ham(self):
        # k1 = 9.6e-12 * exp(-234/T)
        t = 280.0
        k1, _ = _k_dms_oh(jnp.asarray(t), jnp.asarray(2.3e19))
        self.assertAlmostEqual(float(k1), 9.6e-12 * np.exp(-234.0 / t), places=18)


class SulfurGasTendencyTest(unittest.TestCase):
    def _call(self, oh, no3, dms=1.0e-10, so2=1.0e-10):
        shape = (2,)
        return sulfur_gas_tendencies(
            dms=jnp.full(shape, dms),
            so2=jnp.full(shape, so2),
            temperature=jnp.full(shape, 280.0),
            pressure=jnp.full(shape, 9.0e4),
            oh=jnp.full(shape, oh),
            no3=jnp.full(shape, no3),
            dt=jnp.asarray(1800.0),
            soag_production=jnp.asarray(2.0e-15),
        )

    def test_finite(self):
        t = self._call(oh=2.0e6, no3=0.0)
        for v in t.values():
            self.assertTrue(np.all(np.isfinite(np.asarray(v))))

    def test_sulfur_moles_conserved(self):
        # DMS, SO2, H2SO4 each carry one S — molar S rate must sum to ~0
        # (relative to the component magnitudes; float32-appropriate tolerance).
        t = self._call(oh=2.0e6, no3=5.0e7)
        components = [
            np.asarray(t[gas_name(sp)]) / _M[sp] for sp in ("dms", "so2", "h2so4")
        ]
        s_rate = sum(components)
        scale = sum(np.abs(c) for c in components)
        self.assertTrue(
            np.all(np.abs(s_rate) <= 1.0e-5 * np.maximum(scale, 1.0e-30))
        )

    def test_daytime_makes_h2so4_and_consumes_dms(self):
        t = self._call(oh=3.0e6, no3=0.0)
        self.assertLess(float(t[gas_name("dms")][0]), 0.0)      # DMS lost
        self.assertGreater(float(t[gas_name("h2so4")][0]), 0.0)  # H2SO4 made

    def test_nighttime_no3_makes_so2_not_h2so4(self):
        t = self._call(oh=0.0, no3=5.0e7)
        self.assertLess(float(t[gas_name("dms")][0]), 0.0)        # DMS lost via NO3
        self.assertGreater(float(t[gas_name("so2")][0]), 0.0)     # -> SO2
        self.assertAlmostEqual(float(t[gas_name("h2so4")][0]), 0.0)  # no OH -> no H2SO4

    def test_soag_produced(self):
        t = self._call(oh=1.0e6, no3=0.0)
        self.assertGreater(float(t[gas_name("soag")][0]), 0.0)


class SulfurGasTermTest(unittest.TestCase):
    def _setup(self, nlev=3, ncols=2):
        shape = (nlev, ncols)
        tracers = {
            gas_name("dms"): jnp.full(shape, 1.0e-10),
            gas_name("so2"): jnp.full(shape, 1.0e-10),
            gas_name("h2so4"): jnp.zeros(shape),
            gas_name("soag"): jnp.zeros(shape),
        }
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, 280.0), tracers=tracers,
        )
        ox = OxidantField(
            oh=jnp.full(shape, 2.0e6), no3=jnp.full(shape, 1.0e7),
            o3=jnp.full(shape, 1.0e12), h2o2=jnp.full(shape, 1.0e10),
        )
        diagnostics = {
            "oxidants": ox,
            "pressure_full": jnp.full(shape, 9.0e4),
            "_dt_seconds": 1800.0,
        }
        return state, diagnostics

    def test_term_declares_gas_tracers(self):
        names = {s.name for s in SulfurGasChemistry().required_tracers()}
        self.assertEqual(names, {"g_dms", "g_so2", "g_h2so4", "g_soag"})

    def test_term_produces_finite_tendencies(self):
        state, diagnostics = self._setup()
        tend, _ = SulfurGasChemistry()(state, diagnostics, None, None)
        self.assertIn(gas_name("h2so4"), tend.tracers)
        for v in tend.tracers.values():
            self.assertTrue(np.all(np.isfinite(np.asarray(v))))

    def test_grad_through_soag_param_finite(self):
        from jcm.physics.aerosol.jam.chemistry.sulfur_gas import (
            SulfurGasParameters,
        )

        state, diagnostics = self._setup()

        def loss(prod):
            term = SulfurGasChemistry(
                params=SulfurGasParameters(soag_production=prod)
            )
            tend, _ = term(state, diagnostics, None, None)
            return sum(jnp.sum(v ** 2) for v in tend.tracers.values())

        g = jax.grad(loss)(jnp.asarray(2.0e-15))
        self.assertTrue(np.isfinite(float(g)))


if __name__ == "__main__":
    unittest.main()
