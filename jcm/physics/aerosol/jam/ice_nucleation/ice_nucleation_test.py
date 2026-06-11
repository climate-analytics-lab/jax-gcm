"""Tests for heterogeneous ice nucleation (dust/BC) (#494)."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name
from jcm.physics.aerosol.jam.ice_nucleation.in_populations import in_populations
from jcm.physics.aerosol.jam.ice_nucleation.ice_term import IceNucleation
from jcm.physics.aerosol.jam.ice_nucleation.lohmann_diehl import lohmann_diehl_inp
from jcm.physics.aerosol.jam.ice_nucleation.niemand import niemand_inp
from jcm.physics.aerosol.jam.ice_nucleation.params import IceNucleationParameters
from jcm.physics_interface import PhysicsState

_SHAPE = (4, 2)


def _pops(du=2.0e-10, bc=1.0e-11, frac_du_soluble=0.9):
    tracers = {}
    for m in MAM4_SPEC.modes:
        if "du" in m.species:
            tracers[mass_name("du", m.short)] = jnp.full(_SHAPE, du)
        if "bc" in m.species:
            tracers[mass_name("bc", m.short)] = jnp.full(_SHAPE, bc)
    rho = jnp.full(_SHAPE, 1.0)
    return in_populations(
        MAM4_SPEC, tracers, rho, jnp.asarray(frac_du_soluble)
    )


class InPopulationsTest(unittest.TestCase):
    def test_positive_and_scales_with_mass(self):
        lo = _pops(du=1.0e-11)
        hi = _pops(du=1.0e-9)
        self.assertGreater(float(hi["du_number_sol"][0, 0]),
                           float(lo["du_number_sol"][0, 0]))
        for v in lo.values():
            self.assertTrue(np.all(np.asarray(v) >= 0.0))

    def test_solubility_split(self):
        p = _pops(frac_du_soluble=0.75)
        np.testing.assert_allclose(
            float(p["du_number_sol"][0, 0])
            / (float(p["du_number_sol"][0, 0]) + float(p["du_number_insol"][0, 0])),
            0.75, rtol=1e-5,
        )


class NiemandTest(unittest.TestCase):
    def test_immersion_rises_as_temperature_drops(self):
        params = IceNucleationParameters.default()
        pops = _pops()
        warm = niemand_inp(pops, jnp.full(_SHAPE, 268.0), jnp.full(_SHAPE, 1.0), params)
        cold = niemand_inp(pops, jnp.full(_SHAPE, 250.0), jnp.full(_SHAPE, 1.0), params)
        self.assertGreater(float(cold[0, 0]), float(warm[0, 0]))

    def test_capped_by_available_number(self):
        pops = _pops(du=1.0e-9)
        inp = niemand_inp(pops, jnp.full(_SHAPE, 240.0), jnp.full(_SHAPE, 1.0),
                          IceNucleationParameters.default())
        total = pops["du_number_sol"] + pops["bc_number_sol"] + \
            pops["du_number_insol"] + pops["bc_number_insol"]
        self.assertTrue(np.all(np.asarray(inp) <= np.asarray(total) + 1e-6))

    def test_deposition_rises_with_ice_supersaturation(self):
        params = IceNucleationParameters.default()
        pops = _pops()
        sub = niemand_inp(pops, jnp.full(_SHAPE, 245.0), jnp.full(_SHAPE, 1.0), params)
        sup = niemand_inp(pops, jnp.full(_SHAPE, 245.0), jnp.full(_SHAPE, 1.3), params)
        self.assertGreater(float(sup[0, 0]), float(sub[0, 0]))


class LohmannDiehlTest(unittest.TestCase):
    def _inp(self, t=255.0, cooling=1.0e-3, du=2.0e-10, bc=1.0e-11, s_ice=1.0):
        return lohmann_diehl_inp(
            _pops(du=du, bc=bc), jnp.full(_SHAPE, t), jnp.full(_SHAPE, s_ice),
            jnp.full(_SHAPE, cooling), jnp.asarray(1800.0),
            IceNucleationParameters.default(),
        )

    def test_dust_dominates_bc(self):
        dusty = float(self._inp(du=2.0e-10, bc=0.0)[0, 0])
        sooty = float(self._inp(du=0.0, bc=2.0e-10)[0, 0])
        self.assertGreater(dusty, sooty)

    def test_more_ascent_more_freezing(self):
        self.assertGreater(
            float(self._inp(cooling=5.0e-3)[0, 0]),
            float(self._inp(cooling=1.0e-4)[0, 0]),
        )

    def test_finite(self):
        self.assertTrue(np.all(np.isfinite(np.asarray(self._inp()))))


class IceNucleationTermTest(unittest.TestCase):
    def _setup(self, scheme="niemand"):
        tracers = {}
        for m in MAM4_SPEC.modes:
            if "du" in m.species:
                tracers[mass_name("du", m.short)] = jnp.full(_SHAPE, 2.0e-10)
            if "bc" in m.species:
                tracers[mass_name("bc", m.short)] = jnp.full(_SHAPE, 1.0e-11)
        state = PhysicsState.zeros(_SHAPE).copy(
            temperature=jnp.full(_SHAPE, 250.0),
            specific_humidity=jnp.full(_SHAPE, 2.0e-4),
            tracers=tracers,
        )
        diagnostics = {
            "pressure_full": jnp.full(_SHAPE, 4.0e4),
            "air_density": jnp.full(_SHAPE, 0.6),
            "_dt_seconds": 1800.0,
        }
        return state, diagnostics

    def test_invalid_scheme_raises(self):
        with self.assertRaises(ValueError):
            IceNucleation(scheme="bogus")

    def test_both_schemes_produce_finite_ice_nuclei(self):
        for scheme in ("niemand", "lohmann_diehl"):
            state, diagnostics = self._setup()
            _, diags = IceNucleation(scheme=scheme)(state, diagnostics, None, None)
            inp = diags["ice_nuclei"]
            self.assertEqual(inp.shape, _SHAPE)
            self.assertTrue(np.all(np.isfinite(np.asarray(inp))))
            self.assertGreater(float(jnp.max(inp)), 0.0)

    def test_grad_through_params_finite(self):
        state, diagnostics = self._setup()

        def loss(scale):
            base = IceNucleationParameters.default()
            p = IceNucleationParameters(
                frac_du_soluble=base.frac_du_soluble,
                bc_efficiency=base.bc_efficiency,
                deposition_scale=base.deposition_scale,
                scale=scale,
            )
            _, diags = IceNucleation(params=p)(state, diagnostics, None, None)
            return jnp.sum(diags["ice_nuclei"] ** 2)

        g = jax.grad(loss)(jnp.asarray(1.0))
        self.assertTrue(np.isfinite(float(g)) and float(g) != 0.0)


if __name__ == "__main__":
    unittest.main()
