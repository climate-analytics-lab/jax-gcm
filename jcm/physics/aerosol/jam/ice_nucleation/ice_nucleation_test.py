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

    def test_cloud_borne_dust_counts_as_soluble(self):
        # Cloud-borne (mc_) dust is in droplets → immersion-active (soluble),
        # and does not add to the insoluble pool.
        base = {}
        cb = {}
        for m in MAM4_SPEC.modes:
            if "du" in m.species:
                base[mass_name("du", m.short)] = jnp.full(_SHAPE, 1.0e-10)
                cb[mass_name("du", m.short)] = jnp.full(_SHAPE, 1.0e-10)
                cb[mass_name("du", m.short, cloud_borne=True)] = jnp.full(_SHAPE, 5.0e-11)
        rho = jnp.full(_SHAPE, 1.0)
        p0 = in_populations(MAM4_SPEC, base, rho, jnp.asarray(0.9))
        p1 = in_populations(MAM4_SPEC, cb, rho, jnp.asarray(0.9))
        self.assertGreater(
            float(p1["du_number_sol"][0, 0]), float(p0["du_number_sol"][0, 0])
        )
        np.testing.assert_allclose(
            float(p1["du_number_insol"][0, 0]),
            float(p0["du_number_insol"][0, 0]), rtol=1e-5,
        )


class NiemandTest(unittest.TestCase):
    def test_immersion_rises_as_temperature_drops(self):
        params = IceNucleationParameters.default()
        pops = _pops()
        warm, _ = niemand_inp(pops, jnp.full(_SHAPE, 268.0), jnp.full(_SHAPE, 1.0), params)
        cold, _ = niemand_inp(pops, jnp.full(_SHAPE, 250.0), jnp.full(_SHAPE, 1.0), params)
        self.assertGreater(float(cold[0, 0]), float(warm[0, 0]))

    def test_capped_by_available_number(self):
        pops = _pops(du=1.0e-9)
        imm, dep = niemand_inp(pops, jnp.full(_SHAPE, 240.0), jnp.full(_SHAPE, 1.2),
                               IceNucleationParameters.default())
        total = pops["du_number_sol"] + pops["bc_number_sol"] + \
            pops["du_number_insol"] + pops["bc_number_insol"]
        self.assertTrue(np.all(np.asarray(imm + dep) <= np.asarray(total) + 1e-6))

    def test_deposition_rises_with_ice_supersaturation(self):
        params = IceNucleationParameters.default()
        pops = _pops()
        _, sub = niemand_inp(pops, jnp.full(_SHAPE, 245.0), jnp.full(_SHAPE, 1.0), params)
        _, sup = niemand_inp(pops, jnp.full(_SHAPE, 245.0), jnp.full(_SHAPE, 1.3), params)
        self.assertGreater(float(sup[0, 0]), float(sub[0, 0]))


class LohmannDiehlTest(unittest.TestCase):
    def _imm(self, t=255.0, cooling=1.0e-3, du=2.0e-10, bc=1.0e-11, s_ice=1.0):
        imm, _ = lohmann_diehl_inp(
            _pops(du=du, bc=bc), jnp.full(_SHAPE, t), jnp.full(_SHAPE, s_ice),
            jnp.full(_SHAPE, cooling), jnp.asarray(1800.0),
            IceNucleationParameters.default(),
        )
        return imm

    def test_dust_dominates_bc(self):
        dusty = float(self._imm(du=2.0e-10, bc=0.0)[0, 0])
        sooty = float(self._imm(du=0.0, bc=2.0e-10)[0, 0])
        self.assertGreater(dusty, sooty)

    def test_more_ascent_more_freezing(self):
        self.assertGreater(
            float(self._imm(cooling=5.0e-3)[0, 0]),
            float(self._imm(cooling=1.0e-4)[0, 0]),
        )

    def test_finite(self):
        imm, dep = lohmann_diehl_inp(
            _pops(), jnp.full(_SHAPE, 255.0), jnp.full(_SHAPE, 1.2),
            jnp.full(_SHAPE, 1.0e-3), jnp.asarray(1800.0),
            IceNucleationParameters.default(),
        )
        self.assertTrue(np.all(np.isfinite(np.asarray(imm + dep))))


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
            for key in ("ice_nuclei", "ice_nuclei_deposition"):
                inp = diags[key]
                self.assertEqual(inp.shape, _SHAPE)
                self.assertTrue(np.all(np.isfinite(np.asarray(inp))))
            self.assertGreater(float(jnp.max(diags["ice_nuclei"])), 0.0)

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


class FactoryWiringTest(unittest.TestCase):
    def test_ice_term_present_with_default_scheme(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        term = next(
            t for t in jam_aerosol_physics()
            if t.category == "aerosol_ice_nucleation"
        )
        self.assertEqual(term.name, "jam_ice_nucleation")
        self.assertEqual(term._scheme, "niemand")

    def test_scheme_threads_through(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        term = next(
            t for t in jam_aerosol_physics(ice_scheme="lohmann_diehl")
            if t.category == "aerosol_ice_nucleation"
        )
        self.assertEqual(term._scheme, "lohmann_diehl")


class IceNucleationModelTest(unittest.TestCase):
    """End-to-end WIRING guards: each het-ice scheme path compiles and runs
    inside the full ECHAM+JAM+2M model, and the ``ice_nuclei`` coupling
    diagnostic the 2M scheme consumes is emitted.

    These deliberately do NOT assert nonzero ice nuclei or scheme-dependent
    output: on a 3-step cold-start T21 aquaplanet the prognostic dust/BC
    burdens are still ~0, so both schemes measurably produce ``ice_nuclei``
    ≡ 0 and bitwise-identical ``qni`` (verified) — the scheme *physics*
    (active fractions, temperature windows, scheme differences) is pinned
    by the unit tests above on synthetic aerosol inputs. What these guard
    is the trace/compile/coupling path per scheme flag.
    """

    def _run(self, scheme):
        import numpy as onp

        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        coords = get_coords(onp.linspace(0, 1, 21), spectral_truncation=21)
        terrain = TerrainData.aquaplanet(coords)
        model = Model(
            coords=coords, time_step=30, terrain=terrain,
            physics=echam_physics(
                aerosol_module="jam", cloud_scheme="2m", jam_ice_scheme=scheme,
            ),
        )
        return model.run(save_interval=0.0625, total_time=0.0625)

    def _check(self, scheme):
        preds = self._run(scheme)
        dyn = preds.dynamics
        self.assertFalse(bool(jnp.any(jnp.isnan(dyn.temperature))))
        for key in ("qi", "qni"):
            self.assertFalse(bool(jnp.any(jnp.isnan(dyn.tracers[key]))))
        # The coupling contract with the 2M scheme: the term emits the
        # ``ice_nuclei`` diagnostic (finite, non-negative; zero is expected
        # at cold start — see class docstring).
        self.assertIn("ice_nuclei", preds.physics)
        inp = np.asarray(preds.physics["ice_nuclei"])
        self.assertTrue(np.all(np.isfinite(inp)))
        self.assertTrue(np.all(inp >= 0.0))

    def test_niemand_runs_finite(self):
        self._check("niemand")

    def test_lohmann_diehl_runs_finite(self):
        self._check("lohmann_diehl")


# Mark the model-level tests slow.
import pytest  # noqa: E402

IceNucleationModelTest = pytest.mark.slow(IceNucleationModelTest)


if __name__ == "__main__":
    unittest.main()
