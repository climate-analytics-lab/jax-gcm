"""Phase 3 tests: in-cloud aqueous sulfur chemistry (#496)."""

import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.chemistry.aqueous import (
    AqueousSulfur,
    _aqueous_so4,
    _CONV_SO2_SO4_MASS,
)
from jcm.physics.aerosol.jam.chemistry.oxidants import OxidantField
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics_interface import PhysicsState


def _f(**kw):
    base = dict(
        so2=1.0e-10, so4=1.0e-11, h2o2=1.0e10, o3=1.0e12,
        lwc=3.0e-4, rho=1.0, temperature=275.0, dt=1800.0,
    )
    base.update(kw)
    return _aqueous_so4(**{k: jnp.asarray(v) for k, v in base.items()})


class AqueousKernelTest(unittest.TestCase):
    def test_finite_and_bounded_by_available_so2(self):
        dso4 = _f()
        self.assertTrue(np.isfinite(float(dso4)))
        self.assertGreaterEqual(float(dso4), 0.0)
        # SO2 consumed (= dso4 / conv) cannot exceed the SO2 present.
        self.assertLessEqual(float(dso4) / _CONV_SO2_SO4_MASS, 1.0e-10 + 1e-20)

    def test_more_h2o2_more_sulfate(self):
        low = float(_f(h2o2=1.0e8))
        high = float(_f(h2o2=5.0e10))
        self.assertGreater(high, low)

    def test_no_so2_no_sulfate(self):
        self.assertAlmostEqual(float(_f(so2=0.0)), 0.0)

    def test_o3_path_active_without_h2o2(self):
        # With H2O2 ~ 0 the O3 pathway still oxidises some SO2.
        self.assertGreater(float(_f(h2o2=0.0, o3=2.0e12)), 0.0)


class AqueousTermTest(unittest.TestCase):
    def _setup(self, cloud_fraction=0.6, nc=1.0e7, nlev=3, ncols=2):
        from jcm.physics.aerosol.jam import MAM4_SPEC

        shape = (nlev, ncols)
        tracers = {"g_so2": jnp.full(shape, 1.0e-10)}
        for m in (mm.short for mm in MAM4_SPEC.modes if "so4" in mm.species):
            tracers[mass_name("so4", m)] = jnp.full(shape, 1.0e-11)
            tracers[mass_name("so4", m, cloud_borne=True)] = jnp.full(shape, 1.0e-12)
            tracers[number_name(m, cloud_borne=True)] = jnp.full(shape, nc)
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, 275.0), tracers=tracers,
        )
        ox = OxidantField(
            oh=jnp.full(shape, 1.0e6), no3=jnp.zeros(shape),
            o3=jnp.full(shape, 1.0e12), h2o2=jnp.full(shape, 1.0e10),
        )
        diagnostics = {
            "oxidants": ox,
            "clouds": types.SimpleNamespace(
                cloud_fraction=jnp.full(shape, cloud_fraction),
                qc=jnp.full(shape, 2.0e-4),
            ),
            "air_density": jnp.full(shape, 1.0),
            "_dt_seconds": 1800.0,
        }
        return state, diagnostics

    def test_produces_cloud_borne_sulfate_and_consumes_so2(self):
        state, diagnostics = self._setup()
        tend, _ = AqueousSulfur()(state, diagnostics, None, None)
        self.assertIn(mass_name("so4", "acc", cloud_borne=True), tend.tracers)
        self.assertGreater(
            float(tend.tracers[mass_name("so4", "acc", cloud_borne=True)][0, 0]),
            0.0,
        )
        self.assertLess(float(tend.tracers["g_so2"][0, 0]), 0.0)

    def test_sulfur_conserved(self):
        from jcm.physics.aerosol.jam import MAM4_SPEC
        from jcm.physics.aerosol.jam.chemistry.aqueous import _MW_SO2, _MW_SO4

        state, diagnostics = self._setup()
        tend, _ = AqueousSulfur()(state, diagnostics, None, None)
        s_rate = np.asarray(tend.tracers["g_so2"]) / _MW_SO2
        for m in (mm.short for mm in MAM4_SPEC.modes if "so4" in mm.species):
            key = mass_name("so4", m, cloud_borne=True)
            s_rate = s_rate + np.asarray(tend.tracers[key]) / _MW_SO4
        self.assertTrue(np.all(np.abs(s_rate) < 1.0e-18))

    def test_sulfur_conserved_without_cloud_borne_number(self):
        # No cloud-borne number anywhere (spin-up, before the exchange term
        # has populated the mirrors). Production must land in INTERSTITIAL
        # accumulation-mode sulfate — the HAM cloud-borne-coarse fallback fed
        # a tracer nothing scavenged before #602 closed the cycle, which grew
        # ~0.7 mg/m²/day without equilibrium in the first online-emission
        # ne30 year — and must still match the SO2 sink.
        from jcm.physics.aerosol.jam import MAM4_SPEC
        from jcm.physics.aerosol.jam.chemistry.aqueous import _MW_SO2, _MW_SO4

        state, diagnostics = self._setup(nc=0.0)
        tend, _ = AqueousSulfur()(state, diagnostics, None, None)
        # All produced sulfate lands in interstitial accumulation mode…
        self.assertGreater(
            float(tend.tracers[mass_name("so4", "acc")][0, 0]), 0.0,
        )
        # …and none in any cloud-borne tracer.
        for m in (mm.short for mm in MAM4_SPEC.modes if "so4" in mm.species):
            np.testing.assert_allclose(
                np.asarray(tend.tracers[mass_name("so4", m, cloud_borne=True)]),
                0.0,
            )
        s_rate = np.asarray(tend.tracers["g_so2"]) / _MW_SO2
        s_rate = s_rate + np.asarray(tend.tracers[mass_name("so4", "acc")]) / _MW_SO4
        for m in (mm.short for mm in MAM4_SPEC.modes if "so4" in mm.species):
            key = mass_name("so4", m, cloud_borne=True)
            s_rate = s_rate + np.asarray(tend.tracers[key]) / _MW_SO4
        self.assertTrue(np.all(np.abs(s_rate) < 1.0e-18))

    def test_implicit_population_emits_no_cloud_borne_keys(self):
        # With ``spec.cloud_borne = False`` (#602) the whole production is
        # interstitial by construction: no mirror tendencies at all, and the
        # sulfur budget still closes against the SO2 sink.
        import dataclasses
        from jcm.physics.aerosol.jam import MAM4_SPEC
        from jcm.physics.aerosol.jam.chemistry.aqueous import _MW_SO2, _MW_SO4

        spec = dataclasses.replace(MAM4_SPEC, cloud_borne=False)
        state, diagnostics = self._setup()
        tend, _ = AqueousSulfur(spec=spec)(state, diagnostics, None, None)
        self.assertFalse(
            any(nm.startswith(("mc_", "nc_")) for nm in tend.tracers)
        )
        self.assertGreater(
            float(tend.tracers[mass_name("so4", "acc")][0, 0]), 0.0,
        )
        s_rate = np.asarray(tend.tracers["g_so2"]) / _MW_SO2
        s_rate = s_rate + np.asarray(
            tend.tracers[mass_name("so4", "acc")]
        ) / _MW_SO4
        self.assertTrue(np.all(np.abs(s_rate) < 1.0e-18))

    def test_no_clouds_no_production(self):
        state, diagnostics = self._setup(cloud_fraction=0.0)
        tend, _ = AqueousSulfur()(state, diagnostics, None, None)
        self.assertAlmostEqual(
            float(tend.tracers[mass_name("so4", "acc", cloud_borne=True)][0, 0]),
            0.0,
        )

    def test_grad_through_rate_scale_finite(self):
        from jcm.physics.aerosol.jam.chemistry.aqueous import (
            AqueousSulfurParameters,
        )

        state, diagnostics = self._setup()

        def loss(scale):
            term = AqueousSulfur(
                params=AqueousSulfurParameters(rate_scale=scale)
            )
            tend, _ = term(state, diagnostics, None, None)
            return sum(jnp.sum(v ** 2) for v in tend.tracers.values())

        g = jax.grad(loss)(jnp.asarray(1.0))
        self.assertTrue(np.isfinite(float(g)))


class SimpleAqueousSchemeTest(unittest.TestCase):
    def _setup(self, h2o2=1.0e10, so2=1.0e-10, nlev=3, ncols=2):
        from jcm.physics.aerosol.jam import MAM4_SPEC

        shape = (nlev, ncols)
        tracers = {"g_so2": jnp.full(shape, so2)}
        for m in (mm.short for mm in MAM4_SPEC.modes if "so4" in mm.species):
            tracers[mass_name("so4", m)] = jnp.full(shape, 1.0e-11)
            tracers[mass_name("so4", m, cloud_borne=True)] = jnp.full(shape, 1.0e-12)
            tracers[number_name(m, cloud_borne=True)] = jnp.full(shape, 1.0e7)
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, 275.0), tracers=tracers,
        )
        ox = OxidantField(
            oh=jnp.zeros(shape), no3=jnp.zeros(shape),
            o3=jnp.full(shape, 1.0e12), h2o2=jnp.full(shape, h2o2),
        )
        diagnostics = {
            "oxidants": ox,
            "clouds": types.SimpleNamespace(
                cloud_fraction=jnp.full(shape, 0.6), qc=jnp.full(shape, 2.0e-4),
            ),
            "air_density": jnp.full(shape, 1.0),
            "_dt_seconds": 1800.0,
        }
        return state, diagnostics

    def test_invalid_scheme_raises(self):
        with self.assertRaises(ValueError):
            AqueousSulfur(scheme="bogus")

    def test_simple_produces_sulfate_and_conserves_sulfur(self):
        from jcm.physics.aerosol.jam import MAM4_SPEC
        from jcm.physics.aerosol.jam.chemistry.aqueous import _MW_SO2, _MW_SO4

        state, diagnostics = self._setup()
        tend, _ = AqueousSulfur(scheme="simple")(state, diagnostics, None, None)
        self.assertGreater(
            float(tend.tracers[mass_name("so4", "acc", cloud_borne=True)][0, 0]),
            0.0,
        )
        s_rate = np.asarray(tend.tracers["g_so2"]) / _MW_SO2
        for m in (mm.short for mm in MAM4_SPEC.modes if "so4" in mm.species):
            s_rate = s_rate + np.asarray(
                tend.tracers[mass_name("so4", m, cloud_borne=True)]
            ) / _MW_SO4
        self.assertTrue(np.all(np.abs(s_rate) < 1.0e-18))

    def test_simple_is_h2o2_limited(self):
        # Scarce H2O2 caps the sulfate produced below the abundant-H2O2 case.
        lo, _ = AqueousSulfur(scheme="simple")(*self._setup(h2o2=1.0e7)[:2], None, None)
        hi, _ = AqueousSulfur(scheme="simple")(*self._setup(h2o2=1.0e11)[:2], None, None)
        key = mass_name("so4", "acc", cloud_borne=True)
        self.assertLess(
            float(lo.tracers[key][0, 0]), float(hi.tracers[key][0, 0])
        )


if __name__ == "__main__":
    unittest.main()
