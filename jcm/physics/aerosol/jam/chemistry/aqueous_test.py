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
from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY
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
        carry = {}
        for m in (mm.short for mm in MAM4_SPEC.modes if "so4" in mm.species):
            tracers[mass_name("so4", m)] = jnp.full(shape, 1.0e-11)
            carry[mass_name("so4", m, cloud_borne=True)] = jnp.full(
                shape, 1.0e-12
            )
            carry[number_name(m, cloud_borne=True)] = jnp.full(shape, nc)
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, 275.0), tracers=tracers,
        )
        ox = OxidantField(
            oh=jnp.full(shape, 1.0e6), no3=jnp.zeros(shape),
            o3=jnp.full(shape, 1.0e12), h2o2=jnp.full(shape, 1.0e10),
        )
        diagnostics = {
            CARRY_KEY: carry,
            "oxidants": ox,
            "clouds": types.SimpleNamespace(
                cloud_fraction=jnp.full(shape, cloud_fraction),
                qc=jnp.full(shape, 2.0e-4),
            ),
            "air_density": jnp.full(shape, 1.0),
            "_dt_seconds": 1800.0,
        }
        return state, diagnostics

    @staticmethod
    def _cb_rate(diag_in, diag_out, nm, dt=1800.0):
        """Effective cloud-borne production rate from the carry update."""
        import numpy as _np
        return (
            _np.asarray(diag_out[CARRY_KEY][nm])
            - _np.asarray(diag_in[CARRY_KEY][nm])
        ) / dt

    def test_produces_cloud_borne_sulfate_and_consumes_so2(self):
        state, diagnostics = self._setup()
        tend, out = AqueousSulfur()(state, diagnostics, None, None)
        cb = self._cb_rate(
            diagnostics, out, mass_name("so4", "acc", cloud_borne=True),
        )
        self.assertGreater(float(cb[0, 0]), 0.0)
        self.assertLess(float(tend.tracers["g_so2"][0, 0]), 0.0)

    def test_crystal_held_reservoirs_host_no_aqueous_sulfate(self):
        # Sulfate forms in droplets: the split over modes uses the
        # droplet-held part of each mode's cloud-borne number, as the
        # exchange records it.
        from jcm.physics.aerosol.jam import MAM4_SPEC
        from jcm.physics.aerosol.jam.cloud_borne_store import LIQUID_SHARE_KEY

        state, diagnostics = self._setup()
        shape = state.temperature.shape
        modes = [mm.short for mm in MAM4_SPEC.modes if "so4" in mm.species]
        share = {number_name(m, cloud_borne=True): jnp.full(
            shape, 0.0 if m == "cor" else 1.0) for m in modes}
        diag = {**diagnostics, LIQUID_SHARE_KEY: share}
        _, out = AqueousSulfur()(state, diag, None, None)
        cor = self._cb_rate(diag, out, mass_name("so4", "cor", cloud_borne=True))
        acc = self._cb_rate(diag, out, mass_name("so4", "acc", cloud_borne=True))
        np.testing.assert_array_equal(cor, 0.0)
        self.assertGreater(float(acc[0, 0]), 0.0)

    def test_fresh_sulfate_is_droplet_held_in_the_share(self):
        # Wet deposition removes the recorded droplet-held share at the
        # liquid conversion: sulfate formed here in droplets must raise each
        # reservoir's share to (s·m + Δm)/(m + Δm), number shares unchanged.
        from jcm.physics.aerosol.jam import MAM4_SPEC
        from jcm.physics.aerosol.jam.cloud_borne_store import LIQUID_SHARE_KEY

        state, diagnostics = self._setup()
        shape = state.temperature.shape
        modes = [mm.short for mm in MAM4_SPEC.modes if "so4" in mm.species]
        share = {}
        for m in modes:
            share[number_name(m, cloud_borne=True)] = jnp.full(shape, 0.7)
            share[mass_name("so4", m, cloud_borne=True)] = jnp.full(shape, 0.4)
        diag = {**diagnostics, LIQUID_SHARE_KEY: share}
        _, out = AqueousSulfur()(state, diag, None, None)
        for m in modes:
            nm = mass_name("so4", m, cloud_borne=True)
            old = np.asarray(diag[CARRY_KEY][nm])
            fresh = self._cb_rate(diag, out, nm) * 1800.0
            self.assertGreater(float(fresh[0, 0]), 0.0)
            np.testing.assert_allclose(
                np.asarray(out[LIQUID_SHARE_KEY][nm]),
                (0.4 * old + fresh) / (old + fresh), rtol=1e-5)
            nn = number_name(m, cloud_borne=True)
            np.testing.assert_array_equal(
                np.asarray(out[LIQUID_SHARE_KEY][nn]), np.asarray(share[nn]))

    def test_share_left_alone_without_production_or_record(self):
        from jcm.physics.aerosol.jam.cloud_borne_store import LIQUID_SHARE_KEY

        # No cloud: nothing forms, so the recorded share is kept.
        state, diagnostics = self._setup(cloud_fraction=0.0)
        nm = mass_name("so4", "acc", cloud_borne=True)
        share = {nm: jnp.full(state.temperature.shape, 0.4)}
        _, out = AqueousSulfur()(
            state, {**diagnostics, LIQUID_SHARE_KEY: share}, None, None)
        np.testing.assert_allclose(
            np.asarray(out[LIQUID_SHARE_KEY][nm]), np.asarray(share[nm]),
            rtol=1e-6)
        # No exchange record at all: none is invented.
        state, diagnostics = self._setup()
        _, out = AqueousSulfur()(state, diagnostics, None, None)
        self.assertNotIn(LIQUID_SHARE_KEY, out)

    def test_share_gradient_finite_on_empty_reservoirs(self):
        # An empty reservoir with no production takes the discarded branch
        # of the share update; its derivative must stay finite.
        from jcm.physics.aerosol.jam.cloud_borne_store import LIQUID_SHARE_KEY

        for cloud_fraction in (0.0, 0.6):
            state, diagnostics = self._setup(cloud_fraction=cloud_fraction,
                                             nc=0.0)
            nm = mass_name("so4", "acc", cloud_borne=True)
            shape = state.temperature.shape

            def loss(cb, s):
                carry = {**diagnostics[CARRY_KEY], nm: cb}
                diag = {**diagnostics, CARRY_KEY: carry,
                        LIQUID_SHARE_KEY: {nm: s}}
                _, out = AqueousSulfur()(state, diag, None, None)
                return jnp.sum(out[LIQUID_SHARE_KEY][nm])

            grads = jax.grad(loss, argnums=(0, 1))(
                jnp.zeros(shape), jnp.full(shape, 0.4))
            for g in grads:
                self.assertTrue(np.all(np.isfinite(np.asarray(g))),
                                msg=f"cf={cloud_fraction}")

    def test_sulfur_conserved(self):
        from jcm.physics.aerosol.jam import MAM4_SPEC
        from jcm.physics.aerosol.jam.chemistry.aqueous import _MW_SO2, _MW_SO4

        state, diagnostics = self._setup()
        tend, out = AqueousSulfur()(state, diagnostics, None, None)
        s_rate = np.asarray(tend.tracers["g_so2"]) / _MW_SO2
        for m in (mm.short for mm in MAM4_SPEC.modes if "so4" in mm.species):
            key = mass_name("so4", m, cloud_borne=True)
            s_rate = s_rate + self._cb_rate(diagnostics, out, key) / _MW_SO4
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
        tend, out = AqueousSulfur()(state, diagnostics, None, None)
        # All produced sulfate lands in interstitial accumulation mode…
        self.assertGreater(
            float(tend.tracers[mass_name("so4", "acc")][0, 0]), 0.0,
        )
        # …and none in any cloud-borne field.
        s_rate = np.asarray(tend.tracers["g_so2"]) / _MW_SO2
        s_rate = s_rate + np.asarray(tend.tracers[mass_name("so4", "acc")]) / _MW_SO4
        for m in (mm.short for mm in MAM4_SPEC.modes if "so4" in mm.species):
            key = mass_name("so4", m, cloud_borne=True)
            cb = self._cb_rate(diagnostics, out, key)
            np.testing.assert_allclose(cb, 0.0)
            s_rate = s_rate + cb / _MW_SO4
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
        _, out = AqueousSulfur()(state, diagnostics, None, None)
        cb = self._cb_rate(
            diagnostics, out, mass_name("so4", "acc", cloud_borne=True),
        )
        self.assertAlmostEqual(float(cb[0, 0]), 0.0)

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

    def test_cloud_water_gradient_is_finite_in_clear_sky(self):
        """No cloud water, or no cloud at all, still differentiates (#663).

        The chemistry is evaluated everywhere and kept only where active, so
        a clear-sky cell must not hand the discarded evaluation a zero LWC.
        """
        for cloud_fraction, qc in ((0.6, 0.0), (0.0, 0.0), (0.6, 2.0e-4)):
            state, diagnostics = self._setup(cloud_fraction=cloud_fraction)
            shape = state.temperature.shape

            def loss(qc_field, cf_field):
                clouds = types.SimpleNamespace(
                    cloud_fraction=cf_field, qc=qc_field)
                tend, _ = AqueousSulfur()(
                    state, {**diagnostics, "clouds": clouds}, None, None)
                return sum(jnp.sum(v) for v in tend.tracers.values())

            grads = jax.grad(loss, argnums=(0, 1))(
                jnp.full(shape, qc), jnp.full(shape, cloud_fraction))
            for g in grads:
                self.assertTrue(np.all(np.isfinite(np.asarray(g))),
                                msg=f"cf={cloud_fraction}, qc={qc}")


class SimpleAqueousSchemeTest(unittest.TestCase):
    def _setup(self, h2o2=1.0e10, so2=1.0e-10, nlev=3, ncols=2):
        from jcm.physics.aerosol.jam import MAM4_SPEC

        shape = (nlev, ncols)
        tracers = {"g_so2": jnp.full(shape, so2)}
        carry = {}
        for m in (mm.short for mm in MAM4_SPEC.modes if "so4" in mm.species):
            tracers[mass_name("so4", m)] = jnp.full(shape, 1.0e-11)
            carry[mass_name("so4", m, cloud_borne=True)] = jnp.full(
                shape, 1.0e-12
            )
            carry[number_name(m, cloud_borne=True)] = jnp.full(shape, 1.0e7)
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, 275.0), tracers=tracers,
        )
        ox = OxidantField(
            oh=jnp.zeros(shape), no3=jnp.zeros(shape),
            o3=jnp.full(shape, 1.0e12), h2o2=jnp.full(shape, h2o2),
        )
        diagnostics = {
            CARRY_KEY: carry,
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
        tend, out = AqueousSulfur(scheme="simple")(
            state, diagnostics, None, None,
        )
        cb_acc = AqueousTermTest._cb_rate(
            diagnostics, out, mass_name("so4", "acc", cloud_borne=True),
        )
        self.assertGreater(float(cb_acc[0, 0]), 0.0)
        s_rate = np.asarray(tend.tracers["g_so2"]) / _MW_SO2
        for m in (mm.short for mm in MAM4_SPEC.modes if "so4" in mm.species):
            s_rate = s_rate + AqueousTermTest._cb_rate(
                diagnostics, out, mass_name("so4", m, cloud_borne=True),
            ) / _MW_SO4
        self.assertTrue(np.all(np.abs(s_rate) < 1.0e-18))

    def test_simple_is_h2o2_limited(self):
        # Scarce H2O2 caps the sulfate produced below the abundant-H2O2 case.
        key = mass_name("so4", "acc", cloud_borne=True)
        s_lo, d_lo = self._setup(h2o2=1.0e7)
        _, out_lo = AqueousSulfur(scheme="simple")(s_lo, d_lo, None, None)
        s_hi, d_hi = self._setup(h2o2=1.0e11)
        _, out_hi = AqueousSulfur(scheme="simple")(s_hi, d_hi, None, None)
        self.assertLess(
            float(AqueousTermTest._cb_rate(d_lo, out_lo, key)[0, 0]),
            float(AqueousTermTest._cb_rate(d_hi, out_hi, key)[0, 0]),
        )


if __name__ == "__main__":
    unittest.main()
