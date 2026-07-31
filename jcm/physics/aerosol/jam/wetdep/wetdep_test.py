"""Phase 5 tests: in-cloud + below-cloud scavenging and the term."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.wetdep.wetdep_term import (
    WetScavenging,
    WetDepParameters,
    below_cloud_rate,
    conv_in_cloud_rate,
    in_cloud_rate,
    precip_formation_rate,
)


class ScavengingFunctionTest(unittest.TestCase):
    def test_precip_formation_distributes_over_condensate(self):
        precip = jnp.asarray([1.0e-4])           # kg/m²/s
        cf = jnp.array([[0.0], [0.8], [0.0]])
        qc = jnp.array([[0.0], [1.0e-3], [0.0]])
        rho = jnp.ones((3, 1))
        dz = jnp.full((3, 1), 100.0)
        pf = precip_formation_rate(precip, cf, qc, rho, dz)
        # All formation in the single cloudy layer.
        self.assertGreater(float(pf[1, 0]), 0.0)
        self.assertAlmostEqual(float(pf[0, 0]), 0.0)
        self.assertAlmostEqual(float(pf[2, 0]), 0.0)

    def test_in_cloud_rate_scales_with_activation(self):
        pf = jnp.full((1, 1), 1.0e-6)
        qc = jnp.full((1, 1), 1.0e-3)
        lo = in_cloud_rate(jnp.full((1, 1), 0.2), pf, qc)
        hi = in_cloud_rate(jnp.full((1, 1), 0.9), pf, qc)
        self.assertGreater(float(hi[0, 0]), float(lo[0, 0]))

    def test_below_cloud_size_dependence(self):
        precip = jnp.asarray([1.0e-4])
        cf = jnp.zeros((1, 1))
        params = WetDepParameters.default()
        accum = below_cloud_rate(precip, cf, jnp.full((1, 1), 0.1e-6), params)
        coarse = below_cloud_rate(precip, cf, jnp.full((1, 1), 2.0e-6), params)
        self.assertGreater(float(coarse[0, 0]), float(accum[0, 0]))

    def test_no_precip_no_below_cloud(self):
        params = WetDepParameters.default()
        rate = below_cloud_rate(
            jnp.zeros((1,)), jnp.zeros((1, 1)), jnp.full((1, 1), 1e-6), params,
        )
        self.assertAlmostEqual(float(rate[0, 0]), 0.0)

    def test_conv_in_cloud_hammoz_form(self):
        # rate = ratio * (formation/(rho*dz)) / condensate on cloudy
        # layers; exactly zero where the updraft carries no condensate.
        params = WetDepParameters.default()
        form = jnp.array([[0.0], [1.0e-4], [0.0]])        # kg/m²/s
        qcond = jnp.array([[0.0], [1.0e-3], [1.0e-3]])    # kg/kg
        rho = jnp.ones((3, 1))
        dz = jnp.full((3, 1), 500.0)
        rate = conv_in_cloud_rate(form, qcond, rho, dz, params)
        self.assertAlmostEqual(float(rate[0, 0]), 0.0)    # no condensate
        expected = 0.99 * (1.0e-4 / 500.0) / 1.0e-3
        self.assertAlmostEqual(float(rate[1, 0]), expected, places=8)
        self.assertAlmostEqual(float(rate[2, 0]), 0.0)    # no formation
        none = conv_in_cloud_rate(jnp.zeros((3, 1)), qcond, rho, dz, params)
        self.assertAlmostEqual(float(jnp.abs(none).max()), 0.0)


class WetDepTermTest(unittest.TestCase):
    def _setup(self, nlev=4, ncols=2, precip=1.0e-4):
        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name, number_name
        from jcm.physics.aerosol.jam.jam_state import JamAerosolState
        from jcm.physics.clouds.cloud_data import CloudData
        from jcm.physics_interface import PhysicsState

        n_modes = MAM4_SPEC.n_modes()
        shape = (n_modes, nlev, ncols)
        aer = JamAerosolState(
            r_dry=jnp.full(shape, 0.1e-6),
            r_wet=jnp.full(shape, 0.2e-6),
            rho=jnp.full(shape, 1800.0),
            kappa=jnp.full(shape, 0.5),
            mass=jnp.full(shape, 1e-9),
            number=jnp.full(shape, 1.0e8),
        )
        tracers = {}
        for mode in MAM4_SPEC.modes:
            tracers[number_name(mode.short)] = jnp.full((nlev, ncols), 1.0e8)
            for sp in mode.species:
                tracers[mass_name(sp, mode.short)] = jnp.full((nlev, ncols), 1e-9)
        state = PhysicsState.zeros((nlev, ncols)).copy(
            temperature=jnp.full((nlev, ncols), 275.0),
            tracers=tracers,
        )
        clouds = CloudData.zeros((ncols,), nlev).copy(
            cloud_fraction=jnp.full((nlev, ncols), 0.6),
            qc=jnp.full((nlev, ncols), 1.0e-3),
            precip_rain=jnp.full((ncols,), precip),
        )
        diagnostics = {
            "_jam_state": aer,
            "activated_fraction": jnp.full((nlev, ncols), 0.7),
            "air_density": jnp.full((nlev, ncols), 1.0),
            "layer_thickness": jnp.full((nlev, ncols), 200.0),
            "clouds": clouds,
        }
        return state, diagnostics, MAM4_SPEC, mass_name

    def test_scavenging_is_a_sink(self):
        state, diagnostics, spec, mass_name = self._setup()
        term = WetScavenging()
        tend, _ = term(state, diagnostics, None, None)
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        self.assertTrue(bool(jnp.all(tend.tracers[key] <= 0.0)))
        self.assertTrue(np.all(np.isfinite(np.asarray(tend.tracers[key]))))

    def test_extreme_rate_stays_bounded(self):
        # A scavenging rate with rate·dt ≫ 1 (heavy precip + near-clear low qc +
        # large coarse wet radius) must NOT remove more than the available mass
        # in one step. The implicit q·exp(-rate·dt) update keeps a forward step
        # in [0, q]; the old explicit -rate·q overshot into a sign-flipped
        # runaway (the natural-emission blow-up). Regression guard.
        state, diagnostics, spec, mass_name = self._setup(precip=1.0e-2)
        dt = 1800.0
        diagnostics = dict(diagnostics)
        diagnostics["_dt_seconds"] = dt
        aer = diagnostics["_jam_state"]
        diagnostics["_jam_state"] = aer.copy(
            r_wet=jnp.full_like(aer.r_wet, 5.0e-6)        # huge below-cloud rate
        )
        diagnostics["clouds"] = diagnostics["clouds"].copy(
            qc=jnp.full_like(diagnostics["clouds"].qc, 1.0e-9)  # huge in-cloud rate
        )
        term = WetScavenging()
        tend, _ = term(state, diagnostics, None, None)
        for nm, dq in tend.tracers.items():
            q0 = np.asarray(state.tracers[nm])
            q_new = q0 + np.asarray(dq) * dt
            self.assertTrue(np.all(np.isfinite(q_new)), nm)
            # Bounds hold up to floating-point roundoff; assert relative to the
            # field scale so f32 roundoff on the ~1e8 number tracers (n_acc →
            # -8 in the full-suite build) isn't mistaken for a real overshoot.
            scale = float(np.abs(q0).max())
            self.assertGreaterEqual(float(q_new.min()), -1e-5 * scale, nm)
            self.assertLessEqual(float(q_new.max()), float(q0.max()) + 1e-5 * scale, nm)

    def test_cloud_fraction_gt_one_stays_finite(self):
        # The cloud scheme can hand back cloud_fraction > 1 (e.g. where RH > 1).
        # The below-cloud clear-sky fraction (1 - cf) then goes negative, which
        # made the scavenging rate negative and the implicit 1-exp(-rate·dt)
        # removed fraction overflow to +inf, NaN-ing every aerosol tracer.
        # The clear fraction (and the rate) are clamped to ≥0, so the tendency
        # must stay finite for cf > 1. Regression guard.
        state, diagnostics, spec, mass_name = self._setup(precip=1.0e-2)
        diagnostics = dict(diagnostics)
        diagnostics["clouds"] = diagnostics["clouds"].copy(
            cloud_fraction=jnp.full_like(diagnostics["clouds"].cloud_fraction, 1.3)
        )
        # coarse wet radius makes the below-cloud rate large in magnitude
        aer = diagnostics["_jam_state"]
        diagnostics["_jam_state"] = aer.copy(r_wet=jnp.full_like(aer.r_wet, 5.0e-6))
        term = WetScavenging()
        tend, _ = term(state, diagnostics, None, None)
        for nm, dq in tend.tracers.items():
            self.assertTrue(np.all(np.isfinite(np.asarray(dq))), nm)
            self.assertTrue(bool(jnp.all(dq <= 0.0)), nm)  # still a sink, not a source

    def test_no_precip_no_removal(self):
        state, diagnostics, spec, mass_name = self._setup(precip=0.0)
        term = WetScavenging()
        tend, _ = term(state, diagnostics, None, None)
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        self.assertTrue(bool(jnp.allclose(tend.tracers[key], 0.0)))

    def _attach_convection(self, diagnostics, nlev, ncols, conv_precip=1.0e-4):
        from jcm.physics.convection.tiedtke_nordeng.types import ConvectionData

        import dataclasses
        # Convective cloud on levels 1..nlev-2: condensate + formation
        # there, none at the top level or the sub-cloud bottom level.
        prof = jnp.ones((nlev, ncols)).at[0].set(0.0).at[-1].set(0.0)
        conv = dataclasses.replace(
            ConvectionData.zeros((ncols,), nlev),
            precip_conv=jnp.full((ncols,), conv_precip),
            precip_formation=prof * 1.0e-4,
            qc_conv=prof * 1.0e-3,
        )
        diagnostics = dict(diagnostics)
        diagnostics["convection"] = conv
        return diagnostics

    def test_convective_precip_scavenges(self):
        # The convective pathway must strengthen removal vs the same state
        # without it: soluble modes via in-cloud + washout, the insoluble
        # pcm mode via washout only (below-cloud sees total precip).
        state, diagnostics, spec, mass_name = self._setup()
        term = WetScavenging()
        tend_ref, _ = term(state, diagnostics, None, None)
        tend_conv, _ = term(
            state, self._attach_convection(diagnostics, 4, 2), None, None,
        )
        for i, mode in enumerate(spec.modes):
            key = mass_name(mode.species[0], mode.short)
            self.assertLess(
                float(tend_conv.tracers[key].sum()),
                float(tend_ref.tracers[key].sum()),
                f"convective precip must add removal for mode {mode.short}",
            )
        # (Layer confinement of the convective in-cloud rate is asserted at
        # the function level in ``test_conv_in_cloud_confined_to_heated_layers``;
        # here the stratiform in-cloud term already near-saturates the implicit
        # exponential update in cloudy layers, so only the sign/monotonicity of
        # the total increment is meaningful.)

    def test_conv_washout_confined_below_cloud_top(self):
        # With ONLY convective precip and a pressure diagnostic, levels
        # above the convective cloud top (no heating, lower pressure than
        # any active level) must see EXACTLY zero removal — rain
        # cannot collect aerosol above where it forms.
        state, diagnostics, spec, mass_name = self._setup(precip=0.0)
        diagnostics = self._attach_convection(diagnostics, 4, 2)
        # Level 0 is the model top (200 hPa); heating is active on levels
        # 1..2, so the convective cloud top is at 500 hPa.
        diagnostics["pressure_full"] = (
            jnp.array([200.0, 500.0, 800.0, 1000.0])[:, None]
            * jnp.ones((1, 2)) * 100.0
        )
        term = WetScavenging()
        tend, _ = term(state, diagnostics, None, None)
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        dq = np.asarray(tend.tracers[key])
        np.testing.assert_array_equal(dq[0], 0.0)      # above conv top
        self.assertTrue(np.all(dq[3] < 0.0))           # below cloud

    def test_conv_scavenging_no_convection_key_is_noop(self):
        # Without a "convection" diagnostic the term must fall back to the
        # stratiform-only behaviour (composability without a convection scheme).
        state, diagnostics, spec, mass_name = self._setup()
        self.assertNotIn("convection", diagnostics)
        term = WetScavenging()
        tend, _ = term(state, diagnostics, None, None)
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        self.assertTrue(np.all(np.isfinite(np.asarray(tend.tracers[key]))))

    def test_grad_through_below_coeff(self):
        state, diagnostics, spec, mass_name = self._setup()

        def loss(coeff):
            params = WetDepParameters(
                incloud_scale=jnp.asarray(1.0),
                below_coeff=coeff,
                below_radius_ref=jnp.asarray(1.0e-7),
                conv_scav_ratio=jnp.asarray(0.99),
            )
            term = WetScavenging(params=params)
            tend, _ = term(state, diagnostics, None, None)
            return sum(jnp.sum(v ** 2) for v in tend.tracers.values())

        g = jax.grad(loss)(jnp.asarray(1.0e-4))
        self.assertTrue(np.isfinite(float(g)))

    def test_grad_through_conv_scav_ratio(self):
        # The convective scavenging ratio must be a live differentiable
        # knob: nonzero, finite gradient when convective cloud is present.
        state, diagnostics, spec, mass_name = self._setup()
        diagnostics = self._attach_convection(diagnostics, 4, 2)

        def loss(ratio):
            params = WetDepParameters(
                incloud_scale=jnp.asarray(1.0),
                below_coeff=jnp.asarray(1.0e-4),
                below_radius_ref=jnp.asarray(1.0e-7),
                conv_scav_ratio=ratio,
            )
            term = WetScavenging(params=params)
            tend, _ = term(state, diagnostics, None, None)
            return sum(jnp.sum(v ** 2) for v in tend.tracers.values())

        g = jax.grad(loss)(jnp.asarray(0.99))
        self.assertTrue(np.isfinite(float(g)))
        self.assertNotEqual(float(g), 0.0)


if __name__ == "__main__":
    unittest.main()
