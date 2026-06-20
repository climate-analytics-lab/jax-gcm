"""Phase 5 tests: in-cloud + below-cloud scavenging and the term."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.wetdep.wetdep_term import (
    WetScavenging,
    WetDepParameters,
    below_cloud_rate,
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

    def test_no_precip_no_removal(self):
        state, diagnostics, spec, mass_name = self._setup(precip=0.0)
        term = WetScavenging()
        tend, _ = term(state, diagnostics, None, None)
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        self.assertTrue(bool(jnp.allclose(tend.tracers[key], 0.0)))

    def test_grad_through_below_coeff(self):
        state, diagnostics, spec, mass_name = self._setup()

        def loss(coeff):
            params = WetDepParameters(
                incloud_scale=jnp.asarray(1.0),
                below_coeff=coeff,
                below_radius_ref=jnp.asarray(1.0e-7),
            )
            term = WetScavenging(params=params)
            tend, _ = term(state, diagnostics, None, None)
            return sum(jnp.sum(v ** 2) for v in tend.tracers.values())

        g = jax.grad(loss)(jnp.asarray(1.0e-4))
        self.assertTrue(np.isfinite(float(g)))


if __name__ == "__main__":
    unittest.main()
