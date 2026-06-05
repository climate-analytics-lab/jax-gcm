"""Phase 2 tests: Stokes velocity, donor-cell transport, mass conservation."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.sedimentation.sedi_term import (
    StokesSedimentation,
    sediment_column,
    stokes_velocity,
)


class StokesVelocityTest(unittest.TestCase):
    def test_larger_particles_fall_faster(self):
        t = jnp.full((1,), 280.0)
        p = jnp.full((1,), 9.0e4)
        v_small = stokes_velocity(jnp.full((1,), 0.05e-6), jnp.full((1,), 1800.0), t, p)
        v_big = stokes_velocity(jnp.full((1,), 2.0e-6), jnp.full((1,), 1800.0), t, p)
        self.assertGreater(float(v_big[0]), float(v_small[0]))

    def test_velocity_positive_and_finite(self):
        t = jnp.full((3, 2), 270.0)
        p = jnp.full((3, 2), 8.0e4)
        v = stokes_velocity(jnp.full((3, 2), 1.0e-6), jnp.full((3, 2), 2000.0), t, p)
        self.assertTrue(bool(jnp.all(v > 0)))
        self.assertTrue(np.all(np.isfinite(np.asarray(v))))

    def test_coarse_dust_velocity_reasonable(self):
        # 1 µm radius, 2600 kg/m³ → order 1e-4..1e-3 m/s.
        v = stokes_velocity(
            jnp.array([1.0e-6]), jnp.array([2600.0]),
            jnp.array([288.0]), jnp.array([1.0e5]),
        )
        self.assertTrue(1e-5 < float(v[0]) < 1e-2)


class DonorCellTest(unittest.TestCase):
    def test_zero_velocity_no_change(self):
        q = jnp.linspace(1.0, 4.0, 4).reshape(4, 1)
        rho = jnp.ones((4, 1))
        dz = jnp.ones((4, 1)) * 100.0
        dq, surf = sediment_column(q, jnp.zeros((4, 1)), rho, dz)
        self.assertTrue(bool(jnp.allclose(dq, 0.0)))
        self.assertAlmostEqual(float(jnp.squeeze(surf)), 0.0)

    def test_mass_change_equals_bottom_flux(self):
        # Column burden change must equal minus the flux out the bottom.
        nlev = 5
        q = jnp.linspace(2.0, 1.0, nlev).reshape(nlev, 1)
        rho = jnp.full((nlev, 1), 1.0)
        dz = jnp.full((nlev, 1), 50.0)
        v = jnp.full((nlev, 1), 0.01)
        dq, surf = sediment_column(q, v, rho, dz)
        burden_rate = jnp.sum(rho * dz * dq)
        np.testing.assert_allclose(float(burden_rate), -float(jnp.squeeze(surf)), rtol=1e-5)

    def test_mass_moves_downward(self):
        # A single loaded top layer loses mass; the layer below gains.
        nlev = 4
        q = jnp.zeros((nlev, 1)).at[0, 0].set(1.0)
        rho = jnp.ones((nlev, 1))
        dz = jnp.ones((nlev, 1))
        v = jnp.full((nlev, 1), 0.5)
        dq, _ = sediment_column(q, v, rho, dz)
        self.assertLess(float(dq[0, 0]), 0.0)   # top loses
        self.assertGreater(float(dq[1, 0]), 0.0)  # next gains


class SedimentationTermTest(unittest.TestCase):
    def _setup(self, nlev=4, ncols=2):
        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name, number_name
        from jcm.physics.aerosol.jam.jam_state import JamAerosolState
        from jcm.physics_interface import PhysicsState

        n_modes = MAM4_SPEC.n_modes()
        shape = (n_modes, nlev, ncols)
        aer = JamAerosolState(
            r_dry=jnp.full(shape, 0.1e-6),
            r_wet=jnp.full(shape, 0.2e-6),
            rho=jnp.full(shape, 2000.0),
            kappa=jnp.full(shape, 0.4),
            mass=jnp.full(shape, 1e-9),
            number=jnp.full(shape, 1.0e8),
        )
        tracers = {}
        for mode in MAM4_SPEC.modes:
            tracers[number_name(mode.short)] = jnp.full((nlev, ncols), 1.0e8)
            for sp in mode.species:
                tracers[mass_name(sp, mode.short)] = jnp.full((nlev, ncols), 1e-9)
        state = PhysicsState.zeros((nlev, ncols)).copy(
            temperature=jnp.full((nlev, ncols), 280.0),
            tracers=tracers,
        )
        diagnostics = {
            "_jam_state": aer,
            "air_density": jnp.full((nlev, ncols), 1.0),
            "layer_thickness": jnp.full((nlev, ncols), 200.0),
            "pressure_full": jnp.full((nlev, ncols), 9.0e4),
        }
        return state, diagnostics

    def test_term_produces_tracer_tendencies(self):
        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name

        state, diagnostics = self._setup()
        term = StokesSedimentation()
        tend, _ = term(state, diagnostics, None, None)
        key = mass_name(MAM4_SPEC.modes[2].species[0], MAM4_SPEC.modes[2].short)
        self.assertIn(key, tend.tracers)
        self.assertTrue(np.all(np.isfinite(np.asarray(tend.tracers[key]))))
        # Top layer should be losing aerosol (negative tendency).
        self.assertLessEqual(float(tend.tracers[key][0, 0]), 0.0)

    def test_grad_through_velocity_scale_finite(self):
        from jcm.physics.aerosol.jam.sedimentation.sedi_term import SedParameters

        state, diagnostics = self._setup()

        def loss(scale):
            term = StokesSedimentation(
                params=SedParameters(velocity_scale=scale)
            )
            tend, _ = term(state, diagnostics, None, None)
            return sum(jnp.sum(v ** 2) for v in tend.tracers.values())

        g = jax.grad(loss)(jnp.asarray(1.0))
        self.assertTrue(np.isfinite(float(g)))


if __name__ == "__main__":
    unittest.main()
