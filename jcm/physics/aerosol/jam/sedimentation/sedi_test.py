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
    # A monodisperse mode (sigma = 1) reduces the moment weighting to 1, so
    # these shape checks isolate the single-particle physics.
    MONO = dict(geom_std_dev=1.0, moment=0)

    def test_larger_particles_fall_faster(self):
        t = jnp.full((1,), 280.0)
        p = jnp.full((1,), 9.0e4)
        v_small = stokes_velocity(jnp.full((1,), 0.05e-6), jnp.full((1,), 1800.0), t, p,
                                  **self.MONO)
        v_big = stokes_velocity(jnp.full((1,), 2.0e-6), jnp.full((1,), 1800.0), t, p,
                                **self.MONO)
        self.assertGreater(float(v_big[0]), float(v_small[0]))

    def test_velocity_positive_and_finite(self):
        t = jnp.full((3, 2), 270.0)
        p = jnp.full((3, 2), 8.0e4)
        v = stokes_velocity(jnp.full((3, 2), 1.0e-6), jnp.full((3, 2), 2000.0), t, p,
                            **self.MONO)
        self.assertTrue(bool(jnp.all(v > 0)))
        self.assertTrue(np.all(np.isfinite(np.asarray(v))))

    def test_coarse_dust_velocity_reasonable(self):
        # 1 µm radius, 2600 kg/m³ → order 1e-4..1e-3 m/s.
        v = stokes_velocity(
            jnp.array([1.0e-6]), jnp.array([2600.0]),
            jnp.array([288.0]), jnp.array([1.0e5]), **self.MONO,
        )
        self.assertTrue(1e-5 < float(v[0]) < 1e-2)

    def test_matches_seinfeld_pandis_single_particle(self):
        """Monodisperse velocities match Seinfeld & Pandis Table 9.5.

        Unit-density spheres at 1 atm / 298 K. This pins the single-particle
        physics (slip correction, viscosity, mean free path) independently of
        the lognormal weighting below.
        """
        for diameter_um, expected in ((0.1, 8.6e-7), (1.0, 3.5e-5), (10.0, 3.1e-3)):
            v = stokes_velocity(
                jnp.array([diameter_um * 0.5e-6]), jnp.array([1000.0]),
                jnp.array([298.0]), jnp.array([101325.0]), **self.MONO,
            )
            self.assertAlmostEqual(float(v[0]) / expected, 1.0, delta=0.05,
                                   msg=f"D = {diameter_um} um")

    def test_mass_settles_faster_than_number(self):
        """Mass rides the coarse tail of the mode, so it settles faster.

        For v proportional to r^2 the ratio is exp(6 ln^2 sigma) in the
        continuum limit; the slip correction is weaker at the larger
        mass-median radius, so the realised ratio is somewhat smaller.
        """
        args = (jnp.array([0.055e-6]), jnp.array([1770.0]),
                jnp.array([288.0]), jnp.array([1.0e5]))
        v_num = stokes_velocity(*args, geom_std_dev=1.8, moment=0)
        v_mass = stokes_velocity(*args, geom_std_dev=1.8, moment=3)
        ratio = float(v_mass[0] / v_num[0])
        self.assertGreater(ratio, 3.0)
        self.assertLess(ratio, np.exp(6 * np.log(1.8) ** 2))

    def test_matches_cam_modal_aero_depvel_part(self):
        """Reproduce CAM's bulk settling velocity for the accumulation mode.

        Reference values from ``modal_aero_depvel_part``
        (ESCOMP/CAM cam_development, ``aero_model.F90:1575-1596``) evaluated
        for MAM4 accumulation: dg = 0.11 um, sigma = 1.8, rho = 1770 kg/m3.
        CAM's slightly different Sutherland constants keep this to ~1%.
        """
        cases = (
            # (pressure Pa, temperature K, moment, CAM v_grav m/s)
            (1.0e5, 288.0, 0, 3.45e-6),
            (1.0e5, 288.0, 3, 1.58e-5),
            (2.0e4, 220.0, 0, 1.20e-5),
            (2.0e4, 220.0, 3, 4.00e-5),
        )
        for pressure, temperature, moment, expected in cases:
            v = stokes_velocity(
                jnp.array([0.055e-6]), jnp.array([1770.0]),
                jnp.array([temperature]), jnp.array([pressure]),
                geom_std_dev=1.8, moment=moment,
            )
            self.assertAlmostEqual(
                float(v[0]) / expected, 1.0, delta=0.03,
                msg=f"p={pressure} T={temperature} moment={moment}")

    def test_wet_radius_capped(self):
        # HAMMOZ caps the settling diameter at 50 µm (25 µm radius), so a
        # runaway κ-Köhler wet-growth tail can't inflate the fall speed: the
        # velocity at 25 µm and 1 mm must be identical (both clamped to 25 µm),
        # and strictly above a sub-cap radius.
        t, p, rho = jnp.array([280.0]), jnp.array([9.0e4]), jnp.array([1800.0])
        v_cap = stokes_velocity(jnp.array([25.0e-6]), rho, t, p, **self.MONO)
        v_huge = stokes_velocity(jnp.array([1.0e-3]), rho, t, p, **self.MONO)
        v_sub = stokes_velocity(jnp.array([10.0e-6]), rho, t, p, **self.MONO)
        np.testing.assert_allclose(float(v_huge[0]), float(v_cap[0]), rtol=1e-6)
        self.assertGreater(float(v_cap[0]), float(v_sub[0]))


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
            "_dt_seconds": 720.0,
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

    def test_cfl_cap_keeps_tendency_stable_for_extreme_velocity(self):
        # Coarse-mode sea-salt-like extreme: huge wet radius + thin layers +
        # a long step would give a Courant number ≫ 1 and an unstable explicit
        # donor-cell step (the natural-emission blowup). The CFL cap (v ≤ dz/dt)
        # must keep a forward Euler step finite and non-negative.
        state, diagnostics = self._setup(nlev=6, ncols=2)
        dt = 1800.0
        diagnostics = dict(diagnostics)
        diagnostics["_dt_seconds"] = dt
        diagnostics["layer_thickness"] = jnp.full((6, 2), 20.0)  # thin → easy CFL break
        aer = diagnostics["_jam_state"]
        # Inflate every mode's wet radius well past the 25 µm cap.
        diagnostics["_jam_state"] = aer.copy(r_wet=jnp.full_like(aer.r_wet, 1.0e-4))

        term = StokesSedimentation()
        tend, _ = term(state, diagnostics, None, None)
        for nm, dq in tend.tracers.items():
            q_new = np.asarray(state.tracers[nm]) + np.asarray(dq) * dt
            self.assertTrue(np.all(np.isfinite(q_new)), nm)
            # The CFL cap keeps the donor-cell step non-negative up to
            # floating-point roundoff. Assert relative to the field scale: a
            # real Courant>1 runaway drives q_new to O(-q); harmless f32
            # roundoff at the CFL boundary (q(1-v·dt/dz) with v·dt/dz→1+eps) is
            # ~1e-7·q, which an absolute -1e-20 bound spuriously rejects for the
            # ~1e8 number tracers (n_acc → -8 in the full-suite build).
            scale = float(np.abs(q_new).max())
            self.assertGreaterEqual(float(q_new.min()), -1e-5 * scale, nm)

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
