"""Phase 3 tests: dry-deposition resistances, velocity, and the term."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.drydep.drydep_term import (
    DryDepParameters,
    SlinnDryDeposition,
)
from jcm.physics.aerosol.jam.drydep.resistances import (
    aerodynamic_resistance,
    deposition_velocity,
)


class ResistanceTest(unittest.TestCase):
    def test_aerodynamic_resistance_decreases_with_ustar(self):
        ra_calm = aerodynamic_resistance(jnp.asarray(0.05))
        ra_windy = aerodynamic_resistance(jnp.asarray(1.0))
        self.assertGreater(float(ra_calm), float(ra_windy))

    def test_deposition_velocity_positive_finite(self):
        v = deposition_velocity(
            r_wet=jnp.asarray(0.1e-6), v_grav=jnp.asarray(1e-5),
            u_star=jnp.asarray(0.3), temperature=jnp.asarray(285.0),
            pressure=jnp.asarray(1.0e5), air_density=jnp.asarray(1.2),
        )
        self.assertGreater(float(v), 0.0)
        self.assertTrue(np.isfinite(float(v)))

    def test_vdep_has_brownian_minimum_for_small_particles(self):
        # Very small particles deposit faster (Brownian) than ~0.3 µm
        # accumulation-mode particles (the classic deposition minimum).
        kw = dict(v_grav=jnp.asarray(1e-7), u_star=jnp.asarray(0.4),
                  temperature=jnp.asarray(285.0), pressure=jnp.asarray(1.0e5),
                  air_density=jnp.asarray(1.2))
        v_tiny = deposition_velocity(r_wet=jnp.asarray(2e-9), **kw)
        v_accum = deposition_velocity(r_wet=jnp.asarray(0.3e-6), **kw)
        self.assertGreater(float(v_tiny), float(v_accum))


class DryDepTermTest(unittest.TestCase):
    def _setup(self, nlev=4, ncols=2):
        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name, number_name
        from jcm.physics.aerosol.jam.jam_state import JamAerosolState
        from jcm.physics_interface import PhysicsState

        n_modes = MAM4_SPEC.n_modes()
        shape = (n_modes, nlev, ncols)
        aer = JamAerosolState(
            r_dry=jnp.full(shape, 0.1e-6),
            r_wet=jnp.full(shape, 0.15e-6),
            rho=jnp.full(shape, 1800.0),
            kappa=jnp.full(shape, 0.5),
            mass=jnp.full(shape, 1e-9),
            number=jnp.full(shape, 1.0e8),
        )
        from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY

        tracers = {}
        carry = {}
        for mode in MAM4_SPEC.modes:
            tracers[number_name(mode.short)] = jnp.full((nlev, ncols), 1.0e8)
            carry[number_name(mode.short, cloud_borne=True)] = jnp.full(
                (nlev, ncols), 1.0e8
            )
            for sp in mode.species:
                tracers[mass_name(sp, mode.short)] = jnp.full(
                    (nlev, ncols), 1e-9
                )
                carry[mass_name(sp, mode.short, cloud_borne=True)] = (
                    jnp.full((nlev, ncols), 1e-9)
                )
        state = PhysicsState.zeros((nlev, ncols)).copy(
            temperature=jnp.full((nlev, ncols), 285.0),
            tracers=tracers,
        )
        diagnostics = {
            CARRY_KEY: carry,
            "_jam_state": aer,
            "air_density": jnp.full((nlev, ncols), 1.2),
            "layer_thickness": jnp.full((nlev, ncols), 100.0),
            "pressure_full": jnp.full((nlev, ncols), 1.0e5),
        }
        return state, diagnostics, MAM4_SPEC, mass_name

    def test_only_bottom_layer_deposits(self):
        state, diagnostics, spec, mass_name = self._setup()
        term = SlinnDryDeposition()
        tend, _ = term(state, diagnostics, None, None)
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        dq = tend.tracers[key]
        # Loss only at the surface layer; aloft is zero.
        self.assertLess(float(dq[-1, 0]), 0.0)
        self.assertTrue(bool(jnp.all(dq[:-1] == 0.0)))

    def test_cloud_borne_deposits_only_with_explicit_phase(self):
        # With the default (explicit cloud-borne) population the carry
        # fields deposit at the surface too (#602); with the implicit
        # population the carry is untouched and no cloud-borne tendencies
        # are emitted at all.
        import dataclasses
        from jcm.physics.aerosol.jam import MAM4_SPEC
        from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY

        state, diagnostics, spec, mass_name = self._setup()
        _, out = SlinnDryDeposition()(state, diagnostics, None, None)
        cb_key = mass_name(
            spec.modes[0].species[0], spec.modes[0].short, cloud_borne=True,
        )
        delta = (
            np.asarray(out[CARRY_KEY][cb_key])
            - np.asarray(diagnostics[CARRY_KEY][cb_key])
        )
        self.assertLess(float(delta[-1, 0]), 0.0)
        self.assertTrue(bool(jnp.all(delta[:-1] == 0.0)))

        implicit = SlinnDryDeposition(
            spec=dataclasses.replace(MAM4_SPEC, cloud_borne=False)
        )
        tend, out = implicit(state, diagnostics, None, None)
        self.assertFalse(
            any(nm.startswith(("mc_", "nc_")) for nm in tend.tracers)
        )
        np.testing.assert_array_equal(
            np.asarray(out[CARRY_KEY][cb_key]),
            np.asarray(diagnostics[CARRY_KEY][cb_key]),
        )

    def test_uses_vertical_diffusion_ustar_when_present(self):
        from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (
            VerticalDiffusionData,
        )

        state, diagnostics, spec, mass_name = self._setup()
        nlev, ncols = state.temperature.shape
        vd = VerticalDiffusionData.zeros((ncols,), nlev).copy(
            surface_friction_velocity=jnp.full((ncols,), 0.8),
        )
        diagnostics["vertical_diffusion"] = vd
        term = SlinnDryDeposition()
        tend, _ = term(state, diagnostics, None, None)
        key = mass_name(spec.modes[0].species[0], spec.modes[0].short)
        self.assertTrue(np.all(np.isfinite(np.asarray(tend.tracers[key]))))

    def test_grad_finite(self):
        state, diagnostics, spec, mass_name = self._setup()

        def loss(z0):
            term = SlinnDryDeposition(
                params=DryDepParameters(
                    z_ref=jnp.asarray(10.0), z0=z0,
                    u_star_default=jnp.asarray(0.3),
                )
            )
            tend, _ = term(state, diagnostics, None, None)
            return sum(jnp.sum(v ** 2) for v in tend.tracers.values())

        g = jax.grad(loss)(jnp.asarray(1.0e-4))
        self.assertTrue(np.isfinite(float(g)))


if __name__ == "__main__":
    unittest.main()
