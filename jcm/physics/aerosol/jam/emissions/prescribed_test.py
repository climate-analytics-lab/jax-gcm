"""Tests for the CAM6/MAM4-faithful pre-speciated emissions term (#498)."""

import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.emissions.prescribed import PreSpeciatedEmissions
from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name
from jcm.physics_interface import PhysicsState

_NLEV, _NCOLS = 5, 3
_RHO, _DZ = 1.1, 200.0


def _setup(**fields):
    state = PhysicsState.zeros((_NLEV, _NCOLS)).copy(
        temperature=jnp.full((_NLEV, _NCOLS), 280.0))
    diagnostics = {
        "air_density": jnp.full((_NLEV, _NCOLS), _RHO),
        "layer_thickness": jnp.full((_NLEV, _NCOLS), _DZ),
    }
    forcing = types.SimpleNamespace(prescribed_aerosol_emissions=fields or None)
    return state, diagnostics, forcing


def _column_integral(tend):
    """Σ ρ Δz · tend over levels → recovered surface flux [X/m²/s], (ncols,)."""
    return np.asarray(jnp.sum(tend * _RHO * _DZ, axis=0))


class PreSpeciatedEmissionsTest(unittest.TestCase):
    def test_zero_without_forcing(self):
        state, diagnostics, forcing = _setup()
        tend, _ = PreSpeciatedEmissions()(state, diagnostics, forcing, None)
        self.assertEqual(tend.tracers, {})

    def test_surface_field_loads_lowest_layer_mass_conserving(self):
        flux = 2.0e-10
        tracer = mass_name("so4", "acc")
        state, diagnostics, forcing = _setup(
            **{tracer: jnp.full((_NCOLS,), flux)})
        tend, _ = PreSpeciatedEmissions()(state, diagnostics, forcing, None)
        dq = tend.tracers[tracer]
        # All mass enters the lowest layer; the column integral recovers flux.
        self.assertTrue(np.all(np.asarray(dq[:-1]) == 0.0))
        np.testing.assert_allclose(_column_integral(dq), flux, rtol=1e-6)

    def test_volume_field_distributes_over_levels_mass_conserving(self):
        # A 3-D per-layer flux is added across model levels (the elevated /
        # mo_extfrc path); the column integral is the sum of the per-layer flux.
        per_layer = jnp.asarray([0.0, 1e-11, 3e-11, 0.0, 0.0])[:, None]
        field = jnp.broadcast_to(per_layer, (_NLEV, _NCOLS))
        tracer = mass_name("so4", "acc")
        state, diagnostics, forcing = _setup(**{tracer: field})
        tend, _ = PreSpeciatedEmissions()(state, diagnostics, forcing, None)
        np.testing.assert_allclose(
            _column_integral(tend.tracers[tracer]),
            np.asarray(jnp.sum(field, axis=0)), rtol=1e-6)
        # Mass genuinely lands aloft, not just at the surface.
        self.assertGreater(float(tend.tracers[tracer][1, 0]), 0.0)

    def test_multiple_tracers_independent(self):
        state, diagnostics, forcing = _setup(
            **{mass_name("bc", "pcm"): jnp.full((_NCOLS,), 1e-10),
               number_name("pcm"): jnp.full((_NCOLS,), 5e6)})
        tend, _ = PreSpeciatedEmissions()(state, diagnostics, forcing, None)
        self.assertIn(mass_name("bc", "pcm"), tend.tracers)
        self.assertIn(number_name("pcm"), tend.tracers)

    def test_scale_multiplies_emission(self):
        tracer = mass_name("bc", "pcm")
        state, diagnostics, forcing = _setup(
            **{tracer: jnp.full((_NCOLS,), 1e-10)})
        tend, _ = PreSpeciatedEmissions(scale=2.0)(
            state, diagnostics, forcing, None)
        np.testing.assert_allclose(_column_integral(tend.tracers[tracer]),
                                   2.0e-10, rtol=1e-6)

    def test_grad_of_mmr_wrt_emission_field_finite(self):
        # The user-facing point: even without an injection-height parameter,
        # ∂(aerosol mmr tendency)/∂(emission ForcingData field) is well-defined.
        tracer = mass_name("so4", "acc")
        state, diagnostics, _ = _setup()

        def loss(flux_field):
            forcing = types.SimpleNamespace(
                prescribed_aerosol_emissions={tracer: flux_field})
            tend, _ = PreSpeciatedEmissions()(state, diagnostics, forcing, None)
            return jnp.sum(tend.tracers[tracer] ** 2)

        x = jnp.full((_NCOLS,), 2.0e-10)
        g = jax.grad(loss)(x)
        self.assertTrue(np.all(np.isfinite(np.asarray(g))))
        self.assertTrue(np.all(np.asarray(g) > 0.0))


class FactoryWiringTest(unittest.TestCase):
    def test_default_excludes_prescribed(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics
        names = [t.name for t in jam_aerosol_physics()]
        self.assertNotIn("jam_prescribed_aerosol_emissions", names)

    def test_flag_includes_prescribed(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics
        names = [t.name for t in jam_aerosol_physics(prescribed_speciated=True)]
        self.assertIn("jam_prescribed_aerosol_emissions", names)


if __name__ == "__main__":
    unittest.main()
