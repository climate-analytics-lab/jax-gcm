"""Tests for the dycore-side gridpoint tracer filters."""

import unittest

import numpy as np
import jax.numpy as jnp

from jcm.filters import mass_conserving_positivity, MassConservingPositivity


class MassConservingPositivityMathTest(unittest.TestCase):
    """The core hole-filling clip: non-negative + column-mass-conserving."""

    def test_nonnegative_and_conserves_column_mass(self):
        # (nlev, ncol): mix of positive and Gibbs-ringing negative values.
        q = jnp.asarray([[1.0, -0.5, 2.0],
                         [-0.3, 1.0, -1.0],
                         [0.8, 0.4, 3.0]])
        m = jnp.asarray([[1.0, 1.0, 1.0],
                         [2.0, 2.0, 2.0],
                         [1.5, 1.5, 1.5]])          # per-layer air mass ∝ Δp
        out = mass_conserving_positivity(q, m)

        # Non-negative everywhere.
        self.assertTrue(bool(jnp.all(out >= 0.0)))
        # Column mass preserved for columns whose net mass is positive.
        before = jnp.sum(m * q, axis=0)
        after = jnp.sum(m * out, axis=0)
        # All three test columns have positive net mass here.
        self.assertTrue(bool(jnp.all(before > 0.0)))
        np.testing.assert_allclose(np.asarray(after), np.asarray(before), rtol=1e-6)

    def test_net_negative_column_is_zeroed(self):
        # A column whose mass-weighted sum is negative can't be made positive
        # while conserving mass — it's zeroed (the only non-conservation).
        q = jnp.asarray([[-2.0], [-1.0], [0.5]])
        m = jnp.ones_like(q)
        out = mass_conserving_positivity(q, m)
        self.assertTrue(bool(jnp.all(out == 0.0)))

    def test_already_nonnegative_is_unchanged(self):
        q = jnp.asarray([[1.0, 0.0], [2.0, 3.0], [0.5, 1.0]])
        m = jnp.asarray([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]])
        out = mass_conserving_positivity(q, m)
        np.testing.assert_allclose(np.asarray(out), np.asarray(q), rtol=1e-6)

    def test_filter_object_applies_to_every_tracer(self):
        dp = jnp.asarray([[1.0, 1.0], [1.0, 1.0]])
        tracers = {
            'a': jnp.asarray([[1.0, -0.5], [-0.2, 1.0]]),
            'b': jnp.asarray([[0.0, 2.0], [3.0, -1.0]]),
        }
        out = MassConservingPositivity()(tracers, dp)
        self.assertEqual(set(out), {'a', 'b'})
        for k in tracers:
            self.assertTrue(bool(jnp.all(out[k] >= 0.0)))


class DycoreTracerFilterWiringTest(unittest.TestCase):
    """The dycore applies ``tracer_filter`` inside ``to_physics_state`` with a
    correctly-shaped ``dp``, and is an exact no-op when no filter is set.
    """

    def _build_dycore(self, tracer_filter):
        from jcm.dycore.dinosaur.dycore import DinosaurDycore
        from jcm.physics.physics_term import TracerSpec
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords
        sigma_b = np.linspace(0, 1, 9)
        coords = get_coords(sigma_b, spectral_truncation=21)
        terrain = TerrainData.aquaplanet(coords)
        # An additional tracer (beyond specific_humidity, which is a top-level
        # PhysicsState field) so it lands in ``physics_state.tracers`` where the
        # filter operates.
        return DinosaurDycore(
            coords=coords, terrain=terrain, dt_seconds=600.0,
            tracer_specs={'qc': TracerSpec("qc", units="kg/kg")},
            tracer_filter=tracer_filter,
        ), coords

    def test_filter_is_called_with_layer_shaped_dp_and_applied(self):
        captured = {}

        def recording_filter(tracers, dp):
            captured['dp_shape'] = dp.shape
            return {k: v + 1.0 for k, v in tracers.items()}

        dy_filt, coords = self._build_dycore(recording_filter)
        dy_none, _ = self._build_dycore(None)
        state = dy_filt.initial_state(None)

        phys_none = dy_none.to_physics_state(state)
        phys_filt = dy_filt.to_physics_state(state)

        nlev = coords.vertical.layers
        q = phys_none.tracers['qc']
        # dp has the per-layer leading axis and matches the tracer field shape.
        self.assertEqual(captured['dp_shape'][0], nlev)
        self.assertEqual(captured['dp_shape'], q.shape)
        # The filter's effect is visible in the projected physics state.
        np.testing.assert_allclose(
            np.asarray(phys_filt.tracers['qc']),
            np.asarray(q) + 1.0, rtol=1e-6,
        )

    def test_no_filter_is_a_noop(self):
        dy_none, _ = self._build_dycore(None)
        state = dy_none.initial_state(None)
        # Should not raise and should round-trip the tracer untouched.
        phys = dy_none.to_physics_state(state)
        self.assertIn('qc', phys.tracers)


if __name__ == '__main__':
    unittest.main()
