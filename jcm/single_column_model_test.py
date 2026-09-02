"""Tests for ``jcm.single_column_model.SingleColumnModel``."""

import unittest

import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import tree_map
from dinosaur.sigma_coordinates import SigmaCoordinates

from jcm.constants import grav
from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
from jcm.physics.echam.echam_terms import echam_physics
from jcm.physics_interface import Physics, PhysicsState, PhysicsTendency
from jcm.single_column_model import SCMPredictions, SingleColumnModel
from jcm.utils import create_initial_tracers, create_single_column_state


class _IdentityTempPhysics(Physics):
    """Minimal physics whose temperature tendency equals the temperature.

    Lets the free-evolution and ``state_closure`` hooks be tested with exact
    arithmetic: with ``dt`` seconds, ``T_{n+1} = T_n + dt·T_seen``, where
    ``T_seen`` is whatever temperature the closure leaves on the state.
    """

    def compute_tendencies(self, state, forcing, terrain, prev_physics_data=None):
        tend = PhysicsTendency.zeros(state.temperature.shape).copy(
            temperature=state.temperature,
        )
        return tend, (prev_physics_data if prev_physics_data is not None else {})

    def get_empty_data(self, coords):
        return {}

    def initial_carry_state(self, coords):
        return {}


class _DryingPhysics(Physics):
    """Returns a constant strong negative specific-humidity tendency.

    Used to check that a freely evolving ``specific_humidity`` is floored at
    zero in the carry rather than going negative.
    """

    def compute_tendencies(self, state, forcing, terrain, prev_physics_data=None):
        tend = PhysicsTendency.zeros(state.temperature.shape).copy(
            specific_humidity=jnp.full_like(state.specific_humidity, -1.0),
        )
        return tend, (prev_physics_data if prev_physics_data is not None else {})

    def get_empty_data(self, coords):
        return {}

    def initial_carry_state(self, coords):
        return {}


class _ConstantTracerTendencyPhysics(Physics):
    """Gives every tracer the same constant positive tendency."""

    def __init__(self, rate: float = 1e-6):
        self.rate = rate

    def compute_tendencies(self, state, forcing, terrain, prev_physics_data=None):
        tend = PhysicsTendency.zeros(state.temperature.shape).copy(
            tracers={k: jnp.full_like(v, self.rate)
                     for k, v in state.tracers.items()},
        )
        return tend, (prev_physics_data if prev_physics_data is not None else {})

    def get_empty_data(self, coords):
        return {}

    def initial_carry_state(self, coords):
        return {}


def _make_column_state(nlev: int) -> PhysicsState:
    """Build a vertically stratified 1-D column state."""
    z = jnp.linspace(0, 30000, nlev)[::-1]
    t_profile = jnp.maximum(288.0 - 6.5e-3 * z, 200.0)
    q_profile = 0.012 * jnp.exp(-z / 3000.0)
    return PhysicsState(
        u_wind=jnp.full(nlev, 5.0),
        v_wind=jnp.zeros(nlev),
        temperature=t_profile,
        specific_humidity=q_profile,
        geopotential=grav * z,
        normalized_surface_pressure=jnp.asarray(1.0),
        tracers={'qc': jnp.zeros(nlev), 'qi': jnp.zeros(nlev)},
    )


class TestSCMConstruction(unittest.TestCase):
    """Cheap tests for the SCM's coord-stub bookkeeping."""

    def test_init_builds_one_one_grid_at_lat_lon(self):
        scm = SingleColumnModel(
            physics=held_suarez_physics(),
            vertical=SigmaCoordinates.equidistant(8),
            lat_deg=30.0,
            lon_deg=180.0,
        )
        self.assertEqual(scm.coords.horizontal.nodal_shape, (1, 1))
        self.assertEqual(scm.coords.nodal_shape, (8, 1, 1))
        self.assertAlmostEqual(
            float(scm.coords.horizontal.latitudes[0]),
            float(jnp.deg2rad(30.0)),
        )
        self.assertAlmostEqual(
            float(scm.coords.horizontal.longitudes[0]),
            float(jnp.deg2rad(180.0)),
        )

    def test_init_defaults_to_single_column_terrain_and_forcing(self):
        scm = SingleColumnModel(
            physics=held_suarez_physics(),
            vertical=SigmaCoordinates.equidistant(8),
        )
        self.assertEqual(scm.terrain.orog.shape, (1, 1))
        self.assertEqual(scm.forcing.sea_surface_temperature.shape, (1, 1))


class TestSCMHeldSuarez(unittest.TestCase):
    """Held-Suarez SCM run on a small column."""

    def setUp(self):
        self.column_state = _make_column_state(nlev=8)
        self.scm = SingleColumnModel(
            physics=held_suarez_physics(),
            vertical=SigmaCoordinates.equidistant(8),
            lat_deg=0.0,
            lon_deg=0.0,
        )

    def test_run_smoke(self):
        states = [self.column_state, self.column_state, self.column_state]
        predictions = self.scm.run(states)
        self.assertIsInstance(predictions, SCMPredictions)
        # Tendencies should be 1-D in level with a leading time axis.
        self.assertEqual(predictions.tendencies.temperature.shape, (3, 8))
        self.assertIn('qc', predictions.tracer_states)
        self.assertEqual(predictions.tracer_states['qc'].shape, (3, 8))

    def test_disable_tracer_update(self):
        scm = SingleColumnModel(
            physics=held_suarez_physics(),
            vertical=SigmaCoordinates.equidistant(8),
            apply_tracer_tendencies=False,
        )
        states = [self.column_state, self.column_state]
        predictions = scm.run(states)
        self.assertEqual(predictions.tendencies.temperature.shape, (2, 8))


class TestSCMEcham(unittest.TestCase):
    """ECHAM-grey SCM run — exercises tracer evolution."""

    def test_echam_run_smoke(self):
        column_state = _make_column_state(nlev=8)
        scm = SingleColumnModel(
            physics=echam_physics(radiation_scheme='grey'),
            vertical=SigmaCoordinates.equidistant(8),
            lat_deg=0.0,
            lon_deg=0.0,
        )
        predictions = scm.run([column_state, column_state])
        self.assertEqual(predictions.tendencies.temperature.shape, (2, 8))
        self.assertIn('qc', predictions.tracer_states)
        self.assertIn('qi', predictions.tracer_states)

    def test_radiation_step_counter_starts_at_zero(self):
        """Regression: SCM bootstrap must not advance the radiation carry.

        After step 0 the radiation term increments its own ``step`` slot
        from 0→1. If the bootstrap (`initial_physics_data=None` path)
        seeds the carry from a live ``compute_tendencies`` result
        instead of a zero template, step 0 starts at 1 and the
        sub-stepping cadence skews by one — caught by the codex review
        on PR #476.
        """
        column_state = _make_column_state(nlev=8)
        scm = SingleColumnModel(
            physics=echam_physics(radiation_scheme='grey'),
            vertical=SigmaCoordinates.equidistant(8),
            lat_deg=0.0,
            lon_deg=0.0,
        )
        # Three scan steps with the default ``initial_physics_data=None``
        # path. After step N the carry's ``step`` field reads N+1.
        predictions = scm.run([column_state] * 3)
        rad_steps = predictions.physics_data['radiation'].step
        self.assertEqual(tuple(int(s) for s in rad_steps), (1, 2, 3))


class TestSCMHelpers(unittest.TestCase):
    """The SCM-oriented helpers in ``jcm.utils``."""

    def test_create_single_column_state_is_one_dimensional(self):
        nlev = 8
        T = jnp.linspace(280, 220, nlev)
        q = jnp.full((nlev,), 0.005)
        state = create_single_column_state(T, q)
        self.assertEqual(state.temperature.shape, (nlev,))
        self.assertEqual(state.normalized_surface_pressure.shape, ())

    def test_create_initial_tracers(self):
        tracers = create_initial_tracers(4, cloud_water=1e-4)
        self.assertEqual(set(tracers), {'qc', 'qi'})
        self.assertEqual(tracers['qc'].shape, (4,))
        self.assertAlmostEqual(float(tracers['qc'][0]), 1e-4)


def _simple_column(nlev: int, temperature) -> PhysicsState:
    """Build a minimal 1-D column with a given temperature profile (zeros elsewhere)."""
    return PhysicsState(
        u_wind=jnp.zeros(nlev),
        v_wind=jnp.zeros(nlev),
        temperature=jnp.asarray(temperature, dtype=jnp.float32),
        specific_humidity=jnp.zeros(nlev),
        geopotential=jnp.zeros(nlev),
        normalized_surface_pressure=jnp.asarray(1.0),
        tracers={},
    )


class TestSCMFreeEvolveAndClosure(unittest.TestCase):
    """The ``free_evolve`` and ``state_closure`` hooks (issue #523 RCE support)."""

    def test_free_evolve_overlap_raises(self):
        with self.assertRaises(ValueError):
            SingleColumnModel(
                physics=_IdentityTempPhysics(),
                vertical=SigmaCoordinates.equidistant(4),
                relaxation_timescales={'temperature': 100.0},
                free_evolve=('temperature',),
            )

    def test_free_evolve_integrates_physics_tendency(self):
        """With ``dT/dt = T`` and ``dt = 1 s``, T doubles every step (no nudging)."""
        nlev = 4
        T0 = jnp.array([280.0, 260.0, 240.0, 220.0])
        column = _simple_column(nlev, T0)
        scm = SingleColumnModel(
            physics=_IdentityTempPhysics(),
            vertical=SigmaCoordinates.equidistant(nlev),
            dt_seconds=1.0,
            free_evolve=('temperature',),
        )
        preds = scm.run([column, column, column])
        T_hist = preds.relaxed_states['temperature']
        self.assertEqual(T_hist.shape, (3, nlev))
        for k in range(3):
            # Post-step value after k+1 doublings.
            self.assertTrue(
                jnp.allclose(T_hist[k], T0 * 2.0 ** (k + 1), rtol=1e-5),
            )
        # The tendency recorded at step k is the temperature physics *saw*.
        for k in range(3):
            self.assertTrue(
                jnp.allclose(
                    preds.tendencies.temperature[k], T0 * 2.0 ** k, rtol=1e-5,
                ),
            )

    def test_free_evolved_humidity_is_floored_at_zero(self):
        """A freely evolving specific_humidity never carries a negative value."""
        nlev = 4
        column = _simple_column(nlev, jnp.full(nlev, 280.0)).copy(
            specific_humidity=jnp.full(nlev, 0.5),
        )
        scm = SingleColumnModel(
            physics=_DryingPhysics(),
            vertical=SigmaCoordinates.equidistant(nlev),
            dt_seconds=1.0,
            free_evolve=("specific_humidity",),
        )
        preds = scm.run([column, column, column])
        q_hist = preds.relaxed_states["specific_humidity"]
        # dq/dt = -1, dt = 1 would drive 0.5 -> -0.5 on step 0; the clamp holds
        # it at 0 and keeps it there.
        self.assertTrue(bool(jnp.all(q_hist >= 0.0)))
        self.assertTrue(jnp.allclose(q_hist, 0.0))

    def test_state_closure_overwrites_state_before_physics(self):
        """A closure pinning T to a constant makes physics see that constant."""
        nlev = 4
        T0 = jnp.array([280.0, 260.0, 240.0, 220.0])
        C = 300.0
        column = _simple_column(nlev, T0)

        def pin_temperature(state, forcing):
            return state.copy(
                temperature=jnp.full_like(state.temperature, C),
            )

        scm = SingleColumnModel(
            physics=_IdentityTempPhysics(),
            vertical=SigmaCoordinates.equidistant(nlev),
            dt_seconds=1.0,
            free_evolve=('temperature',),
            state_closure=pin_temperature,
        )
        preds = scm.run([column, column, column])
        # Physics always sees the pinned constant, so every recorded tendency is C.
        for k in range(3):
            self.assertTrue(
                jnp.allclose(preds.tendencies.temperature[k], C, rtol=1e-6),
            )
        # Free-evolving T accumulates dt·C each step from the IC.
        T_hist = preds.relaxed_states['temperature']
        for k in range(3):
            self.assertTrue(
                jnp.allclose(T_hist[k], T0 + (k + 1) * C, rtol=1e-5),
            )


# Slow-marked companions — see jcm/runners_test.py for rationale.

@pytest.mark.slow
class TestSCMHeldSuarezSlow(TestSCMHeldSuarez):
    pass


@pytest.mark.slow
class TestSCMEchamSlow(TestSCMEcham):
    pass


class FreeEvolveTracersTest(unittest.TestCase):
    """``free_evolve`` accepts tracers, so a column can prescribe its cloud.

    A prescribed-state column has no ascent, so a freely evolving ``qc`` rains
    out within hours and never re-forms — which leaves any cloud-mediated
    aerosol sink untested. Holding the condensate fixed while the aerosol runs
    is the configuration that exercises one.
    """

    def _column(self, nlev=4):
        from dinosaur.sigma_coordinates import SigmaCoordinates

        from jcm.physics_interface import PhysicsState
        vertical = SigmaCoordinates.equidistant(nlev)
        state = PhysicsState(
            u_wind=jnp.zeros(nlev), v_wind=jnp.zeros(nlev),
            temperature=jnp.full(nlev, 280.0),
            specific_humidity=jnp.full(nlev, 1e-3),
            geopotential=jnp.zeros(nlev),
            normalized_surface_pressure=jnp.asarray(1.0),
        )
        return vertical, state

    def test_named_tracers_evolve_while_the_rest_are_held(self):
        vertical, state = self._column()
        scm = SingleColumnModel(
            physics=_ConstantTracerTendencyPhysics(rate=1e-6),
            vertical=vertical, dt_seconds=900.0,
            apply_tracer_tendencies=False, free_evolve=("free_one",),
        )
        states = tree_map(lambda x: jnp.broadcast_to(x, (4,) + jnp.shape(x)), state)
        seed = {"free_one": jnp.zeros(4), "held_one": jnp.zeros(4)}
        out = scm.run(states, initial_tracers=seed,
                      times=jnp.arange(4) * 900.0 / 86400.0)
        free = np.asarray(out.tracer_states["free_one"])
        held = np.asarray(out.tracer_states["held_one"])
        self.assertGreater(float(free[-1].max()), 0.0)
        np.testing.assert_array_equal(held, np.zeros_like(held))

    def test_unknown_free_evolve_name_raises(self):
        vertical, state = self._column()
        scm = SingleColumnModel(
            physics=_ConstantTracerTendencyPhysics(rate=1e-6),
            vertical=vertical, free_evolve=("not_a_tracer",),
        )
        states = tree_map(lambda x: jnp.broadcast_to(x, (2,) + jnp.shape(x)), state)
        with self.assertRaisesRegex(ValueError, "not_a_tracer"):
            scm.run(states, initial_tracers={"free_one": jnp.zeros(4)},
                    times=jnp.arange(2) * 0.01)


class TestSelectColumn(unittest.TestCase):
    """Nearest-neighbour column extraction for the SCM warm start."""

    def _synthetic(self):
        import xarray as xr

        nt, nlev, nlon, nlat = 3, 4, 5, 6
        lat = np.linspace(-75.0, 75.0, nlat)
        lon = np.linspace(0.0, 300.0, nlon)
        # Distinct per-(lon, lat) signature so the picked column is identifiable.
        col = np.arange(nlon)[:, None] * 100.0 + np.arange(nlat)[None, :]
        temperature = np.broadcast_to(
            col[None, None], (nt, nlev, nlon, nlat)).astype(float)
        surface = np.broadcast_to(col[None], (nt, nlon, nlat)).astype(float)
        ds = xr.Dataset(coords={"lat": lat, "lon": lon})
        states = {
            "temperature": temperature,   # (time, level, lon, lat)
            "surface": surface,           # (time, lon, lat)
        }
        return states, ds, lat, lon

    def test_nearest_neighbour_pick_and_slicing(self):
        from jcm.single_column_model import select_column

        states, ds, lat, lon = self._synthetic()
        # Request a point closest to lon index 2, lat index 4.
        lat_req = float(lat[4]) + 3.0
        lon_req = float(lon[2]) - 4.0
        column, (i_lon, i_lat, actual_lat, actual_lon) = select_column(
            states, ds, lat_req, lon_req)

        self.assertEqual((i_lon, i_lat), (2, 4))
        self.assertEqual(actual_lat, float(lat[4]))
        self.assertEqual(actual_lon, float(lon[2]))
        # 4D column variable collapses to (time, level); 3D surface to (time,).
        self.assertEqual(column["temperature"].shape, (3, 4))
        self.assertEqual(column["surface"].shape, (3,))
        expected = 2 * 100.0 + 4
        np.testing.assert_array_equal(column["temperature"], expected)
        np.testing.assert_array_equal(column["surface"], expected)
