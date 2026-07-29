"""Tests for ``jcm.single_column_model.SingleColumnModel``."""

import unittest

import jax
import jax_datetime as jdt
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from dinosaur.sigma_coordinates import SigmaCoordinates

from jcm.constants import grav
from jcm.date import DateData, absolute_seconds_since_epoch
from jcm.forcing import BY_DATE, ForcingData, SolarGeometry, make_time_series
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


class _ForcingCapturePhysics(Physics):
    """Minimal physics package that exposes the forcing received by the SCM."""

    def get_empty_data(self, coords):
        return {
            "land_temperature": jnp.zeros(coords.horizontal.nodal_shape),
            "tyear": jnp.zeros(()),
        }

    def compute_tendencies(self, state, forcing, terrain, prev_physics_data=None):
        del terrain, prev_physics_data
        return PhysicsTendency.zeros(state.temperature.shape), {
            "land_temperature": forcing.stl_am,
            "tyear": forcing.solar.tyear,
        }


class _ConstantTendencyPhysics(Physics):
    """Minimal physics package for unnudged prognostic-state tests."""

    def get_empty_data(self, coords):
        del coords
        return {}

    def compute_tendencies(self, state, forcing, terrain, prev_physics_data=None):
        del forcing, terrain, prev_physics_data
        return PhysicsTendency.zeros(
            state.temperature.shape,
            temperature=jnp.ones_like(state.temperature),
        ), {}


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


class TestSCMForcing(unittest.TestCase):
    """Forcing selection must happen inside the SCM's jitted scan."""

    def test_run_selects_date_aligned_forcing_and_solar_geometry(self):
        timestamps = [
            jdt.to_datetime("2018-06-01"),
            jdt.to_datetime("2018-06-02"),
        ]
        forcing = ForcingData.zeros(
            (1, 1),
            stl_am=make_time_series(
                jnp.array([[[290.0]], [[300.0]]]),
                jnp.asarray([
                    absolute_seconds_since_epoch(timestamp)
                    for timestamp in timestamps
                ]),
                align_mode=BY_DATE,
            ),
        )
        scm = SingleColumnModel(
            physics=_ForcingCapturePhysics(),
            vertical=SigmaCoordinates.equidistant(8),
            forcing=forcing,
            dt_seconds=86400.0,
            start_date=timestamps[0],
            calendar="gregorian",
        )

        predictions = scm.run([_make_column_state(8)] * 2)

        npt.assert_allclose(
            predictions.physics_data["land_temperature"][:, 0, 0],
            [290.0, 300.0],
        )
        expected_tyear = [
            DateData.set_date(timestamp).tyear("gregorian")
            for timestamp in timestamps
        ]
        npt.assert_allclose(predictions.physics_data["tyear"], expected_tyear)

    def test_free_evolving_variable_advances_without_nudging(self):
        state = _make_column_state(8)
        scm = SingleColumnModel(
            physics=_ConstantTendencyPhysics(),
            vertical=SigmaCoordinates.equidistant(8),
            dt_seconds=2.0,
            free_evolve=("temperature",),
        )

        predictions = scm.run([state] * 3)

        expected = state.temperature + jnp.array([2.0, 4.0, 6.0])[:, None]
        npt.assert_allclose(predictions.relaxed_states["temperature"], expected)

    def test_run_uses_preselected_forcing_trajectory(self):
        state = _make_column_state(8)
        single_forcing = ForcingData.zeros(
            (1, 1),
            stl_am=jnp.array([[290.0]]),
        )
        forcing_steps = jax.tree_util.tree_map(
            lambda x: jnp.stack([x, x]), single_forcing,
        ).copy(
            stl_am=jnp.array([[[290.0]], [[300.0]]]),
            solar=SolarGeometry(
                tyear=jnp.array([0.1, 0.2]),
                orbital_phase=jnp.array([0.0, 0.0]),
                synodic_phase=jnp.array([0.0, 0.0]),
            ),
        )
        scm = SingleColumnModel(
            physics=_ForcingCapturePhysics(),
            vertical=SigmaCoordinates.equidistant(8),
            free_evolve=("temperature",),
        )

        repeated_state = jax.tree_util.tree_map(
            lambda value: jnp.broadcast_to(value, (2,) + value.shape), state
        )
        predictions = scm.run(repeated_state, forcing_steps=forcing_steps)

        npt.assert_allclose(
            predictions.physics_data["land_temperature"][:, 0, 0],
            [290.0, 300.0],
        )
        npt.assert_allclose(predictions.physics_data["tyear"], [0.1, 0.2])

    def test_run_can_be_vmapped_over_independent_windows(self):
        state = _make_column_state(8)
        single_forcing = ForcingData.zeros((1, 1))
        forcing_steps = jax.tree_util.tree_map(
            lambda x: jnp.stack([x, x]), single_forcing,
        ).copy(
            stl_am=jnp.array([[[290.0]], [[300.0]]]),
            solar=SolarGeometry(
                tyear=jnp.array([0.1, 0.2]),
                orbital_phase=jnp.zeros(2),
                synodic_phase=jnp.zeros(2),
            ),
        )
        batched_states = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), state)
        batched_forcing = jax.tree_util.tree_map(
            lambda x: jnp.stack([x, x]), forcing_steps,
        )
        scm = SingleColumnModel(
            physics=_ForcingCapturePhysics(),
            vertical=SigmaCoordinates.equidistant(8),
        )

        predictions = jax.vmap(
            lambda initial, forcing: scm.run(
                jax.tree_util.tree_map(
                    lambda value: jnp.broadcast_to(value, (2,) + value.shape), initial
                ),
                forcing_steps=forcing,
            ),
        )(batched_states, batched_forcing)

        npt.assert_allclose(
            predictions.physics_data["land_temperature"][:, :, 0, 0],
            [[290.0, 300.0], [290.0, 300.0]],
        )


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
