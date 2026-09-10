"""Tests for ``jcm.prescribed_state_model.PrescribedStateModel``."""

import unittest

import jax.numpy as jnp
import pytest

from jcm.constants import grav
from jcm.physics_interface import PhysicsState
from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
from jcm.physics.held_suarez.utils import get_held_suarez_coords
from jcm.prescribed_state_model import (
    PrescribedStateModel,
    PrescribedStatePredictions,
)


def _make_test_state(coords) -> PhysicsState:
    nlev = coords.nodal_shape[0]
    nlon, nlat = coords.horizontal.nodal_shape
    shape = (nlev, nlon, nlat)
    z = jnp.linspace(0, 30000, nlev)[::-1]
    t_profile = jnp.maximum(288.0 - 6.5e-3 * z, 200.0)
    q_profile = 0.012 * jnp.exp(-z / 3000.0)
    return PhysicsState(
        u_wind=jnp.full(shape, 5.0),
        v_wind=jnp.zeros(shape),
        temperature=jnp.broadcast_to(t_profile[:, None, None], shape),
        specific_humidity=jnp.broadcast_to(q_profile[:, None, None], shape),
        geopotential=jnp.broadcast_to((grav * z)[:, None, None], shape),
        normalized_surface_pressure=jnp.ones((nlon, nlat)),
        tracers={},
    )


class TestPrescribedStateModel(unittest.TestCase):
    def setUp(self):
        self.coords = get_held_suarez_coords(layers=8, spectral_truncation=21)
        self.state = _make_test_state(self.coords)

    def test_run_smoke(self):
        model = PrescribedStateModel(
            physics=held_suarez_physics(), coords=self.coords,
        )
        states = [self.state] * 3
        predictions = model.run(states)
        self.assertIsInstance(predictions, PrescribedStatePredictions)
        self.assertEqual(predictions.tendencies.temperature.shape[0], 3)
        self.assertEqual(predictions.times.shape[0], 3)

    def test_run_accepts_stacked_state_and_explicit_times(self):
        # A single PhysicsState whose leading axis is time must behave the
        # same as the list-of-states form, and user-provided ``times`` must
        # be passed through untouched.
        from jax.tree_util import tree_map

        model = PrescribedStateModel(
            physics=held_suarez_physics(), coords=self.coords,
            dt_seconds=900.0,
        )
        stacked = tree_map(lambda a: jnp.stack([a, a], axis=0), self.state)
        times = jnp.array([0.0, 0.5])  # days
        predictions = model.run(stacked, times=times)
        self.assertEqual(predictions.tendencies.temperature.shape[0], 2)
        self.assertTrue(jnp.allclose(predictions.times, times))
        # Identical prescribed states must yield identical tendencies —
        # there is no carry between steps in prescribed mode.
        t_tend = predictions.tendencies.temperature
        self.assertTrue(jnp.allclose(t_tend[0], t_tend[1]))

    def test_default_times_use_dt_seconds(self):
        model = PrescribedStateModel(
            physics=held_suarez_physics(), coords=self.coords,
            dt_seconds=43200.0,  # half a day per step
        )
        predictions = model.run([self.state] * 3)
        self.assertTrue(
            jnp.allclose(predictions.times, jnp.array([0.0, 0.5, 1.0]))
        )

    def test_to_xarray_layout_and_diagnostics(self):
        model = PrescribedStateModel(
            physics=held_suarez_physics(), coords=self.coords,
        )
        predictions = model.run([self.state] * 2)
        ds = predictions.to_xarray()

        nlev = self.coords.nodal_shape[0]
        nlon, nlat = self.coords.horizontal.nodal_shape
        # Column state variables use the prescribed-mode vmap layout.
        self.assertEqual(
            ds["state.temperature"].dims, ("time", "level", "lon", "lat")
        )
        self.assertEqual(
            ds["state.temperature"].shape, (2, nlev, nlon, nlat)
        )
        # Surface variables drop the level axis.
        self.assertEqual(
            ds["state.normalized_surface_pressure"].dims,
            ("time", "lon", "lat"),
        )
        # Tendencies serialised alongside the states.
        self.assertIn("tend.temperature", ds)
        # The prescribed states round-trip bit-exact into the dataset.
        import numpy as np
        np.testing.assert_array_equal(
            ds["state.u_wind"].values[0],
            np.asarray(self.state.u_wind),
        )
        self.assertEqual(ds.sizes["time"], 2)

    def test_to_xarray_physics_data_dict_handling(self):
        # Private ("_"-prefixed) diagnostics are dropped; struct-valued
        # entries expand via asdict; plain arrays serialise directly.
        import numpy as np
        from jax.tree_util import tree_map

        nlev = self.coords.nodal_shape[0]
        base_state = _make_test_state(self.coords)
        # Give the struct a tracer so the nested-dict expansion runs too.
        base_state = PhysicsState(
            **{**base_state.asdict(),
               "tracers": {"qc": jnp.zeros_like(base_state.temperature)}},
        )
        stacked = tree_map(lambda a: jnp.stack([a, a], axis=0), base_state)
        preds = PrescribedStatePredictions(
            states=stacked,
            tendencies=stacked,
            physics_data={
                "_private": jnp.zeros((2,)),
                "scalar_series": jnp.arange(2.0),
                # Struct-valued diagnostics expand field-by-field.
                "blob": stacked,
                # (time, level) columns and rank>4 fall back gracefully.
                "column_series": jnp.zeros((2, nlev)),
                "rank5": jnp.zeros((2, 1, 1, 1, 1)),
            },
            times=jnp.array([0.0, 1.0]),
        )
        ds = preds.to_xarray()
        self.assertNotIn("diag._private", ds)
        self.assertIn("diag.scalar_series", ds)
        np.testing.assert_allclose(
            ds["diag.scalar_series"].values, [0.0, 1.0]
        )
        self.assertIn("diag.blob.temperature", ds)
        self.assertEqual(
            ds["diag.blob.temperature"].dims,
            ("time", "level", "lon", "lat"),
        )
        # Nested tracer dict flattens with a dotted name.
        self.assertIn("diag.blob.tracers.qc", ds)
        self.assertEqual(
            ds["diag.column_series"].dims, ("time", "level"),
        )
        self.assertEqual(
            ds["diag.rank5"].dims,
            ("dim_0", "dim_1", "dim_2", "dim_3", "dim_4"),
        )

    def test_to_xarray_level_is_surface_first_sigma(self):
        # ``run`` threads the vertical table through, so ``to_xarray`` writes a
        # real descending sigma ``level`` coordinate with CF metadata (#739).
        import numpy as np

        model = PrescribedStateModel(
            physics=held_suarez_physics(), coords=self.coords,
        )
        predictions = model.run([self.state] * 2)
        ds = predictions.to_xarray()

        level = ds["level"].values
        # Surface-first file convention: sigma descends from ~1 to ~0.
        self.assertTrue(np.all(np.diff(level) < 0))
        self.assertEqual(ds["level"].attrs.get("positive"), "down")

    def test_to_xarray_round_trips_through_load_states(self):
        # A file written by ``to_xarray`` (surface-first) must come back
        # top-first via ``load_states_from_xarray`` (#739 + #741), recovering
        # the original per-level physics-frame values exactly.
        import numpy as np

        from jcm.utils import load_states_from_xarray

        model = PrescribedStateModel(
            physics=held_suarez_physics(), coords=self.coords,
        )
        predictions = model.run([self.state] * 2)
        ds = predictions.to_xarray()

        reloaded = load_states_from_xarray(
            ds,
            u_wind_var="state.u_wind",
            v_wind_var="state.v_wind",
            temperature_var="state.temperature",
            specific_humidity_var="state.specific_humidity",
            geopotential_var="state.geopotential",
            surface_pressure_var="state.normalized_surface_pressure",
        )
        # ``self.state`` is a single (nlev, ...) column; the predictions stack
        # it over 2 times, so compare each recovered time slice to it.
        np.testing.assert_allclose(
            np.asarray(reloaded.temperature)[0],
            np.asarray(self.state.temperature),
        )
        np.testing.assert_allclose(
            np.asarray(reloaded.temperature)[1],
            np.asarray(self.state.temperature),
        )


# Slow-marked companion — see jcm/runners_test.py for rationale.

@pytest.mark.slow
class TestPrescribedStateModelSlow(TestPrescribedStateModel):
    pass
