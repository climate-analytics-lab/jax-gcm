"""Tests for rebuilding ``ModelPredictions`` host-side context."""

import json
from types import SimpleNamespace
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from jcm.dycore.base import Predictions
from jcm.physics_interface import PhysicsState
from jcm.predictions import ModelPredictions


class _Horizontal:
    nodal_shape = (2, 3)
    nodal_axes = (np.array([0.0, np.pi]), np.array([-1.0, 0.0, 1.0]))


class _Coords:
    horizontal = _Horizontal()
    vertical = SimpleNamespace(layers=2)
    nodal_shape = (2, 2, 3)


class _Term(nnx.Module):
    name = "test_term"

    def __init__(self):
        self.strength = nnx.Param(jnp.array(2.0))


class _Physics:
    def __init__(self):
        self.terms = (_Term(),)

    def output_attrs(self):
        return {"temperature": {"units": "K"}}


class _Dycore:
    dt_seconds = 21600.0

    def to_xarray(self, predictions, times):
        return xr.Dataset(
            {
                "temperature": (
                    ("time", "level", "x", "y"),
                    np.asarray(predictions.dynamics.temperature),
                ),
            },
            coords={"time": np.asarray(times)},
        )


class _Observer:
    name = "station"

    def to_dataset(self, samples, t0_days, dt_seconds):
        values = np.asarray(samples["temperature"])
        times = t0_days + np.arange(values.shape[0]) * dt_seconds / 86400.0
        return xr.Dataset(
            {"temperature": (("time", "point"), values)},
            coords={"time": times},
        )


def _predictions():
    shape = (1, 2, 2, 3)
    dynamics = PhysicsState(
        u_wind=jnp.ones(shape),
        v_wind=jnp.ones(shape),
        temperature=jnp.full(shape, 280.0),
        specific_humidity=jnp.full(shape, 0.005),
        geopotential=jnp.zeros(shape),
        normalized_surface_pressure=jnp.ones((1, 2, 3)),
    )
    return Predictions(dynamics=dynamics, physics={}, times=jnp.array([11.0]))


class ModelPredictionsWithContextTest(unittest.TestCase):
    def setUp(self):
        self.coords = _Coords()
        self.physics = _Physics()
        self.dycore = _Dycore()
        self.observer = _Observer()
        self.model = SimpleNamespace(
            coords=self.coords,
            physics=self.physics,
            dycore=self.dycore,
            observers=(self.observer,),
            dt_si=SimpleNamespace(m=self.dycore.dt_seconds),
        )
        self.observations = (
            {"temperature": jnp.arange(4.0).reshape(4, 1)},
        )
        self.snapshots = {
            "surface.temperature": jnp.arange(12.0).reshape(2, 2, 3),
        }
        self.original = ModelPredictions(
            _predictions(),
            self.coords,
            self.physics,
            dycore=self.dycore,
            observations=self.observations,
            observers=(self.observer,),
            obs_t0_days=10.0,
            obs_dt_seconds=self.dycore.dt_seconds,
            snapshots=self.snapshots,
            snapshot_variables=("surface.temperature",),
            snapshot_interval_days=0.5,
        )

    def test_model_form_rebuilds_tree_map_result_and_all_datasets(self):
        rebuilt_without_context = jax.tree.map(lambda value: value, self.original)
        self.assertIsNone(rebuilt_without_context._coords)
        self.assertIsNone(rebuilt_without_context._physics)
        self.assertEqual(rebuilt_without_context.params, {})

        # Snapshot arrays are intentionally outside the pytree and therefore
        # must be supplied alongside their run-specific cadence. Model-owned
        # context uses the common one-argument spelling.
        restored = rebuilt_without_context.with_context(
            self.model,
            snapshots=self.snapshots,
            snapshot_variables=("surface.temperature",),
            snapshot_interval_days=0.5,
        )

        self.assertIs(restored._coords, self.coords)
        self.assertIs(restored._physics, self.physics)
        self.assertIs(restored._dycore, self.dycore)
        self.assertEqual(restored._observers, (self.observer,))
        self.assertEqual(restored._obs_t0_days, 10.0)
        self.assertEqual(restored._obs_dt_seconds, self.dycore.dt_seconds)
        self.assertEqual(restored._snapshot_variables,
                         ("surface.temperature",))
        self.assertEqual(restored._snapshot_interval_days, 0.5)

        # Context stays outside the pytree: restoring it cannot change the
        # leaves or treedef seen by a later JAX transformation.
        before_leaves, before_tree = jax.tree.flatten(rebuilt_without_context)
        after_leaves, after_tree = jax.tree.flatten(restored)
        self.assertEqual(before_tree, after_tree)
        self.assertEqual(len(before_leaves), len(after_leaves))

        trajectory = restored.to_xarray()
        self.assertEqual(trajectory["temperature"].shape, (1, 2, 2, 3))
        self.assertEqual(trajectory["temperature"].attrs["units"], "K")

        observations = restored.observation_datasets()["station"]
        self.assertEqual(observations["temperature"].shape, (4, 1))
        np.testing.assert_allclose(observations["time"].values[0], 10.0)

        snapshots = restored.snapshot_dataset()
        self.assertEqual(snapshots["surface_temperature"].shape, (2, 2, 3))
        np.testing.assert_allclose(snapshots["snap_time"].values, [0.5, 1.0])

        for dataset in (trajectory, observations, snapshots):
            params = json.loads(dataset.attrs["jcm_prov_params"])
            self.assertIn("test_term.strength", params)
            self.assertIn("parameters_rederived_from_live_context", params)

    def test_rederived_params_are_live_and_explicitly_not_traced(self):
        rebuilt = jax.tree.map(lambda value: value, self.original)
        self.physics.terms[0].strength.set_value(jnp.array(7.0))

        restored = rebuilt.with_context(self.model)

        self.assertEqual(restored.params["test_term.strength"], 7.0)
        self.assertIn("not a trace-time record",
                      restored.params["parameters_rederived_from_live_context"])
        self.assertNotIn("live_parameters_differ_from_compiled",
                         restored.params)

    def test_explicit_coords_physics_form_preserves_present_metadata(self):
        restored = self.original.with_context(
            self.coords,
            self.physics,
            dycore=self.dycore,
        )

        self.assertIs(restored.snapshots, self.snapshots)
        self.assertIs(restored.observations, self.observations)
        self.assertEqual(restored._observers, (self.observer,))
        self.assertEqual(restored._obs_t0_days, 10.0)
        self.assertEqual(restored._snapshot_interval_days, 0.5)

    def test_obvious_grid_shape_mismatch_is_rejected(self):
        rebuilt = jax.tree.map(lambda value: value, self.original)
        bad_coords = SimpleNamespace(
            nodal_shape=(2, 4, 3),
            horizontal=SimpleNamespace(nodal_shape=(4, 3)),
        )
        bad_model = SimpleNamespace(
            coords=bad_coords,
            physics=self.physics,
            dycore=self.dycore,
            observers=(self.observer,),
            dt_si=SimpleNamespace(m=self.dycore.dt_seconds),
        )

        with self.assertRaisesRegex(ValueError, "Prediction grid shape"):
            rebuilt.with_context(bad_model)


if __name__ == "__main__":
    unittest.main()
