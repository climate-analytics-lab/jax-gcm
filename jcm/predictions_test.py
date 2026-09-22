"""Tests for rebuilding ``ModelPredictions`` host-side context."""

import json
from types import SimpleNamespace
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import xarray as xr

from jcm.dycore.base import Predictions
from jcm.physics.composable_physics import ComposablePhysics
from jcm.physics.speedy.speedy_coords import get_speedy_coords
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
                "category": ("time", np.ones(len(times), dtype=np.int32)),
            },
            coords={"time": np.asarray(times)},
        )


class _Observer:
    name = "station"

    def to_dataset(self, samples, start_time, dt_seconds):
        values = np.asarray(samples["temperature"])
        start = jax.device_get(start_time).to_datetime64()
        times = start + np.arange(values.shape[0]) * np.timedelta64(
            int(dt_seconds), "s")
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
    times = jax.tree.map(
        lambda value: value[None],
        jdt.Datetime.from_isoformat("2000-01-12T00:00:00"),
    )
    return Predictions(dynamics=dynamics, physics={}, times=times)


class WaterPositivityOutputTest(unittest.TestCase):
    """Water-correction profiles and columns survive public serialization."""

    def test_column_source_shape_and_attrs_survive_to_xarray(self):
        coords = get_speedy_coords(layers=2, spectral_truncation=21)
        physics = ComposablePhysics(terms=[], vectorize_columns=True)
        physics.cache_coords(coords)
        nlev, nlon, nlat = coords.nodal_shape
        ntime = 2
        ncols = nlon * nlat
        state_shape = (ntime, nlev, nlon, nlat)
        dynamics = PhysicsState(
            u_wind=jnp.zeros(state_shape),
            v_wind=jnp.zeros(state_shape),
            temperature=jnp.full(state_shape, 280.0),
            specific_humidity=jnp.full(state_shape, 1.0e-3),
            geopotential=jnp.zeros(state_shape),
            normalized_surface_pressure=jnp.ones((ntime, nlon, nlat)),
            tracers={},
        )
        correction = {
            # Reproduce the scan output of column-vectorized physics before
            # data_struct_to_dict restores the two horizontal dimensions.
            "specific_humidity_tendency": jnp.zeros(
                (ntime, nlev, ncols),
            ),
            "total_water_tendency": jnp.zeros((ntime, nlev, ncols)),
            "column_water_source": jnp.arange(
                ntime * ncols, dtype=jnp.float32,
            ).reshape(ntime, ncols),
        }
        predictions = Predictions(
            dynamics=dynamics,
            physics={"water_positivity_correction": correction},
            times=jax.tree.map(
                lambda *values: jnp.stack(values),
                jdt.Datetime.from_isoformat("2000-01-01T00:00:00"),
                jdt.Datetime.from_isoformat("2000-01-02T00:00:00"),
            ),
        )

        ds = ModelPredictions(predictions, coords, physics).to_xarray()

        profile = ds[
            "water_positivity_correction.specific_humidity_tendency"
        ]
        source = ds["water_positivity_correction.column_water_source"]
        self.assertEqual(profile.dims, ("time", "level", "lon", "lat"))
        self.assertEqual(profile.shape, (ntime, nlev, nlon, nlat))
        self.assertEqual(source.dims, ("time", "lon", "lat"))
        self.assertEqual(source.shape, (ntime, nlon, nlat))
        self.assertEqual(source.attrs["units"], "kg m-2 s-1")
        self.assertEqual(
            source.attrs["long_name"],
            "column artificial water source from positivity correction",
        )
        self.assertEqual(
            source.attrs["description"],
            "column artificial water source from positivity correction",
        )


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
        self.observer_start_time = jdt.Datetime.from_isoformat(
            "2000-01-10T00:00:00")
        self.snapshot_times = jax.tree.map(
            lambda *values: jnp.stack(values),
            jdt.Datetime.from_isoformat("2000-01-10T12:00:00"),
            jdt.Datetime.from_isoformat("2000-01-11T00:00:00"),
        )
        self.original = ModelPredictions(
            _predictions(),
            self.coords,
            self.physics,
            dycore=self.dycore,
            observations=self.observations,
            observers=(self.observer,),
            observer_start_time=self.observer_start_time,
            obs_dt_seconds=self.dycore.dt_seconds,
            snapshots=self.snapshots,
            snapshot_variables=("surface.temperature",),
            snapshot_times=self.snapshot_times,
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
            snapshot_times=self.snapshot_times,
            observer_start_time=self.observer_start_time,
        )

        self.assertIs(restored._coords, self.coords)
        self.assertIs(restored._physics, self.physics)
        self.assertIs(restored._dycore, self.dycore)
        self.assertEqual(restored._observers, (self.observer,))
        self.assertIs(restored._observer_start_time, self.observer_start_time)
        self.assertEqual(restored._obs_dt_seconds, self.dycore.dt_seconds)
        self.assertEqual(restored._snapshot_variables,
                         ("surface.temperature",))
        self.assertIs(restored._snapshot_times, self.snapshot_times)

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
        self.assertEqual(observations["time"].values[0],
                         np.datetime64("2000-01-10T00:00:00"))

        snapshots = restored.snapshot_dataset()
        self.assertEqual(snapshots["surface_temperature"].shape, (2, 2, 3))
        np.testing.assert_array_equal(
            snapshots["snap_time"].values,
            np.array(["2000-01-10T12:00:00", "2000-01-11T00:00:00"],
                     dtype="datetime64[ns]"))

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
        self.assertIs(restored._observer_start_time, self.observer_start_time)
        self.assertIs(restored._snapshot_times, self.snapshot_times)

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

    def test_interval_bounds_control_exact_midpoint_and_metadata(self):
        predictions = _predictions()
        start = jdt.Datetime.from_isoformat("2000-01-01T00:00:00")
        end = jdt.Datetime.from_isoformat("2000-01-01T00:00:01")
        bounds = jax.tree.map(
            lambda left, right: jnp.stack([left[None], right[None]], axis=1),
            start, end,
        )
        predictions = predictions.replace(
            times=jax.tree.map(lambda value: value[None], start),
            time_bounds=bounds,
            time_cell_method=jnp.asarray(True),
        )

        ds = ModelPredictions(
            predictions, self.coords, self.physics, dycore=self.dycore,
        ).to_xarray()

        self.assertEqual(str(ds.time.values[0]), "2000-01-01T00:00:00.500")
        self.assertEqual(ds.time.attrs["bounds"], "time_bounds")
        self.assertEqual(ds.temperature.attrs["cell_methods"], "time: mean")
        self.assertNotIn("category", ds)
        self.assertEqual(ds.attrs["omitted_interval_mean_variables"],
                         "category")
        self.assertEqual(ds.time.encoding["units"],
                         "seconds since 1970-01-01 00:00:00")
        self.assertEqual(ds.time_bounds.encoding, ds.time.encoding)

    def test_raw_predictions_remain_differentiable(self):
        def loss(temperature):
            predictions = _predictions().replace(
                dynamics=_predictions().dynamics.replace(
                    temperature=temperature))
            wrapped = ModelPredictions(
                predictions, self.coords, self.physics, dycore=self.dycore)
            return jnp.sum(wrapped.dynamics.temperature ** 2)

        temperature = jnp.ones((1, 2, 2, 3))
        np.testing.assert_allclose(jax.grad(loss)(temperature), 2.0)

    def test_interval_mean_rejects_conflicting_declared_cell_method(self):
        class ConflictingDycore(_Dycore):
            def to_xarray(self, predictions, times):
                ds = super().to_xarray(predictions, times)
                ds.temperature.attrs["cell_methods"] = (
                    "area: mean time: mean time: maximum")
                return ds

        predictions = _predictions()
        start = jdt.Datetime.from_isoformat("2000-01-01T00:00:00")
        end = jdt.Datetime.from_isoformat("2000-01-02T00:00:00")
        predictions = predictions.replace(
            times=jax.tree.map(lambda value: value[None], start),
            time_bounds=jax.tree.map(
                lambda left, right: jnp.stack([left[None], right[None]], axis=1),
                start, end),
            time_cell_method=jnp.asarray(True),
        )

        with self.assertRaisesRegex(ValueError, "maximum"):
            ModelPredictions(
                predictions, self.coords, self.physics,
                dycore=ConflictingDycore()).to_xarray()


if __name__ == "__main__":
    unittest.main()
