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

    def data_struct_to_dict(self, struct, nodal_shape=None, sep="."):
        # Output naming belongs to the physics; borrow the shipped packages'.
        return ComposablePhysics(terms=[]).data_struct_to_dict(
            struct, nodal_shape, sep)


class _Dycore:
    dt_seconds = 21600.0

    def to_xarray(self, predictions, times, physics=None):
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


class PhysicsOutputFieldsTest(unittest.TestCase):
    """The one flattening every output path names physics diagnostics with."""

    def test_nested_diagnostics_flatten_and_carry_plumbing_is_dropped(self):
        from jcm.predictions import physics_output_fields

        physics = ComposablePhysics(terms=[], vectorize_columns=True)
        fields = physics_output_fields(
            {"_prev_step": {"q_tendency": jnp.zeros((2, 3, 6))},
             "water_positivity_correction": {
                 "total_water_tendency": jnp.zeros((2, 3, 6))},
             "_convection": {"precnv": jnp.zeros((2, 6))}},
            physics, (3, 2, 3))
        self.assertEqual(sorted(fields), [
            "convection.precnv",
            "water_positivity_correction.total_water_tendency"])
        # Column-vectorized fields get their (lon, lat) axes back.
        self.assertEqual(
            fields["water_positivity_correction.total_water_tendency"].shape,
            (2, 3, 2, 3))
        self.assertEqual(fields["convection.precnv"].shape, (2, 2, 3))

    def test_a_length_one_horizontal_axis_is_not_unflattened_twice(self):
        """The pySES physics layout ``(1, ncol)`` is already on its grid.

        Its column count equals its second horizontal axis, so a field that
        is already ``(..., 1, ncol)`` must not have a second ``(1, ncol)``
        pair inserted, while a flattened ``(..., ncol)`` one still gets its
        pair back.
        """
        from jcm.predictions import physics_output_fields

        physics = ComposablePhysics(terms=[], vectorize_columns=True)
        nodal_shape = (3, 1, 6)            # (nlev, 1, ncol)
        fields = physics_output_fields(
            {"surface": {"on_grid": jnp.zeros((2, 1, 6)),
                         "profile": jnp.zeros((2, 3, 1, 6)),
                         "flattened": jnp.zeros((2, 3, 6))}},
            physics, nodal_shape)
        self.assertEqual(fields["surface.on_grid"].shape, (2, 1, 6))
        self.assertEqual(fields["surface.profile"].shape, (2, 3, 1, 6))
        self.assertEqual(fields["surface.flattened"].shape, (2, 3, 1, 6))

    def test_host_arrays_are_named_like_device_arrays(self):
        """A trajectory fetched with ``jax.device_get`` keeps every field."""
        from jcm.predictions import physics_output_fields

        physics = ComposablePhysics(terms=[], vectorize_columns=True)
        diagnostics = {
            "water_positivity_correction": {
                "total_water_tendency": jnp.ones((2, 3, 6))},
            "_convection": {"precnv": jnp.ones((2, 6))},
            "layer_thickness": jnp.ones((2, 3, 6)),
        }
        on_device = physics_output_fields(diagnostics, physics, (3, 2, 3))
        on_host = physics_output_fields(
            jax.device_get(diagnostics), physics, (3, 2, 3))
        self.assertEqual(sorted(on_host), sorted(on_device))
        for name, value in on_device.items():
            self.assertEqual(on_host[name].shape, value.shape, name)

    def test_diagnostics_without_a_physics_to_name_them_are_refused(self):
        from jcm.predictions import physics_output_fields

        self.assertEqual(physics_output_fields({}, None, (3, 2, 3)), {})
        self.assertEqual(physics_output_fields(None, None, (3, 2, 3)), {})
        with self.assertRaisesRegex(TypeError, "no physics package"):
            physics_output_fields({"a": jnp.zeros((2, 2, 3))}, None,
                                  (3, 2, 3))


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

    def _regridding_dycore(self):
        """Build a pySES-shaped backend that regrids every field to float64."""
        from jcm.predictions import physics_output_fields

        class _Regridding:
            dt_seconds = 21600.0

            def to_xarray(self, predictions, times, physics=None):
                data = {"temperature": (
                    ("time", "level", "x", "y"),
                    np.asarray(predictions.dynamics.temperature))}
                # Named as every backend names them, then regridded.
                fields = physics_output_fields(
                    predictions.physics, physics,
                    predictions.dynamics.temperature.shape[1:])
                for name, leaf in fields.items():
                    data[name] = (("time", "x", "y"),
                                  np.asarray(leaf).astype(np.float64))
                ds = xr.Dataset(data, coords={"time": np.asarray(times)})
                from jcm import cf_metadata
                cf_metadata.apply_cf_attributes(ds)
                return ds

        return _Regridding()

    def _with_interval(self, predictions, mean):
        start = jdt.Datetime.from_isoformat("2000-01-01T00:00:00")
        end = jdt.Datetime.from_isoformat("2000-01-01T06:00:00")
        bounds = jax.tree.map(
            lambda left, right: jnp.stack([left[None], right[None]], axis=1),
            start, end,
        )
        return predictions.replace(
            times=jax.tree.map(lambda value: value[None], end),
            time_bounds=bounds,
            time_cell_method=jnp.asarray(mean),
        )

    def test_regridded_integer_diagnostic_is_omitted_from_means(self):
        """An integer ktype stays categorical after a float64 regrid."""
        physics = {"convection": {
            "ktype": jnp.ones((1, 2, 3), dtype=jnp.int32),
            "precip": jnp.ones((1, 2, 3)),
        }}
        dycore = self._regridding_dycore()
        mean = ModelPredictions(
            self._with_interval(_predictions().replace(physics=physics), True),
            self.coords, self.physics, dycore=dycore).to_xarray()
        self.assertNotIn("convection.ktype", mean)
        self.assertEqual(mean.attrs["omitted_interval_mean_variables"],
                         "convection.ktype")
        self.assertEqual(mean["convection.precip"].attrs["cell_methods"],
                         "time: mean")
        snapshot = ModelPredictions(
            self._with_interval(_predictions().replace(physics=physics), False),
            self.coords, self.physics, dycore=dycore).to_xarray()
        self.assertIn("convection.ktype", snapshot)

    def test_mean_and_snapshot_time_axes_carry_the_same_cf_attrs(self):
        dycore = self._regridding_dycore()
        mean = ModelPredictions(
            self._with_interval(_predictions(), True),
            self.coords, self.physics, dycore=dycore).to_xarray()
        snapshot = ModelPredictions(
            self._with_interval(_predictions(), False),
            self.coords, self.physics, dycore=dycore).to_xarray()
        self.assertIn("standard_name", snapshot.time.attrs)
        strip = lambda attrs: {k: v for k, v in attrs.items()
                               if k not in ("bounds", "cell_methods")}
        self.assertEqual(strip(mean.time.attrs), strip(snapshot.time.attrs))
        self.assertEqual(mean.time.attrs["bounds"], "time_bounds")

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
            def to_xarray(self, predictions, times, physics=None):
                ds = super().to_xarray(predictions, times, physics)
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



def _chunk(start_hour, mean):
    """One two-frame trajectory starting ``start_hour`` into 2000-01-01."""
    import datetime

    base = datetime.datetime(2000, 1, 1)

    def at(hours):
        return jdt.Datetime.from_pydatetime(
            base + datetime.timedelta(hours=hours))

    edges = [at(start_hour + h) for h in (0, 1, 2)]
    stack = lambda values: jax.tree.map(lambda *v: jnp.stack(v), *values)
    return _predictions().replace(
        dynamics=jax.tree.map(lambda x: jnp.concatenate([x, x]),
                              _predictions().dynamics),
        times=stack(edges[1:]),
        time_bounds=jax.tree.map(lambda lo, hi: jnp.stack([lo, hi], axis=1),
                                 stack(edges[:2]), stack(edges[1:])),
        time_cell_method=jnp.asarray(mean),
    )


class StackedTimeCellMethodTest(unittest.TestCase):
    """A stacked trajectory's per-trajectory flag has one public reading (#907).

    A coupler that runs one ``run_from_state_with_carry`` per coupling step
    inside its own ``lax.scan`` gets every leaf stacked on a leading axis,
    ``time_cell_method`` included, so the flag arrives with one entry per
    chunk. ``is_interval_mean`` collapses it when the entries agree and
    refuses a stack that mixes means with samples, and ``time_labels`` /
    ``to_xarray`` read the flag only through it.
    """

    def _wrap(self, predictions):
        return ModelPredictions(predictions, _Coords(), _Physics(),
                                dycore=_Dycore())

    def _tree_stack(self, *chunks):
        return jax.tree.map(lambda *xs: jnp.stack(xs), *chunks)

    def test_tree_map_stack_that_agrees_labels_every_chunk(self):
        for mean in (True, False):
            with self.subTest(mean=mean):
                stacked = self._wrap(self._tree_stack(
                    _chunk(0, mean), _chunk(2, mean), _chunk(4, mean)))
                self.assertEqual(stacked._predictions.time_cell_method.shape,
                                 (3,))
                self.assertIs(stacked.is_interval_mean(), mean)
                labels = stacked.time_labels()
                self.assertEqual(labels.shape, (3, 2))
                # Means are labelled at the interval midpoints, samples at
                # their own times; both keep the stacking axis.
                first = "2000-01-01T00:30" if mean else "2000-01-01T01:00"
                self.assertEqual(labels[0, 0], np.datetime64(first, "ms"))
                self.assertEqual(labels[2, 1] - labels[0, 1],
                                 np.timedelta64(4, "h"))

    def test_tree_map_stack_that_disagrees_is_refused(self):
        stacked = self._wrap(self._tree_stack(
            _chunk(0, True), _chunk(2, False), _chunk(4, True)))
        for read in (stacked.is_interval_mean, stacked.time_labels):
            with self.subTest(read=read.__name__):
                with self.assertRaisesRegex(
                        ValueError, "2 of the 3 stacked time_cell_method"):
                    read()

    def test_lax_scan_stack(self):
        def scanned(flags):
            def body(carry, mean):
                chunk = _chunk(0, False).replace(time_cell_method=mean)
                return carry, chunk
            _, stacked = jax.lax.scan(body, None, jnp.asarray(flags))
            return self._wrap(stacked)

        self.assertTrue(scanned([True, True]).is_interval_mean())
        self.assertFalse(scanned([False, False]).is_interval_mean())
        with self.assertRaisesRegex(ValueError, "1 of the 2"):
            scanned([True, False]).is_interval_mean()

    def test_edge_cases(self):
        # One trajectory (a scalar flag), and a stack of one.
        self.assertTrue(self._wrap(_chunk(0, True)).is_interval_mean())
        self.assertTrue(self._wrap(
            self._tree_stack(_chunk(0, True))).is_interval_mean())
        # No flag at all: labelled as instantaneous samples.
        self.assertFalse(self._wrap(_predictions()).is_interval_mean())
        # A stack of zero trajectories has no answer.
        empty = _chunk(0, True).replace(
            time_cell_method=jnp.zeros((0,), dtype=bool))
        with self.assertRaisesRegex(ValueError, "no entries"):
            self._wrap(empty).is_interval_mean()

    def test_an_unmerged_stack_is_refused_with_the_fix(self):
        stacked = self._wrap(self._tree_stack(_chunk(0, True), _chunk(2, True)))
        with self.assertRaisesRegex(ValueError, "merge its stacking axes"):
            stacked.to_xarray()

    def test_a_merged_stack_serializes_as_one_trajectory(self):
        """The coupler's last step: merge the chunk axis, attach context."""
        stacked = self._tree_stack(_chunk(0, True), _chunk(2, True))
        merged = jax.tree.map(
            lambda x: x.reshape((-1,) + x.shape[2:]) if x.ndim >= 2 else x,
            ModelPredictions(stacked, None, None))
        # The flag keeps one entry per chunk after the merge.
        self.assertEqual(merged._predictions.time_cell_method.shape, (2,))
        ds = merged.with_context(_Coords(), _Physics(),
                                 dycore=_Dycore()).to_xarray()
        np.testing.assert_array_equal(
            ds.time.values,
            np.array(["2000-01-01T00:30", "2000-01-01T01:30",
                      "2000-01-01T02:30", "2000-01-01T03:30"],
                     dtype="datetime64[ms]"))
        self.assertEqual(ds.temperature.attrs["cell_methods"], "time: mean")
        self.assertEqual(ds.time_bounds.shape, (4, 2))


class CoupledScanOverChunksTest(unittest.TestCase):
    """The coupled caller's real shape: ``run_from_state_with_carry`` in a scan."""

    def test_scan_over_chunks_labels_and_serializes(self):
        from jcm.forcing import default_forcing
        from jcm.model import Model
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        from jcm.physics.held_suarez.utils import get_held_suarez_coords
        from jcm.terrain import TerrainData

        coords = get_held_suarez_coords(layers=8, spectral_truncation=21)
        model = Model(coords=coords, terrain=TerrainData.aquaplanet(coords),
                      time_step=30, physics=held_suarez_physics())
        forcing = default_forcing(coords.horizontal)

        def coupling_step(run_state, _):
            dynamics, physics, time, step = run_state
            new, predictions = model.run_from_state_with_carry(
                initial_state=dynamics, initial_physics_state=physics,
                initial_time=time, initial_step=step, forcing=forcing,
                save_interval="1 hour", total_time="2 hours",
                output_averages=True)
            return (new.dynamics, new.physics, new.time, new.step), predictions

        start = (model.initial_state(), model.initial_physics_carry(),
                 model.start_time, jnp.int32(0))
        _, stacked = jax.lax.scan(coupling_step, start, None, length=3)

        self.assertEqual(stacked._predictions.time_cell_method.shape, (3,))
        self.assertTrue(stacked.is_interval_mean())
        labels = stacked.time_labels()
        self.assertEqual(labels.shape, (3, 2))
        self.assertEqual(labels[0, 0],
                         np.datetime64("2000-01-01T00:30", "ms"))
        self.assertEqual(labels[2, 1],
                         np.datetime64("2000-01-01T05:30", "ms"))

        merged = jax.tree.map(
            lambda x: x.reshape((-1,) + x.shape[2:]) if x.ndim >= 2 else x,
            stacked)
        ds = merged.with_context(model).to_xarray()
        np.testing.assert_array_equal(ds.time.values, labels.reshape(-1))
        self.assertEqual(ds.temperature.attrs["cell_methods"], "time: mean")


if __name__ == "__main__":
    unittest.main()
