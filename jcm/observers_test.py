"""Tests for the virtual-observation operators (jcm.observers)."""

import unittest

import pytest

import numpy as np
import jax
import jax.numpy as jnp

from jcm.observers import (
    LocalSolarTimeObserver,
    Observer,
    TrackObserver,
    _times_to_days,
)


def _t21_coords():
    from jcm.physics.held_suarez.utils import get_held_suarez_coords
    return get_held_suarez_coords(spectral_truncation=21)


def _grid_lat_lon_deg(coords):
    lat = np.degrees(np.asarray(coords.horizontal.latitudes))
    lon = np.degrees(np.asarray(coords.horizontal.longitudes)) % 360.0
    return lat, lon


def _synthetic_diagnostics(coords, nlev=8, column_layout=False):
    """Diagnostics dict with analytically known fields.

    z_full decreases linearly with level index (top-first physics frame);
    ``linear_in_z`` is an affine function of z so vertical interpolation is
    exact; ``linear_in_latlon`` is affine in (lat, lon) so horizontal
    bilinear interpolation is exact away from the wrap seam.
    """
    lat, lon = _grid_lat_lon_deg(coords)
    nlon, nlat = coords.horizontal.nodal_shape
    lat2d = np.broadcast_to(lat[None, :], (nlon, nlat))
    lon2d = np.broadcast_to(lon[:, None], (nlon, nlat))

    z_levels = np.linspace(16000.0, 500.0, nlev)  # top-first
    z_full = np.broadcast_to(
        z_levels[:, None, None], (nlev,) + (nlon, nlat)).copy()
    p_full = 101325.0 * np.exp(-z_full / 8000.0)
    linear_in_z = 2.0 * z_full + 5.0
    surface_field = 3.0 * lat2d + 0.5 * lon2d

    def _maybe_cols(a):
        if not column_layout:
            return jnp.asarray(a)
        return jnp.asarray(a.reshape(a.shape[:-2] + (nlon * nlat,)))

    return {
        "_sampler_state": {
            "z_full": _maybe_cols(z_full),
            "p_full": _maybe_cols(p_full),
            "temperature": _maybe_cols(linear_in_z),
            "tracers": {"so4": _maybe_cols(0.1 * z_full)},
        },
        "linear_in_latlon": _maybe_cols(surface_field),
    }


class HorizontalWeightsTest(unittest.TestCase):
    def _stations_observer(self, lats, lons, coords, **kwargs):
        obs = TrackObserver.stations(
            lats, lons, variables=("linear_in_latlon",), **kwargs)
        obs.cache_grid(coords)
        return obs

    def test_weights_collapse_to_node_at_grid_point(self):
        coords = _t21_coords()
        lat, lon = _grid_lat_lon_deg(coords)
        nlat = len(lat)
        obs = self._stations_observer([lat[5]], [lon[7]], coords)
        idx, w = obs._horizontal_weights(
            np.array([[lat[5]]]), np.array([[lon[7]]]))
        flat_node = 7 * nlat + 5
        total = 0.0
        for k in range(4):
            if w[0, 0, k] > 0:
                self.assertEqual(idx[0, 0, k], flat_node)
                total += w[0, 0, k]
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_bilinear_exact_for_affine_field(self):
        coords = _t21_coords()
        lat, lon = _grid_lat_lon_deg(coords)
        diag = _synthetic_diagnostics(coords)
        # Interior points, away from the lon wrap seam and the poles.
        pts_lat = [0.5 * (lat[4] + lat[5]), lat[10] + 0.3 * (lat[11] - lat[10])]
        pts_lon = [33.3, 121.7]
        obs = self._stations_observer(pts_lat, pts_lon, coords)
        xs = obs.prepare(t0_days=0.0, dt_seconds=1800.0, n_steps=1)
        xs_t = jax.tree.map(lambda a: a[0], xs)
        samples = obs.sample(diag, xs_t)
        expected = 3.0 * np.array(pts_lat) + 0.5 * np.array(pts_lon)
        np.testing.assert_allclose(
            np.asarray(samples["linear_in_latlon"]), expected, rtol=1e-5)

    def test_longitude_wraparound(self):
        coords = _t21_coords()
        lat, lon = _grid_lat_lon_deg(coords)
        nlat = len(lat)
        nlon = len(lon)
        # A point between the last and first longitude columns.
        target_lon = (lon[-1] + 360.0 / nlon / 2.0) % 360.0
        obs = self._stations_observer([lat[3]], [target_lon], coords)
        idx, w = obs._horizontal_weights(
            np.array([[lat[3]]]), np.array([[target_lon]]))
        cols = set(int(i) // nlat for i, wt in
                   zip(idx[0, 0], w[0, 0]) if wt > 1e-12)
        self.assertEqual(cols, {0, nlon - 1})

    def test_pole_clamps_to_last_ring(self):
        coords = _t21_coords()
        lat, _ = _grid_lat_lon_deg(coords)
        obs = self._stations_observer([89.9], [10.0], coords)
        idx, w = obs._horizontal_weights(
            np.array([[89.9]]), np.array([[10.0]]))
        nlat = len(lat)
        ring = {int(i) % nlat for i, wt in zip(idx[0, 0], w[0, 0]) if wt > 1e-12}
        self.assertEqual(ring, {int(np.argmax(lat))})
        self.assertAlmostEqual(float(w[0, 0].sum()), 1.0, places=12)


class SamplingTest(unittest.TestCase):
    def test_altitude_interpolation_exact_for_linear_field(self):
        coords = _t21_coords()
        diag = _synthetic_diagnostics(coords)
        targets = [3000.0, 8000.0]
        obs = TrackObserver.stations(
            [10.0, -45.0], [30.0, 200.0], altitudes=targets,
            variables=("temperature", "so4"))
        obs.cache_grid(coords)
        xs = obs.prepare(0.0, 1800.0, 1)
        samples = obs.sample(diag, jax.tree.map(lambda a: a[0], xs))
        np.testing.assert_allclose(
            np.asarray(samples["temperature"]),
            2.0 * np.array(targets) + 5.0, rtol=1e-5)
        np.testing.assert_allclose(
            np.asarray(samples["so4"]), 0.1 * np.array(targets), rtol=1e-4)

    def test_pressure_and_profile_modes(self):
        coords = _t21_coords()
        nlev = 8
        diag = _synthetic_diagnostics(coords, nlev=nlev)
        p_target = 101325.0 * np.exp(-8000.0 / 8000.0)  # p at z=8000 m
        obs_p = TrackObserver.stations(
            [0.0], [100.0], variables=("temperature",),
            pressures=[p_target])
        obs_p.cache_grid(coords)
        xs = obs_p.prepare(0.0, 1800.0, 1)
        sample = obs_p.sample(diag, jax.tree.map(lambda a: a[0], xs))
        # T is linear in z, p exponential in z -> log-p interpolation exact.
        np.testing.assert_allclose(
            float(sample["temperature"][0]), 2.0 * 8000.0 + 5.0, rtol=1e-4)

        obs_prof = TrackObserver.stations(
            [0.0], [100.0], variables=("temperature",), vertical="profile")
        obs_prof.cache_grid(coords)
        xs = obs_prof.prepare(0.0, 1800.0, 1)
        prof = obs_prof.sample(diag, jax.tree.map(lambda a: a[0], xs))
        self.assertEqual(prof["temperature"].shape, (nlev, 1))
        z_levels = np.linspace(16000.0, 500.0, nlev)
        np.testing.assert_allclose(
            np.asarray(prof["temperature"][:, 0]), 2.0 * z_levels + 5.0,
            rtol=1e-5)

    def test_column_and_3d_layouts_agree(self):
        coords = _t21_coords()
        diag_3d = _synthetic_diagnostics(coords, column_layout=False)
        diag_cols = _synthetic_diagnostics(coords, column_layout=True)
        obs = TrackObserver.stations(
            [22.0, -60.0], [77.0, 310.0], altitudes=[2000.0, 11000.0],
            variables=("temperature",))
        obs.cache_grid(coords)
        xs_t = jax.tree.map(lambda a: a[0], obs.prepare(0.0, 1800.0, 1))
        s3 = obs.sample(diag_3d, xs_t)
        sc = obs.sample(diag_cols, xs_t)
        np.testing.assert_array_equal(
            np.asarray(s3["temperature"]), np.asarray(sc["temperature"]))

    def test_track_time_window_masking(self):
        coords = _t21_coords()
        diag = _synthetic_diagnostics(coords)
        dt = 1800.0
        # Track valid for steps 2..5 of an 8-step window starting at t=0.
        track_t = np.array([2, 5]) * dt / 86400.0
        obs = TrackObserver(
            track_t, [0.0, 20.0], [100.0, 120.0], altitudes=[3000.0, 3000.0],
            variables=("temperature",))
        obs.cache_grid(coords)
        xs = obs.prepare(0.0, dt, 8)
        vals = np.stack([
            np.asarray(obs.sample(diag, jax.tree.map(lambda a: a[i], xs))
                       ["temperature"])
            for i in range(8)
        ])
        self.assertTrue(np.all(np.isnan(vals[:2])))
        self.assertTrue(np.all(np.isfinite(vals[2:6])))
        self.assertTrue(np.all(np.isnan(vals[6:])))

    def test_missing_variable_raises_with_help(self):
        coords = _t21_coords()
        diag = _synthetic_diagnostics(coords)
        obs = TrackObserver.stations(
            [0.0], [0.0], variables=("no_such_var",))
        obs.cache_grid(coords)
        xs_t = jax.tree.map(lambda a: a[0], obs.prepare(0.0, 1800.0, 1))
        with self.assertRaisesRegex(KeyError, "no_such_var"):
            obs.sample(diag, xs_t)

    def test_sample_is_differentiable(self):
        coords = _t21_coords()
        obs = TrackObserver.stations(
            [15.0], [50.0], altitudes=[4000.0], variables=("temperature",))
        obs.cache_grid(coords)
        xs_t = jax.tree.map(lambda a: a[0], obs.prepare(0.0, 1800.0, 1))
        base = _synthetic_diagnostics(coords)

        def sampled(t_field):
            diag = {
                "_sampler_state": {
                    **base["_sampler_state"], "temperature": t_field,
                },
                "linear_in_latlon": base["linear_in_latlon"],
            }
            return obs.sample(diag, xs_t)["temperature"][0]

        grad = jax.grad(sampled)(base["_sampler_state"]["temperature"])
        grad = np.asarray(grad)
        self.assertTrue(np.all(np.isfinite(grad)))
        # Gradient support: two vertical levels x up-to-4 horizontal corners.
        self.assertGreater(np.count_nonzero(grad), 0)
        self.assertLessEqual(np.count_nonzero(grad), 8)
        # Weights sum to one across the interpolation stencil.
        self.assertAlmostEqual(float(grad.sum()), 1.0, places=5)


class GeometryTest(unittest.TestCase):
    def test_times_to_days_roundtrip(self):
        t = np.array(["2000-01-01T12:00", "2000-01-02T00:00"],
                     dtype="datetime64[ns]")
        days = _times_to_days(t)
        self.assertAlmostEqual(days[1] - days[0], 0.5)
        np.testing.assert_array_equal(_times_to_days([1.0, 2.5]),
                                      [1.0, 2.5])

    def test_track_interpolation_across_dateline(self):
        dt = 1800.0
        track_t = np.array([0.0, 2 * dt / 86400.0])
        obs = TrackObserver(track_t, [0.0, 0.0], [179.0, 181.0],
                            variables=("x",))
        lat, lon, _, valid = obs._positions_for_times(
            np.array([dt / 86400.0]))
        # Short way round: through 180, not through 0.
        self.assertAlmostEqual(float(lon[0, 0]), 180.0, places=6)
        self.assertTrue(bool(valid[0, 0]))

    def test_local_solar_time_longitude(self):
        obs = LocalSolarTimeObserver(
            variables=("x",), latitudes=[-30.0, 0.0, 30.0],
            local_solar_hour=12.0)
        # At 00 UTC local noon is at 180 deg E; at 12 UTC it is at 0 deg.
        lat, lon, _, valid = obs._positions_for_times(np.array([0.0, 0.5]))
        self.assertEqual(lat.shape, (2, 3))
        np.testing.assert_allclose(lon[0], 180.0)
        np.testing.assert_allclose(lon[1], 0.0)
        self.assertTrue(valid.all())

    def test_local_solar_time_defaults_to_grid_latitudes(self):
        coords = _t21_coords()
        obs = LocalSolarTimeObserver(variables=("x",))
        obs.cache_grid(coords)
        self.assertEqual(obs._latitudes.size,
                         coords.horizontal.nodal_shape[1])

    def test_vertical_mode_validation(self):
        with self.assertRaises(ValueError):
            Observer("bad", ("x",), vertical="bogus")
        with self.assertRaises(ValueError):
            TrackObserver([0.0], [0.0], [0.0], variables=())


@pytest.mark.slow
class ModelIntegrationTest(unittest.TestCase):
    """End-to-end: observers threaded through Model.run / resume.

    Slow-marked so the PR-gate coverage run (slow tests only) exercises
    the observers module end-to-end.
    """

    def _model(self, observers, coords=None, physics=None):
        from jcm.model import Model
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        coords = coords or _t21_coords()
        return Model(
            coords=coords,
            physics=physics if physics is not None else held_suarez_physics(),
            time_step=30.0,
            observers=observers,
        ), coords

    def test_station_sample_matches_saved_frames(self):
        # save_interval == dt: frame k is the post-step state after k+1
        # steps, and the observer samples the physics *input* state, so
        # obs[k+1] must equal frame[k] at the station (placed on a node).
        coords = _t21_coords()
        lat, lon = _grid_lat_lon_deg(coords)
        station = TrackObserver.stations(
            [lat[5]], [lon[7]], variables=("temperature",),
            vertical="profile", name="node_station")
        model, coords = self._model([station], coords=coords)
        dt_days = 30.0 / (60.0 * 24.0)
        preds = model.run(save_interval=dt_days, total_time=4 * dt_days)

        obs = preds.observations[0]["temperature"]  # (4, nlev, 1)
        self.assertEqual(obs.shape, (4, coords.vertical.layers, 1))
        frames = preds.dynamics.temperature  # (4, nlev, nlon, nlat)
        for k in range(3):
            np.testing.assert_allclose(
                np.asarray(obs[k + 1, :, 0]),
                np.asarray(frames[k, :, 7, 5]),
                rtol=1e-5,
            )
        self.assertTrue(np.all(np.isfinite(np.asarray(obs))))

    def test_run_plus_resume_matches_single_run(self):
        coords = _t21_coords()
        lat, lon = _grid_lat_lon_deg(coords)
        station = TrackObserver.stations(
            [lat[3], lat[20]], [lon[2], lon[40]],
            variables=("temperature", "surface_pressure"),
            vertical="profile")
        dt_days = 30.0 / (60.0 * 24.0)

        model_a, _ = self._model([station], coords=coords)
        preds_a = model_a.run(save_interval=2 * dt_days,
                              total_time=4 * dt_days)
        single = jax.device_get(preds_a.observations[0])

        model_b, _ = self._model([station], coords=coords)
        model_b.bootstrap_state(None)
        p1 = model_b.resume(save_interval=2 * dt_days, total_time=2 * dt_days)
        p2 = model_b.resume(save_interval=2 * dt_days, total_time=2 * dt_days)
        chunk1 = jax.device_get(p1.observations[0])
        chunk2 = jax.device_get(p2.observations[0])

        for key in single:
            joined = np.concatenate([chunk1[key], chunk2[key]], axis=0)
            np.testing.assert_allclose(joined, single[key], rtol=1e-6,
                                       err_msg=key)

    def test_vectorized_and_3d_physics_layouts_agree(self):
        from jcm.physics.composable_physics import ComposablePhysics
        coords = _t21_coords()
        lat, lon = _grid_lat_lon_deg(coords)
        station = TrackObserver.stations(
            [lat[8]], [lon[12]], altitudes=[5000.0],
            variables=("temperature",))
        dt_days = 30.0 / (60.0 * 24.0)

        results = {}
        for vectorize in (False, True):
            physics = ComposablePhysics(terms=[],
                                        vectorize_columns=vectorize)
            model, _ = self._model([station], coords=coords,
                                   physics=physics)
            preds = model.run(save_interval=2 * dt_days,
                              total_time=2 * dt_days)
            results[vectorize] = np.asarray(
                jax.device_get(preds.observations[0]["temperature"]))
        np.testing.assert_allclose(results[False], results[True],
                                   rtol=1e-6)

    def test_tracer_carrying_physics_scan_structure(self):
        # Regression (codex P1 on #566): the scan-carry template from
        # ``get_empty_data`` probed with an EMPTY state.tracers dict, while
        # real steps publish ``_sampler_state["tracers"]`` with the declared
        # tracer keys — a lax.scan pytree-structure mismatch for any
        # tracer-carrying physics (ECHAM clouds, JAM). The probe now seeds
        # the declared tracers; tracers must also be sampleable by name.
        from jcm.physics.composable_physics import ComposablePhysics
        from jcm.physics.physics_term import PhysicsTerm, TracerSpec
        from jcm.physics_interface import PhysicsTendency

        class _DeclaresTracer(PhysicsTerm):
            name = "declares_tracer"
            category = "test"

            @classmethod
            def required_tracers(cls):
                return (TracerSpec("qc", initial_value=1e-6),)

            def __call__(self, state, diagnostics, forcing, terrain):
                return (PhysicsTendency.zeros(state.temperature.shape),
                        diagnostics)

        coords = _t21_coords()
        lat, lon = _grid_lat_lon_deg(coords)
        station = TrackObserver.stations(
            [lat[6]], [lon[9]], variables=("qc", "temperature"),
            vertical="profile")
        physics = ComposablePhysics(terms=[_DeclaresTracer()],
                                    vectorize_columns=True)
        model, _ = self._model([station], coords=coords, physics=physics)
        dt_days = 30.0 / (60.0 * 24.0)
        preds = model.run(save_interval=2 * dt_days, total_time=2 * dt_days)
        qc = np.asarray(jax.device_get(preds.observations[0]["qc"]))
        self.assertEqual(qc.shape, (2, coords.vertical.layers, 1))
        # The run completing at all is the regression (the scan used to
        # abort on the carry-structure mismatch); the samples must be finite
        # (tracer resolved by name, not NaN-masked). Value fidelity of
        # tracer sampling is covered by the synthetic so4 test above.
        self.assertTrue(np.all(np.isfinite(qc)))

    def test_sampler_state_not_in_saved_output(self):
        coords = _t21_coords()
        lat, lon = _grid_lat_lon_deg(coords)
        station = TrackObserver.stations(
            [lat[5]], [lon[7]], variables=("temperature",),
            vertical="profile")
        model, _ = self._model([station], coords=coords)
        dt_days = 30.0 / (60.0 * 24.0)
        preds = model.run(save_interval=2 * dt_days, total_time=2 * dt_days)
        self.assertNotIn("_sampler_state", preds.physics)
        ds = preds.to_xarray()
        self.assertFalse(any("sampler" in v for v in ds.data_vars))

    def test_observation_datasets(self):
        coords = _t21_coords()
        lat, lon = _grid_lat_lon_deg(coords)
        swath = LocalSolarTimeObserver(
            variables=("surface_pressure",), latitudes=[-45.0, 0.0, 45.0],
            local_solar_hour=13.5, vertical="surface", name="a_train")
        station = TrackObserver.stations(
            [lat[5]], [lon[7]], variables=("temperature",),
            vertical="profile", name="st")
        model, _ = self._model([station, swath], coords=coords)
        dt_days = 30.0 / (60.0 * 24.0)
        preds = model.run(save_interval=2 * dt_days, total_time=4 * dt_days)

        datasets = preds.observation_datasets()
        self.assertEqual(set(datasets), {"st", "a_train"})
        st = datasets["st"]
        self.assertEqual(st["temperature"].dims, ("time", "level", "point"))
        self.assertEqual(st.sizes["time"], 4)
        # Per-dt cadence: 30-minute spacing on the datetime axis.
        deltas = np.diff(st["time"].values).astype("timedelta64[s]")
        np.testing.assert_array_equal(
            deltas, np.full(3, np.timedelta64(1800, "s")))
        at = datasets["a_train"]
        self.assertEqual(at["surface_pressure"].dims, ("time", "point"))
        self.assertEqual(at.sizes["point"], 3)
        # The local-noon band sweeps 7.5 deg westward per 30-minute step.
        lons = at["longitude"].values
        dlon = (lons[0] - lons[1]) % 360.0
        np.testing.assert_allclose(dlon, np.full(3, 7.5), atol=1e-6)

    def test_profile_output_matches_the_file_vertical_convention(self):
        """A curtain must pair with the trajectory file without a flip (#710).

        Sampling runs top-first (the physics frame); ``to_dataset`` is where
        that becomes the surface-first file convention, so the observer's
        ``level`` axis carries the same sigma coordinate as ``to_xarray``'s.
        """
        coords = _t21_coords()
        lat, lon = _grid_lat_lon_deg(coords)
        station = TrackObserver.stations(
            [lat[5]], [lon[7]], variables=("temperature",),
            vertical="profile", name="st")
        model, _ = self._model([station], coords=coords)
        dt_days = 30.0 / (60.0 * 24.0)
        preds = model.run(save_interval=2 * dt_days, total_time=2 * dt_days)

        st = preds.observation_datasets()["st"]
        np.testing.assert_allclose(
            st["level"].values, preds.to_xarray()["level"].values, rtol=1e-6)
        self.assertEqual(st["level"].attrs["positive"], "down")
        # Sampling is top-first, the dataset surface-first: the same column,
        # reversed.
        raw = np.asarray(jax.device_get(preds.observations[0]["temperature"]))
        np.testing.assert_allclose(
            st["temperature"].values, raw[:, ::-1], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
