"""Tests for ``jcm/nudging.py``.

The previous spectral-space implementation has been promoted to a
:class:`PhysicsTerm` whose reference target rides on :class:`ForcingData`
(sliced per step by the Model). These tests cover the new gridpoint
plumbing.
"""

import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from jcm.forcing import ForcingData
from jcm.model import Model
from jcm.nudging import (
    LatTop, NudgingConfig, NudgingTarget, NudgingTerm, reference_pressure_pa,
    inv_tau_profile, nudging_tendency, window_forcing, with_nudging,
)
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.physics_interface import PhysicsState
from jcm.terrain import TerrainData


def _zero_winds_target_dataset(nlev, nlon, nlat, T_K=250.0):
    """Synthetic 'observation' dataset on the model grid (no time axis).

    Zero winds, isothermal T. Used as a deterministic relaxation target
    to pin the sign of the tendency.
    """
    return xr.Dataset({
        'u':  (('lev', 'lon', 'lat'), np.zeros((nlev, nlon, nlat), dtype=np.float32)),
        'v':  (('lev', 'lon', 'lat'), np.zeros((nlev, nlon, nlat), dtype=np.float32)),
        'T':  (('lev', 'lon', 'lat'), np.full((nlev, nlon, nlat), T_K, dtype=np.float32)),
    })


class TestNudgingConfig(unittest.TestCase):

    def setUp(self):
        coords = get_speedy_coords()
        self.nlev = coords.vertical.layers

    def test_winds_only_zeros_temperature(self):
        cfg = NudgingConfig.winds_only(self.nlev)
        self.assertTrue(jnp.all(cfg.inv_tau_temperature == 0.0))

    def test_winds_only_pbl_mask(self):
        # ``pbl_levels=2`` zeros the bottom two levels of the wind tau.
        cfg = NudgingConfig.winds_only(self.nlev, pbl_levels=2)
        self.assertTrue(jnp.all(cfg.inv_tau_wind[:-2] > 0.0))
        self.assertTrue(jnp.all(cfg.inv_tau_wind[-2:] == 0.0))

    def test_winds_only_zeros_humidity(self):
        cfg = NudgingConfig.winds_only(self.nlev)
        self.assertTrue(jnp.all(cfg.inv_tau_humidity == 0.0))

    def test_temp_humidity_zeros_winds(self):
        # The offline-warmup config nudges T and q, leaves winds free.
        cfg = NudgingConfig.temp_humidity(self.nlev, tau_seconds=86400.0)
        self.assertTrue(jnp.all(cfg.inv_tau_wind == 0.0))
        self.assertTrue(jnp.all(cfg.inv_tau_temperature > 0.0))
        self.assertTrue(jnp.all(cfg.inv_tau_humidity > 0.0))


class TestInvTauProfile(unittest.TestCase):

    def test_sigma_vertical_default_tau(self):
        vertical = get_speedy_coords().vertical
        p_ref = np.asarray(vertical.centers) * 101325.0
        inv_tau = inv_tau_profile(vertical)
        self.assertEqual(inv_tau.shape, (vertical.layers,))
        # Levels above the 60 hPa default cap are zeroed (the top SPEEDY
        # level sits at ~25 hPa); the rest carry the default 6 h tau.
        self.assertTrue(np.all(inv_tau[p_ref < 6000.0] == 0.0))
        np.testing.assert_allclose(
            inv_tau[p_ref >= 6000.0], 1.0 / (6.0 * 3600.0))

    def test_hybrid_vertical_masks_stratosphere(self):
        from jcm.physics.echam.echam_levels import get_echam_levels
        vertical = get_echam_levels(47)
        p_ref = (np.asarray(vertical.a_centers)
                 + np.asarray(vertical.b_centers) * 101325.0)
        inv_tau = inv_tau_profile(vertical, min_pressure_hpa=60.0)
        # Levels above 60 hPa are zeroed; those below carry the tau value.
        self.assertTrue(np.all(inv_tau[p_ref < 6000.0] == 0.0))
        np.testing.assert_allclose(
            inv_tau[p_ref >= 6000.0], 1.0 / (6.0 * 3600.0))
        # The stratosphere is genuinely present in an L47 grid.
        self.assertTrue(np.any(inv_tau == 0.0))

    def test_min_pressure_mask_zeroes_top(self):
        vertical = get_speedy_coords().vertical
        p_ref = np.asarray(vertical.centers) * 101325.0
        # Raise the cap above the highest level so everything is masked.
        cap_hpa = float(p_ref.max()) / 100.0 + 1.0
        inv_tau = inv_tau_profile(vertical, min_pressure_hpa=cap_hpa)
        np.testing.assert_allclose(inv_tau, 0.0)

    def test_pbl_levels_mask_bottom(self):
        vertical = get_speedy_coords().vertical
        # Disable the min-pressure mask so the pbl mask is isolated.
        inv_tau = inv_tau_profile(vertical, min_pressure_hpa=0.0, pbl_levels=2)
        self.assertTrue(np.all(inv_tau[-2:] == 0.0))
        self.assertTrue(np.all(inv_tau[:-2] > 0.0))

    def test_custom_tau_hours(self):
        vertical = get_speedy_coords().vertical
        inv_tau = inv_tau_profile(vertical, tau_hours=12.0)
        np.testing.assert_allclose(
            inv_tau[inv_tau > 0.0], 1.0 / (12.0 * 3600.0))


class TestNudgingTendencyDirection(unittest.TestCase):
    """The relaxation tendency should drive the state toward the target."""

    def test_tendency_points_toward_target(self):
        coords = get_speedy_coords()
        nlev = coords.vertical.layers
        nlon, nlat = coords.horizontal.nodal_shape
        shape = (nlev, nlon, nlat)

        ds = _zero_winds_target_dataset(nlev, nlon, nlat, T_K=250.0)
        # A real humidity reference below the state's 0.01: this case is about
        # the direction of the tendency, not about a missing ``q``.
        ds["q"] = (("lev", "lon", "lat"),
                   np.full((nlev, nlon, nlat), 5e-3, dtype=np.float32))
        target = NudgingTarget.from_dataset(ds, time_var=None)
        config = NudgingConfig(
            inv_tau_wind=jnp.ones(nlev),
            inv_tau_temperature=jnp.ones(nlev),
            inv_tau_humidity=jnp.ones(nlev),
        )
        state = PhysicsState(
            u_wind=jnp.full(shape, 5.0),
            v_wind=jnp.full(shape, -3.0),
            temperature=jnp.full(shape, 280.0),
            specific_humidity=jnp.full(shape, 0.01),
            geopotential=jnp.zeros(shape),
            normalized_surface_pressure=jnp.ones((nlon, nlat)),
            tracers={},
        )
        tend = nudging_tendency(state, target, config)
        self.assertTrue(jnp.all(tend.u_wind <= 0.0))
        self.assertTrue(jnp.all(tend.v_wind >= 0.0))         # state v < target v
        self.assertTrue(jnp.all(tend.temperature <= 0.0))    # state T > target T
        self.assertTrue(jnp.all(tend.specific_humidity <= 0.0))  # state q > target q


class TestNudgingTendencyBroadcasting(unittest.TestCase):
    """The tendency is broadcasting-native: column block agrees with the grid."""

    def test_column_block_matches_grid(self):
        nlev, nlon, nlat = 4, 6, 3
        ncols = nlon * nlat
        key = jax.random.split(jax.random.key(0), 4)
        config = NudgingConfig(
            inv_tau_wind=jnp.linspace(0.5, 1.5, nlev),
            inv_tau_temperature=jnp.linspace(1.0, 2.0, nlev),
            inv_tau_humidity=jnp.linspace(0.1, 0.4, nlev),
        )

        def make(shape, horiz):
            return PhysicsState(
                u_wind=jax.random.normal(key[0], shape),
                v_wind=jax.random.normal(key[1], shape),
                temperature=250.0 + jax.random.normal(key[2], shape),
                specific_humidity=1.0 + jax.random.uniform(key[3], shape),
                geopotential=jnp.zeros(shape),
                normalized_surface_pressure=jnp.ones(horiz),
                tracers={},
            )

        grid = make((nlev, nlon, nlat), (nlon, nlat))
        block = jax.tree_util.tree_map(
            lambda a: a.reshape((nlev, ncols) if a.ndim == 3 else (ncols,)),
            grid)
        target_grid = jax.tree_util.tree_map(jnp.zeros_like, grid)
        target_block = jax.tree_util.tree_map(jnp.zeros_like, block)
        t_grid = nudging_tendency(
            grid, NudgingTarget(target_grid.u_wind, target_grid.v_wind,
                                target_grid.temperature,
                                target_grid.specific_humidity), config)
        t_block = nudging_tendency(
            block, NudgingTarget(target_block.u_wind, target_block.v_wind,
                                 target_block.temperature,
                                 target_block.specific_humidity), config)
        self.assertEqual(t_block.temperature.shape, (nlev, ncols))
        np.testing.assert_allclose(
            np.asarray(t_grid.temperature).reshape(nlev, ncols),
            np.asarray(t_block.temperature), rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(t_grid.specific_humidity).reshape(nlev, ncols),
            np.asarray(t_block.specific_humidity), rtol=1e-6)


class TestNudgingTargetHumidity(unittest.TestCase):
    """``from_dataset`` carries specific humidity when present, ``None`` when not."""

    def test_humidity_absent_stays_none(self):
        """A missing ``q`` must not become a zero-humidity reference.

        ``None`` is the documented "no humidity reference" sentinel, and the
        tendency reads it as "leave humidity alone". Filling zeros instead
        turns a wind/temperature target into a reference for a bone-dry
        atmosphere.
        """
        nlev, nlon, nlat = 4, 8, 6
        ds = _zero_winds_target_dataset(nlev, nlon, nlat)
        target = NudgingTarget.from_dataset(ds, time_var=None)
        self.assertIsNone(target.specific_humidity)

    def test_humidity_absent_does_not_dry_the_atmosphere(self):
        """A q-less target under ``temp_humidity`` leaves humidity untouched."""
        nlev, nlon, nlat = 4, 8, 6
        shape = (nlev, nlon, nlat)
        ds = _zero_winds_target_dataset(nlev, nlon, nlat, T_K=250.0)
        target = NudgingTarget.from_dataset(ds, time_var=None)
        state = PhysicsState(
            u_wind=jnp.zeros(shape), v_wind=jnp.zeros(shape),
            temperature=jnp.full(shape, 280.0),
            specific_humidity=jnp.full(shape, 8e-3),  # kg/kg, a moist column
            geopotential=jnp.zeros(shape),
            normalized_surface_pressure=jnp.ones((nlon, nlat)),
            tracers={},
        )
        tend = nudging_tendency(
            state, target, NudgingConfig.temp_humidity(nlev=nlev))
        # Temperature is still relaxed; humidity is left entirely alone.
        self.assertTrue(jnp.all(tend.temperature < 0.0))
        self.assertTrue(jnp.all(tend.specific_humidity == 0.0))

    def test_humidity_absent_time_varying_stays_none(self):
        """The ``TimeSeries`` path preserves the sentinel too."""
        nlev, nlon, nlat, nt = 4, 8, 6, 2
        zeros = np.zeros((nt, nlev, nlon, nlat), dtype=np.float32)
        dims = ("time", "lev", "lon", "lat")
        ds = xr.Dataset(
            {"u": (dims, zeros), "v": (dims, zeros.copy()),
             "T": (dims, np.full_like(zeros, 250.0))},
            coords={"time": np.array(["2000-01-01", "2000-01-02"],
                                     dtype="datetime64[ns]")},
        )
        target = NudgingTarget.from_dataset(ds)
        self.assertIsNone(target.specific_humidity)

    def test_humidity_present_is_loaded(self):
        nlev, nlon, nlat = 4, 8, 6
        q = np.full((nlev, nlon, nlat), 5e-3, dtype=np.float32)
        ds = _zero_winds_target_dataset(nlev, nlon, nlat)
        ds["q"] = (("lev", "lon", "lat"), q)
        target = NudgingTarget.from_dataset(ds, time_var=None)
        np.testing.assert_allclose(np.asarray(target.specific_humidity), q)


class TestNudgingTermInPhysicsStack(unittest.TestCase):
    """``NudgingTerm`` runs cleanly inside a ``ComposablePhysics`` term list.

    The user adds the term to physics and attaches the target to forcing;
    the Model slices the target per step via ``forcing.select(date, ...)``.
    """

    def test_aquaplanet_winds_shrink_with_nudging(self):
        coords = get_speedy_coords()
        terrain = TerrainData.aquaplanet(coords)
        nlev = coords.vertical.layers
        nlon, nlat = coords.horizontal.nodal_shape

        ds = _zero_winds_target_dataset(nlev, nlon, nlat)
        target = NudgingTarget.from_dataset(ds, time_var=None)
        config = NudgingConfig.winds_only(nlev=nlev, tau_seconds=86400.0)

        forcing = ForcingData.zeros(coords.horizontal.nodal_shape)
        nudging_forcing = forcing.replace(nudging_target=target)

        nudged_physics = with_nudging(speedy_physics(), config)
        preds_nudged = Model(
            coords=coords, terrain=terrain, physics=nudged_physics,
        ).run(forcing=nudging_forcing, save_interval=1, total_time=2)

        preds_free = Model(
            coords=coords, terrain=terrain, physics=speedy_physics(),
        ).run(forcing=forcing, save_interval=1, total_time=2)

        u_n = float(jnp.mean(jnp.abs(preds_nudged.dynamics.u_wind[-1])))
        u_f = float(jnp.mean(jnp.abs(preds_free.dynamics.u_wind[-1])))
        self.assertLess(u_n, u_f,
                        msg=f"nudging didn't shrink winds: |u_n|={u_n} vs |u_f|={u_f}")


class TestNudgingTermInertWithoutTarget(unittest.TestCase):
    """A ``NudgingTerm`` whose forcing carries no target emits zero tendency."""

    def test_default_forcing_makes_term_a_noop(self):
        coords = get_speedy_coords()
        terrain = TerrainData.aquaplanet(coords)
        nlev = coords.vertical.layers
        config = NudgingConfig.winds_only(nlev=nlev, tau_seconds=86400.0)

        # Forcing has nudging_target=None by default — the term should
        # produce no change relative to a baseline run.
        nudged = Model(
            coords=coords, terrain=terrain,
            physics=with_nudging(speedy_physics(), config),
        ).run(save_interval=1, total_time=1)
        plain = Model(
            coords=coords, terrain=terrain, physics=speedy_physics(),
        ).run(save_interval=1, total_time=1)

        self.assertTrue(jnp.allclose(
            nudged.dynamics.u_wind, plain.dynamics.u_wind, atol=1e-6,
        ))


class TestModelRunsNormallyWithoutNudging(unittest.TestCase):
    """A Model without any ``NudgingTerm`` runs as before."""

    def test_default_model_runs_normally(self):
        coords = get_speedy_coords()
        terrain = TerrainData.aquaplanet(coords)
        preds = Model(
            coords=coords, terrain=terrain, physics=speedy_physics(),
        ).run(save_interval=1, total_time=1)
        self.assertEqual(preds.dynamics.u_wind.shape[0], 1)
        self.assertTrue(jnp.all(jnp.isfinite(preds.dynamics.u_wind)))


if __name__ == "__main__":
    unittest.main()


class TestNudgingColumnVectorized(unittest.TestCase):
    """Nudging must work under ``ComposablePhysics(vectorize_columns=True)``.

    The term used to hard-code ``inv_tau[:, None, None]`` and to subtract the
    nodal-shaped target straight from the state, so a column-vectorised host —
    which reshapes the state to ``(nlev, nlon*nlat)`` before calling terms —
    died with "Incompatible shapes for broadcasting: [(47, 192, 96),
    (47, 18432)]". That is what killed the first nudged cluster run of #583.
    """

    NLEV, NLON, NLAT = 4, 8, 6

    def _config(self):
        # Built directly rather than via ``winds_only``, which zeroes the
        # temperature profile — the layout bug must be exercised on BOTH the
        # wind and temperature relaxations.
        inv = 1.0 / (6 * 3600.0)
        return NudgingConfig(
            inv_tau_wind=inv * jnp.ones(self.NLEV),
            inv_tau_temperature=0.5 * inv * jnp.ones(self.NLEV),
        )

    def _target_and_states(self):
        nlev, nlon, nlat = self.NLEV, self.NLON, self.NLAT
        rng = np.random.default_rng(0)
        grid = lambda: jnp.asarray(  # noqa: E731
            rng.normal(size=(nlev, nlon, nlat)))

        target = NudgingTarget(
            u_wind=grid(), v_wind=grid(), temperature=grid() + 280.0)
        state3d = PhysicsState.zeros(
            (nlev, nlon, nlat),
            u_wind=grid(), v_wind=grid(), temperature=grid() + 280.0,
        )
        # Same physical state, flattened exactly as ComposablePhysics does.
        flat = lambda a: a.reshape(nlev, nlon * nlat)  # noqa: E731
        state_col = PhysicsState.zeros(
            (nlev, nlon * nlat),
            u_wind=flat(state3d.u_wind), v_wind=flat(state3d.v_wind),
            temperature=flat(state3d.temperature),
        )
        return target, state3d, state_col

    def test_column_layout_matches_grid_layout(self):
        target, state3d, state_col = self._target_and_states()
        cfg = self._config()

        t3d = nudging_tendency(state3d, target, cfg)
        tcol = nudging_tendency(state_col, target, cfg,
                                nodal_shape=(self.NLON, self.NLAT))

        for name in ("u_wind", "v_wind", "temperature"):
            with self.subTest(field=name):
                a = np.asarray(getattr(t3d, name)).reshape(
                    self.NLEV, self.NLON * self.NLAT)
                b = np.asarray(getattr(tcol, name))
                self.assertEqual(b.shape, (self.NLEV, self.NLON * self.NLAT))
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-12)

    def test_transposed_grid_is_rejected_not_silently_reshaped(self):
        """Equal element count is NOT sufficient grounds to reshape.

        A target stored (nlev, nlat, nlon) has the same size as the state's
        (nlev, nlon, nlat) but needs a transpose. Reshaping it row-major
        would nudge every column toward the wrong reference values and
        corrupt the experiment silently, which is worse than crashing.
        """
        from jcm.nudging import nudging_tendency

        _, _, state_col = self._target_and_states()
        cfg = self._config()
        rng = np.random.default_rng(1)
        # (nlev, nlat, nlon) — axes swapped, same number of elements.
        swapped = lambda: jnp.asarray(  # noqa: E731
            rng.normal(size=(self.NLEV, self.NLAT, self.NLON)))
        target = NudgingTarget(
            u_wind=swapped(), v_wind=swapped(), temperature=swapped())
        # Same size as the column state, so the old size-only check accepted it.
        self.assertEqual(target.u_wind.size, state_col.u_wind.size)
        with self.assertRaisesRegex(ValueError, "looks TRANSPOSED"):
            nudging_tendency(state_col, target, cfg,
                             nodal_shape=(self.NLON, self.NLAT))

    def test_genuinely_mismatched_grid_is_rejected(self):
        target, _, state_col = self._target_and_states()
        cfg = self._config()
        wrong = NudgingTarget(
            u_wind=target.u_wind[:, :, :-1],
            v_wind=target.v_wind[:, :, :-1],
            temperature=target.temperature[:, :, :-1],
        )
        with self.assertRaisesRegex(ValueError, "incompatible with the state"):
            nudging_tendency(state_col, wrong, cfg,
                             nodal_shape=(self.NLON, self.NLAT))


class WindowTargetTest(unittest.TestCase):
    """A chunk's window selects exactly what the run's whole target would."""

    def _target(self, align, persist="strict"):
        import jax_datetime as jdt

        from jcm.forcing import make_time_series
        from jcm.nudging import NudgingTarget
        times = jdt.to_datetime(np.arange(
            np.datetime64("2010-01-01T00"), np.datetime64("2010-01-11T00"),
            np.timedelta64(6, "h")).astype("datetime64[s]"))
        vals = np.arange(times.size, dtype=np.float32)[:, None, None, None]
        vals = vals * np.ones((1, 2, 3, 4), np.float32)
        ts = make_time_series(vals, times, align_mode=align, persist=persist)
        return NudgingTarget(u_wind=ts, v_wind=ts, temperature=ts)

    def test_window_selects_like_the_whole_series(self):
        import jax_datetime as jdt

        from jcm.date import DateData
        from jcm.forcing import BY_DATE, BY_DATE_INTERP, _select_time_series
        from jcm.nudging import samples_per_span, window_target
        chunk = 86400
        for align in (BY_DATE, BY_DATE_INTERP):
            full = self._target(align)
            n = samples_per_span(full, chunk)
            self.assertEqual(n, 6)
            # Chunk starts on, between and near the end of the samples.
            for start in ("2010-01-03T00", "2010-01-04T07:30", "2010-01-09T23"):
                t0 = np.datetime64(start).astype("datetime64[s]")
                start_s = int(t0.astype(np.int64))
                win = window_target(full, start_s, n)
                self.assertEqual(win.u_wind.values.shape[0], n)
                for dt_h in (0, 2.5, 6, 13, 23.9):
                    when = t0 + np.timedelta64(int(dt_h * 3600), "s")
                    if when >= np.datetime64("2010-01-10T18"):
                        continue
                    date = DateData.set_date(jdt.to_datetime(when))
                    np.testing.assert_array_equal(
                        np.asarray(_select_time_series(win.u_wind, date)),
                        np.asarray(_select_time_series(full.u_wind, date)),
                        err_msg=f"align={align} start={start} +{dt_h}h")

    def test_window_at_the_series_end_holds_like_the_whole_series(self):
        # A window that would run past the last sample is clamped to end
        # there, so a persist=hold leaf holds the same last value.
        import jax_datetime as jdt

        from jcm.date import DateData
        from jcm.forcing import BY_DATE_INTERP, _select_time_series
        from jcm.nudging import samples_per_span, window_target
        full = self._target(BY_DATE_INTERP, persist="hold")
        n = samples_per_span(full, 86400)
        t0 = np.datetime64("2010-01-10T12").astype("datetime64[s]")
        win = window_target(full, int(t0.astype(np.int64)), n)
        self.assertEqual(win.u_wind.values.shape[0], n)
        for dt_h in (0, 5, 6, 12, 23):
            date = DateData.set_date(jdt.to_datetime(
                t0 + np.timedelta64(dt_h * 3600, "s")))
            np.testing.assert_array_equal(
                np.asarray(_select_time_series(win.u_wind, date)),
                np.asarray(_select_time_series(full.u_wind, date)))

    def test_static_target_is_returned_whole(self):
        from jcm.nudging import NudgingTarget, samples_per_span, window_target
        static = NudgingTarget(u_wind=jnp.ones((2, 3)), v_wind=jnp.ones((2, 3)),
                               temperature=jnp.ones((2, 3)))
        self.assertIsNone(samples_per_span(static, 86400))
        self.assertIs(window_target(static, 0, 4).u_wind, static.u_wind)


class WindowForcingTest(unittest.TestCase):
    """``window_forcing``: the one public call that puts a segment's window of
    a host-resident target on the device (what ``run_chunked`` uses per chunk).
    """

    SPAN = 6 * 3600

    def _forcing(self, align, persist="strict"):
        target = WindowTargetTest()._target(align, persist)
        # Host-resident, as ``build_run_forcing`` hands it back.
        target = jax.device_get(target)
        return ForcingData.zeros((4, 3)).copy(nudging_target=target)

    def _same_selection(self, win, full, start, offsets_h):
        import jax_datetime as jdt

        from jcm.date import DateData
        from jcm.forcing import _select_time_series
        t0 = np.datetime64(start).astype("datetime64[s]")
        for dt_h in offsets_h:
            date = DateData.set_date(jdt.to_datetime(
                t0 + np.timedelta64(int(dt_h * 3600), "s")))
            for name in ("u_wind", "temperature"):
                np.testing.assert_array_equal(
                    np.asarray(_select_time_series(getattr(win, name), date)),
                    np.asarray(_select_time_series(getattr(full, name), date)),
                    err_msg=f"{name} start={start} +{dt_h}h")

    def test_window_matches_whole_target_at_start_between_samples_and_end(self):
        from jcm.forcing import BY_DATE, BY_DATE_INTERP
        from jcm.nudging import samples_per_span
        for align in (BY_DATE, BY_DATE_INTERP):
            full = self._forcing(align)
            n = samples_per_span(full.nudging_target, self.SPAN)
            # The series start; a start between two samples (1 h past 06:00);
            # a start whose window the series end clamps.
            for start in ("2010-01-01T00", "2010-01-04T07", "2010-01-10T06"):
                win = window_forcing(
                    full, np.datetime64(start).astype("datetime64[s]"),
                    self.SPAN)
                self.assertEqual(win.nudging_target.u_wind.values.shape[0], n)
                self._same_selection(win.nudging_target, full.nudging_target,
                                     start, (0, 1, 2.5, 5.9))

    def test_window_at_the_end_holds_the_last_sample(self):
        from jcm.forcing import BY_DATE_INTERP
        full = self._forcing(BY_DATE_INTERP, persist="hold")
        start = "2010-01-10T12"        # past the last sample (2010-01-10T18 - 6 h)
        win = window_forcing(full, np.datetime64(start), self.SPAN)
        self._same_selection(win.nudging_target, full.nudging_target, start,
                             (0, 6, 12, 23))

    def test_start_accepts_epoch_seconds_datetime_and_model_time(self):
        import datetime

        import jax_datetime as jdt

        from jcm.forcing import BY_DATE
        full = self._forcing(BY_DATE)
        when = datetime.datetime(2010, 1, 3, 7)
        secs = int(np.datetime64(when, "s").astype(np.int64))
        want = np.asarray(window_forcing(full, secs, self.SPAN)
                          .nudging_target.u_wind.values)
        for start in (when, np.datetime64(when),
                      jdt.to_datetime(np.datetime64(when, "s"))):
            np.testing.assert_array_equal(
                np.asarray(window_forcing(full, start, self.SPAN)
                           .nudging_target.u_wind.values), want)

    def test_window_is_on_the_device(self):
        from jcm.forcing import BY_DATE
        full = self._forcing(BY_DATE)
        self.assertIsInstance(full.nudging_target.u_wind.values, np.ndarray)
        win = window_forcing(full, np.datetime64("2010-01-03T00"), self.SPAN)
        self.assertIsInstance(win.nudging_target.u_wind.values, jax.Array)
        self.assertIsInstance(win.nudging_target.u_wind.times.delta.days, jax.Array)

    def test_no_target_or_static_target_is_a_no_op(self):
        bare = ForcingData.zeros((4, 3))
        self.assertIs(window_forcing(bare, 0, self.SPAN), bare)
        static = NudgingTarget(u_wind=jnp.ones((2, 3)), v_wind=jnp.ones((2, 3)),
                               temperature=jnp.ones((2, 3)))
        held = bare.copy(nudging_target=static)
        self.assertIs(window_forcing(held, 0, self.SPAN), held)


class TestLatTop(unittest.TestCase):
    """``LatTop``: the latitude-dependent, tapered top of the nudged region."""

    LT = dict(p_tropical_hpa=150.0, p_polar_hpa=400.0,
              lat_tropical_deg=20.0, lat_polar_deg=35.0, taper_decades=0.1)

    def _hand_top(self, lat):
        a = abs(lat)
        if a <= 20.0:
            w = 1.0
        elif a >= 35.0:
            w = 0.0
        else:
            w = np.cos(0.5 * np.pi * (a - 20.0) / 15.0) ** 2
        return np.exp(w * np.log(150.0) + (1.0 - w) * np.log(400.0))

    def test_weight_matches_hand_computed_profile(self):
        lt = LatTop.from_mapping(self.LT)
        p = np.array([50.0, 100.0, 150.0, 170.0, 188.0, 300.0, 400.0, 500.0, 1000.0])
        lats = np.array([0.0, 10.0, 20.0, 27.5, 35.0, 60.0, -27.5, -90.0])
        w = np.asarray(lt.weight(p, lats))
        self.assertEqual(w.shape, (p.size, lats.size))
        for j, lat in enumerate(lats):
            top = self._hand_top(lat)
            expect = np.clip(np.log10(p / top) / 0.1, 0.0, 1.0)
            np.testing.assert_allclose(w[:, j], expect, atol=1e-5)

    def test_regimes(self):
        lt = LatTop.from_mapping(self.LT)
        np.testing.assert_allclose(lt.top_hpa(jnp.array([0.0, 20.0, -15.0])), 150.0, rtol=1e-5)
        np.testing.assert_allclose(lt.top_hpa(jnp.array([35.0, 80.0, -60.0])), 400.0, rtol=1e-5)
        # North/south symmetric, and the blend is bracketed by the two tops.
        mid = lt.top_hpa(jnp.array([27.5, -27.5]))
        self.assertAlmostEqual(float(mid[0]), float(mid[1]), places=4)
        self.assertTrue(150.0 < float(mid[0]) < 400.0)
        # Un-nudged above the top, fully nudged a taper below it.
        w = np.asarray(lt.weight(np.array([100.0, 150.0, 150.0 * 10 ** 0.1, 800.0]), 0.0))
        np.testing.assert_allclose(w, [0.0, 0.0, 1.0, 1.0], atol=1e-5)

    def test_zero_width_is_a_hard_step(self):
        lt = LatTop.from_mapping({**self.LT, "taper_decades": 0.0})
        w = np.asarray(lt.weight(np.array([149.0, 151.0]), 0.0))
        np.testing.assert_allclose(w, [0.0, 1.0])

    def test_invalid_configs_raise(self):
        bad = [
            {**self.LT, "lat_tropical_deg": 40.0},          # ordering
            {**self.LT, "lat_tropical_deg": 35.0},          # equal
            {**self.LT, "lat_polar_deg": 95.0},
            {**self.LT, "lat_tropical_deg": -5.0},
            {**self.LT, "p_tropical_hpa": 0.0},
            {**self.LT, "p_polar_hpa": -1.0},
            {**self.LT, "taper_decades": -0.1},
            {**self.LT, "taper_decde": 0.1},                # misspelt key
        ]
        for mapping in bad:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                LatTop.from_mapping(mapping)
        # The hard floor may not sit below a top (it would silently bind).
        with self.assertRaises(ValueError):
            LatTop.from_mapping(self.LT, min_pressure_hpa=200.0)
        LatTop.from_mapping(self.LT, min_pressure_hpa=150.0)
        with self.assertRaises(ValueError):
            NudgingTerm(NudgingConfig.winds_only(4),
                        lat_top=LatTop(taper_decades=-1.0))

    def test_weight_gradient(self):
        from jcm.testing import check_gradients
        p = jnp.array([120.0, 175.0, 230.0, 350.0, 600.0])
        lat = jnp.array([5.0, 24.0, 29.0, 33.0, 50.0])

        def f(lt):
            return lt.weight(p, lat)

        lt = LatTop(p_tropical_hpa=150.0, p_polar_hpa=400.0,
                    lat_tropical_deg=20.0, lat_polar_deg=35.0, taper_decades=0.3)
        check_gradients(f, (lt,), rtol=1e-2)


class TestLatTopTerm(unittest.TestCase):
    """``NudgingTerm(lat_top=...)`` on the whole-grid and column layouts."""

    def setUp(self):
        self.coords = get_speedy_coords(layers=8, spectral_truncation=21)
        self.nlev = self.coords.vertical.layers
        self.nlon, self.nlat = self.coords.horizontal.nodal_shape
        self.lat_top = LatTop(p_tropical_hpa=300.0, p_polar_hpa=700.0,
                              lat_tropical_deg=20.0, lat_polar_deg=45.0,
                              taper_decades=0.2)
        rng = np.random.default_rng(3)
        shape = (self.nlev, self.nlon, self.nlat)
        self.state = PhysicsState.zeros(
            shape, u_wind=jnp.asarray(rng.normal(size=shape)),
            v_wind=jnp.asarray(rng.normal(size=shape)),
            temperature=jnp.asarray(rng.normal(size=shape) + 250.0),
            specific_humidity=jnp.asarray(rng.uniform(size=shape)))
        self.target = NudgingTarget(
            u_wind=jnp.zeros(shape), v_wind=jnp.zeros(shape),
            temperature=jnp.full(shape, 250.0),
            specific_humidity=jnp.zeros(shape))
        self.forcing = types.SimpleNamespace(nudging_target=self.target)
        self.config = NudgingConfig(
            inv_tau_wind=jnp.full(self.nlev, 1.0 / 21600.0),
            inv_tau_temperature=jnp.full(self.nlev, 1.0 / 43200.0),
            inv_tau_humidity=jnp.full(self.nlev, 1.0 / 86400.0))

    def _term(self, lat_top, config=None):
        term = NudgingTerm(config or self.config, lat_top=lat_top)
        term.cache_coords(self.coords)
        return term

    def test_default_is_bit_identical_to_plain_relaxation(self):
        term = self._term(None)
        tend, _ = term(self.state, {}, self.forcing, None)
        ref = nudging_tendency(self.state, self.target, self.config,
                               nodal_shape=(self.nlon, self.nlat))
        for name in ("u_wind", "v_wind", "temperature", "specific_humidity"):
            np.testing.assert_array_equal(
                np.asarray(getattr(tend, name)), np.asarray(getattr(ref, name)))

    def test_tendency_is_weight_times_plain_relaxation(self):
        term = self._term(self.lat_top)
        tend, _ = term(self.state, {}, self.forcing, None)
        plain = nudging_tendency(self.state, self.target, self.config,
                                 nodal_shape=(self.nlon, self.nlat))
        lat_deg = np.rad2deg(np.asarray(self.coords.horizontal.latitudes))
        p_hpa = reference_pressure_pa(self.coords.vertical) / 100.0
        w = np.asarray(self.lat_top.weight(p_hpa, lat_deg))   # (nlev, nlat)
        w = w[:, None, :]
        for name in ("u_wind", "v_wind", "temperature", "specific_humidity"):
            np.testing.assert_allclose(
                np.asarray(getattr(tend, name)),
                w * np.asarray(getattr(plain, name)), rtol=1e-5, atol=1e-12)
        # The profile is genuinely latitude dependent here.
        self.assertGreater(float(np.ptp(w[:, 0, :])), 0.5)

    def test_column_layout_matches_grid_layout(self):
        term = self._term(self.lat_top)
        grid, _ = term(self.state, {}, self.forcing, None)
        ncols = self.nlon * self.nlat
        flat = lambda a: a.reshape(self.nlev, ncols)  # noqa: E731
        state_col = PhysicsState.zeros(
            (self.nlev, ncols), u_wind=flat(self.state.u_wind),
            v_wind=flat(self.state.v_wind),
            temperature=flat(self.state.temperature),
            specific_humidity=flat(self.state.specific_humidity))
        col, _ = term(state_col, {}, self.forcing, None)
        for name in ("u_wind", "v_wind", "temperature", "specific_humidity"):
            np.testing.assert_allclose(
                np.asarray(getattr(grid, name)).reshape(self.nlev, ncols),
                np.asarray(getattr(col, name)), rtol=1e-6, atol=1e-12)

    def test_composes_with_pbl_and_floor_masks(self):
        inv = inv_tau_profile(self.coords.vertical, tau_hours=6.0,
                              min_pressure_hpa=100.0, pbl_levels=2)
        config = NudgingConfig(inv_tau_wind=jnp.asarray(inv),
                               inv_tau_temperature=jnp.zeros(self.nlev))
        term = self._term(self.lat_top, config)
        tend, _ = term(self.state, {}, self.forcing, None)
        # Bottom PBL levels stay free; temperature is not nudged.
        self.assertTrue(np.all(np.asarray(tend.u_wind)[-2:] == 0.0))
        self.assertTrue(np.all(np.asarray(tend.temperature) == 0.0))

    def test_gradient_through_tendency(self):
        from jcm.testing import check_gradients
        term = self._term(self.lat_top)

        def f(temperature, u_wind):
            state = self.state.copy(temperature=temperature, u_wind=u_wind)
            tend, _ = term(state, {}, self.forcing, None)
            return tend.temperature, tend.u_wind

        check_gradients(f, (self.state.temperature, self.state.u_wind), rtol=1e-3)

    def test_profile_parameters_are_differentiable_through_the_term(self):
        """A gradient w.r.t. a ``LatTop`` leaf reaches the term's tendency."""
        import jax

        def f(p_tropical_hpa):
            lat_top = self.lat_top.replace(p_tropical_hpa=p_tropical_hpa)
            tend, _ = self._term(lat_top)(self.state, {}, self.forcing, None)
            return jnp.sum(tend.u_wind ** 2)

        grad = jax.grad(f)(jnp.asarray(300.0))
        self.assertTrue(np.isfinite(float(grad)))
        self.assertNotEqual(float(grad), 0.0)


class TestLatTopInModel(unittest.TestCase):
    """The latitude-dependent top survives composition and runs in a Model."""

    def test_model_run_nudges_only_below_the_top(self):
        coords = get_speedy_coords()
        terrain = TerrainData.aquaplanet(coords)
        nlev = coords.vertical.layers
        nlon, nlat = coords.horizontal.nodal_shape
        target = NudgingTarget.from_dataset(
            _zero_winds_target_dataset(nlev, nlon, nlat), time_var=None)
        forcing = ForcingData.zeros(coords.horizontal.nodal_shape).replace(
            nudging_target=target)
        config = NudgingConfig.winds_only(nlev=nlev, tau_seconds=3600.0)

        def run(lat_top):
            physics = with_nudging(speedy_physics(), config, lat_top=lat_top)
            return Model(coords=coords, terrain=terrain, physics=physics).run(
                forcing=forcing, save_interval=1, total_time=2)

        free = run(LatTop(p_tropical_hpa=1.0e4, p_polar_hpa=1.0e4,
                          lat_polar_deg=45.0))  # top below every level: no nudging
        none = Model(coords=coords, terrain=terrain, physics=speedy_physics()).run(
            forcing=forcing, save_interval=1, total_time=2)
        np.testing.assert_allclose(
            np.asarray(free.dynamics.u_wind), np.asarray(none.dynamics.u_wind),
            atol=1e-5)
        deep = run(LatTop(p_tropical_hpa=1.0, p_polar_hpa=1.0, lat_polar_deg=45.0))
        plain = run(None)
        np.testing.assert_allclose(
            np.asarray(deep.dynamics.u_wind), np.asarray(plain.dynamics.u_wind),
            rtol=1e-4, atol=1e-5)
