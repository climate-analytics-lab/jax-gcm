"""Tests for ``jcm/nudging.py``.

The previous spectral-space implementation has been promoted to a
:class:`PhysicsTerm` whose reference target rides on :class:`ForcingData`
(sliced per step by the Model). These tests cover the new gridpoint
plumbing.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from jcm.forcing import ForcingData
from jcm.model import Model
from jcm.nudging import (
    NudgingConfig, NudgingTarget,
    inv_tau_profile, nudging_tendency, with_nudging,
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
            specific_humidity=jnp.full(shape, 8.0),  # g/kg, a moist column
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
