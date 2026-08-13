"""Tests for ``jcm/nudging.py``.

The previous spectral-space implementation has been promoted to a
:class:`PhysicsTerm` whose reference target rides on :class:`ForcingData`
(sliced per step by the Model). These tests cover the new gridpoint
plumbing.
"""

import unittest

import jax.numpy as jnp
import numpy as np
import xarray as xr

from jcm.forcing import ForcingData
from jcm.model import Model
from jcm.nudging import (
    NudgingConfig, NudgingTarget,
    nudging_tendency, with_nudging,
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


class TestNudgingTendencyDirection(unittest.TestCase):
    """The relaxation tendency should drive the state toward the target."""

    def test_tendency_points_toward_target(self):
        coords = get_speedy_coords()
        nlev = coords.vertical.layers
        nlon, nlat = coords.horizontal.nodal_shape
        shape = (nlev, nlon, nlat)

        ds = _zero_winds_target_dataset(nlev, nlon, nlat, T_K=250.0)
        target = NudgingTarget.from_dataset(ds, time_var=None)
        config = NudgingConfig(
            inv_tau_wind=jnp.ones(nlev),
            inv_tau_temperature=jnp.ones(nlev),
        )
        state = PhysicsState(
            u_wind=jnp.full(shape, 5.0),
            v_wind=jnp.full(shape, -3.0),
            temperature=jnp.full(shape, 280.0),
            specific_humidity=jnp.zeros(shape),
            geopotential=jnp.zeros(shape),
            normalized_surface_pressure=jnp.ones((nlon, nlat)),
            tracers={},
        )
        tend = nudging_tendency(state, target, config)
        self.assertTrue(jnp.all(tend.u_wind <= 0.0))
        self.assertTrue(jnp.all(tend.v_wind >= 0.0))         # state v < target v
        self.assertTrue(jnp.all(tend.temperature <= 0.0))    # state T > target T


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
        tcol = nudging_tendency(state_col, target, cfg)

        for name in ("u_wind", "v_wind", "temperature"):
            with self.subTest(field=name):
                a = np.asarray(getattr(t3d, name)).reshape(
                    self.NLEV, self.NLON * self.NLAT)
                b = np.asarray(getattr(tcol, name))
                self.assertEqual(b.shape, (self.NLEV, self.NLON * self.NLAT))
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-12)

    def test_genuinely_mismatched_grid_is_rejected(self):
        target, _, state_col = self._target_and_states()
        cfg = self._config()
        wrong = NudgingTarget(
            u_wind=target.u_wind[:, :, :-1],
            v_wind=target.v_wind[:, :, :-1],
            temperature=target.temperature[:, :, :-1],
        )
        with self.assertRaisesRegex(ValueError, "incompatible with the state"):
            nudging_tendency(state_col, wrong, cfg)
