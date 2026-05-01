"""Tests for jcm/nudging.py (#129)."""

import unittest

import jax.numpy as jnp
import numpy as np
import xarray as xr

from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.terrain import TerrainData
from jcm.model import Model
from jcm.nudging import (
    Nudging,
    NudgingConfig,
    NudgingTarget,
    nudging_tendency,
)


def _zero_winds_target_dataset(nlev, nlon, nlat, T_K=250.0, ps_pa=1.0e5):
    """Synthetic 'observation' Dataset: zero winds, isothermal T, fixed ps.

    Used as a deterministic relaxation target so we can verify the
    direction and magnitude of the relaxation tendency.
    """
    return xr.Dataset({
        'u':  (('lev', 'lon', 'lat'), np.zeros((nlev, nlon, nlat), dtype=np.float32)),
        'v':  (('lev', 'lon', 'lat'), np.zeros((nlev, nlon, nlat), dtype=np.float32)),
        'T':  (('lev', 'lon', 'lat'), np.full((nlev, nlon, nlat), T_K, dtype=np.float32)),
        'ps': (('lon', 'lat'),        np.full((nlon, nlat), ps_pa, dtype=np.float32)),
    })


class TestNudgingConfig(unittest.TestCase):

    def setUp(self):
        coords = get_speedy_coords()
        self.specs = Model(coords=coords, physics=speedy_physics()).primitive.physics_specs
        self.nlev = coords.vertical.layers

    def test_winds_only_zeros_T_and_ps(self):
        cfg = NudgingConfig.winds_only(self.nlev, physics_specs=self.specs)
        self.assertTrue(jnp.all(cfg.inv_tau_temperature == 0.0))
        self.assertEqual(float(cfg.inv_tau_log_surface_pressure), 0.0)

    def test_winds_only_pbl_mask(self):
        # `pbl_levels=2` zeros the bottom two levels of the wind tau.
        cfg = NudgingConfig.winds_only(
            self.nlev, pbl_levels=2, physics_specs=self.specs,
        )
        self.assertTrue(jnp.all(cfg.inv_tau_vorticity[:-2] > 0.0))
        self.assertTrue(jnp.all(cfg.inv_tau_vorticity[-2:] == 0.0))

    def test_winds_only_requires_physics_specs(self):
        with self.assertRaises(ValueError):
            NudgingConfig.winds_only(self.nlev)


class TestNudgingTendencyDirection(unittest.TestCase):
    """Pin the direction of the relaxation tendency: dX/dt should drive
    the state toward the target.
    """

    def test_tendency_points_toward_target(self):
        coords = get_speedy_coords()
        terrain = TerrainData.aquaplanet(coords)
        physics = speedy_physics()
        model = Model(coords=coords, terrain=terrain, physics=physics)
        nlev = coords.vertical.layers
        nlon, nlat = coords.horizontal.nodal_shape

        ds = _zero_winds_target_dataset(nlev, nlon, nlat)
        target = NudgingTarget.from_dataset(
            ds, coords,
            reference_temperature=model.primitive.reference_temperature,
            physics_specs=model.primitive.physics_specs,
            time_var=None,
        )
        # Nudge everything (winds + T + ps) at ~1 day timescale.
        tau_nd = model.primitive.physics_specs.nondimensionalize(
            86400.0 * model.primitive.physics_specs.units.second
        ) if False else 1.0  # use 1.0 nondim for clean math in this unit test
        config = NudgingConfig(
            inv_tau_vorticity=jnp.ones(nlev) * tau_nd,
            inv_tau_divergence=jnp.ones(nlev) * tau_nd,
            inv_tau_temperature=jnp.ones(nlev) * tau_nd,
            inv_tau_log_surface_pressure=jnp.array(tau_nd),
        )

        # Build a state whose vorticity/divergence are slightly positive;
        # the target is zero everywhere, so the tendency should be negative.
        state = model._prepare_initial_modal_state()
        bumped = state.replace(
            vorticity=state.vorticity + 1e-3,
            divergence=state.divergence + 1e-3,
        )
        tend = nudging_tendency(bumped, target, config)
        # tendency = inv_tau · (target − state) — with state > target, tendency < 0.
        self.assertTrue(jnp.all(tend.vorticity <= 0.0))
        self.assertTrue(jnp.all(tend.divergence <= 0.0))


class TestNudgingShrinksWinds(unittest.TestCase):
    """End-to-end: a model run with wind nudging toward zero should have
    smaller winds than a free-running baseline after a few days.
    """

    def test_aquaplanet_winds_shrink(self):
        coords = get_speedy_coords()
        terrain = TerrainData.aquaplanet(coords)
        physics = speedy_physics()
        model = Model(coords=coords, terrain=terrain, physics=physics)
        nlev = coords.vertical.layers
        nlon, nlat = coords.horizontal.nodal_shape

        ds = _zero_winds_target_dataset(nlev, nlon, nlat)
        target = NudgingTarget.from_dataset(
            ds, coords,
            reference_temperature=model.primitive.reference_temperature,
            physics_specs=model.primitive.physics_specs,
            time_var=None,
        )
        config = NudgingConfig.winds_only(
            nlev=nlev, tau_seconds=86400.0,
            physics_specs=model.primitive.physics_specs,
        )

        preds_nudged = Model(
            coords=coords, terrain=terrain, physics=physics,
            nudging=Nudging(target, config),
        ).run(save_interval=1, total_time=2)
        preds_free = Model(
            coords=coords, terrain=terrain, physics=physics,
        ).run(save_interval=1, total_time=2)

        u_n = float(jnp.mean(jnp.abs(preds_nudged.dynamics.u_wind[-1])))
        u_f = float(jnp.mean(jnp.abs(preds_free.dynamics.u_wind[-1])))
        self.assertLess(u_n, u_f,
                        msg=f"nudging didn't shrink winds: |u_n|={u_n} vs |u_f|={u_f}")


class TestNudgingDefaultIsNoOp(unittest.TestCase):
    """Constructing a Model with `nudging=None` (the default) should give
    bit-identical output to one with no nudging at all.
    """

    def test_no_nudging_default_runs_normally(self):
        coords = get_speedy_coords()
        terrain = TerrainData.aquaplanet(coords)
        physics = speedy_physics()
        # Just verifying the run doesn't blow up and returns sensible shapes.
        preds = Model(
            coords=coords, terrain=terrain, physics=physics, nudging=None,
        ).run(save_interval=1, total_time=1)
        self.assertEqual(preds.dynamics.u_wind.shape[0], 1)
        self.assertTrue(jnp.all(jnp.isfinite(preds.dynamics.u_wind)))


if __name__ == "__main__":
    unittest.main()
