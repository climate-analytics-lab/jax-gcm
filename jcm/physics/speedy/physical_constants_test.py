"""Tests for the resolution-aware stability time step (physical_constants.py).

These lock in the §6 "Numerical stability at high nlev" contract from
SPEEDY_VARIABLE_LEVELS.md: SPEEDY's standard 7/8-level configurations keep the
historical 30-minute step (so their validated results are unchanged), while the
high-nlev / high-truncation configurations that previously went non-finite get a
reduced, day-aligned step.
"""

import unittest

from jcm.physics.speedy.physical_constants import (
    _bottom_layer_thickness,
    stable_time_step_from_geometry,
    stable_time_step_minutes,
    _DT_DAY_DIVISORS_MINUTES,
)
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData
from jcm.model import Model
from dinosaur.scales import units


class StableTimeStepTest(unittest.TestCase):
    def test_standard_speedy_levels_keep_30_minutes(self):
        # The standard 7/8-level SPEEDY runs must be on the plateau so their
        # validated behaviour is bit-for-bit unchanged. The empirically
        # stable-at-30 configs (T21 nlev<=24, T31 nlev<=8) must also stay at 30.
        plateau = [(21, 7), (21, 8), (21, 16), (21, 24), (31, 7), (31, 8)]
        for trunc, nlev in plateau:
            self.assertEqual(
                stable_time_step_minutes(nlev, trunc), 30.0,
                msg=f"T{trunc} nlev={nlev} should stay at 30 min",
            )

    def test_unstable_configs_get_reduced_step(self):
        # The three configurations that previously NaN'd at 30 min must be
        # reduced below the empirically measured blow-up boundary.
        # (boundary: T21 n32 -> 30, T31 n16 -> 30, T31 n24 -> 25 min)
        self.assertLessEqual(stable_time_step_minutes(32, 21), 25.0)
        self.assertLessEqual(stable_time_step_minutes(16, 31), 25.0)
        self.assertLessEqual(stable_time_step_minutes(24, 31), 20.0)

    def test_step_is_day_aligned(self):
        # Every returned step must divide a 1440-minute day exactly so saved
        # frames don't drift over long runs.
        for trunc in (21, 31, 42):
            for nlev in (8, 16, 24, 32, 48):
                dt = stable_time_step_minutes(nlev, trunc)
                self.assertIn(dt, _DT_DAY_DIVISORS_MINUTES)
                self.assertAlmostEqual((1440.0 / dt) % 1.0, 0.0)

    def test_step_is_monotone_nonincreasing_in_nlev(self):
        # Thinner bottom layers (more levels) can only require a smaller or
        # equal step, never a larger one.
        for trunc in (21, 31):
            prev = float("inf")
            for nlev in (8, 16, 24, 32, 48, 64):
                dt = stable_time_step_minutes(nlev, trunc)
                self.assertLessEqual(dt, prev + 1e-9)
                prev = dt

    def test_geometry_core_matches_nlev_wrapper(self):
        # The nlev-based convenience wrapper must agree with the geometry-
        # level core evaluated at the same bottom-layer thickness (the core
        # is what the SpeedySurfaceFlux term uses with live coords).
        for trunc in (21, 31):
            for nlev in (8, 16, 32):
                self.assertEqual(
                    stable_time_step_minutes(nlev, trunc),
                    stable_time_step_from_geometry(
                        _bottom_layer_thickness(nlev), trunc,
                    ),
                )

    def test_model_auto_selects_time_step(self):
        # Model(time_step=None) must pick up the resolution-aware step from
        # the SPEEDY physics' stability limit (SpeedySurfaceFlux reads the
        # live coords), and an explicit override must win.
        coords = get_speedy_coords(layers=32, spectral_truncation=21)
        terrain = TerrainData.aquaplanet(coords)

        auto = Model(coords=coords, terrain=terrain)
        auto_min = auto.dt_si.to(units.minute).m
        self.assertAlmostEqual(auto_min, stable_time_step_minutes(32, 21), places=6)
        self.assertLess(auto_min, 30.0)

        override = Model(coords=coords, terrain=terrain, time_step=30.0)
        self.assertAlmostEqual(override.dt_si.to(units.minute).m, 30.0, places=6)

    def test_non_speedy_physics_keeps_default_step(self):
        # The stability limit is SPEEDY's (its explicit surface drag);
        # physics without that term must keep the historical 30-minute
        # default even on grids where SPEEDY would need a shorter step.
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        coords = get_speedy_coords(layers=32, spectral_truncation=21)
        physics = held_suarez_physics()
        self.assertIsNone(physics.stable_time_step_minutes(coords))
        model = Model(coords=coords, terrain=TerrainData.aquaplanet(coords),
                      physics=physics)
        self.assertAlmostEqual(model.dt_si.to(units.minute).m, 30.0, places=6)

    def test_explicit_dycore_owns_the_time_step(self):
        # With an explicit dycore and no time_step, the Model must adopt the
        # dycore's dt (single source of truth) — never auto-select a
        # different one; an explicit mismatching time_step must raise.
        from jcm.dycore.dinosaur.dycore import DinosaurDycore

        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        terrain = TerrainData.aquaplanet(coords)
        dycore = DinosaurDycore(coords=coords, terrain=terrain,
                                dt_seconds=900.0)

        model = Model(dycore)
        self.assertAlmostEqual(model.dt_si.m, 900.0, places=6)

        matching = Model(dycore, time_step=15.0)
        self.assertAlmostEqual(matching.dt_si.m, 900.0, places=6)

        with self.assertRaisesRegex(ValueError, "dt_seconds"):
            Model(dycore, time_step=30.0)


if __name__ == "__main__":
    unittest.main()
