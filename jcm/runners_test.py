"""Unit tests for ``jcm.runners`` and the Hydra config groups.

Verifies that each config-group combination resolves to a sensible model and
that a short integration step runs without raising. Kept deliberately cheap
so it can run in the regular pytest sweep — we do not test the full ICON
T85x47 grid here.
"""

import unittest
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from jcm.runners import (
    build_coords,
    build_diffusion,
    build_model,
    build_physics,
    build_terrain,
    run,
)


CONFIG_DIR = str(Path(__file__).parent / "config")


def _compose(overrides=None):
    overrides = overrides or []
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=overrides)


class TestConfigComposition(unittest.TestCase):
    def test_default_compose(self):
        cfg = _compose()
        self.assertEqual(cfg.physics.name, "speedy")
        self.assertEqual(cfg.grid.vertical, "sigma")
        self.assertEqual(cfg.grid.layers, 8)
        self.assertEqual(cfg.run.time_step, 10)
        self.assertEqual(cfg.init.kind, "isothermal")
        self.assertEqual(cfg.terrain.kind, "aquaplanet")
        self.assertEqual(cfg.forcing.kind, "default")
        self.assertEqual(float(cfg.diffusion.scale), 1.0)

    def test_icon_compose(self):
        cfg = _compose([
            "physics=icon",
            "grid=icon_t42_l8_sigma",
        ])
        self.assertEqual(cfg.physics.name, "icon")
        self.assertEqual(cfg.physics.radiation, "grey")
        self.assertEqual(cfg.grid.vertical, "sigma")

    def test_held_suarez_compose(self):
        cfg = _compose([
            "physics=held_suarez",
            "grid=held_suarez_t31_l8",
        ])
        self.assertEqual(cfg.physics.name, "held_suarez")

    def test_run_smoke_overrides(self):
        cfg = _compose(["run=smoke"])
        self.assertEqual(cfg.run.total_time, 1)
        self.assertEqual(cfg.run.save_interval, 1)

    def test_init_jw_compose(self):
        cfg = _compose(["init=jw"])
        self.assertEqual(cfg.init.kind, "jw")


class TestBuilders(unittest.TestCase):
    def test_build_coords_speedy(self):
        cfg = _compose()
        coords = build_coords(cfg)
        self.assertEqual(coords.horizontal.nodal_shape, (96, 48))

    def test_build_coords_icon_sigma(self):
        cfg = _compose(["grid=icon_t42_l8_sigma"])
        coords = build_coords(cfg)
        self.assertEqual(coords.horizontal.nodal_shape, (128, 64))

    def test_build_physics_speedy(self):
        cfg = _compose()
        physics = build_physics(cfg)
        self.assertIsNotNone(physics)

    def test_build_physics_held_suarez(self):
        cfg = _compose(["physics=held_suarez", "grid=held_suarez_t31_l8"])
        physics = build_physics(cfg)
        self.assertIsNotNone(physics)

    def test_build_terrain_aquaplanet(self):
        cfg = _compose()
        coords = build_coords(cfg)
        terrain = build_terrain(cfg, coords)
        self.assertIsNotNone(terrain.orog)

    def test_build_diffusion_scaled(self):
        cfg = _compose(["diffusion=strong"])
        diffusion = build_diffusion(cfg)
        from jcm.diffusion import DiffusionFilter
        base = DiffusionFilter.default()
        self.assertAlmostEqual(
            float(diffusion.div_timescale),
            float(base.div_timescale) * 0.5,
        )

    def test_build_model_held_suarez(self):
        cfg = _compose([
            "physics=held_suarez",
            "grid=held_suarez_t31_l8",
            "run.time_step=180",
        ])
        model = build_model(cfg)
        self.assertEqual(model.coords.horizontal.nodal_shape, (96, 48))


@pytest.mark.slow
class TestEndToEnd(unittest.TestCase):
    def test_run_held_suarez_smoke(self):
        cfg = _compose([
            "physics=held_suarez",
            "grid=held_suarez_t31_l8",
            "run=smoke",
            "run.time_step=180",
            "run.total_time=2",
            "run.save_interval=1",
        ])
        predictions = run(cfg)
        self.assertEqual(predictions.dynamics.u_wind.shape[0], 2)

    def test_run_speedy_default_smoke(self):
        cfg = _compose([
            "run.time_step=720",
            "run.total_time=2",
            "run.save_interval=1",
        ])
        predictions = run(cfg)
        self.assertEqual(predictions.dynamics.u_wind.shape[0], 2)
