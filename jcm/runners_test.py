"""Unit tests for ``jcm.runners`` and the Hydra config groups.

Verifies that each config-group combination resolves to a sensible model and
that a short integration step runs without raising. Kept deliberately cheap
so it can run in the regular pytest sweep — we do not test the full ECHAM
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

    def test_echam_compose(self):
        cfg = _compose([
            "physics=echam",
            "grid=echam_t42_l8_sigma",
        ])
        self.assertEqual(cfg.physics.name, "echam")
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

    def test_build_coords_echam_sigma(self):
        cfg = _compose(["grid=echam_t42_l8_sigma"])
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

    def test_build_physics_param_overrides(self):
        # Override an ECHAM convection parameter via the cfg.physics.params
        # path; the resulting Parameters should pick up the new value.
        cfg = _compose([
            "physics=echam",
            "grid=echam_t42_l8_sigma",
            "+physics.params.convection.entrpen=4e-4",
        ])
        physics = build_physics(cfg)
        self.assertAlmostEqual(
            float(physics.parameters.convection.entrpen), 4e-4,
        )

    def test_build_physics_unknown_subgroup_raises(self):
        cfg = _compose([
            "physics=echam",
            "grid=echam_t42_l8_sigma",
            "+physics.params.not_a_subgroup.foo=1.0",
        ])
        with self.assertRaisesRegex(ValueError, "Unknown physics parameter subgroup"):
            build_physics(cfg)

    def test_build_physics_curated_preset(self):
        # The echam-strong-conv preset should bump entrpen via the same
        # override pipeline.
        cfg = _compose([
            "physics=echam-strong-conv",
            "grid=echam_t42_l8_sigma",
        ])
        physics = build_physics(cfg)
        self.assertAlmostEqual(
            float(physics.parameters.convection.entrpen), 4e-4,
        )

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


class TestEndToEnd(unittest.TestCase):
    """Tiny end-to-end runs at T31/L8.

    Kept fast so the push CI exercises the full ``runners.run`` +
    ``Model.run`` path.
    """

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


class TestModeDispatch(unittest.TestCase):
    """Cover the ``run.mode = chunked / prescribed / scm`` dispatch paths."""

    def test_chunked_run_writes_per_chunk_netcdfs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _compose([
                "physics=held_suarez",
                "grid=held_suarez_t31_l8",
                "run.time_step=180",
                "run.total_time=2",
                "run.save_interval=1",
                "run.chunk_days=1",
                f"run.output_prefix={tmpdir}/chunk",
            ])
            preds = run(cfg)
            # ``run_chunked`` returns a list of per-chunk health reports.
            self.assertIsInstance(preds, list)
            self.assertGreaterEqual(len(preds), 1)
            self.assertTrue(any(Path(tmpdir).glob("chunk_day*.nc")))

    def _write_state_file(self, path):
        # Run a tiny full simulation and dump it so the prescribed/scm modes
        # have a JCM-shaped state to load.
        cfg = _compose([
            "physics=held_suarez",
            "grid=held_suarez_t31_l8",
            "run.time_step=180",
            "run.total_time=2",
            "run.save_interval=1",
        ])
        preds = run(cfg)
        preds.to_xarray().to_netcdf(path)

    def test_prescribed_mode_runs_from_state_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.nc"
            self._write_state_file(str(state_file))
            cfg = _compose([
                "physics=held_suarez",
                "grid=held_suarez_t31_l8",
                "run.time_step=180",
                "run.mode=prescribed",
                f"run.state_file={state_file}",
            ])
            preds = run(cfg)
            self.assertEqual(preds.tendencies.temperature.shape[0], 2)

    def test_scm_mode_picks_column_from_state_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.nc"
            self._write_state_file(str(state_file))
            cfg = _compose([
                "physics=held_suarez",
                "grid=held_suarez_t31_l8",
                "run.time_step=180",
                "run.mode=scm",
                f"run.state_file={state_file}",
                "run.column.lat_deg=0.0",
                "run.column.lon_deg=0.0",
            ])
            preds = run(cfg)
            # SCM output is 1-D in level with a leading time axis.
            self.assertEqual(preds.tendencies.temperature.shape, (2, 8))


class TestMainCLI(unittest.TestCase):
    """Smoke-test the Hydra CLI entry point at ``jcm.main``."""

    def test_main_writes_netcdf(self):
        # Hydra's testing helpers compose the same config the CLI would and
        # invoke the entry point; this covers ``main`` + ``resolve_output_path``
        # + ``save_predictions`` without spawning a subprocess.
        import tempfile
        from hydra.experimental.callback import Callback  # noqa: F401  (Hydra check)
        from jcm import main as main_module

        with tempfile.TemporaryDirectory() as tmpdir:
            with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
                cfg = compose(
                    config_name="config",
                    overrides=[
                        "physics=held_suarez",
                        "grid=held_suarez_t31_l8",
                        "run.time_step=180",
                        "run.total_time=2",
                        "run.save_interval=1",
                        f"run.output={tmpdir}/cli_test.nc",
                    ],
                    return_hydra_config=True,
                )
                # Hydra's runtime config isn't normally available outside the
                # ``@hydra.main`` decorator; resolve it manually for the test.
                from hydra.core.hydra_config import HydraConfig
                HydraConfig.instance().set_config(cfg)
                main_module.main.__wrapped__(cfg)
            self.assertTrue(Path(tmpdir, "cli_test.nc").exists())


# ---------------------------------------------------------------------------
# Slow-marked companions
#
# The PR CI runs ``pytest -m "slow" --cov-fail-under=80``. The push CI runs
# ``-m "not slow" --cov-fail-under=90``. We need the same end-to-end paths
# exercised in *both* passes so neither coverage threshold drops below the
# bar after we add new code. Subclassing inherits every test method and the
# class-level ``slow`` marker decides which CI pass picks them up.
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEndToEndSlow(TestEndToEnd):
    pass


@pytest.mark.slow
class TestModeDispatchSlow(TestModeDispatch):
    pass


@pytest.mark.slow
class TestMainCLISlow(TestMainCLI):
    pass
