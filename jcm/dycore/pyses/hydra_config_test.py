"""Hydra wiring tests for the pySES backend (dycore config group)."""

import unittest

import pytest
from hydra import compose, initialize_config_dir


def _cfg(overrides):
    from pathlib import Path

    import jcm

    config_dir = str(Path(jcm.__file__).resolve().parent / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config", overrides=overrides)


class PysesHydraConfigTest(unittest.TestCase):
    def test_canonical_config_builds_pyses_model(self):
        pytest.importorskip("pyses")
        import jax.numpy as jnp

        from jcm.dycore.pyses import PysesCamSEDycore
        from jcm.runners import build_model

        cfg = _cfg(["dycore=pyses_ne30l47", "physics=speedy",
                    "run=pyses_year",
                    # test-size grid; the canonical file documents ne30
                    "dycore.nx=3", "dycore.n_sponge=8"])
        model = build_model(cfg)
        dycore = model.dycore
        self.assertIsInstance(dycore, PysesCamSEDycore)
        # The proven-stable settings arrive from the config file.
        self.assertEqual(dycore.dt_seconds, 900.0)
        self.assertEqual(model.dt_si.m, 900.0)  # Model adopted the dycore dt
        self.assertEqual(dycore.nu_top, 2.5e5)
        self.assertEqual(dycore.timestep_config["physics_dynamics_coupling"].name,
                         "lump_tracers_dribble_dynamics")
        self.assertIn("nu_div_factor", dycore.diffusion_config)
        self.assertEqual(dycore.physics_dtype, jnp.float32)
        # Finite-lid sponge appended (T relaxation + implicit uv Rayleigh).
        names = [t.name for t in model.physics.terms]
        self.assertIn("upper_temperature_relaxation", names)

    def test_dinosaur_default_unchanged(self):
        from jcm.dycore.dinosaur.dycore import DinosaurDycore
        from jcm.runners import build_model

        cfg = _cfg([])  # all defaults: dycore=dinosaur
        model = build_model(cfg)
        self.assertIsInstance(model.dycore, DinosaurDycore)

    def test_dinosaur_init_kinds_rejected_on_pyses(self):
        pytest.importorskip("pyses")
        from jcm.runners import build_model

        cfg = _cfg(["dycore=pyses_ne30l47", "init=jw", "dycore.nx=3"])
        with self.assertRaisesRegex(ValueError, "dinosaur-specific"):
            build_model(cfg)

    def test_jam_forcing_files_fail_loudly_on_pyses(self):
        # The column forcing path does not carry the aerosol inputs yet;
        # running silently aerosol-dark is exactly the ne30 JAM failure —
        # it must raise instead.
        pytest.importorskip("pyses")
        from jcm.runners import build_forcing, build_model

        cfg = _cfg(["dycore=pyses_ne30l47", "physics=speedy", "dycore.nx=3",
                    "forcing.dms_file=/nonexistent.nc"])
        model = build_model(cfg)
        with self.assertRaises(NotImplementedError):
            build_forcing(cfg, model.coords, dycore=model.dycore)


if __name__ == "__main__":
    unittest.main()
