"""Unit tests for ``jcm.runners`` and the Hydra config groups.

Verifies that each config-group combination resolves to a sensible model and
that a short integration step runs without raising. Kept deliberately cheap
so it can run in the regular pytest sweep — we do not test the full ECHAM
T85x47 grid here.
"""

import unittest
from pathlib import Path

import numpy as np
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
        self.assertIn("speedy_convection", cfg.physics.terms)
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
        # The default radiation slot in the echam preset is grey two-stream.
        self.assertIn("grey_two_stream_radiation", cfg.physics.terms)
        self.assertIn("tiedtke_convection", cfg.physics.terms)
        self.assertEqual(cfg.grid.vertical, "sigma")

    def test_held_suarez_compose(self):
        cfg = _compose([
            "physics=held_suarez",
            "grid=held_suarez_t31_l8",
        ])
        self.assertIn("held_suarez", cfg.physics.terms)

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
        # Override a per-term parameter via the new
        # ``physics.terms.<term>.params.<field>=...`` CLI path.
        cfg = _compose([
            "physics=echam",
            "grid=echam_t42_l8_sigma",
            "++physics.terms.tiedtke_convection.params.entrpen=4e-4",
        ])
        physics = build_physics(cfg)
        convection_term = next(
            t for t in physics.terms if t.category == "convection"
        )
        self.assertAlmostEqual(
            float(convection_term.params.value.entrpen), 4e-4,
        )

    def test_build_physics_curated_preset(self):
        # The echam-strong-conv preset bumps entrpen via the same
        # term-list pipeline.
        cfg = _compose([
            "physics=echam-strong-conv",
            "grid=echam_t42_l8_sigma",
        ])
        physics = build_physics(cfg)
        convection_term = next(
            t for t in physics.terms if t.category == "convection"
        )
        self.assertAlmostEqual(
            float(convection_term.params.value.entrpen), 4e-4,
        )

    def test_build_physics_swap_radiation_via_preset(self):
        # The echam-rrtmgp preset replaces grey_two_stream_radiation
        # with rrtmgp_radiation in the same logical slot.
        cfg = _compose([
            "physics=echam-rrtmgp",
            "grid=echam_t42_l8_sigma",
        ])
        physics = build_physics(cfg)
        rad_term = next(
            t for t in physics.terms if t.category == "radiation"
        )
        self.assertEqual(rad_term.name, "rrtmgp_radiation")

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


class TestAttachOzonePreservesAquaplanetSST(unittest.TestCase):
    """Regression test for #484 codex P1.

    With ``forcing.kind == default`` and ``ozone_file`` set, the
    attach-ozone helper must keep the aquaplanet cos²-latitude SST
    profile from ``default_forcing(...)`` rather than swap it for the
    uniform 288.15 K placeholder that ``ForcingData.zeros`` would yield.
    """

    def test_default_forcing_with_ozone_keeps_cos2_sst(self):
        import tempfile
        import xarray as xr
        from jcm.forcing import default_forcing
        from jcm.runners import build_coords, build_forcing

        cfg = _compose(["physics=echam", "grid=echam_t42_l8_sigma"])
        coords = build_coords(cfg)
        nlon, nlat = coords.horizontal.nodal_shape
        nlev = coords.nodal_shape[0]

        # Synthetic 12-month ozone file in the (time, level, lat, lon)
        # layout that ``OzoneClimatology.from_file`` expects. Lat/lon
        # coords match the model grid (degrees from radians) so the
        # loader's coordinate-value check passes.
        model_lat_deg = np.asarray(coords.horizontal.latitudes) * 180.0 / np.pi
        model_lon_deg = np.asarray(coords.horizontal.longitudes) * 180.0 / np.pi
        with tempfile.TemporaryDirectory() as tmp:
            ozone_path = Path(tmp) / "ozone.nc"
            xr.Dataset(
                {"O3": (
                    ("time", "level", "lat", "lon"),
                    np.full((12, nlev, nlat, nlon), 1e-6, dtype=np.float32),
                )},
                coords={
                    "time": np.arange(12),
                    "level": np.arange(nlev, dtype=np.int32),
                    "lat": model_lat_deg,
                    "lon": model_lon_deg,
                },
            ).to_netcdf(ozone_path)
            cfg.forcing.kind = "default"
            cfg.forcing.ozone_file = str(ozone_path)

            forcing_with_ozone = build_forcing(cfg, coords)

        baseline = default_forcing(coords.horizontal)
        np.testing.assert_array_equal(
            np.asarray(forcing_with_ozone.sea_surface_temperature),
            np.asarray(baseline.sea_surface_temperature),
        )


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

    def test_chunked_run_resumes_from_checkpoint(self):
        """``cfg.run.checkpoint_path`` makes a chunked run resumable.

        Drives ``run_chunked`` once for 1 of 2 chunks, then re-invokes
        with the same ``checkpoint_path`` and ``total_time=2`` and
        verifies the second invocation only steps the remaining chunk.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = f"{tmpdir}/run.ckpt"
            base_overrides = [
                "physics=held_suarez",
                "grid=held_suarez_t31_l8",
                "run.time_step=180",
                "run.save_interval=1",
                "run.chunk_days=1",
                f"run.output_prefix={tmpdir}/chunk",
                f"run.checkpoint_path={ckpt_path}",
            ]

            # First invocation: run only the first chunk (1 day total).
            cfg1 = _compose(base_overrides + ["run.total_time=1"])
            reports1 = run(cfg1)
            self.assertEqual(len(reports1), 1)
            self.assertTrue(Path(ckpt_path).exists())

            # Second invocation: total 2 days, but the first chunk
            # should be skipped because the checkpoint records day=1.
            cfg2 = _compose(base_overrides + ["run.total_time=2"])
            reports2 = run(cfg2)
            self.assertEqual(len(reports2), 1, "should run only the remaining chunk")
            self.assertAlmostEqual(reports2[0]["elapsed_days"], 2.0, places=5)

    def test_chunked_resume_with_balanced_isothermal_init(self):
        """Resume path bootstraps the physics carry for inject-based inits.

        ``inject_balanced_isothermal_profile`` populates
        ``_final_dycore_state`` but leaves ``_final_physics_state`` for
        ``Model.resume`` to lazy-build. The resume-from-checkpoint code
        path must materialise the carry itself before calling
        ``load_checkpoint``, otherwise the load raises on the
        uninitialised template (codex review on PR #479). Held-Suarez is
        the cheapest physics that supports ``init=balanced_isothermal``.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = f"{tmpdir}/run.ckpt"
            base_overrides = [
                "physics=held_suarez",
                "grid=held_suarez_t31_l8",
                "init=balanced_isothermal",
                "run.time_step=180",
                "run.save_interval=1",
                "run.chunk_days=1",
                f"run.output_prefix={tmpdir}/chunk",
                f"run.checkpoint_path={ckpt_path}",
            ]
            run(_compose(base_overrides + ["run.total_time=1"]))
            self.assertTrue(Path(ckpt_path).exists())
            reports2 = run(_compose(base_overrides + ["run.total_time=2"]))
            self.assertEqual(len(reports2), 1)
            self.assertAlmostEqual(reports2[0]["elapsed_days"], 2.0, places=5)

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
