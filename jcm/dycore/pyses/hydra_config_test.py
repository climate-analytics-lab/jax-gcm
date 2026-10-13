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
    @pytest.mark.requires_extra("pyses")
    def test_canonical_config_builds_pyses_model(self):
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

    def test_delegated_timestep_survives_a_fresh_chunk_on_dinosaur(self):
        """The fast-lane half of the delegated-timestep guard.

        The pySES version below is the end-to-end one, but it is slow and
        needs the ``pyses`` extra, so only the ``extras-tests`` CI job runs
        it. This drives the same per-chunk path — integrate, health-check,
        budget report, netCDF, checkpoint — with ``run.time_step=null`` on a
        tiny dinosaur model, which
        ``run_chunked`` accepts because a pre-built model makes the run config's
        timestep unused. Anything in that path that reads it as a number fails
        here, in the fast lane.
        """
        import tempfile
        from pathlib import Path

        import numpy as np

        from jcm.model import Model
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        from jcm.runners import run_chunked
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        coords = get_coords(np.linspace(0, 1, 9), spectral_truncation=21)
        model = Model(coords=coords, time_step=60,
                      terrain=TerrainData.aquaplanet(coords),
                      physics=held_suarez_physics())
        cfg = _cfg(["physics=held_suarez", "grid=held_suarez_t31_l8",
                    "run.time_step=null", "run.total_time=0.5",
                    "run.save_interval=0.25"])
        self.assertIsNone(cfg.run.time_step)   # the case under test

        with tempfile.TemporaryDirectory() as tmpdir:
            reports = run_chunked(cfg, chunk_days=0.25,
                                  output_prefix=f"{tmpdir}/deleg",
                                  model=model)
            self.assertGreaterEqual(len(reports), 1)
            self.assertTrue(list(Path(tmpdir).glob("deleg_day*.nc")))

    @pytest.mark.slow
    @pytest.mark.requires_extra("pyses")
    def test_delegating_config_completes_a_fresh_first_chunk(self):
        """A config whose timestep the dycore owns must survive chunk 1.

        ``run=pyses_year`` sets ``time_step: null`` deliberately, so anything
        in the per-chunk path that reads it as a number crashes AFTER the
        integration and BEFORE the checkpoint — losing the chunk. That has now
        happened twice in one campaign (a health-check argument here, and the
        scoreable-gate decision of #780), each time in code that ran fine on
        every config that names its own timestep. Drive a real fresh chunk end
        to end: integrate, health-check, budget report, the calendar-month
        stream, checkpoint.

        ``run=pyses_year`` writes calendar-month means only
        (``save_chunks: false``): a chunk leaves the checkpoint and, beside
        it, the pending month it resumes from, and the end of the run flushes
        that (here partial) month to its monthly file. There is no per-chunk
        netCDF; the dinosaur sibling above covers that path.
        """
        import tempfile
        from pathlib import Path

        from jcm.runners import run

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _cfg([
                "dycore=pyses_ne30l47", "physics=held_suarez", "run=pyses_year",
                "dycore.nx=3", "dycore.n_sponge=8",
                # One short chunk: the first fresh chunk is the whole point.
                # Its length must be a whole number of the dycore's 900 s
                # steps (Model.run rejects any other save interval), so six
                # steps: 0.0625 d = 5400 s.
                "run.total_time=0.0625", "run.chunk_days=0.0625",
                "run.save_interval=0.0625",
                f"run.output_prefix={tmpdir}/deleg",
                f"run.checkpoint_path={tmpdir}/deleg.ckpt",
            ])
            self.assertIsNone(cfg.run.time_step)   # the point of the config
            reports = run(cfg)

            # Everything the crash happened BETWEEN: the integration finished
            # and passed its health check, so the checkpoint, the pending
            # month that pairs with it, and the month flushed at the end of
            # the run must all be on disk.
            self.assertIsInstance(reports, list)
            self.assertGreaterEqual(len(reports), 1)
            self.assertTrue(reports[0]["ok"], reports[0].get("reasons"))
            self.assertTrue(Path(f"{tmpdir}/deleg.ckpt").exists())
            self.assertTrue(Path(f"{tmpdir}/deleg.ckpt.monthly").exists())
            self.assertEqual(
                [p.name for p in Path(tmpdir).glob("deleg_monthly_*.nc")],
                ["deleg_monthly_2000-01.nc"])
            self.assertFalse(list(Path(tmpdir).glob("deleg_day*.nc")))

    @pytest.mark.slow
    @pytest.mark.requires_extra("pyses")
    # A float64 value scattered into a float32 field is the same mixed
    # precision, narrowed implicitly; JAX deprecates that (a FutureWarning
    # today, an error later), so it fails here rather than on a JAX upgrade.
    @pytest.mark.filterwarnings(
        "error:scatter inputs have incompatible types:FutureWarning")
    def test_speedy_physics_completes_a_fresh_first_chunk(self):
        """``physics=speedy`` must step on pySES, not only build (#797).

        pySES runs float64 dynamics with float32 physics under
        ``jax_enable_x64``, while its forcing and SPEEDY's cached vertical
        tables stay float64. SPEEDY's ``lax.cond`` branches that recompute
        from those inputs therefore came out float64 against a float32
        pass-through branch, and the first physics step raised
        ``TypeError: cond branches must have equal output types``. The
        build-only test above cannot see it; this is the issue's command,
        run to the end of its one chunk (six 900 s steps: 0.0625 d).
        """
        import tempfile

        import numpy as np
        import xarray as xr

        from jcm.runners import run

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _cfg([
                "dycore=pyses_ne30l47", "physics=speedy", "run=pyses_year",
                "dycore.nx=3", "dycore.n_sponge=8",
                "run.total_time=0.0625", "run.chunk_days=0.0625",
                "run.save_interval=0.0625",
                f"run.output_prefix={tmpdir}/speedy",
                f"run.checkpoint_path={tmpdir}/speedy.ckpt",
            ])
            reports = run(cfg)
            self.assertIsInstance(reports, list)
            self.assertTrue(reports[0]["ok"], reports[0].get("reasons"))
            with xr.open_dataset(f"{tmpdir}/speedy_monthly_2000-01.nc") as ds:
                self.assertTrue(np.isfinite(ds["temperature"].values).all())
                # SPEEDY's diagnostics are named by field, as on dinosaur,
                # not by pytree position, and carry plumbing stays out.
                self.assertIn("surface_flux.hfluxn", ds)
                self.assertFalse([name for name in ds.data_vars
                                  if name.startswith("_")])

    def test_dinosaur_default_unchanged(self):
        from jcm.dycore.dinosaur.dycore import DinosaurDycore
        from jcm.runners import build_model

        cfg = _cfg([])  # all defaults: dycore=dinosaur
        model = build_model(cfg)
        self.assertIsInstance(model.dycore, DinosaurDycore)

    @pytest.mark.requires_extra("pyses")
    def test_dinosaur_init_kinds_rejected_on_pyses(self):
        from jcm.runners import build_model

        cfg = _cfg(["dycore=pyses_ne30l47", "init=jw", "dycore.nx=3"])
        with self.assertRaisesRegex(ValueError, "dinosaur-specific"):
            build_model(cfg)

    @pytest.mark.requires_extra("pyses")
    def test_jam_forcing_files_flow_and_fail_loudly_on_pyses(self):
        # The column path carries the JAM aerosol inputs through
        # attach_jam_forcing; a bad path must still fail loudly rather
        # than run silently aerosol-dark.
        from jcm.runners import build_forcing, build_model

        cfg = _cfg(["dycore=pyses_ne30l47", "physics=speedy", "dycore.nx=3",
                    "forcing.dms_file=/nonexistent.nc"])
        model = build_model(cfg)
        with self.assertRaises((FileNotFoundError, OSError, ValueError)):
            build_forcing(cfg, model.coords, dycore=model.dycore)

    @pytest.mark.requires_extra("pyses", "mam4")
    def test_ma_ne30_configurations_construct_on_pyses(self):
        """The ma-ne30 configuration presets compose and build on pySES.

        echam-jam's TiedtkeConvection declares an ``omega`` requirement
        whenever the mid-level trigger is on (its default); pySES exposes
        no omega provider (#698), so with the trigger on Model construction
        raises. Both configuration YAMLs set ``physics.cu_lmfmid=false`` — the
        launch blocker tracked in #715 — so the composed physics declares no
        omega requirement and the Model builds. Uses a test-size grid; the
        canonical files document ne30.
        """
        from jcm.runners import build_model

        for name in ("ma-ne30-l47", "ma-ne30-l95"):
            cfg = _cfg([f"+configuration={name}", "dycore.nx=3",
                        "dycore.n_sponge=8"])
            model = build_model(cfg)
            self.assertNotIn("omega", model.physics.required_dycore_fields())


if __name__ == "__main__":
    unittest.main()
