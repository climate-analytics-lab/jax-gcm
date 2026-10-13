"""Full-model SPMD sharding regression for the dinosaur dycore.

A full ``Model.run`` on >1 device must stay finite and reproduce the
single-device answer of the *same* spectral-transform algorithm. The
single-vs-multi comparison is done Fast-vs-Fast (both ``FastSphericalHarmonics``,
differing only in device count) so it isolates sharding correctness from the
``RealSphericalHarmonics`` ↔ ``FastSphericalHarmonics`` algorithm difference.

This is the regression that pins the diffusion-filter fix: under SPMD the modal
axis is zero-padded, so the old ``eigenvalues[-1]`` normaliser read 0 and the
hyperdiffusion filters NaN'd every prognostic field within one step.

The suite's default worker has a single CPU device, so the device-dependent
case self-skips there and is driven on 2 CPU devices via a subprocess (the env
var must be set before JAX initialises).
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import subprocess  # noqa: E402
import sys  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402


def _run_speedy_temperature(mesh, days=0.25):
    """Run a short T31 SPEEDY aquaplanet and return the saved temperature field."""
    from jcm.model import Model
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.physics.speedy.speedy_terms import speedy_physics
    from jcm.terrain import TerrainData
    from jcm.forcing import ForcingData

    coords = get_speedy_coords(layers=8, spectral_truncation=31, spmd_mesh=mesh)
    model = Model(
        coords=coords,
        terrain=TerrainData.aquaplanet(coords),
        physics=speedy_physics(),
        time_step=30.0,
    )
    preds = model.run(
        forcing=ForcingData.zeros(coords.horizontal.nodal_shape),
        save_interval=days,
        total_time=days,
    )
    return np.asarray(preds.to_xarray().temperature)


class DinosaurShardingRunTest(unittest.TestCase):
    """Full-model run on >= 2 real devices."""

    def setUp(self):
        if jax.device_count() < 2:
            self.skipTest(
                "needs >= 2 devices; run standalone with "
                "XLA_FLAGS=--xla_force_host_platform_device_count=2"
            )

    def test_full_model_sharded_is_finite_and_matches_single_device(self):
        n = jax.device_count()
        # Both runs use FastSphericalHarmonics (any non-None mesh): the only
        # difference is the device split, so they must agree tightly. A 2-device
        # mesh divides T31's 96 longitudes evenly, so the grid is not padded.
        t_one = _run_speedy_temperature((1, 1, 1))
        t_many = _run_speedy_temperature((n, 1, 1))

        self.assertTrue(np.isfinite(t_many).all(), "sharded run produced NaN/Inf")
        self.assertEqual(t_one.shape, t_many.shape, "grid was padded — mesh must divide it")
        rel = np.nanmax(np.abs(t_one - t_many)) / (np.nanmax(np.abs(t_one)) + 1e-12)
        self.assertLess(float(rel), 1e-4, f"sharded vs single-device rel diff {rel:.2e}")


class MultiDeviceSubprocessTest(unittest.TestCase):
    """Drive the full-model run on 2 CPU devices in a fresh interpreter.

    The default suite has JAX already initialised with a single device, so
    :class:`DinosaurShardingRunTest` self-skips there. Here we re-run it with
    the device count raised from process start — the test that actually
    exercises a sharded dynamical-core integration in CI.
    """

    def test_full_model_on_two_cpu_devices(self):
        repo_root = Path(__file__).resolve().parents[3]
        env = dict(os.environ)
        # Serialised CPU collectives match the documented multi-CPU recipe and
        # keep complex graphs from over-subscribing the CPU thread rendezvous.
        env["XLA_FLAGS"] = (
            "--xla_force_host_platform_device_count=2 "
            "--xla_cpu_enable_concurrency_optimized_scheduler=false"
        )
        env["JAX_PLATFORMS"] = "cpu"
        proc = subprocess.run(
            [
                sys.executable, "-m", "unittest", "-v",
                "jcm.dycore.dinosaur.sharding_test.DinosaurShardingRunTest",
            ],
            env=env, cwd=repo_root, capture_output=True, text=True,
        )
        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, msg=output)
        self.assertNotIn("skipped", output.lower(), msg=output)
        self.assertIn("ok", output.lower(), msg=output)


if __name__ == "__main__":
    unittest.main()
