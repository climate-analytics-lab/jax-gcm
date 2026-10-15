"""Multi-device sharding regression for the opt-in progress bar (#320).

``jax.debug.callback(..., ordered=True)`` raises ``NotImplementedError``
("ordered effects are not supported for more than 1 device") the moment a
model with ``progress=True`` runs under a multi-device ``NamedSharding`` —
exactly the configuration a progress bar matters most for (a multi-day run
split across devices is the one worth watching). ``jcm.progress.set_position``
is the unordered, monotone-position fix (see ``jcm/progress.py``'s module
docstring and the Codex review on #320 it responds to); this regression
pins that a ``progress=True`` run actually COMPILES under a real
multi-device mesh and that the bar reaches the full count, for both output
modes (``output_averages`` True and False each take a different
``_op_split_trajectory`` code path with its own callback site).

The suite's default worker has a single CPU device, so the device-dependent
case self-skips there and is driven on 2 CPU devices via a subprocess (the
env var must be set before JAX initialises) — same pattern as
``jcm/dycore/dinosaur/sharding_test.py`` and
``jcm/physics/composable_physics_sharding_test.py``.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import contextlib  # noqa: E402
import io  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest import mock  # noqa: E402

import jax  # noqa: E402


def _run_sharded(mesh, output_averages):
    """Run a short, sharded Held-Suarez integration with progress=True.

    Forces the fallback printer (rather than whatever ``tqdm`` happens to
    be installed) so the final line's ``n/total`` is deterministic text to
    assert on, matching ``jcm/runners_test.py::TestChunkedProgressBar``'s
    technique. Returns the captured stderr text.
    """
    from jcm.model import Model
    from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
    from jcm.physics.held_suarez.utils import get_held_suarez_coords
    from jcm.terrain import TerrainData
    from jcm import progress

    # Default truncation (31): the same resolution
    # jcm/dycore/dinosaur/sharding_test.py already validates divides evenly
    # across a 2-device longitude-only mesh, so a shape mismatch here would
    # be this test's own setup, not the progress-bar fix under test.
    coords = get_held_suarez_coords(spmd_mesh=mesh)
    terrain = TerrainData.from_coords(coords)
    model = Model(coords=coords, terrain=terrain, time_step=180,
                 physics=held_suarez_physics())
    captured = io.StringIO()
    with mock.patch.object(progress, "_tqdm", None), \
            contextlib.redirect_stderr(captured):
        model.run(save_interval='1 day', total_time='2 days',
                 output_averages=output_averages, progress=True)
    return captured.getvalue()


class ProgressBarShardingRunTest(unittest.TestCase):
    """Full-model ``progress=True`` run on >= 2 real devices."""

    def setUp(self):
        if jax.device_count() < 2:
            self.skipTest(
                "needs >= 2 devices; run standalone with "
                "XLA_FLAGS=--xla_force_host_platform_device_count=2"
            )

    def test_progress_true_compiles_and_reaches_final_count_averaged(self):
        n = jax.device_count()
        output = _run_sharded((n, 1, 1), output_averages=True)
        self.assertIn("jcm: 2/2", output)

    def test_progress_true_compiles_and_reaches_final_count_snapshot(self):
        n = jax.device_count()
        output = _run_sharded((n, 1, 1), output_averages=False)
        self.assertIn("jcm: 2/2", output)


class MultiDeviceSubprocessTest(unittest.TestCase):
    """Drive the sharded ``progress=True`` run on 2 CPU devices in a fresh
    interpreter — the test that actually exercises it in CI, since the
    default suite's single already-initialised CPU device makes
    :class:`ProgressBarShardingRunTest` self-skip there.
    """

    def test_progress_true_on_two_cpu_devices(self):
        repo_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["XLA_FLAGS"] = (
            "--xla_force_host_platform_device_count=2 "
            "--xla_cpu_enable_concurrency_optimized_scheduler=false"
        )
        env["JAX_PLATFORMS"] = "cpu"
        proc = subprocess.run(
            [
                sys.executable, "-m", "unittest", "-v",
                "jcm.progress_sharding_test.ProgressBarShardingRunTest",
            ],
            env=env, cwd=repo_root, capture_output=True, text=True,
        )
        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, msg=output)
        self.assertNotIn("skipped", output.lower(), msg=output)
        self.assertIn("ok", output.lower(), msg=output)


if __name__ == "__main__":
    unittest.main()
