"""Multi-device sharding regression for the opt-in progress bar (#320).

Pins that a ``progress=True`` run actually COMPILES under a real
multi-device mesh and completes, for both output modes (``output_averages``
True and False each take a different ``_op_split_trajectory`` code path).
Earlier versions of this test also asserted an in-jit per-saved-interval
tick count; that reporting path no longer exists (three rounds of Codex
review each found an in-jit ``jax.debug.callback`` broken a different
way — ordered effects fail to compile under a multi-device
``NamedSharding`` in the first place, which is what this regression still
pins; see ``jcm/progress.py``'s module docstring for the full history).
Progress is host-only now, so the only sharding-specific risk left is
compilation itself.

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

    def test_progress_true_compiles_and_completes_averaged(self):
        n = jax.device_count()
        # The call must both compile (the regression this test exists for)
        # and complete, reported host-only as a single unit -- no
        # per-saved-interval granularity any more (#320).
        output = _run_sharded((n, 1, 1), output_averages=True)
        self.assertIn("jcm chunks: 1/1", output)

    def test_progress_true_compiles_and_completes_snapshot(self):
        n = jax.device_count()
        output = _run_sharded((n, 1, 1), output_averages=False)
        self.assertIn("jcm chunks: 1/1", output)


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
