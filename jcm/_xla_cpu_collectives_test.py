"""Tests for the unique collective channel-id workaround."""

import logging
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import jax._src.lax.parallel as parallel

from jcm import _xla_cpu_collectives as xcc

# Lowered in a fresh interpreter because >1 CPU device must be requested
# before JAX initialises (same mechanism as the other sharding tests). Two
# different collectives in one jitted shard_map must carry different channel
# ids in the StableHLO text.
_LOWER_SCRIPT = textwrap.dedent("""
    import re
    import jax, jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from jcm._xla_cpu_collectives import ensure_unique_collective_channel_ids

    assert ensure_unique_collective_channel_ids()
    assert ensure_unique_collective_channel_ids()  # idempotent

    mesh = jax.make_mesh((2,), ("x",))

    @jax.jit
    @jax.shard_map(mesh=mesh, in_specs=P("x"), out_specs=P("x"))
    def f(a):
        b = jax.lax.ppermute(a, "x", [(0, 1), (1, 0)])
        c = jax.lax.psum(a, "x")
        return b + c

    text = f.lower(jnp.arange(4.0)).as_text()
    ids = re.findall(r"channel_handle<handle = (\\d+)", text)
    print("IDS", ids)
    assert len(ids) >= 2, text
    assert len(set(ids)) == len(ids), ids
    assert all(int(i) >= 1 << 16 for i in ids), ids
""")


class UniqueChannelIdTest(unittest.TestCase):

    def test_collectives_get_distinct_channel_ids(self):
        repo_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
        env["JAX_PLATFORMS"] = "cpu"
        proc = subprocess.run(
            [sys.executable, "-c", _LOWER_SCRIPT],
            env=env, cwd=repo_root, capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

    def test_idempotent(self):
        original = parallel.mlir
        try:
            self.assertTrue(xcc.ensure_unique_collective_channel_ids())
            installed = parallel.mlir
            self.assertTrue(xcc.ensure_unique_collective_channel_ids())
            self.assertIs(parallel.mlir, installed)
            first = installed.COLLECTIVE_CHANNEL_ID
            self.assertEqual(installed.COLLECTIVE_CHANNEL_ID, first + 1)
            self.assertIs(installed.ir, installed._real.ir)  # forwarding
        finally:
            parallel.mlir = original

    def test_noop_with_warning_when_jax_internals_move(self):
        original = parallel.mlir

        class _NoChannel:
            pass

        parallel.mlir = _NoChannel()
        try:
            with self.assertLogs(xcc.logger, level=logging.WARNING) as cm:
                self.assertFalse(xcc.ensure_unique_collective_channel_ids())
            self.assertEqual(len(cm.output), 1)
            self.assertIn("rendezvous", cm.output[0])
            self.assertIsInstance(parallel.mlir, _NoChannel)
        finally:
            parallel.mlir = original

    def test_get_coords_installs_only_for_multi_device_cpu_mesh(self):
        from jcm.utils import get_coords
        import jax
        from dinosaur.sigma_coordinates import SigmaCoordinates

        original = parallel.mlir
        try:
            vert = SigmaCoordinates.equidistant(8)
            get_coords(vert, spectral_truncation=31)
            self.assertNotIsInstance(parallel.mlir, xcc._UniqueChannelMlir)
            get_coords(vert, spectral_truncation=31, spmd_mesh=(1, 1, 1))
            self.assertNotIsInstance(parallel.mlir, xcc._UniqueChannelMlir)
            if jax.device_count() >= 2:
                get_coords(vert, spectral_truncation=31,
                           spmd_mesh=(jax.device_count(), 1, 1))
                self.assertIsInstance(parallel.mlir, xcc._UniqueChannelMlir)
        finally:
            parallel.mlir = original


if __name__ == "__main__":
    unittest.main()
