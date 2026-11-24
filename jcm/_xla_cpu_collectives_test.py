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
# before JAX initialises (same mechanism as the other sharding tests).
# Collectives that are identical in primitive, shape and parameters are the
# case that matters: jax's lowering cache would give them one shared id.
_LOWER_SCRIPT = textwrap.dedent("""
    import re
    import jax, jax.numpy as jnp
    import jax._src.lax.parallel as parallel
    from jax.sharding import PartitionSpec as P
    from dinosaur.sigma_coordinates import SigmaCoordinates
    from jcm import _xla_cpu_collectives as xcc
    from jcm.utils import get_coords

    # get_coords installs the workaround for a multi-device CPU mesh.
    assert not isinstance(parallel.mlir, xcc._UniqueChannelMlir)
    get_coords(SigmaCoordinates.equidistant(8), spectral_truncation=31,
               spmd_mesh=(2, 1, 1))
    assert isinstance(parallel.mlir, xcc._UniqueChannelMlir)
    assert xcc.ensure_unique_collective_channel_ids()  # idempotent

    mesh = jax.make_mesh((2,), ("x",), axis_types=(jax.sharding.AxisType.Auto,))
    perm = [(0, 1), (1, 0)]

    def ids_of(fn):
        text = jax.jit(jax.shard_map(
            fn, mesh=mesh, in_specs=P("x"), out_specs=P("x"),
        )).lower(jnp.arange(4.0)).as_text()
        return [int(i) for i in re.findall(r"channel_handle<handle = (\\d+)", text)]

    def f(a):
        b = jax.lax.ppermute(a, "x", perm) + jax.lax.ppermute(a, "x", perm)
        c = jax.lax.psum(a, "x") + jax.lax.psum(a, "x")
        return b + c

    def g(a):
        return jax.lax.ppermute(a, "x", perm) * jax.lax.ppermute(a, "x", perm)

    first = ids_of(f)
    print("IDS", first)
    assert len(first) >= 4, first
    assert len(set(first)) == len(first), first
    assert min(first) >= 1 << 16, first
    # Never reused across modules: two programs can execute at the same time.
    second = ids_of(g)
    assert len(second) >= 2 and not set(second) & set(first), (first, second)
""")


class UniqueChannelIdTest(unittest.TestCase):

    def setUp(self):
        # The proxy is process-global and another in-process sharding test may
        # have installed it through get_coords; start from the real module and
        # put back whatever was there.
        installed = parallel.mlir
        self.addCleanup(setattr, parallel, "mlir", installed)
        if isinstance(installed, xcc._UniqueChannelMlir):
            parallel.mlir = installed._real

    def test_identical_collectives_get_distinct_channel_ids(self):
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

    def test_get_coords_leaves_single_device_runs_alone(self):
        # The multi-device branch needs >1 CPU device and is asserted in the
        # fresh interpreter above.
        from dinosaur.sigma_coordinates import SigmaCoordinates

        from jcm.utils import get_coords

        vert = SigmaCoordinates.equidistant(8)
        get_coords(vert, spectral_truncation=31)
        self.assertNotIsInstance(parallel.mlir, xcc._UniqueChannelMlir)
        get_coords(vert, spectral_truncation=31, spmd_mesh=(1, 1, 1))
        self.assertNotIsInstance(parallel.mlir, xcc._UniqueChannelMlir)

    def test_collective_primitives_lower_uncached(self):
        from jax._src.interpreters import mlir

        self.assertTrue(xcc.ensure_unique_collective_channel_ids())
        for prim in (parallel.ppermute_p, parallel.psum_p, parallel.all_gather_p):
            self.assertIn(prim, mlir._uncacheable_primitives)


if __name__ == "__main__":
    unittest.main()
