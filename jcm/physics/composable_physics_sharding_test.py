"""Sharding of the column-physics path (option B: longitude-only mesh).

These tests cover the SPMD wiring added for multi-device (and multi-CPU)
runs: the ``vectorize_columns`` path must keep the flattened ``(nlev, ncols)``
state sharded across devices, produce results identical to a single-device
run, and merge the mesh axes in lon-major order when building the column
sharding. See docs/design/parallelization.md.

The physics compute path is purely gridpoint ops + per-column vmaps (no
spectral transform), so it can be exercised on real devices in isolation,
independent of the dycore's spectral-transform sharding.
"""

import os

# Expose 2 CPU devices so the multi-device cases have something to shard over.
# This must run before ``import jax``; if another test in the same process
# already initialised the backend with a single device, the multi-device
# cases below self-skip.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import subprocess  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import ClassVar  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpy.testing as npt  # noqa: E402
from flax import nnx  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402
from dinosaur.sigma_coordinates import SigmaCoordinates  # noqa: E402

from jcm.physics.composable_physics import (  # noqa: E402
    ComposablePhysics,
    _flattened_column_sharding,
)
from jcm.physics.physics_term import PhysicsTerm  # noqa: E402
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.forcing import ForcingData  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402
from jcm.utils import get_coords  # noqa: E402


# ---------------------------------------------------------------------------
# Toy layout-agnostic terms (work on (nlev, ncols) column state)
# ---------------------------------------------------------------------------

class LinearHeating(PhysicsTerm):
    """Toy term: temperature tendency proportional to temperature."""

    name: ClassVar[str] = "linear_heating"
    category: ClassVar[str] = "radiation"

    def __init__(self, alpha: float = 0.01):
        """Initialize LinearHeating."""
        self.alpha = nnx.Param(jnp.array(alpha))

    def __call__(self, state, diagnostics, forcing, terrain):
        heating = self.alpha[...] * state.temperature
        tend = PhysicsTendency.zeros(state.temperature.shape).copy(
            temperature=heating,
        )
        return tend, diagnostics


class QuadraticMoistening(PhysicsTerm):
    """Toy term: humidity tendency proportional to temperature squared."""

    name: ClassVar[str] = "quadratic_moistening"
    category: ClassVar[str] = "convection"

    def __init__(self, beta: float = 1e-6):
        """Initialize QuadraticMoistening."""
        self.beta = nnx.Param(jnp.array(beta))

    def __call__(self, state, diagnostics, forcing, terrain):
        source = self.beta[...] * state.temperature ** 2
        tend = PhysicsTendency.zeros(state.temperature.shape).copy(
            specific_humidity=source,
        )
        return tend, diagnostics


# T21 L8: nodal grid is 64 lon x 32 lat.
_NLEV, _NLON, _NLAT = 8, 64, 32


def _make_state(shape):
    """Build a well-conditioned random PhysicsState of shape (nlev, nlon, nlat)."""
    keys = jax.random.split(jax.random.PRNGKey(0), 6)
    surf_shape = shape[1:]
    return PhysicsState(
        u_wind=jax.random.normal(keys[0], shape),
        v_wind=jax.random.normal(keys[1], shape),
        temperature=250.0 + 50.0 * jax.random.uniform(keys[2], shape),
        specific_humidity=1e-3 * jax.random.uniform(keys[3], shape),
        geopotential=jax.random.uniform(keys[4], shape),
        normalized_surface_pressure=1.0 + 0.01 * jax.random.uniform(keys[5], surf_shape),
        tracers={},
    )


def _column_physics(coords):
    phys = ComposablePhysics(
        [LinearHeating(), QuadraticMoistening()],
        vectorize_columns=True,
        checkpoint_terms=False,
    )
    phys.cache_coords(coords)
    return phys


class FlattenedColumnShardingSpecTest(unittest.TestCase):
    """Pure spec-derivation — no real multi-device topology needed."""

    def test_returns_none_without_mesh(self):
        coords = get_coords(
            SigmaCoordinates.equidistant(_NLEV),
            spectral_truncation=21,
            spmd_mesh=None,
        )
        self.assertEqual(_flattened_column_sharding(coords), (None, None))
        # Also for an object that simply has no spmd_mesh attribute set.
        self.assertEqual(
            _flattened_column_sharding(types.SimpleNamespace(spmd_mesh=None)),
            (None, None),
        )

    def test_merges_lon_axes_major(self):
        # Build a mesh from whatever devices exist, shaped (D, 1, 1) so the
        # derivation is exercised regardless of how many devices the worker
        # happens to see.
        devices = jax.devices()
        mesh = jax.sharding.Mesh(
            np.asarray(devices).reshape(len(devices), 1, 1),
            ("x", "y", "z"),
        )
        fake_coords = types.SimpleNamespace(
            spmd_mesh=mesh,
            physics_partition_spec=P(None, ("x", "z"), "y"),
        )
        state_sharding, surface_sharding = _flattened_column_sharding(fake_coords)
        # lon axes ('x', 'z') precede the lat axis ('y') in the merged column
        # axis, matching the lon-major (nlon, nlat) -> ncols reshape.
        self.assertEqual(state_sharding.spec, P(None, ("x", "z", "y")))
        self.assertEqual(surface_sharding.spec, P(("x", "z", "y")))


class ColumnShardingRunTest(unittest.TestCase):
    """End-to-end column-physics behaviour on >= 2 real devices."""

    def setUp(self):
        if jax.device_count() < 2:
            self.skipTest(
                "needs >= 2 devices; run standalone with "
                "XLA_FLAGS=--xla_force_host_platform_device_count=2"
            )
        self.coords_sharded = get_coords(
            SigmaCoordinates.equidistant(_NLEV),
            spectral_truncation=21,
            spmd_mesh=(jax.device_count(), 1, 1),
        )
        self.coords_single = get_coords(
            SigmaCoordinates.equidistant(_NLEV),
            spectral_truncation=21,
            spmd_mesh=None,
        )
        self.state = _make_state((_NLEV, _NLON, _NLAT))
        self.forcing = ForcingData.zeros(self.coords_sharded.horizontal.nodal_shape)
        self.terrain = TerrainData.aquaplanet(self.coords_sharded)

    def test_columns_match_single_device_and_stay_sharded(self):
        phys_sharded = _column_physics(self.coords_sharded)
        phys_single = _column_physics(self.coords_single)

        @jax.jit
        def run_sharded(state):
            tend, _ = phys_sharded.compute_tendencies(
                state, self.forcing, self.terrain,
            )
            return tend

        tend_sharded = run_sharded(self.state)
        tend_single, _ = phys_single.compute_tendencies(
            self.state, self.forcing, self.terrain,
        )

        # Correctness: sharding must not change the answer.
        npt.assert_allclose(
            np.asarray(tend_sharded.temperature),
            np.asarray(tend_single.temperature),
            rtol=1e-6, atol=1e-6,
        )
        npt.assert_allclose(
            np.asarray(tend_sharded.specific_humidity),
            np.asarray(tend_single.specific_humidity),
            rtol=1e-6, atol=1e-6,
        )

        # The constraint must actually take effect: the output is distributed
        # across the mesh rather than gathered onto one device.
        self.assertEqual(tend_sharded.temperature.sharding.mesh.size, jax.device_count())
        self.assertFalse(tend_sharded.temperature.sharding.is_fully_replicated)

    def test_with_physics_sharding_distributes_state(self):
        @jax.jit
        def shard(state):
            return self.coords_sharded.with_physics_sharding(state)

        out = shard(self.state)
        # Level axis replicated, longitude carries the split.
        self.assertFalse(out.temperature.sharding.is_fully_replicated)
        self.assertEqual(out.temperature.sharding.mesh.size, jax.device_count())


class MultiDeviceSubprocessTest(unittest.TestCase):
    """Drive the device-dependent checks on 2 CPU devices.

    Under the default suite JAX is already initialised with a single CPU
    device (test plugins import it before this module's top-level
    ``XLA_FLAGS`` setdefault runs), so :class:`ColumnShardingRunTest`
    self-skips. Here we re-run just that class in a fresh interpreter with
    the device count raised *from process start*, which is the only reliable
    way to get >1 CPU device. This is the test that actually exercises
    cross-device column physics in CI.
    """

    def test_column_sharding_on_two_cpu_devices(self):
        repo_root = Path(__file__).resolve().parents[2]
        env = dict(os.environ)
        env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
        env["JAX_PLATFORMS"] = "cpu"
        proc = subprocess.run(
            [
                sys.executable, "-m", "unittest", "-v",
                "jcm.physics.composable_physics_sharding_test.ColumnShardingRunTest",
            ],
            env=env, cwd=repo_root, capture_output=True, text=True,
        )
        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, msg=output)
        # Guard against a false pass: the child must have actually had 2
        # devices and run the checks, not skipped them.
        self.assertNotIn("skipped", output.lower(), msg=output)
        self.assertIn("ok", output.lower(), msg=output)


if __name__ == "__main__":
    unittest.main()
