"""Column blocks of the RRTMGP solve: layout, block resolution, agreement.

``RRTMGPRadiation`` solves its columns as one ``jax.vmap`` or as a ``lax.map``
over column blocks (``column_block``, ``JCM_RRTMGP_COL_CHUNKS``). The blocks
must (a) reproduce the single vmap to float32 rounding, (b) tile the columns
exactly, and (c) under a sharded column axis take every step's columns from
EVERY device -- mapping contiguous blocks of a sharded axis runs the devices one
after another (8x slower, measured; ``docs/source/design/
rrtmgp_cpu_performance.md``).

The 2-device cases need >= 2 CPU devices; the suite's workers have one, so they
self-skip there and ``TwoDeviceSubprocessTest`` drives them in a fresh
interpreter with the device count raised from process start (the same
mechanism as ``composable_physics_sharding_test``).
"""

import os

# Must run before ``import jax``; if the backend is already initialised with one
# device the multi-device cases below self-skip (and the subprocess drives them).
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import subprocess  # noqa: E402
import sys  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from flax import nnx  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from jcm.physics.radiation.rrtmgp import (  # noqa: E402
    CPU_COLUMN_BLOCK,
    RRTMGPRadiation,
    _check_column_block,
    _maybe_chunked_vmap,
    _merge_columns,
    _split_columns,
    resolve_column_block,
)
from jcm.physics.radiation.rrtmgp_test import _RRTMGPTermFixture  # noqa: E402
from jcm.testing import check_gradients  # noqa: E402


# --------------------------------------------------------------------------
# Block resolution
# --------------------------------------------------------------------------

class TestResolveColumnBlock:
    """``resolve_column_block``: auto / int / None and the divisor rule."""

    def test_auto_is_the_cpu_block_on_cpu_and_off_elsewhere(self):
        assert CPU_COLUMN_BLOCK == 8
        assert resolve_column_block("auto", 768, 1, on_cpu=True) == 8
        assert resolve_column_block("auto", 768, 1, on_cpu=False) is None

    def test_none_is_the_single_vmap_everywhere(self):
        for on_cpu in (True, False):
            assert resolve_column_block(None, 768, 1, on_cpu=on_cpu) is None

    def test_explicit_integer_is_honoured_on_any_backend(self):
        assert resolve_column_block(16, 768, 1, on_cpu=False) == 16

    def test_block_is_the_largest_divisor_of_the_per_device_count(self):
        # 18432 columns / 24 devices = 768 per device: 8 divides it.
        assert resolve_column_block(8, 18432, 24) == 8
        assert resolve_column_block(8, 100, 1) == 5     # 100 = 5 x 20
        assert resolve_column_block(8, 12, 1) == 6
        assert resolve_column_block(4, 13, 1) == 1      # prime: one column

    def test_the_divisor_is_taken_of_the_per_device_count(self):
        # 96 columns over 4 devices = 24 per device: 16 -> 12, not 16 (which
        # divides 96 but is not a tiling of what one device holds).
        assert resolve_column_block(16, 96, 4) == 12

    def test_whole_per_device_count_is_the_single_vmap(self):
        assert resolve_column_block(8, 8, 1) is None
        assert resolve_column_block(8, 5, 1) is None
        assert resolve_column_block(8, 16, 2) is None   # 8 per device
        assert resolve_column_block(8, 35, 5) is None   # 7 per device
        assert resolve_column_block(100, 12, 1) is None
        # A single column (single-column model) never scans.
        assert resolve_column_block("auto", 1, 1, on_cpu=True) is None

    def test_shards_that_do_not_divide_the_columns_count_as_one(self):
        assert resolve_column_block(4, 18, 4) == 3

    def test_invalid_options_are_rejected_at_construction(self):
        for bad in (0, -4, 2.5, True, "eight", "AUTO"):
            with pytest.raises(ValueError, match="column_block"):
                _check_column_block(bad)
        for good in (None, "auto", 1, 8, np.int64(8)):
            assert _check_column_block(good) == good
        assert RRTMGPRadiation()._column_block == "auto"
        with pytest.raises(ValueError, match="column_block"):
            RRTMGPRadiation(column_block=0)


# --------------------------------------------------------------------------
# Layout helpers and the column map (toy function: cheap)
# --------------------------------------------------------------------------

def _toy(x, y, scale):
    """Per-column toy: a ``(tend, diag)`` pair like the real scheme's."""
    return x * scale + y.sum(), (x.sum() * scale, y * 2.0)


_IN_AXES = (0, 0, None)


def _single_vmap(x, y):
    return jax.vmap(_toy, in_axes=_IN_AXES, out_axes=(0, 0))(x, y, 3.0)


class TestColumnLayout:
    """``_split_columns`` / ``_merge_columns`` and ``_maybe_chunked_vmap``."""

    def test_unsharded_split_is_contiguous_blocks(self):
        x = jnp.arange(12.0).reshape(12, 1)
        split = _split_columns(x, 1, 3)
        assert split.shape == (3, 4, 1)
        np.testing.assert_array_equal(split[1, :, 0], np.arange(4.0, 8.0))

    def test_sharded_split_takes_each_step_from_every_device(self):
        # 12 columns on 2 devices: device 0 owns 0..5, device 1 owns 6..11.
        split = _split_columns(jnp.arange(12.0), 2, 3)     # (steps, shards, B)
        assert split.shape == (3, 2, 2)
        np.testing.assert_array_equal(split[0], [[0, 1], [6, 7]])
        np.testing.assert_array_equal(split[2], [[4, 5], [10, 11]])

    @pytest.mark.parametrize("shards,steps", [(1, 4), (2, 3), (3, 2), (4, 1)])
    def test_merge_inverts_split(self, shards, steps):
        x = jnp.arange(24.0 * 3).reshape(24, 3)
        back = _merge_columns(_split_columns(x, shards, steps), shards, 24)
        np.testing.assert_array_equal(back, x)

    @pytest.mark.parametrize(
        "block,shards", [(None, 1), (3, 1), (4, 1), (2, 2), (3, 2), (6, 2)])
    def test_blocked_map_matches_the_single_vmap(self, block, shards):
        n = 12
        x = jnp.linspace(0.0, 1.0, n * 4).reshape(n, 4)
        y = jnp.linspace(1.0, 2.0, n)
        out = _maybe_chunked_vmap(_toy, _IN_AXES, column_block=block,
                                  n_shards=shards)(x, y, 3.0)
        # Row sums reassociate between programs: float32 rounding, no more.
        jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-6),
                     out, _single_vmap(x, y))

    def test_none_is_exactly_the_single_vmap_program(self):
        x, y = jnp.ones((12, 4)), jnp.ones(12)
        plain = jax.make_jaxpr(_single_vmap)(x, y)
        none = jax.make_jaxpr(lambda a, b: _maybe_chunked_vmap(
            _toy, _IN_AXES, column_block=None)(a, b, 3.0))(x, y)
        blocked = jax.make_jaxpr(lambda a, b: _maybe_chunked_vmap(
            _toy, _IN_AXES, column_block=4)(a, b, 3.0))(x, y)
        assert str(none) == str(plain)
        assert "scan" not in str(none)
        assert "scan" in str(blocked)

    def test_auto_follows_the_backend(self):
        x, y = jnp.ones((32, 4)), jnp.ones(32)

        def program(on_cpu):
            return str(jax.make_jaxpr(lambda a, b: _maybe_chunked_vmap(
                _toy, _IN_AXES, column_block="auto", on_cpu=on_cpu)(a, b, 3.0)
            )(x, y))
        assert "scan" in program(True)
        assert "scan" not in program(False)

    def test_col_chunks_wins_and_is_rematerialised(self, monkeypatch):
        monkeypatch.setenv("JCM_RRTMGP_COL_CHUNKS", "4")
        x, y = jnp.ones((16, 4)), jnp.ones(16)
        # ``column_block=None`` alone would be one vmap: the env var wins, and
        # its steps are checkpointed.
        program = str(jax.make_jaxpr(lambda a, b: _maybe_chunked_vmap(
            _toy, _IN_AXES, column_block=None)(a, b, 3.0))(x, y))
        assert "scan" in program
        assert "checkpoint" in program or "remat" in program
        out = _maybe_chunked_vmap(_toy, _IN_AXES, column_block=8)(x, y, 3.0)
        jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-6),
                     out, _single_vmap(x, y))
        # Columns the chunk count does not divide.
        with pytest.raises(ValueError, match="not divisible"):
            _maybe_chunked_vmap(_toy, _IN_AXES)(jnp.ones((18, 4)), jnp.ones(18), 3.0)

    def test_blocked_gradient_matches_the_single_vmap(self):
        x = jnp.linspace(0.1, 1.0, 12 * 4).reshape(12, 4)
        y = jnp.linspace(1.0, 2.0, 12)

        def total(block, xx):
            tend, (s, _) = _maybe_chunked_vmap(
                _toy, _IN_AXES, column_block=block)(xx, y, 3.0)
            return (tend ** 2).sum() + s.sum()

        g_ref = jax.grad(lambda xx: total(None, xx))(x)
        g_blk = jax.grad(lambda xx: total(4, xx))(x)
        np.testing.assert_allclose(g_blk, g_ref, rtol=1e-6)


# --------------------------------------------------------------------------
# The real scheme, through the term
# --------------------------------------------------------------------------

class _BlockFixture(_RRTMGPTermFixture):
    """Columns that differ: temperature, cloud, surface, latitude, longitude."""

    NLEV = 8
    NCOLS = 6
    COMPUTE_CRE = True

    def _build(self, column_block, n_shards=1):
        base, state, diag, forcing = self._term_and_inputs()
        diag, state = self._seed_aerosol_and_cloud(diag, state)
        n = self.NCOLS
        ramp = jnp.linspace(-1.0, 1.0, n)
        # The per-g-point, unrolled placement: the column layout does not
        # depend on it, and XLA's CPU compile of the production placement costs
        # ~11 GiB and minutes (``term_gradients_test._LIGHT_RADIATION``).
        term = RRTMGPRadiation(
            params=base.params.get_value(), compute_cre=True,
            column_block=column_block, gas_optics_mode="per_gpt",
            gpt_chunk=1, use_scan=False)
        term._column_shards = n_shards
        term._lats = nnx.Variable(jnp.linspace(-60.0, 60.0, n))
        term._lons = nnx.Variable(jnp.linspace(0.0, 300.0, n))
        state = state.copy(
            temperature=state.temperature + 6.0 * ramp[None, :],
            tracers={**state.tracers,
                     "qc": state.tracers["qc"] * (1.5 + ramp[None, :])})
        diag = {**diag, "clouds": diag["clouds"].copy(
            cloud_fraction=jnp.clip(
                diag["clouds"].cloud_fraction * (1.0 + 0.8 * ramp[None, :]),
                0.0, 1.0)),
            "surface": diag["surface"].copy(
                surface_temperature=288.0 + 8.0 * ramp)}
        return term, state, diag, forcing

    def _run(self, column_block, **kw):
        term, state, diag, forcing = self._build(column_block, **kw)
        tend, out = term(state, diag, forcing, None)
        return tend, out["radiation"]


_FIELDS = ("lw_heating_rate", "sw_heating_rate", "toa_lw_up", "toa_sw_up",
           "toa_lw_up_clear", "toa_sw_up_clear")


def _assert_same_solve(tend, rad, tend_ref, rad_ref, **tol):
    tol = {"rtol": 1e-4, "atol": 2e-3, **tol}
    np.testing.assert_allclose(
        np.asarray(tend.temperature), np.asarray(tend_ref.temperature),
        rtol=1e-4, atol=1e-9)
    for name in _FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(rad, name)), np.asarray(getattr(rad_ref, name)),
            err_msg=name, **tol)


class TestBlockedSolveAgreesWithSingleVmap(_BlockFixture):
    """Blocked and unblocked solves agree to float32 rounding on a real batch."""

    @pytest.fixture(scope="class")
    def reference(self):
        tend, rad = self._run(None)
        # Anti-vacuity: the columns differ and the sun is up somewhere.
        assert np.ptp(np.asarray(rad.toa_lw_up)) > 1.0
        assert np.asarray(rad.toa_sw_up).max() > 1.0
        return tend, rad

    @pytest.mark.parametrize("block", [
        3,      # divides the 6 columns: 2 steps of 3
        4,      # does not: lowered to its largest divisor <= 4, which is 3
        1,      # one column per step
    ])
    def test_blocked_matches_unblocked(self, reference, block):
        _assert_same_solve(*self._run(block), *reference)

    @pytest.mark.parametrize("block", [None, "auto"])
    def test_whole_batch_blocks_are_the_unblocked_program(self, reference, block):
        # ``auto`` is 8 on CPU and the batch is 6 columns: nothing to scan.
        tend, rad = self._run(block)
        np.testing.assert_array_equal(
            np.asarray(tend.temperature), np.asarray(reference[0].temperature))
        np.testing.assert_array_equal(
            np.asarray(rad.toa_lw_up), np.asarray(reference[1].toa_lw_up))

    @pytest.mark.slow
    def test_gradient_through_the_blocked_path(self):
        """``lax.map`` changes the reverse pass; check it on a tiny batch.

        Slow (~3.5 min): reverse-mode through the real scheme is XLA's CPU
        backward compile, which ``term_gradients_test`` documents; even the
        light placement and 6 columns of 8 levels cannot be made cheaper
        without testing a different invariant than the blocked reverse pass.
        """
        blocked, state, diag, forcing = self._build(3)
        single, *_ = self._build(None)

        def heating(term):
            return lambda t: term(
                state.copy(temperature=t), diag, forcing, None)[0].temperature

        check_gradients(heating(blocked), (state.temperature,),
                        rtol=5e-2, atol=1e-3)
        w = jnp.cos(jnp.arange(state.temperature.size, dtype=jnp.float32)
                    ).reshape(state.temperature.shape)
        g_blk = jax.grad(lambda t: (heating(blocked)(t) * w).sum())(state.temperature)
        g_one = jax.grad(lambda t: (heating(single)(t) * w).sum())(state.temperature)
        assert np.all(np.isfinite(np.asarray(g_blk)))
        np.testing.assert_allclose(
            np.asarray(g_blk), np.asarray(g_one), rtol=1e-3,
            atol=1e-3 * float(np.abs(g_one).max()))


# --------------------------------------------------------------------------
# Two devices
# --------------------------------------------------------------------------

class TwoDeviceColumnBlockTest(_BlockFixture, unittest.TestCase):
    """Strided blocks over a column axis sharded across 2 CPU devices."""

    NCOLS = 8

    def setUp(self):
        if jax.device_count() < 2:
            self.skipTest("needs >= 2 devices; see TwoDeviceSubprocessTest")
        self.mesh = Mesh(np.array(jax.devices()[:2]), ("c",))
        self.col = NamedSharding(self.mesh, P("c"))

    def test_every_step_takes_columns_from_both_devices(self):
        seen = {}

        @jax.jit
        def split(x):
            y = _split_columns(x, 2, 2)
            jax.debug.inspect_array_sharding(
                y, callback=lambda s: seen.update(spec=tuple(s.spec)))
            return y

        out = split(jax.device_put(jnp.arange(16.0), self.col))
        jax.effects_barrier()
        # (steps, shards, B): the shard axis, not the scan axis, is the split one.
        self.assertEqual(seen["spec"], (None, "c"))
        self.assertEqual(len(out[0].sharding.device_set), 2)

    def test_strided_map_matches_and_needs_no_cross_device_traffic(self):
        n = 16
        x = jax.device_put(jnp.linspace(0.0, 1.0, n * 4).reshape(n, 4), self.col)
        y = jax.device_put(jnp.linspace(1.0, 2.0, n), self.col)
        fn = jax.jit(lambda a, b: _maybe_chunked_vmap(
            _toy, _IN_AXES, column_block=2, n_shards=2)(a, b, 3.0),
            out_shardings=self.col)
        jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-6),
                     fn(x, y), _single_vmap(x, y))
        hlo = fn.lower(x, y).compile().as_text()
        for collective in ("all-gather", "all-to-all", "collective-permute",
                           "all-reduce"):
            self.assertNotIn(collective, hlo)

    def test_col_chunks_are_strided_across_the_devices_too(self):
        """The memory path keeps its checkpointed steps but spreads each over both."""
        os.environ["JCM_RRTMGP_COL_CHUNKS"] = "4"
        self.addCleanup(os.environ.pop, "JCM_RRTMGP_COL_CHUNKS")
        n = 16
        x = jax.device_put(jnp.linspace(0.0, 1.0, n * 4).reshape(n, 4), self.col)
        y = jax.device_put(jnp.linspace(1.0, 2.0, n), self.col)
        fn = jax.jit(lambda a, b: _maybe_chunked_vmap(
            _toy, _IN_AXES, n_shards=2)(a, b, 3.0), out_shardings=self.col)
        jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-6),
                     fn(x, y), _single_vmap(x, y))
        hlo = fn.lower(x, y).compile().as_text()
        for collective in ("all-gather", "all-to-all", "collective-permute"):
            self.assertNotIn(collective, hlo)
        # 4 steps over 2 devices: 2 columns per device per step.
        self.assertIn("while", hlo)

    def test_real_solve_sharded_matches_the_single_device_single_vmap(self):
        tend_ref, rad_ref = self._run(None)
        term, state, diag, forcing = self._build(2, n_shards=2)    # 4 per device
        tend, out = jax.jit(
            lambda s, d: term(s, d, forcing, None))(state, diag)
        _assert_same_solve(tend, out["radiation"], tend_ref, rad_ref)


class TwoDeviceSubprocessTest(unittest.TestCase):
    """Drive ``TwoDeviceColumnBlockTest`` in a fresh interpreter with 2 CPUs."""

    def test_column_blocks_on_two_cpu_devices(self):
        repo_root = Path(__file__).resolve().parents[3]
        env = dict(os.environ)
        env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
        env["JAX_PLATFORMS"] = "cpu"
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "-v",
             "jcm.physics.radiation.rrtmgp_column_block_test."
             "TwoDeviceColumnBlockTest"],
            env=env, cwd=repo_root, capture_output=True, text=True)
        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, msg=output)
        # The child must have had 2 devices and run the checks, not skipped.
        self.assertNotIn("skipped", output.lower(), msg=output)
        self.assertIn("ok", output.lower(), msg=output)


if __name__ == "__main__":
    unittest.main()
