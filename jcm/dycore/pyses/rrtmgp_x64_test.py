"""RRTMGP wrapper under ``jax_enable_x64`` with float32 physics inputs.

The pySES backend (and MAM4-JAX) enable ``jax_enable_x64`` process-wide while
jcm physics keeps running float32 (``physics_dtype``). Under that combination
any dtype-less array creation in the RRTMGP input prep silently becomes
float64 and meets float32 ``vmr_fields`` inside the library's gas-optics
``lax.cond`` branches — a trace-time TypeError that killed the first derecho
pySES JAM smoke run. This test reproduces that exact coupling (x64 on, f32
inputs) on a tiny column.

It lives in the pyses test package so ``conftest.py`` schedules it after all
dtype-sensitive tests — flipping x64 here must not contaminate them.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np


class RRTMGPWithX64HostTest(unittest.TestCase):
    def _run(self, dtype):
        jax.config.update("jax_enable_x64", True)
        from jcm.physics.radiation.rrtmgp import radiation_scheme_rrtmgp
        from jcm.physics.radiation.rrtmgp_test import _make_inputs

        inputs = jax.tree_util.tree_map(
            lambda x: (x.astype(dtype)
                       if isinstance(x, jnp.ndarray)
                       and jnp.issubdtype(x.dtype, jnp.floating) else x),
            _make_inputs(nlev=10),
        )
        tendencies, _ = radiation_scheme_rrtmgp(**inputs)
        self.assertTrue(
            bool(np.all(np.isfinite(np.asarray(
                tendencies.temperature_tendency)))),
            f"non-finite heating rate under x64 host / {dtype} physics",
        )

    def test_trace_and_run_with_f32_inputs(self):
        # The nominal coupling: physics_dtype float32 state.
        self._run(jnp.float32)

    def test_trace_and_run_with_f64_inputs(self):
        # The leaky coupling that killed the derecho smoke run: under x64,
        # parts of the state upcast to float64 upstream (e.g. pressures
        # built from float64 hybrid tables) before reaching radiation.
        self._run(jnp.float64)


if __name__ == "__main__":
    unittest.main()
