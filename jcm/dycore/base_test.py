"""Contract tests for the :class:`DynamicalCore` protocol.

The intent here is not to test a backend — it is to test the **shape of the
protocol itself**. A trivial in-test fake exercises every abstract method so
that the ABC machinery raises on any incomplete subclass. The registry and
the :class:`Predictions` pytree-compatibility are also covered.

The richer end-to-end design spike (Phase 0.5) lives in
``jcm/dycore/protocol_test.py``, where a non-trivial fake cubed-sphere dycore
drives real Held-Suarez physics through :class:`Model`.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from jcm.dycore import (
    DynamicalCore,
    Predictions,
    build_dycore,
    list_dycores,
    register_dycore,
)
from jcm.dycore.registry import _REGISTRY


class _TrivialDycore(DynamicalCore):
    """Minimal-but-complete implementation used only to exercise the protocol.

    Every abstract method has a stub that returns a value of the correct kind;
    no real dynamics or physics happen. Lives in this test file so the rest of
    the codebase never imports it.
    """

    def __init__(self):
        # Bare-minimum coords-like object: just enough attribute surface to keep
        # the stub methods honest. The real protocol consumers expect a
        # ``CoordinateSystem`` here.
        self.coords = type("_Coords", (), {"horizontal": None, "vertical": None})()
        self.dt_seconds = 60.0
        # Terrain is opaque to the protocol; ``None`` is fine for the stub.
        self.terrain = None  # type: ignore[assignment]
        self.tracer_specs = {}

    def initial_state(self, physics_state, *, sim_time=0.0, random_seed=0, tracer_specs=None):
        return {"sim_time": jnp.asarray(sim_time, dtype=jnp.float64)}

    def to_physics_state(self, state):
        return state  # the trivial dycore stores its state as a PhysicsState

    def step(self, state, physics_tendency):
        return {**state, "sim_time": state["sim_time"] + self.dt_seconds}

    def sim_time(self, state):
        return state["sim_time"]

    def with_sim_time(self, state, sim_time):
        return {**state, "sim_time": jnp.asarray(sim_time, dtype=jnp.float64)}

    def to_xarray(self, predictions, times, *, additional_coords=None):
        return xr.Dataset({"sim_time": ("time", np.asarray(times))})

    def build_terrain(self, *, source_file=None, **kwargs):
        return None  # type: ignore[return-value]


class DynamicalCoreContractTest(unittest.TestCase):
    """The ABC must reject incomplete implementations and accept complete ones."""

    def test_complete_subclass_is_instantiable(self):
        dycore = _TrivialDycore()
        self.assertIsInstance(dycore, DynamicalCore)

    def test_incomplete_subclass_is_rejected(self):
        # Drop one abstract method to prove the ABC machinery is actually wired.
        class _Incomplete(_TrivialDycore):
            pass

        # _Incomplete inherits all stubs and IS instantiable.
        _Incomplete()

        # Now drop one method explicitly and check Python complains.
        class _MissingStep(DynamicalCore):
            def initial_state(self, *a, **k): ...
            def to_physics_state(self, *a, **k): ...
            def sim_time(self, *a, **k): ...
            def with_sim_time(self, *a, **k): ...
            def to_xarray(self, *a, **k): ...
            def build_terrain(self, *a, **k): ...
            # step deliberately omitted

        with self.assertRaises(TypeError):
            _MissingStep()  # type: ignore[abstract]

    def test_required_tracers_ok_defaults_to_permissive(self):
        # The default implementation accepts anything without raising.
        _TrivialDycore().required_tracers_ok([])

    def test_step_advances_sim_time(self):
        d = _TrivialDycore()
        s0 = d.initial_state(None, sim_time=0.0)
        s1 = d.step(s0, None)
        self.assertAlmostEqual(float(d.sim_time(s1)), d.dt_seconds)


class PredictionsTest(unittest.TestCase):
    """``Predictions`` must round-trip through JAX pytree machinery."""

    def test_tree_map_traverses_all_fields(self):
        p = Predictions(
            dynamics=jnp.ones((3, 2)),
            physics={"foo": jnp.zeros((3, 2))},
            times=jnp.arange(3),
        )
        doubled = jax.tree_util.tree_map(lambda x: x * 2, p)
        self.assertTrue(jnp.allclose(doubled.dynamics, 2.0))
        self.assertTrue(jnp.allclose(doubled.physics["foo"], 0.0))
        self.assertTrue(jnp.allclose(doubled.times, jnp.arange(3) * 2))

    def test_replace_works(self):
        p = Predictions(dynamics=1, physics=2, times=3)
        q = p.replace(times=99)
        self.assertEqual(q.dynamics, 1)
        self.assertEqual(q.physics, 2)
        self.assertEqual(q.times, 99)


class RegistryTest(unittest.TestCase):
    """The dycore name→factory registry must round-trip and complain helpfully."""

    def setUp(self):
        # Snapshot and clear the global registry so tests don't leak.
        self._saved_registry = dict(_REGISTRY)
        _REGISTRY.clear()

    def tearDown(self):
        _REGISTRY.clear()
        _REGISTRY.update(self._saved_registry)

    def test_register_and_build(self):
        @register_dycore("trivial")
        def _make(**kwargs):
            return _TrivialDycore()

        self.assertIn("trivial", list_dycores())
        dycore = build_dycore("trivial")
        self.assertIsInstance(dycore, _TrivialDycore)

    def test_re_registration_overrides(self):
        @register_dycore("trivial")
        def _make_a(**kwargs):
            d = _TrivialDycore()
            d.dt_seconds = 1.0
            return d

        @register_dycore("trivial")
        def _make_b(**kwargs):
            d = _TrivialDycore()
            d.dt_seconds = 2.0
            return d

        self.assertEqual(build_dycore("trivial").dt_seconds, 2.0)

    def test_unknown_name_raises_with_available_list(self):
        @register_dycore("alpha")
        def _alpha(**kwargs):
            return _TrivialDycore()

        with self.assertRaises(KeyError) as cm:
            build_dycore("beta")
        msg = str(cm.exception)
        self.assertIn("beta", msg)
        self.assertIn("alpha", msg)

    def test_empty_name_rejected(self):
        with self.assertRaises(ValueError):
            register_dycore("")


if __name__ == "__main__":
    unittest.main()
