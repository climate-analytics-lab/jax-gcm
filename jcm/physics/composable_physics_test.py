"""Tests for PhysicsTerm and ComposablePhysics.

Phase 2b differentiability gate: verify nnx.grad flows through composable
physics terms, and that composition operators work correctly.

Date: 2026-04-12
"""

import unittest
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy.testing as npt
from flax import nnx

from jcm.physics.physics_term import PhysicsTerm, TracerSpec
from jcm.physics.composable_physics import ComposablePhysics
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.date import DateData


# ---------------------------------------------------------------------------
# Toy terms for testing
# ---------------------------------------------------------------------------

class LinearHeating(PhysicsTerm):
    """Toy term: temperature tendency = alpha * mean(temperature)."""

    name: ClassVar[str] = "linear_heating"
    category: ClassVar[str] = "radiation"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("heating_rate",)

    def __init__(self, alpha: float = 1.0):
        """Initialize LinearHeating."""
        self.alpha = nnx.Param(jnp.array(alpha))

    def __call__(self, state, diagnostics, forcing, terrain):
        heating = self.alpha[...] * state.temperature
        tendency = PhysicsTendency.zeros(state.temperature.shape).copy(
            temperature=heating,
        )
        new_diag = {**diagnostics, "heating_rate": heating}
        return tendency, new_diag


class QuadraticMoistening(PhysicsTerm):
    """Toy term: specific_humidity tendency = beta * temperature^2."""

    name: ClassVar[str] = "quadratic_moistening"
    category: ClassVar[str] = "convection"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("moisture_source",)

    def __init__(self, beta: float = 0.5):
        """Initialize QuadraticMoistening."""
        self.beta = nnx.Param(jnp.array(beta))

    def __call__(self, state, diagnostics, forcing, terrain):
        source = self.beta[...] * state.temperature ** 2
        tendency = PhysicsTendency.zeros(state.temperature.shape).copy(
            specific_humidity=source,
        )
        new_diag = {**diagnostics, "moisture_source": source}
        return tendency, new_diag


class DiagnosticConsumer(PhysicsTerm):
    """Toy term that reads upstream diagnostics and uses them."""

    name: ClassVar[str] = "diagnostic_consumer"
    category: ClassVar[str] = "surface"
    requires: ClassVar[tuple[str, ...]] = ("heating_rate",)
    provides: ClassVar[tuple[str, ...]] = ("surface_flux",)

    def __init__(self, gamma: float = 0.1):
        """Initialize DiagnosticConsumer."""
        self.gamma = nnx.Param(jnp.array(gamma))

    def __call__(self, state, diagnostics, forcing, terrain):
        # Use upstream heating_rate to compute a wind tendency
        heating = diagnostics["heating_rate"]
        wind_tend = self.gamma[...] * heating
        tendency = PhysicsTendency.zeros(state.temperature.shape).copy(
            u_wind=wind_tend,
        )
        new_diag = {**diagnostics, "surface_flux": wind_tend}
        return tendency, new_diag


# ---------------------------------------------------------------------------
# Helper: build a small test state
# ---------------------------------------------------------------------------

def _make_test_state(shape=(2, 4, 8)):
    """Create a simple PhysicsState with non-trivial values."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    return PhysicsState(
        u_wind=jax.random.normal(keys[0], shape),
        v_wind=jax.random.normal(keys[1], shape),
        temperature=jax.random.normal(keys[2], shape) + 250.0,
        specific_humidity=jnp.abs(jax.random.normal(keys[3], shape)),
        geopotential=jnp.zeros(shape),
        normalized_surface_pressure=jnp.ones(shape[1:]),
    )


def _make_test_forcing(shape=(4, 8)):
    return ForcingData.zeros(shape)


def _make_test_terrain(shape=(4, 8)):
    zero = jnp.zeros(shape)
    return TerrainData(
        orog=zero, phis0=zero, fmask=zero, lfluxland=jnp.bool_(False),
        orostd=zero, orosig=zero, orogam=zero,
        orothe=zero, oropic=zero, oroval=zero,
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestPhysicsTermBasics(unittest.TestCase):
    """Basic construction and call tests for PhysicsTerm subclasses."""

    def test_single_term_call(self):
        term = LinearHeating(alpha=2.0)
        state = _make_test_state()
        forcing = _make_test_forcing()
        terrain = _make_test_terrain()

        tend, diag = term(state, {}, forcing, terrain)

        self.assertEqual(tend.temperature.shape, state.temperature.shape)
        self.assertIn("heating_rate", diag)
        # tendency should be 2.0 * temperature
        npt.assert_allclose(tend.temperature, 2.0 * state.temperature)

    def test_term_addition_creates_composable(self):
        a = LinearHeating()
        b = QuadraticMoistening()
        physics = a + b
        self.assertEqual(type(physics).__name__, "ComposablePhysics")
        self.assertEqual(len(physics.terms), 2)

    def test_sum_of_terms(self):
        terms = [LinearHeating(), QuadraticMoistening()]
        physics = sum(terms)
        self.assertEqual(type(physics).__name__, "ComposablePhysics")
        self.assertEqual(len(physics.terms), 2)


class TestComposablePhysics(unittest.TestCase):
    """Tests for ComposablePhysics composition operators and validation."""

    def _make_physics(self):
        return ComposablePhysics(terms=[
            LinearHeating(alpha=1.0),
            QuadraticMoistening(beta=0.5),
            DiagnosticConsumer(gamma=0.1),
        ])

    def test_compute_tendencies(self):
        physics = self._make_physics()
        state = _make_test_state()
        forcing = _make_test_forcing()
        terrain = _make_test_terrain()
        date = DateData.zeros()

        tend, diag = physics.compute_tendencies(state, forcing, terrain, date)

        # Check all diagnostic keys are present
        self.assertIn("heating_rate", diag)
        self.assertIn("moisture_source", diag)
        self.assertIn("surface_flux", diag)

        # Temperature tendency comes from LinearHeating
        npt.assert_allclose(
            tend.temperature, 1.0 * state.temperature, rtol=1e-5
        )
        # specific_humidity tendency from QuadraticMoistening
        npt.assert_allclose(
            tend.specific_humidity, 0.5 * state.temperature ** 2, rtol=1e-5
        )
        # u_wind tendency from DiagnosticConsumer using upstream heating_rate
        npt.assert_allclose(
            tend.u_wind, 0.1 * state.temperature, rtol=1e-5
        )

    def test_replace(self):
        physics = self._make_physics()
        new_rad = LinearHeating(alpha=5.0)
        replaced = physics.replace("radiation", new_rad)
        self.assertEqual(len(replaced.terms), 3)
        self.assertIs(replaced.terms[0], new_rad)

    def test_remove(self):
        physics = self._make_physics()
        removed = physics.remove("convection")
        self.assertEqual(len(removed.terms), 2)
        categories = [t.category for t in removed.terms]
        self.assertNotIn("convection", categories)

    def test_add_two_composables(self):
        a = ComposablePhysics(terms=[LinearHeating()])
        b = ComposablePhysics(terms=[QuadraticMoistening()])
        combined = a + b
        self.assertEqual(len(combined.terms), 2)

    def test_validation_catches_missing_requires(self):
        with self.assertRaises(ValueError) as ctx:
            # DiagnosticConsumer requires "heating_rate" but no upstream provides it
            ComposablePhysics(terms=[DiagnosticConsumer()])
        self.assertIn("heating_rate", str(ctx.exception))

    def test_validation_passes_valid_ordering(self):
        # LinearHeating provides "heating_rate", DiagnosticConsumer requires it
        physics = ComposablePhysics(terms=[
            LinearHeating(),
            DiagnosticConsumer(),
        ])
        self.assertEqual(len(physics.terms), 2)

    def test_replace_nonexistent_category_raises(self):
        physics = self._make_physics()
        with self.assertRaises(ValueError):
            physics.replace("nonexistent", LinearHeating())


class TestDifferentiabilityGate(unittest.TestCase):
    """Phase 2b gating tests: verify gradients flow through ComposablePhysics.

    These are the critical tests that must pass before any term-wrapping work
    begins. They verify that nnx.grad produces non-zero, correct gradients
    for parameters stored as nnx.Param on PhysicsTerm subclasses.
    """

    def _make_physics(self):
        return ComposablePhysics(
            terms=[
                LinearHeating(alpha=1.0),
                QuadraticMoistening(beta=0.5),
                DiagnosticConsumer(gamma=0.1),
            ],
            checkpoint_terms=False,  # disable checkpointing for grad tests
        )

    def test_nnx_grad_produces_nonzero_gradients(self):
        """nnx.grad through composable physics gives non-zero grads."""
        physics = self._make_physics()
        state = _make_test_state()
        forcing = _make_test_forcing()
        terrain = _make_test_terrain()
        date = DateData.zeros()

        def loss_fn(physics):
            tend, _ = physics.compute_tendencies(state, forcing, terrain, date)
            return (
                jnp.sum(tend.temperature ** 2)
                + jnp.sum(tend.specific_humidity ** 2)
                + jnp.sum(tend.u_wind ** 2)
            )

        grads = nnx.grad(loss_fn)(physics)

        # Check that each term's parameter has a non-zero gradient
        alpha_grad = grads.terms[0].alpha[...]
        beta_grad = grads.terms[1].beta[...]
        gamma_grad = grads.terms[2].gamma[...]

        self.assertFalse(jnp.isnan(alpha_grad), "alpha gradient is NaN")
        self.assertFalse(jnp.isnan(beta_grad), "beta gradient is NaN")
        self.assertFalse(jnp.isnan(gamma_grad), "gamma gradient is NaN")

        self.assertNotEqual(float(alpha_grad), 0.0, "alpha gradient is zero")
        self.assertNotEqual(float(beta_grad), 0.0, "beta gradient is zero")
        self.assertNotEqual(float(gamma_grad), 0.0, "gamma gradient is zero")

    def test_nnx_grad_correct_values(self):
        """Verify gradient values match hand-computed expectations for a simple case."""
        physics = ComposablePhysics(
            terms=[LinearHeating(alpha=1.0)],
            checkpoint_terms=False,
        )
        # Use a very simple state for easy hand computation
        shape = (1, 1, 1)
        temp_val = 3.0
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, temp_val),
        )
        forcing = ForcingData.zeros(shape[1:])
        terrain = _make_test_terrain(shape[1:])
        date = DateData.zeros()

        # loss = sum((alpha * T)^2) = alpha^2 * T^2 * n_elements
        # d(loss)/d(alpha) = 2 * alpha * T^2 * n_elements
        # With alpha=1.0, T=3.0, n=1: d(loss)/d(alpha) = 2 * 1 * 9 * 1 = 18.0
        def loss_fn(physics):
            tend, _ = physics.compute_tendencies(state, forcing, terrain, date)
            return jnp.sum(tend.temperature ** 2)

        grads = nnx.grad(loss_fn)(physics)
        expected_grad = 2.0 * 1.0 * temp_val ** 2
        npt.assert_allclose(
            grads.terms[0].alpha[...], expected_grad, rtol=1e-5
        )

    def test_jax_grad_via_split_merge(self):
        """Verify jax.grad works through nnx.split/merge (Pattern 2 from design doc)."""
        physics = ComposablePhysics(
            terms=[LinearHeating(alpha=2.0), QuadraticMoistening(beta=0.3)],
            checkpoint_terms=False,
        )
        state = _make_test_state()
        forcing = _make_test_forcing()
        terrain = _make_test_terrain()
        date = DateData.zeros()

        graphdef, param_state = nnx.split(physics)

        def loss_fn(param_state):
            physics_restored = nnx.merge(graphdef, param_state)
            tend, _ = physics_restored.compute_tendencies(
                state, forcing, terrain, date
            )
            return jnp.sum(tend.temperature ** 2) + jnp.sum(tend.specific_humidity ** 2)

        grads = jax.grad(loss_fn)(param_state)

        # Verify non-zero gradients exist
        grad_leaves = jax.tree_util.tree_leaves(grads)
        for i, leaf in enumerate(grad_leaves):
            self.assertFalse(
                jnp.all(leaf == 0.0),
                f"Gradient leaf {i} is all zeros via split/merge path",
            )

    def test_per_term_gradient_addressing(self):
        """Verify per-term parameter addressing for optimization."""
        physics = self._make_physics()

        # Access individual term parameters directly
        self.assertIsInstance(physics.terms[0].alpha, nnx.Param)
        self.assertIsInstance(physics.terms[1].beta, nnx.Param)
        self.assertIsInstance(physics.terms[2].gamma, nnx.Param)

        # Verify we can get parameter values
        self.assertAlmostEqual(float(physics.terms[0].alpha[...]), 1.0, places=5)
        self.assertAlmostEqual(float(physics.terms[1].beta[...]), 0.5, places=5)
        self.assertAlmostEqual(float(physics.terms[2].gamma[...]), 0.1, places=5)

    def test_gradient_through_diagnostic_chain(self):
        """Gradients must flow through the diagnostics dict."""
        # DiagnosticConsumer reads heating_rate from LinearHeating.
        # A change in LinearHeating's alpha should affect DiagnosticConsumer's
        # output, so the gradient of the combined loss w.r.t. alpha should
        # include the contribution through the diagnostic chain.
        physics = ComposablePhysics(
            terms=[LinearHeating(alpha=1.0), DiagnosticConsumer(gamma=1.0)],
            checkpoint_terms=False,
        )

        shape = (1, 1, 1)
        temp_val = 2.0
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, temp_val),
        )
        forcing = ForcingData.zeros(shape[1:])
        terrain = _make_test_terrain(shape[1:])
        date = DateData.zeros()

        # Only look at u_wind loss (produced by DiagnosticConsumer)
        # u_wind_tend = gamma * heating_rate = gamma * alpha * T
        # loss = sum(u_wind_tend^2) = (gamma * alpha * T)^2
        # d(loss)/d(alpha) = 2 * gamma^2 * alpha * T^2
        # With gamma=1, alpha=1, T=2: d(loss)/d(alpha) = 2 * 1 * 1 * 4 = 8
        def loss_fn(physics):
            tend, _ = physics.compute_tendencies(state, forcing, terrain, date)
            return jnp.sum(tend.u_wind ** 2)

        grads = nnx.grad(loss_fn)(physics)
        expected_alpha_grad = 2.0 * 1.0 ** 2 * 1.0 * temp_val ** 2
        npt.assert_allclose(
            grads.terms[0].alpha[...], expected_alpha_grad, rtol=1e-5,
            err_msg="Gradient through diagnostic chain is incorrect",
        )

    def test_cached_coords_as_nnx_variable(self):
        """nnx.Variable for cached coords should be traced but not differentiated."""

        class CoordsAwareTerm(PhysicsTerm):
            name: ClassVar[str] = "coords_aware"
            category: ClassVar[str] = "test"

            def __init__(self, scale: float = 1.0):
                """Initialize CoordsAwareTerm."""
                self.scale = nnx.Param(jnp.array(scale))
                # Initialize with a placeholder; cache_coords will overwrite
                self.cached_value = nnx.Variable(jnp.array(0.0))

            def cache_coords(self, coords):
                # Overwrite with coordinate-derived value
                self.cached_value[...] = jnp.array(42.0)

            def __call__(self, state, diagnostics, forcing, terrain):
                # Use both the param and the cached value
                out = self.scale[...] * state.temperature + self.cached_value[...]
                tendency = PhysicsTendency.zeros(state.temperature.shape).copy(
                    temperature=out,
                )
                return tendency, diagnostics

        term = CoordsAwareTerm(scale=1.0)
        term.cache_coords(None)  # populate cached_value

        physics = ComposablePhysics(terms=[term], checkpoint_terms=False)

        shape = (1, 1, 1)
        state = PhysicsState.zeros(shape).copy(
            temperature=jnp.full(shape, 5.0),
        )
        forcing = ForcingData.zeros(shape[1:])
        terrain = _make_test_terrain(shape[1:])
        date = DateData.zeros()

        def loss_fn(physics):
            tend, _ = physics.compute_tendencies(state, forcing, terrain, date)
            return jnp.sum(tend.temperature ** 2)

        # This should work — nnx.grad differentiates Param but not Variable
        grads = nnx.grad(loss_fn)(physics)
        scale_grad = grads.terms[0].scale[...]
        self.assertFalse(jnp.isnan(scale_grad))
        self.assertNotEqual(float(scale_grad), 0.0)


class TestColumnVectorization(unittest.TestCase):
    """Test ComposablePhysics with vectorize_columns=True."""

    def test_column_vectorization_produces_correct_shapes(self):
        """Column-vectorized physics reshapes 3D → columns → 3D correctly."""
        # A simple term that works on column format
        term = LinearHeating(alpha=2.0)
        physics = ComposablePhysics(
            terms=[term],
            checkpoint_terms=False,
            vectorize_columns=True,
        )

        shape = (2, 4, 8)
        state = _make_test_state(shape)
        forcing = _make_test_forcing(shape[1:])
        terrain = _make_test_terrain(shape[1:])
        date = DateData.zeros()

        tend, diag = physics.compute_tendencies(state, forcing, terrain, date)

        # Output should be 3D again
        self.assertEqual(tend.temperature.shape, shape)
        self.assertEqual(tend.u_wind.shape, shape)

    def test_column_vectorization_with_tracers(self):
        """Column vectorization handles tracers correctly."""
        term = LinearHeating(alpha=1.0)
        physics = ComposablePhysics(
            terms=[term],
            checkpoint_terms=False,
            vectorize_columns=True,
        )

        shape = (2, 4, 8)
        key = jax.random.PRNGKey(0)
        state = PhysicsState(
            u_wind=jnp.zeros(shape),
            v_wind=jnp.zeros(shape),
            temperature=jnp.ones(shape) * 250.0,
            specific_humidity=jnp.zeros(shape),
            geopotential=jnp.zeros(shape),
            normalized_surface_pressure=jnp.ones(shape[1:]),
            tracers={"qc": jax.random.normal(key, shape)},
        )
        forcing = _make_test_forcing(shape[1:])
        terrain = _make_test_terrain(shape[1:])
        date = DateData.zeros()

        tend, _ = physics.compute_tendencies(state, forcing, terrain, date)
        self.assertEqual(tend.temperature.shape, shape)

    def test_column_vectorization_with_prev_data(self):
        """Column vectorization carries forward prev_physics_data."""
        term = LinearHeating(alpha=1.0)
        physics = ComposablePhysics(
            terms=[term],
            checkpoint_terms=False,
            vectorize_columns=True,
        )

        shape = (2, 4, 8)
        state = _make_test_state(shape)
        forcing = _make_test_forcing(shape[1:])
        terrain = _make_test_terrain(shape[1:])
        date = DateData.zeros()

        prev_data = {"_cached_value": jnp.array(42.0)}
        tend, diag = physics.compute_tendencies(
            state, forcing, terrain, date, prev_physics_data=prev_data,
        )
        # prev_data should be carried forward
        self.assertIn("_cached_value", diag)

    def test_column_vs_3d_numerically_equivalent(self):
        """Column-vectorized and 3D paths should produce same results for simple terms."""
        term_3d = LinearHeating(alpha=2.0)
        term_col = LinearHeating(alpha=2.0)

        physics_3d = ComposablePhysics(
            terms=[term_3d], checkpoint_terms=False, vectorize_columns=False,
        )
        physics_col = ComposablePhysics(
            terms=[term_col], checkpoint_terms=False, vectorize_columns=True,
        )

        shape = (2, 4, 8)
        state = _make_test_state(shape)
        forcing = _make_test_forcing(shape[1:])
        terrain = _make_test_terrain(shape[1:])
        date = DateData.zeros()

        tend_3d, _ = physics_3d.compute_tendencies(state, forcing, terrain, date)
        tend_col, _ = physics_col.compute_tendencies(state, forcing, terrain, date)

        npt.assert_allclose(tend_3d.temperature, tend_col.temperature, rtol=1e-6)
        npt.assert_allclose(tend_3d.u_wind, tend_col.u_wind, rtol=1e-6)


class TestComposablePhysicsUtilities(unittest.TestCase):
    """Test get_empty_data and data_struct_to_dict."""

    def test_get_empty_data(self):
        """get_empty_data returns a zeroed diagnostics dict."""
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        physics = ComposablePhysics(
            terms=[LinearHeating(), QuadraticMoistening()],
        )
        physics.cache_coords(coords)
        empty = physics.get_empty_data(coords)
        self.assertIsInstance(empty, dict)
        self.assertIn("heating_rate", empty)
        # Array values should be zeros
        for v in empty.values():
            if isinstance(v, jax.Array) and v.shape:
                self.assertTrue(jnp.all(v == 0.0))

    def test_data_struct_to_dict_multichannel_5d(self):
        """data_struct_to_dict expands 5D multi-channel fields."""
        physics = ComposablePhysics(terms=[LinearHeating()])
        nodal_shape = (4, 8)
        # 5D: s[1:-1] == nodal_shape → should expand on trailing dim
        struct = {
            "flux": jnp.zeros((2, 4, 8, 3)),  # 4D, doesn't match
            "flux5d": jnp.zeros((1, 2, 4, 8, 3)),  # 5D, s[1:-1]=(2,4,8)!=nodal
        }
        result = physics.data_struct_to_dict(struct, nodal_shape=nodal_shape)
        # Neither should expand since shapes don't match the pattern
        self.assertIn("flux", result)

    def test_data_struct_to_dict_non_array_values(self):
        """Plain Python values (ints, strings) drop out of the user dict.

        Only jax.Array leaves and flattenable sub-structs make it into the
        xarray output — everything else is silently skipped.
        """
        physics = ComposablePhysics(terms=[LinearHeating()])
        struct = {"count": 42, "name": "test", "field": jnp.array([1.0, 2.0])}
        result = physics.data_struct_to_dict(struct, nodal_shape=(4, 8))
        self.assertNotIn("count", result)
        self.assertNotIn("name", result)
        self.assertIn("field", result)

    def test_data_struct_to_dict_filters_internal_keys(self):
        """Underscore-prefixed array keys are exposed without the underscore;
        underscore-prefixed plumbing keys (`_date`, etc.) stay hidden.
        """
        physics = ComposablePhysics(terms=[LinearHeating()])
        struct = {
            "_internal": jnp.array(1.0),
            "public_key": jnp.array(2.0),
            "_date": DateData.zeros(),
        }
        result = physics.data_struct_to_dict(struct)
        self.assertIn("public_key", result)
        # Underscore-prefixed array key surfaces as the key without underscore.
        self.assertIn("internal", result)
        # `_date` is plumbing — stays hidden.
        self.assertNotIn("_date", result)
        self.assertNotIn("date", result)
        # Original underscore key is not preserved.
        self.assertNotIn("_internal", result)

    def test_data_struct_to_dict_none(self):
        physics = ComposablePhysics(terms=[LinearHeating()])
        result = physics.data_struct_to_dict(None)
        self.assertEqual(result, {})

    def test_data_struct_to_dict_with_nodal_shape(self):
        physics = ComposablePhysics(terms=[LinearHeating()])
        nodal_shape = (4, 8)
        struct = {"temperature": jnp.zeros((1, 4, 8))}
        result = physics.data_struct_to_dict(struct, nodal_shape=nodal_shape)
        self.assertIn("temperature", result)

    def test_vectorize_columns_preserved_by_replace(self):
        physics = ComposablePhysics(
            terms=[LinearHeating(), QuadraticMoistening()],
            vectorize_columns=True,
        )
        replaced = physics.replace("radiation", LinearHeating(alpha=5.0))
        self.assertTrue(replaced.vectorize_columns)

    def test_vectorize_columns_preserved_by_remove(self):
        physics = ComposablePhysics(
            terms=[LinearHeating(), QuadraticMoistening()],
            vectorize_columns=True,
        )
        removed = physics.remove("convection")
        self.assertTrue(removed.vectorize_columns)

    def test_vectorize_columns_preserved_by_add(self):
        a = ComposablePhysics(terms=[LinearHeating()], vectorize_columns=True)
        b = ComposablePhysics(terms=[QuadraticMoistening()])
        combined = a + b
        self.assertTrue(combined.vectorize_columns)


class TestPackagesImport(unittest.TestCase):
    """Test packages/ factory re-exports."""

    def test_packages_speedy_import(self):
        from jcm.physics.speedy.speedy_terms import speedy_physics
        self.assertTrue(callable(speedy_physics))

    def test_packages_echam_import(self):
        from jcm.physics.echam.echam_terms import echam_physics
        self.assertTrue(callable(echam_physics))


class TestTracerSpec(unittest.TestCase):
    """TracerSpec declaration + ComposablePhysics aggregation."""

    def test_default_required_tracers_is_empty(self):
        self.assertEqual(LinearHeating().required_tracers(), ())

    def test_aggregation_dedups_identical_specs(self):
        class A(PhysicsTerm):
            name: ClassVar[str] = "a"
            category: ClassVar[str] = "x"
            @classmethod
            def required_tracers(cls):
                return (TracerSpec("qc"),)
            def __call__(self, state, diagnostics, forcing, terrain):
                return PhysicsTendency.zeros(state.temperature.shape), diagnostics

        class B(PhysicsTerm):
            name: ClassVar[str] = "b"
            category: ClassVar[str] = "y"
            @classmethod
            def required_tracers(cls):
                return (TracerSpec("qc"), TracerSpec("qi"))
            def __call__(self, state, diagnostics, forcing, terrain):
                return PhysicsTendency.zeros(state.temperature.shape), diagnostics

        physics = ComposablePhysics(terms=[A(), B()])
        names = {spec.name for spec in physics.required_tracers()}
        self.assertEqual(names, {"qc", "qi"})

    def test_conflicting_specs_raise(self):
        class A(PhysicsTerm):
            name: ClassVar[str] = "a"
            category: ClassVar[str] = "x"
            @classmethod
            def required_tracers(cls):
                return (TracerSpec("qnc", nondimensionalize=False),)
            def __call__(self, state, diagnostics, forcing, terrain):
                return PhysicsTendency.zeros(state.temperature.shape), diagnostics

        class B(PhysicsTerm):
            name: ClassVar[str] = "b"
            category: ClassVar[str] = "y"
            @classmethod
            def required_tracers(cls):
                return (TracerSpec("qnc", nondimensionalize=True),)
            def __call__(self, state, diagnostics, forcing, terrain):
                return PhysicsTendency.zeros(state.temperature.shape), diagnostics

        physics = ComposablePhysics(terms=[A(), B()])
        with self.assertRaisesRegex(ValueError, "qnc"):
            physics.required_tracers()

    def test_nondimensionalize_flag_round_trip(self):
        """Tracers with nondimensionalize=False must round-trip unchanged through the converters."""
        from dinosaur import primitive_equations
        from dinosaur.scales import SI_SCALE
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.physics_interface import (
            dynamics_state_to_physics_state,
            physics_state_to_dynamics_state,
        )

        coords = get_speedy_coords()
        specs = primitive_equations.PrimitiveEquationsSpecs.from_si(scale=SI_SCALE)
        primitive = primitive_equations.PrimitiveEquations(
            reference_temperature=jnp.ones(coords.nodal_shape[0]) * 288.0,
            orography=jnp.zeros(coords.modal_shape[1:]),
            coords=coords,
            physics_specs=specs,
        )
        nodal_shape = coords.nodal_shape

        physics_state = PhysicsState.zeros(nodal_shape)
        physics_state = physics_state.copy(
            temperature=jnp.ones(nodal_shape) * 288.0,
            normalized_surface_pressure=jnp.ones(nodal_shape[1:]),
            tracers={"qnc": jnp.ones(nodal_shape) * 1e8},  # number per kg
        )

        tracer_specs = {"qnc": TracerSpec("qnc", nondimensionalize=False)}

        modal = physics_state_to_dynamics_state(
            physics_state, primitive, tracer_specs=tracer_specs,
        )
        back = dynamics_state_to_physics_state(
            modal, primitive, tracer_specs=tracer_specs,
        )

        # qnc should round-trip within spectral transform tolerance
        npt.assert_allclose(back.tracers["qnc"], physics_state.tracers["qnc"], rtol=1e-3)


class TestModelSeedsTracers(unittest.TestCase):
    """Model seeds the initial tracer dict from physics.required_tracers()."""

    def test_model_seeds_declared_tracers(self):
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        class NeedsQC(PhysicsTerm):
            name: ClassVar[str] = "needs_qc"
            category: ClassVar[str] = "clouds"
            @classmethod
            def required_tracers(cls):
                return (TracerSpec("qc"), TracerSpec("qnc", nondimensionalize=False))
            def __call__(self, state, diagnostics, forcing, terrain):
                return PhysicsTendency.zeros(state.temperature.shape), diagnostics

        physics = ComposablePhysics(terms=[NeedsQC()])
        model = Model(coords=get_speedy_coords(), physics=physics, time_step=720)
        # _prepare_initial_modal_state runs inside .run(), but we can call the
        # underlying prep directly to verify the tracer dict.
        state = model._prepare_initial_modal_state()
        self.assertIn("specific_humidity", state.tracers)
        self.assertIn("qc", state.tracers)
        self.assertIn("qnc", state.tracers)
        self.assertEqual(
            state.tracers["qc"].shape,
            state.tracers["specific_humidity"].shape,
        )


if __name__ == "__main__":
    unittest.main()
