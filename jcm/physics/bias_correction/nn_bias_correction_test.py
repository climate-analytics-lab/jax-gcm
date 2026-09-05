"""Tests for the NN bias-correction PhysicsTerm (issue #356).

Covers: the MLP building blocks, the zero-init no-op, output shapes and the
``correct`` subset, broadcasting-native behaviour across horizontal layouts,
gradient flow (both through the pure function and through the nnx-stored
weights), composition into ``speedy_physics()``, a no-op-vs-plain-SPEEDY
check at the physics-tendency level, and JIT compatibility.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.test_util import check_jvp, check_vjp

from jcm.physics.bias_correction.nn_bias_correction import (
    CONTEXT_FEATURES,
    DenseWeights,
    dense,
    mlp,
    init_mlp,
    bias_correction_tendency,
    build_context,
    polar_taper_factor,
    remap_first_layer,
    widen_first_layer,
    NNBiasCorrection,
    make_bias_correction,
    load_bias_correction,
)
from jcm.physics.radiation.speedy_shortwave import solar
from jcm.physics.speedy.physical_constants import solc
from jcm.physics.speedy.speedy_coords import SpeedyCoords, get_speedy_coords
from jcm.physics_interface import PhysicsState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(shape, key):
    """Build a realistic PhysicsState of the given ``(nlev, *horiz)`` shape."""
    k = jax.random.split(key, 4)
    horiz = shape[1:]
    return PhysicsState(
        u_wind=jax.random.normal(k[0], shape),
        v_wind=jax.random.normal(k[1], shape),
        temperature=250.0 + jax.random.normal(k[2], shape),
        specific_humidity=1e-3 * (1.0 + jax.random.uniform(k[3], shape)),
        geopotential=jnp.zeros(shape),
        normalized_surface_pressure=jnp.ones(horiz),
        tracers={},
    )


def _identity_buffers(nlev):
    """Build placeholder normalisation buffers: identity in, identity out."""
    n_io = 4 * nlev
    return jnp.zeros(n_io), jnp.ones(n_io), jnp.ones(n_io)


def _fake_coords(nlev, lat):
    """Minimal stand-in for a CoordinateSystem.

    ``cache_coords`` reads only the latitudes and, for ``insol``,
    ``nodal_shape[0]`` (via ``SpeedyCoords.from_coordinate_system``).
    """
    return SimpleNamespace(nodal_shape=(nlev,),
                           horizontal=SimpleNamespace(latitudes=lat))


# ---------------------------------------------------------------------------
# MLP building blocks
# ---------------------------------------------------------------------------


class TestMLP(unittest.TestCase):
    """Dense layer, MLP forward, and weight initialisation."""

    def test_dense_output_shape(self):
        w = DenseWeights(kernel=jnp.ones((4, 8)), bias=jnp.zeros(8))
        self.assertEqual(dense(jnp.ones(4), w).shape, (8,))

    def test_mlp_output_shape(self):
        layers = init_mlp(jax.random.key(0), (12, 16, 12), zero_last_layer=False)
        self.assertEqual(mlp(jnp.ones(12), layers).shape, (12,))

    def test_init_layer_shapes(self):
        layers = init_mlp(jax.random.key(0), (12, 16, 32, 12))
        self.assertEqual(layers[0].kernel.shape, (12, 16))
        self.assertEqual(layers[1].kernel.shape, (16, 32))
        self.assertEqual(layers[2].kernel.shape, (32, 12))

    def test_zero_last_layer_gives_zero_output(self):
        """The shipped no-op init must produce exactly zero for any input."""
        layers = init_mlp(jax.random.key(1), (12, 16, 12), zero_last_layer=True)
        out = mlp(jax.random.normal(jax.random.key(2), (12,)), layers)
        np.testing.assert_array_equal(out, jnp.zeros(12))


# ---------------------------------------------------------------------------
# No-op at construction
# ---------------------------------------------------------------------------


class TestNoOpTendency(unittest.TestCase):
    """A zero-initialised term emits an exact-zero tendency."""

    def _assert_all_zero(self, tend):
        for field in (tend.u_wind, tend.v_wind, tend.temperature,
                      tend.specific_humidity):
            np.testing.assert_array_equal(field, jnp.zeros_like(field))

    def test_noop_on_3d_grid(self):
        term = make_bias_correction(nlev=6)            # zero-init by default
        state = _state((6, 4, 3), jax.random.key(0))
        tend, _ = term(state, {}, None, None)
        self._assert_all_zero(tend)

    def test_noop_on_column_block(self):
        term = make_bias_correction(nlev=6)
        state = _state((6, 5), jax.random.key(1))
        tend, _ = term(state, {}, None, None)
        self._assert_all_zero(tend)


# ---------------------------------------------------------------------------
# Shapes and the ``correct`` subset
# ---------------------------------------------------------------------------


class TestShapesAndSubset(unittest.TestCase):
    """Output shapes match the state; uncorrected variables stay zero."""

    def setUp(self):
        self.nlev = 6
        self.shape = (6, 4, 3)
        self.state = _state(self.shape, jax.random.key(0))
        self.layers = init_mlp(jax.random.key(7), (24, 16, 24),
                               zero_last_layer=False)
        self.mean, self.std, self.scale = _identity_buffers(self.nlev)

    def test_output_shapes_match_state(self):
        tend = bias_correction_tendency(
            self.state, self.layers, self.mean, self.std, self.scale,
            correct=("temperature", "specific_humidity", "u_wind", "v_wind"),
        )
        for field in (tend.temperature, tend.specific_humidity,
                      tend.u_wind, tend.v_wind):
            self.assertEqual(field.shape, self.shape)

    def test_uncorrected_fields_are_zero(self):
        tend = bias_correction_tendency(
            self.state, self.layers, self.mean, self.std, self.scale,
            correct=("temperature", "specific_humidity"),
        )
        np.testing.assert_array_equal(tend.u_wind, jnp.zeros(self.shape))
        np.testing.assert_array_equal(tend.v_wind, jnp.zeros(self.shape))
        # corrected fields should be non-trivial with a non-zero net
        self.assertGreater(float(jnp.sum(jnp.abs(tend.temperature))), 0.0)


# ---------------------------------------------------------------------------
# Broadcasting-native: same code, any horizontal layout
# ---------------------------------------------------------------------------


class TestBroadcastingNative(unittest.TestCase):
    """Bare (kx,) column, (kx, ncols) block, and (kx, ix, il) grid all agree."""

    def test_bare_column_matches_block(self):
        # The broadcasting contract includes a bare (kx,) column with no
        # horizontal axis at all, not just flattened blocks and grids.
        nlev, ncols = 6, 5
        layers = init_mlp(jax.random.key(5), (24, 16, 24), zero_last_layer=False)
        mean, std, scale = _identity_buffers(nlev)

        block = _state((nlev, ncols), jax.random.key(6))
        j = 2
        column = PhysicsState(
            u_wind=block.u_wind[:, j],
            v_wind=block.v_wind[:, j],
            temperature=block.temperature[:, j],
            specific_humidity=block.specific_humidity[:, j],
            geopotential=block.geopotential[:, j],
            normalized_surface_pressure=block.normalized_surface_pressure[j],
            tracers={},
        )

        t_block = bias_correction_tendency(block, layers, mean, std, scale,
                                           correct=("temperature",))
        t_col = bias_correction_tendency(column, layers, mean, std, scale,
                                         correct=("temperature",))
        self.assertEqual(t_col.temperature.shape, (nlev,))
        np.testing.assert_allclose(t_col.temperature,
                                   t_block.temperature[:, j],
                                   rtol=1e-6, atol=1e-6)

    def test_column_block_matches_grid(self):
        nlev, ix, il = 6, 4, 3
        ncols = ix * il
        layers = init_mlp(jax.random.key(3), (24, 16, 24), zero_last_layer=False)
        mean, std, scale = _identity_buffers(nlev)

        grid = _state((nlev, ix, il), jax.random.key(4))
        # Same underlying data, flattened to a column block (C-order).
        block = PhysicsState(
            u_wind=grid.u_wind.reshape(nlev, ncols),
            v_wind=grid.v_wind.reshape(nlev, ncols),
            temperature=grid.temperature.reshape(nlev, ncols),
            specific_humidity=grid.specific_humidity.reshape(nlev, ncols),
            geopotential=grid.geopotential.reshape(nlev, ncols),
            normalized_surface_pressure=grid.normalized_surface_pressure.reshape(ncols),
            tracers={},
        )

        t_grid = bias_correction_tendency(grid, layers, mean, std, scale,
                                          correct=("temperature",))
        t_block = bias_correction_tendency(block, layers, mean, std, scale,
                                           correct=("temperature",))
        np.testing.assert_allclose(
            t_grid.temperature, t_block.temperature.reshape(nlev, ix, il),
            rtol=1e-6, atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestOutputCap(unittest.TestCase):
    """The optional soft tanh bound on the dimensionless output."""

    def setUp(self):
        self.nlev = 6
        self.state = _state((self.nlev, 4, 3), jax.random.key(0))
        # Large kernels so the raw dimensionless output far exceeds the cap.
        big = init_mlp(jax.random.key(9), (24, 16, 24), zero_last_layer=False)
        self.layers = tuple(
            DenseWeights(w.kernel * 50.0, w.bias) for w in big)
        self.mean, self.std, self.scale = _identity_buffers(self.nlev)

    def test_none_is_exactly_unbounded(self):
        base = bias_correction_tendency(
            self.state, self.layers, self.mean, self.std, self.scale,
            correct=("temperature",))
        explicit = bias_correction_tendency(
            self.state, self.layers, self.mean, self.std, self.scale,
            correct=("temperature",), output_cap=None)
        np.testing.assert_array_equal(np.asarray(base.temperature),
                                      np.asarray(explicit.temperature))

    def test_cap_bounds_the_output(self):
        cap = 3.0
        tend = bias_correction_tendency(
            self.state, self.layers, self.mean, self.std, self.scale,
            correct=("temperature",), output_cap=cap)
        # out_scale is 1 here, so the physical tendency is the dimensionless
        # output and must respect |out| <= cap (float32 tanh saturates to
        # exactly 1, so equality is reachable).
        self.assertLessEqual(float(jnp.max(jnp.abs(tend.temperature))), cap)
        # And it saturates: with 50x kernels the peak should sit near the cap.
        self.assertGreater(float(jnp.max(jnp.abs(tend.temperature))),
                           0.9 * cap)

    def test_cap_round_trips_through_save_load(self):
        import os
        import tempfile
        term = NNBiasCorrection(self.layers, self.mean, self.std, self.scale,
                                ("temperature",), output_cap=2.5)
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "capped.npz")
        term.save(path)
        loaded = NNBiasCorrection.from_file(path)
        self.assertEqual(loaded.output_cap, 2.5)
        a, _ = term(self.state, {}, None, None)
        b, _ = loaded(self.state, {}, None, None)
        np.testing.assert_allclose(np.asarray(a.temperature),
                                   np.asarray(b.temperature), rtol=1e-6)

    def test_capless_file_loads_with_none(self):
        import os
        import tempfile
        term = NNBiasCorrection(self.layers, self.mean, self.std, self.scale,
                                ("temperature",))
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "uncapped.npz")
        term.save(path)
        loaded = NNBiasCorrection.from_file(path)
        self.assertIsNone(loaded.output_cap)


# ---------------------------------------------------------------------------
# Polar taper: switch the applied correction off near the poles
# ---------------------------------------------------------------------------


class TestPolarTaper(unittest.TestCase):
    """The optional latitude taper that zeros the correction near the poles."""

    def setUp(self):
        self.nlev = 6
        self.shape = (self.nlev, 4, 3)
        self.state = _state(self.shape, jax.random.key(0))
        # Non-zero output layer so the untapered correction is non-trivial.
        self.layers = init_mlp(jax.random.key(9), (24, 16, 24),
                               zero_last_layer=False)
        self.mean, self.std, self.scale = _identity_buffers(self.nlev)

    def test_factor_endpoints_and_monotone(self):
        # Degrees in, radians expected. Factor is 1 in the tropics, 0 at the
        # poles, monotone non-increasing in |lat|. (Also the deg-vs-rad guard.)
        self.assertAlmostEqual(
            float(polar_taper_factor(jnp.array(0.0), 55.0, 75.0)), 1.0)
        self.assertAlmostEqual(
            float(polar_taper_factor(jnp.deg2rad(80.0), 55.0, 75.0)), 0.0)
        self.assertAlmostEqual(
            float(polar_taper_factor(jnp.deg2rad(-80.0), 55.0, 75.0)), 0.0)
        # Full strength equatorward of lat0, exact zero poleward of lat1.
        np.testing.assert_array_equal(
            np.asarray(polar_taper_factor(
                jnp.deg2rad(jnp.array([-50.0, 0.0, 50.0])), 55.0, 75.0)),
            np.ones(3))
        np.testing.assert_array_equal(
            np.asarray(polar_taper_factor(
                jnp.deg2rad(jnp.array([-77.0, 77.0, 89.0])), 55.0, 75.0)),
            np.zeros(3))
        # Monotone non-increasing from equator to pole, and range in [0, 1].
        north = polar_taper_factor(jnp.deg2rad(jnp.linspace(0.0, 90.0, 46)),
                                   55.0, 75.0)
        self.assertTrue(bool(jnp.all(jnp.diff(north) <= 1e-7)))
        self.assertGreaterEqual(float(jnp.min(north)), 0.0)
        self.assertLessEqual(float(jnp.max(north)), 1.0)

    def test_taper_scales_helper_output(self):
        # A (nlat,) taper multiplies each corrected field on the trailing axis.
        taper = jnp.array([1.0, 0.5, 0.0])  # equator ... pole
        base = bias_correction_tendency(
            self.state, self.layers, self.mean, self.std, self.scale,
            correct=("temperature", "specific_humidity"))
        tapered = bias_correction_tendency(
            self.state, self.layers, self.mean, self.std, self.scale,
            correct=("temperature", "specific_humidity"), taper=taper)
        np.testing.assert_allclose(
            np.asarray(tapered.temperature),
            np.asarray(base.temperature) * np.asarray(taper),
            rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(tapered.specific_humidity),
            np.asarray(base.specific_humidity) * np.asarray(taper),
            rtol=1e-6, atol=1e-6)
        # Pole column (taper 0) exactly zeroed; equator column (taper 1) intact.
        np.testing.assert_array_equal(
            np.asarray(tapered.temperature[:, :, 2]),
            np.zeros((self.nlev, 4)))
        np.testing.assert_allclose(
            np.asarray(tapered.temperature[:, :, 0]),
            np.asarray(base.temperature[:, :, 0]), rtol=1e-6, atol=1e-6)

    def test_none_is_exactly_untapered(self):
        base = bias_correction_tendency(
            self.state, self.layers, self.mean, self.std, self.scale,
            correct=("temperature",))
        explicit = bias_correction_tendency(
            self.state, self.layers, self.mean, self.std, self.scale,
            correct=("temperature",), taper=None)
        np.testing.assert_array_equal(np.asarray(base.temperature),
                                      np.asarray(explicit.temperature))

    def test_cache_coords_populates_taper(self):
        term = NNBiasCorrection(self.layers, self.mean, self.std, self.scale,
                                ("temperature",), polar_taper=(55.0, 75.0))
        self.assertIsNone(term._taper_value())  # not cached yet
        lat = jnp.deg2rad(jnp.array([-80.0, -30.0, 30.0, 80.0]))
        coords = SimpleNamespace(
            horizontal=SimpleNamespace(latitudes=lat))
        term.cache_coords(coords)
        vals = term._taper_value()
        self.assertIsNotNone(vals)
        vals = np.asarray(vals)
        self.assertEqual(vals.shape, (4,))
        np.testing.assert_allclose(
            vals, np.asarray(polar_taper_factor(lat, 55.0, 75.0)), rtol=1e-6)
        np.testing.assert_allclose(vals[[0, 3]], np.zeros(2), atol=1e-7)  # poles
        np.testing.assert_array_equal(vals[[1, 2]], np.ones(2))          # mid-lat

    def test_cache_coords_noop_when_unset(self):
        term = NNBiasCorrection(self.layers, self.mean, self.std, self.scale,
                                ("temperature",))
        lat = jnp.deg2rad(jnp.array([-80.0, 0.0, 80.0]))
        coords = SimpleNamespace(
            horizontal=SimpleNamespace(latitudes=lat))
        term.cache_coords(coords)
        self.assertIsNone(term._taper_value())

    def test_taper_round_trips_through_save_load(self):
        import os
        import tempfile
        term = NNBiasCorrection(self.layers, self.mean, self.std, self.scale,
                                ("temperature",), polar_taper=(55.0, 75.0))
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "tapered.npz")
        term.save(path)
        loaded = NNBiasCorrection.from_file(path)
        self.assertEqual(loaded.polar_taper, (55.0, 75.0))

    def test_taperless_file_loads_with_none(self):
        import os
        import tempfile
        term = NNBiasCorrection(self.layers, self.mean, self.std, self.scale,
                                ("temperature",))
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "untapered.npz")
        term.save(path)
        loaded = NNBiasCorrection.from_file(path)
        self.assertIsNone(loaded.polar_taper)


# ---------------------------------------------------------------------------
# Context features: land-sea mask, orography, latitude, surface pressure
# ---------------------------------------------------------------------------


class TestContextFeatures(unittest.TestCase):
    """Per-column context inputs appended after the standardised profiles."""

    CTX = ("fmask", "orog", "sin_lat", "cos_lat", "ps")

    def setUp(self):
        self.nlev = 6
        self.shape = (self.nlev, 4, 3)          # (nlev, nlon, nlat)
        self.ncols = 4 * 3
        self.state = _state(self.shape, jax.random.key(0))
        self.terrain = SimpleNamespace(
            fmask=jnp.linspace(0.0, 1.0, 12).reshape(4, 3),
            orog=jnp.linspace(0.0, 3000.0, 12).reshape(4, 3))
        self.lat = jnp.deg2rad(jnp.array([-60.0, 0.0, 60.0]))
        self.mean, self.std, self.scale = _identity_buffers(self.nlev)

    def test_build_context_shapes_and_values(self):
        ctx = build_context(self.state, self.CTX,
                            terrain=self.terrain, lat=self.lat)
        self.assertEqual(ctx.shape, (self.ncols, len(self.CTX)))
        np.testing.assert_allclose(
            np.asarray(ctx[:, 0]),
            np.asarray(self.terrain.fmask).reshape(-1) - 0.5, rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(ctx[:, 1]),
            np.asarray(self.terrain.orog).reshape(-1) / 3000.0, rtol=1e-6)
        # sin_lat broadcasts over longitude: each lon row repeats the (nlat,)
        # pattern, flattened C-order.
        expect_sin = np.broadcast_to(np.sin(np.asarray(self.lat)),
                                     (4, 3)).reshape(-1)
        np.testing.assert_allclose(np.asarray(ctx[:, 2]), expect_sin,
                                   rtol=1e-6)
        expect_cos = np.broadcast_to(np.cos(np.asarray(self.lat)),
                                     (4, 3)).reshape(-1)
        np.testing.assert_allclose(np.asarray(ctx[:, 3]), expect_cos,
                                   rtol=1e-6)
        # ps is centred on 1 and the test state uses ones -> exactly zero.
        np.testing.assert_array_equal(np.asarray(ctx[:, 4]),
                                      np.zeros(self.ncols))

    def test_build_context_rejects_unknown_feature(self):
        with self.assertRaises(ValueError):
            build_context(self.state, ("no_such_feature",),
                          terrain=self.terrain, lat=self.lat)

    def test_widen_first_layer_is_exact_noop(self):
        # The warm-start guarantee: zero rows for the context inputs make the
        # widened net bit-identical to the original until training moves them.
        layers = init_mlp(jax.random.key(3), (24, 16, 24),
                          zero_last_layer=False)
        widened = widen_first_layer(layers, len(self.CTX))
        self.assertEqual(widened[0].kernel.shape, (24 + len(self.CTX), 16))
        x = jax.random.normal(jax.random.key(4), (7, 24))
        ctx = jax.random.normal(jax.random.key(5), (7, len(self.CTX)))
        y_orig = jax.vmap(lambda f: mlp(f, layers))(x)
        y_wide = jax.vmap(lambda f: mlp(f, widened))(
            jnp.concatenate([x, ctx], axis=1))
        np.testing.assert_array_equal(np.asarray(y_orig), np.asarray(y_wide))

    def test_config_covers_every_static_init_argument(self):
        # The guard that makes config()/rebuild() worth having. Adding a
        # constructor field and forgetting it in config() would silently
        # reintroduce exactly the drop-on-rebuild bug this pair exists to
        # prevent, and no behavioural test would catch it because the rebuilt
        # term would just use the default.
        import inspect
        params = inspect.signature(NNBiasCorrection.__init__).parameters
        array_args = {"self", "layers", "in_mean", "in_std", "out_scale"}
        static = {n for n in params if n not in array_args}
        term = make_bias_correction(nlev=8, hidden=(16, 16))
        self.assertEqual(set(term.config()), static)

    def test_rebuild_preserves_config(self):
        term = make_bias_correction(
            nlev=8, hidden=(16, 16), activation="gelu",
            context_features=("insol",), surface_taper=(0.7, 1.0),
            polar_taper=(60.0, 80.0), output_cap=3.0)
        clone = term.rebuild(term.weights.get_value())
        self.assertEqual(clone.config(), term.config())

    def test_rebuild_applies_overrides(self):
        term = make_bias_correction(nlev=8, hidden=(16, 16))
        clone = term.rebuild(term.weights.get_value(),
                             surface_taper=(0.7, 1.0))
        self.assertEqual(clone.surface_taper, (0.7, 1.0))
        self.assertIsNone(term.surface_taper)          # original untouched
        self.assertEqual(clone.activation, term.activation)

    def test_rebuild_rejects_an_unknown_field(self):
        # A typo'd override must fail loudly rather than being ignored, which
        # is the failure mode of passing kwargs straight to a constructor.
        term = make_bias_correction(nlev=8, hidden=(16, 16))
        with self.assertRaises(ValueError):
            term.rebuild(term.weights.get_value(), surfce_taper=(0.7, 1.0))

    def test_tanh_term_writes_no_activation_key(self):
        # Backward compatibility: tanh is the default and every artifact that
        # predates the registry is tanh, so a tanh term's file must stay
        # byte-identical in key set to the old format or old readers break.
        term = make_bias_correction(nlev=8, hidden=(16, 16))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.npz")
            term.save(path)
            self.assertNotIn("activation", set(np.load(path).files))

    def test_activation_round_trips_and_is_the_only_added_key(self):
        with tempfile.TemporaryDirectory() as d:
            tanh_p = os.path.join(d, "tanh.npz")
            gelu_p = os.path.join(d, "gelu.npz")
            make_bias_correction(nlev=8, hidden=(16, 16)).save(tanh_p)
            make_bias_correction(nlev=8, hidden=(16, 16),
                                 activation="gelu").save(gelu_p)
            keys_t = set(np.load(tanh_p).files)
            keys_g = set(np.load(gelu_p).files)
            self.assertEqual(keys_g - keys_t, {"activation"})
            self.assertEqual(load_bias_correction(gelu_p).activation, "gelu")
            self.assertEqual(load_bias_correction(tanh_p).activation, "tanh")

    def test_file_without_activation_key_loads_as_tanh(self):
        # The inference every pre-registry artifact depends on.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "old.npz")
            make_bias_correction(nlev=8, hidden=(16, 16)).save(path)
            self.assertEqual(load_bias_correction(path).activation, "tanh")

    def test_activation_actually_changes_the_output(self):
        # Guards against the flag being stored but never reaching the forward
        # pass, which would make every activation experiment silently a tanh
        # run.
        key = jax.random.key(0)
        kw = dict(nlev=8, hidden=(16, 16), key=key, zero_last_layer=False)
        t = make_bias_correction(**kw)
        g = make_bias_correction(**kw, activation="gelu")
        state = _state((8, 4, 3), jax.random.key(7))
        out_t, _ = t(state, {}, None, None)
        out_g, _ = g(state, {}, None, None)
        self.assertFalse(np.allclose(np.asarray(out_t.temperature),
                                     np.asarray(out_g.temperature)))

    def test_unknown_activation_raises(self):
        with self.assertRaises(ValueError):
            make_bias_correction(nlev=8, hidden=(16, 16), activation="selu")

    def test_remap_first_layer_agrees_with_widen_on_a_pure_append(self):
        # Appending a suffix is the case every shipped term actually used, so
        # the new path has to reproduce widen_first_layer exactly there.
        # Kernel is 24 profile rows + 1 context row ("insol").
        layers = init_mlp(jax.random.key(3), (25, 16, 24), zero_last_layer=False)

        widened = widen_first_layer(layers, 1)
        remapped = remap_first_layer(layers, 24, ("insol",), ("insol", "sice"))

        np.testing.assert_array_equal(np.asarray(remapped[0].kernel),
                                      np.asarray(widened[0].kernel))
        np.testing.assert_array_equal(np.asarray(remapped[0].bias),
                                      np.asarray(widened[0].bias))

    def test_remap_first_layer_survives_a_mid_tuple_insert(self):
        # The bug this replaces: context order is re-derived by filtering
        # CONTEXT_FEATURES, so adding `fmask` to a term trained with `insol`
        # yields ("fmask", "insol") -- a REORDER, not an append. Appending a
        # zero row at the bottom would leave the trained insol weights feeding
        # fmask. Keyed by name, insol keeps its own row.
        layers = init_mlp(jax.random.key(3), (25, 16, 24), zero_last_layer=False)
        insol_row = np.asarray(layers[0].kernel[24]).copy()

        out = remap_first_layer(layers, 24, ("insol",), ("fmask", "insol"))

        self.assertEqual(out[0].kernel.shape, (26, 16))
        # insol moved to slot 1, and slot 0 (the new fmask) is still zero.
        np.testing.assert_array_equal(np.asarray(out[0].kernel[25]), insol_row)
        np.testing.assert_array_equal(np.asarray(out[0].kernel[24]),
                                      np.zeros(16, dtype=insol_row.dtype))
        np.testing.assert_array_equal(np.asarray(out[0].kernel[:24]),
                                      np.asarray(layers[0].kernel[:24]))

    def test_remap_first_layer_rejects_a_wrong_row_count(self):
        # Silent mislabelling is the failure mode being designed out, so a
        # layout that does not match the kernel must raise rather than pad.
        layers = init_mlp(jax.random.key(3), (25, 16, 24), zero_last_layer=False)
        with self.assertRaises(ValueError):
            remap_first_layer(layers, 24, ("insol", "sice"), ("insol",))

    def test_widened_term_matches_unwidened_term(self):
        # Same guarantee at the term level, through cache_coords and __call__.
        layers = init_mlp(jax.random.key(6), (24, 16, 24),
                          zero_last_layer=False)
        base = NNBiasCorrection(layers, self.mean, self.std, self.scale,
                                ("temperature", "specific_humidity"))
        ctx_term = NNBiasCorrection(
            widen_first_layer(layers, len(self.CTX)),
            self.mean, self.std, self.scale,
            ("temperature", "specific_humidity"),
            context_features=self.CTX)
        coords = SimpleNamespace(
            horizontal=SimpleNamespace(latitudes=self.lat))
        ctx_term.cache_coords(coords)
        a, _ = base(self.state, {}, None, self.terrain)
        b, _ = ctx_term(self.state, {}, None, self.terrain)
        np.testing.assert_array_equal(np.asarray(a.temperature),
                                      np.asarray(b.temperature))
        np.testing.assert_array_equal(np.asarray(a.specific_humidity),
                                      np.asarray(b.specific_humidity))

    def test_gradient_reaches_context_rows(self):
        # After widening, the context rows must receive gradient (the context
        # inputs are non-zero, so d loss / d new_rows != 0). Standardise with
        # real feature statistics: with identity buffers the raw T ~ 250
        # saturates the first tanh exactly (tanh' == 0 in float32) and ALL
        # first-layer gradients vanish, which would mask the wiring.
        from jcm.physics.bias_correction.nn_bias_correction import (
            state_to_features)
        feats = state_to_features(self.state)
        mean = jnp.mean(feats, axis=0)
        std = jnp.std(feats, axis=0) + 1e-6
        layers = widen_first_layer(
            init_mlp(jax.random.key(7), (24, 16, 24), zero_last_layer=False),
            len(self.CTX))
        ctx = build_context(self.state, self.CTX,
                            terrain=self.terrain, lat=self.lat)

        def loss(ls):
            tend = bias_correction_tendency(
                self.state, ls, mean, std, self.scale,
                correct=("temperature",), context=ctx)
            return jnp.sum(tend.temperature ** 2)

        grads = jax.grad(loss)(layers)
        ctx_rows = grads[0].kernel[24:, :]
        self.assertEqual(ctx_rows.shape, (len(self.CTX), 16))
        self.assertGreater(float(jnp.max(jnp.abs(ctx_rows))), 0.0)

    def test_make_bias_correction_widens_input(self):
        term = make_bias_correction(nlev=self.nlev, context_features=self.CTX,
                                    zero_last_layer=False)
        first = term.weights.get_value()[0]
        self.assertEqual(first.kernel.shape[0],
                         4 * self.nlev + len(self.CTX))
        coords = SimpleNamespace(
            horizontal=SimpleNamespace(latitudes=self.lat))
        term.cache_coords(coords)
        tend, _ = term(self.state, {}, None, self.terrain)
        self.assertEqual(tend.temperature.shape, self.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(tend.temperature))))

    def test_context_round_trips_through_save_load(self):
        import os
        import tempfile
        term = make_bias_correction(nlev=self.nlev, context_features=self.CTX)
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "context.npz")
        term.save(path)
        loaded = NNBiasCorrection.from_file(path)
        self.assertEqual(loaded.context_features, self.CTX)
        self.assertEqual(loaded.weights.get_value()[0].kernel.shape[0],
                         4 * self.nlev + len(self.CTX))

    def test_contextless_file_loads_with_empty_tuple(self):
        import os
        import tempfile
        term = make_bias_correction(nlev=self.nlev)
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "plain.npz")
        term.save(path)
        loaded = NNBiasCorrection.from_file(path)
        self.assertEqual(loaded.context_features, ())


# ---------------------------------------------------------------------------
# The insolation context feature: the only time-varying input
# ---------------------------------------------------------------------------


class TestInsolationContext(unittest.TestCase):
    """Daily-mean TOA insolation as a season / surface-state proxy.

    Every other context feature is constant in time, so the network cannot
    tell a frozen January column from the same column in July. Insolation
    varies with latitude *and* time of year together, so unlike a bare
    day-of-year phase it separates northern winter from southern summer, and
    it collapses to zero through the polar night, which is the regime the
    winter-over-land bias lives in.
    """

    def setUp(self):
        # SPEEDY's vertical coordinates are defined for 7 or 8 levels only,
        # and `insol` reaches them via SpeedyCoords, so use the real 8.
        self.nlev = 8
        self.shape = (self.nlev, 4, 3)          # (nlev, nlon, nlat)
        self.ncols = 4 * 3
        self.state = _state(self.shape, jax.random.key(0))
        self.lat = jnp.deg2rad(jnp.array([-60.0, 0.0, 75.0]))
        self.speedy_coords = SpeedyCoords.single_column_coords(
            num_levels=self.nlev).copy(
                radang=self.lat, sia=jnp.sin(self.lat), coa=jnp.cos(self.lat))
        self.jan = jnp.asarray(0.04)            # mid-January
        self.jul = jnp.asarray(0.54)            # mid-July

    def _ctx(self, tyear):
        return build_context(self.state, ("insol",),
                             speedy_coords=self.speedy_coords, tyear=tyear)

    def test_matches_speedy_solar_scaled_and_centred(self):
        # Reuses the shortwave scheme's insolation rather than re-deriving it,
        # so this pins the two together.
        ctx = self._ctx(self.jan)
        self.assertEqual(ctx.shape, (self.ncols, 1))
        expect = np.broadcast_to(
            np.asarray(solar(self.jan, self.speedy_coords)) / solc - 1.0,
            (4, 3)).reshape(-1)
        np.testing.assert_allclose(np.asarray(ctx[:, 0]), expect, rtol=1e-6)

    def test_finite_and_order_one_across_the_whole_year(self):
        # The zero-row warm start does NOT protect against a NaN feature,
        # because 0 * NaN is NaN and would poison every gradient. The polar
        # arccos is the place this could bite, so sweep the full year against
        # the real T31 latitudes, not just the toy ones.
        real = get_speedy_coords()
        sc = SpeedyCoords.from_coordinate_system(real)
        state = _state((8,) + tuple(real.nodal_shape[1:]), jax.random.key(1))
        for tyear in np.linspace(0.0, 1.0, 25):
            ctx = np.asarray(build_context(
                state, ("insol",), speedy_coords=sc,
                tyear=jnp.asarray(float(tyear))))
            self.assertTrue(np.isfinite(ctx).all(), f"non-finite at {tyear}")
            self.assertLess(np.abs(ctx).max(), 2.0)

    def test_saturates_through_polar_night(self):
        # Mid-January at 75N: the sun never rises, so the raw insolation is
        # exactly zero and the centred feature sits at its floor of -1.
        ctx = np.asarray(self._ctx(self.jan)).reshape(4, 3)
        np.testing.assert_allclose(ctx[:, 2], -1.0, atol=1e-6)
        # ...and the same column in July is lit, which is the whole point.
        summer = np.asarray(self._ctx(self.jul)).reshape(4, 3)
        self.assertGreater(summer[0, 2], -0.5)

    def test_raises_clearly_when_inputs_are_missing(self):
        # A mis-wired call should fail with a readable message rather than an
        # AttributeError from inside a jit trace.
        with self.assertRaises(ValueError):
            build_context(self.state, ("insol",), speedy_coords=None,
                          tyear=None)

    def test_varies_with_time_of_year(self):
        # The property no other context feature has.
        jan = np.asarray(self._ctx(self.jan))
        jul = np.asarray(self._ctx(self.jul))
        self.assertGreater(np.abs(jan - jul).max(), 0.1)

    def test_hemispheres_are_opposite_in_january(self):
        # A bare sin/cos of day-of-year is a global scalar and could not do
        # this: in January the southern column must be sunnier than the
        # northern one.
        ctx = np.asarray(self._ctx(self.jan)).reshape(4, 3)
        self.assertGreater(ctx[0, 0], ctx[0, 2])     # 60S brighter than 75N

    def test_cache_coords_populates_speedy_coords_only_when_requested(self):
        coords = _fake_coords(self.nlev, self.lat)
        with_insol = make_bias_correction(nlev=self.nlev,
                                          context_features=("insol",))
        with_insol.cache_coords(coords)
        self.assertIsNotNone(with_insol._speedy_coords_value())
        without = make_bias_correction(nlev=self.nlev)
        without.cache_coords(coords)
        self.assertIsNone(without._speedy_coords_value())

    def test_term_call_is_seasonal_and_finite(self):
        # End to end through __call__, reading tyear off forcing.solar the way
        # the live model repopulates it every step.
        coords = _fake_coords(self.nlev, self.lat)
        term = make_bias_correction(nlev=self.nlev, key=jax.random.key(11),
                                    context_features=("insol",),
                                    zero_last_layer=False)
        term.cache_coords(coords)
        forcing_jan = SimpleNamespace(solar=SimpleNamespace(tyear=self.jan))
        forcing_jul = SimpleNamespace(solar=SimpleNamespace(tyear=self.jul))
        tend_jan, _ = term(self.state, {}, forcing_jan, None)
        tend_jul, _ = term(self.state, {}, forcing_jul, None)
        self.assertTrue(np.isfinite(np.asarray(tend_jan.temperature)).all())
        # Identical state, different season -> different correction. Without a
        # time-varying input these would be bit-identical.
        self.assertGreater(
            np.abs(np.asarray(tend_jan.temperature)
                   - np.asarray(tend_jul.temperature)).max(), 0.0)

    def test_round_trips_through_save_load(self):
        import os
        import tempfile
        term = make_bias_correction(nlev=self.nlev,
                                    context_features=("insol",))
        path = os.path.join(tempfile.mkdtemp(), "insol.npz")
        term.save(path)
        loaded = NNBiasCorrection.from_file(path)
        self.assertEqual(loaded.context_features, ("insol",))
        self.assertEqual(loaded.weights.get_value()[0].kernel.shape[0],
                         4 * self.nlev + 1)

    def test_is_appended_after_the_original_features(self):
        # widen_first_layer can only add kernel rows at the bottom, and a
        # term's feature order is re-derived by filtering this tuple.
        # Filtering preserves relative order, so APPENDING a new name is safe
        # but INSERTING one mid-tuple silently pairs every later feature with
        # the wrong kernel row of an already-trained term.
        #
        # The invariant is therefore the prefix, not that any one name is
        # last: everything up to and including insol must keep its position.
        self.assertEqual(
            CONTEXT_FEATURES[:6],
            ("fmask", "orog", "sin_lat", "cos_lat", "ps", "insol"))
        self.assertEqual(CONTEXT_FEATURES.index("insol"), 5)


class TestGradientFlow(unittest.TestCase):
    """Gradients must reach the weights, or the term cannot be trained."""

    def test_gradient_through_pure_function(self):
        nlev = 6
        state = _state((nlev, 5), jax.random.key(0))
        layers = init_mlp(jax.random.key(8), (24, 16, 24), zero_last_layer=False)
        mean, std, scale = _identity_buffers(nlev)

        def loss(L):
            tend = bias_correction_tendency(state, L, mean, std, scale,
                                            correct=("temperature", "specific_humidity"))
            return jnp.sum(tend.temperature ** 2) + jnp.sum(tend.specific_humidity ** 2)

        grads = jax.grad(loss)(layers)
        total = sum(float(jnp.sum(jnp.abs(g)))
                    for g in jax.tree_util.tree_leaves(grads))
        self.assertGreater(total, 0.0)

    def test_gradient_through_nnx_param(self):
        """The make-or-break check: weights stored in nnx.Param are trainable."""
        term = make_bias_correction(nlev=6, key=jax.random.key(9),
                                    zero_last_layer=False)
        state = _state((6, 5), jax.random.key(0))

        def loss(t):
            tend, _ = t(state, {}, None, None)
            return jnp.sum(tend.temperature ** 2) + jnp.sum(tend.specific_humidity ** 2)

        grads = nnx.grad(loss)(term)
        total = sum(float(jnp.sum(jnp.abs(g)))
                    for g in jax.tree_util.tree_leaves(grads)
                    if hasattr(g, "shape"))
        self.assertGreater(total, 0.0)


# ---------------------------------------------------------------------------
# Composition into SPEEDY
# ---------------------------------------------------------------------------


class TestComposesIntoSpeedy(unittest.TestCase):
    """The term snaps onto speedy_physics() with ``+`` and passes ordering."""

    def test_appends_and_validates(self):
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.physics.speedy.speedy_terms import speedy_physics
        from jcm.physics.composable_physics import ComposablePhysics

        coords = get_speedy_coords()
        physics = speedy_physics() + make_bias_correction(coords)

        self.assertIsInstance(physics, ComposablePhysics)
        names = [t.name for t in physics.terms]
        self.assertIn("nn_bias_correction", names)
        # appended last; requires=() means ordering validation already passed
        self.assertEqual(names[-1], "nn_bias_correction")


class TestNoOpVsSpeedy(unittest.TestCase):
    """Adding the zero-init term changes SPEEDY's tendencies by exactly nothing."""

    @pytest.mark.slow
    def test_speedy_tendencies_unchanged(self):
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.physics.speedy.speedy_terms import speedy_physics
        from jcm.forcing import ForcingData
        from jcm.terrain import TerrainData

        coords = get_speedy_coords()
        nodal_shape = coords.horizontal.nodal_shape
        nlev = coords.nodal_shape[0]
        shape_3d = (nlev,) + nodal_shape

        # Well-conditioned probe state (288 K, q=0); same pattern as
        # ComposablePhysics.get_empty_data so radiation terms avoid 0/0.
        probe = PhysicsState.zeros(shape_3d).copy(
            temperature=jnp.full(shape_3d, 288.0),
            normalized_surface_pressure=jnp.ones(nodal_shape),
        )
        forcing = ForcingData.zeros(nodal_shape)
        terrain = TerrainData.aquaplanet(coords)

        speedy = speedy_physics()
        speedy.cache_coords(coords)
        with_nn = speedy_physics() + make_bias_correction(coords)
        with_nn.cache_coords(coords)

        base, _ = speedy.compute_tendencies(probe, forcing, terrain)
        corrected, _ = with_nn.compute_tendencies(probe, forcing, terrain)

        for name in ("temperature", "specific_humidity", "u_wind", "v_wind"):
            np.testing.assert_array_equal(
                getattr(base, name), getattr(corrected, name),
                err_msg=f"{name} tendency changed by the no-op term",
            )


class TestShippedTermOnSpeedy(unittest.TestCase):
    """The shipped artifact loads, composes onto SPEEDY and runs on the current physics.

    Guards the file itself as much as the code: a term saved with a stale
    config key, a wrong input width for its context feature, or weights that
    blow up inside the real physics would all pass the synthetic tests above
    and fail here.
    """

    @pytest.mark.slow
    def test_shipped_term_loads_and_runs(self):
        from pathlib import Path

        import jax_datetime as jdt

        import jcm
        from jcm.forcing import ForcingData
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.physics.speedy.speedy_terms import speedy_physics
        from jcm.terrain import TerrainData

        data = Path(jcm.__file__).resolve().parent / "data"
        term = load_bias_correction(
            data / "bias_correction" / "online_term_t31_big_insol_vt.npz")
        cfg = term.config()
        self.assertEqual(tuple(cfg["context_features"]), ("insol",))
        self.assertEqual(tuple(cfg["surface_taper"]), (0.7, 1.0))
        self.assertEqual(tuple(cfg["correct"]),
                         ("temperature", "specific_humidity"))
        n_params = sum(int(np.prod(layer.kernel.shape)) + int(layer.bias.size)
                       for layer in term.weights.get_value())
        self.assertEqual(n_params, 148_512)

        coords = get_speedy_coords(layers=8, spectral_truncation=31)
        bc = data / "bc" / "t30" / "clim"
        terrain = TerrainData.from_file(bc / "terrain.nc", coords=coords)
        forcing = ForcingData.from_file(bc / "forcing.nc", coords=coords)
        model = Model(coords=coords, terrain=terrain,
                      physics=speedy_physics() + term,
                      start_date=jdt.to_datetime("2001-01-01"),
                      calendar="365_day", time_step=30)
        # Half a model day: enough steps for the correction, the insolation
        # context and the taper to run inside the real physics, cheap on CPU.
        ds = model.run(forcing=forcing, save_interval=0.5,
                       total_time=0.5).to_xarray()
        for name in ("temperature", "specific_humidity"):
            self.assertTrue(np.isfinite(np.asarray(ds[name].values)).all(),
                            f"{name} went non-finite with the shipped term")


# ---------------------------------------------------------------------------
# JIT
# ---------------------------------------------------------------------------


class TestJIT(unittest.TestCase):
    """The pure tendency helper compiles under jax.jit."""

    def test_pure_helper_jits(self):
        nlev = 6
        state = _state((nlev, 5), jax.random.key(0))
        layers = init_mlp(jax.random.key(5), (24, 16, 24), zero_last_layer=False)
        mean, std, scale = _identity_buffers(nlev)

        @jax.jit
        def run(s, L, m, sd, sc):
            return bias_correction_tendency(s, L, m, sd, sc,
                                            correct=("temperature",)).temperature

        out = run(state, layers, mean, std, scale)
        self.assertEqual(out.shape, (nlev, 5))


if __name__ == "__main__":
    unittest.main()


class TestGradientAgainstFiniteDifferences(unittest.TestCase):
    """The derivative is right, not merely present.

    Every other gradient test here asserts the gradient is finite and non-zero,
    which catches a severed graph but not a wrong derivative -- and a wrong
    derivative is the failure mode that matters, because online training
    backpropagates through hundreds of these calls and a subtly wrong gradient
    still descends, just to the wrong place. ``check_vjp`` / ``check_jvp``
    against finite differences is the repo convention for differentiable
    physics (see ``speedy_condensation_test``).

    Run in float64: a centred difference of a float32 function loses about half
    its significant digits, so a float32 check could only pass at a tolerance
    loose enough to also pass a genuinely wrong gradient. x64 is a global JAX
    setting, so it is turned on and off around this one test rather than left
    on for the module.
    """

    def setUp(self):
        jax.config.update("jax_enable_x64", True)

    def tearDown(self):
        jax.config.update("jax_enable_x64", False)

    def test_tendency_vjp_and_jvp_match_finite_differences(self):
        nlev, ncols = 3, 2
        nfeat = 4 * nlev

        state = _state((nlev, ncols), jax.random.key(11))
        layers = init_mlp(jax.random.key(12), (nfeat, 8, nfeat),
                          zero_last_layer=False)
        kernels = tuple(jnp.asarray(w.kernel, jnp.float64) for w in layers)
        biases = tuple(jnp.asarray(w.bias, jnp.float64) for w in layers)
        in_mean = jnp.zeros(nfeat, jnp.float64)
        in_std = jnp.ones(nfeat, jnp.float64)
        # Unit output scale keeps the tendency O(1). The real scale is
        # ~1e-5 K/s, and finite differences of a 1e-5-sized output are
        # dominated by rounding, which would test the tolerance rather
        # than the gradient.
        out_scale = jnp.ones(nfeat, jnp.float64)

        def f(kernels, biases, temperature, humidity):
            ws = [DenseWeights(kernel=k, bias=b)
                  for k, b in zip(kernels, biases)]
            st = PhysicsState(
                u_wind=jnp.asarray(state.u_wind, jnp.float64),
                v_wind=jnp.asarray(state.v_wind, jnp.float64),
                temperature=temperature,
                specific_humidity=humidity,
                geopotential=jnp.asarray(state.geopotential, jnp.float64),
                normalized_surface_pressure=jnp.asarray(
                    state.normalized_surface_pressure, jnp.float64),
                tracers={},
            )
            tend = bias_correction_tendency(
                st, ws, in_mean, in_std, out_scale)
            return tend.temperature, tend.specific_humidity

        args = (kernels, biases,
                jnp.asarray(state.temperature, jnp.float64),
                jnp.asarray(state.specific_humidity, jnp.float64))

        check_vjp(f, functools.partial(jax.vjp, f), args=args,
                  atol=1e-5, rtol=1e-5, eps=1e-6)
        check_jvp(f, functools.partial(jax.jvp, f), args=args,
                  atol=1e-5, rtol=1e-5, eps=1e-6)


