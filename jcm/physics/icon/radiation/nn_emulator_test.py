"""Tests for the neural network radiation emulator.

Covers: GRU cell, sequence processing, SW/LW emulators, flux
reconstruction, heating rates, weight initialization, and gradient
flow through the NN weights.

Date: 2026-04-11
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.icon.radiation.nn_emulator import (
    gru_cell,
    gru_forward_sequence,
    gru_backward_sequence,
    dense,
    softsign,
    sigmoid,
    preprocess_sw_inputs,
    preprocess_lw_inputs,
    sw_emulator_column,
    lw_emulator_column,
    reconstruct_sw_fluxes,
    reconstruct_lw_fluxes,
    flux_to_heating_rate,
    init_gru_weights,
    init_dense_weights,
    init_sw_emulator_weights,
    init_lw_emulator_weights,
    init_emulator_weights,
    DenseWeights,
    SWEmulatorWeights,
    LWEmulatorWeights,
    EmulatorWeights,
    InputScaling,
)


class TestActivations(unittest.TestCase):
    """Basic activation function sanity checks."""

    def test_softsign_values(self):
        x = jnp.array([-2.0, 0.0, 2.0])
        y = softsign(x)
        np.testing.assert_allclose(y, np.array([-2/3, 0.0, 2/3]), atol=1e-6)

    def test_sigmoid_range(self):
        x = jnp.linspace(-5, 5, 20)
        y = sigmoid(x)
        self.assertTrue(jnp.all(y >= 0.0))
        self.assertTrue(jnp.all(y <= 1.0))


class TestDense(unittest.TestCase):
    """Tests for the dense layer."""

    def test_output_shape(self):
        key = jax.random.key(0)
        w = init_dense_weights(4, 8, key)
        x = jnp.ones(4)
        y = dense(x, w)
        self.assertEqual(y.shape, (8,))

    def test_batched_output_shape(self):
        """TimeDistributed pattern: (seq_len, features)."""
        key = jax.random.key(0)
        w = init_dense_weights(4, 8, key)
        x = jnp.ones((10, 4))
        y = dense(x, w)
        self.assertEqual(y.shape, (10, 8))

    def test_activation_applied(self):
        w = DenseWeights(kernel=jnp.eye(2), bias=jnp.zeros(2))
        x = jnp.array([-1.0, 1.0])
        y = dense(x, w, activation=jax.nn.relu)
        np.testing.assert_allclose(y, np.array([0.0, 1.0]))


class TestGRUCell(unittest.TestCase):
    """Tests for a single GRU step."""

    def setUp(self):
        self.input_dim = 7
        self.units = 16
        self.key = jax.random.key(42)
        self.weights = init_gru_weights(self.input_dim, self.units, self.key)

    def test_output_shape(self):
        x = jnp.ones(self.input_dim)
        h = jnp.zeros(self.units)
        h_new = gru_cell(x, h, self.weights)
        self.assertEqual(h_new.shape, (self.units,))

    def test_zero_input_nonzero_output(self):
        """Even with zero input, the candidate gate should produce output."""
        x = jnp.zeros(self.input_dim)
        h = jnp.ones(self.units) * 0.5
        h_new = gru_cell(x, h, self.weights)
        self.assertFalse(jnp.allclose(h_new, jnp.zeros(self.units)))

    def test_deterministic(self):
        x = jax.random.normal(self.key, (self.input_dim,))
        h = jnp.zeros(self.units)
        h1 = gru_cell(x, h, self.weights)
        h2 = gru_cell(x, h, self.weights)
        np.testing.assert_array_equal(h1, h2)


class TestGRUSequence(unittest.TestCase):
    """Tests for forward and backward GRU sequence processing."""

    def setUp(self):
        self.input_dim = 7
        self.units = 16
        self.seq_len = 40
        self.key = jax.random.key(1)
        self.weights = init_gru_weights(self.input_dim, self.units, self.key)

    def test_forward_output_shape(self):
        x_seq = jnp.ones((self.seq_len, self.input_dim))
        h0 = jnp.zeros(self.units)
        hidden = gru_forward_sequence(x_seq, h0, self.weights)
        self.assertEqual(hidden.shape, (self.seq_len, self.units))

    def test_backward_output_shape(self):
        x_seq = jnp.ones((self.seq_len, self.input_dim))
        h0 = jnp.zeros(self.units)
        hidden = gru_backward_sequence(x_seq, h0, self.weights)
        self.assertEqual(hidden.shape, (self.seq_len, self.units))

    def test_backward_reversal(self):
        """Backward GRU reverses before processing, then reverses output."""
        k1, k2 = jax.random.split(self.key)
        x_seq = jax.random.normal(k1, (self.seq_len, self.input_dim))
        h0 = jnp.zeros(self.units)
        # backward sequence should match forward on reversed input, then reversed
        fwd_on_rev = gru_forward_sequence(x_seq[::-1], h0, self.weights)
        bwd = gru_backward_sequence(x_seq, h0, self.weights)
        np.testing.assert_allclose(bwd, fwd_on_rev[::-1], atol=1e-6)


class TestSWEmulator(unittest.TestCase):
    """Tests for the shortwave emulator column function."""

    def setUp(self):
        self.nlev = 40
        self.n_features = 7
        self.units = 16
        self.key = jax.random.key(10)
        self.weights = init_sw_emulator_weights(
            n_features=self.n_features, units=self.units, key=self.key
        )

    def test_output_shape(self):
        x_seq = jnp.ones((self.nlev, self.n_features))
        albedo = jnp.array([0.3])
        out = sw_emulator_column(x_seq, albedo, self.weights)
        self.assertEqual(out.shape, (self.nlev, 2))

    def test_output_in_0_1(self):
        """Sigmoid output should be in [0, 1]."""
        x_seq = jax.random.normal(self.key, (self.nlev, self.n_features))
        albedo = jnp.array([0.3])
        out = sw_emulator_column(x_seq, albedo, self.weights)
        self.assertTrue(jnp.all(out >= 0.0))
        self.assertTrue(jnp.all(out <= 1.0))


class TestLWEmulator(unittest.TestCase):
    """Tests for the longwave emulator column function."""

    def setUp(self):
        self.nlev = 40
        self.n_features = 7
        self.units = 16
        self.key = jax.random.key(20)
        self.weights = init_lw_emulator_weights(
            n_features=self.n_features, units=self.units, key=self.key
        )

    def test_output_shape(self):
        """LW emulator outputs at nlev+1 interfaces."""
        x_seq = jnp.ones((self.nlev, self.n_features))
        emissivity = jnp.array([0.97])
        out = lw_emulator_column(x_seq, emissivity, self.weights)
        self.assertEqual(out.shape, (self.nlev + 1, 2))

    def test_output_in_0_1(self):
        x_seq = jax.random.normal(self.key, (self.nlev, self.n_features))
        emissivity = jnp.array([0.97])
        out = lw_emulator_column(x_seq, emissivity, self.weights)
        self.assertTrue(jnp.all(out >= 0.0))
        self.assertTrue(jnp.all(out <= 1.0))


class TestFluxReconstruction(unittest.TestCase):
    """Tests for SW and LW flux reconstruction."""

    def test_sw_flux_shapes(self):
        nlev = 40
        nn_output = jnp.ones((nlev, 2)) * 0.5
        toa_sw_down = jnp.array(1361.0)
        albedo = jnp.array(0.3)
        rsd, rsu = reconstruct_sw_fluxes(nn_output, toa_sw_down, albedo)
        self.assertEqual(rsd.shape, (nlev + 1,))
        self.assertEqual(rsu.shape, (nlev + 1,))

    def test_sw_toa_boundary(self):
        """TOA downwelling should equal toa_sw_down."""
        nlev = 40
        nn_output = jnp.ones((nlev, 2)) * 0.5
        toa_sw_down = jnp.array(1361.0)
        albedo = jnp.array(0.3)
        rsd, _ = reconstruct_sw_fluxes(nn_output, toa_sw_down, albedo)
        np.testing.assert_allclose(rsd[0], 1361.0, atol=1e-5)

    def test_sw_surface_reflection(self):
        """Surface upwelling should equal albedo * surface downwelling."""
        nlev = 10
        nn_output = jnp.ones((nlev, 2)) * 0.8
        toa_sw_down = jnp.array(1000.0)
        albedo = jnp.array(0.25)
        rsd, rsu = reconstruct_sw_fluxes(nn_output, toa_sw_down, albedo)
        np.testing.assert_allclose(rsu[-1], albedo * rsd[-1], atol=1e-5)

    def test_lw_flux_shapes(self):
        nlev = 40
        nn_output = jnp.ones((nlev + 1, 2)) * 0.5
        surface_temp = jnp.array(288.0)
        emissivity = jnp.array(0.97)
        rld, rlu = reconstruct_lw_fluxes(nn_output, surface_temp, emissivity)
        self.assertEqual(rld.shape, (nlev + 1,))
        self.assertEqual(rlu.shape, (nlev + 1,))

    def test_lw_fluxes_positive(self):
        """LW fluxes should be non-negative for positive NN output."""
        nlev = 40
        nn_output = jnp.ones((nlev + 1, 2)) * 0.5
        surface_temp = jnp.array(288.0)
        emissivity = jnp.array(0.97)
        rld, rlu = reconstruct_lw_fluxes(nn_output, surface_temp, emissivity)
        self.assertTrue(jnp.all(rld >= 0.0))
        self.assertTrue(jnp.all(rlu >= 0.0))


class TestHeatingRate(unittest.TestCase):
    """Tests for flux → heating rate conversion."""

    def test_output_shape(self):
        nlev = 40
        flux_down = jnp.linspace(1000, 800, nlev + 1)
        flux_up = jnp.linspace(200, 300, nlev + 1)
        p_int = jnp.linspace(100.0, 101325.0, nlev + 1)
        hr = flux_to_heating_rate(flux_down, flux_up, p_int)
        self.assertEqual(hr.shape, (nlev,))

    def test_zero_flux_divergence(self):
        """Constant net flux → zero heating rate."""
        nlev = 10
        flux_down = jnp.ones(nlev + 1) * 500.0
        flux_up = jnp.ones(nlev + 1) * 200.0
        p_int = jnp.linspace(100.0, 101325.0, nlev + 1)
        hr = flux_to_heating_rate(flux_down, flux_up, p_int)
        np.testing.assert_allclose(hr, 0.0, atol=1e-10)


class TestPreprocessing(unittest.TestCase):
    """Tests for input preprocessing."""

    def test_sw_preprocessing_shape(self):
        nlev = 40
        scaling = InputScaling(x_max=jnp.ones(7) * 1000.0)
        x = preprocess_sw_inputs(
            temperature=jnp.full(nlev, 250.0),
            pressure=jnp.linspace(100, 101325, nlev),
            h2o=jnp.full(nlev, 0.01),
            o3=jnp.full(nlev, 5e-6),
            cloud_water=jnp.zeros(nlev),
            cloud_ice=jnp.zeros(nlev),
            cos_zenith=jnp.array(0.5),
            scaling=scaling,
        )
        self.assertEqual(x.shape, (nlev, 7))

    def test_lw_preprocessing_shape(self):
        nlev = 40
        scaling = InputScaling(x_max=jnp.ones(7) * 1000.0)
        x = preprocess_lw_inputs(
            temperature=jnp.full(nlev, 250.0),
            pressure=jnp.linspace(100, 101325, nlev),
            h2o=jnp.full(nlev, 0.01),
            o3=jnp.full(nlev, 5e-6),
            cloud_water=jnp.zeros(nlev),
            cloud_ice=jnp.zeros(nlev),
            co2_vmr=400e-6,
            scaling=scaling,
        )
        self.assertEqual(x.shape, (nlev, 7))


class TestWeightInitialization(unittest.TestCase):
    """Tests for random weight initialization."""

    def test_gru_shapes(self):
        w = init_gru_weights(7, 16, jax.random.key(0))
        self.assertEqual(w.kernel.shape, (7, 48))
        self.assertEqual(w.recurrent_kernel.shape, (16, 48))
        self.assertEqual(w.bias.shape, (2, 48))

    def test_dense_shapes(self):
        w = init_dense_weights(4, 8, jax.random.key(0))
        self.assertEqual(w.kernel.shape, (4, 8))
        self.assertEqual(w.bias.shape, (8,))

    def test_sw_emulator_init(self):
        w = init_sw_emulator_weights(n_features=7, units=16)
        self.assertEqual(w.gru_fwd.kernel.shape, (7, 48))
        self.assertEqual(w.gru2.kernel.shape, (32, 48))
        self.assertEqual(w.output_dense.kernel.shape, (16, 2))

    def test_lw_emulator_init(self):
        w = init_lw_emulator_weights(n_features=7, units=16)
        self.assertEqual(w.gru_fwd.kernel.shape, (7, 48))
        self.assertEqual(w.surface_dense.kernel.shape, (17, 16))  # units + 1
        self.assertEqual(w.gru3.kernel.shape, (32, 48))

    def test_full_emulator_init(self):
        w = init_emulator_weights()
        self.assertIsInstance(w, EmulatorWeights)
        self.assertIsInstance(w.sw, SWEmulatorWeights)
        self.assertIsInstance(w.lw, LWEmulatorWeights)


class TestGradientFlow(unittest.TestCase):
    """Test that gradients flow through the NN weights."""

    def test_sw_gradient_wrt_weights(self):
        """jax.grad of a scalar loss w.r.t. SW emulator weights should be non-zero."""
        nlev = 10
        n_features = 7
        units = 8
        weights = init_sw_emulator_weights(
            n_features=n_features, units=units, key=jax.random.key(99)
        )
        x_seq = jnp.ones((nlev, n_features)) * 0.5
        albedo = jnp.array([0.3])

        def loss_fn(w):
            out = sw_emulator_column(x_seq, albedo, w)
            return jnp.sum(out)

        grads = jax.grad(loss_fn)(weights)
        # Check that at least one gradient is non-zero
        grad_flat = jax.tree_util.tree_leaves(grads)
        total_abs = sum(jnp.sum(jnp.abs(g)) for g in grad_flat)
        self.assertGreater(float(total_abs), 0.0)

    def test_lw_gradient_wrt_weights(self):
        """jax.grad of a scalar loss w.r.t. LW emulator weights should be non-zero."""
        nlev = 10
        n_features = 7
        units = 8
        weights = init_lw_emulator_weights(
            n_features=n_features, units=units, key=jax.random.key(100)
        )
        x_seq = jnp.ones((nlev, n_features)) * 0.5
        emissivity = jnp.array([0.97])

        def loss_fn(w):
            out = lw_emulator_column(x_seq, emissivity, w)
            return jnp.sum(out)

        grads = jax.grad(loss_fn)(weights)
        grad_flat = jax.tree_util.tree_leaves(grads)
        total_abs = sum(jnp.sum(jnp.abs(g)) for g in grad_flat)
        self.assertGreater(float(total_abs), 0.0)

    def test_full_pipeline_gradient(self):
        """Gradient through SW emulator + flux reconstruction + heating rate."""
        nlev = 10
        n_features = 7
        units = 8
        sw_weights = init_sw_emulator_weights(
            n_features=n_features, units=units, key=jax.random.key(101)
        )
        p_int = jnp.linspace(100.0, 101325.0, nlev + 1)

        def loss_fn(w):
            x_seq = jnp.ones((nlev, n_features)) * 0.5
            albedo = jnp.array([0.3])
            nn_out = sw_emulator_column(x_seq, albedo, w)
            toa_sw = jnp.array(1361.0)
            rsd, rsu = reconstruct_sw_fluxes(nn_out, toa_sw, jnp.array(0.3))
            hr = flux_to_heating_rate(rsd, rsu, p_int)
            return jnp.mean(hr ** 2)

        grads = jax.grad(loss_fn)(sw_weights)
        grad_flat = jax.tree_util.tree_leaves(grads)
        total_abs = sum(jnp.sum(jnp.abs(g)) for g in grad_flat)
        self.assertGreater(float(total_abs), 0.0)


class TestJITCompatibility(unittest.TestCase):
    """Ensure all main functions can be JIT-compiled."""

    def test_sw_emulator_jit(self):
        weights = init_sw_emulator_weights(n_features=7, units=8)
        x_seq = jnp.ones((20, 7))
        albedo = jnp.array([0.3])

        @jax.jit
        def run(x, a, w):
            return sw_emulator_column(x, a, w)

        out = run(x_seq, albedo, weights)
        self.assertEqual(out.shape, (20, 2))

    def test_lw_emulator_jit(self):
        weights = init_lw_emulator_weights(n_features=7, units=8)
        x_seq = jnp.ones((20, 7))
        emissivity = jnp.array([0.97])

        @jax.jit
        def run(x, e, w):
            return lw_emulator_column(x, e, w)

        out = run(x_seq, emissivity, weights)
        self.assertEqual(out.shape, (21, 2))

    def test_heating_rate_jit(self):
        nlev = 20
        p_int = jnp.linspace(100, 101325, nlev + 1)
        fd = jnp.linspace(1000, 800, nlev + 1)
        fu = jnp.linspace(200, 300, nlev + 1)

        hr = jax.jit(flux_to_heating_rate)(fd, fu, p_int)
        self.assertEqual(hr.shape, (nlev,))


if __name__ == "__main__":
    unittest.main()
