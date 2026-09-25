"""Forward-parity and AD tests for the ice fall-speed surrogate derivative."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from .echam_1m import (
    Echam1MMicrophysics,
    MicrophysicsParameters,
    _ice_fall_speed_density_power,
    cloud_microphysics_column_sweep,
)
from .sundqvist import saturation_specific_humidity


_CUTOFF = 1.0e-10
_EXPONENT = 0.16


@pytest.fixture(autouse=True, scope="module")
def _enable_x64():
    """Exercise explicit float64 cases even when JAX defaults to float32."""
    with jax.enable_x64():
        yield


def _legacy_power(x, d_epsilon):
    return jnp.maximum(x, d_epsilon) ** _EXPONENT


def _gated_power(x, cutoff):
    power = _ice_fall_speed_density_power(x, 1.0e-30, cutoff)
    return jnp.where(x > 0.0, power, 0.0)


def test_invalid_derivative_cutoff_is_rejected_in_primal_and_ad():
    for cutoff in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="derivative_cutoff"):
            _ice_fall_speed_density_power(jnp.asarray(1.0e-12), 1.0e-30, cutoff)
        with pytest.raises(ValueError, match="derivative_cutoff"):
            jax.jvp(
                lambda x: _ice_fall_speed_density_power(x, 1.0e-30, cutoff),
                (jnp.asarray(1.0e-12),),
                (jnp.asarray(1.0),),
            )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_forward_is_exact_legacy_for_scalar_and_array(dtype):
    values = jnp.asarray(
        [0.0, 1.0e-30, 1.0e-20, _CUTOFF, 1.0e-6], dtype=dtype)
    epsilon = jnp.asarray(1.0e-30, dtype=dtype)
    expected = _legacy_power(values, epsilon)
    np.testing.assert_array_equal(
        _ice_fall_speed_density_power(values, epsilon), expected)
    np.testing.assert_array_equal(
        _ice_fall_speed_density_power(values, epsilon, 0.0), expected)
    for value, reference in zip(values, expected, strict=True):
        np.testing.assert_array_equal(
            _ice_fall_speed_density_power(value, epsilon), reference)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_surrogate_jvp_vjp_are_finite_and_transposes(dtype):
    x = jnp.asarray([1e-30, 1e-20, _CUTOFF, 1e-6], dtype=dtype)
    x_dot = jnp.asarray([0.25, -0.5, 0.75, 1.25], dtype=dtype)
    cotangent = jnp.asarray([-0.2, 0.4, 0.6, -0.8], dtype=dtype)
    epsilon = jnp.asarray(1.0e-30, dtype=dtype)
    epsilon_dot = jnp.asarray(0.3, dtype=dtype)
    fn = lambda a, b: _ice_fall_speed_density_power(a, b, _CUTOFF)
    primal, tangent = jax.jvp(fn, (x, epsilon), (x_dot, epsilon_dot))
    _, pullback = jax.vjp(fn, x, epsilon)
    x_bar, epsilon_bar = pullback(cotangent)
    assert jnp.all(jnp.isfinite(primal))
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.all(jnp.isfinite(x_bar))
    assert jnp.isfinite(epsilon_bar)
    np.testing.assert_allclose(
        jnp.vdot(tangent, cotangent),
        jnp.vdot(x_dot, x_bar) + epsilon_dot * epsilon_bar,
        rtol=4e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_below_cutoff_uses_c1_continuation_slope(dtype):
    cutoff = 2.0e-9
    x = jnp.asarray(0.5 * cutoff, dtype=dtype)
    actual = jax.grad(
        lambda y: _ice_fall_speed_density_power(y, 1.0e-30, cutoff))(x)
    t = x / jnp.asarray(cutoff, dtype=dtype)
    expected = jnp.asarray(cutoff, dtype=dtype) ** (_EXPONENT - 1.0) * (
        (2.0 - _EXPONENT) + 2.0 * (_EXPONENT - 1.0) * t)
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=0.0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_cutoff_zero_and_resolved_ice_use_true_legacy_derivative(dtype):
    epsilon = jnp.asarray(1.0e-30, dtype=dtype)
    for value in (1.0e-20, _CUTOFF, 1.0e-6):
        x = jnp.asarray(value, dtype=dtype)
        legacy = jax.grad(lambda y: _legacy_power(y, epsilon))(x)
        zero = jax.grad(
            lambda y: _ice_fall_speed_density_power(y, epsilon, 0.0))(x)
        np.testing.assert_array_equal(zero, legacy)
        if value >= _CUTOFF:
            surrogate = jax.grad(
                lambda y: _ice_fall_speed_density_power(y, epsilon))(x)
            np.testing.assert_allclose(surrogate, legacy, rtol=4e-6, atol=0.0)


def test_d_epsilon_tangent_is_preserved():
    x = jnp.asarray(0.5e-20, dtype=jnp.float64)
    epsilon = jnp.asarray(1.0e-20, dtype=jnp.float64)
    fn = lambda e: _ice_fall_speed_density_power(x, e)
    actual = jax.grad(fn)(epsilon)
    expected = _EXPONENT * epsilon ** (_EXPONENT - 1.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


def test_resolved_derivative_respects_unusual_d_epsilon_floor():
    cutoff = 1.0e-10
    x = jnp.asarray(2.0e-10, dtype=jnp.float64)
    epsilon = jnp.asarray(4.0e-10, dtype=jnp.float64)
    actual = jax.grad(
        lambda y: _ice_fall_speed_density_power(y, epsilon, cutoff))(x)
    assert actual == 0.0


def test_original_outer_gate_has_exact_zero_value_and_derivative():
    for cutoff in (0.0, _CUTOFF):
        zero = jnp.asarray(0.0)
        assert _gated_power(zero, cutoff) == 0.0
        assert jax.grad(lambda x: _gated_power(x, cutoff))(zero) == 0.0


def _cold_column(cloud_ice):
    nlev = cloud_ice.size
    temperature = jnp.linspace(225.0, 250.0, nlev)
    pressure = jnp.linspace(25000.0, 80000.0, nlev)
    humidity = 0.9 * jax.vmap(saturation_specific_humidity)(
        pressure, temperature)
    cloud_water = jnp.zeros(nlev)
    cloud_fraction = jnp.full(nlev, 0.7)
    air_density = pressure / (287.0 * temperature)
    layer_thickness = jnp.full(nlev, 500.0)
    droplet_number = jnp.full(nlev, 1.0e8)
    return (temperature, humidity, pressure, cloud_water, cloud_ice,
            cloud_fraction, air_density, layer_thickness, droplet_number)


def test_full_sweep_forward_parity_but_gradient_difference():
    config = MicrophysicsParameters.default()
    ice = jnp.asarray([2e-7, 2e-12, 1e-20, 3e-11, 8e-6, 0.0])

    def objective(cloud_ice, cutoff):
        tendency, state = cloud_microphysics_column_sweep(
            *_cold_column(cloud_ice), 900.0, config, None, cutoff)
        return jnp.sum(tendency.dqidt) + state.precip_snow

    default = cloud_microphysics_column_sweep(
        *_cold_column(ice), 900.0, config)
    legacy = cloud_microphysics_column_sweep(
        *_cold_column(ice), 900.0, config, None, 0.0)
    for candidate, reference in zip(
            jax.tree.leaves(default), jax.tree.leaves(legacy), strict=True):
        np.testing.assert_array_equal(candidate, reference)

    surrogate_grad = jax.grad(lambda x: objective(x, _CUTOFF))(ice)
    legacy_grad = jax.grad(lambda x: objective(x, 0.0))(ice)
    assert jnp.all(jnp.isfinite(surrogate_grad))
    assert jnp.any(surrogate_grad != legacy_grad)


def test_default_term_derivative_cutoff_is_static_under_jit():
    term = Echam1MMicrophysics()
    assert term.ice_fall_speed_derivative_cutoff == _CUTOFF
    power = nnx.jit(lambda module, x: _ice_fall_speed_density_power(
        x, 1.0e-30, module.ice_fall_speed_derivative_cutoff))
    expected = _legacy_power(jnp.asarray(1.0e-12), 1.0e-30)
    np.testing.assert_array_equal(power(term, jnp.asarray(1.0e-12)), expected)


def test_column_derivative_cutoff_is_static_under_jit():
    config = MicrophysicsParameters.default()
    column = _cold_column(jnp.asarray([1e-12, 2e-7, 1e-5]))
    run = jax.jit(lambda *args: cloud_microphysics_column_sweep(
        *args, 900.0, config, None, _CUTOFF)[0].dqidt)
    result = run(*column)
    assert jnp.all(jnp.isfinite(result))
