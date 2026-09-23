"""Shape, AD, and full-sweep tests for the ice fall-speed continuation."""

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
    """Exercise the explicit float64 cases even when JAX defaults to f32."""
    with jax.enable_x64():
        yield


def test_negative_continuation_cutoff_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        _ice_fall_speed_density_power(jnp.asarray(1.0e-12), 1.0e-30, -1.0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_continuation_is_monotone_safe_and_exact_above_join(dtype):
    """Both branches stay finite; resolved ice retains the original law."""
    x0 = jnp.asarray(_CUTOFF, dtype=dtype)
    values = jnp.asarray(
        [0.0, _CUTOFF * 1e-6, _CUTOFF * 0.5,
         _CUTOFF, _CUTOFF * (1 + 1e-5), 1e12],
        dtype=dtype,
    )
    actual = jax.vmap(
        lambda x: _ice_fall_speed_density_power(x, 1e-30, _CUTOFF)
    )(values)
    assert jnp.all(jnp.isfinite(actual))
    assert jnp.all(actual >= 0.0)
    assert jnp.all(jnp.diff(actual) >= 0.0)
    np.testing.assert_allclose(actual[3:], values[3:] ** _EXPONENT,
                               rtol=4e-6, atol=0.0)

    expected_join_slope = _EXPONENT * x0 ** (_EXPONENT - 1.0)
    join_slope = jax.grad(
        lambda x: _ice_fall_speed_density_power(x, 1e-30, _CUTOFF)
    )(x0)
    np.testing.assert_allclose(join_slope, expected_join_slope,
                               rtol=4e-6, atol=0.0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_origin_and_far_resolved_jvp_vjp_are_finite(dtype):
    """The enabled map has a bounded right slope and safe inactive branches."""
    f = lambda x: _ice_fall_speed_density_power(x, 1e-30, _CUTOFF)
    for value in (0.0, _CUTOFF, 1e12):
        x = jnp.asarray(value, dtype=dtype)
        primal, tangent = jax.jvp(f, (x,), (jnp.ones_like(x),))
        _, pullback = jax.vjp(f, x)
        cotangent = pullback(jnp.ones_like(primal))[0]
        assert jnp.isfinite(primal)
        assert jnp.isfinite(tangent)
        assert jnp.isfinite(cotangent)
    origin_slope = jax.grad(f)(jnp.asarray(0.0, dtype=dtype))
    expected = (2.0 - _EXPONENT) * jnp.asarray(
        _CUTOFF, dtype=dtype) ** (_EXPONENT - 1.0)
    np.testing.assert_allclose(origin_slope, expected, rtol=4e-6, atol=0.0)


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


def test_full_sweep_default_and_resolved_continuation_are_identical():
    """Refactoring is default-neutral and the joined law is exact above x0."""
    config = MicrophysicsParameters.default()
    column = _cold_column(jnp.full(6, 1.0e-5))
    default = cloud_microphysics_column_sweep(
        *column, 900.0, config)
    explicit_zero = cloud_microphysics_column_sweep(
        *column, 900.0, config, None, 0.0)
    bounded = cloud_microphysics_column_sweep(
        *column, 900.0, config, None, _CUTOFF)
    for reference, zero, candidate in zip(
            jax.tree.leaves(default), jax.tree.leaves(explicit_zero),
            jax.tree.leaves(bounded), strict=True):
        np.testing.assert_array_equal(zero, reference)
        np.testing.assert_array_equal(candidate, reference)


def test_trace_ice_full_sweep_closes_column_water_budget():
    """The enabled trace-ice path adds no column water source or sink."""
    config = MicrophysicsParameters.default()
    cloud_ice = jnp.asarray([2e-7, 2e-12, 0.0, 3e-11, 8e-6, 0.0])
    column = _cold_column(cloud_ice)
    tendency, state = cloud_microphysics_column_sweep(
        *column, 900.0, config, None, _CUTOFF)
    layer_mass = column[6] * column[7]
    water_tendency = jnp.sum(
        (tendency.dqdt + tendency.dqcdt + tendency.dqidt) * layer_mass)
    surface_precip = state.precip_rain + state.precip_snow
    residual = water_tendency + surface_precip
    # This trace column retains essentially all ice, so surface precipitation
    # is near zero and a relative residual is ill-conditioned. Test the
    # independently dimensioned kg/m2/s closure directly instead.
    assert float(jnp.abs(residual)) < 1e-10


def test_cutoff_is_static_under_jit():
    """The expert cutoff remains static when the NNX term crosses JIT."""
    term = Echam1MMicrophysics(
        ice_fall_speed_continuation_cutoff=_CUTOFF)
    power = nnx.jit(lambda module, x: _ice_fall_speed_density_power(
        x, 1.0e-30, module.ice_fall_speed_continuation_cutoff))
    assert jnp.isfinite(power(term, jnp.asarray(1.0e-12)))


def test_column_cutoff_is_static_under_jit():
    """A closure-captured cutoff compiles through the column sweep."""
    config = MicrophysicsParameters.default()
    column = _cold_column(jnp.asarray([1e-12, 2e-7, 1e-5]))
    run = jax.jit(lambda *args: cloud_microphysics_column_sweep(
        *args, 900.0, config, None, _CUTOFF)[0].dqidt)
    result = run(*column)
    assert jnp.all(jnp.isfinite(result))
