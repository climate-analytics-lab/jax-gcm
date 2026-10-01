"""Tests of the reference-exact value / surrogate-derivative wrapper."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from jcm.physics.surrogate_gradient import with_surrogate_gradient
from jcm.testing import check_surrogate_gradient

WIDTH = 0.02


def hard_clip(x):
    return jnp.clip(x, 0.0, 1.0)


def soft_clip(x):
    return (WIDTH * jax.nn.softplus(x / WIDTH)
            - WIDTH * jax.nn.softplus((x - 1.0) / WIDTH))


def hard_switch(temperature, liquid, ice):
    return jnp.where(temperature < 250.0, ice, liquid)


def soft_switch(temperature, liquid, ice):
    weight = jax.nn.sigmoid((temperature - 250.0) / 5.0)
    return weight * liquid + (1.0 - weight) * ice


def singular_power(x):
    return jnp.where(x > 0.0, jnp.maximum(x, 1e-30) ** 0.16, 0.0)


def bounded_power(x):
    # C1 continuation of x**0.16 below the cutoff: a parabola through the
    # origin that matches value and slope at the cutoff.
    cutoff = 1.0e-3
    t = jnp.minimum(x, cutoff) / cutoff
    low = cutoff ** 0.16 * ((2.0 - 0.16) * t + (0.16 - 1.0) * t * t)
    high = jnp.maximum(x, cutoff) ** 0.16
    return jnp.where(x < cutoff, low, high)


clip = with_surrogate_gradient(hard_clip, soft_clip)
switch = with_surrogate_gradient(hard_switch, soft_switch)
power = with_surrogate_gradient(singular_power, bounded_power)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_value_is_the_reference_bit_for_bit(dtype):
    with jax.enable_x64(dtype == jnp.float64):
        x = jnp.linspace(-0.5, 1.5, 4001, dtype=dtype)
        np.testing.assert_array_equal(clip(x), hard_clip(x))
        np.testing.assert_array_equal(jax.jit(clip)(x), hard_clip(x))
        # Exactly 0 and exactly 1 on the plateaux, which a smooth clip never
        # reaches.
        assert float(clip(jnp.asarray(-0.2, dtype))) == 0.0
        assert float(clip(jnp.asarray(1.2, dtype))) == 1.0


def test_derivative_is_the_surrogates_in_both_modes():
    x = jnp.linspace(-0.2, 1.2, 57)
    check_surrogate_gradient(clip, hard_clip, soft_clip, (x,))
    temperature = jnp.linspace(230.0, 270.0, 41)
    liquid = jnp.full(41, 2.0e-4)
    ice = jnp.full(41, 5.0e-5)
    check_surrogate_gradient(
        switch, hard_switch, soft_switch, (temperature, liquid, ice))
    density = jnp.array([0.0, 1e-20, 1e-8, 1e-4, 1e-3, 1e-2, 0.5])
    check_surrogate_gradient(
        power, singular_power, bounded_power, (density,))


def test_plateau_has_a_live_gradient_where_the_reference_has_none():
    # Just outside the clip the reference derivative is identically zero;
    # the surrogate's is small and positive, so an optimiser can see which
    # way the clipped quantity would move.
    below = jnp.asarray(-0.01)
    assert float(jax.grad(hard_clip)(below)) == 0.0
    assert float(jax.grad(clip)(below)) > 0.0
    # Far inside the linear range the two agree.
    inside = jnp.asarray(0.5)
    np.testing.assert_allclose(jax.grad(clip)(inside), 1.0, rtol=1e-6)


def test_switch_passes_a_gradient_to_the_switching_variable():
    args = (jnp.asarray(250.0), jnp.asarray(2.0e-4), jnp.asarray(5.0e-5))
    assert float(jax.grad(hard_switch)(*args)) == 0.0
    expected = jax.grad(soft_switch)(*args)
    assert float(expected) != 0.0
    np.testing.assert_array_equal(jax.grad(switch)(*args), expected)


def test_singular_slope_is_bounded():
    tiny = jnp.asarray(1.0e-20)
    assert float(jax.grad(singular_power)(tiny)) > 1.0e15
    assert float(jax.grad(power)(tiny)) < 1.0e3
    assert float(power(tiny)) == float(singular_power(tiny))


def test_integer_and_boolean_arguments_carry_no_tangent():
    def exact(x, mask, level):
        return jnp.where(mask, jnp.clip(x, 0.0, 1.0), 0.0) * level

    def surrogate(x, mask, level):
        return jnp.where(mask, soft_clip(x), 0.0) * level

    f = with_surrogate_gradient(exact, surrogate)
    x = jnp.linspace(-0.2, 1.2, 8)
    mask = jnp.arange(8) % 2 == 0
    level = jnp.arange(8, dtype=jnp.int32)
    np.testing.assert_array_equal(f(x, mask, level), exact(x, mask, level))
    gradient = jax.grad(lambda y: jnp.sum(f(y, mask, level)))(x)
    expected = jax.grad(lambda y: jnp.sum(surrogate(y, mask, level)))(x)
    np.testing.assert_array_equal(gradient, expected)


def test_composes_with_vmap_scan_and_jit():
    def column_sum(column):
        def step(carry, value):
            new = carry + clip(value)
            return new, new
        total, _ = lax.scan(step, jnp.zeros(()), column)
        return total

    def reference_sum(column):
        def step(carry, value):
            new = carry + soft_clip(value)
            return new, new
        total, _ = lax.scan(step, jnp.zeros(()), column)
        return total

    columns = jnp.linspace(-0.3, 1.3, 60).reshape(5, 12)
    value = jax.jit(jax.vmap(column_sum))(columns)
    np.testing.assert_allclose(
        value, jnp.sum(hard_clip(columns), axis=1), rtol=1e-6)
    gradient = jax.jit(jax.vmap(jax.grad(column_sum)))(columns)
    expected = jax.vmap(jax.grad(reference_sum))(columns)
    # The soft clip's slope above 1 is a difference of two sigmoids that are
    # both ~1, so in float32 it carries an absolute rounding of a few 1e-8
    # that differs between the jitted and the eager graph.
    np.testing.assert_allclose(gradient, expected, rtol=1e-5, atol=1e-7)


def test_second_derivative_is_the_surrogates():
    x = jnp.asarray(0.01)
    second = jax.grad(jax.grad(clip))(x)
    expected = jax.grad(jax.grad(soft_clip))(x)
    assert float(expected) != 0.0
    np.testing.assert_allclose(second, expected, rtol=1e-5)
