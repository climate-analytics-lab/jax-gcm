"""The diagnostic's first buoyant layer, including LFC/EL and base changes."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .column_surrogate_test import _inputs
from .switches import cape_cin_window
from .tiedtke_nordeng import (
    _calculate_cape_cin, _cape_cin_surrogate, calculate_cape_cin, find_cloud_base,
)
from .types import ConvectionParameters


def test_soft_window_is_the_expected_first_buoyant_layer():
    """Enumerate paths independently of the scan; later layers cannot restart."""
    b = jnp.asarray([-.04, .01, -.02, .08, -.03])
    dz = jnp.asarray([100., 200., 300., 400., 500.])
    width = .01
    mask = jnp.asarray([False, True, True, True, True])
    prob = np.where(mask, jax.nn.sigmoid(b / width), 0.)
    positive = np.asarray(width * jax.nn.softplus(b / width) * dz)
    negative = np.asarray(width * jax.nn.softplus(-b / width) * dz)
    expected = np.zeros(2)
    for path in itertools.product((False, True), repeat=b.size):
        p = np.prod(np.where(path, prob, 1-prob))
        active = np.flatnonzero(path)
        if active.size:
            lfc = active[0]
            el = next((k for k in range(lfc+1, b.size) if not path[k]), b.size)
            expected += p * np.array([
                positive[lfc:el].sum(), negative[:lfc][np.asarray(mask[:lfc])].sum()])
    np.testing.assert_allclose(cape_cin_window(b, dz, mask, width), expected, rtol=2e-6)
    # With a narrow width, a negative level ends the first layer even if
    # a much stronger positive layer follows it.
    b = jnp.asarray([-.2, .4, -.3, 2., -.2])
    got = cape_cin_window(b, dz, jnp.ones(5, bool), 1e-4)
    np.testing.assert_allclose(got, [80., 20.], rtol=2e-6)
    dry = cape_cin_window(-jnp.ones(5), dz, jnp.ones(5, bool), .01)
    np.testing.assert_allclose(dry, 0., atol=1e-30)
    with jax.enable_x64(True):
        near = jnp.asarray([-.01, .002, 0., .003, -.004], jnp.float64)
        loss = lambda x: jnp.sum(jnp.stack(cape_cin_window(x, dz, mask, .01)))
        hessian = jax.hessian(loss)(near)
        assert np.all(np.isfinite(hessian))
        np.testing.assert_allclose(hessian, hessian.T, rtol=1e-12, atol=1e-8)


@pytest.mark.parametrize("transition", ["lfc", "el", "cloud_base"])
def test_diagnostic_values_stay_exact_and_gradients_follow_surrogate(transition):
    with jax.enable_x64(True):
        inputs, _ = _inputs()
        t, q, p, dz = (inputs[k] for k in (
            "temperature", "humidity", "pressure", "layer_thickness"))
        config = ConvectionParameters.default(cloud_base_interpolation=transition == "cloud_base")

        def result(x, fn=calculate_cape_cin, reverse=False):
            temp = t if transition == "cloud_base" else t.at[18 if transition == "lfc" else 17].add(x)
            humid = x*q if transition == "cloud_base" else q
            base, _ = find_cloud_base(temp, humid, p, config)
            if reverse:
                return jnp.stack(fn(temp[::-1], humid[::-1], p[::-1], dz[::-1],
                                    t.size-1-base, config))
            return jnp.stack(fn(temp, humid, p, dz, base, config))

        point = jnp.asarray({"el": 12.21287546714738,
                             "lfc": 6.709094992611426, "cloud_base": .93}[transition])
        exact = jax.jit(result)
        smooth = jax.jit(lambda x: result(x, _cape_cin_surrogate))
        reference = jax.jit(lambda x: result(x, _calculate_cape_cin))
        for x in (point-.001, point, point+.001):
            np.testing.assert_array_equal(exact(x), reference(x))
        forward = jax.jit(jax.jacfwd(result))(point)
        reverse = jax.jit(jax.jacrev(result))(point)
        for step in (1e-5, 5e-6, 2.5e-6):
            fd = (smooth(point+step)-smooth(point-step))/(2*step)
            np.testing.assert_allclose(forward, fd, rtol=2e-5, atol=1e-5)
        np.testing.assert_allclose(reverse, forward, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(jax.jacfwd(lambda x: result(x, reverse=True))(point),
                                   forward, rtol=1e-12, atol=1e-9)
        if transition == "el":
            assert abs(float(forward[0])) > 1000.
            assert abs(float(exact(point-.001)[0] - exact(point+.001)[0])) > 1e4
            assert abs(float(smooth(point-.001)[0] - smooth(point+.001)[0])) < 100.
        elif transition == "lfc":
            assert float(exact(point-.001)[1]) == 0.
            assert float(exact(point+.001)[1]) > 0.
            assert float(smooth(point)[1]) > 0.
        else:
            edge = .9451928837221855
            assert abs(float(exact(edge-1e-7)[0] - exact(edge+1e-7)[0])) > 10.
            assert abs(float(smooth(edge-1e-7)[0] - smooth(edge+1e-7)[0])) < .1


def test_zero_width_disables_the_window_surrogate():
    inputs, _ = _inputs()
    args = [inputs[k] for k in ("temperature", "humidity", "pressure", "layer_thickness")]
    config = ConvectionParameters.default(cloud_base_interpolation=False, cape_buoyancy_width=0.)
    base, _ = find_cloud_base(*args[:3], config)
    for fn in (calculate_cape_cin, _calculate_cape_cin):
        value, grad = jax.jvp(lambda t: jnp.stack(fn(t, *args[1:], base, config)),
                             (args[0],), (jnp.ones_like(args[0]),))
        if fn is calculate_cape_cin:
            reference_value, reference_grad = value, grad
        else:
            np.testing.assert_array_equal(value, reference_value)
            np.testing.assert_array_equal(grad, reference_grad)


def test_wetbulb_has_live_derivatives_without_the_old_conditional():
    """The wet-bulb branch cited in #665 is now the continuous cuadjtq clip."""
    from .cuadjtq import saturation_mixing_ratio
    from .downdraft import wetbulb_temperature

    with jax.enable_x64(True):
        t, p = jnp.asarray(285.), jnp.asarray(85000.)
        qs = saturation_mixing_ratio(p, t)
        output = lambda rh: jnp.stack(wetbulb_temperature(t, rh*qs, p))
        for rh in (.8, .999, 1.001):
            x = jnp.asarray(rh)
            fd = (output(x+1e-6)-output(x-1e-6))/(2e-6)
            grad = jax.jacfwd(output)(x)
            np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-8)
            assert np.all(np.isfinite(grad))
            assert float(grad[1]) > 0.
