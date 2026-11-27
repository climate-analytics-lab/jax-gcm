"""Whole-column derivatives across cloud-base and plume-type choices."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from jcm.physics.convection.tiedtke_nordeng.half_levels import reconstruct_pressure_half
from jcm.physics.convection.tiedtke_nordeng.surrogate_gradients_test import _deep_column
from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
    _convection_surrogate, _tiedtke_convection_toa_first,
    tiedtke_nordeng_convection,
)
from jcm.physics.convection.tiedtke_nordeng.types import (
    ConvectionParameters, SURROGATE_WIDTH_FIELDS,
)


def _inputs():
    t, q, p, dz, rho = _deep_column()
    zeros = jnp.zeros_like(q)
    mass = jnp.diff(reconstruct_pressure_half(p)) / c.grav
    return dict(
        temperature=t, humidity=q, pressure=p, layer_thickness=dz, rho=rho,
        u_wind=zeros, v_wind=zeros, qc=zeros, qi=zeros, dt=jnp.asarray(900.),
        moisture_supply=jnp.asarray(3e-5),
        moisture_tend_profile=zeros.at[-4:].set(3e-5 / jnp.sum(mass[-4:])),
        qte_dynamics=zeros.at[10:16].set(3e-5 / jnp.sum(mass[10:16]))), mass


@pytest.mark.parametrize("choice", ["cloud_base", "deep_shallow"])
def test_column_derivative_follows_complete_branch_interpolation(choice):
    with jax.enable_x64(True):
        inputs, mass = _inputs()
        widths = {name: 0.0 for name in SURROGATE_WIDTH_FIELDS}
        if choice == "deep_shallow":
            widths["deep_convergence_width"] = 2e-7
        config = ConvectionParameters.default(
            cloud_base_interpolation=choice == "cloud_base", **widths)
        q = inputs["humidity"]
        previous = q - inputs["dt"] * jnp.zeros_like(q).at[10:16].set(
            (0.1 * inputs["moisture_supply"] + 1e-7) / jnp.sum(mass[10:16]))

        def output(x, surrogate=False):
            # Both state and parameter derivatives must reach every branch.
            column = dict(inputs, humidity=x[0] * q)
            params = config.replace(tau=config.tau * x[1])
            if choice == "deep_shallow":
                column["qte_dynamics"] = (column["humidity"] - previous) / inputs["dt"]
            tend, state = (_convection_surrogate(column, params) if surrogate
                           else tiedtke_nordeng_convection(**column, config=params))
            water = (jnp.sum((tend.dqdt + tend.dqc_dt + tend.dqi_dt) * mass)
                     + tend.precip_conv - tend.precip_floor_source)
            heat = jnp.sum(tend.dtedt * mass)
            return jnp.stack([tend.precip_conv, heat, water, jnp.sum(state.mfu)])

        exact = jax.jit(output)
        smooth = jax.jit(lambda x: output(x, True))
        forward = jax.jit(jax.jacfwd(output))
        reverse = jax.jit(jax.jacrev(output))
        x = jnp.asarray([0.93 if choice == "cloud_base" else 1.0, 1.0])
        step = 1e-7 if choice == "cloud_base" else 1e-9
        fd = jnp.stack([
            (smooth(x.at[i].add(step)) - smooth(x.at[i].add(-step))) / (2 * step)
            for i in range(2)], axis=1)
        np.testing.assert_allclose(forward(x), fd, rtol=2e-4, atol=1e-5)
        np.testing.assert_allclose(reverse(x), fd, rtol=2e-4, atol=1e-5)
        # Both nonlinear and tangent water ledgers close, including condensate.
        assert abs(float(smooth(x)[2])) < 1e-12
        np.testing.assert_allclose(forward(x)[2], 0.0, atol=1e-10)
        assert np.all(np.isfinite(np.asarray(forward(x))))
        assert float(smooth(x)[0]) > 0

        if choice == "cloud_base":
            # The exact 17 -> 18 base change is discontinuous. Interpolation
            # joins at the common candidate, without changing the forward run.
            edge = 0.9451928837221855
            left = jnp.asarray([edge - 1e-7, 1.0])
            right = jnp.asarray([edge + 1e-7, 1.0])
            assert abs(float(exact(right)[0] - exact(left)[0])) > 5e-4
            assert abs(float(smooth(right)[0] - smooth(left)[0])) < 1e-8
            column = dict(inputs, humidity=x[0] * q)
            reference = _tiedtke_convection_toa_first(**column, config=config)
            actual = tiedtke_nordeng_convection(**column, config=config)
            for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(reference)):
                np.testing.assert_array_equal(got, want)
            lower = _tiedtke_convection_toa_first(
                **column, config=config, _cloud_base=reference[1].kbase + 1)
            interpolated = _convection_surrogate(column, config)
            # The approximation error is bounded by the candidate separation.
            for value, a, b in zip(jax.tree.leaves(interpolated),
                                   jax.tree.leaves(reference), jax.tree.leaves(lower)):
                if jnp.issubdtype(value.dtype, jnp.floating):
                    np.testing.assert_array_less(
                        np.abs(value - a), np.abs(b - a) + 1e-10)


def test_batched_float32_gradients_and_vertical_ordering():
    inputs, _ = _inputs()
    inputs = jax.tree.map(lambda x: jnp.asarray(x, jnp.float32), inputs)
    config = ConvectionParameters.default()

    def output(scale, reverse=False):
        column = dict(inputs, humidity=scale * inputs["humidity"])
        if reverse:
            column = {k: (v[::-1] if v.ndim else v) for k, v in column.items()}
        tend, state = tiedtke_nordeng_convection(**column, config=config)
        return jnp.stack([tend.precip_conv, jnp.sum(tend.dtedt), jnp.sum(state.mfu)])

    def run(scale, reverse=False):
        return jax.jvp(lambda s: output(s, reverse), (scale,), (jnp.ones_like(scale),))

    scales = jnp.asarray([0., .7, .93, 1.05], jnp.float32)
    value, tangent = jax.jit(jax.vmap(run))(scales)
    flipped, flipped_tangent = jax.jit(jax.vmap(lambda s: run(s, True)))(scales)
    assert np.all(np.isfinite(np.asarray(tangent)))
    np.testing.assert_allclose(flipped, value, rtol=2e-5, atol=1e-8)
    np.testing.assert_allclose(flipped_tangent, tangent, rtol=2e-5, atol=1e-7)
    np.testing.assert_array_equal(value[0], 0.)
    np.testing.assert_array_equal(tangent[0], 0.)
