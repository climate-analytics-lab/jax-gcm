"""Tests for shared cloud diagnostics."""

import jax.numpy as jnp

from jcm.physics.clouds.cloud_data import CloudData, radiation_cloud_fields
from jcm.physics_interface import PhysicsState


def test_radiation_cloud_fields_match_echam_cover_then_radiation_order():
    """Radiation uses fresh cover but pre-cloud-step condensate tracers."""
    nlev, ncols = 3, 2
    shape = (nlev, ncols)
    tracer_qc = jnp.full(shape, 1.0e-9)
    tracer_qi = jnp.full(shape, 2.0e-9)
    post_cloud_qc = jnp.arange(nlev * ncols, dtype=jnp.float32).reshape(shape) * 1e-5
    post_cloud_qi = post_cloud_qc + 1e-4
    diagnosed_cf = jnp.clip(post_cloud_qc * 1e4, 0.0, 1.0)

    state = PhysicsState.zeros(
        shape,
        temperature=jnp.ones(shape) * 280.0,
        specific_humidity=jnp.ones(shape) * 1e-3,
        tracers={"qc": tracer_qc, "qi": tracer_qi},
    )
    clouds = CloudData.zeros((ncols,), nlev).copy(
        qc=post_cloud_qc,
        qi=post_cloud_qi,
        cloud_fraction=diagnosed_cf,
    )

    cloud_water, cloud_ice, cloud_fraction = radiation_cloud_fields(
        state, {"clouds": clouds},
    )

    assert jnp.allclose(cloud_water, tracer_qc)
    assert jnp.allclose(cloud_ice, tracer_qi)
    assert jnp.allclose(cloud_fraction, diagnosed_cf)
    assert not jnp.allclose(cloud_water, post_cloud_qc)
    assert not jnp.allclose(cloud_ice, post_cloud_qi)
