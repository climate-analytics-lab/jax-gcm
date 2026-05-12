"""Tests for shared cloud diagnostics."""

import jax.numpy as jnp

from jcm.physics.clouds.cloud_data import CloudData, radiation_cloud_fields
from jcm.physics_interface import PhysicsState


def test_radiation_cloud_fields_use_current_cloud_diagnostics():
    """Radiation should see post-cloud-scheme condensate, not stale tracers."""
    nlev, ncols = 3, 2
    shape = (nlev, ncols)
    stale_qc = jnp.full(shape, 1.0e-9)
    stale_qi = jnp.full(shape, 2.0e-9)
    current_qc = jnp.arange(nlev * ncols, dtype=jnp.float32).reshape(shape) * 1e-5
    current_qi = current_qc + 1e-4
    current_cf = jnp.clip(current_qc * 1e4, 0.0, 1.0)

    state = PhysicsState.zeros(
        shape,
        temperature=jnp.ones(shape) * 280.0,
        specific_humidity=jnp.ones(shape) * 1e-3,
        tracers={"qc": stale_qc, "qi": stale_qi},
    )
    clouds = CloudData.zeros((ncols,), nlev).copy(
        qc=current_qc,
        qi=current_qi,
        cloud_fraction=current_cf,
    )

    cloud_water, cloud_ice, cloud_fraction = radiation_cloud_fields(
        state, {"clouds": clouds},
    )

    assert jnp.allclose(cloud_water, current_qc)
    assert jnp.allclose(cloud_ice, current_qi)
    assert jnp.allclose(cloud_fraction, current_cf)
    assert not jnp.allclose(cloud_water, stale_qc)
    assert not jnp.allclose(cloud_ice, stale_qi)
