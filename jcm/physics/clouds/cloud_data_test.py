"""Tests for shared cloud diagnostics."""

import jax.numpy as jnp
import pytest

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


def _state_with(qc, qi):
    shape = qc.shape
    return PhysicsState.zeros(
        shape, temperature=jnp.full(shape, 260.0),
        tracers={"qc": qc, "qi": qi})


def test_radiation_cover_is_masked_where_there_is_no_condensate():
    """``mo_radiation.f90`` l.428-434: clip at 0, keep cover only with condensate.

    Levels: liquid only, ice only, both, neither, negative liquid (clipped
    to 0 and so treated as none), negative liquid with positive ice.
    """
    qc = jnp.array([[1e-5], [0.0], [1e-5], [0.0], [-1e-8], [-1e-8]])
    qi = jnp.array([[0.0], [1e-6], [1e-6], [0.0], [0.0], [1e-7]])
    cf = jnp.full((6, 1), 0.4)
    clouds = CloudData.zeros((1,), 6).copy(cloud_fraction=cf)
    cw, ci, got = radiation_cloud_fields(_state_with(qc, qi),
                                         {"clouds": clouds})
    import numpy as np
    np.testing.assert_array_equal(
        np.asarray(got[:, 0]), np.asarray(cf[:, 0]) * np.array([1, 1, 1, 0, 0, 1]))
    assert float(jnp.min(cw)) == 0.0 and float(cw[5, 0]) == 0.0
    assert float(ci[5, 0]) == pytest.approx(1e-7)
    # the cover field itself is untouched
    np.testing.assert_array_equal(np.asarray(clouds.cloud_fraction),
                                  np.asarray(cf))


def _echam_total_cover(cf):
    """``mo_radiation.f90`` l.436-442, ECHAM's maximum-random ``cld_cvr``."""
    import numpy as np
    clear = 1.0 - cf[0]
    for k in range(1, len(cf)):
        clear *= (1.0 - max(cf[k], cf[k - 1])) / (
            1.0 - min(cf[k - 1], 1.0 - np.finfo(float).eps))
    return 1.0 - clear


def test_condensate_free_layer_separates_two_cloud_banks_in_mcica():
    """A cloudless-by-condensate layer between two clouds is clear to McICA.

    ECHAM's maximum-random sampler (``mo_cld_sampling.f90`` l.66-83) keeps
    a rank only below a layer that has cover, and it samples the masked
    cover, so the empty layer breaks the bank: the two clouds overlap
    randomly, total cover 1 - 0.5·0.5 = 0.75. Unmasked, the empty layer's
    diagnosed 0.5 would bridge them into one maximally overlapped bank, 0.5.
    The maximum-random checks run on the default overlap rule, which is
    ECHAM's. Under the exponential rule (a jcm option) the correlation does
    not depend on the cover, so the mask removes only the empty layer's own,
    optically empty, sampled cloud.
    """
    import jax
    import numpy as np
    from jcm.physics.radiation.mcica import (
        effective_cloud_fraction, expected_total_cover, generate_subcolumns)
    from jcm.physics.radiation.radiation_types import (
        RadiationParameters, cloud_overlap_name)

    default_rule = cloud_overlap_name(
        int(RadiationParameters.default().cloud_overlap))
    assert default_rule == "maximum_random"

    qc = jnp.array([[1e-5], [0.0], [1e-5]])
    qi = jnp.zeros((3, 1))
    clouds = CloudData.zeros((1,), 3).copy(cloud_fraction=jnp.full((3, 1), 0.5))
    _, _, cf = radiation_cloud_fields(_state_with(qc, qi), {"clouds": clouds})
    cf = effective_cloud_fraction(cf[:, 0])
    dz = jnp.full(3, 500.0)

    masked = float(expected_total_cover(cf, dz, default_rule))
    unmasked = float(expected_total_cover(jnp.full(3, 0.5), dz,
                                          default_rule))
    assert masked == pytest.approx(0.75) == _echam_total_cover(np.asarray(cf))
    assert unmasked == pytest.approx(0.5)

    masks = generate_subcolumns(cf, dz, n_subcols=20000,
                                overlap=default_rule,
                                key=jax.random.PRNGKey(0))
    sampled = float(jnp.mean(jnp.max(masks, axis=1)))
    assert sampled == pytest.approx(0.75, abs=0.01)
    assert float(jnp.max(masks[:, 1])) == 0.0          # the empty layer

    # exponential: same correlation either way; only the empty layer differs
    exp_masked = generate_subcolumns(cf, dz, n_subcols=64,
                                     overlap="exponential",
                                     key=jax.random.PRNGKey(1))
    exp_unmasked = generate_subcolumns(jnp.full(3, 0.5), dz, n_subcols=64,
                                       overlap="exponential",
                                       key=jax.random.PRNGKey(1))
    np.testing.assert_array_equal(exp_masked[:, [0, 2]],
                                  exp_unmasked[:, [0, 2]])
