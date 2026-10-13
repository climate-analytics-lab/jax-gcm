"""Tests for shared cloud diagnostics."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.clouds.cloud_data import (
    CLOUD_OUTPUT_ATTRS,
    CloudData,
    radiation_cloud_fields,
)
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


def test_radiation_condensate_derivative_is_one_sided_at_zero():
    """d(in-cloud condensate)/dq is 1/cf at zero condensate in a cloudy cell.

    Levels (cover 0.5): pure ice, pure liquid, both, neither. The clip's
    derivative at zero is the one-sided 1 (a symmetric ``max`` gives 1/2),
    so d(in-cloud liquid)/dqc is 1/cf in the pure-ice cell and d(in-cloud
    ice)/dqi in the pure-liquid cell. The covered cell with no condensate
    keeps the mask's reference derivative, zero.
    """
    import jax
    from jcm.physics.radiation.mcica import in_cloud_condensate
    qc0 = jnp.array([[0.0], [1e-4], [1e-4], [0.0]])
    qi0 = jnp.array([[1e-5], [0.0], [1e-5], [0.0]])
    cf = jnp.full((4, 1), 0.5)
    clouds = CloudData.zeros((1,), 4).copy(cloud_fraction=cf)

    def in_cloud(qc, qi, which):
        cw, ci, cov = radiation_cloud_fields(_state_with(qc, qi), {"clouds": clouds})
        return jnp.sum(in_cloud_condensate(cw if which == "liq" else ci, cov, eps=1e-3))

    d_qc = np.asarray(jax.grad(lambda q: in_cloud(q, qi0, "liq"))(qc0))[:, 0]
    d_qi = np.asarray(jax.grad(lambda q: in_cloud(qc0, q, "ice"))(qi0))[:, 0]
    np.testing.assert_allclose(d_qc, [2.0, 2.0, 2.0, 0.0])
    np.testing.assert_allclose(d_qi, [2.0, 2.0, 2.0, 0.0])
    # Negative condensate is clipped: no derivative.
    g = jax.grad(lambda q: in_cloud(q, qi0, "liq"))(jnp.full((4, 1), -1e-8))
    assert np.all(np.asarray(g) == 0.0)


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


def test_zeros_seeds_the_detrainment_fields():
    """The seed carries the convective detrainment as (nlev, ncols) zeros."""
    nlev, ncols = 5, 3
    clouds = CloudData.zeros((ncols,), nlev)
    for name in ("conv_detrainment_qc", "conv_detrainment_qi"):
        field = getattr(clouds, name)
        assert field.shape == (nlev, ncols)
        assert jnp.all(field == 0.0)


def test_copy_round_trips_every_field():
    """``copy()`` keeps every field and replaces only the ones it is given.

    Enumerated from the struct itself so a field added to the class but
    forgotten in ``copy()`` fails here rather than raising (or, worse,
    silently reverting) inside a term.
    """
    nlev, ncols = 4, 2
    base = CloudData.zeros((ncols,), nlev)
    names = [f.name for f in dataclasses.fields(CloudData)]
    distinct = base.copy(**{
        name: jnp.full(getattr(base, name).shape, float(i + 1))
        for i, name in enumerate(names)
    })
    same = distinct.copy()
    for name in names:
        assert jnp.array_equal(getattr(same, name), getattr(distinct, name)), name

    new_qi = jnp.full((nlev, ncols), -7.0)
    changed = distinct.copy(conv_detrainment_qi=new_qi)
    assert jnp.array_equal(changed.conv_detrainment_qi, new_qi)
    for name in names:
        if name != "conv_detrainment_qi":
            assert jnp.array_equal(
                getattr(changed, name), getattr(distinct, name)), name


class TestTotalCloudCover:
    """``clouds.total_cloud_cover`` is ECHAM's ``aclcov`` of the fraction it sits beside."""

    # Top-first profile: a contiguous 0.6/0.5 deck, a clear layer, a 0.4 deck.
    DECKS = jnp.asarray([[0.6], [0.5], [0.0], [0.4]])

    def test_the_seed_is_cloudless(self):
        clouds = CloudData.zeros((3,), 5)
        assert clouds.total_cloud_cover.shape == (3,)
        assert jnp.all(clouds.total_cloud_cover == 0.0)

    def test_replacing_the_fraction_recomputes_the_cover(self):
        clouds = CloudData.zeros((1,), 4).copy(cloud_fraction=self.DECKS)
        np.testing.assert_allclose(np.asarray(clouds.total_cloud_cover), 0.76,
                                   atol=1e-6)
        # A later writer (the microphysics' write-back) clears the lower deck:
        # the cover follows, so the saved cover is that of the FINAL fraction
        # and no writer of ``cloud_fraction`` can leave it stale.
        cleared = clouds.copy(cloud_fraction=self.DECKS.at[3].set(0.0))
        np.testing.assert_allclose(np.asarray(cleared.total_cloud_cover), 0.6,
                                   atol=1e-6)

    def test_other_fields_leave_the_cover_alone(self):
        clouds = CloudData.zeros((1,), 4).copy(cloud_fraction=self.DECKS)
        moved = clouds.copy(qc=jnp.ones((4, 1)))
        assert jnp.array_equal(moved.total_cloud_cover, clouds.total_cloud_cover)

    def test_an_explicit_cover_wins_over_the_recomputed_one(self):
        clouds = CloudData.zeros((1,), 4).copy(
            cloud_fraction=self.DECKS, total_cloud_cover=jnp.asarray([0.123]))
        np.testing.assert_allclose(np.asarray(clouds.total_cloud_cover), 0.123)

    def test_the_cover_is_the_offline_overlap_of_the_same_fraction(self):
        import xarray as xr

        from jcm.analysis import total_cloud_cover
        cf = jnp.asarray(np.random.default_rng(1).uniform(0, 1, (12, 6)))
        online = CloudData.zeros((6,), 12).copy(cloud_fraction=cf)
        offline = total_cloud_cover(
            xr.DataArray(np.asarray(cf, dtype=np.float64), dims=("level", "col")))
        np.testing.assert_allclose(np.asarray(online.total_cloud_cover),
                                   np.asarray(offline), atol=1e-6)

    def test_time_mean_of_the_cover_exceeds_the_overlap_of_the_mean_profile(self):
        """Why the gate scores the online field, not the overlap of a saved mean.

        A 0.5 cloud that alternates between the top and the bottom layer is
        0.5 cover on every step, so 0.5 on average (what ECHAM accumulates).
        Its mean profile is 0.25 in each layer, and the overlap of *that* is
        1 - 0.75 * 0.75 = 0.4375: the clear layer between the two means breaks
        the maximum-overlap chain that the instantaneous cloud never had to
        cross. (The inequality is not a theorem: clouds that fill several
        layers together at the same times push it the other way.)
        """
        import xarray as xr

        from jcm.analysis import total_cloud_cover
        up = jnp.asarray([[0.5], [0.0], [0.0]])
        down = jnp.asarray([[0.0], [0.0], [0.5]])
        steps = [CloudData.zeros((1,), 3).copy(cloud_fraction=cf)
                 for cf in (up, down)]
        online_mean = float(np.mean([float(s.total_cloud_cover[0])
                                     for s in steps]))
        mean_profile = (steps[0].cloud_fraction + steps[1].cloud_fraction) / 2.0
        offline = float(total_cloud_cover(
            xr.DataArray(np.asarray(mean_profile[:, 0], dtype=np.float64),
                         dims=("level",))))
        np.testing.assert_allclose(online_mean, 0.5, atol=1e-6)
        np.testing.assert_allclose(offline, 0.4375, atol=1e-6)
        assert online_mean > offline


def test_every_field_has_output_attrs():
    """Every ``clouds.<field>`` written to output has units and a long name."""
    names = {f"clouds.{f.name}" for f in dataclasses.fields(CloudData)}
    assert names == set(CLOUD_OUTPUT_ATTRS)
    for key, attrs in CLOUD_OUTPUT_ATTRS.items():
        assert attrs.get("units"), key
        assert attrs.get("long_name"), key
    for name in ("conv_detrainment_qc", "conv_detrainment_qi"):
        assert CLOUD_OUTPUT_ATTRS[f"clouds.{name}"]["units"] == "kg kg-1 s-1"


def test_checkpoint_without_detrainment_fields_restores_them_as_zeros():
    """A checkpoint written before the fields existed still restores.

    Restores match carry leaves by name and fill a leaf the file lacks from
    a freshly built carry (``CloudData.zeros``), so the detrainment fields
    come back as zeros while every stored field keeps its stored value.
    """
    from jcm.checkpoint import _match_by_name, _named_leaves

    nlev, ncols = 3, 2
    template = _named_leaves({"clouds": CloudData.zeros((ncols,), nlev)})
    new_names = {"clouds.conv_detrainment_qc", "clouds.conv_detrainment_qi"}
    assert new_names <= {name for name, _ in template}
    stored = {name: np.full(leaf.shape, 3.0, leaf.dtype)
              for name, leaf in template if name not in new_names}
    fresh = dict(_named_leaves({"clouds": CloudData.zeros((ncols,), nlev)}))

    leaves, seeded, dropped = _match_by_name(
        stored, template, path="old.msgpack", group="physics carry",
        fill_missing=True, seeds=lambda: fresh,
    )
    assert set(seeded) == new_names and dropped == []
    for (name, _), leaf in zip(template, leaves):
        expected = 0.0 if name in new_names else 3.0
        np.testing.assert_array_equal(np.asarray(leaf), expected, err_msg=name)
