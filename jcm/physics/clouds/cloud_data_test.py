"""Tests for shared cloud diagnostics."""

import dataclasses

import jax.numpy as jnp
import numpy as np

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
