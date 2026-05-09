"""Tests for the Räisänen McICA sub-column generator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.radiation.mcica import (
    column_key,
    generate_subcolumns,
    in_cloud_path,
)


_NLEV = 20
_DEFAULT_DZ = 500.0  # m, uniform 500 m layers (~10 km column)


def _uniform_cloud(nlev=_NLEV, fraction=0.4):
    """Cloud fraction = constant ``fraction`` everywhere."""
    return jnp.full((nlev,), fraction, dtype=jnp.float32)


def _layer_thickness(nlev=_NLEV, dz=_DEFAULT_DZ):
    return jnp.full((nlev,), dz, dtype=jnp.float32)


def test_clear_sky_limit():
    """``cloud_fraction = 0`` → all sub-columns are clear at every level."""
    cf = jnp.zeros((_NLEV,), dtype=jnp.float32)
    masks = generate_subcolumns(
        cf, _layer_thickness(),
        n_subcols=64, overlap="exponential",
        key=jax.random.PRNGKey(0),
    )
    assert masks.shape == (64, _NLEV)
    assert float(jnp.max(masks)) == 0.0


def test_overcast_limit():
    """``cloud_fraction = 1`` → all sub-columns are cloudy everywhere."""
    cf = jnp.ones((_NLEV,), dtype=jnp.float32)
    masks = generate_subcolumns(
        cf, _layer_thickness(),
        n_subcols=64, overlap="exponential",
        key=jax.random.PRNGKey(0),
    )
    assert float(jnp.min(masks)) == 1.0


def test_layer_mean_recovers_cloud_fraction():
    """Averaging the binary mask across many sub-columns recovers ``cf``."""
    target = _uniform_cloud(fraction=0.4)
    # Many sub-columns to drive Monte-Carlo error down to ~0.5/sqrt(N).
    n = 4096
    masks = generate_subcolumns(
        target, _layer_thickness(),
        n_subcols=n, overlap="random",
        key=jax.random.PRNGKey(42),
    )
    layer_mean = jnp.mean(masks, axis=0)
    # 1σ on a Bernoulli(0.4) mean of N draws is ~0.0076; 5σ ≈ 0.038.
    np.testing.assert_allclose(np.array(layer_mean), 0.4, atol=0.04)


def test_random_overlap_is_independent_per_layer():
    """Random overlap → adjacent layers uncorrelated (within MC error)."""
    cf = _uniform_cloud(fraction=0.5)
    masks = generate_subcolumns(
        cf, _layer_thickness(),
        n_subcols=8192, overlap="random",
        key=jax.random.PRNGKey(7),
    )
    # Pearson correlation between layer 0 and layer 1 across sub-columns.
    a = masks[:, 0] - jnp.mean(masks[:, 0])
    b = masks[:, 1] - jnp.mean(masks[:, 1])
    corr = float(jnp.sum(a * b) / jnp.sqrt(jnp.sum(a * a) * jnp.sum(b * b)))
    assert abs(corr) < 0.05  # ≪ 1 expected for random overlap


def test_maximum_random_correlates_within_cloud_bank():
    """Two adjacent cloudy layers should be near-perfectly correlated."""
    cf = _uniform_cloud(fraction=0.5)
    masks = generate_subcolumns(
        cf, _layer_thickness(),
        n_subcols=4096, overlap="maximum_random",
        key=jax.random.PRNGKey(11),
    )
    a = masks[:, 5] - jnp.mean(masks[:, 5])
    b = masks[:, 6] - jnp.mean(masks[:, 6])
    corr = float(jnp.sum(a * b) / jnp.sqrt(jnp.sum(a * a) * jnp.sum(b * b)))
    # Maximum-random within a continuous bank → identical sub-columns.
    assert corr > 0.99


def test_exponential_overlap_decays_with_distance():
    """Inter-layer correlation should decrease with distance for
    exponential overlap.
    """
    cf = _uniform_cloud(fraction=0.5)
    masks = generate_subcolumns(
        cf, _layer_thickness(dz=_DEFAULT_DZ),
        n_subcols=8192, overlap="exponential",
        decorrelation_km=2.0,
        key=jax.random.PRNGKey(13),
    )

    def corr(i, j):
        a = masks[:, i] - jnp.mean(masks[:, i])
        b = masks[:, j] - jnp.mean(masks[:, j])
        return float(jnp.sum(a * b) / jnp.sqrt(jnp.sum(a * a) * jnp.sum(b * b)))

    # Adjacent layers (Δz = 0.5 km, L = 2 km) should be ~exp(-0.25) = 0.78.
    c_near = corr(8, 9)
    # Layers 5 apart (Δz = 2.5 km) should be much weaker.
    c_far = corr(5, 10)
    assert c_near > c_far + 0.1
    assert c_near > 0.5
    assert c_far < c_near


def test_reproducibility():
    """Same key → same masks (bit-exact)."""
    cf = _uniform_cloud(fraction=0.4)
    key = jax.random.PRNGKey(99)
    a = generate_subcolumns(
        cf, _layer_thickness(),
        n_subcols=16, overlap="exponential", key=key,
    )
    b = generate_subcolumns(
        cf, _layer_thickness(),
        n_subcols=16, overlap="exponential", key=key,
    )
    np.testing.assert_array_equal(np.array(a), np.array(b))


def test_column_key_is_deterministic():
    """``column_key`` composes step+column into a reproducible PRNG key."""
    base = jax.random.PRNGKey(0)
    k1 = column_key(base, model_step=10, column_index=42)
    k2 = column_key(base, model_step=10, column_index=42)
    np.testing.assert_array_equal(np.array(k1), np.array(k2))
    # Different (step, col) → different key.
    k3 = column_key(base, model_step=10, column_index=43)
    assert not np.array_equal(np.array(k1), np.array(k3))


def test_in_cloud_path_scales_correctly():
    """``LWP_grid = f * LWP_in_cloud`` ⇒ ``LWP_in_cloud = LWP_grid / f``."""
    grid = jnp.array([0.1, 0.2, 0.3])
    f = jnp.array([0.5, 0.5, 0.5])
    expected = jnp.array([0.2, 0.4, 0.6])
    np.testing.assert_allclose(
        np.array(in_cloud_path(grid, f)), np.array(expected),
        rtol=1e-6,
    )


def test_in_cloud_path_floors_zero_cloud():
    """Clear cells (f=0) get a finite divisor to avoid NaN."""
    grid = jnp.array([0.0])
    f = jnp.array([0.0])
    out = in_cloud_path(grid, f, eps=1e-3)
    assert jnp.isfinite(out).all()
