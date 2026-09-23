"""Tests for the Räisänen McICA sub-column generator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.radiation.mcica import (
    column_key,
    column_total_cover,
    effective_cloud_fraction,
    expected_total_cover,
    generate_subcolumns,
    in_cloud_path,
)
from jcm.physics.radiation.mcica import _alpha_from_overlap
from jcm.testing import check_gradients


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


def test_in_cloud_path_zeroes_clear_cells():
    """Clear cells (f <= 2*eps) get a ZERO in-cloud path.

    Mirrors ECHAM ``mo_psrad_interface.f90:232-237`` which zeros the
    in-cloud water where ``cld_frc_vr <= 2*EPSILON``. Even with grid-mean
    condensate present (e.g. decorrelated 1M residue), the in-cloud path
    must be exactly 0 in a clear cell so a vanishing cloud fraction can
    never inflate the cloud optical depth.
    """
    eps = 1e-3
    grid = jnp.array([1e-3, 1e-3, 1e-3])      # nonzero grid-mean condensate
    f = jnp.array([0.0, eps, 5.0 * eps])      # clear, clear, thin-but-real
    out = in_cloud_path(grid, f, eps=eps)
    assert jnp.isfinite(out).all()
    # f=0 and f=eps are <= 2*eps -> zeroed.
    np.testing.assert_array_equal(np.array(out[:2]), np.array([0.0, 0.0]))
    # f=5*eps is a real (if thin) cloud -> normal grid/f conversion.
    np.testing.assert_allclose(float(out[2]), 1e-3 / (5.0 * eps), rtol=1e-6)


def test_in_cloud_path_no_inflation_for_decorrelated_condensate():
    """A tiny cloud fraction with large grid-mean condensate must not
    inflate the in-cloud path (the RRTMGP-NaN failure mode).
    """
    eps = 1e-3
    grid = jnp.array([5e-3])     # large grid-mean condensate (5 g/kg)
    f = jnp.array([1e-6])        # essentially clear
    out = in_cloud_path(grid, f, eps=eps)
    assert float(out[0]) == 0.0


def test_effective_cloud_fraction_matches_in_cloud_path_zeroing():
    """The sampler/cover fraction is zeroed on the SAME cells in_cloud_path is.

    A cell whose in-cloud condensate is zeroed (cf <= 2*eps) is optically
    empty, so its effective cloud fraction must be 0 too — otherwise the
    McICA sampler and the cover diagnostic disagree with the optics.
    """
    eps = 1e-3
    grid = jnp.ones(4)
    f = jnp.array([0.0, eps, 2.0 * eps, 5.0 * eps])
    eff = effective_cloud_fraction(f, eps=eps)
    zeroed_condensate = in_cloud_path(grid, f, eps=eps) == 0.0
    # Wherever the condensate is zeroed the effective fraction is 0, and only
    # there — the thin-but-real cloud (5*eps) keeps its fraction.
    np.testing.assert_array_equal(np.array(eff == 0.0), np.array(zeroed_condensate))
    np.testing.assert_allclose(float(eff[3]), 5.0 * eps, rtol=1e-6)


def test_effective_fraction_removes_spurious_cover_from_optically_empty_layer():
    """A sub-threshold layer must not report cover or bridge overlap.

    Quantifies the marginal-layer behaviour change at the 1e-3 default: a
    column whose only nonzero fraction is a sub-threshold layer (cf = 1.5e-3)
    reports nonzero max-random cover from the RAW fraction but ZERO from the
    effective fraction, matching the zeroed in-cloud condensate there.
    """
    eps = 1e-3
    f = jnp.array([0.0, 1.5e-3, 0.0])   # cf in (eps, 2*eps]: condensate zeroed
    # Raw fraction still reports cover (max-random overlap code 1).
    np.testing.assert_allclose(float(column_total_cover(f, 1)), 1.5e-3, rtol=1e-5)
    # Effective fraction reports none — consistent with the empty optics.
    eff = effective_cloud_fraction(f, eps=eps)
    assert float(column_total_cover(eff, 1)) == 0.0
    # A genuine (supra-threshold) cloud in the same column is untouched.
    f2 = f.at[0].set(0.6)
    np.testing.assert_allclose(
        float(column_total_cover(effective_cloud_fraction(f2, eps=eps), 1)),
        0.6, rtol=1e-5)


def _unrolled_expected_total_cover(cloud_fraction, layer_thickness, overlap):
    """Evaluate the former recurrence as an independent test oracle."""
    cf = jnp.clip(cloud_fraction, 0.0, 1.0)
    alpha = _alpha_from_overlap(cf, layer_thickness, overlap, 2.0)
    clear = [jnp.ones(cf.shape[1:]), 1.0 - cf[0]]
    for k in range(1, cf.shape[0]):
        contrib = jnp.zeros(cf.shape[1:])
        prod_a = jnp.ones(cf.shape[1:])
        seg_max = cf[k]
        for s in range(k, -1, -1):
            seg_max = jnp.maximum(seg_max, cf[s])
            start = (1.0 - alpha[s - 1]) if s > 0 else 1.0
            contrib = contrib + start * prod_a * clear[s] * (1.0 - seg_max)
            if s > 0:
                prod_a = prod_a * alpha[s - 1]
        clear.append(contrib)
    return 1.0 - clear[cf.shape[0]]


@pytest.mark.parametrize("overlap", ["random", "maximum_random", "exponential"])
@pytest.mark.parametrize("fractions", [
    [0.0],
    [1.0],
    [0.0, 0.5, 0.0, 0.8, 0.0],
    [0.4, 0.4, 0.7, 0.4, 0.7],
    [0.1, 0.9, 0.3, 0.6, 0.2],
])
def test_expected_total_cover_matches_unrolled_value_and_derivatives(overlap, fractions):
    """The scan preserves values, tangent, and cotangent at ties and gaps."""
    cf = jnp.asarray(fractions, dtype=jnp.float32)
    dz = jnp.linspace(200.0, 800.0, len(fractions))
    tangent_cf = jnp.linspace(0.1, 0.3, len(fractions))
    tangent_dz = jnp.linspace(0.2, 0.4, len(fractions))
    reference = lambda c, d: _unrolled_expected_total_cover(c, d, overlap)
    scanned = lambda c, d: expected_total_cover(c, d, overlap)
    np.testing.assert_allclose(scanned(cf, dz), reference(cf, dz), rtol=2e-6, atol=2e-6)
    old_jvp = jax.jvp(reference, (cf, dz), (tangent_cf, tangent_dz))
    new_jvp = jax.jvp(scanned, (cf, dz), (tangent_cf, tangent_dz))
    for actual, expected in zip(new_jvp, old_jvp):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    old_vjp = jax.grad(lambda c, d: jnp.sum(reference(c, d)), argnums=(0, 1))(cf, dz)
    new_vjp = jax.grad(lambda c, d: jnp.sum(scanned(c, d)), argnums=(0, 1))(cf, dz)
    for actual, expected in zip(new_vjp, old_vjp):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("overlap", ["random", "maximum_random", "exponential"])
def test_expected_total_cover_multicolumn_47_levels(overlap):
    """A realistic vertical dimension and column batch preserve values."""
    cf = jnp.asarray(np.random.default_rng(9).uniform(size=(47, 3)), dtype=jnp.float32)
    dz = jnp.full((47, 3), 350.0)
    np.testing.assert_allclose(
        expected_total_cover(cf, dz, overlap),
        _unrolled_expected_total_cover(cf, dz, overlap),
        rtol=2e-6, atol=2e-6,
    )


def test_expected_total_cover_float64_value_and_derivatives():
    """Keep the former promotion behavior when 64-bit arrays are enabled."""
    with jax.enable_x64():
        cf = jnp.array([0.2, 0.6, 0.6, 0.3], dtype=jnp.float64)
        dz = jnp.array([200.0, 400.0, 600.0, 800.0], dtype=jnp.float64)
        reference = lambda c, d: _unrolled_expected_total_cover(c, d, "exponential")
        scanned = lambda c, d: expected_total_cover(c, d, "exponential")
        np.testing.assert_allclose(scanned(cf, dz), reference(cf, dz), rtol=1e-12)
        old_grad = jax.grad(reference, argnums=(0, 1))(cf, dz)
        new_grad = jax.grad(scanned, argnums=(0, 1))(cf, dz)
        for actual, expected in zip(new_grad, old_grad):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("fractions", [[0.3], [0.2, 0.6, 0.6, 0.3]])
def test_expected_total_cover_float32_under_x64(fractions):
    """Preserve the old result dtype and derivatives with x64 enabled."""
    with jax.enable_x64():
        cf = jnp.asarray(fractions, dtype=jnp.float32)
        dz = jnp.full(cf.shape, 400.0, dtype=jnp.float32)
        reference = lambda c, d: _unrolled_expected_total_cover(c, d, "exponential")
        scanned = lambda c, d: expected_total_cover(c, d, "exponential")
        old_value = reference(cf, dz)
        new_value = scanned(cf, dz)
        assert new_value.dtype == old_value.dtype
        np.testing.assert_allclose(new_value, old_value, rtol=1e-6, atol=1e-6)
        old_jvp = jax.jvp(reference, (cf, dz), (cf, dz))
        new_jvp = jax.jvp(scanned, (cf, dz), (cf, dz))
        for actual, expected in zip(new_jvp, old_jvp):
            assert actual.dtype == expected.dtype
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        old_grad = jax.grad(reference, argnums=(0, 1))(cf, dz)
        new_grad = jax.grad(scanned, argnums=(0, 1))(cf, dz)
        for actual, expected in zip(new_grad, old_grad):
            assert actual.dtype == expected.dtype
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


# A deck profile with real clear gaps, and a thin one whose random-overlap
# cover stays well away from 1.
_DECK_CLOUD = jnp.array([0.02, 0.05, 0.35, 0.60, 0.55, 0.08,
                         0.05, 0.20, 0.75, 0.80, 0.10, 0.03])
_THIN_CLOUD = jnp.linspace(0.04, 0.26, 12)
_GRADIENT_DZ = _layer_thickness(nlev=12)


class TestMcicaGradients:
    """AD against a central difference through the McICA helpers (#820).

    Every *continuous* entry point here is green. The sampler itself is the
    exception, and deliberately so — see
    ``test_subcolumn_masks_carry_no_gradient``.

    The random-overlap fixtures use ``_THIN_CLOUD`` rather than the deck
    profile: random overlap of a 12-layer deck column gives a total cover of
    0.999, and ``1 - prod(1 - f)`` that close to 1 leaves a float32 secant
    only a handful of ulps of signal, so a comparison there would be
    measuring the reference's precision rather than the derivative. The
    correlated rules do not saturate and use the deck profile.
    """

    @pytest.mark.parametrize("seed", [0, 4])
    def test_in_cloud_path(self, seed):
        """Cloud fractions clear of the ``2*eps`` zeroing threshold."""
        check_gradients(
            in_cloud_path,
            (jnp.linspace(1.0e-5, 3.0e-4, 12), _DECK_CLOUD), rtol=1e-3,
            seed=seed)

    @pytest.mark.parametrize("seed", [0, 4])
    def test_effective_cloud_fraction(self, seed):
        """The same threshold, applied to the fraction itself."""
        check_gradients(effective_cloud_fraction, (_DECK_CLOUD,), rtol=1e-3,
                        seed=seed)

    @pytest.mark.parametrize("seed", [0, 4])
    @pytest.mark.parametrize("overlap_code", [0, 1, 2],
                             ids=["random", "maximum_random", "exponential"])
    def test_column_total_cover(self, overlap_code, seed):
        """The grey beam-split's closed-form cover, for all three codes."""
        check_gradients(lambda f: column_total_cover(f, overlap_code),
                        (_THIN_CLOUD,), rtol=1e-3, seed=seed)

    @pytest.mark.parametrize("seed", [0, 4])
    @pytest.mark.parametrize("overlap,cloud_fraction", [
        ("random", _THIN_CLOUD),
        ("maximum_random", _DECK_CLOUD),
        ("exponential", _DECK_CLOUD),
    ])
    def test_expected_total_cover(self, overlap, cloud_fraction, seed):
        """The O(nlev^2) rank-segment recursion, for each overlap rule.

        This is the differentiable counterpart of the sampler — the cover the
        NN emulator publishes — so it is the one place in this module where a
        cloud-fraction gradient survives, and worth a fence for that reason
        alone.
        """
        check_gradients(
            lambda f, dz: expected_total_cover(f, dz, overlap=overlap),
            (cloud_fraction, _GRADIENT_DZ), rtol=1e-3, seed=seed)

    def test_subcolumn_masks_carry_no_gradient(self):
        """The sampler is a hard Bernoulli draw, so its gradient is zero.

        ``per_subcol`` returns ``(r < cloud_fraction).astype(float32)``: a
        comparison feeding an integer-to-float cast. Both kill the
        derivative, so ``d(mask)/d(cloud_fraction)`` is identically zero —
        finite, but carrying no information. Radiative gradients with respect
        to cloud fraction therefore do **not** flow through the McICA path;
        they flow through the in-cloud condensate (``in_cloud_path``, which
        is differentiable) and, for the schemes that use it, through
        ``expected_total_cover``.

        That is the Raisanen (2004) sampler as ECHAM runs it, not a defect to
        guard away — making it differentiable means a relaxed or
        straight-through estimator, which changes what is sampled and needs
        its own validation. This test pins the present behaviour so that such
        a change cannot land unnoticed, and asserts finiteness because a
        ``nan`` would be a different matter entirely.
        """
        key = jax.random.PRNGKey(20260918)
        masks = generate_subcolumns(
            _DECK_CLOUD, _GRADIENT_DZ, n_subcols=32,
            overlap="exponential", key=key)
        # The draw is a real one, not an all-clear column that would make the
        # zero gradient uninteresting.
        assert 0.0 < float(jnp.mean(masks)) < 1.0

        gradients = jax.grad(
            lambda f, dz: jnp.sum(generate_subcolumns(
                f, dz, n_subcols=32, overlap="exponential", key=key) ** 2),
            argnums=(0, 1),
        )(_DECK_CLOUD, _GRADIENT_DZ)
        for name, gradient in zip(("cloud_fraction", "layer_thickness"),
                                  gradients):
            assert jnp.all(jnp.isfinite(gradient)), (
                f"d/d{name} is not finite: {gradient}")
            assert jnp.all(gradient == 0.0), (
                f"d/d{name} is no longer identically zero: {gradient}. The "
                f"sampler has become differentiable; that is a change to "
                f"what McICA samples, not a test to relax.")
