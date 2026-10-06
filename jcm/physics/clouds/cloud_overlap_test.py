"""Tests for the maximum-random overlap kernel (ECHAM's ``aclcov``).

The kernel is the single copy of ``mo_cloud.f90`` section 10.2 shared by the
in-model ``clouds.total_cloud_cover`` and the offline
:func:`jcm.analysis.total_cloud_cover`. The tests pin three things: that it is
the Fortran recurrence (including at overcast layers, where the Fortran form
is the one that breaks in float32); that it is orientation-free and
broadcasting-native, because the physics frame is top-first and the saved
output surface-first; and that it is differentiable without NaN, because it
sits in the scan carry of every step.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from jcm.analysis import total_cloud_cover
from jcm.physics.clouds.cloud_overlap import (
    ZEPSEC,
    column_max_random_cover,
    max_random_cover,
)
from jcm.testing import check_gradients


def _fortran_aclcov(profile):
    """``mo_cloud.f90`` section 10.2 as written, in float64, top level first.

    Deliberately the Fortran's own ``1 - MIN(c, zxsec)`` form and a plain loop,
    independent of the kernel's ``MAX(1 - c, zepsec)`` rewrite.
    """
    c = np.clip(np.asarray(profile, dtype=np.float64), 0.0, 1.0)
    zxsec = 1.0 - ZEPSEC
    zclcov = 1.0 - c[0]
    for k in range(1, len(c)):
        zclcov *= (1.0 - max(c[k], c[k - 1])) / (1.0 - min(c[k - 1], zxsec))
    return 1.0 - zclcov


def _profiles(n=64, nlev=47, seed=0):
    """Random columns with clear gaps and some overcast layers, ``(nlev, n)``."""
    rng = np.random.default_rng(seed)
    cf = rng.uniform(0.0, 1.0, (nlev, n))
    cf[rng.uniform(size=cf.shape) < 0.55] = 0.0
    cf[rng.uniform(size=cf.shape) < 0.03] = 1.0
    return cf


# A top-first profile of the kind the physics sees: a contiguous 0.6/0.5 deck,
# a clear layer, a 0.4 deck. Maximum overlap inside a deck and random overlap
# between them gives 1 - (1 - 0.6)(1 - 0.4) = 0.76.
_DECKS = np.array([0.6, 0.5, 0.0, 0.4])


class TestRecurrence:

    def test_known_values(self):
        cover = lambda p: float(max_random_cover(lambda k: p[k], len(p), np))  # noqa: E731
        np.testing.assert_allclose(cover([0.37]), 0.37)
        np.testing.assert_allclose(cover([0.5] * 5), 0.5)            # maximum
        np.testing.assert_allclose(cover([0.3, 0.7, 0.4]), 0.7)
        np.testing.assert_allclose(cover([0.4, 0.0, 0.6]), 0.76)     # random
        np.testing.assert_allclose(cover(_DECKS), 0.76)
        np.testing.assert_allclose(cover([0.0, 0.0, 0.0]), 0.0)

    def test_is_the_fortran_recurrence_in_float64(self):
        """The ``MAX(1 - c, zepsec)`` form is ECHAM's ``1 - MIN(c, zxsec)``."""
        cf = _profiles()
        got = np.asarray(max_random_cover(
            lambda k: np.clip(cf[k], 0.0, 1.0), cf.shape[0], np))
        want = np.array([_fortran_aclcov(cf[:, j]) for j in range(cf.shape[1])])
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-14)

    def test_overcast_layer_gives_cover_one_at_any_level(self):
        # c = 1 makes the Fortran's 1 / (1 - c) singular; the floor at zepsec
        # leaves the matching numerator, which is exactly zero.
        for p in ([1.0, 0.3, 0.2], [0.3, 1.0, 0.2], [0.3, 0.2, 1.0],
                  [1.0, 1.0, 1.0]):
            np.testing.assert_allclose(
                float(max_random_cover(lambda k: np.float64(p[k]), 3, np)), 1.0)

    def test_nan_in_any_layer_propagates(self):
        p = np.array([0.3, np.nan, 0.2])
        assert np.isnan(float(max_random_cover(lambda k: p[k], 3, np)))
        assert np.isnan(float(column_max_random_cover(jnp.asarray(p))))


class TestColumnWrapper:
    """``column_max_random_cover``: the jnp entry the physics step calls."""

    def test_agrees_with_the_fortran_transcription(self):
        cf = _profiles()
        got = np.asarray(column_max_random_cover(jnp.asarray(cf, jnp.float32)))
        want = np.array([_fortran_aclcov(cf[:, j]) for j in range(cf.shape[1])])
        np.testing.assert_allclose(got, want, rtol=0, atol=2e-6)

    @pytest.mark.parametrize("reverse", [False, True])
    def test_online_equals_offline_for_a_time_constant_profile(self, reverse):
        """One step of a time-constant profile: the in-model cover is the
        offline scorer's, in either vertical orientation.

        The physics frame is top-first and saved output surface-first; the
        overlap is the same set of adjacent-pair factors either way, so no
        orientation guard exists and none is needed.
        """
        profile = _DECKS[::-1] if reverse else _DECKS
        online = float(column_max_random_cover(jnp.asarray(profile, jnp.float32)))
        offline = float(total_cloud_cover(xr.DataArray(profile, dims=("level",))))
        np.testing.assert_allclose(online, offline, atol=1e-6)
        np.testing.assert_allclose(online, 0.76, atol=1e-6)

    def test_orientation_does_not_matter_on_random_columns(self):
        cf = jnp.asarray(_profiles(seed=3), jnp.float32)
        np.testing.assert_allclose(
            np.asarray(column_max_random_cover(cf)),
            np.asarray(column_max_random_cover(cf[::-1])), atol=2e-6)
        # ...and the offline scorer reads the same number from the same field,
        # up and down.
        da = xr.DataArray(np.asarray(cf, dtype=np.float64),
                          dims=("level", "col"))
        off = np.asarray(total_cloud_cover(da))
        np.testing.assert_allclose(np.asarray(column_max_random_cover(cf)),
                                   off, atol=2e-6)
        np.testing.assert_allclose(
            np.asarray(column_max_random_cover(cf[::-1])), off, atol=2e-6)

    def test_is_the_kernel_the_offline_scorer_runs(self):
        """Bitwise, not merely close: the two callers share one recurrence."""
        cf = _profiles(seed=5)
        got = np.asarray(max_random_cover(
            lambda k: np.clip(cf[k], 0.0, 1.0), cf.shape[0], np))
        off = np.asarray(total_cloud_cover(xr.DataArray(cf, dims=("level", "col"))))
        np.testing.assert_array_equal(got, off)

    def test_broadcasts_over_any_trailing_shape(self):
        """A ``(nlev,)`` column, a ``(nlev, ncols)`` block and a
        ``(nlev, nlat, nlon)`` grid agree column by column.
        """
        cf = jnp.asarray(_profiles(n=12, nlev=20, seed=2), jnp.float32)
        block = np.asarray(column_max_random_cover(cf))
        assert block.shape == (12,)
        grid = np.asarray(column_max_random_cover(cf.reshape(20, 3, 4)))
        assert grid.shape == (3, 4)
        np.testing.assert_array_equal(grid.reshape(12), block)
        for j in range(12):
            single = column_max_random_cover(cf[:, j])
            assert single.shape == ()
            np.testing.assert_allclose(float(single), block[j], atol=1e-7)

    def test_clips_out_of_range_input_and_keeps_the_dtype(self):
        out = column_max_random_cover(jnp.asarray([-0.2, 0.4, 1.3], jnp.float32))
        assert out.dtype == jnp.float32
        np.testing.assert_allclose(float(out), 1.0)
        out = column_max_random_cover(jnp.asarray([-0.5, -1e-9], jnp.float32))
        np.testing.assert_allclose(float(out), 0.0)

    def test_float32_overcast_layer_is_finite_in_value_and_gradient(self):
        """The Fortran's ``zxsec = 1 - 1e-12`` is exactly 1.0 in float32, which
        turns an overcast layer's guarded 0 / 1e-12 into 0 / 0. The kernel
        floors ``1 - c`` at ``zepsec`` instead, so neither the cover nor its
        gradient (which sits in every step's backward pass) can be NaN.
        """
        assert np.float32(1.0 - ZEPSEC) == np.float32(1.0)
        cf = jnp.asarray([[0.4, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0],
                          [0.0, 0.0, 0.0, 0.0]], jnp.float32).T      # (4, 3)
        cover = column_max_random_cover(cf)
        np.testing.assert_allclose(np.asarray(cover), [1.0, 1.0, 0.0])
        grad = jax.grad(lambda x: jnp.sum(column_max_random_cover(x)))(cf)
        assert np.all(np.isfinite(np.asarray(grad)))
        # A zero cotangent through the kernel stays zero (0 * finite), which is
        # the case that matters: a carry diagnostic nothing reads.
        zero = jax.vjp(column_max_random_cover, cf)[1](jnp.zeros(3, jnp.float32))[0]
        np.testing.assert_array_equal(np.asarray(zero), 0.0)

    def test_gradient_matches_a_finite_difference(self):
        # An interior point: no clip edge, no max tie, a contiguous deck and a
        # separated one, so the derivative is smooth and non-trivial.
        cf = jnp.asarray([0.2, 0.55, 0.35, 0.1, 0.05, 0.3, 0.45, 0.15],
                         jnp.float32)
        check_gradients(column_max_random_cover, (cf,), rtol=1e-3)

    def test_gradient_is_the_analytic_one_for_two_layers(self):
        # Two layers, c1 > c0: clear = (1 - c0)(1 - c1)/(1 - c0) = 1 - c1.
        grad = jax.grad(column_max_random_cover)(jnp.asarray([0.3, 0.6]))
        np.testing.assert_allclose(np.asarray(grad), [0.0, 1.0], atol=1e-6)
