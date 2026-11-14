"""Forward-mode tangents stay finite at singular-but-finite primal points.

Each case is a site where the primal is finite but a jvp coefficient is
singular or overflows float32 (division by a squared tiny denominator, sqrt at
an exact zero); the guard must leave the primal unchanged.
"""
import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.gravity_waves.hines.hines import _rms_total_and_per_azimuth_8
from jcm.physics.radiation.cloud_optics import effective_radius_liquid
from jcm.physics.vertical_diffusion.tte_tke.matrix_solver import forward_sweep


def test_hines_rms_dead_spectrum_has_finite_tangent():
    s = jnp.zeros(8)
    (sig_t, sig_a), (dt, da) = jax.jvp(_rms_total_and_per_azimuth_8, (s,), (jnp.ones_like(s),))
    assert np.all(np.asarray(sig_t) == 0.0) and np.all(np.asarray(sig_a) == 0.0)
    assert np.all(np.isfinite(dt)) and np.all(np.isfinite(da))


def test_hines_rms_live_spectrum_unchanged():
    s = jnp.arange(1.0, 9.0)
    sig_t, _ = _rms_total_and_per_azimuth_8(s)
    np.testing.assert_allclose(float(sig_t), 6.0, rtol=1e-6)


def test_tridiagonal_tiny_pivot_has_finite_tangent():
    ncol, nlev = 2, 4
    a = jnp.full((ncol, nlev), -0.5)
    b = jnp.ones((ncol, nlev)).at[:, 0].set(1e-25)      # clamped first pivot
    c = jnp.full((ncol, nlev), -0.5)
    d = jnp.ones((ncol, nlev))
    out, tan = jax.jvp(lambda bb: forward_sweep(a, bb, c, d), (b,), (jnp.ones_like(b),))
    for x in jax.tree_util.tree_leaves(tan):
        assert np.all(np.isfinite(np.asarray(x)))
    assert np.all(np.isfinite(np.asarray(out[0])))


def test_liquid_radius_at_zero_cdnc_factor():
    r, dr = jax.jvp(effective_radius_liquid, (jnp.array([0.0, 1.0]),), (jnp.array([0.0, 1.0]),))
    assert np.isinf(float(r[0])) and np.isfinite(float(r[1]))
    assert np.all(np.isfinite(np.asarray(dr)[1:])) and not np.isnan(float(dr[0]))
