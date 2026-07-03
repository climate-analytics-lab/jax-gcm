"""Gradient pins for the smoothed Sundqvist constructs (review B.2.4).

The hard ``clip(b0, 0, 1)`` and the argmax inversion pick made
``crt``/``csatsc``/``cinv`` gradient-dead over most of state space
(exactly zero in the review's probes). These tests pin the unlock on a
marine-stratocumulus column tuned into the enhancement-sensitive window
(BL RH between rhc·csatsc-boosted saturation and the upper plateau).
"""

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.clouds.sundqvist import (
    CloudParameters,
    calculate_cloud_fraction,
    saturation_specific_humidity,
)


def _marine_sc_column(nlev=20, bl_rh=0.68):
    p = jnp.linspace(2e4, 1.013e5, nlev)
    t = jnp.linspace(220.0, 288.0, nlev)
    # Genuine BL-top inversion: level 16 warmer than the level below it,
    # at ~600-1100 m — inside the inversion_z_min/max search range.
    t = t.at[16].set(t[17] + 2.0)
    qs = jax.vmap(saturation_specific_humidity)(p, t)
    rh = jnp.where(jnp.arange(nlev) >= 15, bl_rh, 0.4)
    return t, rh * qs, p


class TestSundqvistSmoothGradients:
    def _grad(self, field, bl_rh=0.68):
        t, q, p = _marine_sc_column(bl_rh=bl_rh)
        base = CloudParameters.default()

        def cf_sum(x):
            cfg = base.__class__(**{**base.__dict__, field: x})
            cf, _ = calculate_cloud_fraction(t, q, p, p[-1], cfg)
            return jnp.sum(cf)

        return jax.grad(cf_sum)(jnp.asarray(getattr(base, field)))

    def test_csatsc_gradient_nonzero(self):
        """Sc-enhancement strength was dead behind the one-hot .at[knvb]."""
        g = self._grad('csatsc')
        assert np.isfinite(float(g)) and abs(float(g)) > 1e-6, g

    def test_cinv_gradient_nonzero(self):
        """Cinv appeared only in the stability inequality before."""
        g = self._grad('cinv')
        assert np.isfinite(float(g)) and abs(float(g)) > 1e-12, g

    def test_crt_gradient_nonzero(self):
        """Critical-RH gradient survives inside the ramp window."""
        g = self._grad('crt')
        assert np.isfinite(float(g)) and abs(float(g)) > 1e-6, g

    def test_cf_bounded_and_finite(self):
        """Soft-clip keeps cf in [0, 1] and finite across an RH sweep,
        including super-saturated columns (the old 1-sqrt(1-b0) had an
        infinite slope at b0=1).
        """
        base = CloudParameters.default()
        for rh in (0.2, 0.6, 0.9, 1.0, 1.2):
            t, _, p = _marine_sc_column()
            qs = jax.vmap(saturation_specific_humidity)(p, t)
            cf, _ = calculate_cloud_fraction(t, rh * qs, p, p[-1], base)
            assert bool(jnp.all(jnp.isfinite(cf)))
            assert float(jnp.min(cf)) >= 0.0
            assert float(jnp.max(cf)) <= 1.0 + 1e-6

            g = jax.grad(
                lambda q: jnp.sum(
                    calculate_cloud_fraction(t, q, p, p[-1], base)[0]
                )
            )(rh * qs)
            assert bool(jnp.all(jnp.isfinite(g))), rh
