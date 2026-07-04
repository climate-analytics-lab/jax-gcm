"""Gradient tests for the smoothed Tiedtke gates (maintainability review B).

Each test pins the specific unlock: a parameter whose gradient was
exactly zero (or undefined) under the hard gates must now be finite and
nonzero on a physically appropriate column. These are the calibration
paths the smoothing PR exists for — if one regresses to zero the
corresponding sigmoid has been re-hardened.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
    ConvectionParameters,
    tiedtke_nordeng_convection,
)


def _moist_tropical_column(nlev=20, t_sfc=300.0):
    """TOA-first moist conditionally-unstable column."""
    p = jnp.linspace(10_000.0, 101_325.0, nlev)
    t = t_sfc - 6.5e-3 * 16_000.0 * (1.0 - p / p[-1]) ** 0.8
    t = jnp.clip(t, 200.0, t_sfc)
    from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
        saturation_mixing_ratio,
    )
    q = 0.85 * jax.vmap(saturation_mixing_ratio)(p, t)
    rho = p / (287.0 * t)
    dz = jnp.diff(p, prepend=p[:1] * 0.5) / (rho * 9.81)
    return t, q, p, jnp.abs(dz), rho


def _precip(params, t_sfc=300.0, supply=2e-5):
    t, q, p, dz, rho = _moist_tropical_column(t_sfc=t_sfc)
    tend, state = tiedtke_nordeng_convection(
        t, q, p, dz, rho,
        jnp.zeros_like(t), jnp.zeros_like(t),
        jnp.zeros_like(t), jnp.zeros_like(t),
        dt=900.0, config=params,
        moisture_supply=jnp.asarray(supply),
        land_fraction=jnp.asarray(0.0),
    )
    return tend.precip_conv


def _grad_wrt(field, **precip_kwargs):
    base = ConvectionParameters.default()

    def loss(x):
        params = base.__class__(**{
            **{k: getattr(base, k) for k in base.__dict__}, field: x,
        })
        return _precip(params, **precip_kwargs)

    return loss, jnp.asarray(getattr(base, field))


class TestSmoothTriggerGradients:
    def test_entrpen_gradient_nonzero(self):
        """Deep entrainment must be learnable on a convecting column."""
        loss, x0 = _grad_wrt('entrpen')
        g = jax.grad(loss)(x0)
        assert np.isfinite(float(g)) and float(g) != 0.0, g

    def test_tau_gradient_nonzero(self):
        """The CAPE-closure timescale was dead behind lax.switch."""
        loss, x0 = _grad_wrt('tau', supply=0.0)  # CAPE closure path
        g = jax.grad(loss)(x0)
        assert np.isfinite(float(g)) and float(g) != 0.0, g

    def test_zdnoprc_gradient_nonzero(self):
        """Precip-onset depth appeared only in an inequality before."""
        loss, x0 = _grad_wrt('cu_dnoprc_ocean')
        g = jax.grad(loss)(x0)
        assert np.isfinite(float(g)) and float(g) != 0.0, g

    def test_trigger_threshold_gradient_nonzero(self):
        """The activation threshold is calibratable NEAR the threshold.

        A sigmoid trigger saturates (correctly) when CAPE is hundreds of
        widths past the threshold — the synthetic sounding here carries
        CAPE ~ 1.5e4 J/kg, so the gradient at the default 100 J/kg is a
        legitimate exact zero. The calibration property is sensitivity
        near the crossing: evaluate d(precip)/d(trigger_cape) at a
        threshold placed just below the column's CAPE, where the fuzzy
        trigger is on its ramp.
        """
        from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
            calculate_cape_cin, find_cloud_base,
        )
        t, q, p, dz, _ = _moist_tropical_column()
        cfg = ConvectionParameters.default()
        cb, _has = find_cloud_base(t, q, p, cfg)
        cape, _ = calculate_cape_cin(t, q, p, dz, cb, cfg)
        # supply=0: with a moisture supply the OR-branch floor weight
        # saturates the trigger for any buoyant column (by design — the
        # #529 continuous-convection path), so the main threshold only
        # binds on the CAPE-only path.
        loss, _ = _grad_wrt('trigger_cape', supply=0.0)
        x_near = jnp.asarray(float(cape) - 30.0)
        g = jax.grad(loss)(x_near)
        assert np.isfinite(float(g)) and float(g) != 0.0, g
        # And below-threshold sensitivity has the right sign: raising the
        # threshold toward CAPE weakens convection.
        assert float(g) < 0.0, g

    def test_all_default_param_gradients_finite(self):
        """No NaN through any leaf of the params struct (B.0 class)."""
        base = ConvectionParameters.default()

        # lmfdudv is a boolean switch leaf — differentiate w.r.t. the
        # float leaves only.
        float_fields = [
            k for k, v in base.__dict__.items()
            if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)
        ]

        def loss(float_vals):
            params = base.__class__(**{**base.__dict__, **float_vals})
            return _precip(params)

        grads = jax.grad(loss)(
            {k: jnp.asarray(getattr(base, k)) for k in float_fields}
        )
        for k, g in grads.items():
            assert bool(jnp.all(jnp.isfinite(g))), (k, g)

    def test_stable_column_still_exactly_off(self):
        """The rescaled sigmoid keeps zero-CAPE columns at exactly zero
        flux — smoothing must not introduce phantom convection.
        """
        base = ConvectionParameters.default()
        # Isothermal (statically stable) column.
        nlev = 20
        p = jnp.linspace(10_000.0, 101_325.0, nlev)
        t = jnp.full(nlev, 280.0)
        q = jnp.full(nlev, 1e-3)
        rho = p / (287.0 * t)
        dz = jnp.abs(jnp.diff(p, prepend=p[:1] * 0.5)) / (rho * 9.81)
        tend, state = tiedtke_nordeng_convection(
            t, q, p, dz, rho,
            jnp.zeros_like(t), jnp.zeros_like(t),
            jnp.zeros_like(t), jnp.zeros_like(t),
            dt=900.0, config=base,
            moisture_supply=jnp.asarray(2e-5),
            land_fraction=jnp.asarray(0.0),
        )
        assert float(tend.precip_conv) == 0.0
        assert float(jnp.max(jnp.abs(tend.dtedt))) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
