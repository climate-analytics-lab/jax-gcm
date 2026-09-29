"""Tests of ECHAM's saturation functions (``echam_saturation.py``)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from jcm.physics.clouds import echam_saturation as es


@pytest.fixture(autouse=True)
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _fit(t, a):
    return np.exp(a[0] / t + a[1] + a[2] * 0.01 * t + a[3] * t * t * 1e-5
                  + a[4] * np.log(t))


def _dfit(t, a):
    return (-a[0] / t ** 2 + a[2] * 0.01 + a[3] * 2e-5 * t + a[4] / t) * _fit(t, a)


def _echam_spline(t, a):
    """ECHAM's table lookup: cubic Hermite on 0.025 K knots.

    ``fetch_ua_spline`` (mo_echam_convect_tables.f90 l.394-437) with knot
    values and knot slopes from the analytic fit (l.262-309), indexed by
    ``prepare_ua_index_spline`` (l.637-676).
    """
    h = 0.025
    x = t / h
    k = np.floor(x)
    alpha = x - k
    t0, t1 = k * h, (k + 1) * h
    y0, y1 = _fit(t0, a), _fit(t1, a)
    d0, d1 = h * _dfit(t0, a), h * _dfit(t1, a)
    dx = y1 - y0
    ddx = d1 + d0
    aa = ddx - 2.0 * dx
    bb = 3.0 * dx - ddx - d0
    return y0 + alpha * (d0 + alpha * (bb + alpha * aa))


def test_coefficients_are_echams():
    """``cavl1..5`` and ``cavi1..5`` of mo_echam_convect_tables.f90 l.42-52."""
    assert es.WATER_COEFFICIENTS == (
        -6096.9385, 21.2409642, -2.711193, 1.673952, 2.433502)
    assert es.ICE_COEFFICIENTS == (
        -6024.5282, 29.32707, 1.0613868, -1.3198825, -0.49382577)


def test_values_at_known_points():
    """Sonntag (1990): 611.2 Pa over water at 0 degC, 611.15 over ice."""
    assert float(es.es_water(273.15)) == pytest.approx(611.2, rel=2e-4)
    assert float(es.es_ice(273.15)) == pytest.approx(611.15, rel=2e-4)
    assert float(es.es_water(293.15)) == pytest.approx(2339.2, rel=2e-3)
    assert float(es.es_ice(253.15)) == pytest.approx(103.2, rel=3e-3)


def test_the_spline_tabulates_the_fit_to_float_precision():
    """The fit differs from ECHAM's 0.025 K spline by < 1e-10 of the value."""
    t = np.random.default_rng(0).uniform(150.0, 330.0, 20000)
    for coefficients, func in ((es.WATER_COEFFICIENTS, es.es_water),
                               (es.ICE_COEFFICIENTS, es.es_ice)):
        spline = _echam_spline(t, coefficients)
        fit = np.asarray(func(jnp.asarray(t)))
        assert np.max(np.abs(fit / spline - 1.0)) < 1e-10


@pytest.mark.parametrize("phase", ["water", "ice"])
def test_log_derivative_is_the_fits(phase):
    es_f = getattr(es, f"es_{phase}")
    dln = getattr(es, f"dlnes_dT_{phase}")
    t = jnp.linspace(180.0, 320.0, 15)
    ad = jax.vmap(jax.grad(lambda x: jnp.log(es_f(x))))(t)
    np.testing.assert_allclose(np.asarray(dln(t)), np.asarray(ad), rtol=1e-12)


def test_lo2_is_echams_strict_switch():
    cthomi, csecfrl = c.tmelt - 35.0, 5e-6
    t = jnp.array([230.0, cthomi, 255.0, 255.0, 255.0, c.tmelt, 280.0])
    qi = jnp.array([0.0, 0.0, 0.0, 5e-6, 6e-6, 1e-3, 1e-3])
    got = np.asarray(es.lo2_ice_phase(t, qi, csecfrl, cthomi))
    assert list(got) == [True, False, False, False, True, False, False]


def test_qsat_form_and_cap():
    """``x/(1 - vtmpc1·x)``, ``x = min(es·rd/rv/p, 0.5)`` (mo_cover l.221-223)."""
    e = jnp.array([100.0, 2000.0, 60000.0])
    p = jnp.array([50000.0, 100000.0, 60000.0])
    eps = c.rd / c.rv
    got = np.asarray(es.qsat_from_es(e, p))
    direct = eps * np.asarray(e) / (np.asarray(p) - (1 - eps) * np.asarray(e))
    np.testing.assert_allclose(got[:2], direct[:2], rtol=1e-12)
    capped = 0.5 / (1.0 - c.vtmpc1 * 0.5)
    assert got[2] == pytest.approx(capped)
