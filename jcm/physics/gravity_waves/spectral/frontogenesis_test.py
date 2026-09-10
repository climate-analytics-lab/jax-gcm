"""Tests for the lat-lon frontogenesis-function provider."""

import unittest

import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics.gravity_waves.spectral.frontogenesis import (
    frontogenesis_function,
)


class AnalyticDeformationTest(unittest.TestCase):
    """Deformation flow with a known frontogenesis function.

    On a near-equatorial patch take (periodic-in-lon) fields

        theta = theta0 + G a sin(lon) + H a lat
        u     = alpha a sin(lon),   v = -alpha a lat

    for which (exactly, with spherical metrics)

        theta_x = G cos(lon)/cos(lat),  theta_y = H
        u_x = alpha cos(lon)/cos(lat),  u_y = 0, v_x = 0, v_y = -alpha
        F = alpha * (H^2 - G^2 cos^3(lon)/cos^3(lat))

    The lon-derivatives of sin are second-order accurate; everything in
    lat is linear so the lat differences are exact.
    """

    def test_matches_analytic(self):
        a = c.rearth
        alpha = 1.0e-5           # deformation rate [1/s]
        G = 1.0e-5               # zonal theta gradient [K/m]
        H = 2.0e-5               # meridional theta gradient [K/m]
        nlon, nlat = 128, 41
        lons = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-0.15, 0.15, nlat)
        lam = lons[:, None]
        phi = lats[None, :]

        theta = 300.0 + G * a * np.sin(lam) + H * a * phi
        u = alpha * a * np.sin(lam) + 0.0 * phi
        v = -alpha * a * phi + 0.0 * lam

        F = np.asarray(frontogenesis_function(
            jnp.asarray(u), jnp.asarray(v), jnp.asarray(theta),
            jnp.asarray(lons), jnp.asarray(lats)))

        expected = alpha * (H**2 - G**2 * np.cos(lam)**3 / np.cos(phi)**3)
        # Second-order in dlon = 2*pi/128; relative error ~ dlon^2/6.
        np.testing.assert_allclose(F, expected, rtol=0, atol=3e-3 * np.abs(
            expected).max())

    def test_pure_shear_units_scale(self):
        # F has units K^2/m^2/s: doubling the theta gradient quadruples F,
        # doubling the deformation rate doubles it.
        a = c.rearth
        nlon, nlat = 64, 21
        lons = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-0.1, 0.1, nlat)
        lam, phi = lons[:, None], lats[None, :]
        theta = 300.0 + 1e-5 * a * np.sin(lam) + 0.0 * phi
        u = 1e-5 * a * np.sin(lam) + 0.0 * phi
        v = np.zeros_like(u)
        args = (jnp.asarray(lons), jnp.asarray(lats))
        F1 = np.asarray(frontogenesis_function(
            jnp.asarray(u), jnp.asarray(v), jnp.asarray(theta), *args))
        F2 = np.asarray(frontogenesis_function(
            jnp.asarray(u), jnp.asarray(v), jnp.asarray(2.0 * theta - 300.0),
            *args))
        F3 = np.asarray(frontogenesis_function(
            jnp.asarray(2.0 * u), jnp.asarray(v), jnp.asarray(theta), *args))
        atol = 1e-6 * float(np.abs(F1).max())
        np.testing.assert_allclose(F2, 4.0 * F1, rtol=1e-4, atol=atol)
        np.testing.assert_allclose(F3, 2.0 * F1, rtol=1e-4, atol=atol)


class PeriodicityAndShapeTest(unittest.TestCase):
    def test_periodic_in_longitude(self):
        rng = np.random.default_rng(1)
        nlev, nlon, nlat = 3, 32, 16
        lons = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False)
        lats = np.linspace(-1.4, 1.4, nlat)   # up to ~80 degrees
        u = rng.normal(0, 10, (nlev, nlon, nlat))
        v = rng.normal(0, 10, (nlev, nlon, nlat))
        theta = 300 + rng.normal(0, 5, (nlev, nlon, nlat))
        F = np.asarray(frontogenesis_function(
            jnp.asarray(u), jnp.asarray(v), jnp.asarray(theta),
            jnp.asarray(lons), jnp.asarray(lats)))
        self.assertEqual(F.shape, (nlev, nlon, nlat))
        self.assertTrue(np.all(np.isfinite(F)))
        # Rolling all fields in longitude rolls the output identically
        # (wrap-around stencil, no boundary artifacts).
        shift = 7
        F_roll = np.asarray(frontogenesis_function(
            jnp.asarray(np.roll(u, shift, axis=1)),
            jnp.asarray(np.roll(v, shift, axis=1)),
            jnp.asarray(np.roll(theta, shift, axis=1)),
            jnp.asarray(lons), jnp.asarray(lats)))
        np.testing.assert_allclose(F_roll, np.roll(F, shift, axis=1),
                                   rtol=1e-5,
                                   atol=1e-7 * float(np.abs(F).max()))

    def test_nonuniform_gaussian_like_latitudes(self):
        # Gaussian-quadrature-style (non-uniform) latitudes must work and
        # produce finite values, including the one-sided end rows.
        nlon, nlat = 16, 24
        x, _ = np.polynomial.legendre.leggauss(nlat)
        lats = np.arcsin(x)
        lons = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False)
        lam, phi = lons[:, None], lats[None, :]
        theta = 300.0 + 10.0 * np.sin(phi) + 0.0 * lam
        u = 20.0 * np.cos(phi) + 0.0 * lam
        v = np.zeros_like(u)
        F = np.asarray(frontogenesis_function(
            jnp.asarray(u), jnp.asarray(v), jnp.asarray(theta),
            jnp.asarray(lons), jnp.asarray(lats)))
        self.assertTrue(np.all(np.isfinite(F)))


if __name__ == "__main__":
    unittest.main()
