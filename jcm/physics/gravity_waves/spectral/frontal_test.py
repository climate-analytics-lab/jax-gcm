"""Tests for the frontal gravity-wave source (gw_front.F90 port)."""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.gravity_waves.spectral.frontal import (
    flat_spectrum,
    gaussian_spectrum,
    gw_cm_src,
)
from jcm.physics.gravity_waves.spectral.solver import GWBand, get_unit_vector

NGWV = 8
DC = 2.5
BAND = GWBand(dc=DC, fcrit2=1.0, wavelength=1.0e5, ngwv=NGWV)
KSRC = 5
NLEV = 12


class SpectrumTest(unittest.TestCase):
    def test_flat_spectrum(self):
        tau = np.asarray(flat_spectrum(BAND, 1.5e-3))
        self.assertEqual(tau.shape, (2 * NGWV + 1,))
        self.assertEqual(tau[NGWV], 0.0)          # l = 0 prohibited
        mask = np.ones_like(tau, dtype=bool)
        mask[NGWV] = False
        np.testing.assert_allclose(tau[mask], 1.5e-3)

    def test_gaussian_spectrum_is_bin_average(self):
        height, width = 1.25e-3, 30.0
        with jax.enable_x64():
            tau = np.asarray(gaussian_spectrum(
                GWBand(dc=DC, fcrit2=1.0, wavelength=1.0e5, ngwv=NGWV),
                height, width))
            cref = DC * np.arange(-NGWV, NGWV + 1)
            for i, c0 in enumerate(cref):
                if i == NGWV:
                    self.assertEqual(tau[i], 0.0)  # l = 0 prohibited
                    continue
                # Dense quadrature of the Gaussian over the bin.
                x = np.linspace(c0 - 0.5 * DC, c0 + 0.5 * DC, 20001)
                avg = np.trapezoid(
                    height * np.exp(-((x / width) ** 2)), x) / DC
                np.testing.assert_allclose(tau[i], avg, rtol=1e-6)

    def test_gaussian_spectrum_symmetry_and_center(self):
        tau = np.asarray(gaussian_spectrum(BAND, 1.0e-3, 30.0))
        # Centered on c=0: symmetric about l=0 (float32 erfc rounding).
        np.testing.assert_allclose(tau, tau[::-1], rtol=1e-5)
        # Shifted center moves the peak.
        tau_shift = np.asarray(gaussian_spectrum(BAND, 1.0e-3, 10.0,
                                                 center=10.0))
        self.assertEqual(int(np.argmax(tau_shift)), NGWV + 4)  # c = +10 m/s


class GetUnitVectorTest(unittest.TestCase):
    def test_magnitude_and_zero_safety(self):
        u = jnp.asarray([3.0, 0.0, -1.0])
        v = jnp.asarray([4.0, 0.0, 0.0])
        xv, yv, mag = get_unit_vector(u, v)
        np.testing.assert_allclose(np.asarray(mag), [5.0, 0.0, 1.0])
        np.testing.assert_allclose(np.asarray(xv), [0.6, 0.0, -1.0])
        np.testing.assert_allclose(np.asarray(yv), [0.8, 0.0, 0.0])

    def test_grad_finite_at_zero(self):
        def f(u, v):
            xv, yv, mag = get_unit_vector(u, v)
            return jnp.sum(xv**2 + yv**2 + mag**2)

        gu = jax.grad(f)(jnp.zeros(3), jnp.zeros(3))
        self.assertTrue(bool(jnp.all(jnp.isfinite(gu))))


class GwCmSrcTest(unittest.TestCase):
    def _uv(self, ncols=3):
        rng = np.random.default_rng(0)
        u = jnp.asarray(rng.normal(10.0, 5.0, (NLEV, ncols)))
        v = jnp.asarray(rng.normal(0.0, 3.0, (NLEV, ncols)))
        return u, v

    def test_launch_mask_thresholding(self):
        u, v = self._uv()
        src_tau = flat_spectrum(BAND, 1.5e-3)
        frontgf = jnp.asarray([0.0, 2.9e-15, 5.0e-15])
        src = gw_cm_src(BAND, KSRC, u, v, frontgf, 3.0e-15, src_tau)
        np.testing.assert_array_equal(np.asarray(src.launch),
                                      [False, False, True])
        # Non-launching columns get exactly zero stress.
        self.assertEqual(float(jnp.abs(src.tau_src[:, :2]).max()), 0.0)
        # Launching column carries the spectrum.
        np.testing.assert_allclose(np.asarray(src.tau_src[:, 2]),
                                   np.asarray(src_tau))

    def test_phase_speeds_are_cref_plus_source_wind(self):
        u, v = self._uv()
        src_tau = flat_spectrum(BAND, 1.5e-3)
        src = gw_cm_src(BAND, KSRC, u, v, jnp.zeros(3), 3.0e-15, src_tau)
        cref = DC * np.arange(-NGWV, NGWV + 1)
        expected = cref[:, None] + np.asarray(src.ubi[KSRC + 1])[None, :]
        np.testing.assert_allclose(np.asarray(src.c), expected, rtol=1e-6)
        # The source interface wind equals the source wind magnitude.
        usrc = 0.5 * (np.asarray(u)[KSRC + 1] + np.asarray(u)[KSRC])
        vsrc = 0.5 * (np.asarray(v)[KSRC + 1] + np.asarray(v)[KSRC])
        np.testing.assert_allclose(np.asarray(src.ubi[KSRC + 1]),
                                   np.hypot(usrc, vsrc), rtol=1e-6)

    def test_projection_identities(self):
        u, v = self._uv()
        src = gw_cm_src(BAND, KSRC, u, v, jnp.zeros(3), 3.0e-15,
                        flat_spectrum(BAND, 1.5e-3))
        # Unit vector.
        np.testing.assert_allclose(
            np.asarray(src.xv**2 + src.yv**2), 1.0, rtol=1e-6)
        # ubm is the projection of (u, v).
        np.testing.assert_allclose(
            np.asarray(src.ubm),
            np.asarray(u) * np.asarray(src.xv) + np.asarray(v) * np.asarray(src.yv),
            rtol=1e-6)
        # Top interface takes the top midpoint value.
        np.testing.assert_allclose(np.asarray(src.ubi[0]),
                                   np.asarray(src.ubm[0]), rtol=1e-6)

    def test_column_vs_batch_broadcasting(self):
        u, v = self._uv()
        src_tau = flat_spectrum(BAND, 1.5e-3)
        frontgf = jnp.asarray([0.0, 2.9e-15, 5.0e-15])
        batch = gw_cm_src(BAND, KSRC, u, v, frontgf, 3.0e-15, src_tau)
        for i in range(3):
            single = gw_cm_src(BAND, KSRC, u[:, i], v[:, i], frontgf[i],
                               3.0e-15, src_tau)
            np.testing.assert_allclose(np.asarray(batch.ubm[:, i]),
                                       np.asarray(single.ubm), rtol=1e-6)
            np.testing.assert_allclose(np.asarray(batch.tau_src[:, i]),
                                       np.asarray(single.tau_src), rtol=1e-6)
            np.testing.assert_allclose(np.asarray(batch.c[:, i]),
                                       np.asarray(single.c), rtol=1e-6)

    def test_flat_reference_values(self):
        # Cross-check the Gaussian builder against a math.erfc loop
        # (the reference form in gw_front.F90).
        height, width = 1.25e-3, 30.0
        cref = DC * np.arange(-NGWV, NGWV + 1)
        bounds = np.concatenate([cref - 0.5 * DC, [cref[-1] + 0.5 * DC]])
        integ = np.array([math.erfc(b / width) for b in bounds])
        integ *= height * width * math.sqrt(math.pi) / 2.0
        expected = (integ[:-1] - integ[1:]) / DC
        expected[NGWV] = 0.0
        tau = np.asarray(gaussian_spectrum(BAND, height, width))
        np.testing.assert_allclose(tau, expected, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
