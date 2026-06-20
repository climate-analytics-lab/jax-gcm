"""Tests for the Mie kernel and lookup-table interpolation."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.optics.mie import mie_efficiencies
from jcm.physics.aerosol.jam.optics.mie_lut import build_mie_lut, interp_mie


class MieKernelTest(unittest.TestCase):
    def test_reference_values(self):
        # Validated against an independent scipy-Bessel Mie implementation.
        cases = {
            (1.0, 1.5, 0.0): (0.2151, 1.0, 0.1989),
            (10.0, 1.33, 0.0): (2.2065, 1.0, 0.7125),
            (2.0, 1.53, 0.006): (2.0193, 0.9729, 0.6165),
            (8.0, 1.55, 0.4): (2.4234, 0.4894, 0.8978),
        }
        for (x, mr, mi), (qe, ss, g) in cases.items():
            q, s, gg = mie_efficiencies(x, mr, mi)
            self.assertAlmostEqual(q, qe, places=3)
            self.assertAlmostEqual(s, ss, places=3)
            self.assertAlmostEqual(gg, g, places=3)

    def test_nonabsorbing_ssa_unity(self):
        for x in (0.3, 1.0, 5.0, 30.0):
            _, ssa, _ = mie_efficiencies(x, 1.5, 0.0)
            self.assertAlmostEqual(ssa, 1.0, places=4)

    def test_large_x_qext_approaches_two(self):
        q, _, _ = mie_efficiencies(95.0, 1.5, 0.0)
        self.assertTrue(1.9 < q < 2.3)

    def test_rayleigh_g_small(self):
        _, _, g = mie_efficiencies(0.05, 1.5, 0.0)
        self.assertLess(abs(g), 0.05)

    def test_asymmetry_increases_with_size(self):
        _, _, g_small = mie_efficiencies(0.5, 1.5, 0.0)
        _, _, g_big = mie_efficiencies(8.0, 1.5, 0.0)
        self.assertGreater(g_big, g_small)


class MieLUTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lut = build_mie_lut(nx=48, nmr=20, nmi=20)

    def test_interp_matches_kernel_offgrid(self):
        # Off-grid points should interpolate close to the direct kernel.
        pts = [(1.7, 1.44, 0.01), (6.3, 1.6, 0.05), (0.7, 1.5, 1e-8),
               (20.0, 1.52, 0.003)]
        x = jnp.array([p[0] for p in pts])
        mr = jnp.array([p[1] for p in pts])
        mi = jnp.array([p[2] for p in pts])
        qe, ssa, g = interp_mie(self.lut, x, mr, mi)
        for i, (px, pmr, pmi) in enumerate(pts):
            rqe, rss, rg = mie_efficiencies(px, pmr, pmi)
            self.assertAlmostEqual(float(qe[i]), rqe, delta=0.15 * max(rqe, 0.1))
            self.assertAlmostEqual(float(ssa[i]), rss, delta=0.05)
            self.assertAlmostEqual(float(g[i]), rg, delta=0.06)

    def test_interp_finite_and_bounded(self):
        x = jnp.linspace(0.05, 99.0, 50)
        mr = jnp.full((50,), 1.55)
        mi = jnp.full((50,), 0.02)
        qe, ssa, g = interp_mie(self.lut, x, mr, mi)
        self.assertTrue(np.all(np.isfinite(np.asarray(qe))))
        self.assertTrue(bool(jnp.all((ssa >= 0) & (ssa <= 1))))
        self.assertTrue(bool(jnp.all((g >= -1) & (g <= 1))))

    def test_interp_differentiable(self):
        lut = self.lut

        def loss(x):
            qe, _, _ = interp_mie(lut, x, jnp.full_like(x, 1.5), jnp.full_like(x, 0.01))
            return jnp.sum(qe)

        g = jax.grad(loss)(jnp.array([1.0, 5.0, 20.0]))
        self.assertTrue(np.all(np.isfinite(np.asarray(g))))


if __name__ == "__main__":
    unittest.main()
