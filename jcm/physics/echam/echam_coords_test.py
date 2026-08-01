"""Tests for EchamCoords with both sigma and hybrid vertical coordinates.

Verifies that:
1. EchamCoords.from_coordinate_system builds correctly for both coord types
2. calculate_pressure_full/half return correct pressure for both
3. For pure sigma coords, a=0 and b=sigma_boundaries, recovering p = sigma * P_s
4. For hybrid coords, p = a + b * P_s exactly (not the old sigma approximation)
"""

import unittest
import jax.numpy as jnp
import numpy as np
from dinosaur.sigma_coordinates import SigmaCoordinates

from jcm.physics.echam.echam_coords import EchamCoords


class TestEchamCoordsSigma(unittest.TestCase):
    """EchamCoords built from pure sigma coordinates."""

    def setUp(self):
        from jcm.utils import get_coords
        self.sigma_vertical = SigmaCoordinates.equidistant(8)
        self.coords = get_coords(self.sigma_vertical, spectral_truncation=21)

    def test_from_coordinate_system_sigma(self):
        ic = EchamCoords.from_coordinate_system(self.coords)
        # Sigma coords are stored as a=0, b=sigma.
        np.testing.assert_allclose(ic.a_half, 0.0, atol=1e-6)
        np.testing.assert_allclose(
            ic.b_half, self.sigma_vertical.boundaries, atol=1e-6
        )

    def test_pressure_at_reference_surface(self):
        """For sigma coords, pressure at full levels is exactly sigma * P_s."""
        ic = EchamCoords.from_coordinate_system(self.coords)
        p_s = jnp.array([101325.0, 90000.0, 100000.0])  # 3 columns
        p_full = ic.calculate_pressure_full(p_s)
        expected = jnp.asarray(self.sigma_vertical.centers)[:, None] * p_s[None, :]
        np.testing.assert_allclose(p_full, expected, rtol=1e-6)

    def test_pressure_at_half_levels_sigma(self):
        ic = EchamCoords.from_coordinate_system(self.coords)
        p_s = jnp.array([101325.0])
        p_half = ic.calculate_pressure_half(p_s)
        expected = jnp.asarray(self.sigma_vertical.boundaries)[:, None] * p_s[None, :]
        np.testing.assert_allclose(p_half, expected, rtol=1e-6)


class TestEchamCoordsHybrid(unittest.TestCase):
    """EchamCoords built from ECHAM hybrid coordinates."""

    def setUp(self):
        from jcm.utils import get_coords
        from jcm.physics.echam.echam_levels import get_echam_levels
        self.hybrid = get_echam_levels(47)
        self.coords = get_coords(self.hybrid, spectral_truncation=31)

    def test_from_coordinate_system_hybrid(self):
        """Hybrid EchamCoords stores a/b coefficients directly (both in Pa)."""
        ic = EchamCoords.from_coordinate_system(self.coords)
        np.testing.assert_allclose(
            ic.a_half, self.hybrid.a_boundaries, atol=1e-6
        )
        np.testing.assert_allclose(
            ic.b_half, self.hybrid.b_boundaries, atol=1e-6
        )

    def test_pressure_at_reference_surface(self):
        """p_full = a_full + b_full * P_s at the reference pressure."""
        ic = EchamCoords.from_coordinate_system(self.coords)
        p_s_ref = 101325.0
        p_s = jnp.array([p_s_ref])
        p_full = ic.calculate_pressure_full(p_s)
        a_centers = 0.5 * (self.hybrid.a_boundaries[:-1] + self.hybrid.a_boundaries[1:])
        b_centers = 0.5 * (self.hybrid.b_boundaries[:-1] + self.hybrid.b_boundaries[1:])
        expected = a_centers[:, None] + b_centers[:, None] * p_s_ref
        np.testing.assert_allclose(p_full, expected, rtol=1e-4)

    def test_pressure_varies_correctly_with_surface_pressure(self):
        """A hybrid model must give different p_full for different P_s columns
        following p = a + b*P_s, NOT p = (a/P_s_ref + b) * P_s.

        This is the bug that broke the hybrid runs: the old EchamCoords stored
        only sigma = (a + b*P_s_ref)/P_s_ref, which is only correct at P_s_ref.
        """
        ic = EchamCoords.from_coordinate_system(self.coords)
        p_s_ref = 101325.0
        p_s_low = 70000.0   # e.g. terrain-high grid point
        p_s = jnp.array([p_s_ref, p_s_low])
        p_full = ic.calculate_pressure_full(p_s)

        a_centers = 0.5 * (self.hybrid.a_boundaries[:-1] + self.hybrid.a_boundaries[1:])
        b_centers = 0.5 * (self.hybrid.b_boundaries[:-1] + self.hybrid.b_boundaries[1:])
        expected = a_centers[:, None] + b_centers[:, None] * p_s[None, :]
        np.testing.assert_allclose(p_full, expected, rtol=1e-4)

        # The old sigma-only formula would give:
        fsg_approx = a_centers / p_s_ref + b_centers
        sigma_style = fsg_approx[:, None] * p_s[None, :]
        # ...which differs from the correct answer by a meaningful amount in
        # the stratosphere where b ≈ 0 and a dominates.
        strato_correct = expected[0, 1]       # top level, low P_s column
        strato_sigma = sigma_style[0, 1]
        self.assertFalse(
            jnp.isclose(strato_correct, strato_sigma, rtol=0.01),
            f"At low P_s, hybrid pressure {float(strato_correct)} should differ "
            f"from sigma approx {float(strato_sigma)} by >1%"
        )

    def test_pressure_monotonic(self):
        """Pressure must increase monotonically from TOA to surface for any P_s."""
        ic = EchamCoords.from_coordinate_system(self.coords)
        for p_s in [40000.0, 70000.0, 101325.0, 105000.0]:
            p_full = ic.calculate_pressure_full(jnp.array([p_s]))
            dp = jnp.diff(p_full[:, 0])
            self.assertTrue(
                jnp.all(dp > 0),
                f"p_full not monotonic for P_s={p_s}: dp={dp}"
            )


if __name__ == "__main__":
    unittest.main()
