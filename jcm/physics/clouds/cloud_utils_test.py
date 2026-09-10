"""Tests for the shared cloud microphysics utilities."""

from math import pi

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from .cloud_utils import (
    breadth_factor,
    eff_liquid_droplet_radius,
    ice_volume_mean_radius,
)
from .lohmann_2m_params import CloudParams2M

_EPS = 1.1920929e-7  # float32 machine epsilon, as CloudParams2M.eps


def _echam_reference_radius(qc_in_cloud, air_density, cdnc_m3):
    """ECHAM6 ``mo_cloud_optics.f90`` form: 62.035 um * kappa * (LWC/N)^(1/3)
    with LWC in g/m^3 and N in cm^-3 — an independent recomputation of the law.
    """
    zfact = 1.0e6 * (3.0e-9 / (4.0 * pi * c.rhow)) ** (1.0 / 3.0)
    kappa = 4.5e-10 * cdnc_m3 + 1.18
    lwc_gm3 = qc_in_cloud * air_density * 1.0e3
    cdnc_cm3 = cdnc_m3 * 1.0e-6
    return zfact * kappa * (lwc_gm3 / cdnc_cm3) ** (1.0 / 3.0)


class TestEffLiquidDropletRadius:
    def test_matches_echam_reference_law(self):
        qc = np.array([2.0e-4, 5.0e-5, 1.0e-3])
        rho = np.array([1.0, 0.8, 1.2])
        cdnc = np.array([1.0e8, 5.0e7, 2.0e8])

        got = eff_liquid_droplet_radius(
            jnp.asarray(qc), jnp.asarray(rho), jnp.asarray(cdnc), _EPS,
        )
        np.testing.assert_allclose(
            np.asarray(got), _echam_reference_radius(qc, rho, cdnc), rtol=1e-5,
        )

    def test_hand_computed_value(self):
        # qc = 2e-4 kg/kg, rho = 1 kg/m^3, N = 1e8 m^-3 (100 cm^-3):
        # kappa = 1.225, 62.0345 * (0.2/100)^(1/3) = 7.8157 um -> 9.574 um.
        got = eff_liquid_droplet_radius(
            jnp.array(2.0e-4), jnp.array(1.0), jnp.array(1.0e8), _EPS,
        )
        np.testing.assert_allclose(float(got), 9.5742, rtol=1e-4)

    def test_zero_liquid_is_exactly_zero(self):
        qc = jnp.array([0.0, 1.0e-4, 0.0, -1.0e-9])
        got = eff_liquid_droplet_radius(
            qc, jnp.array(1.0), jnp.array(1.0e8), _EPS,
        )
        # Radiation selects on ``r_eff > 0``, so these must be exact zeros.
        assert float(got[0]) == 0.0
        assert float(got[2]) == 0.0
        assert float(got[3]) == 0.0
        assert float(got[1]) > 0.0

    def test_liquid_cloud_flag_masks_to_exact_zero(self):
        qc = jnp.full((3,), 1.0e-4)
        flag = jnp.array([True, False, True])
        got = eff_liquid_droplet_radius(
            qc, jnp.array(1.0), jnp.array(1.0e8), _EPS, liquid_cloud_flag=flag,
        )
        assert float(got[1]) == 0.0
        assert float(got[0]) > 0.0 and float(got[2]) > 0.0

    def test_increases_with_liquid_water_at_fixed_cdnc(self):
        qc = jnp.array([1.0e-5, 1.0e-4, 1.0e-3])
        got = eff_liquid_droplet_radius(
            qc, jnp.array(1.0), jnp.array(1.0e8), _EPS,
        )
        assert np.all(np.diff(np.asarray(got)) > 0.0)
        # Cube-root scaling: a 10x LWC increase is a 10^(1/3) radius increase.
        np.testing.assert_allclose(
            float(got[1] / got[0]), 10.0 ** (1.0 / 3.0), rtol=1e-4,
        )

    def test_gradient_finite_at_zero_liquid(self):
        # The cube root's derivative is infinite at 0; the double-``where``
        # guard is what stops the reverse pass returning NaN there.
        def total(qc):
            return jnp.sum(
                eff_liquid_droplet_radius(
                    qc, jnp.full((3,), 1.0), jnp.full((3,), 1.0e8), _EPS,
                )
            )

        grad_zero = jax.grad(total)(jnp.zeros((3,)))
        assert jnp.all(jnp.isfinite(grad_zero))
        assert np.all(np.asarray(grad_zero) == 0.0)

        grad_mixed = jax.grad(total)(jnp.array([0.0, 1.0e-4, 0.0]))
        assert jnp.all(jnp.isfinite(grad_mixed))
        assert float(grad_mixed[1]) > 0.0

    def test_cdnc_gradient_finite_at_zero_liquid(self):
        def total(cdnc):
            return jnp.sum(
                eff_liquid_droplet_radius(
                    jnp.zeros((3,)), jnp.full((3,), 1.0), cdnc, _EPS,
                )
            )

        grad = jax.grad(total)(jnp.full((3,), 1.0e8))
        assert jnp.all(jnp.isfinite(grad))

    def test_uses_breadth_factor(self):
        cdnc = jnp.array(2.0e8)
        got = eff_liquid_droplet_radius(
            jnp.array(1.0e-4), jnp.array(1.0), cdnc, _EPS,
        )
        base = (3.0 / (4.0 * pi * c.rhow)) * 1.0e-4 * 1.0 / 2.0e8
        np.testing.assert_allclose(
            float(got),
            float(1.0e6 * breadth_factor(cdnc) * base ** (1.0 / 3.0)),
            rtol=1e-6,
        )


class TestIceVolumeMeanRadius:
    """The ``prid`` contract: a VOLUME-mean radius in METRES (#725).

    ``update_in_cloud_water`` inverts this as
    ``N = rho*q_i / ((4/3)*pi*prid^3*rho_ice)``, so the unit is load-bearing:
    returning the microns that ``eff_ice_crystal_radius`` produces understates
    crystal number by ~1e18, pinning ICNC at ``icemin`` and saturating the ice
    effective radius at ``ceffmax``.
    """

    _P = CloudParams2M.default()

    def test_result_is_metres_not_microns(self):
        # Realistic cirrus: 0.01 g/m^3 in-cloud IWC, 50 crystals per litre.
        r = ice_volume_mean_radius(
            jnp.array([1.0e-2]), jnp.array([5.0e4]), self._P,
        )
        # A crystal radius in metres is O(1e-5); in microns it would be O(10).
        assert jnp.all(r > 1.0e-6) and jnp.all(r < 1.0e-3), r

    def test_matches_the_schumann_chain(self):
        ice_gm3, icnc = jnp.array([1.0e-2]), jnp.array([5.0e4])
        # Independent transcription: Lohmann (2008) r_eff, clip, Schumann (2011).
        base = ice_gm3 / (self._P.fact_PK * icnc)
        r_eff = np.clip(
            0.5e4 * base ** (1.0 / self._P.pow_PK),
            self._P.ceffmin, self._P.ceffmax,
        )
        zrih = -2261.0 + np.sqrt(5113188.0 + 2809.0 * r_eff**3)
        expected = 1.0e-6 * zrih ** (1.0 / 3.0)
        got = ice_volume_mean_radius(ice_gm3, icnc, self._P)
        np.testing.assert_allclose(got, expected, rtol=1e-5)

    def test_inverts_to_a_physical_crystal_number(self):
        """Round-trip: the radius must return ICNC to the order it came from.

        This is the property the #725 bug broke -- the number came back ~1e18
        too small, so every cell fell to the ``icemin`` floor.
        """
        icnc_in, ice_gm3 = 5.0e4, 1.0e-2
        r = float(ice_volume_mean_radius(
            jnp.array([ice_gm3]), jnp.array([icnc_in]), self._P,
        )[0])
        # N = rho*q_i / ((4/3) pi r^3 rho_ice), with rho*q_i = IWC in kg/m^3.
        n_back = (ice_gm3 * 1.0e-3) / (4.0 / 3.0 * pi * r**3 * float(self._P.rhoice))
        assert 1.0e3 < n_back < 1.0e7, n_back
        assert n_back > 100.0 * float(self._P.icemin)

    def test_larger_ice_gives_larger_crystals(self):
        icnc = jnp.full((3,), 5.0e4)
        r = ice_volume_mean_radius(jnp.array([1e-3, 1e-2, 1e-1]), icnc, self._P)
        assert jnp.all(jnp.diff(r) > 0.0)

    def test_gradient_is_finite_at_zero_ice(self):
        g = jax.grad(
            lambda x: ice_volume_mean_radius(x, jnp.array([5.0e4]), self._P).sum(),
        )(jnp.array([0.0]))
        assert jnp.all(jnp.isfinite(g)), g
