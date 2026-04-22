"""RCE-style integration test for the Tiedtke-Nordeng convection scheme.

Covers the full scheme end-to-end on a tropical sounding with CAPE > 1000
J/kg, validating that our fixes (iterative saturation adjustment, wired-up
post-convection adjustment, dynamic LNB termination, Nordeng organized
entrainment) produce physically sensible tendencies: latent heating in
the cloud layer, drying of the boundary layer, and positive precipitation.

These are the RCE signatures that were missing / wrong before the fixes.
"""

import unittest
import jax.numpy as jnp

from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
    ConvectionParameters,
    tiedtke_nordeng_convection,
    saturation_mixing_ratio,
)


def _tropical_sounding(nlev: int = 47, surface_T: float = 302.0,
                      surface_rh: float = 0.8, lapse_K_per_km: float = 6.5):
    """Build a conditionally-unstable tropical sounding (index 0 = TOA)."""
    # Pressure: 10 hPa (TOA) → 1000 hPa (surface)
    p = jnp.logspace(jnp.log10(1000.0), jnp.log10(100_000.0), nlev)
    z_km = -8.4 * jnp.log(p / 100_000.0)  # approx hypsometric height

    # T: standard lapse rate to 15 km, isothermal above
    T = jnp.maximum(surface_T - lapse_K_per_km * z_km, 200.0)

    # Humidity: prescribed RH, drying aloft
    qs = saturation_mixing_ratio(p, T)
    rh = jnp.where(p > 50_000.0, surface_rh, surface_rh * (p / 50_000.0))
    q = rh * qs

    # Density and layer thickness (approximate hydrostatic)
    rho = p / (287.0 * T)
    # layer_thickness in meters: dz ≈ - dp / (ρ g); use level midpoints
    dp = jnp.concatenate([jnp.diff(p), jnp.array([p[-1] * 0.02])])
    dz = jnp.abs(dp) / (rho * 9.81)

    return T, q, p, dz, rho


class TestRCEConvection(unittest.TestCase):
    """Full-scheme RCE-style integration tests."""

    def test_tropical_sounding_fires_convection(self):
        """On a sounding with CAPE > 1000 J/kg the scheme should produce:
        - non-zero tendencies
        - positive precipitation
        - non-zero updraft mass flux
        """
        # Use a very warm, moist sounding to guarantee CAPE > 1000 J/kg
        # (deep convection threshold in the scheme).
        T, q, p, dz, rho = _tropical_sounding(
            surface_T=305.0, surface_rh=0.9, lapse_K_per_km=7.0
        )
        nlev = T.shape[0]
        u = jnp.zeros(nlev)
        v = jnp.zeros(nlev)
        qc = jnp.zeros(nlev)
        qi = jnp.zeros(nlev)
        dt = 1800.0  # 30 min
        cfg = ConvectionParameters.default()

        tendencies, state = tiedtke_nordeng_convection(
            T, q, p, dz, rho, u, v, qc, qi, dt, cfg,
        )
        # Should have nonzero temperature tendency somewhere
        self.assertGreater(
            float(jnp.max(jnp.abs(tendencies.dtedt))), 1e-6,
            "Convection should produce nonzero T tendency on unstable sounding",
        )
        # Surface precipitation should be positive
        self.assertGreater(
            float(tendencies.precip_conv), 0.0,
            "Unstable sounding should produce positive precipitation",
        )
        # Updraft mass flux should be active somewhere in the cloud
        self.assertGreater(
            float(jnp.max(state.mfu)), 1e-4,
            "Updraft mass flux should activate on unstable sounding",
        )

    def test_stable_sounding_no_convection(self):
        """On a stable sounding (cold surface) the scheme should return zero
        tendencies — ensures we haven't introduced spurious activation.
        """
        T, q, p, dz, rho = _tropical_sounding(
            surface_T=260.0, surface_rh=0.5, lapse_K_per_km=2.0
        )
        nlev = T.shape[0]
        u = jnp.zeros(nlev)
        v = jnp.zeros(nlev)
        qc = jnp.zeros(nlev)
        qi = jnp.zeros(nlev)
        cfg = ConvectionParameters.default()

        tendencies, state = tiedtke_nordeng_convection(
            T, q, p, dz, rho, u, v, qc, qi, 1800.0, cfg,
        )
        # No convection → no tendencies, no precip
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(tendencies.dtedt))), 0.0, places=8,
        )
        self.assertAlmostEqual(float(tendencies.precip_conv), 0.0, places=8)

    def test_convective_heating_pattern(self):
        """Latent heat release from convection should warm the cloud layer
        (mid-troposphere) and leave the boundary layer approximately
        unchanged or slightly cooled (downdraft detrainment + evaporation).
        """
        T, q, p, dz, rho = _tropical_sounding(
            surface_T=305.0, surface_rh=0.9, lapse_K_per_km=7.0
        )
        nlev = T.shape[0]
        cfg = ConvectionParameters.default()

        tendencies, _ = tiedtke_nordeng_convection(
            T, q, p, dz, rho,
            jnp.zeros(nlev), jnp.zeros(nlev),
            jnp.zeros(nlev), jnp.zeros(nlev),
            1800.0, cfg,
        )
        # Mid-troposphere (e.g. 400-700 hPa) is where condensation heating
        # dominates. Find those indices: p in [40000, 70000].
        mid_mask = jnp.logical_and(p > 40_000.0, p < 70_000.0)
        mid_heating = jnp.where(mid_mask, tendencies.dtedt, 0.0)
        self.assertGreater(
            float(jnp.sum(mid_heating)), 0.0,
            f"Expected net positive heating in 400-700 hPa; "
            f"got dtedt[mid] sum = {float(jnp.sum(mid_heating)):.3e}",
        )

    def test_convective_drying_in_cloud_layer(self):
        """Condensation removes vapour from the column during convection —
        the integrated q tendency should be negative (condensed → precip)
        minus what was transported up from the BL.
        """
        T, q, p, dz, rho = _tropical_sounding(surface_T=302.0, surface_rh=0.85)
        nlev = T.shape[0]
        cfg = ConvectionParameters.default()

        tendencies, _ = tiedtke_nordeng_convection(
            T, q, p, dz, rho,
            jnp.zeros(nlev), jnp.zeros(nlev),
            jnp.zeros(nlev), jnp.zeros(nlev),
            1800.0, cfg,
        )
        # Some level should have dqdt < 0 (drying) from condensation
        self.assertLess(
            float(jnp.min(tendencies.dqdt)), 0.0,
            f"Expected some drying tendency from condensation; "
            f"min dqdt = {float(jnp.min(tendencies.dqdt)):.3e}",
        )

    def test_saturation_adjustment_leaves_no_supersaturation(self):
        """After the scheme runs (including the post-convection adjustment
        we wired up), applying the tendencies should leave the column at
        or below saturation everywhere.
        """
        T, q, p, dz, rho = _tropical_sounding(surface_T=302.0, surface_rh=0.85)
        nlev = T.shape[0]
        cfg = ConvectionParameters.default()
        dt = 1800.0

        tendencies, _ = tiedtke_nordeng_convection(
            T, q, p, dz, rho,
            jnp.zeros(nlev), jnp.zeros(nlev),
            jnp.zeros(nlev), jnp.zeros(nlev),
            dt, cfg,
        )
        T_new = T + tendencies.dtedt * dt
        q_new = q + tendencies.dqdt * dt
        qs_new = saturation_mixing_ratio(p, T_new)
        supersaturation = jnp.maximum(q_new - qs_new, 0.0)
        max_super = float(jnp.max(supersaturation))
        self.assertLess(
            max_super, 1e-4,
            f"Post-tendency state still supersaturated by "
            f"{max_super*1000:.3f} g/kg; the post-convection saturation "
            f"adjustment is not working",
        )


if __name__ == "__main__":
    unittest.main()
