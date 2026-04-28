"""Sea-ice scheme regression tests.

Targeted tests covering the ice_thickness_evolution path. Currently
focused on the frazil-ice formation mechanism added to address Bug F5
in fortran_harness/PLAN.md (open-ocean cells dropping below the
saline freezing point need to grow new ice rather than continuing to
cool).
"""

import unittest
import jax.numpy as jnp
import numpy as np

from jcm.physics.surface.icon.sea_ice import ice_thickness_evolution
from jcm.physics.surface.icon.surface_types import (
    SurfaceParameters, SurfaceFluxes,
)


def _zero_fluxes(ncol, nsfc=3):
    """A SurfaceFluxes struct with all zero arrays — a quiet surface."""
    z2 = jnp.zeros((ncol, nsfc))
    z1 = jnp.zeros(ncol)
    return SurfaceFluxes(
        sensible_heat=z2,
        latent_heat=z2,
        longwave_net=z2,
        shortwave_net=z2,
        ground_heat=z2,
        momentum_u=z2,
        momentum_v=z2,
        evaporation=z2,
        transpiration=z2,
        sensible_heat_mean=z1,
        latent_heat_mean=z1,
        momentum_u_mean=z1,
        momentum_v_mean=z1,
        evaporation_mean=z1,
    )


class TestFrazilIceFormation(unittest.TestCase):
    """Frazil ice formation when ocean drops below ctfreez (Bug F5)."""

    def setUp(self):
        self.params = SurfaceParameters.default()
        self.dt = 1800.0
        self.ncol = 1
        self.nice_layers = 2
        # No existing ice, no ambient surface fluxes.
        self.zero_ice = jnp.zeros((self.ncol, self.nice_layers))
        # Ice "temperature" set to tmelt so it's not in melting regime;
        # with zero thickness this is mostly a placeholder.
        self.zero_ice_T = jnp.full((self.ncol, self.nice_layers), 273.15)
        self.fluxes = _zero_fluxes(self.ncol)

    def test_no_frazil_when_ocean_above_freezing(self):
        """SST = 280 K → no frazil contribution to top layer.

        Other paths (bottom melting, etc.) may produce small tendencies
        depending on ice conduction; we specifically check that the
        top-layer frazil-ice growth contribution is zero.
        """
        ocean_T = jnp.array([280.0])
        tendency = ice_thickness_evolution(
            self.zero_ice, self.zero_ice_T,
            self.fluxes, ocean_T, self.dt, self.params,
        )
        # Top-layer tendency should not be positive (no frazil growth).
        top_tend = float(tendency[0, 0])
        self.assertLessEqual(top_tend, 0.0,
                             msg=f"Got top-layer tendency {top_tend} for warm "
                                 "ocean — frazil should not fire.")

    def test_no_frazil_at_exactly_ctfreez(self):
        """SST = 271.38 K → no frazil (boundary case, ocean_below_freezing False)."""
        ocean_T = jnp.array([271.38])
        # Save the warm-ocean baseline to subtract off non-frazil contributions.
        warm_tendency = ice_thickness_evolution(
            self.zero_ice, self.zero_ice_T,
            self.fluxes, jnp.array([280.0]), self.dt, self.params,
        )
        cold_tendency = ice_thickness_evolution(
            self.zero_ice, self.zero_ice_T,
            self.fluxes, ocean_T, self.dt, self.params,
        )
        # Top-layer contribution from frazil specifically: difference
        # between cold and warm cases at top layer.
        frazil_contribution = float(cold_tendency[0, 0] - warm_tendency[0, 0])
        self.assertLess(abs(frazil_contribution), 1e-10,
                        msg=f"Got frazil contribution {frazil_contribution} "
                            "at the freezing point.")

    def test_frazil_fires_when_ocean_below_freezing(self):
        """SST = 268 K (below ctfreez) → frazil ice grows at top layer."""
        ocean_T = jnp.array([268.0])
        tendency = ice_thickness_evolution(
            self.zero_ice, self.zero_ice_T,
            self.fluxes, ocean_T, self.dt, self.params,
        )
        # Top-layer tendency should be positive (ice growing).
        top_tend = float(tendency[0, 0])
        self.assertGreater(top_tend, 0.0,
                           msg=f"Expected ice growth at supercooled ocean; "
                               f"got top-layer tendency {top_tend} m/s.")

        # Magnitude check: deficit_heat = (271.38 - 268) * rho_w * cp_w * h_ml
        # = 3.38 K * 1025 kg/m³ * 3994 J/kg/K * 50 m
        # ≈ 6.92e8 J/m². Frazil thickness over dt=1800s:
        # ~6.92e8 / (910 kg/m³ * 334000 J/kg) = 2.28 m total.
        # Per second: 2.28 / 1800 = 1.27e-3 m/s.
        # (Allow ~10% tolerance — params may differ slightly.)
        expected_rate = (
            (271.38 - 268.0)
            * float(self.params.rho_water)
            * float(self.params.cp_water)
            * float(self.params.ml_depth)
            / (float(self.params.rho_ice) * 334000.0 * self.dt)
        )
        self.assertAlmostEqual(top_tend, expected_rate, delta=0.1 * expected_rate,
                               msg=f"Frazil rate {top_tend:.4e} ≠ expected "
                                   f"{expected_rate:.4e} m/s.")


if __name__ == "__main__":
    unittest.main()
