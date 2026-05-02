"""Tests for ECHAM exchange coefficient coupling and surface physics integration.

Tests that exchange coefficients computed by vertical diffusion are properly
passed to surface physics, vary with atmospheric stability and surface type,
and have physically reasonable magnitudes.
"""

import unittest
import numpy as np
import jax.numpy as jnp

from jcm.physics.echam.echam_physics import (
    _prepare_common_physics_state,
    apply_vertical_diffusion,
    apply_surface,
)
from jcm.physics.echam.echam_physics_data import PhysicsData
from jcm.physics.echam.echam_coords import EchamCoords
from jcm.physics.echam.parameters import Parameters
from jcm.physics_interface import PhysicsState
from jcm.date import DateData
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.utils import get_coords


def _make_echam_state(nlev=40, nlat=32, nlon=64, surface_temp_k=300.0,
                     atm_temp_k=290.0, wind_speed=5.0, specific_humidity=0.01,
                     fmask=0.0):
    """Create a realistic ECHAM physics state in column format for testing.

    Returns state, physics_data, parameters, forcing, terrain already prepared
    with diagnostics (pressure, height, density) computed.
    """
    sigma_boundaries = np.linspace(0, 1, nlev + 1)
    coords = get_coords(sigma_boundaries, nodal_shape=(nlon, nlat))
    echam_coords = EchamCoords.from_coordinate_system(coords)
    ncols = nlat * nlon

    # Build a realistic temperature profile (warm at surface, cooling aloft)
    sigma_mid = (sigma_boundaries[:-1] + sigma_boundaries[1:]) / 2
    # Lapse-rate-based profile: T = T_surface * sigma^(R*gamma/g)
    temp_profile = atm_temp_k * (sigma_mid ** 0.19)  # ~6.5 K/km lapse rate
    temperature = jnp.broadcast_to(
        jnp.array(temp_profile)[:, jnp.newaxis], (nlev, ncols)
    )

    # Humidity decreasing with height
    q_profile = specific_humidity * sigma_mid ** 3
    specific_hum = jnp.broadcast_to(
        jnp.array(q_profile)[:, jnp.newaxis], (nlev, ncols)
    )

    # Geopotential from hydrostatic balance (approximate)
    from jcm.constants import physical_constants as pc
    height_profile = -pc.rd * atm_temp_k / pc.grav * np.log(sigma_mid)
    geopotential = jnp.broadcast_to(
        jnp.array(height_profile * pc.grav)[:, jnp.newaxis], (nlev, ncols)
    )

    tracers = {
        'qc': jnp.zeros((nlev, ncols)),
        'qi': jnp.zeros((nlev, ncols)),
    }
    state = PhysicsState(
        temperature=temperature,
        specific_humidity=specific_hum,
        u_wind=jnp.ones((nlev, ncols)) * wind_speed,
        v_wind=jnp.zeros((nlev, ncols)),
        geopotential=geopotential,
        normalized_surface_pressure=jnp.ones(ncols),
        tracers=tracers,
    )

    date = DateData.zeros()
    terrain_obj = TerrainData.aquaplanet(coords)
    if fmask > 0:
        # Override fmask for land fraction tests
        terrain_obj = TerrainData.from_coords(
            coords, fmask=fmask * jnp.ones(coords.horizontal.nodal_shape)
        )

    nodal_shape = coords.horizontal.nodal_shape  # (nlon, nlat)
    physics_data = PhysicsData.zeros(
        (ncols,), nlev, echam_coords=echam_coords,
        model_step=date.model_step, dt_seconds=date.dt_seconds,
    )

    # Set surface temperature
    surface_data = physics_data.surface.copy(
        surface_temperature=jnp.ones(ncols) * surface_temp_k,
        roughness_length=jnp.ones(ncols) * 1e-3,
    )
    physics_data = physics_data.copy(surface=surface_data)

    parameters = Parameters.default()
    forcing = ForcingData.zeros(nodal_shape)

    # Prepare diagnostics (pressure, height, density)
    _, physics_data = _prepare_common_physics_state(
        state, physics_data, parameters, forcing, terrain_obj
    )

    return state, physics_data, parameters, forcing, terrain_obj


class TestVdiffProducesExchangeCoefficients(unittest.TestCase):
    """Test that vertical diffusion produces nonzero surface exchange coefficients."""

    def test_vdiff_produces_nonzero_surface_exchange_coefficients(self):
        state, physics_data, parameters, forcing, terrain = _make_echam_state(
            surface_temp_k=300.0, atm_temp_k=290.0, wind_speed=5.0
        )
        _, updated_data = apply_vertical_diffusion(
            state, physics_data, parameters, forcing, terrain
        )
        vd = updated_data.vertical_diffusion
        ncols = state.temperature.shape[1]

        # Shape checks
        self.assertEqual(vd.surface_exchange_heat.shape, (ncols, 3))
        self.assertEqual(vd.surface_exchange_moisture.shape, (ncols, 3))
        self.assertEqual(vd.surface_exchange_momentum.shape, (ncols, 3))

        # Not all zeros
        self.assertTrue(jnp.any(vd.surface_exchange_heat != 0),
                        "surface_exchange_heat should not be all zeros")
        self.assertTrue(jnp.any(vd.surface_exchange_momentum != 0),
                        "surface_exchange_momentum should not be all zeros")

    def test_surface_exchange_coefficients_reasonable_magnitudes(self):
        state, physics_data, parameters, forcing, terrain = _make_echam_state(
            surface_temp_k=300.0, atm_temp_k=295.0, wind_speed=5.0
        )
        _, updated_data = apply_vertical_diffusion(
            state, physics_data, parameters, forcing, terrain
        )
        vd = updated_data.vertical_diffusion

        # No NaN or Inf
        self.assertFalse(jnp.any(jnp.isnan(vd.surface_exchange_heat)),
                         "surface_exchange_heat contains NaN")
        self.assertFalse(jnp.any(jnp.isinf(vd.surface_exchange_heat)),
                         "surface_exchange_heat contains Inf")
        self.assertFalse(jnp.any(jnp.isnan(vd.surface_exchange_momentum)),
                         "surface_exchange_momentum contains NaN")

        # Physically reasonable range for Ch*|V| type coefficients
        # Typical values: O(1e-3) to O(1e-1) m/s
        max_heat = jnp.max(jnp.abs(vd.surface_exchange_heat))
        max_mom = jnp.max(jnp.abs(vd.surface_exchange_momentum))
        self.assertGreater(float(max_heat), 0,
                           "exchange_heat should have nonzero values")
        self.assertLess(float(max_heat), 10.0,
                        f"exchange_heat max {max_heat} seems unreasonably large")
        self.assertLess(float(max_mom), 10.0,
                        f"exchange_momentum max {max_mom} seems unreasonably large")


class TestExchangeCoefficientsCoupling(unittest.TestCase):
    """Test that exchange coefficients flow from vertical diffusion to surface physics."""

    def test_exchange_coefficients_flow_to_surface_physics(self):
        state, physics_data, parameters, forcing, terrain = _make_echam_state(
            surface_temp_k=305.0, atm_temp_k=285.0, wind_speed=5.0
        )

        # Run vertical diffusion to get exchange coefficients
        _, pd_after_vdiff = apply_vertical_diffusion(
            state, physics_data, parameters, forcing, terrain
        )

        # Run surface physics with the computed exchange coefficients
        _, pd_after_surface = apply_surface(
            state, pd_after_vdiff, parameters, forcing, terrain
        )

        # Surface fluxes should be nonzero when exchange coefficients are nonzero
        self.assertTrue(
            jnp.any(pd_after_surface.surface.sensible_heat_flux != 0),
            "Sensible heat flux should be nonzero with computed exchange coefficients"
        )
        self.assertTrue(
            jnp.any(pd_after_surface.surface.evaporation != 0),
            "Evaporation should be nonzero with computed exchange coefficients"
        )

    def test_vdiff_coefficients_stored_in_physics_data(self):
        state, physics_data, parameters, forcing, terrain = _make_echam_state(
            surface_temp_k=305.0, atm_temp_k=285.0, wind_speed=5.0
        )

        # Before vdiff, exchange coefficients are zero
        self.assertTrue(jnp.all(physics_data.vertical_diffusion.surface_exchange_heat == 0))

        # Run vertical diffusion
        _, pd_after_vdiff = apply_vertical_diffusion(
            state, physics_data, parameters, forcing, terrain
        )

        # After vdiff, exchange coefficients should be populated
        self.assertTrue(
            jnp.any(pd_after_vdiff.vertical_diffusion.surface_exchange_heat != 0),
            "Vdiff should populate surface_exchange_heat"
        )

        # After surface physics, ch/cm should be stored
        _, pd_after_surface = apply_surface(
            state, pd_after_vdiff, parameters, forcing, terrain
        )
        self.assertTrue(
            jnp.any(pd_after_surface.surface.ch != 0),
            "Surface ch should be stored after apply_surface"
        )


class TestSurfaceFluxesVaryWithStability(unittest.TestCase):
    """Test that surface fluxes respond to atmospheric stability."""

    def test_unstable_vs_stable_exchange_coefficients(self):
        # Unstable: warm surface, cool atmosphere
        state_u, pd_u, params, forcing, terrain = _make_echam_state(
            surface_temp_k=310.0, atm_temp_k=280.0, wind_speed=5.0
        )
        _, pd_u = apply_vertical_diffusion(
            state_u, pd_u, params, forcing, terrain
        )

        # Stable: cool surface, warm atmosphere
        state_s, pd_s, _, _, _ = _make_echam_state(
            surface_temp_k=270.0, atm_temp_k=290.0, wind_speed=5.0
        )
        _, pd_s = apply_vertical_diffusion(
            state_s, pd_s, params, forcing, terrain
        )

        # Exchange coefficients should be larger in unstable conditions
        # (enhanced turbulent mixing)
        mean_kh_unstable = jnp.mean(jnp.abs(pd_u.vertical_diffusion.surface_exchange_heat))
        mean_kh_stable = jnp.mean(jnp.abs(pd_s.vertical_diffusion.surface_exchange_heat))

        self.assertGreater(
            float(mean_kh_unstable), float(mean_kh_stable),
            f"Unstable exchange coeffs ({mean_kh_unstable:.6f}) should exceed "
            f"stable ({mean_kh_stable:.6f})"
        )

    def test_sensible_heat_flux_sign_with_stability(self):
        # Unstable: warm surface heats atmosphere
        state_u, pd_u, params, forcing, terrain = _make_echam_state(
            surface_temp_k=310.0, atm_temp_k=280.0, wind_speed=5.0
        )
        _, pd_u = apply_vertical_diffusion(state_u, pd_u, params, forcing, terrain)
        _, pd_u = apply_surface(state_u, pd_u, params, forcing, terrain)

        # Stable: cold surface cools atmosphere
        state_s, pd_s, _, _, _ = _make_echam_state(
            surface_temp_k=260.0, atm_temp_k=290.0, wind_speed=5.0
        )
        _, pd_s = apply_vertical_diffusion(state_s, pd_s, params, forcing, terrain)
        _, pd_s = apply_surface(state_s, pd_s, params, forcing, terrain)

        mean_shf_warm = float(jnp.mean(pd_u.surface.sensible_heat_flux))
        mean_shf_cold = float(jnp.mean(pd_s.surface.sensible_heat_flux))

        # Warm surface should produce positive (upward) sensible heat flux
        # Cold surface should produce negative (downward) sensible heat flux
        # or at least much less than the warm case
        self.assertGreater(
            mean_shf_warm, mean_shf_cold,
            f"Warm surface SHF ({mean_shf_warm:.2f}) should exceed "
            f"cold surface SHF ({mean_shf_cold:.2f})"
        )


class TestSurfaceFluxesVaryWithSurfaceType(unittest.TestCase):
    """Test that surface fluxes differ between ocean and land."""

    def test_ocean_vs_land_surface_fluxes(self):
        # Pure ocean (fmask=0)
        state_o, pd_o, params, forcing_o, terrain_o = _make_echam_state(
            surface_temp_k=300.0, atm_temp_k=290.0, wind_speed=5.0, fmask=0.0
        )
        _, pd_o = apply_vertical_diffusion(
            state_o, pd_o, params, forcing_o, terrain_o
        )
        _, pd_o = apply_surface(
            state_o, pd_o, params, forcing_o, terrain_o
        )

        # Pure land (fmask=1)
        state_l, pd_l, _, forcing_l, terrain_l = _make_echam_state(
            surface_temp_k=300.0, atm_temp_k=290.0, wind_speed=5.0, fmask=1.0
        )
        _, pd_l = apply_vertical_diffusion(
            state_l, pd_l, params, forcing_l, terrain_l
        )
        _, pd_l = apply_surface(
            state_l, pd_l, params, forcing_l, terrain_l
        )

        # Surface fluxes should differ due to different roughness lengths and
        # surface properties
        ocean_shf = pd_o.surface.sensible_heat_flux
        land_shf = pd_l.surface.sensible_heat_flux

        self.assertFalse(
            jnp.allclose(ocean_shf, land_shf, atol=1e-6),
            "Ocean and land sensible heat fluxes should differ"
        )


class TestApplyVerticalDiffusionOrchestrator(unittest.TestCase):
    """Test the apply_vertical_diffusion orchestrator function."""

    def test_no_nans_in_tendencies_and_exchange_coefficients(self):
        state, physics_data, parameters, forcing, terrain = _make_echam_state()
        tendencies, updated_data = apply_vertical_diffusion(
            state, physics_data, parameters, forcing, terrain
        )

        # Tendencies should be NaN-free
        self.assertFalse(jnp.any(jnp.isnan(tendencies.temperature)),
                         "Temperature tendency contains NaN")
        self.assertFalse(jnp.any(jnp.isnan(tendencies.specific_humidity)),
                         "Humidity tendency contains NaN")
        self.assertFalse(jnp.any(jnp.isnan(tendencies.u_wind)),
                         "U wind tendency contains NaN")
        self.assertFalse(jnp.any(jnp.isnan(tendencies.v_wind)),
                         "V wind tendency contains NaN")

        # Key diagnostics should be NaN-free
        vd = updated_data.vertical_diffusion
        self.assertFalse(jnp.any(jnp.isnan(vd.km)), "km contains NaN")
        self.assertFalse(jnp.any(jnp.isnan(vd.kh)), "kh contains NaN")
        self.assertFalse(jnp.any(jnp.isnan(vd.surface_exchange_heat)),
                         "surface_exchange_heat contains NaN")
        self.assertFalse(jnp.any(jnp.isnan(vd.surface_exchange_momentum)),
                         "surface_exchange_momentum contains NaN")
        self.assertFalse(jnp.any(jnp.isnan(vd.pbl_height)),
                         "PBL height contains NaN")

    def test_unstable_profile_mixes(self):
        # Strong surface heating should produce mixing
        state, physics_data, parameters, forcing, terrain = _make_echam_state(
            surface_temp_k=320.0, atm_temp_k=280.0, wind_speed=5.0
        )
        tendencies, updated_data = apply_vertical_diffusion(
            state, physics_data, parameters, forcing, terrain
        )

        # PBL height should be nontrivial
        mean_pbl = float(jnp.mean(updated_data.vertical_diffusion.pbl_height))
        self.assertGreater(mean_pbl, 50.0,
                           f"PBL height ({mean_pbl:.0f}m) too low for unstable conditions")

    def test_stable_profile_weaker_exchange_coefficients(self):
        # Cold surface, warm atmosphere - should have weaker exchange
        state_s, pd_s, params, forcing, terrain = _make_echam_state(
            surface_temp_k=260.0, atm_temp_k=300.0, wind_speed=5.0
        )
        _, pd_s = apply_vertical_diffusion(
            state_s, pd_s, params, forcing, terrain
        )

        # Unstable case for comparison
        state_u, pd_u, _, _, _ = _make_echam_state(
            surface_temp_k=320.0, atm_temp_k=280.0, wind_speed=5.0
        )
        _, pd_u = apply_vertical_diffusion(
            state_u, pd_u, params, forcing, terrain
        )

        mean_kh_stable = float(jnp.mean(jnp.abs(
            pd_s.vertical_diffusion.surface_exchange_heat)))
        mean_kh_unstable = float(jnp.mean(jnp.abs(
            pd_u.vertical_diffusion.surface_exchange_heat)))

        self.assertLess(
            mean_kh_stable, mean_kh_unstable,
            f"Stable exchange coeff ({mean_kh_stable:.6f}) should be smaller than "
            f"unstable ({mean_kh_unstable:.6f})"
        )

    def test_tke_stays_positive(self):
        state, physics_data, parameters, forcing, terrain = _make_echam_state()
        _, updated_data = apply_vertical_diffusion(
            state, physics_data, parameters, forcing, terrain
        )
        self.assertTrue(
            jnp.all(updated_data.vertical_diffusion.tke >= 0),
            "TKE should remain non-negative"
        )


class TestApplySurfaceOrchestrator(unittest.TestCase):
    """Test the apply_surface orchestrator function."""

    def test_no_nans_in_output(self):
        state, physics_data, parameters, forcing, terrain = _make_echam_state()
        # Need vdiff first to populate exchange coefficients
        _, pd = apply_vertical_diffusion(
            state, physics_data, parameters, forcing, terrain
        )
        tendencies, updated_data = apply_surface(
            state, pd, parameters, forcing, terrain
        )

        self.assertFalse(jnp.any(jnp.isnan(tendencies.temperature)),
                         "Temperature tendency contains NaN")
        self.assertFalse(jnp.any(jnp.isnan(tendencies.specific_humidity)),
                         "Humidity tendency contains NaN")
        self.assertFalse(jnp.any(jnp.isnan(updated_data.surface.sensible_heat_flux)),
                         "Sensible heat flux contains NaN")
        self.assertFalse(jnp.any(jnp.isnan(updated_data.surface.latent_heat_flux)),
                         "Latent heat flux contains NaN")

    def test_warm_ocean_heats_atmosphere(self):
        state, physics_data, parameters, forcing, terrain = _make_echam_state(
            surface_temp_k=305.0, atm_temp_k=285.0, wind_speed=5.0
        )
        _, pd = apply_vertical_diffusion(
            state, physics_data, parameters, forcing, terrain
        )
        tendencies, updated_data = apply_surface(
            state, pd, parameters, forcing, terrain
        )

        # Warm ocean should heat the atmosphere (positive SHF)
        mean_shf = float(jnp.mean(updated_data.surface.sensible_heat_flux))
        self.assertGreater(mean_shf, 0,
                           f"SHF ({mean_shf:.2f} W/m²) should be positive over warm ocean")

        # Temperature tendency at lowest level should be positive
        mean_t_tend_sfc = float(jnp.mean(tendencies.temperature[-1, :]))
        self.assertGreater(mean_t_tend_sfc, 0,
                           f"Surface T tendency ({mean_t_tend_sfc:.2e}) should be positive")

    def test_cold_surface_cools_atmosphere(self):
        state, physics_data, parameters, forcing, terrain = _make_echam_state(
            surface_temp_k=260.0, atm_temp_k=290.0, wind_speed=5.0
        )
        _, pd = apply_vertical_diffusion(
            state, physics_data, parameters, forcing, terrain
        )
        tendencies, updated_data = apply_surface(
            state, pd, parameters, forcing, terrain
        )

        # Cold surface should cool the atmosphere (negative SHF)
        mean_shf = float(jnp.mean(updated_data.surface.sensible_heat_flux))
        self.assertLess(mean_shf, 0,
                        f"SHF ({mean_shf:.2f} W/m²) should be negative over cold surface")

    def test_surface_flux_magnitudes_reasonable(self):
        state, physics_data, parameters, forcing, terrain = _make_echam_state(
            surface_temp_k=300.0, atm_temp_k=290.0, wind_speed=5.0
        )
        _, pd = apply_vertical_diffusion(
            state, physics_data, parameters, forcing, terrain
        )
        _, updated_data = apply_surface(
            state, pd, parameters, forcing, terrain
        )

        max_shf = float(jnp.max(jnp.abs(updated_data.surface.sensible_heat_flux)))
        max_lhf = float(jnp.max(jnp.abs(updated_data.surface.latent_heat_flux)))
        max_tau = float(jnp.max(jnp.abs(updated_data.surface.momentum_flux_u)))

        self.assertLess(max_shf, 1000.0,
                        f"|SHF| max ({max_shf:.1f} W/m²) unreasonably large")
        self.assertLess(max_lhf, 2000.0,
                        f"|LHF| max ({max_lhf:.1f} W/m²) unreasonably large")
        self.assertLess(max_tau, 10.0,
                        f"|tau_u| max ({max_tau:.2f} N/m²) unreasonably large")


class TestFastIntegration(unittest.TestCase):
    """Fast integration tests for ECHAM physics (< 60s)."""

    def test_echam_physics_2_timesteps_physical_reasonableness(self):
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics

        sigma_boundaries = np.linspace(0, 1, 41)
        coords = get_coords(sigma_boundaries, spectral_truncation=31)
        terrain = TerrainData.aquaplanet(coords)

        model = Model(
            coords=coords,
            time_step=30,
            terrain=terrain,
            physics=echam_physics(),
        )

        # Run for 1 hour (2 timesteps)
        predictions = model.run(save_interval=1/24., total_time=1/24.)
        dynamics = predictions.dynamics

        # No NaN/Inf
        self.assertFalse(jnp.any(jnp.isnan(dynamics.temperature)),
                         "Temperature contains NaN")
        self.assertFalse(jnp.any(jnp.isnan(dynamics.u_wind)),
                         "U wind contains NaN")
        self.assertFalse(jnp.any(jnp.isnan(dynamics.specific_humidity)),
                         "Specific humidity contains NaN")

        # Physically reasonable ranges
        temp_min = float(jnp.min(dynamics.temperature))
        temp_max = float(jnp.max(dynamics.temperature))
        self.assertGreater(temp_min, 150.0,
                           f"Min temperature {temp_min:.1f}K too cold")
        self.assertLess(temp_max, 350.0,
                        f"Max temperature {temp_max:.1f}K too hot")

        # Surface pressure reasonable
        sp_min = float(jnp.min(dynamics.normalized_surface_pressure))
        sp_max = float(jnp.max(dynamics.normalized_surface_pressure))
        self.assertGreater(sp_min, 0.5, f"Min surface pressure {sp_min:.3f} too low")
        self.assertLess(sp_max, 1.5, f"Max surface pressure {sp_max:.3f} too high")

    def test_echam_physics_exchange_coefficients_active_in_integration(self):
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.physics_interface import PhysicsState
        from jcm.date import DateData
        from jcm.forcing import ForcingData

        sigma_boundaries = np.linspace(0, 1, 41)
        coords = get_coords(sigma_boundaries, spectral_truncation=31)
        terrain = TerrainData.aquaplanet(coords)
        nlev, nlon, nlat = coords.nodal_shape
        shape_3d = (nlev, nlon, nlat)

        physics = echam_physics()
        physics.cache_coords(coords)

        state = PhysicsState(
            u_wind=jnp.ones(shape_3d) * 5.0,
            v_wind=jnp.zeros(shape_3d),
            temperature=jnp.ones(shape_3d) * 280.0,
            specific_humidity=jnp.ones(shape_3d) * 0.005,
            geopotential=jnp.zeros(shape_3d),
            normalized_surface_pressure=jnp.ones((nlon, nlat)),
            tracers={},
        )
        forcing = ForcingData.zeros((nlon, nlat))
        _, physics_data = physics.compute_tendencies(
            state, forcing, terrain, DateData.zeros(),
        )

        # Composable ECHAM stores per-term state under '_<category>' keys.
        vd = physics_data["_vertical_diffusion"]
        self.assertTrue(
            jnp.any(vd.surface_exchange_heat != 0),
            "surface_exchange_heat should be nonzero after coupling"
        )

        surface = physics_data["_surface"]
        self.assertFalse(jnp.all(surface.sensible_heat_flux == 0),
                         "Sensible heat flux should not be all zero")


if __name__ == '__main__':
    unittest.main()
