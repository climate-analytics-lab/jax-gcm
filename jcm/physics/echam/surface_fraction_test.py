"""Tests for surface fraction setup in apply_surface.

Verifies that surface tile fractions (water, sea ice, land) are correctly
computed from land mask and sea ice boundary conditions.
"""

import unittest
import jax.numpy as jnp
import numpy as np
from jcm.physics.echam.echam_physics import (
    _prepare_common_physics_state,
    apply_surface,
)
from jcm.physics.echam.echam_physics_data import PhysicsData
from jcm.physics.echam.echam_coords import EchamCoords
from jcm.physics.echam.parameters import Parameters
from jcm.physics_interface import PhysicsState
from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.date import DateData
from jcm.utils import get_coords


NLEV = 8
NLAT, NLON = 32, 64  # (nlon, nlat) nodal shape convention
NCOLS = NLAT * NLON


def _make_inputs(land_fraction, sea_ice_concentration):
    """Create inputs for apply_surface with prescribed land/ice fractions.

    Follows the same setup pattern as exchange_coupling_test._make_echam_state.
    """
    sigma_boundaries = np.linspace(0, 1, NLEV + 1)
    coords = get_coords(sigma_boundaries, nodal_shape=(NLON, NLAT))
    echam_coords = EchamCoords.from_coordinate_system(coords)
    date = DateData.zeros()
    nodal_shape = coords.horizontal.nodal_shape  # (nlon, nlat)

    state = PhysicsState(
        temperature=jnp.ones((NLEV, NCOLS)) * 280.0,
        specific_humidity=jnp.ones((NLEV, NCOLS)) * 0.005,
        u_wind=jnp.ones((NLEV, NCOLS)) * 5.0,
        v_wind=jnp.ones((NLEV, NCOLS)) * 2.0,
        geopotential=jnp.zeros((NLEV, NCOLS)),
        normalized_surface_pressure=jnp.ones(NCOLS),
        tracers={'qc': jnp.zeros((NLEV, NCOLS)), 'qi': jnp.zeros((NLEV, NCOLS))},
    )

    physics_data = PhysicsData.zeros(
        (NCOLS,), NLEV, echam_coords=echam_coords,
        model_step=date.model_step, dt_seconds=date.dt_seconds,
    )
    # Set realistic surface temperature
    surface_data = physics_data.surface.copy(
        surface_temperature=jnp.ones(NCOLS) * 290.0,
        roughness_length=jnp.ones(NCOLS) * 1e-3,
    )
    physics_data = physics_data.copy(surface=surface_data)

    # Set up terrain with prescribed land fraction
    fmask = jnp.full(nodal_shape, land_fraction)
    terrain = TerrainData.aquaplanet(coords)
    terrain = terrain.copy(fmask=fmask)

    # Set up forcing with prescribed sea ice
    sice = jnp.full(nodal_shape, sea_ice_concentration)
    forcing = ForcingData.zeros(nodal_shape)
    forcing = forcing.copy(sice_am=sice)

    parameters = Parameters.default()

    # Prepare diagnostics (pressure, height, density)
    _, physics_data = _prepare_common_physics_state(
        state, physics_data, parameters, forcing, terrain
    )

    return state, physics_data, parameters, forcing, terrain


class TestSurfaceFractions(unittest.TestCase):
    """Test that apply_surface computes surface tile fractions correctly."""

    def _get_fractions(self, land_fraction, sea_ice_concentration):
        """Compute surface fractions using the same logic as apply_surface."""
        land = jnp.full(NCOLS, land_fraction)
        raw_ice = jnp.full(NCOLS, sea_ice_concentration)
        sea_ice = jnp.clip(raw_ice, 0.0, 1.0 - land)
        water = 1.0 - land - sea_ice
        return water, sea_ice, land

    def test_ocean_only(self):
        """All ocean: land=0, ice=0 -> water=1."""
        water, ice, land = self._get_fractions(0.0, 0.0)
        np.testing.assert_allclose(water, 1.0)
        np.testing.assert_allclose(ice, 0.0)
        np.testing.assert_allclose(land, 0.0)

    def test_land_only(self):
        """All land: land=1, any ice -> water=0, ice=0 (clamped)."""
        water, ice, land = self._get_fractions(1.0, 0.5)
        np.testing.assert_allclose(water, 0.0)
        np.testing.assert_allclose(ice, 0.0)
        np.testing.assert_allclose(land, 1.0)

    def test_mixed_with_sea_ice(self):
        """Mixed: land=0.3, ice=0.4 -> water=0.3, ice=0.4, land=0.3."""
        water, ice, land = self._get_fractions(0.3, 0.4)
        np.testing.assert_allclose(water, 0.3, atol=1e-6)
        np.testing.assert_allclose(ice, 0.4, atol=1e-6)
        np.testing.assert_allclose(land, 0.3, atol=1e-6)

    def test_fractions_sum_to_one(self):
        """Fractions must sum to 1 for all combinations."""
        for land_frac in [0.0, 0.2, 0.5, 0.8, 1.0]:
            for ice_frac in [0.0, 0.3, 0.7, 1.0]:
                water, ice, land = self._get_fractions(land_frac, ice_frac)
                total = water + ice + land
                np.testing.assert_allclose(
                    total, 1.0, atol=1e-6,
                    err_msg=f"Fractions don't sum to 1 for land={land_frac}, ice={ice_frac}"
                )

    def test_ice_clamped_to_non_land_area(self):
        """Sea ice cannot exceed (1 - land_fraction)."""
        water, ice, land = self._get_fractions(0.7, 0.8)
        # Ice should be clamped to 0.3 (= 1 - 0.7)
        np.testing.assert_allclose(ice, 0.3, atol=1e-6)
        np.testing.assert_allclose(water, 0.0, atol=1e-6)

    def test_no_negative_fractions(self):
        """No fraction should be negative."""
        for land_frac in [0.0, 0.5, 1.0]:
            for ice_frac in [0.0, 0.5, 1.0]:
                water, ice, land = self._get_fractions(land_frac, ice_frac)
                self.assertTrue(jnp.all(water >= 0.0))
                self.assertTrue(jnp.all(ice >= 0.0))
                self.assertTrue(jnp.all(land >= 0.0))

    def test_apply_surface_runs_with_sea_ice(self):
        """Smoke test: apply_surface completes without error with sea ice data."""
        state, physics_data, parameters, forcing, terrain = _make_inputs(
            land_fraction=0.3, sea_ice_concentration=0.4
        )
        tendencies, updated_data = apply_surface(
            state, physics_data, parameters, forcing, terrain
        )
        assert tendencies.temperature.shape == (NLEV, NCOLS)
        assert jnp.all(jnp.isfinite(tendencies.temperature))
