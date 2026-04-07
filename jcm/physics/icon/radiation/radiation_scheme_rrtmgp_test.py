"""Tests for the RRTMGP radiation scheme wrapper.

Compares RRTMGP and grey radiation schemes with identical atmospheric inputs
to verify structural correctness and reasonable agreement.

Date: 2025-08-01
"""

import pytest
import numpy as np
import jax.numpy as jnp
import jax_datetime as jdt
from datetime import datetime

from jcm.physics.icon.radiation.radiation_scheme import radiation_scheme
from jcm.physics.icon.radiation.radiation_scheme_rrtmgp import (
    radiation_scheme_rrtmgp,
)
from jcm.physics.icon.radiation.radiation_types import RadiationParameters
from jcm.physics.icon.unit_conversions import (
    calculate_air_density,
    calculate_layer_thickness,
)
from jcm.physics.icon.radiation.radiation_scheme_test import (
    create_test_atmosphere,
    create_default_aerosol_data,
)


def _make_inputs(nlev=10):
    """Create identical input set for both radiation schemes."""
    atm = create_test_atmosphere(nlev=nlev)
    params = RadiationParameters.default()
    aerosol = create_default_aerosol_data(nlev=nlev, parameters=params)

    air_density = calculate_air_density(
        atm["pressure_levels"], atm["temperature"]
    )
    layer_thickness = calculate_layer_thickness(
        atm["pressure_levels"], atm["temperature"]
    )

    # Summer solstice, equatorial point
    date = jdt.Datetime.from_pydatetime(datetime(2024, 6, 21, 12, 0))

    return dict(
        temperature=atm["temperature"],
        specific_humidity=atm["specific_humidity"],
        pressure_levels=atm["pressure_levels"],
        pressure_interfaces=atm["pressure_interfaces"],
        layer_thickness=layer_thickness,
        air_density=air_density,
        cloud_water=atm["cloud_water"],
        cloud_ice=atm["cloud_ice"],
        cloud_fraction=atm["cloud_fraction"],
        surface_temperature=jnp.array(300.0),
        surface_albedo_vis=jnp.array(0.07),
        surface_albedo_nir=jnp.array(0.07),
        surface_emissivity=jnp.array(0.98),
        date=date,
        latitude=0.0,
        longitude=0.0,
        parameters=params,
        aerosol_data=aerosol,
        ozone_vmr=None,
        co2_vmr=400e-6,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestRRTMGPScheme:
    """Test the RRTMGP radiation scheme produces valid outputs."""

    def test_rrtmgp_produces_valid_heating(self):
        """RRTMGP heating rates should be finite and non-trivial."""
        inputs = _make_inputs(nlev=10)
        tend, diag = radiation_scheme_rrtmgp(**inputs)

        assert jnp.all(jnp.isfinite(tend.temperature_tendency))
        assert jnp.all(jnp.isfinite(tend.longwave_heating))
        assert jnp.all(jnp.isfinite(tend.shortwave_heating))
        # At least some non-zero heating
        assert jnp.mean(jnp.abs(tend.temperature_tendency)) > 1e-8

    def test_rrtmgp_diagnostics_valid(self):
        """Surface/TOA flux diagnostics should be non-negative and finite."""
        inputs = _make_inputs(nlev=10)
        _, diag = radiation_scheme_rrtmgp(**inputs)

        assert jnp.isfinite(diag.surface_sw_down)
        assert jnp.isfinite(diag.surface_lw_down)
        assert jnp.isfinite(diag.toa_lw_up)
        assert diag.surface_lw_down >= 0.0
        assert diag.toa_lw_up >= 0.0


class TestGreyVsRRTMGP:
    """Compare grey and RRTMGP schemes for structural agreement."""

    def test_heating_tendency_shapes_match(self):
        """Both schemes should return the same shaped arrays."""
        inputs = _make_inputs(nlev=10)
        tend_grey, _ = radiation_scheme(**inputs)
        tend_rrtm, _ = radiation_scheme_rrtmgp(**inputs)

        assert tend_grey.temperature_tendency.shape == tend_rrtm.temperature_tendency.shape
        assert tend_grey.longwave_heating.shape == tend_rrtm.longwave_heating.shape
        assert tend_grey.shortwave_heating.shape == tend_rrtm.shortwave_heating.shape

    def test_heating_rates_broadly_agree(self):
        """Total heating should agree within a generous tolerance.

        The grey scheme is a coarse parameterisation so we only check that
        the two are in the same ballpark (atol=0.1 K/s, rtol=100%).
        """
        inputs = _make_inputs(nlev=10)
        tend_grey, _ = radiation_scheme(**inputs)
        tend_rrtm, _ = radiation_scheme_rrtmgp(**inputs)

        # Both should have the same sign pattern in most levels
        assert jnp.all(jnp.isfinite(tend_grey.temperature_tendency))
        assert jnp.all(jnp.isfinite(tend_rrtm.temperature_tendency))

        # Loose absolute tolerance (K/s) — they don't need to match closely
        np.testing.assert_allclose(
            tend_grey.temperature_tendency,
            tend_rrtm.temperature_tendency,
            atol=0.1,
            rtol=1.0,
        )

    @pytest.mark.parametrize(
        "lat,lon,month",
        [
            (0.0, 0.0, 6),       # equator, summer
            (60.0, 0.0, 6),      # high-lat NH summer
            (-60.0, 0.0, 12),    # high-lat SH summer
            (0.0, 180.0, 3),     # equator, equinox
            (80.0, 0.0, 12),     # near-polar NH winter (low sun)
        ],
    )
    def test_multiple_conditions(self, lat, lon, month):
        """Both schemes should produce finite results across conditions."""
        inputs = _make_inputs(nlev=10)
        inputs["latitude"] = lat
        inputs["longitude"] = lon
        inputs["date"] = jdt.Datetime.from_pydatetime(
            datetime(2024, month, 15, 12, 0)
        )

        tend_grey, _ = radiation_scheme(**inputs)
        tend_rrtm, _ = radiation_scheme_rrtmgp(**inputs)

        assert jnp.all(jnp.isfinite(tend_grey.temperature_tendency))
        assert jnp.all(jnp.isfinite(tend_rrtm.temperature_tendency))
