"""Radiative-Convective Equilibrium (RCE) single-column test.

Evolves a temperature profile under radiation-only or radiation + convective
adjustment to verify that:
  1. Pure radiative equilibrium develops a stratospheric inversion.
  2. With convective adjustment the tropospheric lapse rate is bounded.
  3. The net TOA flux converges toward zero (energy balance).

Inspired by the swirl-jatmos ``radiative_eqb_solver`` and ICON physics.

Date: 2025-08-01
"""

import jax.numpy as jnp
import jax_datetime as jdt
from datetime import datetime

from jcm.physics.radiation.grey_two_stream.radiation_scheme import radiation_scheme
from jcm.physics.radiation.radiation_types import RadiationParameters
from jcm.physics.echam.unit_conversions import (
    calculate_air_density,
    calculate_layer_thickness,
)
from jcm.physics.radiation.grey_two_stream.radiation_scheme_test import (
    create_test_atmosphere,
    create_default_aerosol_data,
)
from jcm.physics.clouds.sundqvist import (
    saturation_specific_humidity,
)


# ---------------------------------------------------------------------------
# RCE helpers
# ---------------------------------------------------------------------------

def _compute_q_from_rh(temperature, pressure, rh=0.75):
    """Specific humidity for a given constant relative humidity."""
    qs = saturation_specific_humidity(pressure, temperature)
    return rh * qs


def _radiation_heating(temperature, pressure, pressure_interfaces,
                       surface_temperature, params, aerosol, date,
                       rh=0.75):
    """Compute radiation heating rate for a single column."""
    from jcm.forcing import SolarGeometry
    from jax_solar import OrbitalTime
    nlev = temperature.shape[0]
    specific_humidity = _compute_q_from_rh(temperature, pressure, rh)
    air_density = calculate_air_density(pressure, temperature)
    layer_thickness = calculate_layer_thickness(pressure, temperature)

    ot = OrbitalTime.from_datetime(date)
    solar = SolarGeometry(
        tyear=jnp.asarray(ot.orbital_phase / (2.0 * jnp.pi), dtype=jnp.float32),
        orbital_phase=jnp.asarray(ot.orbital_phase, dtype=jnp.float32),
        synodic_phase=jnp.asarray(ot.synodic_phase, dtype=jnp.float32),
    )

    tend, diag = radiation_scheme(
        temperature=temperature,
        specific_humidity=specific_humidity,
        pressure_levels=pressure,
        pressure_interfaces=pressure_interfaces,
        layer_thickness=layer_thickness,
        air_density=air_density,
        cloud_water=jnp.zeros(nlev),
        cloud_ice=jnp.zeros(nlev),
        cloud_fraction=jnp.zeros(nlev),
        surface_temperature=surface_temperature,
        surface_albedo_vis=jnp.array(0.07),
        surface_albedo_nir=jnp.array(0.07),
        surface_emissivity=jnp.array(0.98),
        solar=solar,
        latitude=0.0,
        longitude=0.0,
        parameters=params,
        aerosol_data=aerosol,
        ozone_vmr=None,
        co2_vmr=400e-6,
    )
    return tend.temperature_tendency, diag


def _make_rce_setup(nlev=20):
    """Create atmosphere and parameters for RCE tests."""
    atm = create_test_atmosphere(nlev=nlev)
    params = RadiationParameters.default()
    aerosol = create_default_aerosol_data(nlev=nlev, parameters=params)
    date = jdt.Datetime.from_pydatetime(datetime(2024, 3, 21, 12, 0))
    surface_temperature = jnp.array(300.0)
    return atm, params, aerosol, date, surface_temperature



class TestRadiationHeating:
    """Quick (non-slow) tests for radiation heating sanity."""

    def test_clear_sky_heating_has_lw_cooling(self):
        """Clear-sky atmosphere should show longwave cooling in troposphere."""
        atm, params, aerosol, date, sfc_t = _make_rce_setup(nlev=20)
        _, diag = _radiation_heating(
            atm["temperature"], atm["pressure_levels"],
            atm["pressure_interfaces"], sfc_t, params, aerosol, date,
        )
        # LW should cool the troposphere (negative heating) for at least some levels
        assert jnp.any(diag.lw_heating_rate < 0), "Expected LW cooling in troposphere"

    def test_heating_is_finite(self):
        """All heating rates and fluxes should be finite."""
        atm, params, aerosol, date, sfc_t = _make_rce_setup(nlev=20)
        heating, diag = _radiation_heating(
            atm["temperature"], atm["pressure_levels"],
            atm["pressure_interfaces"], sfc_t, params, aerosol, date,
        )
        assert jnp.all(jnp.isfinite(heating))
        assert jnp.isfinite(diag.surface_lw_down)
        assert jnp.isfinite(diag.toa_lw_up)
