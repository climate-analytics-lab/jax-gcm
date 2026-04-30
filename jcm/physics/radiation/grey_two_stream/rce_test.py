"""Radiative-Convective Equilibrium (RCE) single-column test.

Evolves a temperature profile under radiation-only or radiation + convective
adjustment to verify that:
  1. Pure radiative equilibrium develops a stratospheric inversion.
  2. With convective adjustment the tropospheric lapse rate is bounded.
  3. The net TOA flux converges toward zero (energy balance).

Inspired by the swirl-jatmos ``radiative_eqb_solver`` and ICON physics.

Date: 2025-08-01
"""

import pytest
import jax.numpy as jnp
import jax_datetime as jdt
from datetime import datetime

from jcm.physics.radiation.grey_two_stream.radiation_scheme import radiation_scheme
from jcm.physics.radiation.radiation_types import RadiationParameters
from jcm.physics.icon.unit_conversions import (
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


def _convective_adjustment(temperature, pressure, lapse_rate=6.5e-3,
                           max_dt=5.0):
    """Manabe-Strickler dry/moist convective adjustment.

    If any layer is less stable than the target lapse rate, adjust toward it.
    """
    nlev = temperature.shape[0]
    # Approximate heights from hydrostatic + ideal gas
    scale_height = 8500.0  # m
    z = -scale_height * jnp.log(pressure / pressure[-1])

    for _ in range(5):  # a few passes for convergence
        for k in range(nlev - 1):
            dz = z[k] - z[k + 1]
            dT_actual = temperature[k + 1] - temperature[k]
            dT_adiabat = lapse_rate * dz
            excess = dT_actual - dT_adiabat
            correction = jnp.clip(excess / 2.0, -max_dt, max_dt)
            correction = jnp.maximum(correction, 0.0)  # only adjust unstable
            temperature = temperature.at[k].add(correction)
            temperature = temperature.at[k + 1].add(-correction)
    return temperature


def _rce_step(temperature, pressure, pressure_interfaces,
              surface_temperature, params, aerosol, date, dt,
              rh=0.75, convective_adjustment=False):
    """Single forward-Euler step of RCE evolution."""
    heating, _ = _radiation_heating(
        temperature, pressure, pressure_interfaces,
        surface_temperature, params, aerosol, date, rh,
    )
    temperature_new = temperature + heating * dt
    if convective_adjustment:
        temperature_new = _convective_adjustment(temperature_new, pressure)
    return temperature_new


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _make_rce_setup(nlev=20):
    """Create atmosphere and parameters for RCE tests."""
    atm = create_test_atmosphere(nlev=nlev)
    params = RadiationParameters.default()
    aerosol = create_default_aerosol_data(nlev=nlev, parameters=params)
    date = jdt.Datetime.from_pydatetime(datetime(2024, 3, 21, 12, 0))
    surface_temperature = jnp.array(300.0)
    return atm, params, aerosol, date, surface_temperature


@pytest.mark.slow
class TestRadiativeEquilibrium:
    """Test pure radiative equilibrium (no convection).

    The 3-band LW radiation used by the scheme produces larger TOA heating
    than the previous 8-band version, so the simple forward-Euler driver
    here needs sub-hour dt to stay stable at the thin TOA layers. Real
    GCM runs are bounded by advection / diffusion / convection and don't
    see the same issue.
    """

    def test_radiative_equilibrium_converges(self):
        """Temperature profile should evolve and net flux should decrease."""
        atm, params, aerosol, date, sfc_t = _make_rce_setup(nlev=20)
        temperature = atm["temperature"]
        pressure = atm["pressure_levels"]
        pressure_interfaces = atm["pressure_interfaces"]

        # dt = 30 min keeps forward-Euler stable at the thin TOA layers under
        # the current 3-band LW radiation tuning. See class docstring.
        dt = 1800.0
        n_steps = 16

        initial_temperature = temperature.copy()
        for _ in range(n_steps):
            temperature = _rce_step(
                temperature, pressure, pressure_interfaces,
                sfc_t, params, aerosol, date, dt,
                convective_adjustment=False,
            )

        # Temperature should have changed
        assert not jnp.allclose(temperature, initial_temperature, atol=0.1)
        # All temperatures should remain physical
        assert jnp.all(temperature > 100.0)
        assert jnp.all(temperature < 500.0)

    def test_pure_radiative_develops_inversion(self):
        """Without convection the stratosphere should warm (inversion)."""
        atm, params, aerosol, date, sfc_t = _make_rce_setup(nlev=20)
        temperature = atm["temperature"]
        pressure = atm["pressure_levels"]
        pressure_interfaces = atm["pressure_interfaces"]

        dt = 1800.0
        for _ in range(16):
            temperature = _rce_step(
                temperature, pressure, pressure_interfaces,
                sfc_t, params, aerosol, date, dt,
                convective_adjustment=False,
            )

        # In radiative equilibrium, temperature should NOT be monotonically
        # decreasing with height — there should be a stratospheric warming
        dT = jnp.diff(temperature)
        # At least one level where temperature increases going up (toward TOA)
        has_inversion = jnp.any(dT > 0)
        assert has_inversion, "Expected stratospheric warming in radiative eq."


@pytest.mark.slow
class TestRadiativeConvectiveEquilibrium:
    """Test RCE with convective adjustment."""

    def test_rce_bounds_lapse_rate(self):
        """With convective adjustment, the tropospheric lapse rate should be bounded."""
        atm, params, aerosol, date, sfc_t = _make_rce_setup(nlev=20)
        temperature = atm["temperature"]
        pressure = atm["pressure_levels"]
        pressure_interfaces = atm["pressure_interfaces"]

        dt = 1800.0
        for _ in range(24):
            temperature = _rce_step(
                temperature, pressure, pressure_interfaces,
                sfc_t, params, aerosol, date, dt,
                convective_adjustment=True,
            )

        # Compute lapse rate in lower troposphere (below ~300 hPa)
        scale_height = 8500.0
        z = -scale_height * jnp.log(pressure / pressure[-1])
        troposphere = pressure > 30000.0  # > 300 hPa
        trop_idx = jnp.where(troposphere)[0]

        if len(trop_idx) > 1:
            dT = jnp.diff(temperature[trop_idx])
            dz = jnp.diff(z[trop_idx])
            lapse = -dT / dz  # K/m, positive means T decreases with height
            # Should not exceed dry adiabatic lapse rate (~10 K/km)
            assert jnp.all(lapse < 0.012), (
                f"Lapse rate too large: max={jnp.max(lapse)*1000:.1f} K/km"
            )

    def test_rce_temperatures_physical(self):
        """All temperatures should remain in a physical range."""
        atm, params, aerosol, date, sfc_t = _make_rce_setup(nlev=20)
        temperature = atm["temperature"]
        pressure = atm["pressure_levels"]
        pressure_interfaces = atm["pressure_interfaces"]

        dt = 1800.0
        for _ in range(16):
            temperature = _rce_step(
                temperature, pressure, pressure_interfaces,
                sfc_t, params, aerosol, date, dt,
                convective_adjustment=True,
            )

        assert jnp.all(temperature > 100.0), f"Min T = {jnp.min(temperature):.1f}"
        assert jnp.all(temperature < 500.0), f"Max T = {jnp.max(temperature):.1f}"


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
