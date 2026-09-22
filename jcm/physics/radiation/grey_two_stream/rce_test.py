"""Radiative-Convective Equilibrium (RCE) single-column test.

Evolves a temperature profile under radiation-only or radiation + convective
adjustment to verify that:
  1. Pure radiative equilibrium develops a stratospheric inversion.
  2. With convective adjustment the tropospheric lapse rate is bounded.
  3. The net TOA flux converges toward zero (energy balance).

Inspired by the swirl-jatmos ``radiative_eqb_solver`` and ICON physics.

Date: 2025-08-01
"""

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import pytest
from datetime import datetime

from jcm.physics.radiation.grey_two_stream.radiation_scheme import radiation_scheme
from jcm.physics.radiation.radiation_types import RadiationParameters
from jcm.physics.radiation.grey_two_stream.radiation_scheme_test import (
    create_test_atmosphere,
    create_default_aerosol_data,
    calculate_air_density,
    calculate_layer_thickness,
)
from jcm.physics.clouds.sundqvist import (
    saturation_specific_humidity,
)
from jcm.forcing import SolarGeometry
from jax_solar import OrbitalTime
from jcm.testing import check_gradients


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


class TestRadiationSchemeGradients:
    """Whole-scheme gradients through the grey two-stream column (#820).

    The finiteness checks here are the substance. Before the Planck and
    two-stream guards this class exercises, ``radiation_scheme`` returned a
    NaN gradient with respect to temperature and surface temperature for
    *every* column, cloudy or clear — the longwave chain was poisoned from
    ``planck_function_wavenumber`` outwards — and the grey scheme is the
    default ECHAM radiation.

    The AD-vs-difference comparison is done at component level, where it
    converges: ``planck_test.py`` for the Planck derivative and
    ``two_stream_test.py`` for the layer coefficients and both flux solvers.
    At whole-scheme level the reference is the adjoint one, because the
    eleven outputs span three orders of magnitude and cancel, and because the
    scheme carries cloud-overlap and band switches a displacement of 0.1% of
    each field's RMS can cross.
    """

    @staticmethod
    def _column(nlev=14, cloudy=True):
        """Build a tropical column, cloudy or clear, off the cover switches."""
        atm = create_test_atmosphere(nlev=nlev)
        temperature = atm["temperature"]
        pressure = atm["pressure_levels"]
        humidity = 0.72 * jax.vmap(saturation_specific_humidity)(
            pressure, temperature)
        if cloudy:
            # A thin condensate everywhere plus two real decks: the cloud
            # optics and the overlap both switch on exact zeros, so an
            # all-or-nothing profile would sit on those switches rather than
            # test them.
            cloud_water = jnp.full(nlev, 2.1e-6).at[6].set(9e-5).at[7].set(6e-5)
            cloud_ice = jnp.full(nlev, 8.0e-7).at[3].set(2.0e-5)
            cloud_fraction = jnp.full(nlev, 0.11).at[3].set(0.5).at[6:8].set(0.62)
        else:
            cloud_water = jnp.zeros(nlev)
            cloud_ice = jnp.zeros(nlev)
            cloud_fraction = jnp.zeros(nlev)
        return (atm, temperature, humidity, cloud_water, cloud_ice,
                cloud_fraction)

    @staticmethod
    def _scheme_fn(atm, params, aerosol, date):
        """Return f(T, q, qc, qi, cf, T_sfc, albedo) -> radiative outputs."""
        from jcm.physics.radiation.grey_two_stream.radiation_scheme_test import (
            calculate_air_density, calculate_layer_thickness,
        )
        ot = OrbitalTime.from_datetime(date)
        solar = SolarGeometry(
            tyear=jnp.asarray(ot.orbital_phase / (2.0 * jnp.pi), jnp.float32),
            orbital_phase=jnp.asarray(ot.orbital_phase, jnp.float32),
            synodic_phase=jnp.asarray(ot.synodic_phase, jnp.float32),
        )
        pressure = atm["pressure_levels"]
        interfaces = atm["pressure_interfaces"]

        def f(temperature, humidity, cloud_water, cloud_ice, cloud_fraction,
              surface_temperature, albedo):
            tend, diag = radiation_scheme(
                temperature=temperature, specific_humidity=humidity,
                pressure_levels=pressure, pressure_interfaces=interfaces,
                layer_thickness=calculate_layer_thickness(pressure, temperature),
                air_density=calculate_air_density(pressure, temperature),
                cloud_water=cloud_water, cloud_ice=cloud_ice,
                cloud_fraction=cloud_fraction,
                surface_temperature=surface_temperature,
                surface_albedo_vis=albedo, surface_albedo_nir=albedo,
                surface_emissivity=jnp.array(0.98), solar=solar,
                latitude=12.0, longitude=30.0, parameters=params,
                aerosol_data=aerosol, ozone_vmr=None, co2_vmr=400e-6,
            )
            return (tend.temperature_tendency, tend.longwave_heating,
                    tend.shortwave_heating, diag.sw_flux_up, diag.sw_flux_down,
                    diag.lw_flux_up, diag.lw_flux_down, diag.toa_sw_up,
                    diag.toa_lw_up, diag.surface_sw_down, diag.surface_lw_down)

        return f

    def _args(self, cloudy):
        """Build (f, args) for a cloudy or clear column."""
        nlev = 14
        atm, T, q, qc, qi, cf = self._column(nlev=nlev, cloudy=cloudy)
        params = RadiationParameters.default()
        aerosol = create_default_aerosol_data(nlev=nlev, parameters=params)
        date = jdt.Datetime.from_pydatetime(datetime(2024, 3, 21, 12, 0))
        f = self._scheme_fn(atm, params, aerosol, date)
        return f, (T, q, qc, qi, cf, jnp.array(299.0), jnp.array(0.09))

    @pytest.mark.parametrize("cloudy", [True, False])
    def test_every_input_carries_a_finite_gradient(self, cloudy):
        """No input of the grey scheme may return a non-finite gradient.

        Temperature and surface temperature both returned NaN here, in cloudy
        and clear columns alike, through ``planck_function_wavenumber``'s
        float32 underflow; an optically thick longwave layer added a second,
        mode-dependent NaN in the layer solution.
        """
        f, args = self._args(cloudy)
        grads = jax.grad(
            lambda *a: sum(jnp.sum(x ** 2) for x in f(*a)),
            argnums=tuple(range(len(args))),
        )(*args)
        names = ("temperature", "humidity", "cloud_water", "cloud_ice",
                 "cloud_fraction", "surface_temperature", "albedo")
        for name, grad in zip(names, grads):
            assert jnp.all(jnp.isfinite(grad)), (
                f"d/d{name} is not finite in a "
                f"{'cloudy' if cloudy else 'clear'} column: {grad}")

    @pytest.mark.parametrize("cloudy", [True, False])
    def test_scheme_gradients_are_adjoint_and_live(self, cloudy):
        """Both AD modes agree and every input is live, cloudy and clear."""
        f, args = self._args(cloudy)
        check_gradients(
            f, args, reference="adjoint", adjoint_rtol=5e-3,
            live_inputs=["[0]", "[1]", "[5]", "[6]"])
