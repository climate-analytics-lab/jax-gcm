"""Emulated radiation scheme using bidirectional GRU neural networks.

Drop-in replacement for ``radiation_scheme_rrtmgp`` that uses trained neural
networks to predict shortwave and longwave fluxes. The NN weights are passed
as JAX arrays through the ``emulator_weights`` argument, making them
fully differentiable for gradient-based optimization.

Reference architecture: Ukkonen (2024), https://github.com/peterukk/rte-rrtmgp-nn

Date: 2026-04-11
"""

from typing import Tuple, Optional

import jax.numpy as jnp

from jcm.physics.radiation.radiation_types import (
    RadiationParameters,
    RadiationTendencies,
)
from jcm.physics.icon.icon_physics_data import RadiationData
from jcm.physics.radiation.nn_emulator import (
    EmulatorWeights,
    InputScaling,
    preprocess_sw_inputs,
    preprocess_lw_inputs,
    sw_emulator_column,
    lw_emulator_column,
    reconstruct_sw_fluxes,
    reconstruct_lw_fluxes,
    flux_to_heating_rate,
)
from jcm.constants import PhysicalConstants


def radiation_scheme_emulated(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure_levels: jnp.ndarray,
    pressure_interfaces: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    air_density: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    surface_albedo_vis: jnp.ndarray,
    surface_albedo_nir: jnp.ndarray,
    surface_emissivity: jnp.ndarray,
    date,
    latitude: float,
    longitude: float,
    parameters: RadiationParameters,
    aerosol_data,
    ozone_vmr: Optional[jnp.ndarray] = None,
    co2_vmr: float = 400e-6,
    emulator_weights: Optional[EmulatorWeights] = None,
    sw_scaling: Optional[InputScaling] = None,
    lw_scaling: Optional[InputScaling] = None,
) -> Tuple[RadiationTendencies, RadiationData]:
    """Emulated radiation scheme — drop-in replacement for ``radiation_scheme_rrtmgp``.

    Uses bidirectional GRU neural networks to predict shortwave and longwave
    fluxes, then derives heating rates from flux divergence. The call
    signature matches the other radiation schemes so it can be used
    interchangeably.

    Additional Args:
        emulator_weights: Trained NN weights (``EmulatorWeights``). Must be
            provided; passed through the parameters mechanism in IconPhysics.
        sw_scaling: Input normalization for SW network.
        lw_scaling: Input normalization for LW network.
    """
    from jax_solar import OrbitalTime, radiation_flux, get_solar_sin_altitude

    nlev = temperature.shape[0]
    phys = PhysicalConstants()

    # --- Solar geometry ---
    actual_date = getattr(date, "dt", date)
    orbital_time = OrbitalTime.from_datetime(actual_date)
    toa_flux = radiation_flux(
        orbital_time, longitude, latitude, parameters.solar_constant
    )
    sin_altitude = get_solar_sin_altitude(orbital_time, longitude, latitude)
    cos_zenith = jnp.maximum(sin_altitude, parameters.min_cos_zenith)

    # --- Prepare inputs common to SW and LW ---
    # Water vapour mixing ratio
    eps = phys.eps  # Mv/Md ≈ 0.622
    h2o_vmr = specific_humidity / (eps * (1.0 - specific_humidity) + specific_humidity)

    # Ozone
    if ozone_vmr is None:
        ozone_vmr = jnp.full(nlev, 5e-6)

    # Cloud water/ice paths (kg/m^2)
    cwp = cloud_water * air_density * layer_thickness * cloud_fraction
    cip = cloud_ice * air_density * layer_thickness * cloud_fraction

    # Default scaling if not provided
    if sw_scaling is None:
        sw_scaling = InputScaling(x_max=jnp.ones(7))
    if lw_scaling is None:
        lw_scaling = InputScaling(x_max=jnp.ones(7))

    # --- Shortwave ---
    sw_input = preprocess_sw_inputs(
        temperature, pressure_levels, h2o_vmr, ozone_vmr,
        cwp, cip, cos_zenith, sw_scaling,
    )
    surface_albedo = 0.5 * (surface_albedo_vis + surface_albedo_nir)
    sw_nn_output = sw_emulator_column(
        sw_input, jnp.atleast_1d(surface_albedo), emulator_weights.sw,
    )
    toa_sw_down = jnp.maximum(toa_flux, 0.0)
    sw_flux_down, sw_flux_up = reconstruct_sw_fluxes(
        sw_nn_output, toa_sw_down, surface_albedo,
    )

    # --- Longwave ---
    lw_input = preprocess_lw_inputs(
        temperature, pressure_levels, h2o_vmr, ozone_vmr,
        cwp, cip, co2_vmr, lw_scaling,
    )
    lw_nn_output = lw_emulator_column(
        lw_input, jnp.atleast_1d(surface_emissivity), emulator_weights.lw,
    )
    lw_flux_down, lw_flux_up = reconstruct_lw_fluxes(
        lw_nn_output, surface_temperature, surface_emissivity,
    )

    # --- Heating rates ---
    sw_heating = flux_to_heating_rate(sw_flux_down, sw_flux_up, pressure_interfaces)
    lw_heating = flux_to_heating_rate(lw_flux_down, lw_flux_up, pressure_interfaces)
    total_heating = sw_heating + lw_heating

    tendencies = RadiationTendencies(
        temperature_tendency=total_heating,
        longwave_heating=lw_heating,
        shortwave_heating=sw_heating,
    )

    diagnostics = RadiationData(
        cos_zenith=cos_zenith,
        surface_albedo_vis=jnp.atleast_1d(surface_albedo_vis),
        surface_albedo_nir=jnp.atleast_1d(surface_albedo_nir),
        surface_emissivity=jnp.atleast_1d(surface_emissivity),
        sw_flux_up=sw_flux_up,
        sw_flux_down=sw_flux_down,
        sw_heating_rate=sw_heating,
        lw_flux_up=lw_flux_up,
        lw_flux_down=lw_flux_down,
        lw_heating_rate=lw_heating,
        surface_sw_down=sw_flux_down[-1],
        surface_lw_down=lw_flux_down[-1],
        surface_sw_up=sw_flux_up[-1],
        surface_lw_up=lw_flux_up[-1],
        toa_sw_up=sw_flux_up[0],
        toa_lw_up=lw_flux_up[0],
        toa_sw_down=toa_sw_down,
    )

    return tendencies, diagnostics
