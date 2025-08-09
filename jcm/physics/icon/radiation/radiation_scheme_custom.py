"""
Radiation scheme dispatcher for ICON physics.

This module provides a unified interface to dispatch between different
radiation schemes (e.g., ICON native and RRTMGP) based on configuration.
"""

from typing import Tuple
import jax.numpy as jnp

from jcm.physics.icon.radiation.radiation_scheme import radiation_scheme as icon_radiation_scheme
from jcm.physics.icon.radiation.radiation_types import RadiationParameters, RadiationTendencies
from ..icon_physics_data import RadiationData


def radiation_scheme_rrtmgp(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure_levels: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    air_density: jnp.ndarray,
    cloud_water: jnp.ndarray,
    cloud_ice: jnp.ndarray,
    cloud_fraction: jnp.ndarray,
    day_of_year: float,
    seconds_since_midnight: float,
    latitude: float,
    longitude: float,
    parameters: RadiationParameters,
    aerosol_data,
    ozone_vmr: jnp.ndarray = None,
    co2_vmr: float = 400e-6
) -> Tuple[RadiationTendencies, RadiationData]:
    """
    RRTMGP-based radiation scheme wrapper.
    
    This function will interface with jax-rrtmgp to provide radiation calculations
    compatible with the ICON radiation interface.
    
    For now, this is a placeholder that calls the ICON scheme.
    TODO: Implement actual RRTMGP interface.
    
    Args:
        Same as icon_radiation_scheme
        
    Returns:
        RadiationTendencies and RadiationData in same format as ICON scheme
    """
    # Placeholder: Use ICON scheme for now
    # TODO: In the future, this will call jax-rrtmgp
    return icon_radiation_scheme(
        temperature=temperature,
        specific_humidity=specific_humidity,
        pressure_levels=pressure_levels,
        layer_thickness=layer_thickness,
        air_density=air_density,
        cloud_water=cloud_water,
        cloud_ice=cloud_ice,
        cloud_fraction=cloud_fraction,
        day_of_year=day_of_year,
        seconds_since_midnight=seconds_since_midnight,
        latitude=latitude,
        longitude=longitude,
        parameters=parameters,
        aerosol_data=aerosol_data,
        ozone_vmr=ozone_vmr,
        co2_vmr=co2_vmr
    )