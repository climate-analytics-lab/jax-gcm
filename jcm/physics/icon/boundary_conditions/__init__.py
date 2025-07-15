"""
Boundary conditions and forcing data for ICON physics

This module handles external forcings and boundary conditions including:
- Solar irradiance
- Greenhouse gas concentrations
- Ozone concentrations
- Aerosol optical properties
- Sea surface temperature and sea ice
"""

from .simple_boundary_conditions import (
    BoundaryConditionParameters,
    BoundaryConditionState,
    simple_boundary_conditions,
    create_idealized_boundary_conditions,
    compute_solar_zenith_angle,
    compute_solar_irradiance,
    compute_surface_properties
)

__all__ = [
    'BoundaryConditionParameters',
    'BoundaryConditionState',
    'simple_boundary_conditions',
    'create_idealized_boundary_conditions',
    'compute_solar_zenith_angle',
    'compute_solar_irradiance',
    'compute_surface_properties'
]