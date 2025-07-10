"""
ICON Atmospheric Physics Package for JAX-GCM

This package contains JAX implementations of the ICON atmospheric physics
parameterizations originally written in Fortran. The modules are organized
by physics category and follow the same structure as the SPEEDY physics
implementation.

Physics Categories:
- constants: Physical constants and lookup tables
- boundary_conditions: Forcing data and boundary conditions
- radiation: Shortwave and longwave radiation
- convection: Convective parameterizations
- clouds: Large-scale cloud microphysics
- vertical_diffusion: Boundary layer and turbulent mixing
- surface: Land-atmosphere exchange
- gravity_waves: Atmospheric gravity wave drag
- chemistry: Simple chemistry schemes
- diagnostics: Physics diagnostics and utilities

The conversion follows a modular approach where each physics process is
implemented as a separate JAX function that can be composed together.
"""

from jcm.physics.icon.constants import physical_constants
from jcm.physics.icon.icon_physics import IconPhysics
from jcm.physics.icon.parameters import Parameters
from jcm.physics.icon.diagnostics import wmo_tropopause

__all__ = [
    'physical_constants',
    'IconPhysics',
    'Parameters',
    'wmo_tropopause',
]