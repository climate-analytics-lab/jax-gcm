"""ICON Atmospheric Physics Package for JAX-GCM

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

__all__ = [
    'physical_constants',
]


def __getattr__(name):
    """Lazy imports to avoid circular dependencies after reorganization."""
    if name == "icon_physics":
        from jcm.physics.icon.icon_terms import icon_physics
        return icon_physics
    if name == "Parameters":
        from jcm.physics.icon.parameters import Parameters
        return Parameters
    if name == "wmo_tropopause":
        from jcm.physics.diagnostics.wmo_tropopause import wmo_tropopause
        return wmo_tropopause
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")