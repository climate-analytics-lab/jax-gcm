"""
Surface parameterizations for ICON physics

This module contains land-atmosphere exchange processes including
heat, moisture, and momentum fluxes, vegetation effects, and soil processes.
"""

from .surface_types import (
    SurfaceParameters, SurfaceState, AtmosphericForcing, 
    SurfaceFluxes, SurfaceTendencies, SurfaceDiagnostics, SurfaceResistances
)
from .surface_physics import (
    initialize_surface_state, surface_physics_step, update_surface_state
)
from .turbulent_fluxes import (
    compute_bulk_richardson_number, compute_stability_functions,
    compute_exchange_coefficients, compute_turbulent_fluxes
)
from .ocean import ocean_physics_step
from .sea_ice import sea_ice_physics_step
from .land import land_surface_physics_step

__all__ = [
    # Data structures
    'SurfaceParameters', 'SurfaceState', 'AtmosphericForcing',
    'SurfaceFluxes', 'SurfaceTendencies', 'SurfaceDiagnostics', 'SurfaceResistances',
    
    # Main interface
    'initialize_surface_state', 'surface_physics_step', 'update_surface_state',
    
    # Turbulent fluxes
    'compute_bulk_richardson_number', 'compute_stability_functions',
    'compute_exchange_coefficients', 'compute_turbulent_fluxes',
    
    # Surface type physics
    'ocean_physics_step', 'sea_ice_physics_step', 'land_surface_physics_step'
]