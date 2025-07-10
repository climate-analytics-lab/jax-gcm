"""
Vertical diffusion and boundary layer parameterizations for ICON physics

This module contains vertical diffusion schemes including turbulent mixing
in the boundary layer and free atmosphere, using implicit matrix solvers.

The implementation follows ICON's approach with:
- Tridiagonal matrix solver for implicit time stepping
- Exchange coefficient calculations based on mixing length theory
- Richardson number stability corrections
- Surface boundary layer parameterization
"""

from .vertical_diffusion_types import (
    VDiffParameters,
    VDiffState,
    VDiffTendencies,
    VDiffDiagnostics,
    VDiffMatrixSystem
)

from .turbulence_coefficients import (
    compute_richardson_number,
    compute_mixing_length,
    compute_exchange_coefficients,
    compute_boundary_layer_height,
    compute_friction_velocity,
    compute_turbulence_diagnostics
)

from .matrix_solver import (
    setup_matrix_system,
    solve_tridiagonal_system,
    vertical_diffusion_step
)

from .vertical_diffusion import (
    vertical_diffusion_scheme,
    vertical_diffusion_scheme_vectorized,
    prepare_vertical_diffusion_state,
    compute_dry_static_energy,
    compute_virtual_temperature
)

__all__ = [
    # Types
    "VDiffParameters",
    "VDiffState", 
    "VDiffTendencies",
    "VDiffDiagnostics",
    "VDiffMatrixSystem",
    
    # Turbulence coefficients
    "compute_richardson_number",
    "compute_mixing_length",
    "compute_exchange_coefficients",
    "compute_boundary_layer_height",
    "compute_friction_velocity",
    "compute_turbulence_diagnostics",
    
    # Matrix solver
    "setup_matrix_system",
    "solve_tridiagonal_system",
    "vertical_diffusion_step",
    
    # Main interface
    "vertical_diffusion_scheme",
    "vertical_diffusion_scheme_vectorized",
    "prepare_vertical_diffusion_state",
    "compute_dry_static_energy",
    "compute_virtual_temperature"
]