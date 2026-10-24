"""Vertical diffusion and boundary layer parameterizations for ECHAM physics

This module contains vertical diffusion schemes including turbulent mixing
in the boundary layer and free atmosphere, using implicit matrix solvers.

The implementation is ECHAM6.3's ``vdiff``:
- the TKE closure (``closure.py``): PBL extension, Blackadar/Holtslag-Boville
  mixing length, Louis (1979) stability functions, implicit TKE and the
  theta_v-variance budget, exchange coefficients on the interfaces
- the moist, cloud-weighted buoyancy (``moist_buoyancy.py``)
- the surface layer (``surface_layer.py``)
- the implicit tridiagonal column solve (``matrix_solver.py``)
"""

from .vertical_diffusion_types import (
    VDiffParameters,
    VDiffState,
    VDiffTendencies,
    VDiffDiagnostics,
    VDiffMatrixSystem,
    VDiffSurfaceFluxes,
    InterfaceConductances,
)

from .turbulence_coefficients import (
    compute_richardson_number,
    compute_surface_exchange_coefficients,
    compute_friction_velocity,
    compute_turbulence_diagnostics
)
from .closure import (
    InteriorClosure,
    PblExtension,
    closure_constants,
    geopotential_above_surface,
    interior_closure,
    louis_stability,
    lowest_level_height_above_surface,
    mixing_length,
    pbl_extension,
    surface_tke_boundary,
    thv_variance_source_update,
    tke_source_update,
)
from .surface_layer import (
    compute_surface_exchange_coefficients_echam_louis,
    surface_bulk_richardson,
)
from .moist_buoyancy import (
    InteriorBuoyancyTerms,
    cloud_weighted_buoyancy_multipliers,
    interior_buoyancy_and_shear,
    interior_buoyancy_terms,
    richardson_number,
)

from .matrix_solver import (
    setup_matrix_system,
    solve_tridiagonal_system,
    vertical_diffusion_step,
    couple_surface_tiles,
)

from .vertical_diffusion import (
    TteTkeVerticalDiffusion,
    vertical_diffusion_scheme,
    vertical_diffusion_scheme_vectorized,
    prepare_vertical_diffusion_state,
    vertical_diffusion_column,
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
    "VDiffSurfaceFluxes",
    "InterfaceConductances",

    # Turbulence coefficients
    "compute_richardson_number",
    "compute_surface_exchange_coefficients",
    "compute_surface_exchange_coefficients_echam_louis",
    "surface_bulk_richardson",
    "InteriorBuoyancyTerms",
    "cloud_weighted_buoyancy_multipliers",
    "interior_buoyancy_and_shear",
    "interior_buoyancy_terms",
    "richardson_number",
    "compute_friction_velocity",
    "compute_turbulence_diagnostics",

    # ECHAM closure
    "InteriorClosure",
    "PblExtension",
    "closure_constants",
    "geopotential_above_surface",
    "interior_closure",
    "louis_stability",
    "lowest_level_height_above_surface",
    "mixing_length",
    "pbl_extension",
    "surface_tke_boundary",
    "thv_variance_source_update",
    "tke_source_update",

    # Matrix solver
    "setup_matrix_system",
    "solve_tridiagonal_system",
    "vertical_diffusion_step",
    "couple_surface_tiles",

    # Main interface
    "TteTkeVerticalDiffusion",
    "vertical_diffusion_scheme",
    "vertical_diffusion_scheme_vectorized",
    "prepare_vertical_diffusion_state",
    "vertical_diffusion_column",
    "compute_dry_static_energy",
    "compute_virtual_temperature"
]