"""
Main vertical diffusion scheme for ICON physics.

This module provides the main interface for vertical diffusion and boundary layer
physics, integrating turbulence coefficient calculations with the matrix solver.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

from jcm.physics.icon.constants.physical_constants import PhysicalConstants
from .vertical_diffusion_types import (
    VDiffState, VDiffParameters, VDiffTendencies, VDiffDiagnostics
)

# Create constants instance
PHYS_CONST = PhysicalConstants()
from .turbulence_coefficients import (
    compute_richardson_number, compute_mixing_length, compute_exchange_coefficients,
    compute_turbulence_diagnostics
)
from .matrix_solver import vertical_diffusion_step


@jax.jit
def compute_dry_static_energy(
    temperature: jnp.ndarray,
    geopotential: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute dry static energy.
    
    Args:
        temperature: Temperature [K]
        geopotential: Geopotential [m²/s²]
        
    Returns:
        Dry static energy [J/kg]
    """
    return PHYS_CONST.cp * temperature + geopotential


@jax.jit
def compute_virtual_temperature(
    temperature: jnp.ndarray,
    qv: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute virtual temperature.
    
    Args:
        temperature: Temperature [K]
        qv: Water vapor mixing ratio [kg/kg]
        
    Returns:
        Virtual temperature [K]
    """
    return temperature * (1.0 + 0.608 * qv)


@jax.jit
def prepare_vertical_diffusion_state(
    u: jnp.ndarray,
    v: jnp.ndarray,
    temperature: jnp.ndarray,
    qv: jnp.ndarray,
    qc: jnp.ndarray,
    qi: jnp.ndarray,
    pressure_full: jnp.ndarray,
    pressure_half: jnp.ndarray,
    geopotential: jnp.ndarray,
    height_full: jnp.ndarray,
    height_half: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    surface_fraction: jnp.ndarray,
    roughness_length: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    tke: jnp.ndarray,
    thv_variance: jnp.ndarray
) -> VDiffState:
    """
    Prepare the vertical diffusion state from input variables.
    
    Args:
        u: Zonal wind [m/s] (ncol, nlev)
        v: Meridional wind [m/s] (ncol, nlev)
        temperature: Temperature [K] (ncol, nlev)
        qv: Water vapor mixing ratio [kg/kg] (ncol, nlev)
        qc: Cloud water mixing ratio [kg/kg] (ncol, nlev)
        qi: Cloud ice mixing ratio [kg/kg] (ncol, nlev)
        pressure_full: Full level pressure [Pa] (ncol, nlev)
        pressure_half: Half level pressure [Pa] (ncol, nlev+1)
        geopotential: Geopotential [m²/s²] (ncol, nlev)
        height_full: Full level height [m] (ncol, nlev)
        height_half: Half level height [m] (ncol, nlev+1)
        surface_temperature: Surface temperature [K] (ncol, nsfc_type)
        surface_fraction: Surface type fraction [-] (ncol, nsfc_type)
        roughness_length: Roughness length [m] (ncol, nsfc_type)
        ocean_u: Ocean u-velocity [m/s] (ncol,)
        ocean_v: Ocean v-velocity [m/s] (ncol,)
        tke: Turbulent kinetic energy [m²/s²] (ncol, nlev)
        thv_variance: Variance of theta_v [K²] (ncol, nlev)
        
    Returns:
        Complete vertical diffusion state
    """
    # Compute air masses
    dp = jnp.diff(pressure_half, axis=1)
    air_mass = dp / PHYS_CONST.grav
    
    # Approximate dry air mass (could be more sophisticated)
    dry_air_mass = air_mass * (1.0 - qv)
    
    return VDiffState(
        u=u,
        v=v,
        temperature=temperature,
        qv=qv,
        qc=qc,
        qi=qi,
        pressure_full=pressure_full,
        pressure_half=pressure_half,
        geopotential=geopotential,
        air_mass=air_mass,
        dry_air_mass=dry_air_mass,
        surface_temperature=surface_temperature,
        surface_fraction=surface_fraction,
        roughness_length=roughness_length,
        height_full=height_full,
        height_half=height_half,
        tke=tke,
        thv_variance=thv_variance,
        ocean_u=ocean_u,
        ocean_v=ocean_v
    )


@jax.jit
def vertical_diffusion_column(
    state: VDiffState,
    params: VDiffParameters,
    dt: float
) -> Tuple[VDiffTendencies, VDiffDiagnostics]:
    """
    Compute vertical diffusion for a single column.
    
    Args:
        state: Vertical diffusion state
        params: Vertical diffusion parameters
        dt: Time step [s]
        
    Returns:
        Tuple of (tendencies, diagnostics)
    """
    # Compute turbulence coefficients
    ri = compute_richardson_number(
        state.u, state.v, state.temperature,
        state.height_full, state.height_half
    )
    
    # Estimate boundary layer height (initial guess)
    pbl_height_guess = jnp.full(state.u.shape[0], 1000.0)
    
    mixing_length = compute_mixing_length(
        state.height_full, state.height_half, ri, pbl_height_guess
    )
    
    exchange_coeff_momentum, exchange_coeff_heat, exchange_coeff_moisture = (
        compute_exchange_coefficients(state, params, mixing_length, ri)
    )
    
    # Compute diagnostics
    diagnostics = compute_turbulence_diagnostics(
        state, params, exchange_coeff_momentum, 
        exchange_coeff_heat, exchange_coeff_moisture
    )
    
    # Perform vertical diffusion step
    tendencies = vertical_diffusion_step(
        state, params, exchange_coeff_momentum,
        exchange_coeff_heat, exchange_coeff_moisture, dt
    )
    
    return tendencies, diagnostics


@jax.jit
def vertical_diffusion_scheme(
    u: jnp.ndarray,
    v: jnp.ndarray,
    temperature: jnp.ndarray,
    qv: jnp.ndarray,
    qc: jnp.ndarray,
    qi: jnp.ndarray,
    pressure_full: jnp.ndarray,
    pressure_half: jnp.ndarray,
    geopotential: jnp.ndarray,
    height_full: jnp.ndarray,
    height_half: jnp.ndarray,
    surface_temperature: jnp.ndarray,
    surface_fraction: jnp.ndarray,
    roughness_length: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    tke: jnp.ndarray,
    thv_variance: jnp.ndarray,
    dt: float,
    params: VDiffParameters
) -> Tuple[VDiffTendencies, VDiffDiagnostics]:
    """
    Main vertical diffusion scheme interface.
    
    Args:
        u: Zonal wind [m/s] (ncol, nlev)
        v: Meridional wind [m/s] (ncol, nlev)
        temperature: Temperature [K] (ncol, nlev)
        qv: Water vapor mixing ratio [kg/kg] (ncol, nlev)
        qc: Cloud water mixing ratio [kg/kg] (ncol, nlev)
        qi: Cloud ice mixing ratio [kg/kg] (ncol, nlev)
        pressure_full: Full level pressure [Pa] (ncol, nlev)
        pressure_half: Half level pressure [Pa] (ncol, nlev+1)
        geopotential: Geopotential [m²/s²] (ncol, nlev)
        height_full: Full level height [m] (ncol, nlev)
        height_half: Half level height [m] (ncol, nlev+1)
        surface_temperature: Surface temperature [K] (ncol, nsfc_type)
        surface_fraction: Surface type fraction [-] (ncol, nsfc_type)
        roughness_length: Roughness length [m] (ncol, nsfc_type)
        ocean_u: Ocean u-velocity [m/s] (ncol,)
        ocean_v: Ocean v-velocity [m/s] (ncol,)
        tke: Turbulent kinetic energy [m²/s²] (ncol, nlev)
        thv_variance: Variance of theta_v [K²] (ncol, nlev)
        dt: Time step [s]
        params: Vertical diffusion parameters
        
    Returns:
        Tuple of (tendencies, diagnostics)
    """
    # Prepare state
    state = prepare_vertical_diffusion_state(
        u, v, temperature, qv, qc, qi,
        pressure_full, pressure_half, geopotential,
        height_full, height_half,
        surface_temperature, surface_fraction, roughness_length,
        ocean_u, ocean_v, tke, thv_variance
    )
    
    # Compute vertical diffusion
    tendencies, diagnostics = vertical_diffusion_column(state, params, dt)
    
    return tendencies, diagnostics


# Vectorized version for multiple columns
vertical_diffusion_scheme_vectorized = jax.vmap(
    vertical_diffusion_scheme,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
    out_axes=(0, 0)
)