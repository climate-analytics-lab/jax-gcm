"""
Turbulent kinetic energy (TKE) budget calculations for vertical diffusion.

This module computes the TKE budget according to the standard TKE equation:
d(TKE)/dt = Shear Production + Buoyancy Production - Dissipation + Transport

The TKE equation is:
d(e)/dt = P_s + P_b - ε + ∂/∂z(K_e ∂e/∂z)

where:
- e = TKE (turbulent kinetic energy)
- P_s = Shear production = K_m * (∂u/∂z)²
- P_b = Buoyancy production = -K_h * (g/θ) * (∂θ/∂z)
- ε = Dissipation = C_ε * e^(3/2) / l
- K_e = TKE exchange coefficient
"""

import jax
import jax.numpy as jnp
from typing import Tuple

from jcm.physics.icon.constants.physical_constants import PhysicalConstants
from .vertical_diffusion_types import VDiffState, VDiffParameters

# Create constants instance
PHYS_CONST = PhysicalConstants()


@jax.jit
def compute_shear_production(
    u: jnp.ndarray,
    v: jnp.ndarray,
    height_full: jnp.ndarray,
    exchange_coeff_momentum: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute shear production term in TKE budget.
    
    P_s = K_m * [(∂u/∂z)² + (∂v/∂z)²]
    
    Args:
        u: Zonal wind [m/s] (ncol, nlev)
        v: Meridional wind [m/s] (ncol, nlev)
        height_full: Full level heights [m] (ncol, nlev)
        exchange_coeff_momentum: Momentum exchange coefficient [m²/s] (ncol, nlev)
        
    Returns:
        Shear production [m²/s³] (ncol, nlev)
    """
    # Compute vertical wind shear
    du_dz = jnp.diff(u, axis=1) / jnp.diff(height_full, axis=1)
    dv_dz = jnp.diff(v, axis=1) / jnp.diff(height_full, axis=1)
    
    # Extend to full levels (nlev) by padding with boundary values
    du_dz_extended = jnp.concatenate([
        du_dz[:, :1],  # Extend top value
        du_dz          # Interior values (nlev-1)
    ], axis=1)
    
    dv_dz_extended = jnp.concatenate([
        dv_dz[:, :1],  # Extend top value
        dv_dz          # Interior values (nlev-1)
    ], axis=1)
    
    # Shear production: P_s = K_m * (S²)
    # where S² = (∂u/∂z)² + (∂v/∂z)²
    shear_squared = du_dz_extended**2 + dv_dz_extended**2
    
    # Multiply by exchange coefficient
    shear_production = exchange_coeff_momentum * shear_squared
    
    return shear_production


@jax.jit
def compute_buoyancy_production(
    temperature: jnp.ndarray,
    height_full: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    gravity: float = PHYS_CONST.grav
) -> jnp.ndarray:
    """
    Compute buoyancy production term in TKE budget.
    
    P_b = -K_h * (g/θ) * (∂θ/∂z)
    
    Args:
        temperature: Temperature [K] (ncol, nlev)
        height_full: Full level heights [m] (ncol, nlev)
        exchange_coeff_heat: Heat exchange coefficient [m²/s] (ncol, nlev)
        gravity: Gravitational acceleration [m/s²]
        
    Returns:
        Buoyancy production [m²/s³] (ncol, nlev)
    """
    # Compute vertical temperature gradient
    dt_dz = jnp.diff(temperature, axis=1) / jnp.diff(height_full, axis=1)
    
    # Extend to full levels (nlev) by padding with boundary values
    dt_dz_extended = jnp.concatenate([
        dt_dz[:, :1],  # Extend top value
        dt_dz          # Interior values (nlev-1)
    ], axis=1)
    
    # Average temperature for buoyancy frequency
    temp_avg = temperature  # Use full level temperature directly
    
    # Buoyancy production: P_b = -K_h * (g/T) * (dT/dz + g/cp)
    # Note: The dry adiabatic lapse rate g/cp is included for stability
    lapse_rate = gravity / PHYS_CONST.cp
    buoyancy_freq = (gravity / temp_avg) * (dt_dz_extended + lapse_rate)
    
    # Buoyancy production (negative for stable stratification)
    buoyancy_production = -exchange_coeff_heat * buoyancy_freq
    
    return buoyancy_production


@jax.jit
def compute_dissipation(
    tke: jnp.ndarray,
    mixing_length: jnp.ndarray,
    c_dissipation: float = 0.19
) -> jnp.ndarray:
    """
    Compute dissipation term in TKE budget.
    
    ε = C_ε * e^(3/2) / l
    
    Args:
        tke: Turbulent kinetic energy [m²/s²] (ncol, nlev)
        mixing_length: Mixing length [m] (ncol, nlev)
        c_dissipation: Dissipation constant [-]
        
    Returns:
        Dissipation rate [m²/s³] (ncol, nlev)
    """
    # Ensure TKE is positive
    tke_positive = jnp.maximum(tke, 1e-8)
    
    # Dissipation: ε = C_ε * e^(3/2) / l
    # Use sqrt(tke) * tke / mixing_length for numerical stability
    dissipation = c_dissipation * jnp.sqrt(tke_positive) * tke_positive / mixing_length
    
    return dissipation


@jax.jit
def compute_tke_exchange_coefficient(
    tke: jnp.ndarray,
    mixing_length: jnp.ndarray,
    c_tke: float = 0.1
) -> jnp.ndarray:
    """
    Compute TKE exchange coefficient.
    
    K_e = C_tke * l * sqrt(e)
    
    Args:
        tke: Turbulent kinetic energy [m²/s²] (ncol, nlev)
        mixing_length: Mixing length [m] (ncol, nlev)
        c_tke: TKE transport coefficient [-]
        
    Returns:
        TKE exchange coefficient [m²/s] (ncol, nlev)
    """
    # Ensure TKE is positive
    tke_positive = jnp.maximum(tke, 1e-8)
    
    # TKE exchange coefficient: K_e = C_tke * l * sqrt(e)
    tke_exchange_coeff = c_tke * mixing_length * jnp.sqrt(tke_positive)
    
    return tke_exchange_coeff


@jax.jit
def compute_tke_tendency(
    state: VDiffState,
    params: VDiffParameters,
    exchange_coeff_momentum: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    mixing_length: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute complete TKE tendency from budget equation.
    
    d(TKE)/dt = Shear Production + Buoyancy Production - Dissipation + Transport
    
    Args:
        state: Atmospheric state
        params: Vertical diffusion parameters
        exchange_coeff_momentum: Momentum exchange coefficient [m²/s] (ncol, nlev)
        exchange_coeff_heat: Heat exchange coefficient [m²/s] (ncol, nlev)
        mixing_length: Mixing length [m] (ncol, nlev)
        
    Returns:
        TKE tendency [m²/s³] (ncol, nlev)
    """
    # Shear production
    shear_production = compute_shear_production(
        state.u, state.v, state.height_full, exchange_coeff_momentum
    )
    
    # Buoyancy production
    buoyancy_production = compute_buoyancy_production(
        state.temperature, state.height_full, exchange_coeff_heat
    )
    
    # Dissipation
    dissipation = compute_dissipation(state.tke, mixing_length)
    
    # TKE exchange coefficient for transport term
    tke_exchange_coeff = compute_tke_exchange_coefficient(state.tke, mixing_length)
    
    # Transport term: ∂/∂z(K_e ∂e/∂z)
    # For now, we'll compute this as part of the matrix solver
    # Here we just sum the source terms
    transport_term = jnp.zeros_like(state.tke)  # Will be handled by matrix solver
    
    # Total TKE tendency
    tke_tendency = (shear_production + buoyancy_production - dissipation + transport_term)
    
    return tke_tendency


@jax.jit
def compute_tke_diagnostics(
    state: VDiffState,
    params: VDiffParameters,
    exchange_coeff_momentum: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    mixing_length: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute TKE budget diagnostics for analysis.
    
    Args:
        state: Atmospheric state
        params: Vertical diffusion parameters
        exchange_coeff_momentum: Momentum exchange coefficient [m²/s] (ncol, nlev)
        exchange_coeff_heat: Heat exchange coefficient [m²/s] (ncol, nlev)
        mixing_length: Mixing length [m] (ncol, nlev)
        
    Returns:
        Tuple of:
        - Shear production [m²/s³] (ncol, nlev)
        - Buoyancy production [m²/s³] (ncol, nlev)
        - Dissipation [m²/s³] (ncol, nlev)
        - TKE exchange coefficient [m²/s] (ncol, nlev)
    """
    shear_production = compute_shear_production(
        state.u, state.v, state.height_full, exchange_coeff_momentum
    )
    
    buoyancy_production = compute_buoyancy_production(
        state.temperature, state.height_full, exchange_coeff_heat
    )
    
    dissipation = compute_dissipation(state.tke, mixing_length)
    
    tke_exchange_coeff = compute_tke_exchange_coefficient(state.tke, mixing_length)
    
    return shear_production, buoyancy_production, dissipation, tke_exchange_coeff


@jax.jit
def minimum_tke_constraint(
    tke: jnp.ndarray,
    min_tke: float = 1e-6
) -> jnp.ndarray:
    """
    Apply minimum TKE constraint to prevent negative values.
    
    Args:
        tke: Turbulent kinetic energy [m²/s²] (ncol, nlev)
        min_tke: Minimum TKE value [m²/s²]
        
    Returns:
        Constrained TKE [m²/s²] (ncol, nlev)
    """
    return jnp.maximum(tke, min_tke)