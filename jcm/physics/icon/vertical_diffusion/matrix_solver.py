"""
Tridiagonal matrix solver for vertical diffusion.

This module implements the implicit tridiagonal matrix solver used in ICON's
vertical diffusion scheme, following the downward sweep/upward sweep approach.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

from jcm.physics.icon.constants.physical_constants import PhysicalConstants
from .vertical_diffusion_types import (
    VDiffState, VDiffParameters, VDiffMatrixSystem, VDiffTendencies
)

# Create constants instance
PHYS_CONST = PhysicalConstants()


@jax.jit
def setup_matrix_system(
    state: VDiffState,
    params: VDiffParameters,
    exchange_coeff_momentum: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    exchange_coeff_moisture: jnp.ndarray,
    dt: float,
    tke_exchange_coeff: jnp.ndarray = None
) -> VDiffMatrixSystem:
    """
    Set up the tridiagonal matrix system for vertical diffusion.
    
    Args:
        state: Atmospheric state
        params: Vertical diffusion parameters
        exchange_coeff_momentum: Momentum exchange coefficient [m²/s]
        exchange_coeff_heat: Heat exchange coefficient [m²/s]
        exchange_coeff_moisture: Moisture exchange coefficient [m²/s]
        dt: Time step [s]
        
    Returns:
        Matrix system ready for solution
    """
    ncol, nlev = state.u.shape
    nsfc_type = 3  # Fixed number of surface types (water, ice, land)
    
    # Number of variables and matrices
    # Variables: u, v, T, qv, qc, qi, TKE, thv_var (fixed 8 variables)
    nvar_total = 8  # Fixed number of variables (no additional tracers)
    
    # Matrix types: momentum, heat, moisture, hydrometeors, TKE, thv_var
    nmatrix = 6
    
    # Initialize matrices
    matrix_coeffs = jnp.zeros((ncol, nlev, 3, nmatrix))
    matrix_bottom = jnp.zeros((ncol, 3, nsfc_type, 2))  # Only heat and moisture need surface BC
    rhs_vectors = jnp.zeros((ncol, nlev, nvar_total))
    rhs_surface = jnp.zeros((ncol, nsfc_type, 2))
    
    # Variable to matrix mapping
    variable_to_matrix = jnp.array([
        0, 0,  # u, v -> momentum matrix
        1,     # T -> heat matrix
        2,     # qv -> moisture matrix
        3, 3,  # qc, qi -> hydrometeor matrix
        4,     # TKE -> TKE matrix
        5      # thv_var -> thv_var matrix
    ])
    
    # No additional tracers - qc and qi are already included in the base variables
    
    # Reciprocal air mass for matrix coefficients
    recip_air_mass = 1.0 / state.air_mass
    recip_dry_air_mass = 1.0 / state.dry_air_mass
    
    # Time step factors
    dt_factor = dt * params.tpfac1
    
    # Setup momentum matrix (u, v)
    matrix_coeffs = setup_momentum_matrix(
        matrix_coeffs, exchange_coeff_momentum, recip_air_mass, dt_factor, 0
    )
    
    # Setup heat matrix (T)
    matrix_coeffs = setup_heat_matrix(
        matrix_coeffs, exchange_coeff_heat, recip_air_mass, dt_factor, 1
    )
    
    # Setup moisture matrix (qv)
    matrix_coeffs = setup_moisture_matrix(
        matrix_coeffs, exchange_coeff_moisture, recip_dry_air_mass, dt_factor, 2
    )
    
    # Setup hydrometeor matrix (qc, qi, tracers)
    matrix_coeffs = setup_hydrometeor_matrix(
        matrix_coeffs, exchange_coeff_heat, recip_dry_air_mass, dt_factor, 3
    )
    
    # Setup TKE matrix (use TKE exchange coefficient)
    matrix_coeffs = setup_tke_matrix(
        matrix_coeffs, tke_exchange_coeff, recip_air_mass, dt_factor, 4
    )
    
    # Setup theta_v variance matrix
    matrix_coeffs = setup_thv_matrix(
        matrix_coeffs, exchange_coeff_heat, recip_air_mass, dt_factor, 5
    )
    
    # Setup right-hand side vectors
    rhs_vectors = setup_rhs_vectors(state, params)
    
    return VDiffMatrixSystem(
        matrix_coeffs=matrix_coeffs,
        matrix_bottom=matrix_bottom,
        rhs_vectors=rhs_vectors,
        rhs_surface=rhs_surface,
        variable_to_matrix=variable_to_matrix
    )


@jax.jit
def setup_momentum_matrix(
    matrix_coeffs: jnp.ndarray,
    exchange_coeff: jnp.ndarray,
    recip_air_mass: jnp.ndarray,
    dt_factor: float,
    matrix_idx: int
) -> jnp.ndarray:
    """Set up tridiagonal matrix for momentum equations."""
    ncol, nlev = exchange_coeff.shape
    
    # Exchange coefficient on half levels (between full levels)
    # Surface flux is handled separately
    k_half = jnp.zeros((ncol, nlev + 1))
    k_half = k_half.at[:, 1:nlev].set(
        0.5 * (exchange_coeff[:, :-1] + exchange_coeff[:, 1:])
    )
    # k_half[:, 0] = 0 (no flux at top)
    # k_half[:, nlev] will be set by surface conditions
    
    # Scaled exchange coefficients
    k_scaled = k_half * dt_factor
    
    # Build tridiagonal matrix

    # Sub-diagonal (connection to level below)
    sub_diagonal_vals = -k_scaled[:, 1:-1] * recip_air_mass[:, :-1]  # shape: [ncol, nlev-1]
    matrix_coeffs = matrix_coeffs.at[:, :-1, 0, matrix_idx].set(sub_diagonal_vals)

    # Super-diagonal (connection to level above)  
    super_diagonal_vals = -k_scaled[:, 1:-1] * recip_air_mass[:, 1:]  # shape: [ncol, nlev-1]
    matrix_coeffs = matrix_coeffs.at[:, 1:, 2, matrix_idx].set(super_diagonal_vals)

    # Diagonal
    above_contrib = jnp.concatenate([
        jnp.zeros((k_scaled.shape[0], 1)),  # k=0 has no above contribution
        -super_diagonal_vals
    ], axis=1)

    below_contrib = jnp.concatenate([
        -sub_diagonal_vals,  # k<nlev-1 contributions  
        jnp.zeros((k_scaled.shape[0], 1))  # k=nlev-1 has no below contribution
    ], axis=1)

    diagonal_vals = 1.0 + above_contrib + below_contrib
    matrix_coeffs = matrix_coeffs.at[:, :, 1, matrix_idx].set(diagonal_vals)

    return matrix_coeffs


@jax.jit
def setup_heat_matrix(
    matrix_coeffs: jnp.ndarray,
    exchange_coeff: jnp.ndarray,
    recip_air_mass: jnp.ndarray,
    dt_factor: float,
    matrix_idx: int
) -> jnp.ndarray:
    """Set up tridiagonal matrix for heat equation."""
    return setup_momentum_matrix(
        matrix_coeffs, exchange_coeff, recip_air_mass, dt_factor, matrix_idx
    )


@jax.jit
def setup_moisture_matrix(
    matrix_coeffs: jnp.ndarray,
    exchange_coeff: jnp.ndarray,
    recip_dry_air_mass: jnp.ndarray,
    dt_factor: float,
    matrix_idx: int
) -> jnp.ndarray:
    """Set up tridiagonal matrix for moisture equation."""
    return setup_momentum_matrix(
        matrix_coeffs, exchange_coeff, recip_dry_air_mass, dt_factor, matrix_idx
    )


@jax.jit
def setup_hydrometeor_matrix(
    matrix_coeffs: jnp.ndarray,
    exchange_coeff: jnp.ndarray,
    recip_dry_air_mass: jnp.ndarray,
    dt_factor: float,
    matrix_idx: int
) -> jnp.ndarray:
    """Set up tridiagonal matrix for hydrometeor equations."""
    # Hydrometeors have no surface flux, so bottom boundary condition is different; however, surface flux is handled separately
    return setup_momentum_matrix(
        matrix_coeffs, exchange_coeff, recip_dry_air_mass, dt_factor, matrix_idx
    )


@jax.jit
def setup_tke_matrix(
    matrix_coeffs: jnp.ndarray,
    exchange_coeff: jnp.ndarray,
    recip_air_mass: jnp.ndarray,
    dt_factor: float,
    matrix_idx: int
) -> jnp.ndarray:
    """Set up tridiagonal matrix for TKE equation."""
    # TKE uses its own exchange coefficient (from TKE budget)
    # but same matrix structure as other variables
    return setup_momentum_matrix(
        matrix_coeffs, exchange_coeff, recip_air_mass, dt_factor, matrix_idx
    )


@jax.jit
def setup_thv_matrix(
    matrix_coeffs: jnp.ndarray,
    exchange_coeff: jnp.ndarray,
    recip_air_mass: jnp.ndarray,
    dt_factor: float,
    matrix_idx: int
) -> jnp.ndarray:
    """Set up tridiagonal matrix for theta_v variance equation."""
    return setup_momentum_matrix(
        matrix_coeffs, exchange_coeff, recip_air_mass, dt_factor, matrix_idx
    )


@jax.jit
def setup_rhs_vectors(
    state: VDiffState,
    params: VDiffParameters
) -> jnp.ndarray:
    """Set up right-hand side vectors for the linear system."""
    ncol, nlev = state.u.shape
    # Fixed number of variables: u, v, T, qv, qc, qi, TKE, thv_var
    rhs = jnp.zeros((ncol, nlev, 8))
    
    # Current values as initial guess (implicit time stepping)
    rhs = rhs.at[:, :, 0].set(state.u * params.tpfac2)  # u
    rhs = rhs.at[:, :, 1].set(state.v * params.tpfac2)  # v
    rhs = rhs.at[:, :, 2].set(state.temperature * params.tpfac2)  # T
    rhs = rhs.at[:, :, 3].set(state.qv * params.tpfac2)  # qv
    rhs = rhs.at[:, :, 4].set(state.qc * params.tpfac2)  # qc
    rhs = rhs.at[:, :, 5].set(state.qi * params.tpfac2)  # qi
    rhs = rhs.at[:, :, 6].set(state.tke * params.tpfac2)  # TKE
    rhs = rhs.at[:, :, 7].set(state.thv_variance * params.tpfac2)  # thv_var
    
    return rhs


@jax.jit
def solve_tridiagonal_system(
    matrix_coeffs: jnp.ndarray,
    rhs_vectors: jnp.ndarray,
    variable_to_matrix: jnp.ndarray
) -> jnp.ndarray:
    """
    Solve the tridiagonal matrix system using Thomas algorithm.
    
    Args:
        matrix_coeffs: Coefficient matrices [ncol, nlev, 3, nmatrix]
        rhs_vectors: Right-hand side vectors [ncol, nlev, nvar]
        variable_to_matrix: Mapping from variables to matrix types
        
    Returns:
        Solution vectors [ncol, nlev, nvar]
    """
    ncol, nlev, nvar = rhs_vectors.shape
    solution = jnp.zeros_like(rhs_vectors)
    
    # Process each variable
    for ivar in range(nvar):
        matrix_idx = variable_to_matrix[ivar]
        
        # Get matrix coefficients for this variable
        a = matrix_coeffs[:, :, 0, matrix_idx]  # sub-diagonal
        b = matrix_coeffs[:, :, 1, matrix_idx]  # diagonal
        c = matrix_coeffs[:, :, 2, matrix_idx]  # super-diagonal
        d = rhs_vectors[:, :, ivar]             # RHS
        
        # Solve tridiagonal system for this variable
        solution = solution.at[:, :, ivar].set(
            solve_tridiagonal_single(a, b, c, d)
        )
    
    return solution


@jax.jit
def solve_tridiagonal_single(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    d: jnp.ndarray
) -> jnp.ndarray:
    """
    Solve a single tridiagonal system using Thomas algorithm.
    
    Args:
        a: Sub-diagonal [ncol, nlev]
        b: Diagonal [ncol, nlev]
        c: Super-diagonal [ncol, nlev]
        d: Right-hand side [ncol, nlev]
        
    Returns:
        Solution [ncol, nlev]
    """
    ncol, nlev = b.shape
    
    # Forward sweep (elimination)
    # Initialize first row
    cp_0 = c[:, 0] / b[:, 0]
    dp_0 = d[:, 0] / b[:, 0]
    
    # Remaining rows
    def forward_step(carry, inputs):
        cp_prev, dp_prev = carry
        a_i, b_i, c_i, d_i = inputs

        denom_i = b_i - a_i * cp_prev
        cp_i = c_i / denom_i
        dp_i = (d_i - a_i * dp_prev) / denom_i
        
        return (cp_i, dp_i), (cp_i, dp_i)
    
    _, forward_outputs = jax.lax.scan(
        forward_step,
        (cp_0, dp_0), # initial carry
        (a[:, 1:].T, b[:, 1:].T, c[:, 1:].T, d[:, 1:].T) # inputs
    )
    
    # Reconstruct cp and dp arrays
    cp_rest, dp_rest = forward_outputs
    cp = jnp.concatenate([cp_0[None, :], cp_rest], axis=0).T
    dp = jnp.concatenate([dp_0[None, :], dp_rest], axis=0).T

    # Back substitution
    x_last = dp[:, -1]
    def backward_step(carry, inputs):
        """Backward substitution step for scan."""
        x_next = carry
        cp_i, dp_i = inputs
        
        x_i = dp_i - cp_i * x_next
        
        return x_i, x_i
    
    # Prepare inputs for backward scan (reverse order, skip last element)
    backward_inputs = (cp[:, :-1].T[::-1], dp[:, :-1].T[::-1])

    _, backward_outputs = jax.lax.scan(backward_step, x_last, backward_inputs)
    
    # Reconstruct solution array (reverse the outputs and add last element)
    x_rest = backward_outputs[::-1]
    x = jnp.concatenate([x_rest, x_last[None, :]], axis=0).T
    
    return x


@jax.jit
def compute_tendencies_from_solution(
    solution: jnp.ndarray,
    state: VDiffState,
    params: VDiffParameters,
    dt: float
) -> VDiffTendencies:
    """
    Compute tendencies from the solution of the matrix system.
    
    Args:
        solution: Solution vectors [ncol, nlev, nvar]
        state: Original atmospheric state
        params: Vertical diffusion parameters
        dt: Time step [s]
        
    Returns:
        Tendencies for all variables
    """
    ncol, nlev = state.u.shape
    
    # Extract solutions for each variable
    u_new = solution[:, :, 0]
    v_new = solution[:, :, 1]
    t_new = solution[:, :, 2]
    qv_new = solution[:, :, 3]
    qc_new = solution[:, :, 4]
    qi_new = solution[:, :, 5]
    tke_new = solution[:, :, 6]
    thv_var_new = solution[:, :, 7]
    
    # Compute tendencies
    u_tend = (u_new - state.u) / dt
    v_tend = (v_new - state.v) / dt
    t_tend = (t_new - state.temperature) / dt
    qv_tend = (qv_new - state.qv) / dt
    qc_tend = (qc_new - state.qc) / dt
    qi_tend = (qi_new - state.qi) / dt
    tke_tend = (tke_new - state.tke) / dt
    thv_var_tend = (thv_var_new - state.thv_variance) / dt
    
    # Convert temperature tendency to heating rate
    heating_rate = t_tend * state.air_mass * PHYS_CONST.cp
    
    return VDiffTendencies(
        u_tendency=u_tend,
        v_tendency=v_tend,
        temperature_tendency=t_tend,
        heating_rate=heating_rate,
        qv_tendency=qv_tend,
        qc_tendency=qc_tend,
        qi_tendency=qi_tend,
        tke_tendency=tke_tend,
        thv_var_tendency=thv_var_tend
    )


@jax.jit
def vertical_diffusion_step(
    state: VDiffState,
    params: VDiffParameters,
    exchange_coeff_momentum: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    exchange_coeff_moisture: jnp.ndarray,
    dt: float,
    tke_exchange_coeff: jnp.ndarray = None
) -> VDiffTendencies:
    """
    Perform one vertical diffusion time step.
    
    Args:
        state: Atmospheric state
        params: Vertical diffusion parameters
        exchange_coeff_momentum: Momentum exchange coefficient
        exchange_coeff_heat: Heat exchange coefficient
        exchange_coeff_moisture: Moisture exchange coefficient
        dt: Time step [s]
        
    Returns:
        Tendencies for all variables
    """
    # Default TKE exchange coefficient if not provided
    if tke_exchange_coeff is None:
        tke_exchange_coeff = exchange_coeff_momentum
    
    # Set up matrix system
    matrix_system = setup_matrix_system(
        state, params, exchange_coeff_momentum, 
        exchange_coeff_heat, exchange_coeff_moisture, dt, tke_exchange_coeff
    )
    
    # Solve the system
    solution = solve_tridiagonal_system(
        matrix_system.matrix_coeffs,
        matrix_system.rhs_vectors,
        matrix_system.variable_to_matrix
    )
    
    # Compute tendencies
    tendencies = compute_tendencies_from_solution(
        solution, state, params, dt
    )
    
    return tendencies