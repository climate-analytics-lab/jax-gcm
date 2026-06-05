"""Tridiagonal matrix solver for vertical diffusion.

This module implements the implicit tridiagonal matrix solver used in ICON's
vertical diffusion scheme, following the downward sweep/upward sweep approach.
"""

import jax
import jax.numpy as jnp

import jcm.constants as c
from .vertical_diffusion_types import (
    VDiffState, VDiffParameters, VDiffMatrixSystem, VDiffTendencies
)


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
    """Set up the tridiagonal matrix system for vertical diffusion.

    Following ICON's mo_vdiff_solver.f90 and mo_turbulence_diag.f90:
    - The matrix coefficient is: K* = dt * tpfac1 * K * prefactor
    - where prefactor = rho / dz = p / (Tv * Rd * dz) at half levels
    - This gets divided by air_mass to give: K* / dm = dt * tpfac1 * K / dz²

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

    # Reciprocal air mass for matrix coefficients
    recip_air_mass = 1.0 / state.air_mass
    recip_dry_air_mass = 1.0 / state.dry_air_mass

    # Compute layer thickness dz at half levels (needed for prefactor)
    # dz_half[k] = height_full[k] - height_full[k+1] (distance between full levels)
    dz_half = jnp.diff(state.height_full, axis=1)  # (ncol, nlev-1)
    # Ensure positive and avoid division by zero (10m floor prevents
    # artificial prefactor inflation with thin uniform sigma layers)
    dz_half = jnp.maximum(jnp.abs(dz_half), 10.0)

    # Compute prefactor at half levels: pprfac = rho / dz = p / (Tv * Rd * dz)
    # We use pressure and virtual temperature at half levels (average of adjacent full levels)
    p_half = 0.5 * (state.pressure_full[:, :-1] + state.pressure_full[:, 1:])  # (ncol, nlev-1)
    t_half = 0.5 * (state.temperature[:, :-1] + state.temperature[:, 1:])  # Use T as proxy for Tv
    prefactor_half = p_half / (c.rd * t_half * dz_half)  # (ncol, nlev-1)

    # Time step factor
    dt_factor = dt * params.tpfac1

    # Combine dt_factor with prefactor for passing to matrix setup functions
    # The setup functions will apply: K_scaled = K * (dt_factor * prefactor)
    scaled_prefactor = dt_factor * prefactor_half  # (ncol, nlev-1)

    # Setup momentum matrix (u, v)
    matrix_coeffs = setup_momentum_matrix_with_prefactor(
        matrix_coeffs, exchange_coeff_momentum, recip_air_mass, scaled_prefactor, 0
    )

    # Setup heat matrix (T)
    matrix_coeffs = setup_momentum_matrix_with_prefactor(
        matrix_coeffs, exchange_coeff_heat, recip_air_mass, scaled_prefactor, 1
    )

    # Setup moisture matrix (qv)
    matrix_coeffs = setup_momentum_matrix_with_prefactor(
        matrix_coeffs, exchange_coeff_moisture, recip_dry_air_mass, scaled_prefactor, 2
    )

    # Setup hydrometeor matrix (qc, qi, tracers)
    matrix_coeffs = setup_momentum_matrix_with_prefactor(
        matrix_coeffs, exchange_coeff_heat, recip_dry_air_mass, scaled_prefactor, 3
    )

    # Setup TKE matrix (use TKE exchange coefficient)
    matrix_coeffs = setup_momentum_matrix_with_prefactor(
        matrix_coeffs, tke_exchange_coeff, recip_air_mass, scaled_prefactor, 4
    )

    # Setup theta_v variance matrix
    matrix_coeffs = setup_momentum_matrix_with_prefactor(
        matrix_coeffs, exchange_coeff_heat, recip_air_mass, scaled_prefactor, 5
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
def setup_momentum_matrix_with_prefactor(
    matrix_coeffs: jnp.ndarray,
    exchange_coeff: jnp.ndarray,
    recip_air_mass: jnp.ndarray,
    scaled_prefactor: jnp.ndarray,
    matrix_idx: int
) -> jnp.ndarray:
    """Set up tridiagonal matrix for vertical diffusion with proper prefactor.

    Following ICON's mo_vdiff_solver.f90:
    - zkstar = pprfac * pcfm (scaled exchange coefficient at half levels)
    - aa(jc,jk,1,im) = -zkstar(jc,jk-1) * prmairm(jc,jk)  (sub-diagonal)
    - aa(jc,jk,3,im) = -zkstar(jc,jk)   * prmairm(jc,jk)  (super-diagonal)
    - aa(jc,jk,2,im) = 1 - aa(jk,1) - aa(jk,3)  (diagonal)

    Args:
        matrix_coeffs: Matrix coefficients array [ncol, nlev, 3, nmatrix]
        exchange_coeff: Exchange coefficient [m²/s] (ncol, nlev)
        recip_air_mass: Reciprocal air mass [m²/kg] (ncol, nlev)
        scaled_prefactor: dt * tpfac1 * (rho/dz) at half levels (ncol, nlev-1)
        matrix_idx: Index of the matrix type

    Returns:
        Updated matrix coefficients

    """
    ncol, nlev = exchange_coeff.shape

    # Exchange coefficient at half levels (between full levels)
    # k_half[k] is at interface between full levels k and k+1
    k_half = 0.5 * (exchange_coeff[:, :-1] + exchange_coeff[:, 1:])  # (ncol, nlev-1)

    # Scaled exchange coefficients: K* = K * (dt * tpfac1 * rho/dz)
    k_scaled = k_half * scaled_prefactor  # (ncol, nlev-1)

    # Build tridiagonal matrix
    # Note: In Fortran, k_half has indices [itop:klev] where klev is surface
    # Here we have k_scaled with shape (ncol, nlev-1) for interfaces 0..nlev-2

    # Sub-diagonal: aa(jk, 1) = -zkstar(jk-1) * recip_air_mass(jk)
    # This connects level jk to level jk-1 (above)
    # For jk=1..nlev-1, use k_scaled indices 0..nlev-2
    sub_diagonal_vals = -k_scaled * recip_air_mass[:, 1:]  # shape: [ncol, nlev-1]
    matrix_coeffs = matrix_coeffs.at[:, 1:, 0, matrix_idx].set(sub_diagonal_vals)

    # Super-diagonal: aa(jk, 3) = -zkstar(jk) * recip_air_mass(jk)
    # This connects level jk to level jk+1 (below)
    # For jk=0..nlev-2, use k_scaled indices 0..nlev-2
    super_diagonal_vals = -k_scaled * recip_air_mass[:, :-1]  # shape: [ncol, nlev-1]
    matrix_coeffs = matrix_coeffs.at[:, :-1, 2, matrix_idx].set(super_diagonal_vals)

    # Diagonal: aa(jk, 2) = 1 - aa(jk, 1) - aa(jk, 3)
    # Need contributions from both sub and super diagonals

    # Contribution from super-diagonal (for level jk, this is -aa(jk, 3))
    super_contrib = jnp.concatenate([
        -super_diagonal_vals,
        jnp.zeros((ncol, 1))  # Level nlev-1 has no super-diagonal contribution
    ], axis=1)

    # Contribution from sub-diagonal (for level jk, this is -aa(jk, 1))
    sub_contrib = jnp.concatenate([
        jnp.zeros((ncol, 1)),  # Level 0 has no sub-diagonal contribution
        -sub_diagonal_vals
    ], axis=1)

    diagonal_vals = 1.0 + super_contrib + sub_contrib
    matrix_coeffs = matrix_coeffs.at[:, :, 1, matrix_idx].set(diagonal_vals)

    return matrix_coeffs


@jax.jit
def setup_momentum_matrix(
    matrix_coeffs: jnp.ndarray,
    exchange_coeff: jnp.ndarray,
    recip_air_mass: jnp.ndarray,
    dt_factor: float,
    matrix_idx: int
) -> jnp.ndarray:
    """Set up tridiagonal matrix for momentum equations (legacy version without prefactor)."""
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
    """Set up right-hand side vectors for the linear system.

    Following ICON's semi-implicit time stepping (mo_vdiff_solver.f90):
    - Matrix equation: (I - dt*tpfac1*L) * bb = tpfac2 * X_old
    - New value: X_new = bb + tpfac3 * X_old
    - where tpfac1=1.5, tpfac2=1/tpfac1=0.667, tpfac3=1-tpfac2=0.333

    The tpfac2 factor scales the RHS to achieve the semi-implicit scheme.
    """
    ncol, nlev = state.u.shape
    # Fixed number of variables: u, v, T, qv, qc, qi, TKE, thv_var
    rhs = jnp.zeros((ncol, nlev, 8))

    # Apply tpfac2 scaling to RHS as in ICON
    tpfac2 = params.tpfac2

    rhs = rhs.at[:, :, 0].set(tpfac2 * state.u)  # u
    rhs = rhs.at[:, :, 1].set(tpfac2 * state.v)  # v
    rhs = rhs.at[:, :, 2].set(tpfac2 * state.temperature)  # T
    rhs = rhs.at[:, :, 3].set(tpfac2 * state.qv)  # qv
    rhs = rhs.at[:, :, 4].set(tpfac2 * state.qc)  # qc
    rhs = rhs.at[:, :, 5].set(tpfac2 * state.qi)  # qi
    rhs = rhs.at[:, :, 6].set(tpfac2 * state.tke)  # TKE
    rhs = rhs.at[:, :, 7].set(tpfac2 * state.thv_variance)  # thv_var

    return rhs


@jax.jit
def solve_tridiagonal_system(
    matrix_coeffs: jnp.ndarray,
    rhs_vectors: jnp.ndarray,
    variable_to_matrix: jnp.ndarray
) -> jnp.ndarray:
    """Solve the tridiagonal matrix system using Thomas algorithm.
    
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
    """Solve a single tridiagonal system using Thomas algorithm.
    
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
    # Guard pivots from underflow to prevent NaN with ill-conditioned matrices.
    # The previous form ``jnp.sign(x) * 1e-20 + 1e-20`` returned exactly 0
    # when ``x`` was a tiny *negative* number (sign(-eps)*1e-20 + 1e-20 ==
    # -1e-20 + 1e-20 == 0) — so subsequent ``/_safe(x)`` divisions produced
    # inf, which after a few back-substitutions explodes the solution by
    # ~18 orders of magnitude. The new form preserves sign and is never
    # exactly zero.
    def _safe(x):
        eps = 1e-20
        return jnp.where(
            jnp.abs(x) > eps,
            x,
            jnp.where(x < 0, -eps, eps),
        )

    # Initialize first row
    cp_0 = c[:, 0] / _safe(b[:, 0])
    dp_0 = d[:, 0] / _safe(b[:, 0])

    # Remaining rows
    def forward_step(carry, inputs):
        cp_prev, dp_prev = carry
        a_i, b_i, c_i, d_i = inputs

        denom_i = _safe(b_i - a_i * cp_prev)
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
    """Compute tendencies from the solution of the matrix system.

    Following ICON's semi-implicit time stepping (mo_vdiff_solver.f90:840-851):
    - bb is the matrix solution (solution of (I - dt*tpfac1*L) * bb = tpfac2 * X_old)
    - X_new = bb + tpfac3 * X_old
    - tendency = (X_new - X_old) / dt = (bb + tpfac3 * X_old - X_old) / dt
                                      = (bb - tpfac2 * X_old) / dt  (since tpfac2 + tpfac3 = 1)

    Args:
        solution: Solution vectors [ncol, nlev, nvar] (this is bb)
        state: Original atmospheric state
        params: Vertical diffusion parameters
        dt: Time step [s]

    Returns:
        Tendencies for all variables

    """
    ncol, nlev = state.u.shape

    # Extract solutions for each variable (these are bb values)
    bb_u = solution[:, :, 0]
    bb_v = solution[:, :, 1]
    bb_t = solution[:, :, 2]
    bb_qv = solution[:, :, 3]
    bb_qc = solution[:, :, 4]
    bb_qi = solution[:, :, 5]
    bb_tke = solution[:, :, 6]
    bb_thv_var = solution[:, :, 7]

    # Reconstruct new values: X_new = bb + tpfac3 * X_old
    tpfac3 = params.tpfac3
    u_new = bb_u + tpfac3 * state.u
    v_new = bb_v + tpfac3 * state.v
    t_new = bb_t + tpfac3 * state.temperature
    qv_new = bb_qv + tpfac3 * state.qv
    qc_new = bb_qc + tpfac3 * state.qc
    qi_new = bb_qi + tpfac3 * state.qi
    tke_new = bb_tke + tpfac3 * state.tke
    thv_var_new = bb_thv_var + tpfac3 * state.thv_variance

    # Compute tendencies: (X_new - X_old) / dt
    u_tend = (u_new - state.u) / dt
    v_tend = (v_new - state.v) / dt
    t_tend = (t_new - state.temperature) / dt
    qv_tend = (qv_new - state.qv) / dt
    qc_tend = (qc_new - state.qc) / dt
    qi_tend = (qi_new - state.qi) / dt
    tke_tend = (tke_new - state.tke) / dt
    thv_var_tend = (thv_var_new - state.thv_variance) / dt

    # Convert temperature tendency to heating rate
    heating_rate = t_tend * state.air_mass * c.cpd

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
    """Perform one vertical diffusion time step.
    
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