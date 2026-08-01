"""Tridiagonal matrix solver for vertical diffusion.

This module implements the implicit tridiagonal matrix solver used in ICON's
vertical diffusion scheme, following the downward sweep/upward sweep approach.
"""

import jax
import jax.numpy as jnp

import jcm.constants as c
from .vertical_diffusion_types import (
    VDiffState, VDiffParameters, VDiffMatrixSystem, VDiffTendencies,
    VDiffSurfaceFluxes,
)


def _surface_air_density(state: VDiffState) -> jnp.ndarray:
    """ρ_s = p_half[K+1/2] / (Rd · T[K]) — surface air density (ncol,)."""
    return state.pressure_half[:, -1] / (c.rd * state.temperature[:, -1])


@jax.jit
def setup_matrix_system(
    state: VDiffState,
    params: VDiffParameters,
    exchange_coeff_momentum: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    exchange_coeff_moisture: jnp.ndarray,
    dt: float,
    tke_exchange_coeff: jnp.ndarray = None,
    surface_exchange: tuple = None,
    surface_target: tuple = None,
) -> VDiffMatrixSystem:
    """Set up the tridiagonal matrix system for vertical diffusion.

    Following ICON's mo_vdiff_solver.f90 and mo_turbulence_diag.f90:
    - The matrix coefficient is: K* = dt * tpfac1 * K * prefactor
    - where prefactor = rho / dz = p / (Tv * Rd * dz) at half levels
    - This gets divided by air_mass to give: K* / dm = dt * tpfac1 * K / dz²

    Surface boundary condition (ECHAM-faithful Robin row)
    -----------------------------------------------------
    When ``surface_exchange``/``surface_target`` are given, the bulk surface
    exchange enters the BOTTOM row of the momentum (u, v), heat (T) and
    moisture (qv) matrices exactly as ECHAM's ``zcfh_sfc·zqdp`` does in the
    Richtmyer–Morton bottom elimination (``mo_surface_ocean.f90:510-515``):

        k_sfc = dt · tpfac1 · ρ_s · C_grid · recip_air_mass[:, K]
        aa[:, K, diag, im] += k_sfc
        rhs[:, K, ivar]    += tpfac2 · k_sfc · X_s_eff

    with ``ρ_s = p_half[K+1/2]/(Rd·T_K)`` and ``C_grid`` the tile-collapsed
    exchange *velocity* [m/s] (CH·|U| etc.). This is the same dimensionless
    row entry as ECHAM's ``zcfh_sfc·zqdp = α·Δt·g·ρ_s·C/Δp_K``; the RHS
    constant is divided by α because the solver works in ``bb = X̂/α`` units
    (``X̂ = α·X_new + (1−α)·X_old``). The hydrometeor (qc/qi), TKE and
    thv_var matrices get NO surface term — matching ECHAM's bottom
    elimination 5.4, which handles only xl/xi with a zero surface
    coefficient (``vdiff.f90:933-941``).

    Args:
        state: Atmospheric state
        params: Vertical diffusion parameters
        exchange_coeff_momentum: Momentum exchange coefficient [m²/s]
        exchange_coeff_heat: Heat exchange coefficient [m²/s]
        exchange_coeff_moisture: Moisture exchange coefficient [m²/s]
        dt: Time step [s]
        tke_exchange_coeff: TKE exchange coefficient [m²/s]
        surface_exchange: Optional ``(C_m, C_h, C_q)`` grid-collapsed surface
            exchange velocities [m/s], each (ncol,). ``None`` keeps the
            legacy zero-flux (insulating, free-slip) bottom boundary.
        surface_target: Optional ``(u_s, v_s, T_s_eff, q_s_eff)`` surface
            target values, each (ncol,). ``T_s_eff`` must already include
            the ``−φ_K/cpd`` dry-static-energy compensation (the solver
            diffuses T, not s = cp·T + gz; see ``vertical_diffusion_column``).

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

    # Reciprocal air mass for matrix coefficients. ALL rows — including
    # moisture and hydrometeors — use the moist layer mass Δp/g, exactly
    # like ECHAM's single ``zqdp = 1/(paphm1(k+1)−paphm1(k))`` in
    # ``vdiff.f90``. jcm's specific humidity is per MOIST mass and every
    # column-budget in the model (convection, microphysics, the composed
    # water-closure test, the dycore itself) integrates with dp/g, so the
    # qv row must conserve the same measure: with a dry-air mass here the
    # solve conserved Σ dm_dry·dq instead, and the composed column budget
    # opened by O(internal transport × qv) — up to ~15 % of E in the
    # sharp post-convective-burst profiles (ICON uses dry mass because its
    # tracers are per dry mass; jcm's are not).
    recip_air_mass = 1.0 / state.air_mass

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
        matrix_coeffs, exchange_coeff_moisture, recip_air_mass, scaled_prefactor, 2
    )

    # Setup hydrometeor matrix (qc, qi, tracers)
    matrix_coeffs = setup_momentum_matrix_with_prefactor(
        matrix_coeffs, exchange_coeff_heat, recip_air_mass, scaled_prefactor, 3
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

    # Surface Robin term on the bottom row (see docstring). Only u, v, T,
    # qv couple to the surface; qc/qi/TKE/thv_var keep the zero-flux bottom.
    if surface_exchange is not None:
        c_mom, c_heat, c_moist = surface_exchange
        u_s, v_s, t_s_eff, q_s_eff = surface_target

        # k_sfc = dt·tpfac1·ρ_s·C_grid·recip_air_mass[K] for every row —
        # the same moist Δp/g measure as the interior rows (ECHAM zqdp),
        # so the delivered-E identity Σ (dp/g)·dq/dt == E is exact in the
        # model's own budget convention.
        rho_s = _surface_air_density(state)
        dt_tp1_rho = dt * params.tpfac1 * rho_s
        k_sfc_mom = dt_tp1_rho * c_mom * recip_air_mass[:, -1]
        k_sfc_heat = dt_tp1_rho * c_heat * recip_air_mass[:, -1]
        k_sfc_moist = dt_tp1_rho * c_moist * recip_air_mass[:, -1]

        # Bottom diagonals: ECHAM's "+ zcfhw*zqdp" inside zdiscw
        # (mo_surface_ocean.f90:510).
        matrix_coeffs = matrix_coeffs.at[:, -1, 1, 0].add(k_sfc_mom)
        matrix_coeffs = matrix_coeffs.at[:, -1, 1, 1].add(k_sfc_heat)
        matrix_coeffs = matrix_coeffs.at[:, -1, 1, 2].add(k_sfc_moist)

        # Bottom RHS: tpfac2·k_sfc·X_s — ECHAM's "+ zcfh_sfc·zqdp·X_s" term,
        # divided by α because rhs is loaded in X_old/α (bb) units.
        tpfac2 = params.tpfac2
        rhs_vectors = rhs_vectors.at[:, -1, 0].add(tpfac2 * k_sfc_mom * u_s)
        rhs_vectors = rhs_vectors.at[:, -1, 1].add(tpfac2 * k_sfc_mom * v_s)
        rhs_vectors = rhs_vectors.at[:, -1, 2].add(tpfac2 * k_sfc_heat * t_s_eff)
        rhs_vectors = rhs_vectors.at[:, -1, 3].add(tpfac2 * k_sfc_moist * q_s_eff)

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
def diagnose_surface_fluxes(
    solution: jnp.ndarray,
    state: VDiffState,
    params: VDiffParameters,
    surface_exchange: tuple,
    surface_target: tuple,
) -> VDiffSurfaceFluxes:
    """Diagnose the delivered surface fluxes from the implicit solution.

    Port of ECHAM's post-solve flux diagnosis (``mo_surface_ocean.f90:
    620-634``): the flux is evaluated at the α-weighted implicit bottom-level
    value ``X̂_K = tpfac1·bb_K`` the solver itself used, so reported flux ==
    delivered flux by construction:

        E  = ρ_s·C_q·(q_s_eff − X̂_K)         [kg/m²/s, positive up]
        SH = ρ_s·cpd·C_h·(T_s_eff − T̂_K)     [W/m²,   positive up]
        LH = alhc·E
        τ  = ρ_s·C_m·(Û_K − û_s)             [N/m², stress on the atmosphere]

    Implementation note: the fluxes are written in the algebraically
    equivalent form ``ρ_s·C·tpfac1·(tpfac2·X_s − bb_K)``. With ECHAM's exact
    ``tpfac2 = 1/tpfac1`` this IS the formula above; with the port's rounded
    defaults (0.667/0.333) this form is the one that keeps the ``pev_vdiff``
    column-budget identity ``Σ_k dm_k·dX_k/dt == flux`` (``vdiff.f90:
    1544-1551``) exact to round-off, because ``tpfac2·k_sfc·X_s`` is what the
    bottom RHS actually carried into the solve.
    """
    c_mom, c_heat, c_moist = surface_exchange
    u_s, v_s, t_s_eff, q_s_eff = surface_target

    rho_s = _surface_air_density(state)
    tp1 = params.tpfac1
    tp2 = params.tpfac2

    bb_u = solution[:, -1, 0]
    bb_v = solution[:, -1, 1]
    bb_t = solution[:, -1, 2]
    bb_qv = solution[:, -1, 3]

    evaporation = rho_s * c_moist * tp1 * (tp2 * q_s_eff - bb_qv)
    sensible_heat = rho_s * c.cpd * c_heat * tp1 * (tp2 * t_s_eff - bb_t)
    latent_heat = c.alhc * evaporation
    # Stress the atmosphere exerts (positive with the wind); the delivered
    # column momentum change is −τ (drag), matching the old surface-term
    # convention of publishing τ = ρ·C_M·u and applying −τ/(ρ·dz).
    stress_u = rho_s * c_mom * tp1 * (bb_u - tp2 * u_s)
    stress_v = rho_s * c_mom * tp1 * (bb_v - tp2 * v_s)

    return VDiffSurfaceFluxes(
        evaporation=evaporation,
        sensible_heat=sensible_heat,
        latent_heat=latent_heat,
        stress_u=stress_u,
        stress_v=stress_v,
    )


@jax.jit
def vertical_diffusion_step(
    state: VDiffState,
    params: VDiffParameters,
    exchange_coeff_momentum: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    exchange_coeff_moisture: jnp.ndarray,
    dt: float,
    tke_exchange_coeff: jnp.ndarray = None,
    surface_exchange: tuple = None,
    surface_target: tuple = None,
) -> tuple:
    """Perform one vertical diffusion time step.

    Args:
        state: Atmospheric state
        params: Vertical diffusion parameters
        exchange_coeff_momentum: Momentum exchange coefficient
        exchange_coeff_heat: Heat exchange coefficient
        exchange_coeff_moisture: Moisture exchange coefficient
        dt: Time step [s]
        tke_exchange_coeff: TKE exchange coefficient
        surface_exchange: Optional ``(C_m, C_h, C_q)`` surface exchange
            velocities [m/s] — enables the ECHAM Robin bottom row (see
            :func:`setup_matrix_system`). ``None`` keeps the legacy
            zero-flux bottom boundary.
        surface_target: Optional ``(u_s, v_s, T_s_eff, q_s_eff)`` targets.

    Returns:
        ``(tendencies, surface_fluxes)`` — tendencies for all variables and
        the delivered surface fluxes (zeros for the zero-flux boundary).

    """
    # Default TKE exchange coefficient if not provided
    if tke_exchange_coeff is None:
        tke_exchange_coeff = exchange_coeff_momentum

    # Set up matrix system
    matrix_system = setup_matrix_system(
        state, params, exchange_coeff_momentum,
        exchange_coeff_heat, exchange_coeff_moisture, dt, tke_exchange_coeff,
        surface_exchange=surface_exchange, surface_target=surface_target,
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

    # Diagnose the delivered surface fluxes from the implicit solution
    # (reported == delivered by construction; zero for the zero-flux BC).
    if surface_exchange is not None:
        surface_fluxes = diagnose_surface_fluxes(
            solution, state, params, surface_exchange, surface_target,
        )
    else:
        surface_fluxes = VDiffSurfaceFluxes.zeros(state.u.shape[0])

    return tendencies, surface_fluxes