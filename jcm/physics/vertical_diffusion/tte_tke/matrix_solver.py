"""Tridiagonal matrix solver for vertical diffusion.

This module implements the implicit tridiagonal matrix solver used in ICON's
vertical diffusion scheme, following the downward sweep/upward sweep approach.
"""

import jax
import jax.numpy as jnp

import jcm.constants as c
from jcm.physics.surface.echam import jsbach_land
from .vertical_diffusion_types import (
    VDiffState, VDiffParameters, VDiffMatrixSystem, VDiffTendencies,
    VDiffSurfaceFluxes, SurfaceTiles, LandBalanceOutputs,
)

#: Tile order of the TTE-TKE surface: open water, sea ice, land.
LAND_TILE = 2


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
    surface_momentum: tuple = None,
) -> VDiffMatrixSystem:
    """Set up the tridiagonal matrix system for vertical diffusion.

    Following ICON's mo_vdiff_solver.f90 and mo_turbulence_diag.f90:
    - The matrix coefficient is: K* = dt * tpfac1 * K * prefactor
    - where prefactor = rho / dz = p / (Tv * Rd * dz) at half levels
    - This gets divided by air_mass to give: K* / dm = dt * tpfac1 * K / dz²

    Surface boundary condition
    --------------------------
    Momentum: with ``surface_momentum = (C_m, u_s, v_s)`` the bulk drag enters
    the BOTTOM row of the u/v matrix as a Robin term, exactly as ECHAM's
    fraction-weighted ``cdum`` does (``mo_surface.f90:1205-1219`` →
    ``vdiff.f90:1099-1100``)::

        k_sfc = dt · tpfac1 · ρ_s · C_m · recip_air_mass[:, K]
        aa[:, K, diag, mom] += k_sfc
        rhs[:, K, u/v]      += tpfac2 · k_sfc · u_s/v_s

    with ``ρ_s = p_half[K+1/2]/(Rd·T_K)``: the dimensionless row entry of
    ECHAM's ``zcfm·zqdp``, the RHS constant divided by α because the solver
    works in ``bb = X̂/α`` units. Heat and moisture get no surface term here:
    they couple tile by tile through the Richtmyer–Morton relations in
    :func:`vertical_diffusion_step`. The hydrometeor (qc/qi), TKE and thv_var
    matrices get no surface term at all, matching ECHAM's bottom elimination
    5.4, which handles only xl/xi with a zero surface coefficient
    (``vdiff.f90:933-941``).

    Args:
        state: Atmospheric state
        params: Vertical diffusion parameters
        exchange_coeff_momentum: Momentum exchange coefficient [m²/s]
        exchange_coeff_heat: Heat exchange coefficient [m²/s]
        exchange_coeff_moisture: Moisture exchange coefficient [m²/s]
        dt: Time step [s]
        tke_exchange_coeff: TKE exchange coefficient [m²/s]
        surface_momentum: Optional ``(C_m, u_s, v_s)``: the fraction-weighted
            momentum exchange velocity [m/s] and the surface velocity, each
            (ncol,). ``None`` keeps the free-slip bottom boundary.

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

    # Momentum Robin term on the bottom row (see docstring).
    if surface_momentum is not None:
        c_mom, u_s, v_s = surface_momentum
        rho_s = _surface_air_density(state)
        k_sfc_mom = dt * params.tpfac1 * rho_s * c_mom * recip_air_mass[:, -1]
        matrix_coeffs = matrix_coeffs.at[:, -1, 1, 0].add(k_sfc_mom)
        tpfac2 = params.tpfac2
        rhs_vectors = rhs_vectors.at[:, -1, 0].add(tpfac2 * k_sfc_mom * u_s)
        rhs_vectors = rhs_vectors.at[:, -1, 1].add(tpfac2 * k_sfc_mom * v_s)

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
    - where tpfac1=1.5, tpfac2=1/tpfac1, tpfac3=1-tpfac2 (ECHAM's cvdifts)

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


def _safe_pivot(x):
    """Keep a pivot's sign and keep it away from zero.

    ``jnp.sign(x)·eps + eps`` returned exactly 0 for a tiny negative pivot,
    and the following divisions produced inf that grew ~18 orders of magnitude
    in back-substitution; this form is never zero.
    """
    eps = 1e-20
    return jnp.where(jnp.abs(x) > eps, x, jnp.where(x < 0, -eps, eps))


def forward_sweep(a, b, c, d):
    """Thomas elimination from the top: ``(cp, dp, pivot)``, each [ncol, nlev].

    Row ``k`` is reduced to ``x_k = dp_k − cp_k·x_{k+1}``. ``pivot_k`` is the
    eliminated diagonal ``b_k − a_k·cp_{k−1}``, so at the bottom row
    ``pivot_K`` and ``dp_K·pivot_K`` are ECHAM's eliminated ``1 + zfac·(1 −
    zebsh_{K−1})`` and ``ztdif_K + zfac·ztdif_{K−1}`` (``vdiff.f90:887-931``,
    with ``cp = −zebsh``), the inputs of the Richtmyer–Morton coefficients.
    """
    pivot_0 = _safe_pivot(b[:, 0])
    cp_0 = c[:, 0] / pivot_0
    dp_0 = d[:, 0] / pivot_0

    def step(carry, inputs):
        cp_prev, dp_prev = carry
        a_i, b_i, c_i, d_i = inputs
        pivot_i = _safe_pivot(b_i - a_i * cp_prev)
        cp_i = c_i / pivot_i
        dp_i = (d_i - a_i * dp_prev) / pivot_i
        return (cp_i, dp_i), (cp_i, dp_i, pivot_i)

    _, (cp_r, dp_r, pv_r) = jax.lax.scan(
        step, (cp_0, dp_0), (a[:, 1:].T, b[:, 1:].T, c[:, 1:].T, d[:, 1:].T))
    cp = jnp.concatenate([cp_0[None, :], cp_r], axis=0).T
    dp = jnp.concatenate([dp_0[None, :], dp_r], axis=0).T
    pivot = jnp.concatenate([pivot_0[None, :], pv_r], axis=0).T
    return cp, dp, pivot


def back_substitute(cp, dp, x_bottom):
    """Upward sweep ``x_k = dp_k − cp_k·x_{k+1}`` from a given bottom value."""
    def step(x_next, inputs):
        cp_i, dp_i = inputs
        x_i = dp_i - cp_i * x_next
        return x_i, x_i

    _, x_rest = jax.lax.scan(step, x_bottom, (cp[:, :-1].T[::-1], dp[:, :-1].T[::-1]))
    return jnp.concatenate([x_rest[::-1], x_bottom[None, :]], axis=0).T


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
    cp, dp, _ = forward_sweep(a, b, c, d)
    return back_substitute(cp, dp, dp[:, -1])


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


def _momentum_stress(solution, state, params, surface_momentum):
    """Stress the atmosphere exerts on the surface, from the implicit solution.

    ``τ = ρ_s·C_m·tpfac1·(bb_K − tpfac2·u_s)`` — ECHAM's ``zcfm·zudif``
    diagnosis (``mo_surface_land.f90::update_stress_land``) on the α-weighted
    bottom value the solver used, so the stress equals the column's momentum
    change (positive with the wind; the column receives −τ).
    """
    c_mom, u_s, v_s = surface_momentum
    rho_s = _surface_air_density(state)
    tp1, tp2 = params.tpfac1, params.tpfac2
    stress_u = rho_s * c_mom * tp1 * (solution[:, -1, 0] - tp2 * u_s)
    stress_v = rho_s * c_mom * tp1 * (solution[:, -1, 1] - tp2 * v_s)
    return stress_u, stress_v


def couple_surface_tiles(state: VDiffState, params: VDiffParameters,
                         matrix_system: VDiffMatrixSystem, dt: float,
                         tiles: SurfaceTiles):
    """Heat and moisture coupled to the surface tile by tile, as ECHAM does.

    ECHAM eliminates the column from the top once (``vdiff.f90:887-931``),
    after which the lowest level obeys, for each tile ``t`` on its own exchange
    coefficient ``k_t = Δt·α·ρ_s·C_t·g/Δp_K`` (``richtmyer_land``, ``_ocean``,
    ``_ice``; :func:`~jcm.physics.surface.echam.jsbach_land.richtmyer_morton`)::

        X̂_K,t = E_t·X̂_s,t + F_t

    The prescribed tiles (open water at the SST, sea ice at ``min(SST,
    ctfreez)``) give ``X̂_K,t`` directly. The land tile, when
    ``tiles.land`` is given, first solves its skin energy balance against its
    own ``E``/``F`` (``update_surfacetemp``; ``mo_soil.f90:1843-1853``), with the
    snow-melt cap of ``update_soil`` 1859-1863. The column's bottom value is
    the fraction-weighted blend ``bb_K = tpfac2·Σ_t f_t·X̂_K,t``
    (``blend_zq_zt``), and back-substitution completes the solve.

    Each tile's flux is evaluated against its OWN lowest-level value
    (``postproc_ocean``/``_ice``, ``update_soil`` 1892-1911); because the
    eliminated bottom row then reads ``D·bb_K = R + tpfac2·Σ_t f_t·k_t·(X̂_s,t −
    X̂_K,t)``, the grid-mean flux is exactly what the column receives (the
    ``pev_vdiff`` identity) when ``tpfac1·tpfac2 = 1``. Heat is carried as
    ``T`` with the surface value ``T_s − φ_K/c_pd`` (the solver diffuses T,
    not s = c_p·T + φ; the shift makes the bottom exchange the dry-static-energy
    flux), so the land balance reads ECHAM's dry static energies as
    ``s_s = c_pd·T̂_s`` and ``s_K = E·s_s + c_pd·F + (1 − E)·φ_K``.

    Returns:
        ``(T_solution, q_solution, VDiffSurfaceFluxes-without-stress parts,
        land outputs or None)``: the bb-unit solutions ``(ncol, nlev)``, a dict
        of grid fluxes, and :class:`LandBalanceOutputs`.

    """
    tp1, tp2, tp3 = params.tpfac1, params.tpfac2, params.tpfac3
    mc, rhs = matrix_system.matrix_coeffs, matrix_system.rhs_vectors

    def eliminate(ivar, imat):
        cp, dp, pivot = forward_sweep(mc[:, :, 0, imat], mc[:, :, 1, imat],
                                      mc[:, :, 2, imat], rhs[:, :, ivar])
        return cp, dp, pivot[:, -1], dp[:, -1] * pivot[:, -1]

    cp_t, dp_t, den_t, r_t = eliminate(2, 1)
    cp_q, dp_q, den_q, r_q = eliminate(3, 2)

    rho_s = _surface_air_density(state)
    k_scale = (dt * tp1 * rho_s / state.air_mass[:, -1])[:, None]
    k_h = k_scale * tiles.exchange_heat
    k_q = k_scale * tiles.exchange_moisture
    en, fn, eq, fq = jsbach_land.richtmyer_morton(
        den_t[:, None], den_q[:, None], r_t[:, None], r_q[:, None], k_h, k_q,
        tiles.cair, tiles.csat, tp1)

    cpd = c.cpd
    phi_k = c.grav * (state.height_full[:, -1] - state.height_half[:, -1])
    x_s = tiles.temperature - (phi_k / cpd)[:, None]
    q_s = tiles.saturation_humidity

    land = tiles.land
    if land is not None:
        il = LAND_TILE
        t_old = land.temperature
        rn_old = (land.net_shortwave + land.emissivity * land.longwave_down
                  - land.emissivity * c.sbc * t_old ** 4)
        s_hat = jsbach_land.update_surfacetemp(
            cpd, en[:, il], cpd * fn[:, il] + (1.0 - en[:, il]) * phi_k, eq[:, il], fq[:, il],
            cpd * t_old, q_s[:, il], land.saturation_slope, rn_old,
            land.conductance * (land.soil_temperature - t_old),
            rho_s * tiles.exchange_heat[:, il], tiles.cair[:, il], tiles.csat[:, il],
            tiles.sublimation_fraction[:, il], land.heat_capacity + dt * land.conductance,
            dt, tp1, land.emissivity)
        t_hat = s_hat / cpd
        # ECHAM forms the new (unfiltered) temperature from the uncapped ŝ and
        # then melts the excess, but evaluates the fluxes at the capped ŝ.
        t_hat = jsbach_land.melt_cap(t_hat, land.melt_capped, land.params)
        t_new = jsbach_land.melt_cap(tp2 * s_hat / cpd + tp3 * t_old,
                                     land.melt_capped, land.params)
        q_hat = q_s[:, il] + land.saturation_slope * (t_hat - t_old)
        # The land constants are float64 leaves under x64; the tiles keep the
        # state's precision.
        x_s = x_s.at[:, il].set((t_hat - phi_k / cpd).astype(x_s.dtype))
        q_s = q_s.at[:, il].set(q_hat.astype(q_s.dtype))

    x_k = en * x_s + fn                      # X̂_K per tile (update_land)
    q_k = eq * q_s + fq
    frac = tiles.fraction
    sol_t = back_substitute(cp_t, dp_t, tp2 * jnp.sum(frac * x_k, axis=1))
    sol_q = back_substitute(cp_q, dp_q, tp2 * jnp.sum(frac * q_k, axis=1))

    rho = rho_s[:, None]
    sh_t = rho * cpd * tiles.exchange_heat * (x_s - x_k)
    e_t = rho * tiles.exchange_moisture * (tiles.csat * q_s - tiles.cair * q_k)
    e_pot_t = rho * tiles.exchange_moisture * (q_s - q_k)
    lh_t = c.alhc * e_t + (c.alhs - c.alhc) * tiles.sublimation_fraction * e_pot_t
    fluxes = dict(
        evaporation=jnp.sum(frac * e_t, axis=1),
        sensible_heat=jnp.sum(frac * sh_t, axis=1),
        latent_heat=jnp.sum(frac * lh_t, axis=1),
    )

    land_out = None
    if land is not None:
        rn = rn_old - 4.0 * land.emissivity * c.sbc * t_old ** 3 * (t_hat - t_old)
        ground = land.conductance * (t_new - land.soil_temperature)
        storage = land.heat_capacity * (t_new - t_old) / dt
        sh_l, lh_l = sh_t[:, LAND_TILE], lh_t[:, LAND_TILE]
        land_out = LandBalanceOutputs(
            temperature=t_new,
            ground_heat_flux=ground,
            net_radiation=rn,
            sensible_heat_flux=sh_l,
            latent_heat_flux=lh_l,
            evaporation=e_t[:, LAND_TILE],
            heat_storage=storage,
            melt_heat_flux=rn - sh_l - lh_l - ground - storage,
        )
    return sol_t, sol_q, fluxes, land_out


@jax.jit
def vertical_diffusion_step(
    state: VDiffState,
    params: VDiffParameters,
    exchange_coeff_momentum: jnp.ndarray,
    exchange_coeff_heat: jnp.ndarray,
    exchange_coeff_moisture: jnp.ndarray,
    dt: float,
    tke_exchange_coeff: jnp.ndarray = None,
    surface_momentum: tuple = None,
    surface_tiles: SurfaceTiles = None,
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
        surface_momentum: Optional ``(C_m, u_s, v_s)``: the momentum Robin row
            (see :func:`setup_matrix_system`). ``None``: free slip.
        surface_tiles: Optional :class:`SurfaceTiles`: heat and moisture
            coupled tile by tile (:func:`couple_surface_tiles`). ``None``:
            insulating, zero-flux bottom.

    Returns:
        ``(tendencies, surface_fluxes, land)`` — tendencies for all variables,
        the delivered grid-mean surface fluxes (zero for the zero-flux
        boundary) and the land tile's :class:`LandBalanceOutputs` (``None``
        unless ``surface_tiles.land`` is given).

    """
    # Default TKE exchange coefficient if not provided
    if tke_exchange_coeff is None:
        tke_exchange_coeff = exchange_coeff_momentum

    matrix_system = setup_matrix_system(
        state, params, exchange_coeff_momentum,
        exchange_coeff_heat, exchange_coeff_moisture, dt, tke_exchange_coeff,
        surface_momentum=surface_momentum,
    )
    solution = solve_tridiagonal_system(
        matrix_system.matrix_coeffs,
        matrix_system.rhs_vectors,
        matrix_system.variable_to_matrix
    )

    ncol = state.u.shape[0]
    zero = jnp.zeros(ncol)
    fluxes = dict(evaporation=zero, sensible_heat=zero, latent_heat=zero)
    land = None
    if surface_tiles is not None:
        sol_t, sol_q, fluxes, land = couple_surface_tiles(
            state, params, matrix_system, dt, surface_tiles)
        solution = solution.at[:, :, 2].set(sol_t).at[:, :, 3].set(sol_q)
    stress_u = stress_v = zero
    if surface_momentum is not None:
        stress_u, stress_v = _momentum_stress(solution, state, params, surface_momentum)

    tendencies = compute_tendencies_from_solution(solution, state, params, dt)
    surface_fluxes = VDiffSurfaceFluxes(stress_u=stress_u, stress_v=stress_v, **fluxes)
    return tendencies, surface_fluxes, land
