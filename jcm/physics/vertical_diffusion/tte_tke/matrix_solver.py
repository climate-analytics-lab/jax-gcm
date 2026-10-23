"""Tridiagonal matrix solver for vertical diffusion.

The implicit column solve of ECHAM6.3's ``vdiff.f90`` (sections 3.8, 4 and 5):
full-level variables exchange across the interior interfaces with the
coefficients of :mod:`.closure`, TKE and the θ_v variance on the interface rows,
and heat and moisture couple to the surface tiles through the Richtmyer–Morton
relations.
"""

import jax
import jax.numpy as jnp

import jcm.constants as c
from jcm.physics.surface.echam import jsbach_land
from jcm.physics.thermodynamics import jsbach_surface_humidity_boundary
from .closure import geopotential_above_surface
from .vertical_diffusion_types import (
    VDiffState, VDiffParameters, VDiffMatrixSystem, VDiffTendencies,
    VDiffSurfaceFluxes, SurfaceTiles, LandBalanceOutputs, InterfaceConductances,
)

#: Tile order of the TTE-TKE surface: open water, sea ice, land.
LAND_TILE = 2


def _surface_air_density(state: VDiffState) -> jnp.ndarray:
    """ρ_s = p_s / (R_d · T_v,K) — the surface layer's air density (ncol,).

    ECHAM's ``zcons = zcons12·paphm1/(ptm1·(1 + vtmpc1·pqm1 − zx))`` of the
    surface tiles (``mo_surface_land.f90::precalc_land`` and the ocean and ice
    analogues): the lowest level's virtual temperature, condensate loading
    included, at the surface pressure.
    """
    tv = state.temperature[:, -1] * (
        1.0 + c.vtmpc1 * state.qv[:, -1] - (state.qc[:, -1] + state.qi[:, -1]))
    return state.pressure_half[:, -1] / (c.rd * tv)


def surface_geopotential_offset(state: VDiffState) -> jnp.ndarray:
    """Geopotential above the surface ``φ − φ_s`` [m²/s²] (ncol, nlev).

    The ``pgeom1`` the closure uses (:func:`.closure.geopotential_above_surface`),
    the frame in which the surface's dry static energy is ``c_pd·T_s``.
    """
    tv = state.temperature * (1.0 + c.vtmpc1 * state.qv - (state.qc + state.qi))
    return geopotential_above_surface(
        state.geopotential.T, tv.T, state.pressure_half.T).T


def heat_variable(state: VDiffState) -> jnp.ndarray:
    """Return the heat variable the solve diffuses, ``s/c_pd`` [K].

    ECHAM diffuses the dry static energy ``zcptgz = φ + c_pd·T·(1 +
    vtmpc2·q)`` (``vdiff.f90`` l.669-670, 890), with ``φ`` above the surface,
    so a well-mixed layer has uniform ``s``, not uniform ``T``, and the surface
    exchanges ``s_s − s_K`` with ECHAM's surface values ``zcptw``/``zcpti``/
    ``zcptl`` (:func:`surface_heat_variable`). It is carried here divided by
    ``c_pd``, in kelvin.
    """
    return (state.temperature * (1.0 + c.vtmpc2 * state.qv)
            + surface_geopotential_offset(state) / c.cpd)


def surface_heat_variable(temperature, surface_humidity):
    """``s_s/c_pd = T_s·(1 + vtmpc2·q_s)`` of a surface tile [K].

    ECHAM's ``zcptw``/``zcpti``/``zcptl`` (``precalc_ocean``/``_ice``/
    ``_land``) over ``c_pd``: the tile's dry static energy, its geopotential
    zero, with the surface humidity the tile exchanges (``csat·q_s + (1 −
    cair)·q_a`` over land).
    """
    return temperature * (1.0 + c.vtmpc2 * surface_humidity)


@jax.jit
def setup_matrix_system(
    state: VDiffState,
    params: VDiffParameters,
    conductances: InterfaceConductances,
    dt: float,
    surface_momentum: tuple = None,
) -> VDiffMatrixSystem:
    """Set up the tridiagonal systems of ECHAM's implicit vertical diffusion.

    Full-level variables (u, v, T, q, qc, qi; ``vdiff.f90`` sections 4 and 5)
    exchange across the interior interfaces. With ``G_k`` the conductance
    ``g·ρ·K/Δz`` of the interface between full levels ``k`` and ``k + 1``
    (ECHAM's ``zcf/(cvdifts·Δt)``) and ``Δp_k`` the pressure thickness of
    layer ``k`` (``zqdp = 1/Δp_k``), row ``k`` couples to its neighbours with
    ``α·Δt·G_{k−1}/Δp_k`` and ``α·Δt·G_k/Δp_k`` (``α = tpfac1``). Momentum uses
    ``zcfm``; heat, moisture and the hydrometeors ``zcfh`` (ECHAM's ``zcfv``
    is formed but never used, l.854, 901-941).

    TKE and the θ_v variance (section 3.8, l.1235-1300) live on the interface
    rows of :class:`~.vertical_diffusion_types.VDiffState`: row ``k`` is the
    interface below full level ``k``, the last row the surface layer. Row
    ``k < nlev − 1`` has the pressure thickness between full levels ``k``
    and ``k + 1`` (``papm1(k+1) − papm1(k)``) and couples to the rows above
    and below with the mean of the two rows' transport coefficients
    (``(zcdum(k) + zcdum(k+1))/2``); the last row is the surface value,
    a Dirichlet boundary the row above it reads (l.1276-1293).

    Surface boundary condition
    --------------------------
    Momentum: with ``surface_momentum = (C_m, u_s, v_s)`` the bulk drag enters
    the BOTTOM row of the u/v matrix as a Robin term, exactly as ECHAM's
    fraction-weighted ``cdum`` does (``mo_surface.f90:1205-1219`` →
    ``vdiff.f90:1099-1100``)::

        k_sfc = dt · tpfac1 · ρ_s · C_m · g / Δp_K
        aa[:, K, diag, mom] += k_sfc
        rhs[:, K, u/v]      += tpfac2 · k_sfc · u_s/v_s

    with ``ρ_s = p_s/(R_d·T_v,K)``: the dimensionless row entry of ECHAM's
    ``zcfm·zqdp``, the RHS constant divided by α because the solver works in
    ``bb = X̂/α`` units. Heat and moisture get no surface term here: they
    couple tile by tile through the Richtmyer–Morton relations in
    :func:`vertical_diffusion_step`. The hydrometeors get none at all,
    matching ECHAM's bottom elimination 5.4, which handles only xl/xi with a
    zero surface coefficient (``vdiff.f90:933-941``).

    Args:
        state: Atmospheric state; ``tke``/``thv_variance`` are the post-source
            values on the interface rows.
        params: Vertical diffusion parameters
        conductances: :class:`~.vertical_diffusion_types.InterfaceConductances`.
        dt: Time step [s]
        surface_momentum: Optional ``(C_m, u_s, v_s)``: the fraction-weighted
            momentum exchange velocity [m/s] and the surface velocity, each
            (ncol,). ``None`` keeps the free-slip bottom boundary.

    Returns:
        Matrix system ready for solution

    """
    ncol, nlev = state.u.shape
    nsfc_type = 3  # Fixed number of surface types (water, ice, land)

    # Matrix types: momentum, heat, moisture, hydrometeors, TKE, thv_var
    nmatrix = 6

    matrix_coeffs = jnp.zeros((ncol, nlev, 3, nmatrix))
    matrix_bottom = jnp.zeros((ncol, 3, nsfc_type, 2))  # Only heat and moisture need surface BC
    rhs_surface = jnp.zeros((ncol, nsfc_type, 2))

    variable_to_matrix = jnp.array([
        0, 0,  # u, v -> momentum matrix
        1,     # T -> heat matrix
        2,     # qv -> moisture matrix
        3, 3,  # qc, qi -> hydrometeor matrix
        4,     # TKE -> TKE matrix
        5      # thv_var -> thv_var matrix
    ])

    # Every full-level row — momentum, heat, moisture and hydrometeors — uses
    # the moist layer pressure thickness, ECHAM's one ``zqdp =
    # 1/(paphm1(k+1)−paphm1(k))``: jcm's specific humidity is per moist mass
    # and every column budget in the model integrates with dp/g, so the qv row
    # conserves that measure.
    dt_factor = dt * params.tpfac1
    dp = jnp.diff(state.pressure_half, axis=1)                    # (ncol, nlev)
    matrix_coeffs = _full_level_rows(matrix_coeffs, dt_factor * conductances.momentum, dp, 0)
    matrix_coeffs = _full_level_rows(matrix_coeffs, dt_factor * conductances.heat, dp, 1)
    matrix_coeffs = _full_level_rows(matrix_coeffs, dt_factor * conductances.heat, dp, 2)
    matrix_coeffs = _full_level_rows(matrix_coeffs, dt_factor * conductances.heat, dp, 3)
    dp_mid = jnp.diff(state.pressure_full, axis=1)                # (ncol, nlev-1)
    matrix_coeffs = _interface_rows(matrix_coeffs, dt_factor * conductances.tke, dp_mid, 4)
    matrix_coeffs = _interface_rows(matrix_coeffs, dt_factor * conductances.thv, dp_mid, 5)

    rhs_vectors = setup_rhs_vectors(state, params)

    # Momentum Robin term on the bottom row (see docstring).
    if surface_momentum is not None:
        c_mom, u_s, v_s = surface_momentum
        rho_s = _surface_air_density(state)
        k_sfc_mom = dt_factor * rho_s * c_mom * c.grav / dp[:, -1]
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


def _set_rows(matrix_coeffs, sub, sup, matrix_idx):
    """Write a tridiagonal block from its off-diagonals (diag = 1 − sub − sup)."""
    matrix_coeffs = matrix_coeffs.at[:, :, 0, matrix_idx].set(sub)
    matrix_coeffs = matrix_coeffs.at[:, :, 2, matrix_idx].set(sup)
    return matrix_coeffs.at[:, :, 1, matrix_idx].set(1.0 - sub - sup)


def _full_level_rows(matrix_coeffs, scaled_conductance, dp, matrix_idx):
    """Rows of a full-level variable from interior-interface coefficients.

    ``scaled_conductance`` is ``α·Δt·G`` (ncol, nlev − 1), ECHAM's ``zcf``;
    row ``k`` couples upward with ``zcf(k−1)/Δp_k`` and downward with
    ``zcf(k)/Δp_k`` (``vdiff.f90`` l.913-923 and 1373-1379).
    """
    zero = jnp.zeros_like(dp[:, :1])
    sub = -jnp.concatenate([zero, scaled_conductance], axis=1) / dp
    sup = -jnp.concatenate([scaled_conductance, zero], axis=1) / dp
    return _set_rows(matrix_coeffs, sub, sup, matrix_idx)


def _interface_rows(matrix_coeffs, scaled_coefficient, dp_mid, matrix_idx):
    """Rows of an interface variable (TKE, θ_v variance), l.1242-1293.

    ``scaled_coefficient`` is ``α·Δt`` times the transport coefficient of
    every interface row (ncol, nlev), ECHAM's ``zcdum``/``zcthv``, the last
    the surface layer's. Interior row ``k`` has the thickness ``Δp`` between
    full levels ``k`` and ``k + 1`` and couples to row ``k ∓ 1`` with the mean
    coefficient of the two rows; the last row is the surface value, held fixed.
    """
    mean = 0.5 * (scaled_coefficient[:, :-1] + scaled_coefficient[:, 1:])  # (ncol, nlev-1)
    zero = jnp.zeros_like(dp_mid[:, :1])
    # Interior rows 0 .. nlev-2: up to row k-1 with mean[k-1], down with mean[k].
    sub_int = -jnp.concatenate([zero, mean[:, :-1]], axis=1) / dp_mid
    sup_int = -mean / dp_mid
    sub = jnp.concatenate([sub_int, zero], axis=1)
    sup = jnp.concatenate([sup_int, zero], axis=1)
    return _set_rows(matrix_coeffs, sub, sup, matrix_idx)


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
    Heat is the dry static energy over ``c_pd`` (:func:`heat_variable`).
    """
    ncol, nlev = state.u.shape
    # Fixed number of variables: u, v, s/cp, qv, qc, qi, TKE, thv_var
    rhs = jnp.zeros((ncol, nlev, 8))

    # Apply tpfac2 scaling to RHS as in ICON
    tpfac2 = params.tpfac2

    rhs = rhs.at[:, :, 0].set(tpfac2 * state.u)  # u
    rhs = rhs.at[:, :, 1].set(tpfac2 * state.v)  # v
    rhs = rhs.at[:, :, 2].set(tpfac2 * heat_variable(state))  # s/cp
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

    Following ECHAM's semi-implicit time stepping (``vdiff.f90`` l.1305-1310,
    1412-1456):
    - bb is the matrix solution (solution of (I - dt*tpfac1*L) * bb = tpfac2 * X_old)
    - X_new = bb + tpfac3 * X_old
    - tendency = (X_new - X_old) / dt = (bb + tpfac3 * X_old - X_old) / dt
                                      = (bb - tpfac2 * X_old) / dt  (since tpfac2 + tpfac3 = 1)

    The temperature is recovered from the new dry static energy with the
    kinetic energy the momentum diffusion removed added back as heat (ECHAM's
    ``zdis``, l.1418-1421 and 1445)::

        zdis  = ½·[(tpfac2·u − bb_u)(tpfac4·u + bb_u) + (same for v)]
        T_new = (s_new + zdis − φ)/(c_pd·(1 + vtmpc2·q))

    with the step-start humidity ``q``, as ECHAM divides by ``pqm1``'s heat
    capacity.

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
    tpfac2 = params.tpfac2
    tpfac4 = 1.0 + tpfac3
    dissipation = 0.5 * ((tpfac2 * state.u - bb_u) * (tpfac4 * state.u + bb_u)
                         + (tpfac2 * state.v - bb_v) * (tpfac4 * state.v + bb_v))
    x_old = heat_variable(state)
    x_new = bb_t + tpfac3 * x_old
    moist = 1.0 + c.vtmpc2 * state.qv
    t_new = state.temperature + (x_new - x_old + dissipation / c.cpd) / moist
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

    # The column's dry static energy change [W/m²] per layer (ECHAM's zcpten,
    # l.1447, whose column sum is psh_vdiff).
    heating_rate = (x_new - x_old) / dt * state.air_mass * c.cpd

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
    snow-melt cap of ``update_soil`` 1859-1863; with
    ``land_temperature="prescribed"`` it keeps the prescribed value instead.
    The column's bottom value is
    the fraction-weighted blend ``bb_K = tpfac2·Σ_t f_t·X̂_K,t``
    (``blend_zq_zt``), and back-substitution completes the solve.

    Each tile's flux is evaluated against its OWN lowest-level value
    (``postproc_ocean``/``_ice``, ``update_soil`` 1892-1911); because the
    eliminated bottom row then reads ``D·bb_K = R + tpfac2·Σ_t f_t·k_t·(X̂_s,t −
    X̂_K,t)``, the grid-mean flux is exactly what the column receives (the
    ``pev_vdiff`` identity) when ``tpfac1·tpfac2 = 1``. Heat is the dry
    static energy over ``c_pd`` (:func:`heat_variable`), whose surface value is
    ECHAM's ``T_s·(1 + vtmpc2·q_s)`` (:func:`surface_heat_variable`), so the
    land balance solves ECHAM's dry static energy ``ŝ = c_p,s·T̂_s`` with the
    tile's heat capacity ``c_p,s = c_pd·(1 + vtmpc2·q_s)`` (``zcpq``,
    ``mo_surface_land.f90`` l.329-331) and ``s_K = E·ŝ + c_pd·F``.

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
    q_s = tiles.saturation_humidity
    # The humidity each tile exchanges, csat·q_s + (1 − cair)·q_a (q_s over
    # water and ice), sets its surface heat capacity (zcptw/zcpti/zcptl).
    q_surface = jsbach_surface_humidity_boundary(
        q_s, state.qv[:, -1:], tiles.cair, tiles.csat)
    x_s = surface_heat_variable(tiles.temperature, q_surface)

    land = tiles.land
    prescribed = (land is not None and jsbach_land.check_land_temperature_mode(land.params)
                  == "prescribed")
    if land is not None:
        il = LAND_TILE
        t_old = land.temperature
        rn_old = (land.net_shortwave + land.emissivity * land.longwave_down
                  - land.emissivity * c.sbc * t_old ** 4)
    if prescribed:
        # Fixed land temperature: the skin stays at the prescribed value the
        # tiles already carry; nothing is solved and the budget stays open.
        t_hat = t_new = t_old
    elif land is not None:
        cp_land = cpd * (1.0 + c.vtmpc2 * q_surface[:, il])          # zcpq
        s_hat = jsbach_land.update_surfacetemp(
            cp_land, en[:, il], cpd * fn[:, il], eq[:, il], fq[:, il],
            cp_land * t_old, q_s[:, il], land.saturation_slope, rn_old,
            land.conductance * (land.soil_temperature - t_old),
            rho_s * tiles.exchange_heat[:, il], tiles.cair[:, il], tiles.csat[:, il],
            tiles.sublimation_fraction[:, il], land.heat_capacity + dt * land.conductance,
            dt, tp1, land.emissivity)
        t_hat = s_hat / cp_land
        # ECHAM forms the new (unfiltered) temperature from the uncapped ŝ and
        # then melts the excess, but evaluates the fluxes at the capped ŝ.
        t_hat = jsbach_land.melt_cap(t_hat, land.melt_capped, land.params)
        t_new = jsbach_land.melt_cap(tp2 * s_hat / cp_land + tp3 * t_old,
                                     land.melt_capped, land.params)
        q_hat = q_s[:, il] + land.saturation_slope * (t_hat - t_old)
        # The land constants are float64 leaves under x64; the tiles keep the
        # state's precision.
        x_s = x_s.at[:, il].set((t_hat * cp_land / cpd).astype(x_s.dtype))
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
        sh_l, lh_l = sh_t[:, LAND_TILE], lh_t[:, LAND_TILE]
        residual = rn - sh_l - lh_l
        if prescribed:
            ground = storage = melt = jnp.zeros_like(rn)
        else:
            ground = land.conductance * (t_new - land.soil_temperature)
            storage = land.heat_capacity * (t_new - t_old) / dt
            melt = residual - ground - storage
        land_out = LandBalanceOutputs(
            temperature=t_new,
            ground_heat_flux=ground,
            net_radiation=rn,
            sensible_heat_flux=sh_l,
            latent_heat_flux=lh_l,
            evaporation=e_t[:, LAND_TILE],
            heat_storage=storage,
            melt_heat_flux=melt,
            energy_residual=residual,
        )
    return sol_t, sol_q, fluxes, land_out


@jax.jit
def vertical_diffusion_step(
    state: VDiffState,
    params: VDiffParameters,
    conductances: InterfaceConductances,
    dt: float,
    surface_momentum: tuple = None,
    surface_tiles: SurfaceTiles = None,
) -> tuple:
    """Perform one vertical diffusion time step.

    Args:
        state: Atmospheric state
        params: Vertical diffusion parameters
        conductances: Interface coefficients of the closure
            (:class:`~.vertical_diffusion_types.InterfaceConductances`).
        dt: Time step [s]
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
    matrix_system = setup_matrix_system(
        state, params, conductances, dt, surface_momentum=surface_momentum,
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
