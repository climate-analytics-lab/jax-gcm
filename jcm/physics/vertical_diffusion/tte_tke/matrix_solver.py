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
from jcm.physics.surface.slim import slim_land
from jcm.physics.thermodynamics import jsbach_surface_humidity_boundary
from .closure import geopotential_above_surface
from .vertical_diffusion_types import (
    VDiffState, VDiffParameters, VDiffMatrixSystem, VDiffTendencies,
    VDiffSurfaceFluxes, SurfaceTiles, LandBalanceOutputs, InterfaceConductances,
    SlimColumnOutputs,
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
    sub_mom, diag_mom, sup_mom = _full_level_triples(
        dt_factor * conductances.momentum, dp)
    block_heat = _full_level_block(dt_factor * conductances.heat, dp)
    dp_mid = jnp.diff(state.pressure_full, axis=1)                # (ncol, nlev-1)
    block_tke = _interface_block(dt_factor * conductances.tke, dp_mid)
    block_thv = _interface_block(dt_factor * conductances.thv, dp_mid)

    rhs_vectors = setup_rhs_vectors(state, params)

    # Momentum Robin term on the bottom row (see docstring).
    if surface_momentum is not None:
        c_mom, u_s, v_s = surface_momentum
        rho_s = _surface_air_density(state)
        k_sfc_mom = dt_factor * rho_s * c_mom * c.grav / dp[:, -1]
        diag_mom = jnp.concatenate(
            [diag_mom[:, :-1], (diag_mom[:, -1] + k_sfc_mom)[:, None]], axis=1)
        tpfac2 = params.tpfac2
        u_rhs = jnp.concatenate(
            [rhs_vectors[:, :-1, 0],
             (rhs_vectors[:, -1, 0] + tpfac2 * k_sfc_mom * u_s)[:, None]],
            axis=1)
        v_rhs = jnp.concatenate(
            [rhs_vectors[:, :-1, 1],
             (rhs_vectors[:, -1, 1] + tpfac2 * k_sfc_mom * v_s)[:, None]],
            axis=1)
        rhs_vectors = jnp.concatenate(
            [u_rhs[:, :, None], v_rhs[:, :, None], rhs_vectors[:, :, 2:]],
            axis=2)

    block_mom = jnp.stack([sub_mom, diag_mom, sup_mom], axis=2)
    matrix_coeffs = jnp.stack(
        [block_mom, block_heat, block_heat, block_heat, block_tke, block_thv],
        axis=3)

    return VDiffMatrixSystem(
        matrix_coeffs=matrix_coeffs,
        matrix_bottom=matrix_bottom,
        rhs_vectors=rhs_vectors,
        rhs_surface=rhs_surface,
        variable_to_matrix=variable_to_matrix
    )


def _full_level_triples(scaled_conductance, dp, axis=1):
    """Return ``(sub, diag, sup)`` for a full-level variable along ``axis``.

    ``scaled_conductance`` is ``α·Δt·G`` (shape ``nlev − 1`` on ``axis``),
    ECHAM's ``zcf``; row ``k`` couples upward with ``zcf(k−1)/Δp_k`` and
    downward with ``zcf(k)/Δp_k`` (``vdiff.f90`` l.913-923 and 1373-1379).
    """
    zero_slice = (slice(None), slice(0, 1)) if axis == 1 else (slice(0, 1),)
    zero = jnp.zeros_like(dp[zero_slice])
    sub = -jnp.concatenate([zero, scaled_conductance], axis=axis) / dp
    sup = -jnp.concatenate([scaled_conductance, zero], axis=axis) / dp
    return sub, 1.0 - sub - sup, sup


def _interface_triples(scaled_coefficient, dp_mid, axis=1):
    """Return ``(sub, diag, sup)`` for an interface variable (TKE, θ_v variance).

    ``scaled_coefficient`` is ``α·Δt`` times the transport coefficient of
    every interface row (shape ``nlev`` on ``axis``), ECHAM's ``zcdum``/``zcthv``,
    the last the surface layer's. Interior row ``k`` has the thickness ``Δp``
    between full levels ``k`` and ``k + 1`` and couples to row ``k ∓ 1`` with
    the mean coefficient of the two rows; the last row is the surface value,
    held fixed (``vdiff.f90`` l.1242-1293).
    """
    if axis == 1:
        mean = 0.5 * (scaled_coefficient[:, :-1] + scaled_coefficient[:, 1:])
        zero = jnp.zeros_like(dp_mid[:, :1])
        sub_int = -jnp.concatenate([zero, mean[:, :-1]], axis=1) / dp_mid
    else:
        mean = 0.5 * (scaled_coefficient[:-1] + scaled_coefficient[1:])
        zero = jnp.zeros_like(dp_mid[:1])
        sub_int = -jnp.concatenate([zero, mean[:-1]], axis=0) / dp_mid
    sup_int = -mean / dp_mid
    sub = jnp.concatenate([sub_int, zero], axis=axis)
    sup = jnp.concatenate([sup_int, zero], axis=axis)
    return sub, 1.0 - sub - sup, sup


def _full_level_block(scaled_conductance, dp):
    """Build ``(ncol, nlev, 3)`` tridiagonal block for a full-level variable."""
    return jnp.stack(_full_level_triples(scaled_conductance, dp, axis=1), axis=2)


def _interface_block(scaled_coefficient, dp_mid):
    """Build ``(ncol, nlev, 3)`` tridiagonal block for an interface variable."""
    return jnp.stack(
        _interface_triples(scaled_coefficient, dp_mid, axis=1), axis=2)


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
    tpfac2 = params.tpfac2
    return jnp.stack([
        tpfac2 * state.u,
        tpfac2 * state.v,
        tpfac2 * heat_variable(state),
        tpfac2 * state.qv,
        tpfac2 * state.qc,
        tpfac2 * state.qi,
        tpfac2 * state.tke,
        tpfac2 * state.thv_variance,
    ], axis=2)


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
    # Transpose to level-major ``(nlev, ..., ncol)`` so ``ncol`` sits on the
    # minor axis and all ``nvar`` systems share one forward/backward scan.
    mc_lm = jnp.moveaxis(matrix_coeffs, (0, 1, 2, 3), (3, 0, 1, 2))
    mc_var = jnp.take(mc_lm, variable_to_matrix, axis=2)  # (nlev, 3, nvar, ncol)
    rhs_lm = jnp.moveaxis(rhs_vectors, (0, 1, 2), (2, 0, 1))  # (nlev, nvar, ncol)
    cp, dp, _ = _forward_sweep_level_major(
        mc_var[:, 0], mc_var[:, 1], mc_var[:, 2], rhs_lm)
    sol_lm = _back_substitute_level_major(cp, dp, dp[-1])
    return jnp.moveaxis(sol_lm, (0, 1, 2), (1, 2, 0))


def _safe_pivot(x):
    """Keep a pivot's sign and keep it away from zero.

    ``jnp.sign(x)·eps + eps`` returned exactly 0 for a tiny negative pivot,
    and the following divisions produced inf that grew ~18 orders of magnitude
    in back-substitution; this form is never zero.
    """
    eps = 1e-20
    return jnp.where(jnp.abs(x) > eps, x, jnp.where(x < 0, -eps, eps))


def _forward_sweep_level_major(a, b, c, d):
    """Thomas elimination along axis 0 for ``(nlev, *batch)`` arrays."""
    pivot_0 = _safe_pivot(b[0])
    cp_0 = c[0] / pivot_0
    dp_0 = d[0] / pivot_0

    def step(carry, inputs):
        cp_prev, dp_prev = carry
        a_i, b_i, c_i, d_i = inputs
        pivot_i = _safe_pivot(b_i - a_i * cp_prev)
        cp_i = c_i / pivot_i
        dp_i = (d_i - a_i * dp_prev) / pivot_i
        return (cp_i, dp_i), (cp_i, dp_i, pivot_i)

    _, (cp_r, dp_r, pv_r) = jax.lax.scan(
        step, (cp_0, dp_0), (a[1:], b[1:], c[1:], d[1:]))
    cp = jnp.concatenate([cp_0[None], cp_r], axis=0)
    dp = jnp.concatenate([dp_0[None], dp_r], axis=0)
    pivot = jnp.concatenate([pivot_0[None], pv_r], axis=0)
    return cp, dp, pivot


def _back_substitute_level_major(cp, dp, x_bottom):
    """Upward sweep ``x_k = dp_k − cp_k·x_{k+1}`` along axis 0."""
    def step(x_next, inputs):
        cp_i, dp_i = inputs
        x_i = dp_i - cp_i * x_next
        return x_i, x_i

    _, x_rest = jax.lax.scan(
        step, x_bottom, (cp[:-1], dp[:-1]), reverse=True)
    return jnp.concatenate([x_rest, x_bottom[None]], axis=0)


def forward_sweep(a, b, c, d):
    """Thomas elimination from the top: ``(cp, dp, pivot)``, each [ncol, nlev].

    Row ``k`` is reduced to ``x_k = dp_k − cp_k·x_{k+1}``. ``pivot_k`` is the
    eliminated diagonal ``b_k − a_k·cp_{k−1}``, so at the bottom row
    ``pivot_K`` and ``dp_K·pivot_K`` are ECHAM's eliminated ``1 + zfac·(1 −
    zebsh_{K−1})`` and ``ztdif_K + zfac·ztdif_{K−1}`` (``vdiff.f90:887-931``,
    with ``cp = −zebsh``), the inputs of the Richtmyer–Morton coefficients.
    """
    cp, dp, pivot = _forward_sweep_level_major(a.T, b.T, c.T, d.T)
    return cp.T, dp.T, pivot.T


def back_substitute(cp, dp, x_bottom):
    """Upward sweep ``x_k = dp_k − cp_k·x_{k+1}`` from a given bottom value."""
    return _back_substitute_level_major(cp.T, dp.T, x_bottom).T


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
    return _momentum_stress_from_bottom(
        solution[:, -1, 0], solution[:, -1, 1], state, params, surface_momentum)


def _momentum_stress_from_bottom(bb_u_bottom, bb_v_bottom, state, params,
                                 surface_momentum):
    c_mom, u_s, v_s = surface_momentum
    rho_s = _surface_air_density(state)
    tp1, tp2 = params.tpfac1, params.tpfac2
    stress_u = rho_s * c_mom * tp1 * (bb_u_bottom - tp2 * u_s)
    stress_v = rho_s * c_mom * tp1 * (bb_v_bottom - tp2 * v_s)
    return stress_u, stress_v


def _couple_surface_from_eliminated(
    state: VDiffState,
    params: VDiffParameters,
    dt: float,
    tiles: SurfaceTiles,
    den_t: jnp.ndarray,
    r_t: jnp.ndarray,
    den_q: jnp.ndarray,
    r_q: jnp.ndarray,
):
    """Solve tile Richtmyer–Morton relations from bottom-eliminated rows."""
    tp1, tp2, tp3 = params.tpfac1, params.tpfac2, params.tpfac3
    rho_s = _surface_air_density(state)
    k_scale = (dt * tp1 * rho_s / state.air_mass[:, -1])[:, None]
    k_h = k_scale * tiles.exchange_heat
    k_q = k_scale * tiles.exchange_moisture

    land = tiles.land
    slim = None if land is None else land.slim
    il = LAND_TILE
    cair_t, csat_t, sfrac_t = tiles.cair, tiles.csat, tiles.sublimation_fraction
    if slim is not None:
        # SLIM's evaporation E = ρ(q_s − q_a)·β_eff/(r_evap + r_a) with this
        # step's r_a = 1/(C_q|U|): cair = csat = β_eff/(1 + C_q|U|·r_evap)
        # (slim_land.evaporation_factor), limited so the step cannot take more
        # than the snow and water present (mml_main.F90 ~1072). Sublimation
        # (L_s) where there is snow or glacier: the sublimating share of the
        # potential evaporation is snow_on·csat, so LH = L·E exactly.
        chu = tiles.exchange_moisture[:, il]
        factor = slim_land.evaporation_factor(
            slim.beta_effective, slim.snow_on, chu, slim.evaporative_resistance, slim.params)
        potential = rho_s * chu * (tiles.saturation_humidity[:, il] - state.qv[:, -1])
        factor = slim_land.cap_evaporation_factor(
            factor, potential, slim.available_water, dt).astype(cair_t.dtype)
        cair_t = cair_t.at[:, il].set(factor)
        csat_t = csat_t.at[:, il].set(factor)
        sfrac_t = sfrac_t.at[:, il].set((slim.snow_on * factor).astype(sfrac_t.dtype))

    en, fn, eq, fq = jsbach_land.richtmyer_morton(
        den_t[:, None], den_q[:, None], r_t[:, None], r_q[:, None], k_h, k_q,
        cair_t, csat_t, tp1)

    cpd = c.cpd
    q_s = tiles.saturation_humidity
    # The humidity each tile exchanges, csat·q_s + (1 − cair)·q_a (q_s over
    # water and ice), sets its surface heat capacity (zcptw/zcpti/zcptl).
    q_surface = jsbach_surface_humidity_boundary(
        q_s, state.qv[:, -1:], cair_t, csat_t)
    x_s = surface_heat_variable(tiles.temperature, q_surface)

    prescribed = (land is not None and jsbach_land.check_land_temperature_mode(land.params)
                  == "prescribed")
    if land is not None:
        t_old = land.temperature
        rn_old = (land.net_shortwave + land.emissivity * land.longwave_down
                  - land.emissivity * c.sbc * t_old ** 4)
    if prescribed:
        # Fixed land temperature: the skin stays at the prescribed value the
        # tiles already carry; nothing is solved and the budget stays open.
        t_hat = t_new = t_old
    elif slim is not None:
        # SLIM: the surface is the top soil layer. The rest of the column,
        # eliminated from the bottom, gives the balance its implicit capacity
        # den·Δt and flux num (ECHAM's pgrndcapc/pgrndhflx), so the surface,
        # the soil column and the lowest level are solved together. SLIM's
        # melt then takes the energy that would lift the provisional top
        # layer above the freezing point, up to the snow present.
        cp_land = cpd * (1.0 + c.vtmpc2 * q_surface[:, il])          # zcpq
        soil_den, soil_num, soil_e, soil_f = slim_land.eliminate_soil_column(
            slim.soil_temperature, slim.heat_capacity, slim.interface_conductivity,
            slim.depth, slim.thickness, dt)
        zcolin, zcohfl, zcoind = jsbach_land.surfacetemp_coefficients(
            cp_land, en[:, il], cpd * fn[:, il], eq[:, il], fq[:, il],
            cp_land * t_old, q_s[:, il], land.saturation_slope, rn_old, soil_num,
            rho_s * tiles.exchange_heat[:, il], cair_t[:, il], csat_t[:, il],
            sfrac_t[:, il], soil_den * dt, dt, tp1, land.emissivity)
        balance = zcolin + zcohfl
        s_free = (zcolin * cp_land * t_old + zcoind) / balance
        t_free = tp2 * s_free / cp_land + tp3 * t_old
        melt_heat = slim_land.melt_energy(
            t_free, balance * cp_land / dt, slim.snow, slim.glacier, dt, slim.params)
        s_hat = s_free - tp1 * dt * melt_heat / balance
        t_hat = s_hat / cp_land
        t_new = tp2 * t_hat + tp3 * t_old
        q_hat = q_s[:, il] + land.saturation_slope * (t_hat - t_old)
        x_s = x_s.at[:, il].set((t_hat * cp_land / cpd).astype(x_s.dtype))
        q_s = q_s.at[:, il].set(q_hat.astype(q_s.dtype))
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
        x_land = (t_hat * cp_land / cpd).astype(x_s.dtype)[:, None]
        q_land = q_hat.astype(q_s.dtype)[:, None]
        x_s = jnp.concatenate([x_s[:, :il], x_land, x_s[:, il + 1:]], axis=1)
        q_s = jnp.concatenate([q_s[:, :il], q_land, q_s[:, il + 1:]], axis=1)

    x_k = en * x_s + fn                      # X̂_K per tile (update_land)
    q_k = eq * q_s + fq
    frac = tiles.fraction
    x_bot_t = tp2 * jnp.sum(frac * x_k, axis=1)
    x_bot_q = tp2 * jnp.sum(frac * q_k, axis=1)

    rho = rho_s[:, None]
    sh_t = rho * cpd * tiles.exchange_heat * (x_s - x_k)
    e_t = rho * tiles.exchange_moisture * (csat_t * q_s - cair_t * q_k)
    e_pot_t = rho * tiles.exchange_moisture * (q_s - q_k)
    lh_t = c.alhc * e_t + (c.alhs - c.alhc) * sfrac_t * e_pot_t
    # ``sh_t`` is ECHAM's ``ahfs`` (mo_surface_ocean.f90::postproc_ocean,
    # ``zcoefw*(ztdif - zcptw)``): a dry-static-energy flux carried with the
    # MOIST heat capacity ``cpd*(1 + vtmpc2*q)``, so it includes the enthalpy
    # ``(cpv - cpd)*T_s*E`` of the evaporated vapour (``cpd*vtmpc2 == cpv -
    # cpd``). The dry-cp sensible heat flux, the CMIP/MERRA-2 ``hfss``
    # definition, removes that term with the same tile skin temperature and
    # evaporation the solve used, so ``sensible_heat_dry + enthalpy ==
    # sensible_heat`` holds exactly.
    t_skin = tiles.temperature
    if land is not None:
        t_skin = t_skin.at[:, LAND_TILE].set(t_hat.astype(t_skin.dtype))
    sh_dry_t = sh_t - cpd * c.vtmpc2 * t_skin * e_t
    fluxes = dict(
        evaporation=jnp.sum(frac * e_t, axis=1),
        sensible_heat=jnp.sum(frac * sh_t, axis=1),
        sensible_heat_dry=jnp.sum(frac * sh_dry_t, axis=1),
        latent_heat=jnp.sum(frac * lh_t, axis=1),
        sensible_heat_tile=sh_t,
        latent_heat_tile=lh_t,
    )

    land_out = None
    if land is not None:
        rn = rn_old - 4.0 * land.emissivity * c.sbc * t_old ** 3 * (t_hat - t_old)
        sh_l, lh_l = sh_t[:, LAND_TILE], lh_t[:, LAND_TILE]
        residual = rn - sh_l - lh_l
        slim_out = None
        if prescribed:
            ground = storage = melt = jnp.zeros_like(rn)
        elif slim is not None:
            # Complete the column from the top layer's increment, then freeze
            # or thaw its water (mml_main.F90::phase_change). The top layer's
            # storage and the conduction below it are the balance's storage
            # and ground flux: rn − sh − lh = storage + ground + melt.
            cv, dz = slim.heat_capacity, slim.thickness[:, None]
            t_soil = slim_land.complete_soil_column(
                slim.soil_temperature, t_new - t_old, soil_e, soil_f)
            storage = cv[0] * dz[0] * (t_new - t_old) / dt
            ground = jnp.sum(cv[1:] * dz[1:] * (t_soil[1:] - slim.soil_temperature[1:]),
                             axis=0) / dt
            melt = melt_heat
            t_soil, soil_ice, phase_heat = slim_land.phase_change(
                t_soil, slim.soil_ice, cv, slim.thickness, slim.max_water, dt)
            slim_out = SlimColumnOutputs(
                soil_temperature=t_soil, soil_ice=soil_ice,
                snow_melt=melt_heat / c.alhf, melt_heat_flux=melt_heat,
                phase_change_heat_flux=phase_heat, soil_heat_flux=ground,
                evaporation_factor=csat_t[:, il])
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
            slim=slim_out,
        )
    return x_bot_t, x_bot_q, fluxes, land_out


def couple_surface_tiles(state: VDiffState, params: VDiffParameters,
                         matrix_system: VDiffMatrixSystem, dt: float,
                         tiles: SurfaceTiles):
    """Heat and moisture coupled to the surface tile by tile, as ECHAM does.

    ECHAM eliminates the column from the top once (``vdiff.f90:887-931``),
    after which the lowest level obeys, for each tile ``t`` on its own exchange
    coefficient ``k_t = Δt·α·ρ_s·C_t·g/Δp_K`` (``richtmyer_land``, ``_ocean``,
    ``_ice``; :func:`~jcm.physics.surface.echam.jsbach_land.richtmyer_morton`)::

        X̂_K,t = E_t·X̂_s,t + F_t

    The prescribed tile (open water at the SST) and the sea-ice tile (at its
    carried surface temperature, advanced by ``s_sicetemp`` after the solve
    from the tile fluxes returned here) give ``X̂_K,t`` directly. The land tile, when
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
    mc, rhs = matrix_system.matrix_coeffs, matrix_system.rhs_vectors
    a_tq = jnp.stack([mc[:, :, 0, 1].T, mc[:, :, 0, 2].T], axis=1)
    b_tq = jnp.stack([mc[:, :, 1, 1].T, mc[:, :, 1, 2].T], axis=1)
    c_tq = jnp.stack([mc[:, :, 2, 1].T, mc[:, :, 2, 2].T], axis=1)
    d_tq = jnp.stack([rhs[:, :, 2].T, rhs[:, :, 3].T], axis=1)
    cp_tq, dp_tq, pv_tq = _forward_sweep_level_major(a_tq, b_tq, c_tq, d_tq)
    den_t, den_q = pv_tq[-1, 0], pv_tq[-1, 1]
    r_t, r_q = dp_tq[-1, 0] * den_t, dp_tq[-1, 1] * den_q

    x_bot_t, x_bot_q, fluxes, land_out = _couple_surface_from_eliminated(
        state, params, dt, tiles, den_t, r_t, den_q, r_q)
    sol_tq = _back_substitute_level_major(
        cp_tq, dp_tq, jnp.stack([x_bot_t, x_bot_q], axis=0))
    return sol_tq[:, 0].T, sol_tq[:, 1].T, fluxes, land_out


def _compute_tendencies_level_major(
    sol_lm: jnp.ndarray,
    heat_lm: jnp.ndarray,
    state: VDiffState,
    params: VDiffParameters,
    dt: float,
) -> VDiffTendencies:
    """Compute tendencies directly in ``(nlev, ncol)`` layout."""
    u_lm = state.u.T
    v_lm = state.v.T
    t_lm = state.temperature.T
    qv_lm = state.qv.T
    qc_lm = state.qc.T
    qi_lm = state.qi.T
    tke_lm = state.tke.T
    thv_var_lm = state.thv_variance.T

    bb_u = sol_lm[:, 0]
    bb_v = sol_lm[:, 1]
    bb_t = sol_lm[:, 2]
    bb_qv = sol_lm[:, 3]
    bb_qc = sol_lm[:, 4]
    bb_qi = sol_lm[:, 5]
    bb_tke = sol_lm[:, 6]
    bb_thv_var = sol_lm[:, 7]

    tpfac2 = params.tpfac2
    tpfac3 = params.tpfac3
    tpfac4 = 1.0 + tpfac3
    u_new = bb_u + tpfac3 * u_lm
    v_new = bb_v + tpfac3 * v_lm
    dissipation = 0.5 * ((tpfac2 * u_lm - bb_u) * (tpfac4 * u_lm + bb_u)
                         + (tpfac2 * v_lm - bb_v) * (tpfac4 * v_lm + bb_v))
    x_new = bb_t + tpfac3 * heat_lm
    moist = 1.0 + c.vtmpc2 * qv_lm
    t_new = t_lm + (x_new - heat_lm + dissipation / c.cpd) / moist
    qv_new = bb_qv + tpfac3 * qv_lm
    qc_new = bb_qc + tpfac3 * qc_lm
    qi_new = bb_qi + tpfac3 * qi_lm
    tke_new = bb_tke + tpfac3 * tke_lm
    thv_var_new = bb_thv_var + tpfac3 * thv_var_lm

    heating_rate = (x_new - heat_lm) / dt * state.air_mass.T * c.cpd

    return VDiffTendencies(
        u_tendency=((u_new - u_lm) / dt).T,
        v_tendency=((v_new - v_lm) / dt).T,
        temperature_tendency=((t_new - t_lm) / dt).T,
        heating_rate=heating_rate.T,
        qv_tendency=((qv_new - qv_lm) / dt).T,
        qc_tendency=((qc_new - qc_lm) / dt).T,
        qi_tendency=((qi_new - qi_lm) / dt).T,
        tke_tendency=((tke_new - tke_lm) / dt).T,
        thv_var_tendency=((thv_var_new - thv_var_lm) / dt).T,
    )


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
    # Build and solve all 8 tridiagonal systems in ``(nlev, 8, ncol)`` layout
    # so ``ncol`` stays on the minor axis and Thomas elimination runs in a
    # single forward + backward scan pair across all variables.
    dt_factor = dt * params.tpfac1
    tpfac2 = params.tpfac2
    dp_lm = jnp.diff(state.pressure_half.T, axis=0)
    dp_mid_lm = jnp.diff(state.pressure_full.T, axis=0)

    sub_mom, diag_mom, sup_mom = _full_level_triples(
        dt_factor * conductances.momentum.T, dp_lm, axis=0)
    sub_heat, diag_heat, sup_heat = _full_level_triples(
        dt_factor * conductances.heat.T, dp_lm, axis=0)
    sub_tke, diag_tke, sup_tke = _interface_triples(
        dt_factor * conductances.tke.T, dp_mid_lm, axis=0)
    sub_thv, diag_thv, sup_thv = _interface_triples(
        dt_factor * conductances.thv.T, dp_mid_lm, axis=0)

    t_lm = state.temperature.T
    qv_lm = state.qv.T
    qc_lm = state.qc.T
    qi_lm = state.qi.T
    tv_lm = t_lm * (1.0 + c.vtmpc1 * qv_lm - (qc_lm + qi_lm))
    pgeom1_lm = geopotential_above_surface(
        state.geopotential.T, tv_lm, state.pressure_half.T)
    heat_lm = t_lm * (1.0 + c.vtmpc2 * qv_lm) + pgeom1_lm / c.cpd

    rhs_u = tpfac2 * state.u.T
    rhs_v = tpfac2 * state.v.T
    if surface_momentum is not None:
        c_mom, u_s, v_s = surface_momentum
        rho_s = _surface_air_density(state)
        k_sfc_mom = dt_factor * rho_s * c_mom * c.grav / dp_lm[-1]
        diag_mom = jnp.concatenate(
            [diag_mom[:-1], (diag_mom[-1] + k_sfc_mom)[None]], axis=0)
        rhs_u = jnp.concatenate(
            [rhs_u[:-1], (rhs_u[-1] + tpfac2 * k_sfc_mom * u_s)[None]], axis=0)
        rhs_v = jnp.concatenate(
            [rhs_v[:-1], (rhs_v[-1] + tpfac2 * k_sfc_mom * v_s)[None]], axis=0)

    a_lm = jnp.stack(
        [sub_mom, sub_mom, sub_heat, sub_heat, sub_heat, sub_heat, sub_tke, sub_thv],
        axis=1)
    b_lm = jnp.stack(
        [diag_mom, diag_mom, diag_heat, diag_heat, diag_heat, diag_heat, diag_tke, diag_thv],
        axis=1)
    c_lm = jnp.stack(
        [sup_mom, sup_mom, sup_heat, sup_heat, sup_heat, sup_heat, sup_tke, sup_thv],
        axis=1)
    d_lm = jnp.stack([
        rhs_u,
        rhs_v,
        tpfac2 * heat_lm,
        tpfac2 * qv_lm,
        tpfac2 * qc_lm,
        tpfac2 * qi_lm,
        tpfac2 * state.tke.T,
        tpfac2 * state.thv_variance.T,
    ], axis=1)

    cp_lm, dp_lm_elim, pv_lm = _forward_sweep_level_major(a_lm, b_lm, c_lm, d_lm)

    ncol = state.u.shape[0]
    zero = jnp.zeros(ncol)
    fluxes = dict(evaporation=zero, sensible_heat=zero, sensible_heat_dry=zero,
                  latent_heat=zero)
    land = None
    if surface_tiles is not None:
        den_t, den_q = pv_lm[-1, 2], pv_lm[-1, 3]
        r_t, r_q = dp_lm_elim[-1, 2] * den_t, dp_lm_elim[-1, 3] * den_q
        x_bot_t, x_bot_q, fluxes, land = _couple_surface_from_eliminated(
            state, params, dt, surface_tiles, den_t, r_t, den_q, r_q)
        x_bottom = jnp.concatenate([
            dp_lm_elim[-1, :2],
            jnp.stack([x_bot_t, x_bot_q], axis=0),
            dp_lm_elim[-1, 4:],
        ], axis=0)
    else:
        x_bottom = dp_lm_elim[-1]

    sol_lm = _back_substitute_level_major(cp_lm, dp_lm_elim, x_bottom)

    stress_u = stress_v = zero
    if surface_momentum is not None:
        stress_u, stress_v = _momentum_stress_from_bottom(
            sol_lm[-1, 0], sol_lm[-1, 1], state, params, surface_momentum)

    tendencies = _compute_tendencies_level_major(sol_lm, heat_lm, state, params, dt)
    surface_fluxes = VDiffSurfaceFluxes(stress_u=stress_u, stress_v=stress_v, **fluxes)
    return tendencies, surface_fluxes, land
