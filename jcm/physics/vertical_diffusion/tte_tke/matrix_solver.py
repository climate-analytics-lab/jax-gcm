"""Tridiagonal matrix solver for vertical diffusion.

The implicit column solve of ECHAM6.3's ``vdiff.f90`` (sections 3.8, 4 and 5):
full-level variables exchange across the interior interfaces with the
coefficients of :mod:`.closure`, TKE and the θ_v variance on the interface rows,
and heat and moisture couple to the surface tiles through the Richtmyer–Morton
relations.

:func:`vertical_diffusion_step` is the composition of the public pieces, and
production runs exactly that composition::

    system = setup_matrix_system(...)                 # build (+ momentum Robin row)
    cp, dp, pivot = forward_sweep(system.lower, system.diag, system.upper, system.rhs)
    bottom, fluxes, land = couple_surface_tiles(...)  # heat/moisture bottom values
    solution = back_substitute(cp, dp, bottom)
    tendencies = compute_tendencies_from_solution(solution, system, state, params, dt)

Every array of the solve is level-major, ``(nlev, ...)`` with the column as the
contiguous minor axis, and the eight systems share one ``(nlev, NVAR, ncol)``
array (:class:`~.vertical_diffusion_types.VDiffMatrixSystem`): the Thomas
sweeps are then one ``lax.scan`` pair over the levels for every variable and
column at once, with no per-column ``vmap`` and no gathers or scatters, which
is what keeps the solve fast on accelerators. :class:`VDiffState` stays
column-leading ``(ncol, nlev)``; the solve transposes it on entry and its
tendencies on exit.
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
    SlimColumnOutputs, IU, IV, IH, IQV, IQC, IQI, ITKE, ITHV,
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


def heat_variable(state: VDiffState) -> jnp.ndarray:
    """Return the heat variable the solve diffuses, ``s/c_pd`` [K], (nlev, ncol).

    ECHAM diffuses the dry static energy ``zcptgz = φ + c_pd·T·(1 +
    vtmpc2·q)`` (``vdiff.f90`` l.669-670, 890), with ``φ`` the geopotential
    above the surface (``pgeom1``, :func:`.closure.geopotential_above_surface`),
    so a well-mixed layer has uniform ``s``, not uniform ``T``, and the surface
    exchanges ``s_s − s_K`` with ECHAM's surface values ``zcptw``/``zcpti``/
    ``zcptl`` (:func:`surface_heat_variable`). It is carried here divided by
    ``c_pd``, in kelvin, level-major like every array of the solve.
    """
    t = state.temperature.T
    qv = state.qv.T
    tv = t * (1.0 + c.vtmpc1 * qv - (state.qc.T + state.qi.T))
    geopotential = geopotential_above_surface(
        state.geopotential.T, tv, state.pressure_half.T)
    return t * (1.0 + c.vtmpc2 * qv) + geopotential / c.cpd


def surface_heat_variable(temperature, surface_humidity):
    """``s_s/c_pd = T_s·(1 + vtmpc2·q_s)`` of a surface tile [K].

    ECHAM's ``zcptw``/``zcpti``/``zcptl`` (``precalc_ocean``/``_ice``/
    ``_land``) over ``c_pd``: the tile's dry static energy, its geopotential
    zero, with the surface humidity the tile exchanges (``csat·q_s + (1 −
    cair)·q_a`` over land).
    """
    return temperature * (1.0 + c.vtmpc2 * surface_humidity)


def _with_bottom(x, bottom):
    """``x`` (nlev, ...) with its last level replaced by ``bottom``.

    Built by concatenation rather than ``x.at[-1].set``: the indexed update
    lowers to a scatter, which is slow on accelerators.
    """
    return jnp.concatenate([x[:-1], bottom[None]], axis=0)


def _with_tile(x, tile, value):
    """``x`` (ncol, nsfc_type) with tile column ``tile`` replaced by ``value``.

    Concatenated rather than ``x.at[:, tile].set`` to avoid a scatter, slow on
    accelerators; the value is cast to the tiles' precision (the land
    constants are float64 leaves under x64, the tiles keep the state's
    precision).
    """
    return jnp.concatenate(
        [x[:, :tile], value.astype(x.dtype)[:, None], x[:, tile + 1:]], axis=1)


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

    The right-hand side is ECHAM's semi-implicit ``tpfac2·X_old``
    (``mo_vdiff_solver.f90``): the system ``(I − α·Δt·L)·bb = tpfac2·X_old``
    is solved for ``bb``, and ``X_new = bb + tpfac3·X_old``
    (:func:`compute_tendencies_from_solution`). Heat is the dry static energy
    over ``c_pd`` (:func:`heat_variable`).

    Surface boundary condition
    --------------------------
    Momentum: with ``surface_momentum = (C_m, u_s, v_s)`` the bulk drag enters
    the BOTTOM row of the u/v systems as a Robin term, exactly as ECHAM's
    fraction-weighted ``cdum`` does (``mo_surface.f90:1205-1219`` →
    ``vdiff.f90:1099-1100``)::

        k_sfc = dt · tpfac1 · ρ_s · C_m · g / Δp_K
        diag[K, IU/IV] += k_sfc
        rhs[K, IU/IV]  += tpfac2 · k_sfc · u_s/v_s

    with ``ρ_s = p_s/(R_d·T_v,K)``: the dimensionless row entry of ECHAM's
    ``zcfm·zqdp``, the RHS constant divided by α because the solver works in
    ``bb = X̂/α`` units. Heat and moisture get no surface term here: they
    couple tile by tile through the Richtmyer–Morton relations in
    :func:`couple_surface_tiles`. The hydrometeors get none at all,
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
        The level-major :class:`~.vertical_diffusion_types.VDiffMatrixSystem`.

    """
    # Every full-level row — momentum, heat, moisture and hydrometeors — uses
    # the moist layer pressure thickness, ECHAM's one ``zqdp =
    # 1/(paphm1(k+1)−paphm1(k))``: jcm's specific humidity is per moist mass
    # and every column budget in the model integrates with dp/g, so the qv row
    # conserves that measure.
    dt_factor = dt * params.tpfac1
    tpfac2 = params.tpfac2
    dpres = jnp.diff(state.pressure_half.T, axis=0)               # (nlev, ncol)
    dpres_mid = jnp.diff(state.pressure_full.T, axis=0)           # (nlev-1, ncol)
    mom = _full_level_triples(dt_factor * conductances.momentum.T, dpres)
    heat = _full_level_triples(dt_factor * conductances.heat.T, dpres)
    tke = _interface_triples(dt_factor * conductances.tke.T, dpres_mid)
    thv = _interface_triples(dt_factor * conductances.thv.T, dpres_mid)

    previous = jnp.stack([
        state.u.T, state.v.T, heat_variable(state), state.qv.T, state.qc.T,
        state.qi.T, state.tke.T, state.thv_variance.T,
    ], axis=1)                                                    # (nlev, NVAR, ncol)
    rhs = tpfac2 * previous

    # Momentum Robin term on the bottom row (see docstring).
    if surface_momentum is not None:
        c_mom, u_s, v_s = surface_momentum
        rho_s = _surface_air_density(state)
        k_sfc_mom = dt_factor * rho_s * c_mom * c.grav / dpres[-1]
        sub_mom, diag_mom, sup_mom = mom
        mom = (sub_mom, _with_bottom(diag_mom, diag_mom[-1] + k_sfc_mom), sup_mom)
        rhs_uv_bottom = rhs[-1, :IH] + tpfac2 * k_sfc_mom * jnp.stack([u_s, v_s])
        rhs = _with_bottom(rhs, jnp.concatenate([rhs_uv_bottom, rhs[-1, IH:]], axis=0))

    # u, v on the momentum coefficients; heat, qv, qc, qi on the heat ones.
    lower, diag, upper = (
        jnp.stack([m, m, h, h, h, h, k, v], axis=1)
        for m, h, k, v in zip(mom, heat, tke, thv))
    return VDiffMatrixSystem(
        lower=lower, diag=diag, upper=upper, rhs=rhs, previous=previous)


def _full_level_triples(scaled_conductance, dpres):
    """Return level-major ``(sub, diag, sup)`` (nlev, ncol) for a full-level variable.

    ``scaled_conductance`` (nlev − 1, ncol) is ``α·Δt·G``, ECHAM's ``zcf``,
    and ``dpres`` (nlev, ncol) the layer thickness; row ``k`` couples upward
    with ``zcf(k−1)/Δp_k`` and downward with ``zcf(k)/Δp_k`` (``vdiff.f90``
    l.913-923 and 1373-1379). The top row has no level above and the bottom
    row none below (its surface term is added by the caller).
    """
    zero = jnp.zeros_like(dpres[:1])
    sub = -jnp.concatenate([zero, scaled_conductance], axis=0) / dpres
    sup = -jnp.concatenate([scaled_conductance, zero], axis=0) / dpres
    return sub, 1.0 - sub - sup, sup


def _interface_triples(scaled_coefficient, dpres_mid):
    """Return level-major ``(sub, diag, sup)`` (nlev, ncol) for an interface variable.

    ``scaled_coefficient`` (nlev, ncol) is ``α·Δt`` times the transport
    coefficient of every interface row (TKE or θ_v variance), ECHAM's
    ``zcdum``/``zcthv``, the last the surface layer's. Interior row ``k`` has
    the thickness ``dpres_mid`` (nlev − 1, ncol) between full levels ``k`` and
    ``k + 1`` and couples to row ``k ∓ 1`` with the mean coefficient of the two
    rows; the last row is the surface value, held fixed (``vdiff.f90``
    l.1242-1293).
    """
    mean = 0.5 * (scaled_coefficient[:-1] + scaled_coefficient[1:])
    zero = jnp.zeros_like(dpres_mid[:1])
    sub_interior = -jnp.concatenate([zero, mean[:-1]], axis=0) / dpres_mid
    sup_interior = -mean / dpres_mid
    sub = jnp.concatenate([sub_interior, zero], axis=0)
    sup = jnp.concatenate([sup_interior, zero], axis=0)
    return sub, 1.0 - sub - sup, sup


def _safe_pivot(x):
    """Keep a pivot's sign and keep it away from zero.

    ``jnp.sign(x)·eps + eps`` returned exactly 0 for a tiny negative pivot,
    and the following divisions produced inf that grew ~18 orders of magnitude
    in back-substitution; this form is never zero.
    """
    eps = 1e-20
    return jnp.where(jnp.abs(x) > eps, x, jnp.where(x < 0, -eps, eps))


def forward_sweep(lower, diag, upper, rhs):
    """Thomas elimination from the top along axis 0: ``(cp, dp, pivot)``.

    Each argument is level-major ``(nlev, *batch)``; any trailing axes (the
    ``(NVAR, ncol)`` of a :class:`~.vertical_diffusion_types.VDiffMatrixSystem`,
    or just ``ncol``) are independent systems solved by the same scan. Row ``k``
    is reduced to ``x_k = dp_k − cp_k·x_{k+1}``. ``pivot_k`` is the eliminated
    diagonal ``diag_k − lower_k·cp_{k−1}``, so at the bottom row ``pivot_K``
    and ``dp_K·pivot_K`` are ECHAM's eliminated ``1 + zfac·(1 − zebsh_{K−1})``
    and ``ztdif_K + zfac·ztdif_{K−1}`` (``vdiff.f90:887-931``, with ``cp =
    −zebsh``), the inputs of the Richtmyer–Morton coefficients
    (:func:`couple_surface_tiles`).
    """
    pivot_0 = _safe_pivot(diag[0])
    cp_0 = upper[0] / pivot_0
    dp_0 = rhs[0] / pivot_0

    def step(carry, inputs):
        cp_prev, dp_prev = carry
        a_i, b_i, c_i, d_i = inputs
        pivot_i = _safe_pivot(b_i - a_i * cp_prev)
        cp_i = c_i / pivot_i
        dp_i = (d_i - a_i * dp_prev) / pivot_i
        return (cp_i, dp_i), (cp_i, dp_i, pivot_i)

    _, (cp_r, dp_r, pv_r) = jax.lax.scan(
        step, (cp_0, dp_0), (lower[1:], diag[1:], upper[1:], rhs[1:]))
    cp = jnp.concatenate([cp_0[None], cp_r], axis=0)
    dp = jnp.concatenate([dp_0[None], dp_r], axis=0)
    pivot = jnp.concatenate([pivot_0[None], pv_r], axis=0)
    return cp, dp, pivot


def back_substitute(cp, dp, x_bottom):
    """Upward sweep ``x_k = dp_k − cp_k·x_{k+1}`` along axis 0 from ``x_bottom``.

    ``x_bottom`` (``*batch``) is the bottom level's value: ``dp[-1]`` for the
    eliminated system's own solution, or the surface-coupled value
    :func:`couple_surface_tiles` returns.
    """
    def step(x_next, inputs):
        cp_i, dp_i = inputs
        x_i = dp_i - cp_i * x_next
        return x_i, x_i

    _, x_rest = jax.lax.scan(
        step, x_bottom, (cp[:-1], dp[:-1]), reverse=True)
    return jnp.concatenate([x_rest, x_bottom[None]], axis=0)


def solve_tridiagonal(lower, diag, upper, rhs):
    """Solve level-major ``(nlev, *batch)`` tridiagonal systems (Thomas algorithm).

    :func:`forward_sweep` then :func:`back_substitute` from the eliminated
    bottom row, i.e. with no further surface condition.
    """
    cp, dp, _ = forward_sweep(lower, diag, upper, rhs)
    return back_substitute(cp, dp, dp[-1])


def solve_tridiagonal_system(system: VDiffMatrixSystem) -> jnp.ndarray:
    """Solve all eight systems of ``system`` at once: ``bb`` (nlev, NVAR, ncol).

    The free-bottom solve (:func:`solve_tridiagonal`); :func:`vertical_diffusion_step`
    instead completes heat and moisture from the surface tiles between the
    two sweeps.
    """
    return solve_tridiagonal(system.lower, system.diag, system.upper, system.rhs)


def compute_tendencies_from_solution(
    solution: jnp.ndarray,
    system: VDiffMatrixSystem,
    state: VDiffState,
    params: VDiffParameters,
    dt: float
) -> VDiffTendencies:
    """Compute tendencies from the solution of the matrix system.

    Following ECHAM's semi-implicit time stepping (``vdiff.f90`` l.1305-1310,
    1412-1456):
    - bb is the matrix solution (solution of (I - dt*tpfac1*L) * bb = tpfac2 * X_old)
    - X_new = bb + tpfac3 * X_old
    - tendency = (X_new - X_old) / dt

    The temperature is recovered from the new dry static energy with the
    kinetic energy the momentum diffusion removed added back as heat (ECHAM's
    ``zdis``, l.1418-1421 and 1445)::

        zdis  = ½·[(tpfac2·u − bb_u)(tpfac4·u + bb_u) + (same for v)]
        T_new = (s_new + zdis − φ)/(c_pd·(1 + vtmpc2·q))

    with the step-start humidity ``q``, as ECHAM divides by ``pqm1``'s heat
    capacity.

    Args:
        solution: ``bb`` (nlev, NVAR, ncol), level-major like ``system``.
        system: The system solved; its ``previous`` holds ``X_old``.
        state: Original atmospheric state
        params: Vertical diffusion parameters
        dt: Time step [s]

    Returns:
        Tendencies for all variables, column-leading (ncol, nlev) like ``state``.

    """
    x_old = system.previous
    tpfac2 = params.tpfac2
    tpfac3 = params.tpfac3
    tpfac4 = 1.0 + tpfac3
    x_new = solution + tpfac3 * x_old
    tendency = (x_new - x_old) / dt                               # (nlev, NVAR, ncol)

    u, v = x_old[:, IU], x_old[:, IV]
    bb_u, bb_v = solution[:, IU], solution[:, IV]
    dissipation = 0.5 * ((tpfac2 * u - bb_u) * (tpfac4 * u + bb_u)
                         + (tpfac2 * v - bb_v) * (tpfac4 * v + bb_v))
    t_old = state.temperature.T
    moist = 1.0 + c.vtmpc2 * x_old[:, IQV]
    t_new = t_old + (x_new[:, IH] - x_old[:, IH] + dissipation / c.cpd) / moist

    # The column's dry static energy change [W/m²] per layer (ECHAM's zcpten,
    # l.1447, whose column sum is psh_vdiff).
    heating_rate = tendency[:, IH] * state.air_mass.T * c.cpd

    return VDiffTendencies(
        u_tendency=tendency[:, IU].T,
        v_tendency=tendency[:, IV].T,
        temperature_tendency=((t_new - t_old) / dt).T,
        heating_rate=heating_rate.T,
        qv_tendency=tendency[:, IQV].T,
        qc_tendency=tendency[:, IQC].T,
        qi_tendency=tendency[:, IQI].T,
        tke_tendency=tendency[:, ITKE].T,
        thv_var_tendency=tendency[:, ITHV].T,
    )


def _momentum_stress(solution, state, params, surface_momentum):
    """Stress the atmosphere exerts on the surface, from the implicit solution.

    ``τ = ρ_s·C_m·tpfac1·(bb_K − tpfac2·u_s)`` — ECHAM's ``zcfm·zudif``
    diagnosis (``mo_surface_land.f90::update_stress_land``) on the α-weighted
    bottom value the solver used, so the stress equals the column's momentum
    change (positive with the wind; the column receives −τ). ``solution`` is
    the level-major ``bb`` (nlev, NVAR, ncol).
    """
    c_mom, u_s, v_s = surface_momentum
    rho_s = _surface_air_density(state)
    tp1, tp2 = params.tpfac1, params.tpfac2
    stress_u = rho_s * c_mom * tp1 * (solution[-1, IU] - tp2 * u_s)
    stress_v = rho_s * c_mom * tp1 * (solution[-1, IV] - tp2 * v_s)
    return stress_u, stress_v


def couple_surface_tiles(state: VDiffState, params: VDiffParameters,
                         eliminated: tuple, dt: float, tiles: SurfaceTiles):
    """Heat and moisture coupled to the surface tile by tile, as ECHAM does.

    ECHAM eliminates the column from the top once (``vdiff.f90:887-931``;
    ``eliminated`` is :func:`forward_sweep` of the whole system), after which
    the lowest level obeys, for each tile ``t`` on its own exchange
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

    Args:
        state: Atmospheric state.
        params: Vertical diffusion parameters.
        eliminated: ``(cp, dp, pivot)`` of :func:`forward_sweep` over the
            level-major system, each (nlev, NVAR, ncol); only the bottom rows
            of the heat and moisture systems are read.
        dt: Time step [s].
        tiles: The surface as the solve sees it.

    Returns:
        ``(bottom, fluxes, land)``: the bottom values (NVAR, ncol) to
        back-substitute from (:func:`back_substitute`), heat and moisture the
        tile blend and every other variable its free value ``dp[-1]``; a dict
        of the grid fluxes without the stress; and the land tile's
        :class:`LandBalanceOutputs` (``None`` without ``tiles.land``).

    """
    _, dp, pivot = eliminated
    dp_bottom = dp[-1]
    den_t, den_q = pivot[-1, IH], pivot[-1, IQV]
    r_t, r_q = dp_bottom[IH] * den_t, dp_bottom[IQV] * den_q

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
        x_s = _with_tile(x_s, il, t_hat * cp_land / cpd)
        q_s = _with_tile(q_s, il, q_hat)

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
    # The heat and moisture rows end at the tile blend; every other variable
    # keeps its free bottom (the eliminated row's own value).
    bottom = jnp.concatenate(
        [dp_bottom[:IH], jnp.stack([x_bot_t, x_bot_q]), dp_bottom[IQV + 1:]], axis=0)
    return bottom, fluxes, land_out


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

    Builds the eight systems (:func:`setup_matrix_system`), eliminates them
    from the top in one sweep (:func:`forward_sweep`), takes the heat and
    moisture bottom values from the surface tiles (:func:`couple_surface_tiles`)
    and back-substitutes (:func:`back_substitute`).

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
    system = setup_matrix_system(state, params, conductances, dt, surface_momentum)
    eliminated = forward_sweep(system.lower, system.diag, system.upper, system.rhs)
    cp, dp, _ = eliminated

    zero = jnp.zeros(state.u.shape[0])
    fluxes = dict(evaporation=zero, sensible_heat=zero, sensible_heat_dry=zero,
                  latent_heat=zero)
    land = None
    bottom = dp[-1]
    if surface_tiles is not None:
        bottom, fluxes, land = couple_surface_tiles(
            state, params, eliminated, dt, surface_tiles)
    solution = back_substitute(cp, dp, bottom)

    stress_u = stress_v = zero
    if surface_momentum is not None:
        stress_u, stress_v = _momentum_stress(solution, state, params, surface_momentum)

    tendencies = compute_tendencies_from_solution(solution, system, state, params, dt)
    surface_fluxes = VDiffSurfaceFluxes(stress_u=stress_u, stress_v=stress_v, **fluxes)
    return tendencies, surface_fluxes, land
