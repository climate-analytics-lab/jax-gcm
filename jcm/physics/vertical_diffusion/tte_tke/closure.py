"""ECHAM6.3's TKE closure of the vertical diffusion, ``vdiff.f90`` r7492.

What ``vdiff.f90::vdiff`` does with the moist buoyancy and shear of every
interface (:mod:`.moist_buoyancy`) to obtain the exchange coefficients, the new
turbulent kinetic energy and the new variance of the virtual potential
temperature:

* the PBL extension ``ihpbl`` (l.730-765): the higher of the dynamic height
  ``0.3·u*/f`` and the lowest level whose dry static energy exceeds the lowest
  level's (:func:`pbl_extension`);
* the mixing length (l.801-817): Blackadar's ``κz/(1 + κz/λ)`` with
  ``λ = clam`` inside the PBL extension and Holtslag and Boville's
  height-dependent ``λ(z)`` above it (:func:`mixing_length`);
* the stability functions of Louis (1979) (l.819-832, :func:`louis_stability`);
* the implicit TKE source and dissipation (l.837-848, :func:`tke_source_update`)
  and the θ_v-variance budget (l.857-860, :func:`thv_variance_source_update`);
* the exchange coefficients of the interface (l.849-856,
  :func:`interior_closure`), and the TKE boundary condition the surface layer
  supplies for the lowest interface (``mo_surface_ocean.f90::precalc_ocean``
  l.387-388 and 422-457, repeated in ``precalc_land`` and ``precalc_ice``;
  :func:`surface_tke_boundary`).

**Staggering.** As in ECHAM, TKE, the θ_v variance and the exchange
coefficients live on interfaces. An array of these has ``nlev`` rows and row
``k`` is the interface *below* full level ``k`` (ECHAM's index ``jk = k + 1``):
rows ``0 .. nlev-2`` are the interior interfaces, between full levels ``k`` and
``k + 1``, and row ``nlev - 1`` is the surface-layer value (ECHAM's ``klev``
row, which the surface supplies). The model top interface carries nothing. A
consumer that reads row ``k`` "at level ``k``" reads what ECHAM's own consumers
of ``ptkem1(jl,jk)`` read.

**Arrays.** Vertical on axis 0, model top first; any trailing axes are
horizontal and broadcast (the identical code runs on a ``(nlev,)`` column and a
``(nlev, ncols)`` block). Interior-interface arrays have ``nlev - 1`` rows.

**Units of the coefficients.** ECHAM forms ``zcfm = zsm·√e·zcons18·zalf`` with
``zcons18 = cvdifts·Δt·g²`` and ``zalf = p_{k+1/2}/(T̄_v·Δφ·R_d) = ρ/Δφ``, and
the solver divides by the layer's pressure thickness. The coefficients here are
``zcfm/(cvdifts·Δt)`` — the *conductance* ``g²·ρ/Δφ·K = g·ρ·K/Δz`` [Pa/s] of the
interface — so the time weighting stays the solver's; ``K = zsm·√e`` [m²/s] is
published alongside.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

import jcm.constants as c

#: ECHAM's security floor under ``|f|`` in the dynamic PBL height,
#: ``zepcor = 5e-5`` s⁻¹ (``vdiff.f90`` l.574).
CORIOLIS_FLOOR = 5.0e-5

#: Upper bound on the geopotential of the PBL extension that enters the
#: convective velocity scale, ``MIN(50000, pgeom1(ihpbl))`` m²/s² (l.761).
PBL_GEOPOTENTIAL_CAP = 50000.0

#: The surface layer's floor on the convective scale ``zwst``,
#: ``zepsr = 1e-10`` (``mo_surface_ocean.f90`` l.302).
CONVECTIVE_SCALE_FLOOR = 1.0e-10

#: ECHAM's initial TKE [m²/s²], θ_v variance [K²] and its standard deviation
#: ``thvsig`` [K] (``ioinitial.f90`` l.418-421): what a cold start begins from.
INITIAL_TKE = 1.0e-4
INITIAL_THV_VARIANCE = 1.0e-4
INITIAL_THV_SIGMA = 1.0e-2


class ClosureConstants(NamedTuple):
    """The neutral stability coefficients ``zshn``, ``zsmn`` and ``zda1``.

    ``zshn = zh1·zh2·√2``, ``zsmn = zshn·zm1·zm2/zm4``, ``zda1 = 1/zsmn³``
    (``vdiff.f90`` l.597-604): Mellor and Yamada's level-2 constants, which
    make ``zsmn³`` the dissipation constant of the TKE budget. Formed from
    the differentiable :class:`~.vertical_diffusion_types.VDiffParameters`
    leaves, never stored.
    """

    shn: jnp.ndarray
    smn: jnp.ndarray
    da1: jnp.ndarray


def closure_constants(params) -> ClosureConstants:
    """``(zshn, zsmn, zda1)`` from ``params.closure_h1 .. closure_m4``."""
    shn = params.closure_h1 * params.closure_h2 * jnp.sqrt(2.0)
    smn = shn * params.closure_m1 * params.closure_m2 / params.closure_m4
    return ClosureConstants(shn=shn, smn=smn, da1=1.0 / smn ** 3)


def _level_index(field):
    """Level index ``k`` shaped to broadcast against ``field`` (axis 0 vertical)."""
    n = field.shape[0]
    return jnp.arange(n).reshape((n,) + (1,) * (field.ndim - 1))


def _hybrid_level_alpha(pressure_half_upper, pressure_half_lower):
    """ECHAM's layer weight ``α_K`` (``auxhyb.f90`` l.87, 136-137).

    ``auxhyb.f90`` actually forms ``palpha = R_d·α_K`` in one step (``zrd =
    rd`` at l.87, then ``palpha(jk) = zrd − ph(jk)·plnpr(jk)/Δp_K`` with
    ``plnpr(jk) = zrd·ln(p_{k+1/2}/p_k)`` at l.136-137, which factors to
    ``R_d·(1 − p_k·ln(p_{k+1/2}/p_k)/Δp_K)``); the ``R_d`` is split back out
    here so both this module's consumers multiply it in themselves, next to
    whichever virtual temperature they pair it with. Deliberately the SAME
    ``log(p_{k+1/2}/p_k)`` form ``auxhyb.f90`` itself uses (not the
    numerically kinder ``log1p(Δp_K/p_k)`` :func:`lowest_level_height_above_surface`
    uses for the surface layer alone): the interior closure this feeds
    (:func:`geopotential_above_surface`) is pinned bit-for-bit against a
    float64 Fortran reference (``closure_test.py``'s harness), so changing
    its arithmetic here — even to an equally valid, more precise reordering
    — would just trade one rounding pattern for a different one against
    that reference, for interior layers thick enough that the swap buys
    nothing anyway (see the surface-layer function's docstring for where it
    does matter).
    """
    return 1.0 - (pressure_half_upper / (pressure_half_lower - pressure_half_upper)
                  * jnp.log(pressure_half_lower / pressure_half_upper))


def geopotential_above_surface(geopotential, virtual_temperature, pressure_half):
    """Geopotential above the surface ``φ − φ_s`` [m²/s²], ECHAM's ``pgeom1``.

    ECHAM hands ``vdiff`` the geopotential relative to the surface
    (``physc.f90`` l.266-274 calls ``geopot`` with zero surface geopotential,
    ``zgeo``). The model geopotential here includes the orography, and its
    lowest level sits ``α_K·R_d·T_v,K`` above the surface in the hydrostatic
    discretisation both the dynamical core and ECHAM use (``geopot.f90``
    l.82: ``phi(nlev) = palpha(nlev)·ptv(nlev) + phis``; ``auxhyb.f90``, see
    :func:`_hybrid_level_alpha`). Subtracting the surface value that
    discretisation implies recovers ECHAM's field exactly, and every
    difference between levels is the model's own.

    Args:
        geopotential: Full-level geopotential [m²/s²], ``(nlev, *h)``.
        virtual_temperature: ``T(1 + vtmpc1·q − q_c − q_i)`` [K],
            ``(nlev, *h)`` (ECHAM's ``ptvm1``).
        pressure_half: Interface pressure [Pa], ``(nlev + 1, *h)``.

    Returns:
        ``pgeom1`` [m²/s²], ``(nlev, *h)``.

    """
    alpha = _hybrid_level_alpha(pressure_half[-2], pressure_half[-1])
    surface = geopotential[-1] - c.rd * alpha * virtual_temperature[-1]
    return geopotential - surface[None]


def lowest_level_height_above_surface(virtual_temperature_lowest,
                                      pressure_half_upper, pressure_half_lower):
    """ECHAM's surface-layer reference height ``pgeom1(klev)/g`` [m].

    The lowest full level's height above the surface, from the hydrostatic
    discretisation of the lowest layer ALONE (``α_K·R_d·T_v,K/g``) — **not**
    half the gap between the two lowest full levels, which is what
    ``height_full[-1] − height_half[-1]`` gives here: ``height_half[-1]`` is
    an extrapolated bottom half-level that keeps layer thickness consistent
    for OTHER diagnostics (``moist_air_state.py``'s output ``height_half``),
    not the ECHAM surface-layer reference height (jax-gcm#1171). ECHAM's own
    surface tiles read ``pgeom1`` the same way — e.g. ``mo_surface_land.
    f90``'s ``zril = pgeom1·zbuoy/(...)``  (l.236) and ``zcdnl = (κ/ln(1 +
    pgeom1/(g·z0)))²`` (l.238), with the matching ``mo_surface_ocean.f90``
    (l.376-383) and ``mo_surface_ice.f90`` (l.636-646) forms.

    This is mathematically exactly :func:`geopotential_above_surface`'s
    lowest row divided by ``g``: that function's own ``surface =
    geopotential[-1] − R_d·α_K·T_v,K`` makes ``(geopotential −
    surface)[-1] == R_d·α_K·T_v,K`` regardless of the model geopotential's
    absolute value, so this needs none of the profile above the lowest
    level. That lets a caller outside this module's ``(nlev, *h)``,
    model-top-first convention — the surface layer's ``(ncol, nlev)``
    ``VDiffState`` arrays, or a physics term whose own level axis runs the
    other way — pass just its own lowest-level slices, in whatever shape
    they broadcast, rather than transpose a whole column through the
    profile this module otherwise uses.

    Deliberately does NOT call :func:`_hybrid_level_alpha` (the interior
    closure's α_K, ``log(p_{K+1/2}/p_K)`` form): for the lowest, thinnest
    layer α_K itself is the small difference between that near-1 log term
    and 1, so computing the ratio first and logging it loses precision in
    float32 — measured as ~1e-3 relative noise in α_K on L47's lowest
    layer (and hence in the surface Richardson number and drag this
    feeds), swamping the ~1e-7 float32 epsilon that should bound it.
    ``log1p`` on the already-small ``Δp_K/p_K`` directly (mathematically
    the same: ``p_{K+1/2}/p_K = 1 + Δp_K/p_K``) keeps this well-conditioned
    and matches a float64 evaluation to float32 epsilon instead. The
    interior closure's own α_K keeps the un-reordered form (see that
    function's docstring) because it is pinned bit-for-bit against a
    float64 Fortran reference and its layers are thick enough that the
    swap buys nothing there.

    Args:
        virtual_temperature_lowest: Lowest full level's virtual temperature
            ``T(1 + vtmpc1·q_v − q_c − q_i)`` [K], ``ECHAM``'s ``ptvm1(klev)``,
            any shape.
        pressure_half_upper: Interface pressure just above the lowest level,
            ``p_{K−1/2}`` [Pa], same shape.
        pressure_half_lower: Surface (lowest) interface pressure,
            ``p_{K+1/2}`` [Pa], same shape.

    Returns:
        ``pgeom1(klev)/g`` [m], unfloored — ECHAM floors the quantities
        DERIVED from it (the stability functions' ``z/z0`` etc.), not
        ``pgeom1`` itself, so callers apply their own floor at their own
        point of use.

    """
    delta_p = pressure_half_lower - pressure_half_upper
    alpha = 1.0 - (pressure_half_upper / delta_p
                   * jnp.log1p(delta_p / pressure_half_upper))
    return c.rd * alpha * virtual_temperature_lowest / c.grav


class PblExtension(NamedTuple):
    """ECHAM's PBL extension (``vdiff.f90`` l.737-765), per column.

    The indices are 0-based full levels counted from the model top (ECHAM's
    ``jk − 1``); ``nlev − 1`` is the lowest level, ECHAM's ``klev``.
    """

    level: jnp.ndarray              # ihpbl: the higher of the two below
    convective_level: jnp.ndarray   # ihpblc
    dynamic_level: jnp.ndarray      # ihpbld
    geopotential: jnp.ndarray       # zghabl = min(50000, pgeom1(ihpbl)) [m²/s²]
    dynamic_height: jnp.ndarray     # zhdyn [m]


def pbl_extension(pgeom1, dry_static_energy, friction_velocity, coriolis, params):
    """Diagnose the PBL extension ``ihpbl`` and its geopotential ``zghabl``.

    ``vdiff.f90`` l.737-765::

        zhdyn  = min(pgeom1(top)/g, zchneu·u*/max(|f|, zepcor))
        ihpblc = lowest level above the lowest one whose dry static energy
                 exceeds the lowest level's (klev when there is none)
        ihpbld = lowest level at or above zhdyn
        ihpbl  = min(ihpblc, ihpbld)          (the higher of the two)
                 ihpblc                       under lrce (l.756-760)
        zghabl = min(50000, pgeom1(ihpbl))

    ``u*`` is the previous step's grid-mean friction velocity: ECHAM's
    ``ustarm`` comes from the surface call of the step before
    (l.732-735). ``params.lrce`` is ECHAM's RCE switch, under which the
    convective level alone sets the extension.

    Args:
        pgeom1: Geopotential above the surface [m²/s²], ``(nlev, *h)``.
        dry_static_energy: ``zcptgz = pgeom1 + c_pd·T·(1 + vtmpc2·q)``
            [J/kg], ``(nlev, *h)``.
        friction_velocity: Previous-step grid-mean ``u*`` [m/s], ``(*h)``.
        coriolis: Coriolis parameter ``f`` [1/s], ``(*h)``.
        params: :class:`~.vertical_diffusion_types.VDiffParameters`
            (``pbl_dynamic_height_factor`` = ECHAM's ``zchneu``, ``lrce``).

    Returns:
        :class:`PblExtension`.

    """
    nlev = pgeom1.shape[0]
    g = c.grav
    corr = jnp.maximum(jnp.abs(coriolis), CORIOLIS_FLOOR)
    height = pgeom1 / g
    zhdyn = jnp.minimum(height[0],
                        params.pbl_dynamic_height_factor * friction_velocity / corr)
    k = _level_index(pgeom1)
    above_lowest = k < nlev - 1
    convective = above_lowest & (dry_static_energy - dry_static_energy[-1][None] > 0.0)
    # zhdyn never exceeds the top level's height, so the top level always
    # qualifies (ECHAM evaluates the same expression twice and gets exactly
    # 0 there); stated outright rather than left to two roundings agreeing.
    dynamic = above_lowest & ((height - zhdyn[None] >= 0.0) | (k == 0))
    # ECHAM scans upwards from klev-1 and keeps the first level that meets the
    # condition, i.e. the largest index; klev when none does.
    ihpblc = jnp.max(jnp.where(convective, k, -1), axis=0)
    ihpblc = jnp.where(ihpblc < 0, nlev - 1, ihpblc)
    ihpbld = jnp.max(jnp.where(dynamic, k, -1), axis=0)
    ihpbld = jnp.where(ihpbld < 0, nlev - 1, ihpbld)
    ihpbl = jnp.where(params.lrce, ihpblc, jnp.minimum(ihpblc, ihpbld))
    geopotential = jnp.take_along_axis(pgeom1, ihpbl[None], axis=0)[0]
    return PblExtension(
        level=ihpbl, convective_level=ihpblc, dynamic_level=ihpbld,
        geopotential=jnp.minimum(PBL_GEOPOTENTIAL_CAP, geopotential),
        dynamic_height=zhdyn)


def mixing_length(pgeom1, pbl_level, params):
    """Mixing length of every interior interface [m], ``(nlev − 1, *h)``.

    ``vdiff.f90`` l.801-817: Blackadar's length ``zmix = κz/(1 + κz/λ)``
    with ``z`` the interface height (the mean of the two levels' heights).
    At and below the PBL extension (interface ``k`` with ``k >= ihpbl``)
    ``λ = clam``; above it ``λ`` decays with height to 1 m following
    Holtslag and Boville (1993)::

        λ(z) = 1 + (clam − 1)·exp(1 − φ_k/φ_ihpbl)

    where ``φ_k`` is the geopotential of the level above the interface.

    Args:
        pgeom1: Geopotential above the surface [m²/s²], ``(nlev, *h)``.
        pbl_level: ``ihpbl`` (0-based), ``(*h)``.
        params: ``mixing_length_asymptote`` = ECHAM's ``clam``.

    Returns:
        ``zmix`` [m], ``(nlev − 1, *h)``.

    """
    clam = params.mixing_length_asymptote
    cons2 = 0.5 * c.karman_const / c.grav                      # zcons2
    phi_pbl = jnp.take_along_axis(pgeom1, pbl_level[None], axis=0)
    upper = pgeom1[:-1]
    hexp = jnp.exp(1.0 - upper / phi_pbl)                       # zhexp
    lam = 1.0 + (clam - 1.0) * hexp                             # zlam (zzzlam = 1)
    k = _level_index(upper)
    cons23 = jnp.where(k >= pbl_level[None], cons2 / clam, cons2 / lam)
    two_geo = pgeom1[:-1] + pgeom1[1:]                          # z2geomf
    return cons2 * two_geo / (1.0 + cons23 * two_geo)


def louis_stability(ri, zmix, pgeom1, params):
    """Louis (1979) stability functions ``(zsm, zsh)`` [m], ``(nlev − 1, *h)``.

    ``vdiff.f90`` l.819-832, the neutral values times the mixing length::

        unstable (Ri < 0):
            zucf = 1/(1 + 3·b·c·g²·l²·√(|Ri|·((φ_k/φ_{k+1})^(1/3) − 1)³
                                             /(φ_k − φ_{k+1})³/φ_{k+1}))
            zsh = zshn·(1 − 3b·Ri·zucf)·l      zsm = zsmn·(1 − 2b·Ri·zucf)·l
        stable (Ri >= 0):
            zsh = zshn/(1 + 2b·Ri·√(1 + Ri))·l zsm = zsmn/(1 + 2b·Ri/√(1 + Ri))·l

    with ``b = cb``, ``c = cc`` (the surface layer's ``louis_cb``/``louis_cc``,
    one pair in ECHAM as here). Each branch sees ``Ri`` only on its own side
    of zero, so the branch ``where`` discards never meets the root at
    ``Ri = 0`` or a negative base, and the gradient stays finite there.

    Args:
        ri: Richardson number ``zri`` of the interior interfaces,
            ``(nlev − 1, *h)``.
        zmix: Mixing length [m], ``(nlev − 1, *h)``.
        pgeom1: Geopotential above the surface [m²/s²], ``(nlev, *h)``.
        params: ``louis_cb``, ``louis_cc`` and the closure constants.

    Returns:
        ``(zsm, zsh)`` [m].

    """
    k_ = closure_constants(params)
    cb, cc = params.louis_cb, params.louis_cc
    g = c.grav
    unstable = ri < 0.0
    ri_u = jnp.where(unstable, ri, -1.0)
    ri_s = jnp.where(unstable, 0.0, ri)
    upper, lower = pgeom1[:-1], pgeom1[1:]
    shape = (((upper / lower) ** (1.0 / 3.0) - 1.0) / (upper - lower)) ** 3 / lower
    ucf = 1.0 / (1.0 + 3.0 * cb * cc * g * g * zmix * zmix
                 * jnp.sqrt(jnp.abs(ri_u) * shape))
    sh_unstable = k_.shn * (1.0 - 3.0 * cb * ri_u * ucf) * zmix
    sm_unstable = k_.smn * (1.0 - 2.0 * cb * ri_u * ucf) * zmix
    root = jnp.sqrt(1.0 + ri_s)
    sh_stable = k_.shn / (1.0 + 2.0 * cb * ri_s * root) * zmix
    sm_stable = k_.smn / (1.0 + 2.0 * cb * ri_s / root) * zmix
    return (jnp.where(unstable, sm_unstable, sm_stable),
            jnp.where(unstable, sh_unstable, sh_stable))


def _sqrt_nonnegative(x):
    """``√x`` for ``x >= 0`` whose derivative at ``x = 0`` is 0, not ``inf``.

    The double-``where`` keeps a cold-started or decayed TKE of exactly zero
    (the carry's zero seed, a companion scheme's zero placeholder) from
    poisoning the reverse pass, which ``sqrt(maximum(x, 0))`` does not
    (its tie at 0 still multiplies ``sqrt'(0) = inf``).
    """
    positive = x > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)


def tke_source_update(prev_tke, shear, buoyancy, zsm, zsh, zmix, dt, params):
    """Implicit TKE source and dissipation, ``vdiff.f90`` l.837-844.

    With ``e`` the TKE, the budget ``de/dt = √e·(S²·zsm − N²·zsh) −
    e^{3/2}/(zda1·l)`` is integrated implicitly in ``u = √e``, a quadratic
    whose positive root is::

        zzb    = zshear·zsm − zbuoy·zsh
        zdisl  = zda1·zmix/Δt
        zktest = 1 + (zzb·Δt + 2·√e_old)/zdisl
        e_new  = ztkemin                              if zktest <= 1
                 max(ztkemin, (zdisl·(√zktest − 1))²) otherwise

    so the dissipation constant is ``1/zda1 = zsmn³ ≈ 0.167``.

    Args:
        prev_tke: TKE of the interior interfaces at the previous step
            [m²/s²], ``(nlev − 1, *h)``.
        shear, buoyancy: ``zshear``, ``zbuoy`` [1/s²], ``(nlev − 1, *h)``.
        zsm, zsh: Stability functions times the mixing length [m].
        zmix: Mixing length [m].
        dt: Time step [s].
        params: ``totte_min`` (ECHAM's ``ztkemin``) and the closure constants.

    Returns:
        Post-source TKE ``ztkevn`` [m²/s²], ``(nlev − 1, *h)``.

    """
    da1 = closure_constants(params).da1
    zzb = shear * zsm - buoyancy * zsh
    zdisl = da1 * zmix / dt
    zktest = 1.0 + (zzb * dt + 2.0 * _sqrt_nonnegative(prev_tke)) / zdisl
    growing = zktest > 1.0
    root = jnp.sqrt(jnp.where(growing, zktest, 1.0))
    return jnp.where(growing,
                     jnp.maximum(params.totte_min, (zdisl * (root - 1.0)) ** 2),
                     params.totte_min)


def thv_variance_source_update(prev_variance, thv_gradient, zsh, tke_sqrt, zmix, dt,
                               params):
    """Explicit θ_v-variance production and dissipation, ``vdiff.f90`` l.857-860.

    ::

        zthvprod = 2·zsh·√e·(∂θ_v/∂z)²
        zthvdiss = var·√e/(zda1·zmix)
        var_new  = max(ztkemin, var + (zthvprod − zthvdiss)·Δt)

    with ``√e = ztkesq = √max(ztkemin, e_old)``, the previous TKE (l.849),
    not the post-source one.

    Args:
        prev_variance: θ_v variance of the interior interfaces at the previous
            step [K²], ``(nlev − 1, *h)``.
        thv_gradient: ``zthvirdif = ∂θ_v/∂z`` [K/m], ``(nlev − 1, *h)``.
        zsh: Heat stability function times the mixing length [m].
        tke_sqrt: ``ztkesq`` [m/s].
        zmix: Mixing length [m].
        dt: Time step [s].
        params: ``totte_min`` and the closure constants.

    Returns:
        Post-source variance [K²], ``(nlev − 1, *h)``.

    """
    da1 = closure_constants(params).da1
    production = 2.0 * zsh * tke_sqrt * thv_gradient ** 2
    dissipation = prev_variance * tke_sqrt / (da1 * zmix)
    return jnp.maximum(params.totte_min,
                       prev_variance + (production - dissipation) * dt)


def virtual_potential_temperature_gradient(temperature, qv, qc, qi, pressure_full,
                                           geopotential):
    """``zthvirdif = ∂θ_v/∂z`` of the interior interfaces [K/m], ``vdiff.f90`` l.791.

    ``θ_v = T·(p₀/p)^κ·(1 + vtmpc1·q − q_c − q_i)`` (``ztvir1``, l.671), the
    difference of the two levels over their geopotential thickness times
    ``g``. Condensate loading is part of the reference definition: it is what
    makes a cloud-topped boundary layer's variance differ from a clear one.

    Args:
        temperature, qv, qc, qi: Full-level state [K, kg/kg], ``(nlev, *h)``.
        pressure_full: Full-level pressure [Pa], ``(nlev, *h)``.
        geopotential: Full-level geopotential [m²/s²], ``(nlev, *h)``.

    Returns:
        ``(nlev − 1, *h)`` [K/m].

    """
    exner = (c.p0 / pressure_full) ** (c.rd / c.cpd)
    thv = temperature * exner * (1.0 + c.vtmpc1 * qv - (qc + qi))
    zhh = geopotential[:-1] - geopotential[1:]
    return (thv[:-1] - thv[1:]) / zhh * c.grav


class InteriorClosure(NamedTuple):
    """ECHAM's closure of the interior interfaces, ``vdiff.f90`` l.730-860.

    Interface arrays have ``nlev − 1`` rows (the interior interfaces, row
    ``k`` between full levels ``k`` and ``k + 1``); the conductances are
    ``zcf/(cvdifts·Δt)`` [Pa/s] (see the module docstring).
    """

    pbl: PblExtension
    geopotential_above_surface: jnp.ndarray   # pgeom1 [m²/s²] (nlev, *h)
    mixing_length: jnp.ndarray                # zmix [m]
    sm: jnp.ndarray                           # zsm [m]
    sh: jnp.ndarray                           # zsh [m]
    km: jnp.ndarray                           # zsm·ztkesq [m²/s]
    kh: jnp.ndarray                           # zsh·ztkesq [m²/s]
    tke: jnp.ndarray                          # ztkevn, post-source [m²/s²]
    thv_variance: jnp.ndarray                 # zthvvar, post-source [K²]
    conductance_momentum: jnp.ndarray         # zcfm  [Pa/s]
    conductance_heat: jnp.ndarray             # zcfh  [Pa/s]
    conductance_tke: jnp.ndarray              # zcdum [Pa/s]
    conductance_thv: jnp.ndarray              # zcthv [Pa/s]


def interior_closure(*, temperature, qv, qc, qi, pressure_full, pressure_half,
                     geopotential, buoyancy, shear, ri, prev_tke, prev_thv_variance,
                     friction_velocity, coriolis, dt, params) -> InteriorClosure:
    """ECHAM's closure of every interior interface, ``vdiff.f90`` l.730-860.

    Args:
        temperature, qv, qc, qi: Full-level state [K, kg/kg], ``(nlev, *h)``.
        pressure_full: Full-level pressure [Pa], ``(nlev, *h)``.
        pressure_half: Interface pressure [Pa], ``(nlev + 1, *h)``.
        geopotential: Full-level geopotential (orography included) [m²/s²],
            ``(nlev, *h)``.
        buoyancy, shear, ri: ``zbuoy``, ``zshear`` and ``zri`` of the interior
            interfaces (:mod:`.moist_buoyancy`), ``(nlev − 1, *h)``.
        prev_tke, prev_thv_variance: Previous-step TKE [m²/s²] and θ_v variance
            [K²] of the interior interfaces, ``(nlev − 1, *h)``.
        friction_velocity: Previous-step grid-mean ``u*`` [m/s], ``(*h)``.
        coriolis: Coriolis parameter [1/s], ``(*h)``.
        dt: Time step [s].
        params: :class:`~.vertical_diffusion_types.VDiffParameters`.

    Returns:
        :class:`InteriorClosure`.

    """
    g = c.grav
    x = qc + qi
    tv = temperature * (1.0 + c.vtmpc1 * qv - x)                 # ptvm1
    pgeom1 = geopotential_above_surface(geopotential, tv, pressure_half)
    dry_static_energy = pgeom1 + temperature * c.cpd * (1.0 + c.vtmpc2 * qv)   # zcptgz
    pbl = pbl_extension(pgeom1, dry_static_energy, friction_velocity, coriolis, params)
    zmix = mixing_length(pgeom1, pbl.level, params)
    zsm, zsh = louis_stability(ri, zmix, pgeom1, params)

    # A carried value of exactly 0 is an interface with no previous step (the
    # carry's seed; the closure floors every stepped value at ztkemin > 0).
    # ECHAM starts from tkem1 = thvvar = 1e-4 (ioinitial.f90:418-420) and, on
    # its first step (lstart), forms the coefficients from the post-source TKE
    # rather than the previous one (vdiff.f90 l.845-847).
    cold = prev_tke <= 0.0
    tke_new = tke_source_update(jnp.where(cold, INITIAL_TKE, prev_tke), shear,
                                buoyancy, zsm, zsh, zmix, dt, params)
    prev_thv_variance = jnp.where(prev_thv_variance <= 0.0, INITIAL_THV_VARIANCE,
                                  prev_thv_variance)
    tke_sqrt = jnp.sqrt(jnp.maximum(params.totte_min,
                                    jnp.where(cold, tke_new, prev_tke)))  # ztkesq

    zhh = geopotential[:-1] - geopotential[1:]
    zalf = pressure_half[1:-1] / (0.5 * (tv[:-1] + tv[1:]) * zhh * c.rd)
    km = zsm * tke_sqrt
    kh = zsh * tke_sqrt
    g2alf = g * g * zalf
    # zcdum/zcthv: the coefficients rescaled from the previous TKE to the
    # post-source one (l.855-856), the transport coefficients of TKE and of the
    # variance.
    rescale = jnp.sqrt(tke_new) / tke_sqrt

    thv_gradient = virtual_potential_temperature_gradient(
        temperature, qv, qc, qi, pressure_full, geopotential)
    var_new = thv_variance_source_update(prev_thv_variance, thv_gradient, zsh,
                                         tke_sqrt, zmix, dt, params)
    return InteriorClosure(
        pbl=pbl, geopotential_above_surface=pgeom1, mixing_length=zmix,
        sm=zsm, sh=zsh, km=km, kh=kh, tke=tke_new, thv_variance=var_new,
        conductance_momentum=g2alf * km, conductance_heat=g2alf * kh,
        conductance_tke=g2alf * km * rescale, conductance_thv=g2alf * kh * rescale)


def surface_tke_boundary(*, momentum_exchange, heat_exchange, wind_speed,
                         thv_surface, thv_air, fraction, pbl_geopotential,
                         lowest_geopotential, params):
    """TKE of the surface layer, the lowest interface's value ``tkevn``.

    Mailhot and Benoit (1982) as ECHAM's surface tiles form it
    (``mo_surface_ocean.f90::precalc_ocean`` l.387-388 and 438-456; the same
    statements in ``precalc_land`` and ``precalc_ice``), fraction-weighted
    over the tiles (``mo_surface.f90::surface_box_average``)::

        u*²    = C_m|U|·√zdu2                      zdu2 = max(1, |U|²)
        zwst   = max(0, θ_v,s − θ_v,1)·√zdu2/θ̄_v   θ̄_v = (θ_v,1 + θ_v,s)/2
        zconvs = (zwst·C_h·zghabl)^(1/3)            only where zwst > 1e-10
        L      = u*³/(κ·g·zwst·C_h)
        zstabf = min(3·zustf, (z_1/L)^(2/3))
        tkev   = (zustf + zstabf)·u*² + zwstf·zconvs²

    with ``zustf = 1/zsmn²``, ``zwstf = surface_tke_convective_factor``
    (0.2), ``C_h = C_h|U|/√zdu2`` the tile's heat transfer coefficient and
    ``zghabl`` the geopotential of the PBL extension. ECHAM's ``u*`` here is
    ``zcfm2m``'s, which differs from ``C_m|U|`` only for a roughness above
    ``zepz0o = 2`` m; no jcm tile has one.

    Tile arrays have the tiles on the last axis, ``(*h, ntile)``; per-column
    arrays are ``(*h)``.

    Args:
        momentum_exchange: ``C_m|U|`` per tile [m/s].
        heat_exchange: ``C_h|U|`` per tile [m/s].
        wind_speed: Lowest-level wind speed [m/s], ``(*h)``.
        thv_surface: Virtual potential temperature of each tile's surface
            [K] (ECHAM's ``ztvw``/``ztvi``/``ztvl``).
        thv_air: Virtual potential temperature of the lowest level [K]
            (``ztvir1(klev)``), ``(*h)``.
        fraction: Tile fractions [-].
        pbl_geopotential: ``zghabl`` [m²/s²], ``(*h)``.
        lowest_geopotential: ``pgeom1(klev)`` [m²/s²], ``(*h)``.
        params: ``surface_tke_convective_factor`` and the closure constants.

    Returns:
        ``tkevn`` [m²/s²], ``(*h)``.

    """
    ustf = 1.0 / closure_constants(params).smn ** 2
    wstf = params.surface_tke_convective_factor
    sqrt_du2 = jnp.sqrt(jnp.maximum(wind_speed ** 2, 1.0))[..., None]
    thv_air = thv_air[..., None]
    ustar2 = momentum_exchange * sqrt_du2
    dthv = jnp.maximum(0.0, thv_surface - thv_air)
    wst = dthv * sqrt_du2 / (0.5 * (thv_air + thv_surface))
    ch = heat_exchange / sqrt_du2
    active = wst > CONVECTIVE_SCALE_FLOOR
    wst_a = jnp.where(active, wst, 1.0)
    flux = wst_a * ch
    convs2 = jnp.where(active, (flux * pbl_geopotential[..., None]) ** (2.0 / 3.0), 0.0)
    ustar3 = ustar2 * jnp.sqrt(ustar2)
    monin_obukhov = ustar3 / (c.karman_const * c.grav * flux)
    stabf = jnp.minimum(
        3.0 * ustf,
        (lowest_geopotential[..., None] / (c.grav * monin_obukhov)) ** (2.0 / 3.0))
    stabf = jnp.where(active, stabf, 0.0)
    tke = (ustf + stabf) * ustar2 + wstf * convs2
    return jnp.sum(fraction * tke, axis=-1)
