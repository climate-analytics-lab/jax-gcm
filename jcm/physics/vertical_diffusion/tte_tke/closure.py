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


def geopotential_above_surface(geopotential, virtual_temperature, pressure_half):
    """Geopotential above the surface ``φ − φ_s`` [m²/s²], ECHAM's ``pgeom1``.

    ECHAM hands ``vdiff`` the geopotential relative to the surface
    (``physc.f90`` calls ``geopot`` with zero surface geopotential). The model
    geopotential here includes the orography, and its lowest level sits
    ``α_K·R_d·T_v,K`` above the surface in the hydrostatic discretisation both
    the dynamical core and ECHAM use (``geopot.f90``, ``auxhyb.f90``):
    ``α_K = 1 − p_{K−1/2}·ln(p_{K+1/2}/p_{K−1/2})/Δp_K``. Subtracting the
    surface value that discretisation implies recovers ECHAM's field exactly,
    and every difference between levels is the model's own.

    Args:
        geopotential: Full-level geopotential [m²/s²], ``(nlev, *h)``.
        virtual_temperature: ``T(1 + vtmpc1·q − q_c − q_i)`` [K],
            ``(nlev, *h)`` (ECHAM's ``ptvm1``).
        pressure_half: Interface pressure [Pa], ``(nlev + 1, *h)``.

    Returns:
        ``pgeom1`` [m²/s²], ``(nlev, *h)``.

    """
    p_up = pressure_half[-2]
    p_lo = pressure_half[-1]
    alpha = 1.0 - p_up / (p_lo - p_up) * jnp.log(p_lo / p_up)
    surface = geopotential[-1] - c.rd * alpha * virtual_temperature[-1]
    return geopotential - surface[None]


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
        zghabl = min(50000, pgeom1(ihpbl))

    ``u*`` is the previous step's grid-mean friction velocity: ECHAM's
    ``ustarm`` comes from the surface call of the step before
    (l.732-735). ECHAM's RCE switch ``lrce``, which drops the dynamic height,
    has no counterpart: jcm's RCE testbed is not an ECHAM ``lrce`` run.

    Args:
        pgeom1: Geopotential above the surface [m²/s²], ``(nlev, *h)``.
        dry_static_energy: ``zcptgz = pgeom1 + c_pd·T·(1 + vtmpc2·q)``
            [J/kg], ``(nlev, *h)``.
        friction_velocity: Previous-step grid-mean ``u*`` [m/s], ``(*h)``.
        coriolis: Coriolis parameter ``f`` [1/s], ``(*h)``.
        params: :class:`~.vertical_diffusion_types.VDiffParameters`
            (``pbl_dynamic_height_factor`` = ECHAM's ``zchneu``).

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
    ihpbl = jnp.minimum(ihpblc, ihpbld)
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
