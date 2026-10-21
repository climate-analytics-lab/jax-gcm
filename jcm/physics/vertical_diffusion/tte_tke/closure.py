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
