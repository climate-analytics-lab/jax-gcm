"""Lott & Miller (1997) + Lott (1999) sub-grid orographic GW drag.

Faithful JAX port of ECHAM ``mo_ssodrag.f90`` (atm_phy_echam version,
1564 lines). References:

- Lott & Miller (1997) QJRMS 123: 101-127 (anisotropic hill drag,
  blocked flow)
- Lott (1999) MWR 127: 788-801 (blocking-depth refinement)

The port targets the production-default control flow:

- ``gkdrag = 0.2``  wave drag enabled
- ``gkwake = 1.0``  blocked-flow wake drag enabled
- ``gklift = 0.0``  mountain-lift branch DISABLED (skipped entirely)
- ``nktopg = 1``  top level for orography effects
- ``ntop   = 1``  top level for stress profile

The mountain-lift branch (``orolift``) is NOT ported because the default
ECHAM config disables it. It can be added later if a non-zero ``gklift``
becomes interesting.

Subroutine map (Fortran → JAX):

- ``ssodrag``   → :func:`sso_drag` (entry)
- ``orodrag``   → ``_orodrag`` (driver: setup + stress + profile + tendencies)
- ``orosetup``  → ``_orosetup`` (geometry, mean low-level flow, blocking)
- ``gwstress``  → ``_gwstress`` (surface stress)
- ``gwprofil``  → ``_gwprofil`` (vertical stress profile + saturation)
- ``orolift``   → not ported (gklift=0 in production)

The half-level/full-level convention follows the Fortran: half levels run
``[0..nlev]`` (length ``nlev+1``); full levels run ``[0..nlev-1]`` (length
``nlev``). Index 0 = top, index ``nlev`` = surface (half) or
``nlev-1`` = lowest full level.
"""
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import tree_math

# ICON physical constants (mo_physical_constants.f90)
_GRAV = 9.80665
_RD = 287.04
_CPD = 1004.64
_PI = 3.141592653589793

# Module-level PARAMETER constants from mo_echam_sso_config.
_GFRCRIT = 0.5     # critical non-dim mountain height
_GRCRIT = 0.25     # critical Richardson number
_GRAHILO = 1.0     # trapped-wave fraction
_GSIGCR = 0.80     # min blocked-flow depth
_GSSEC = 1.0e-4    # min low-level B-V freq
_GTSEC = 1.0e-5    # min anisotropy / GW stress
_GVSEC = 0.10      # min ulow


@tree_math.struct
class SSOParameters:
    """Parameters for the Lott & Miller (1997) SSO drag scheme."""

    gpicmea: jnp.ndarray
    gstd: jnp.ndarray
    gkdrag: jnp.ndarray
    gkwake: jnp.ndarray
    gklift: jnp.ndarray
    nktopg: jnp.ndarray
    ntop: jnp.ndarray

    @classmethod
    def default(
        cls,
        gpicmea: float = 1.0,
        gstd: float = 1.0,
        gkdrag: float = 0.2,
        gkwake: float = 1.0,
        gklift: float = 0.0,
        nktopg: int = 1,
        ntop: int = 1,
    ) -> "SSOParameters":
        return cls(
            gpicmea=jnp.asarray(gpicmea),
            gstd=jnp.asarray(gstd),
            gkdrag=jnp.asarray(gkdrag),
            gkwake=jnp.asarray(gkwake),
            gklift=jnp.asarray(gklift),
            nktopg=jnp.asarray(nktopg),
            ntop=jnp.asarray(ntop),
        )


class SSOState(NamedTuple):
    """Diagnostic outputs from the SSO scheme."""

    u_stress: jnp.ndarray
    v_stress: jnp.ndarray
    dissip_total: jnp.ndarray


class SSOTendencies(NamedTuple):
    """Tendencies from the SSO scheme."""

    dudt: jnp.ndarray
    dvdt: jnp.ndarray
    dissip: jnp.ndarray


# ---------------------------------------------------------------------------
# orosetup — geometry, mean low-level flow, blocking
# ---------------------------------------------------------------------------

def _orosetup(
    paphm1: jnp.ndarray,    # (nlev+1,) half pressure
    papm1: jnp.ndarray,     # (nlev,)   full pressure
    pmair: jnp.ndarray,     # (nlev,)   air mass
    pum1: jnp.ndarray,      # (nlev,)
    pvm1: jnp.ndarray,      # (nlev,)
    ptm1: jnp.ndarray,      # (nlev,)
    phgeo: jnp.ndarray,     # (nlev,)   height above surface
    pmea: jnp.ndarray,      # scalar
    ppic: jnp.ndarray,      # scalar
    pval: jnp.ndarray,      # scalar
    ptheta: jnp.ndarray,    # scalar (degrees)
    pgam: jnp.ndarray,      # scalar
    nktopg: int,
):
    """Set up orographic geometry and low-level flow.

    Returns a dict of intermediate quantities used by ``_gwstress``,
    ``_gwprofil``, and ``_orodrag``.

    Index convention: 0-based, top=0. Fortran's ``klev+1`` (surface half
    level) maps to index ``nlev`` in 0-based half-level arrays.
    """
    nlev = pum1.shape[0]
    ilevh = nlev // 3   # Fortran's ilevh = klev/3 (integer divide)

    # Anisotropy guard (line 647)
    pgam_safe = jnp.maximum(pgam, _GTSEC)

    # ----- Determine levels kknu, kknu2, kknub via 3 separate sweeps -------
    # Fortran scans jk = klev .. ilevh (downward). We work 0-based with
    # klev = nlev-1 (last full level) and ilevh = ilevh-1 in 0-based... but
    # the Fortran ilevh stays as a pure integer index - same in 0-based.
    # Convention: jk_1based - 1 = jk_0based, so "jk=klev" → "jk=nlev-1".

    # Fortran: ll1[jk] = (phgeo[jk] > zhcrit), and kknu = jk where ll1
    # changes between jk and jk+1 going DOWN. The semantically equivalent
    # 0-based statement: kknu = the deepest h (largest h, lowest altitude)
    # where above[h] is True. ilevh-as-1-based-count → use ilevh-1 as the
    # 0-based fallback index.
    ilevh_idx = ilevh - 1   # 0-based equivalent of Fortran's 1-based ilevh
    def _lowest_above(thresh: jnp.ndarray) -> jnp.ndarray:
        above = phgeo > thresh   # bool (nlev,)
        # Last True going top-down = deepest True. Compute via reversed argmax.
        rev_first = jnp.argmax(above[::-1].astype(jnp.int32))
        idx_last = nlev - 1 - rev_first
        kknu = jnp.where(jnp.any(above), idx_last, nlev - 1)
        # Floor at ilevh (line 679: "IF(.NOT.ll1(jl,ilevh))kknu(jl)=ilevh")
        kknu = jnp.where(above[ilevh_idx], kknu, ilevh_idx)
        return kknu

    kknu = _lowest_above(ppic - pval)            # peaks elevation
    kknu2 = _lowest_above(ppic - pmea)           # peaks above mean
    kknub = _lowest_above(jnp.minimum(ppic - pmea, pmea - pval))

    # Bound by nktopg (line 717-720). Fortran uses MIN, then sets kknul=klev.
    kknu = jnp.minimum(kknu, nktopg - 1)         # 1-based nktopg → 0-based
    kknu2 = jnp.minimum(kknu2, nktopg - 1)
    kknub = jnp.minimum(kknub, nktopg - 1)
    kknul = nlev - 1

    # ----- kkcrit: top of low-level flow, where p/p_sfc >= gsigcr (line 670) -
    # Find lowest 0-based jk such that paphm1[jk]/paphm1[nlev] >= gsigcr.
    # Fortran initialises kkcrit = klev = nlev-1 (1-based) = nlev-2 (0-based)?
    # Looking at line 736: kkcrith(:) = klev (initialised). And kcrit(:) = 1.
    # Then kkcrit is set inside the loop. Fortran kkcrit is used in
    # `IF(kcrit(jl) >= kkcrith(jl)) kcrit=1` (line 948), but kkcrit itself
    # isn't read elsewhere in this code (the variable kcrit is used).
    # We skip computing kkcrit since it's unused.

    # ----- Density + stability at half levels (lines 745-761) ---------------
    # rho[jk] = 2*paphm1[jk]/(rd*(T[jk]+T[jk-1])) for jk = 1..nlev (1-based)
    # In 0-based, this maps to half-levels [1..nlev], where each uses
    # T_full[jk] and T_full[jk-1]. Note 1-based jk means full-level index jk
    # which in 0-based is jk-1.
    # Fortran half index 'jk' takes T(jk) and T(jk-1). In 0-based half index
    # h, these are full-levels h-1 and h-2? Let me re-read:
    #
    #   prho(jl,jk)=2._wp*paphm1(jl,jk)*zcons1/(ptm1(jl,jk)+ptm1(jl,jk-1))
    #
    # paphm1 has shape (klev+1) and is half-level. ptm1 has shape (klev) and
    # is full-level. ptm1(jk) and ptm1(jk-1) correspond to the full levels
    # immediately above and below the half level jk. With jk=2..klev
    # (Fortran 1-based), the half-level is between full levels jk and jk-1.
    #
    # In 0-based: half-level index h (h=1..nlev-1) is between full levels
    # h and h-1. So:
    #   rho[h] = 2*paphm1[h] / (rd*(T[h] + T[h-1]))    for h = 1..nlev-1
    #
    # And similarly for stab. Half levels 0 and nlev are not set by this
    # loop — they are handled separately (initialised to 0 / 9999, etc.).
    h_idx = jnp.arange(1, nlev)
    zdp_half = papm1[h_idx] - papm1[h_idx - 1]         # (nlev-1,)
    rho_half = (2.0 * paphm1[h_idx] / _RD
                / (ptm1[h_idx] + ptm1[h_idx - 1]))
    stab_half = (2.0 * _GRAV ** 2 / _CPD / (ptm1[h_idx] + ptm1[h_idx - 1])
                 * (1.0 - _CPD * rho_half * (ptm1[h_idx] - ptm1[h_idx - 1])
                    / zdp_half))
    stab_half = jnp.maximum(stab_half, _GSSEC)
    # Pad to length nlev+1 with the surface and top initial values.
    prho = jnp.concatenate([jnp.zeros(1), rho_half, jnp.zeros(1)])
    pstab = jnp.concatenate([jnp.zeros(1), stab_half, jnp.zeros(1)])
    zdp = jnp.concatenate([jnp.zeros(1), zdp_half])     # full-level diff
    # zdp has length nlev; index 0 unused, indices 1..nlev-1 = papm1[k]-papm1[k-1].

    # ----- Mass-weighted low-level mean (lines 767-797) ---------------------
    # Fortran loops jk = klev..ilevh (1-based, going down), with the IF
    # condition jk >= kknu2 AND jk <= kknul further restricting. In 0-based:
    # h = nlev-1..ilevh-1, with kknu2 <= h <= kknul.
    h_arr = jnp.arange(nlev)
    levmask = ((h_arr >= kknu2) & (h_arr <= kknul) & (h_arr >= ilevh_idx))
    # weights = pmair * mask
    w = pmair * levmask.astype(pmair.dtype)
    zmair = jnp.sum(w)
    pulow = jnp.sum(pum1 * w) / zmair
    pvlow = jnp.sum(pvm1 * w) / zmair
    # stab and rho at klev+1 (surface half) are layer-mass-weighted averages
    # of the half-level values. Fortran uses pstab[jk] and prho[jk] at the
    # FULL-level index jk, but they're only filled at HALF levels above. So
    # the indexing in Fortran here aligns half-level h with the layer above
    # it. In 0-based: at full-level k, take half-level value at h=k.
    # This is delicate; let's match the Fortran semantics literally:
    #   pstab[klev+1] += pstab[jk] * pmair[jk]    (jk = full-level index)
    # In 0-based: pstab[nlev] += pstab[k] * pmair[k] for k = kknu2..nlev-1.
    # But pstab in Fortran has shape (klev+1) and is on half levels.
    # Indexing pstab[jk] when jk is a full-level index pulls the half-level
    # value at half-index jk (= full level k in 0-based, half-level k means
    # the half level between full levels k-1 and k).
    sum_stab = jnp.sum(pstab[:nlev] * w)
    sum_rho = jnp.sum(prho[:nlev] * w)
    pstab_sfc = sum_stab / zmair
    prho_sfc = sum_rho / zmair
    pstab = pstab.at[nlev].set(pstab_sfc)
    prho = prho.at[nlev].set(prho_sfc)

    znorm = jnp.maximum(jnp.sqrt(pulow ** 2 + pvlow ** 2), _GVSEC)
    pvph_sfc = znorm

    # ----- Anisotropy & wave-stress orientation (lines 802-819) -------------
    zu = jnp.where((pulow > -_GVSEC) & (pulow < _GVSEC),
                   pulow + 2.0 * _GVSEC, pulow)
    zphi = jnp.arctan(pvlow / zu)
    psi_sfc = ptheta * _PI / 180.0 - zphi
    zb = 1.0 - 0.18 * pgam_safe - 0.04 * pgam_safe ** 2
    zc = 0.48 * pgam_safe + 0.30 * pgam_safe ** 2
    pd1 = zb - (zb - zc) * jnp.sin(psi_sfc) ** 2
    pd2 = (zb - zc) * jnp.sin(psi_sfc) * jnp.cos(psi_sfc)
    pdmod = jnp.sqrt(pd1 ** 2 + pd2 ** 2)

    # ----- Project flow into wave-stress plane (lines 824-837) --------------
    zvt1 = pulow * pum1 + pvlow * pvm1     # (nlev,)
    zvt2 = -pvlow * pum1 + pulow * pvm1
    zvpf = (zvt1 * pd1 + zvt2 * pd2) / (znorm * pdmod)

    # ----- pvph: vertical interpolation onto half levels (lines 844-862) ----
    # For jk = 2..klev (1-based) → h = 1..nlev-1 (0-based):
    #   pvph[h] = ((paphm1[h]-papm1[h-1])*zvpf[h] + (papm1[h]-paphm1[h])*zvpf[h-1])
    #            / (papm1[h]-papm1[h-1])
    #
    # In Fortran: zdp(jk) = papm1(jk) - papm1(jk-1) for full-level index jk.
    # Mapping to 0-based half index h: full-level jk in Fortran 1-based is
    # full-level h in 0-based — the Fortran "jk" of value 2 means "between
    # half-levels 1 and 2" which is half-level 1 in 0-based. zdp(2) =
    # papm1(2) - papm1(1) = zdp_0based[1] = papm1[1] - papm1[0].
    # OK so zdp[h] = papm1[h] - papm1[h-1] for h=1..nlev-1.
    h = jnp.arange(1, nlev)
    pvph_int = (((paphm1[h] - papm1[h - 1]) * zvpf[h]
                 + (papm1[h] - paphm1[h]) * zvpf[h - 1])
                / zdp[h])  # (nlev-1,)
    pvph = jnp.concatenate([jnp.zeros(1), pvph_int, jnp.array([pvph_sfc])])
    # kcrit (line 859): jk where pvph drops below gvsec, but only for jk<klev.
    # Find lowest 0-based half-level h in [1..nlev-1] where pvph[h] < gvsec
    # AND h-as-full-level is < nlev (i.e. h < nlev-1).
    kcrit_mask = (pvph[1:nlev] < _GVSEC) & (jnp.arange(1, nlev) < nlev - 1)
    pvph_int_clamped = jnp.maximum(pvph_int, _GVSEC)
    pvph = jnp.concatenate([jnp.zeros(1), pvph_int_clamped,
                            jnp.array([pvph_sfc])])
    has_kcrit = jnp.any(kcrit_mask)
    # Highest h with the condition (Fortran loops jk=2..klev, last assignment wins).
    kcrit_idx = jnp.where(
        has_kcrit,
        nlev - 1 - jnp.argmax(kcrit_mask[::-1].astype(jnp.int32)),
        jnp.int32(0),  # Fortran initial: kcrit=1 (1-based) = 0 (0-based)
    )

    # ----- Richardson number at half levels (lines 866-879) -----------------
    zdwind = jnp.maximum(jnp.abs(zvpf[h] - zvpf[h - 1]), _GVSEC)
    pri_int = pstab[1:nlev] * (zdp[1:nlev] / (_GRAV * prho[1:nlev] * zdwind)) ** 2
    pri_int = jnp.maximum(pri_int, _GRCRIT)
    pri = jnp.concatenate([jnp.zeros(1), pri_int, jnp.array([9999.0])])

    # ----- kkenvh: top of envelope/blocked layer (lines 886-910) ------------
    # Cumulative integral pnu over jk = 2..klev-1 going UP (Fortran loop is
    # in increasing jk = 2..klev-1 with values appearing only when jk >= kknu).
    # Crossing of pnu=gfrcrit → kkenvh.
    # In 0-based: jk = 1..nlev-2.
    def cum_pnu_step(carry, k):
        pnu_prev, kkenvh = carry
        active = k >= kknu
        zwind_dotted = ((pulow * pum1[k] + pvlow * pvm1[k])
                        / jnp.maximum(jnp.sqrt(pulow ** 2 + pvlow ** 2),
                                      _GVSEC))
        zwind = jnp.maximum(jnp.sqrt(zwind_dotted ** 2), _GVSEC)
        zstabm = jnp.sqrt(jnp.maximum(pstab[k], _GSSEC))
        zstabp = jnp.sqrt(jnp.maximum(pstab[k + 1], _GSSEC))
        zrhom = prho[k]
        zrhop = prho[k + 1]
        increment = pmair[k] * ((zstabp / zrhop + zstabm / zrhom) * 0.5) / zwind
        increment = jnp.where(active, increment, 0.0)
        pnu_new = pnu_prev + increment
        crossed = (pnu_prev <= _GFRCRIT) & (pnu_new > _GFRCRIT) & (kkenvh == nlev - 1)
        kkenvh_new = jnp.where(crossed & active, k, kkenvh).astype(jnp.int32)
        return (pnu_new, kkenvh_new), None

    init_pnu = (jnp.array(0.0), jnp.int32(nlev - 1))
    (pnu_final, kkenvh), _ = lax.scan(
        cum_pnu_step, init_pnu, jnp.arange(1, nlev - 1, dtype=jnp.int32),
    )

    # ----- kkcrith: dynamical-mixing-height cumulative integral (lines 921-941)
    # Same accumulation but loops jk = klev-1 down to 2 (decreasing); crosses pi/2.
    def cum_kkcrith_step(carry, k):
        znup_prev, kkcrith = carry
        zwind_dotted = ((pulow * pum1[k] + pvlow * pvm1[k])
                        / jnp.maximum(jnp.sqrt(pulow ** 2 + pvlow ** 2),
                                      _GVSEC))
        zwind = jnp.maximum(jnp.sqrt(zwind_dotted ** 2), _GVSEC)
        zstabm = jnp.sqrt(jnp.maximum(pstab[k], _GSSEC))
        zstabp = jnp.sqrt(jnp.maximum(pstab[k + 1], _GSSEC))
        zrhom = prho[k]
        zrhop = prho[k + 1]
        increment = pmair[k] * ((zstabp / zrhop + zstabm / zrhom) * 0.5) / zwind
        znup_new = znup_prev + increment
        crossed = (znup_prev <= _PI / 2) & (znup_new > _PI / 2) & (kkcrith == nlev - 1)
        kkcrith_new = jnp.where(crossed, k, kkcrith).astype(jnp.int32)
        return (znup_new, kkcrith_new), None

    init_kkcrith = (jnp.array(0.0), jnp.int32(nlev - 1))
    (_, kkcrith), _ = lax.scan(
        cum_kkcrith_step, init_kkcrith,
        jnp.arange(nlev - 2, 0, -1, dtype=jnp.int32),
    )
    kkcrith = jnp.minimum(kkcrith, kknu)
    # Fortran clamps kkcrith >= 2*ilevh (1-based) → 2*ilevh-1 (0-based).
    kkcrith = jnp.maximum(kkcrith, 2 * ilevh_idx + 1)
    # If kcrit >= kkcrith: reset kcrit to 0 (1-based 1; line 948).
    kcrit_idx = jnp.where(kcrit_idx >= kkcrith, jnp.int32(0), kcrit_idx)

    # ----- ppsi at all full levels (lines 953-971): blocking direction ------
    # zphi_per_level = atan(v/u) per level, with u guarded by gvsec.
    zu_per = jnp.where((pum1 > -_GVSEC) & (pum1 < _GVSEC),
                       pum1 + 2.0 * _GVSEC, pum1)
    zphi_per = jnp.arctan(pvm1 / zu_per)
    ppsi_full = ptheta * _PI / 180.0 - zphi_per   # (nlev,)

    # ----- zzdep (vertical leakiness for blocked drag, lines 975-985) ------
    # Fortran loop range: jk = ilevh..klev (1-based, inclusive) → h in
    # [ilevh-1, nlev-1]. Inside, pzdep is nonzero only when jk >= kkenvh
    # AND kkenvh != klev. Outside the loop range, pzdep stays at the
    # initial 0.
    kkenvh_safe = jnp.maximum(kkenvh, 1)
    denom = phgeo[kkenvh_safe] - phgeo[nlev - 1]
    pzdep_raw = (phgeo[kkenvh_safe - 1] - phgeo) / denom
    levmask_dep = ((jnp.arange(nlev) >= kkenvh)
                   & (kkenvh != nlev - 1)
                   & (jnp.arange(nlev) >= ilevh_idx))
    pzdep = jnp.where(levmask_dep, pzdep_raw, 0.0)

    return dict(
        prho=prho, pstab=pstab, pri=pri, pvph=pvph,
        ppsi_full=ppsi_full, ppsi_sfc=psi_sfc,
        pulow=pulow, pvlow=pvlow,
        pd1=pd1, pd2=pd2, pdmod=pdmod,
        pzdep=pzdep,
        kkenvh=kkenvh, kkcrith=kkcrith, kcrit=kcrit_idx,
        znorm=znorm,
    )


# ---------------------------------------------------------------------------
# gwstress — surface stress
# ---------------------------------------------------------------------------

def _gwstress(pstd, psig, ppic, pval, prho_sfc, pstab_sfc, pvph_sfc,
              pdmod, kkenvh, gkdrag, nlev):
    """Compute the surface gravity-wave stress (lines 1042-1063)."""
    # Effective mountain height above blocked flow.
    zeff_full = ppic - pval
    zeff_blocked = jnp.minimum(
        _GFRCRIT * pvph_sfc / jnp.sqrt(pstab_sfc),
        zeff_full,
    )
    zeff = jnp.where(kkenvh < nlev - 1, zeff_blocked, zeff_full)
    ptau0 = (gkdrag * prho_sfc
             * psig * pdmod / 4.0 / pstd
             * pvph_sfc * jnp.sqrt(pstab_sfc)
             * zeff ** 2)
    return ptau0


# ---------------------------------------------------------------------------
# gwprofil — vertical stress profile + saturation
# ---------------------------------------------------------------------------

def _gwprofil(paphm1, prho, pri, pstab, pvph, pdmod, ptau0, pstd, psig,
              kkcrith, kcrit, ntop, nlev):
    """Compute the vertical profile of gravity-wave stress (lines 1131-1291).

    Returns ``ptau`` of length ``nlev+1``.
    """
    # zoro per column (line 1141)
    zoro = psig * pdmod / 4.0 / pstd

    # Initial ztau (line 1142-1143): ztau[nlev] = ptau0; ztau[kkcrith] = grahilo*ptau0.
    # Then loop 430 fills ptau for jk = klev+1..2 (1-based, descending) =
    # h = nlev..1 (0-based). Index h=0 is NEVER touched by the init loop and
    # stays at the initial 0 value (line 1131: ptau(:,:) = 0). For h > kkcrith
    # the formula is a linear-in-pressure interpolation between ztau[nlev]
    # (= ptau0) and ztau[kkcrith] (= grahilo*ptau0); for h <= kkcrith,
    # ptau = ztau[kkcrith] = grahilo*ptau0.
    paphm1_sfc = paphm1[nlev]
    paphm1_kc = paphm1[kkcrith]
    zdelpt = paphm1_kc - paphm1_sfc
    h = jnp.arange(nlev + 1)
    interp = (ptau0
              + (paphm1 - paphm1_sfc) / zdelpt * (_GRAHILO * ptau0 - ptau0))
    ptau_init = jnp.where(
        h > kkcrith, interp,
        jnp.where(h > 0, _GRAHILO * ptau0, 0.0),
    )

    # ----- Saturation sweep (lines 1191-1231) -------------------------------
    # Process h = nlev down to 1 (Fortran: jk = klev down to 2). At each
    # half-level h, compute znorm, zdz2 from current ptau, then update ptau
    # if h < kkcrith via the saturation criterion. ptau[h] depends on
    # ptau[h+1] (one above going UP, but loop is going DOWN) — so this is
    # sequential and needs lax.scan.
    #
    # NOTE: Fortran loops jk = klev, klev-1, ..., 2 (DOWN in 1-based, which
    # is UP in 0-based). At each iteration it reads ptau[jk+1] (already
    # processed in previous iter; for jk=klev that's the surface ptau[klev+1]
    # which equals ptau0). So state is ptau[h+1] from previous iteration.

    def sat_step(carry, h):
        ptau = carry
        # h goes nlev, nlev-1, ..., 1
        znorm = prho[h] * jnp.sqrt(pstab[h]) * pvph[h]
        zdz2 = ptau[h] / jnp.maximum(znorm, _GSSEC) / zoro

        # Saturation only for h < kkcrith
        active = h < kkcrith
        # Two break conditions:
        zero_branch = (ptau[h + 1] < _GTSEC) | (h <= kcrit)
        # Else: compute zriw and possibly cap ptau.
        zsqr = jnp.sqrt(jnp.maximum(pri[h], 1e-30))
        zalfa = jnp.sqrt(jnp.maximum(pstab[h] * zdz2, 0.0)) / pvph[h]
        zriw = pri[h] * (1.0 - zalfa) / (1.0 + zalfa * zsqr) ** 2
        zdel = 4.0 / zsqr / _GRCRIT + 1.0 / _GRCRIT ** 2 + 4.0 / _GRCRIT
        zb = 1.0 / _GRCRIT + 2.0 / zsqr
        zalpha = 0.5 * (-zb + jnp.sqrt(jnp.maximum(zdel, 0.0)))
        zdz2n = (pvph[h] * zalpha) ** 2 / jnp.maximum(pstab[h], _GSSEC)
        ptau_capped = znorm * zdz2n * zoro
        # If zriw < grcrit, replace with capped value; otherwise keep current.
        ptau_new_uncapped = jnp.where(zriw < _GRCRIT, ptau_capped, ptau[h])
        # Limit: ptau[h] = min(ptau[h], ptau[h+1])
        ptau_after_min = jnp.minimum(ptau_new_uncapped, ptau[h + 1])
        # Apply zero_branch
        ptau_h_new = jnp.where(zero_branch, 0.0, ptau_after_min)
        # Only modify if active (h < kkcrith)
        ptau_h_final = jnp.where(active, ptau_h_new, ptau[h])
        return ptau.at[h].set(ptau_h_final), None

    indices_down = jnp.arange(nlev, 0, -1)
    ptau_after_sat, _ = lax.scan(sat_step, ptau_init, indices_down)

    # ----- Reorganization at low level + model top (lines 1242-1291) -------
    # In Fortran 1-based: loop jk=1..klev. If jk > kkcrith: linear interp.
    # If jk < ntop: ptau = ztau[ntop]. ptau[klev+1] is left unchanged.
    # 0-based: loop h=0..nlev-1. Convert ntop→ntop-1 for the comparison
    # since "jk < ntop" (1-based) ≡ "h < ntop-1" (0-based).
    ztau_kc = ptau_after_sat[kkcrith]
    ztau_top = ptau_after_sat[ntop - 1]   # ntop-1 = 0-based equivalent
    ztau_sfc = ptau_after_sat[nlev]
    interp2 = (ztau_sfc
               + (paphm1 - paphm1[nlev]) / (paphm1[kkcrith] - paphm1[nlev])
               * (ztau_kc - ztau_sfc))
    full_idx = jnp.arange(nlev + 1)
    ptau_final = jnp.where(
        (full_idx > kkcrith) & (full_idx < nlev),
        interp2, ptau_after_sat,
    )
    ptau_final = jnp.where(full_idx < ntop - 1, ztau_top, ptau_final)
    return ptau_final


# ---------------------------------------------------------------------------
# orodrag — combine wave-drag tendency + blocked-flow wake drag
# ---------------------------------------------------------------------------

def _orodrag(paphm1, papm1, pmair, pum1, pvm1, ptm1, phgeo,
             pmea, pstd, psig, pgam, pthe, ppic, pval,
             pdtime, gkdrag, gkwake, nktopg, ntop):
    """Top-level wave + blocked-flow drag for one column."""
    nlev = pum1.shape[0]
    setup = _orosetup(paphm1, papm1, pmair, pum1, pvm1, ptm1, phgeo,
                      pmea, ppic, pval, pthe, pgam, nktopg)
    pgam_safe = jnp.maximum(pgam, _GTSEC)
    prho = setup["prho"]
    pstab = setup["pstab"]
    pri = setup["pri"]
    pvph = setup["pvph"]
    ppsi_full = setup["ppsi_full"]
    pulow, pvlow = setup["pulow"], setup["pvlow"]
    pd1, pd2, pdmod = setup["pd1"], setup["pd2"], setup["pdmod"]
    pzdep = setup["pzdep"]
    kkenvh, kkcrith, kcrit = setup["kkenvh"], setup["kkcrith"], setup["kcrit"]
    pvph_sfc = pvph[nlev]
    prho_sfc = prho[nlev]
    pstab_sfc = pstab[nlev]

    # Surface stress
    ptau0 = _gwstress(pstd, psig, ppic, pval, prho_sfc, pstab_sfc, pvph_sfc,
                      pdmod, kkenvh, gkdrag, nlev)
    # Vertical stress profile
    ptau = _gwprofil(paphm1, prho, pri, pstab, pvph, pdmod, ptau0,
                     pstd, psig, kkcrith, kcrit, ntop, nlev)

    # Wave-drag tendencies (lines 401-436)
    # ztemp[jk] = -(ptau[jk+1]-ptau[jk]) / (pvph_sfc * pmair[jk]) per FULL level jk
    # In 0-based full-level indexing: jk=0..nlev-1 reads ptau[jk] and ptau[jk+1]
    ztemp = -(ptau[1:] - ptau[:-1]) / (pvph_sfc * pmair)
    zdudt_wave = (pulow * pd1 - pvlow * pd2) * ztemp / pdmod
    zdvdt_wave = (pvlow * pd1 + pulow * pd2) * ztemp / pdmod
    # Overshoot guard (line 423-429)
    zforc = jnp.sqrt(zdudt_wave ** 2 + zdvdt_wave ** 2)
    ztend = jnp.sqrt(pum1 ** 2 + pvm1 ** 2) / pdtime
    rover = 0.25
    factor = jnp.where(zforc >= rover * ztend,
                       rover * ztend / jnp.maximum(zforc, 1e-30),
                       1.0)
    zdudt_wave = zdudt_wave * factor
    zdvdt_wave = zdvdt_wave * factor

    # Disable wave drag if gkdrag == 0
    use_wave = gkdrag != 0.0
    zdudt_wave = jnp.where(use_wave, zdudt_wave, 0.0)
    zdvdt_wave = jnp.where(use_wave, zdvdt_wave, 0.0)

    # Blocked-flow drag (lines 442-477) — replaces zdudt/zdvdt where active
    zb = 1.0 - 0.18 * pgam_safe - 0.04 * pgam_safe ** 2
    zc = 0.48 * pgam_safe + 0.30 * pgam_safe ** 2
    zconb = 2.0 * pdtime * gkwake * psig / (4.0 * pstd)
    zabsv = jnp.sqrt(pum1 ** 2 + pvm1 ** 2) / 2.0
    cos_psi = jnp.cos(ppsi_full)
    sin_psi = jnp.sin(ppsi_full)
    zzd1 = zb * cos_psi ** 2 + zc * sin_psi ** 2
    ratio = ((cos_psi ** 2 + pgam_safe * sin_psi ** 2)
             / (pgam_safe * cos_psi ** 2 + sin_psi ** 2))
    zbet = (jnp.maximum(0.0, 2.0 - 1.0 / ratio)
            * zconb * pzdep * zzd1 * zabsv)
    block_du = -pum1 / pdtime * zbet / (1.0 + zbet)
    block_dv = -pvm1 / pdtime * zbet / (1.0 + zbet)
    use_block = (gkwake != 0.0) & (jnp.arange(nlev) > kkenvh)
    zdudt = jnp.where(use_block, block_du, zdudt_wave)
    zdvdt = jnp.where(use_block, block_dv, zdvdt_wave)

    # Energy dissipation (lines 481-494)
    zust = pum1 + pdtime * zdudt
    zvst = pvm1 + pdtime * zdvdt
    zdis_pre = 0.5 * (pum1 ** 2 + pvm1 ** 2 - zust ** 2 - zvst ** 2)
    # If zdis < 0: rescale tendencies so KE conserved.
    safe_denom = jnp.maximum(zust ** 2 + zvst ** 2, 1e-30)
    zred = jnp.sqrt((pum1 ** 2 + pvm1 ** 2) / safe_denom)
    zust_corr = zust * zred
    zvst_corr = zvst * zred
    new_du = (zust_corr - pum1) / pdtime
    new_dv = (zvst_corr - pvm1) / pdtime
    zdudt = jnp.where(zdis_pre < 0.0, new_du, zdudt)
    zdvdt = jnp.where(zdis_pre < 0.0, new_dv, zdvdt)
    zust_final = pum1 + pdtime * zdudt
    zvst_final = pvm1 + pdtime * zdvdt
    zdis = 0.5 * (pum1 ** 2 + pvm1 ** 2 - zust_final ** 2 - zvst_final ** 2)
    pdis = zdis / pdtime
    return zdudt, zdvdt, pdis


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def sso_drag(
    pdtime: jnp.ndarray,
    pcoriol: jnp.ndarray,
    pzf: jnp.ndarray,
    pzs: jnp.ndarray,
    paphm1: jnp.ndarray,
    papm1: jnp.ndarray,
    pmair: jnp.ndarray,
    ptm1: jnp.ndarray,
    pum1: jnp.ndarray,
    pvm1: jnp.ndarray,
    pmea: jnp.ndarray,
    pstd: jnp.ndarray,
    psig: jnp.ndarray,
    pgam: jnp.ndarray,
    pthe: jnp.ndarray,
    ppic: jnp.ndarray,
    pval: jnp.ndarray,
    psftlf: jnp.ndarray,
    config: SSOParameters,
) -> Tuple[SSOTendencies, SSOState]:
    """Compute Lott-Miller SSO drag for a single column.

    Mirrors ``ssodrag`` in ``mo_ssodrag.f90`` line 26. ``pcoriol`` is read
    by the (unported) mountain-lift branch only. The seven SSO descriptors
    (``pmea``..``pval``) are scalars per column from boundary data.
    """
    nlev = pum1.shape[0]
    nktopg = int(config.nktopg)
    ntop = int(config.ntop)

    # Activation criterion (lines 173-180): scheme is active only if
    # (ppic - pmea) > gpicmea AND pstd > gstd.
    active = (ppic - pmea > config.gpicmea) & (pstd > config.gstd)

    # Above-surface height (line 158).
    phgeo = pzf - pzs

    zdudt, zdvdt, zdis = _orodrag(
        paphm1, papm1, pmair, pum1, pvm1, ptm1, phgeo,
        pmea, pstd, psig, pgam, pthe, ppic, pval,
        pdtime, config.gkdrag, config.gkwake, nktopg, ntop,
    )

    # Mask by activation; scale by land fraction (lines 226-244).
    zero = jnp.zeros_like(zdudt)
    zdudt = jnp.where(active, zdudt, zero) * psftlf
    zdvdt = jnp.where(active, zdvdt, zero) * psftlf
    zdis = jnp.where(active, zdis, zero) * psftlf

    # Column-integrated stresses and dissipation (lines 223-231).
    u_stress = jnp.sum(zdudt * pmair)
    v_stress = jnp.sum(zdvdt * pmair)
    dissip_total = jnp.sum(zdis * pmair)

    return (
        SSOTendencies(dudt=zdudt, dvdt=zdvdt, dissip=zdis),
        SSOState(u_stress=u_stress, v_stress=v_stress,
                 dissip_total=dissip_total),
    )
