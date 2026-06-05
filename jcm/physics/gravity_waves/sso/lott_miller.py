"""Lott & Miller (1997) sub-grid orographic gravity-wave drag.

Algorithm (column-mode, executed at each grid column):

1. **Activation gate.** A column is processed only if its sub-grid
   orography is large enough — both the peak-minus-mean elevation
   and the orographic standard deviation must exceed the user-supplied
   thresholds (``min_peak_minus_mean_elevation`` and ``min_orog_std``).
   Inactive columns return zero tendencies.

2. **Mean low-level flow & geometry** (the ``_orosetup`` helper).
   Build mass-weighted mean wind and Brunt-Vaisala stratification over
   a layer running from the surface up to the level that "sees" the
   mountain peaks. Project the flow onto the orography's principal
   axis using its anisotropy and orientation angle to obtain a single
   ``pvph`` magnitude per column. Locate three vertically-stacked
   levels:

   - ``kkenvh`` — top of the blocked-flow layer, where the cumulative
     non-dim mountain height (Froude integral) crosses ``gfrcrit``
     going up. Below this level the flow goes around (not over) the
     mountain.
   - ``kkcrith`` — top of the low-level wave-trapping layer, where
     the cumulative wave-action integral crosses pi/2.
   - ``kcrit`` — critical level above which the projected wind drops
     below a small floor.

3. **Surface gravity-wave stress** (the ``_gwstress`` helper).
   Stress at the surface follows Phillips (1979) for flow over an
   anisotropic ellipse, scaled by ``wave_drag_coeff`` and reduced by
   the effective mountain height (``ppic - pval`` capped at
   ``gfrcrit · pvph / N``).

4. **Vertical stress profile + wave breaking** (the ``_gwprofil``
   helper). Below ``kkcrith`` stress is constant (linear-in-pressure
   between ``ptau0`` at the surface and ``grahilo·ptau0`` at
   ``kkcrith``). Above ``kkcrith`` a saturation criterion based on the
   wave Richardson number caps stress to keep the wave amplitude
   below the breaking threshold; once stress drops below the noise
   floor or hits the critical level it is zeroed all the way up.

5. **Tendencies** (the top-level ``_orodrag``). Wave-drag tendency is
   the negative vertical divergence of stress in the principal-axis
   plane, projected back to (u, v). Below ``kkenvh`` the wave drag is
   replaced by **blocked-flow drag**: form drag from each layer of
   blocked air being deflected around the obstacle, scaled by
   ``blocked_flow_drag_coeff``. An overshoot guard caps any
   tendency at 25% of the local KE per timestep, and a final
   KE-conservation correction rescales tendencies if the resulting
   flow gains energy (numerical artefact).

**Implementation choices and gaps:**

- The mountain-lift branch (``orolift`` in the ECHAM source, controlled
  by ``mountain_lift_coeff``) is **not** implemented. It is disabled by
  default in production ECHAM-A configurations (``gklift = 0``) and is
  the only piece of the original scheme intentionally omitted here.
  Setting ``mountain_lift_coeff`` to non-zero is currently a no-op; if
  a non-default lift is needed in the future, port the ``orolift``
  routine alongside the existing helpers.

- The vertical-grid convention follows the convention used elsewhere
  in JCM: half levels are length ``nlev+1`` indexed top-to-bottom
  (``[0..nlev]``); full levels are length ``nlev`` (``[0..nlev-1]``).
  Index 0 = top of the model, index ``nlev`` = surface half level.

- A few internal level-index quantities (``nktopg`` for the top-most
  level orography may "see"; ``ntop`` for the stress-profile model
  top) are passed as Python integer kwargs to :func:`sso_drag` rather
  than living on the parameters tree, because they index into level
  arrays and must be static at JIT trace time.

- The seven sub-grid orography descriptors required by the scheme
  (mean elevation, std-dev, slope, anisotropy, orientation, peak,
  valley) are normally produced by an offline preprocessor from a
  high-resolution DEM. JCM derives them on the fly from mean
  orography via :func:`jcm.terrain.derive_sso_descriptors` (Baines-
  Palmer statistics) or :func:`jcm.terrain.get_simplified_sso_descriptors`
  (rough heuristic from mean orography only) when real preprocessed
  data is not available.
"""
from typing import NamedTuple, Tuple

import jax.numpy as jnp
from jax import lax
import tree_math

import jcm.constants as c

# Tunable thresholds taken straight from the ECHAM ``mo_echam_sso_config``
# PARAMETER block. They protect the algorithm from degenerate inputs
# (zero stratification, near-singular anisotropy, etc.) and set the
# critical Froude/Richardson numbers used by the saturation criterion.
_CRITICAL_FROUDE = 0.5      # critical non-dim mountain height (gfrcrit)
_CRITICAL_RICHARDSON = 0.25  # critical Richardson number (grcrit)
_TRAPPED_WAVE_FRAC = 1.0    # fraction of stress trapped at low levels (grahilo)
_MIN_BLOCKED_DEPTH = 0.80   # security floor on blocked-flow depth (gsigcr)
_MIN_BV_FREQ = 1.0e-4       # security floor on B-V frequency (gssec)
_MIN_ANISOTROPY = 1.0e-5    # security floor on anisotropy / stress (gtsec)
_MIN_LOW_LEVEL_WIND = 0.10  # security floor on low-level wind (gvsec)


@tree_math.struct
class SSOParameters:
    """Tunable parameters for the Lott & Miller (1997) SSO drag scheme.

    Static loop / level-index knobs (``nktopg`` for the top model level
    that orography may "see"; ``ntop`` for the stress-profile model
    top) are passed as Python kwargs to :func:`sso_drag` because they
    index into level arrays at JIT trace time.

    Attributes:
        min_peak_minus_mean_elevation: Activation threshold (m). The
            scheme is inactive in any column where ``peak_elevation -
            mean_orography`` does not exceed this value. Default 1.0.
        min_orog_std: Activation threshold (m) on the sub-grid
            orography standard deviation. The scheme is inactive in
            any column where ``orography_std`` does not exceed this
            value. Default 1.0.
        wave_drag_coeff: Coefficient on the wave-drag branch
            (``gkdrag`` in the ECHAM source). Default 0.2.
        blocked_flow_drag_coeff: Coefficient on the low-level
            blocked-flow form-drag branch (``gkwake``). Default 1.0.
        mountain_lift_coeff: Coefficient on the mountain-lift branch
            (``gklift``). The lift branch is **not** ported in this
            scheme; setting this to non-zero is currently a no-op.
            Default 0.0.

    """

    min_peak_minus_mean_elevation: jnp.ndarray
    min_orog_std: jnp.ndarray
    wave_drag_coeff: jnp.ndarray
    blocked_flow_drag_coeff: jnp.ndarray
    mountain_lift_coeff: jnp.ndarray

    @classmethod
    def default(
        cls,
        min_peak_minus_mean_elevation: float = 1.0,
        min_orog_std: float = 1.0,
        wave_drag_coeff: float = 0.2,
        blocked_flow_drag_coeff: float = 1.0,
        mountain_lift_coeff: float = 0.0,
    ) -> "SSOParameters":
        return cls(
            min_peak_minus_mean_elevation=jnp.asarray(min_peak_minus_mean_elevation),
            min_orog_std=jnp.asarray(min_orog_std),
            wave_drag_coeff=jnp.asarray(wave_drag_coeff),
            blocked_flow_drag_coeff=jnp.asarray(blocked_flow_drag_coeff),
            mountain_lift_coeff=jnp.asarray(mountain_lift_coeff),
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
    pressure_half: jnp.ndarray,           # (nlev+1,)
    pressure_full: jnp.ndarray,           # (nlev,)
    layer_mass: jnp.ndarray,              # (nlev,)
    u_wind: jnp.ndarray,                  # (nlev,)
    v_wind: jnp.ndarray,                  # (nlev,)
    temperature: jnp.ndarray,             # (nlev,)
    height_above_ground: jnp.ndarray,     # (nlev,)
    mean_orography: jnp.ndarray,          # scalar
    peak_elevation: jnp.ndarray,          # scalar
    valley_elevation: jnp.ndarray,        # scalar
    orography_orientation: jnp.ndarray,   # scalar (degrees)
    orography_anisotropy: jnp.ndarray,    # scalar
    nktopg: int,
):
    """Build the geometric / mean-flow inputs needed for the SSO scheme.

    For one column this routine:
    1. Identifies the three special vertical levels:
       - ``layer_seeing_peaks`` — first layer above ground whose
         geopotential height exceeds the peak-to-valley range.
       - ``layer_above_mean`` — first layer above the mean orography.
       - ``layer_above_mean_above_valleys`` — first layer above
         ``min(peak−mean, mean−valley)``.
    2. Computes the half-level density and Brunt-Vaisala frequency
       between adjacent full levels.
    3. Forms a mass-weighted mean of u, v, density and stratification
       over the layer running from the surface up to
       ``layer_seeing_peaks_above_mean``.
    4. Builds the wave-stress orientation factors (``pd1``, ``pd2``,
       ``pdmod``) from the orography anisotropy and the angle between
       the low-level mean wind and the orographic principal axis.
    5. Projects the full-column wind onto the wave-stress plane,
       interpolates onto half levels, and computes the per-half-level
       Richardson number.
    6. Locates ``kkenvh`` (top of the blocked-flow layer) by integrating
       N/U upward and finding where it first exceeds the critical
       Froude number.
    7. Locates ``kkcrith`` (top of the wave-trapping low-level layer)
       similarly using a pi/2 threshold.
    8. Computes ``pzdep`` (the per-layer "leakiness" factor used by the
       blocked-flow drag).

    Returns a dict packing all of the above for later use by
    ``_gwstress``, ``_gwprofil`` and ``_orodrag``.

    All half-level arrays have length nlev+1 indexed 0=top, nlev=surface;
    full-level arrays have length nlev.
    """
    # Local aliases keep the literal port readable below — the original
    # 8-character Fortran names are still used internally for parts of
    # the algorithm that mirror the source line-for-line.
    paphm1 = pressure_half
    papm1 = pressure_full
    pmair = layer_mass
    pum1 = u_wind
    pvm1 = v_wind
    ptm1 = temperature
    phgeo = height_above_ground
    pmea = mean_orography
    ppic = peak_elevation
    pval = valley_elevation
    ptheta = orography_orientation
    pgam = orography_anisotropy
    nlev = pum1.shape[0]
    ilevh = nlev // 3   # Fortran's ilevh = klev/3 (integer divide)

    # Anisotropy guard (line 647)
    pgam_safe = jnp.maximum(pgam, _MIN_ANISOTROPY)

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
    rho_half = (2.0 * paphm1[h_idx] / c.rd
                / (ptm1[h_idx] + ptm1[h_idx - 1]))
    stab_half = (2.0 * c.grav ** 2 / c.cpd / (ptm1[h_idx] + ptm1[h_idx - 1])
                 * (1.0 - c.cpd * rho_half * (ptm1[h_idx] - ptm1[h_idx - 1])
                    / zdp_half))
    stab_half = jnp.maximum(stab_half, _MIN_BV_FREQ)
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

    znorm = jnp.maximum(jnp.sqrt(pulow ** 2 + pvlow ** 2), _MIN_LOW_LEVEL_WIND)
    pvph_sfc = znorm

    # ----- Anisotropy & wave-stress orientation (lines 802-819) -------------
    zu = jnp.where((pulow > -_MIN_LOW_LEVEL_WIND) & (pulow < _MIN_LOW_LEVEL_WIND),
                   pulow + 2.0 * _MIN_LOW_LEVEL_WIND, pulow)
    zphi = jnp.arctan(pvlow / zu)
    psi_sfc = ptheta * jnp.pi / 180.0 - zphi
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
    kcrit_mask = (pvph[1:nlev] < _MIN_LOW_LEVEL_WIND) & (jnp.arange(1, nlev) < nlev - 1)
    pvph_int_clamped = jnp.maximum(pvph_int, _MIN_LOW_LEVEL_WIND)
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
    zdwind = jnp.maximum(jnp.abs(zvpf[h] - zvpf[h - 1]), _MIN_LOW_LEVEL_WIND)
    pri_int = pstab[1:nlev] * (zdp[1:nlev] / (c.grav * prho[1:nlev] * zdwind)) ** 2
    pri_int = jnp.maximum(pri_int, _CRITICAL_RICHARDSON)
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
                                      _MIN_LOW_LEVEL_WIND))
        zwind = jnp.maximum(jnp.sqrt(zwind_dotted ** 2), _MIN_LOW_LEVEL_WIND)
        zstabm = jnp.sqrt(jnp.maximum(pstab[k], _MIN_BV_FREQ))
        zstabp = jnp.sqrt(jnp.maximum(pstab[k + 1], _MIN_BV_FREQ))
        zrhom = prho[k]
        zrhop = prho[k + 1]
        increment = pmair[k] * ((zstabp / zrhop + zstabm / zrhom) * 0.5) / zwind
        increment = jnp.where(active, increment, 0.0)
        pnu_new = pnu_prev + increment
        crossed = (pnu_prev <= _CRITICAL_FROUDE) & (pnu_new > _CRITICAL_FROUDE) & (kkenvh == nlev - 1)
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
                                      _MIN_LOW_LEVEL_WIND))
        zwind = jnp.maximum(jnp.sqrt(zwind_dotted ** 2), _MIN_LOW_LEVEL_WIND)
        zstabm = jnp.sqrt(jnp.maximum(pstab[k], _MIN_BV_FREQ))
        zstabp = jnp.sqrt(jnp.maximum(pstab[k + 1], _MIN_BV_FREQ))
        zrhom = prho[k]
        zrhop = prho[k + 1]
        increment = pmair[k] * ((zstabp / zrhop + zstabm / zrhom) * 0.5) / zwind
        znup_new = znup_prev + increment
        crossed = (znup_prev <= jnp.pi / 2) & (znup_new > jnp.pi / 2) & (kkcrith == nlev - 1)
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
    zu_per = jnp.where((pum1 > -_MIN_LOW_LEVEL_WIND) & (pum1 < _MIN_LOW_LEVEL_WIND),
                       pum1 + 2.0 * _MIN_LOW_LEVEL_WIND, pum1)
    zphi_per = jnp.arctan(pvm1 / zu_per)
    ppsi_full = ptheta * jnp.pi / 180.0 - zphi_per   # (nlev,)

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
        _CRITICAL_FROUDE * pvph_sfc / jnp.sqrt(pstab_sfc),
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
              + (paphm1 - paphm1_sfc) / zdelpt * (_TRAPPED_WAVE_FRAC * ptau0 - ptau0))
    ptau_init = jnp.where(
        h > kkcrith, interp,
        jnp.where(h > 0, _TRAPPED_WAVE_FRAC * ptau0, 0.0),
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
        zdz2 = ptau[h] / jnp.maximum(znorm, _MIN_BV_FREQ) / zoro

        # Saturation only for h < kkcrith
        active = h < kkcrith
        # Two break conditions:
        zero_branch = (ptau[h + 1] < _MIN_ANISOTROPY) | (h <= kcrit)
        # Else: compute zriw and possibly cap ptau.
        zsqr = jnp.sqrt(jnp.maximum(pri[h], 1e-30))
        zalfa = jnp.sqrt(jnp.maximum(pstab[h] * zdz2, 0.0)) / pvph[h]
        zriw = pri[h] * (1.0 - zalfa) / (1.0 + zalfa * zsqr) ** 2
        zdel = 4.0 / zsqr / _CRITICAL_RICHARDSON + 1.0 / _CRITICAL_RICHARDSON ** 2 + 4.0 / _CRITICAL_RICHARDSON
        zb = 1.0 / _CRITICAL_RICHARDSON + 2.0 / zsqr
        zalpha = 0.5 * (-zb + jnp.sqrt(jnp.maximum(zdel, 0.0)))
        zdz2n = (pvph[h] * zalpha) ** 2 / jnp.maximum(pstab[h], _MIN_BV_FREQ)
        ptau_capped = znorm * zdz2n * zoro
        # If zriw < grcrit, replace with capped value; otherwise keep current.
        ptau_new_uncapped = jnp.where(zriw < _CRITICAL_RICHARDSON, ptau_capped, ptau[h])
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
    pgam_safe = jnp.maximum(pgam, _MIN_ANISOTROPY)
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
    dt: jnp.ndarray,
    coriolis: jnp.ndarray,
    height_full: jnp.ndarray,
    surface_height: jnp.ndarray,
    pressure_half: jnp.ndarray,
    pressure_full: jnp.ndarray,
    layer_mass: jnp.ndarray,
    temperature: jnp.ndarray,
    u_wind: jnp.ndarray,
    v_wind: jnp.ndarray,
    mean_orography: jnp.ndarray,
    orography_std: jnp.ndarray,
    orography_slope: jnp.ndarray,
    orography_anisotropy: jnp.ndarray,
    orography_orientation: jnp.ndarray,
    peak_elevation: jnp.ndarray,
    valley_elevation: jnp.ndarray,
    land_fraction: jnp.ndarray,
    config: SSOParameters,
    *,
    nktopg: int = 1,
    ntop: int = 1,
) -> Tuple[SSOTendencies, SSOState]:
    """Compute Lott-Miller SSO drag tendencies for a single column.

    Args:
        dt: physics timestep (s).
        coriolis: Coriolis parameter at this column (1/s). Read only by
            the (unported) mountain-lift branch — currently a no-op.
        height_full: full-level geopotential height above sea level
            (m), shape (nlev,).
        surface_height: surface elevation (m), scalar.
        pressure_half: pressure on half levels (Pa), shape (nlev+1,).
            Index 0 = top, index nlev = surface.
        pressure_full: pressure on full levels (Pa), shape (nlev,).
        layer_mass: full-level air mass per unit area (kg/m^2), shape
            (nlev,) — i.e. ``Δp / g``.
        temperature: full-level temperature (K), shape (nlev,).
        u_wind, v_wind: full-level zonal/meridional wind (m/s), shape
            (nlev,).
        mean_orography: mean elevation of the column's sub-grid
            orography (m), scalar.
        orography_std: standard deviation of the sub-grid orography
            (m), scalar. Drives both the activation gate and the
            surface-stress amplitude.
        orography_slope: mean slope of the sub-grid orography
            (dimensionless), scalar.
        orography_anisotropy: anisotropy factor of the sub-grid
            orographic stress ellipse (0=pure ridge, 1=isotropic),
            scalar.
        orography_orientation: orientation angle (deg) of the principal
            axis of the orographic stress ellipse, measured from east.
            Scalar.
        peak_elevation: characteristic peak elevation in the column
            (m, above sea level), scalar.
        valley_elevation: characteristic valley elevation in the column
            (m, above sea level), scalar.
        land_fraction: fraction of the column over land+lakes (0-1),
            scalar. Tendencies are scaled by this since SSO descriptors
            are valid only over land.
        config: tunable :class:`SSOParameters`.
        nktopg: top-most 1-based level index that orography is allowed
            to "see" (Python int, static at trace time). Default 1
            (orography may extend up to the model top).
        ntop: 1-based level index above which the wave-stress profile
            is held constant (Python int, static at trace time).
            Default 1.

    Returns:
        ``(tendencies, state)`` — see :class:`SSOTendencies` and
        :class:`SSOState` for field documentation.

    """
    # Activation gate: skip columns whose sub-grid orography is too small
    # to support meaningful drag.
    active = ((peak_elevation - mean_orography
               > config.min_peak_minus_mean_elevation)
              & (orography_std > config.min_orog_std))

    # Layer height above ground (full levels).
    height_above_ground = height_full - surface_height

    drag_u, drag_v, dissipation = _orodrag(
        pressure_half, pressure_full, layer_mass,
        u_wind, v_wind, temperature, height_above_ground,
        mean_orography, orography_std, orography_slope,
        orography_anisotropy, orography_orientation,
        peak_elevation, valley_elevation,
        dt, config.wave_drag_coeff, config.blocked_flow_drag_coeff,
        nktopg, ntop,
    )

    # Apply activation mask and scale by land fraction (the descriptors
    # are valid only over the land portion of the cell).
    zero = jnp.zeros_like(drag_u)
    drag_u = jnp.where(active, drag_u, zero) * land_fraction
    drag_v = jnp.where(active, drag_v, zero) * land_fraction
    dissipation = jnp.where(active, dissipation, zero) * land_fraction

    u_stress = jnp.sum(drag_u * layer_mass)
    v_stress = jnp.sum(drag_v * layer_mass)
    dissip_total = jnp.sum(dissipation * layer_mass)

    return (
        SSOTendencies(dudt=drag_u, dvdt=drag_v, dissip=dissipation),
        SSOState(u_stress=u_stress, v_stress=v_stress,
                 dissip_total=dissip_total),
    )


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

import jax  # noqa: E402
from flax import nnx  # noqa: E402

from jcm import constants as _physical_constants  # noqa: E402
from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics.physics_term import PhysicsTerm  # noqa: E402
from jcm.physics_interface import PhysicsState, PhysicsTendency  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402


class LottMillerSso(PhysicsTerm):
    """Lott-Miller (1997) sub-grid orographic GWD as a composable term.

    Wraps :func:`sso_drag` over columns. Reads ``pressure_full``,
    ``pressure_half``, ``height_full`` from the moist-air diagnostics
    dict; reads orography descriptors (``orog``, ``orostd``, ``orosig``,
    ``orogam``, ``orothe``, ``oropic``, ``oroval``, ``fmask``) from
    :class:`TerrainData`. Writes only u/v/T tendencies — no Data
    sub-struct.

    The unported mountain-lift branch (``orolift``, ``gklift=0`` in the
    production namelist) is not exercised; coriolis is therefore set to
    zero.
    """

    name: ClassVar[str] = "lott_miller_sso"
    category: ClassVar[str] = "sso"
    requires: ClassVar[tuple[str, ...]] = (
        "pressure_full", "pressure_half", "height_full",
    )
    provides: ClassVar[tuple[str, ...]] = ()

    def __init__(self, params: SSOParameters | None = None):
        """Hold the scheme-native :class:`SSOParameters`."""
        self.params = nnx.Param(params or SSOParameters.default())

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute u/v/T tendencies from Lott-Miller SSO."""
        nlev, ncols = state.temperature.shape
        dt = diagnostics["_dt_seconds"]
        params = self.params.get_value()

        pressure_full = diagnostics["pressure_full"]
        pressure_half = diagnostics["pressure_half"]
        height_full = diagnostics["height_full"]

        layer_mass = (
            (pressure_half[1:, :] - pressure_half[:-1, :])
            / _physical_constants.grav
        )
        # Coriolis is consumed only by the unported mountain-lift branch.
        coriolis = jnp.zeros((ncols,))

        def _sso_one_col(
            pressure_full_c, pressure_half_c, layer_mass_c,
            temperature_c, u_wind_c, v_wind_c, height_full_c,
            surface_height_c, mean_orography_c, orography_std_c,
            orography_slope_c, orography_anisotropy_c,
            orography_orientation_c, peak_elevation_c,
            valley_elevation_c, coriolis_c, land_fraction_c,
        ):
            return sso_drag(
                jnp.asarray(dt), coriolis_c, height_full_c,
                surface_height_c,
                pressure_half_c, pressure_full_c, layer_mass_c,
                temperature_c, u_wind_c, v_wind_c,
                mean_orography_c, orography_std_c, orography_slope_c,
                orography_anisotropy_c, orography_orientation_c,
                peak_elevation_c, valley_elevation_c,
                land_fraction_c, params,
                nktopg=1, ntop=1,
            )

        tend, _state = jax.vmap(
            _sso_one_col,
            in_axes=(1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            out_axes=(0, 0),
        )(
            pressure_full, pressure_half, layer_mass,
            state.temperature, state.u_wind, state.v_wind, height_full,
            terrain.orog.reshape(-1), terrain.orog.reshape(-1),
            terrain.orostd.reshape(-1), terrain.orosig.reshape(-1),
            terrain.orogam.reshape(-1), terrain.orothe.reshape(-1),
            terrain.oropic.reshape(-1), terrain.oroval.reshape(-1),
            coriolis, terrain.fmask.reshape(-1),
        )

        dt_temperature = tend.dissip / _physical_constants.cpd

        return PhysicsTendency(
            u_wind=tend.dudt.T,
            v_wind=tend.dvdt.T,
            temperature=dt_temperature.T,
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers={},
        ), diagnostics
