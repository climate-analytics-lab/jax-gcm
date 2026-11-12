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

- A few internal level-index quantities (``nktopg``, the security level
  that bounds the blocked-flow layer from *below*; ``ntop`` for the
  stress-profile model top) are passed as Python integer kwargs to
  :func:`sso_drag` rather than living on the parameters tree, because
  they index into level arrays and must be static at JIT trace time.
  ``nktopg`` is a property of the vertical grid, computed exactly as
  ECHAM's ``mo_ssodrag.f90::sugwd`` does by :func:`echam_nktopg`; the
  host term derives it in ``cache_coords``.

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

# Security floor on the sub-grid orography standard deviation [m]. The whole
# SSO computation divides by orography statistics (``pstd``, and ``zoro`` /
# ``pdmod`` derived from them), which are exactly zero over flat terrain
# (ocean, aquaplanet). The drag there is legitimately zero, but ``x / 0``
# still produces an ``inf`` cotangent in reverse mode that a downstream mask
# cannot clean (``0·inf = nan``) — the multi-step reverse-mode NaN of issue
# #558. Flooring these denominators leaves the forward drag unchanged
# (numerator orography factors are also zero there) while keeping every VJP
# finite.
_MIN_OROG_STD = 1.0e-6      # [m]

# Floor on the squared wind speed [m²/s²] that divides the energy-conserving
# cap's quotient in ``_orodrag``. Both the forward ``0/0`` at an exactly calm
# level and the reverse-mode ``den**-2`` need it above float32's underflow of
# the square (see the comment at the cap); it is a speed of 1e-9 m/s, whose
# correction (``<= speed/dt``) is ~1e-12 m/s².
_MIN_SPEED2 = 1.0e-18


# ``mo_ssodrag.f90::sugwd``: the reference surface pressure [Pa] and the
# sigma at or below which a level counts as "near the surface" when locating
# ``nktopg``.
_NKTOPG_REFERENCE_PRESSURE = 80000.0   # zpr
_NKTOPG_SIGMA = 0.94                   # zsigt


def echam_nktopg(a_half, b_half) -> int:
    """ECHAM's ``nktopg`` for a hybrid grid (``mo_ssodrag.f90::sugwd``).

    ``nktopg`` is the 1-based index (top = 1) of the highest full level whose
    mid-level sigma, evaluated at a reference surface pressure of 800 hPa, is
    at least 0.94 — i.e. the top of the lowest few model levels. ``orosetup``
    applies it as ``kknu = MIN(kknu, nktopg)`` (and likewise ``kknu2``,
    ``kknub``): with the level index counting down from the model top, that
    *raises* the top of the blocked-flow / low-level averaging layer to at
    least ``nktopg``, so the layer is never thinner than the near-surface
    levels. It is a floor on the layer depth, not a ceiling, which is why the
    value must be this grid-derived level and not the model top: with
    ``nktopg = 1`` every ``kknu`` collapses to the model top and the "low-level"
    wind, stability and density are averaged over the whole column (jet and
    stratosphere included), giving surface stresses of O(10 N/m²) and an
    orographic momentum sink several times the surface friction.

    Args:
        a_half: hybrid ``A`` coefficients on the ``nlev + 1`` half levels, in
            Pa, top-first (``0`` for a pure sigma grid).
        b_half: hybrid ``B`` coefficients on the half levels, top-first.

    Returns:
        ``nktopg`` as a 1-based Python ``int`` in ``[1, nlev]``. On a grid
        whose lowest full level already sits above sigma 0.94 the Fortran loop
        never assigns ``nktopg``; the lowest full level (``nlev``) is returned
        there, the shallowest layer the scheme can represent.

    """
    import numpy as np

    a = np.asarray(a_half, dtype=np.float64)
    b = np.asarray(b_half, dtype=np.float64)
    nlev = a.shape[0] - 1
    zpr = _NKTOPG_REFERENCE_PRESSURE
    sigma_full = 0.5 * (a[:-1] + a[1:] + zpr * (b[:-1] + b[1:])) / zpr
    near_surface = np.nonzero(sigma_full >= _NKTOPG_SIGMA)[0]
    if near_surface.size == 0:
        return nlev
    return int(near_surface.min()) + 1


def _safe_denom(x, floor):
    """Floor ``|x|`` at ``floor`` (sign-preserving) for use as a divisor.

    Keeps ``x / _safe_denom(x, ...)`` finite in both the forward and — the
    point here — the reverse pass, where an unfloored ``1/x`` at ``x → 0``
    gives a NaN cotangent (issue #558). Inert wherever ``|x|`` already
    exceeds ``floor``.
    """
    return jnp.where(
        jnp.abs(x) < floor,
        jnp.where(x < 0.0, -floor, floor),
        x,
    )


def _lev_idx(n: int, horiz_ndim: int, dtype=jnp.int32) -> jnp.ndarray:
    """Return ``0..n-1`` shaped ``(n, 1, ..., 1)`` to broadcast over ``horiz_ndim`` axes."""
    return jnp.arange(n, dtype=dtype).reshape((n,) + (1,) * horiz_ndim)


def _pick_level(arr: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """Select ``arr[idx]`` along leading vertical axis 0 via masked reduction.

    Supports both 1-D ``(n,)`` single-column inputs with scalar ``idx`` and
    level-major ``(n, *horiz)`` batches with ``(*horiz)`` integer ``idx`` while
    avoiding per-column dynamic gathers that pad minor tiles on TPU.
    """
    levels = _lev_idx(arr.shape[0], arr.ndim - 1, dtype=idx.dtype)
    mask = levels == jnp.expand_dims(idx, axis=0)
    return jnp.sum(jnp.where(mask, arr, 0), axis=0)


@tree_math.struct
class SSOParameters:
    """Tunable parameters for the Lott & Miller (1997) SSO drag scheme.

    Static loop / level-index knobs (``nktopg``, the grid-derived floor
    on the blocked-layer depth — see :func:`echam_nktopg`; ``ntop`` for
    the stress-profile model top) are passed as Python kwargs to
    :func:`sso_drag` because they index into level arrays at JIT trace
    time.

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
    horiz = pum1.shape[1:]
    horiz_ndim = len(horiz)
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
        above = phgeo > jnp.expand_dims(thresh, axis=0)   # bool (nlev, *horiz)
        # Last True going top-down = deepest True. Compute via reversed argmax.
        rev_first = jnp.argmax(jnp.flip(above, axis=0).astype(jnp.int32), axis=0)
        idx_last = (nlev - 1 - rev_first).astype(jnp.int32)
        kknu = jnp.where(jnp.any(above, axis=0), idx_last, jnp.int32(nlev - 1))
        # Floor at ilevh (line 679: "IF(.NOT.ll1(jl,ilevh))kknu(jl)=ilevh")
        kknu = jnp.where(above[ilevh_idx], kknu, jnp.int32(ilevh_idx))
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
    # In 0-based: half-level index h (h=1..nlev-1) sits between full levels
    # h and h-1:
    #   rho[h] = 2*paphm1[h] / (rd*(T[h] + T[h-1]))    for h = 1..nlev-1
    # Half levels 0 and nlev are handled separately (top = 0, surface =
    # layer-mass-weighted low-level mean below).
    zero_lvl = jnp.zeros((1,) + horiz, dtype=pum1.dtype)
    zdp_half = papm1[1:nlev] - papm1[:nlev - 1]
    t_pair = ptm1[1:nlev] + ptm1[:nlev - 1]
    rho_half = 2.0 * paphm1[1:nlev] / c.rd / t_pair
    stab_half = (
        2.0 * c.grav ** 2 / c.cpd / t_pair
        * (1.0 - c.cpd * rho_half * (ptm1[1:nlev] - ptm1[:nlev - 1]) / zdp_half)
    )
    stab_half = jnp.maximum(stab_half, _MIN_BV_FREQ)
    prho_top = jnp.concatenate([zero_lvl, rho_half], axis=0)
    pstab_top = jnp.concatenate([zero_lvl, stab_half], axis=0)

    # ----- Mass-weighted low-level mean (lines 767-797) ---------------------
    # Fortran loops jk = klev..ilevh (1-based, going down), with the IF
    # condition jk >= kknu2 AND jk <= kknul further restricting. In 0-based:
    # h = nlev-1..ilevh-1, with kknu2 <= h <= kknul.
    h_arr = _lev_idx(nlev, horiz_ndim)
    levmask = (
        (h_arr >= jnp.expand_dims(kknu2, axis=0))
        & (h_arr <= kknul)
        & (h_arr >= ilevh_idx)
    )
    w = pmair * levmask.astype(pmair.dtype)
    zmair = jnp.sum(w, axis=0)
    pulow = jnp.sum(pum1 * w, axis=0) / zmair
    pvlow = jnp.sum(pvm1 * w, axis=0) / zmair
    # stab and rho at klev+1 (surface half) are layer-mass-weighted averages
    # of the half-level values aligned at full-level k -> half-level h = k.
    pstab_sfc = jnp.sum(pstab_top * w, axis=0) / zmair
    prho_sfc = jnp.sum(prho_top * w, axis=0) / zmair
    pstab = jnp.concatenate([pstab_top, jnp.expand_dims(pstab_sfc, axis=0)], axis=0)
    prho = jnp.concatenate([prho_top, jnp.expand_dims(prho_sfc, axis=0)], axis=0)

    znorm = jnp.sqrt(jnp.maximum(pulow ** 2 + pvlow ** 2, _MIN_LOW_LEVEL_WIND ** 2))
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
    pdmod = jnp.sqrt(jnp.maximum(pd1 ** 2 + pd2 ** 2, 1.0e-30))

    # ----- Project flow into wave-stress plane (lines 824-837) --------------
    zvt1 = pulow * pum1 + pvlow * pvm1
    zvt2 = -pvlow * pum1 + pulow * pvm1
    zvpf = (zvt1 * pd1 + zvt2 * pd2) / (znorm * pdmod)

    # ----- pvph: vertical interpolation onto half levels (lines 844-862) ----
    # For jk = 2..klev (1-based) → h = 1..nlev-1 (0-based):
    #   pvph[h] = ((paphm1[h]-papm1[h-1])*zvpf[h] + (papm1[h]-paphm1[h])*zvpf[h-1])
    #            / (papm1[h]-papm1[h-1])
    pvph_int = (
        (paphm1[1:nlev] - papm1[:nlev - 1]) * zvpf[1:nlev]
        + (papm1[1:nlev] - paphm1[1:nlev]) * zvpf[:nlev - 1]
    ) / zdp_half
    # kcrit (line 859): jk where pvph drops below gvsec, for h in [1..nlev-2].
    h_int = _lev_idx(nlev - 1, horiz_ndim) + 1
    kcrit_mask = (pvph_int < _MIN_LOW_LEVEL_WIND) & (h_int < nlev - 1)
    pvph_int_clamped = jnp.maximum(pvph_int, _MIN_LOW_LEVEL_WIND)
    pvph = jnp.concatenate(
        [zero_lvl, pvph_int_clamped, jnp.expand_dims(pvph_sfc, axis=0)],
        axis=0,
    )
    has_kcrit = jnp.any(kcrit_mask, axis=0)
    # Highest h with the condition (Fortran loops jk=2..klev, last assignment wins).
    kcrit_idx = jnp.where(
        has_kcrit,
        (nlev - 1 - jnp.argmax(jnp.flip(kcrit_mask, axis=0).astype(jnp.int32), axis=0)).astype(jnp.int32),
        jnp.int32(0),  # Fortran initial: kcrit=1 (1-based) = 0 (0-based)
    )

    # ----- Richardson number at half levels (lines 866-879) -----------------
    zdwind = jnp.maximum(jnp.abs(zvpf[1:nlev] - zvpf[:nlev - 1]), _MIN_LOW_LEVEL_WIND)
    pri_int = pstab[1:nlev] * (zdp_half / (c.grav * prho[1:nlev] * zdwind)) ** 2
    pri_int = jnp.maximum(pri_int, _CRITICAL_RICHARDSON)
    pri_sfc = jnp.full((1,) + horiz, 9999.0, dtype=pum1.dtype)
    pri = jnp.concatenate([zero_lvl, pri_int, pri_sfc], axis=0)

    # ----- kkenvh & kkcrith: cumulative Froude / wave-action integrals ------
    # Both integrals accumulate the same per-layer increment over full levels
    # k = 1..nlev-2 (Fortran jk = 2..klev-1): kkenvh integrates downward from
    # k >= kknu until crossing _CRITICAL_FROUDE, and kkcrith integrates upward
    # from k = nlev-2..1 until crossing pi/2.
    if nlev > 2:
        k_interior = _lev_idx(nlev - 2, horiz_ndim) + 1
        zwind_dotted = (
            (pulow * pum1[1:nlev - 1] + pvlow * pvm1[1:nlev - 1]) / znorm
        )
        zwind = jnp.sqrt(jnp.maximum(zwind_dotted ** 2, _MIN_LOW_LEVEL_WIND ** 2))
        zstabm = jnp.sqrt(jnp.maximum(pstab[1:nlev - 1], _MIN_BV_FREQ))
        zstabp = jnp.sqrt(jnp.maximum(pstab[2:nlev], _MIN_BV_FREQ))
        dz_froude = (
            pmair[1:nlev - 1]
            * ((zstabp / prho[2:nlev] + zstabm / prho[1:nlev - 1]) * 0.5)
            / zwind
        )

        active_pnu = k_interior >= jnp.expand_dims(kknu, axis=0)
        inc_pnu = jnp.where(active_pnu, dz_froude, 0.0)
        pnu_new = jnp.cumsum(inc_pnu, axis=0)
        pnu_prev = jnp.concatenate([zero_lvl, pnu_new[:-1]], axis=0)
        crossed_pnu = (
            (pnu_prev <= _CRITICAL_FROUDE)
            & (pnu_new > _CRITICAL_FROUDE)
            & active_pnu
        )
        kkenvh = jnp.where(
            jnp.any(crossed_pnu, axis=0),
            (jnp.argmax(crossed_pnu.astype(jnp.int32), axis=0) + 1).astype(jnp.int32),
            jnp.int32(nlev - 1),
        )

        inc_kc = jnp.flip(dz_froude, axis=0)
        znup_new = jnp.cumsum(inc_kc, axis=0)
        znup_prev = jnp.concatenate([zero_lvl, znup_new[:-1]], axis=0)
        crossed_kc = (znup_prev <= jnp.pi / 2) & (znup_new > jnp.pi / 2)
        kkcrith = jnp.where(
            jnp.any(crossed_kc, axis=0),
            (nlev - 2 - jnp.argmax(crossed_kc.astype(jnp.int32), axis=0)).astype(jnp.int32),
            jnp.int32(nlev - 1),
        )
    else:
        kkenvh = jnp.full(horiz, nlev - 1, dtype=jnp.int32)
        kkcrith = jnp.full(horiz, nlev - 1, dtype=jnp.int32)

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
    ppsi_full = ptheta * jnp.pi / 180.0 - zphi_per

    # ----- zzdep (vertical leakiness for blocked drag, lines 975-985) ------
    # Fortran loop range: jk = ilevh..klev (1-based, inclusive) → h in
    # [ilevh-1, nlev-1]. Inside, pzdep is nonzero only when jk >= kkenvh
    # AND kkenvh != klev. Outside the loop range, pzdep stays at the
    # initial 0.
    kkenvh_safe = jnp.maximum(kkenvh, 1)
    denom = _pick_level(phgeo, kkenvh_safe) - phgeo[nlev - 1]
    pzdep_raw = (_pick_level(phgeo, kkenvh_safe - 1) - phgeo) / _safe_denom(denom, 1.0)
    kkenvh_bc = jnp.expand_dims(kkenvh, axis=0)
    levmask_dep = (
        (h_arr >= kkenvh_bc)
        & (kkenvh_bc != nlev - 1)
        & (h_arr >= ilevh_idx)
    )
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
    pstab_sfc_safe = jnp.maximum(pstab_sfc, _MIN_BV_FREQ)
    zeff_full = ppic - pval
    zeff_blocked = jnp.minimum(
        _CRITICAL_FROUDE * pvph_sfc / jnp.sqrt(pstab_sfc_safe),
        zeff_full,
    )
    zeff = jnp.where(kkenvh < nlev - 1, zeff_blocked, zeff_full)
    ptau0 = (gkdrag * prho_sfc
             * psig * pdmod / 4.0 / _safe_denom(pstd, _MIN_OROG_STD)
             * pvph_sfc * jnp.sqrt(pstab_sfc_safe)
             * zeff ** 2)
    return ptau0


# ---------------------------------------------------------------------------
# gwprofil — vertical stress profile + saturation
# ---------------------------------------------------------------------------

def _gwprofil(paphm1, prho, pri, pstab, pvph, pdmod, ptau0, pstd, psig,
              kkcrith, kcrit, ntop, nlev):
    """Compute the vertical profile of gravity-wave stress (lines 1131-1291).

    Returns ``ptau`` of leading length ``nlev+1``.
    """
    horiz_ndim = paphm1.ndim - 1
    # zoro per column (line 1141)
    zoro = psig * pdmod / 4.0 / _safe_denom(pstd, _MIN_OROG_STD)

    # Initial ztau (line 1142-1143): ztau[nlev] = ptau0; ztau[kkcrith] = grahilo*ptau0.
    # Loop 430 fills ptau for h = nlev..1 (0-based); h = 0 stays at 0.
    paphm1_sfc = paphm1[nlev]
    paphm1_kc = _pick_level(paphm1, kkcrith)
    zdelpt = paphm1_kc - paphm1_sfc
    # Floor the interpolation-depth denominator (sign-preserving) so reverse-
    # mode cotangents stay finite where ``kkcrith == nlev`` (issue #558).
    zdelpt = jnp.where(
        jnp.abs(zdelpt) < 1.0,
        jnp.where(zdelpt < 0.0, -1.0, 1.0),
        zdelpt,
    )
    full_idx = _lev_idx(nlev + 1, horiz_ndim)
    kkcrith_bc = jnp.expand_dims(kkcrith, axis=0)
    kcrit_bc = jnp.expand_dims(kcrit, axis=0)
    interp = (
        ptau0
        + (paphm1 - paphm1_sfc) / zdelpt * (_TRAPPED_WAVE_FRAC * ptau0 - ptau0)
    )
    ptau_init = jnp.where(
        full_idx > kkcrith_bc, interp,
        jnp.where(full_idx > 0, _TRAPPED_WAVE_FRAC * ptau0, 0.0),
    )

    # ----- Saturation sweep (lines 1191-1231) -------------------------------
    # Across half-levels h = nlev..1 (upward from kkcrith), the wave Richardson
    # criterion caps the local stress at ptau_new_uncapped[h] and enforces
    # ptau[h] <= ptau[h+1], zeroing stress once ptau[h+1] < _MIN_ANISOTROPY or
    # h <= kcrit. Because min(., ptau[h+1]) is monotone upward from kkcrith,
    # the upward recurrence equals a reverse cumulative minimum seeded with
    # ptau_init[kkcrith] at h == kkcrith and zeroed where the level below has
    # already dropped below _MIN_ANISOTROPY.
    prho_s = prho[1:]
    pstab_s = pstab[1:]
    pvph_s = pvph[1:]
    pri_s = pri[1:]
    ptau_init_s = ptau_init[1:]
    pstab_safe = jnp.maximum(pstab_s, _MIN_BV_FREQ)

    znorm = prho_s * jnp.sqrt(pstab_safe) * pvph_s
    zdz2 = (
        ptau_init_s / jnp.maximum(znorm, _MIN_BV_FREQ)
        / _safe_denom(zoro, _MIN_ANISOTROPY)
    )
    zsqr = jnp.sqrt(jnp.maximum(pri_s, 1e-30))
    zalfa = (
        jnp.sqrt(jnp.maximum(pstab_s * zdz2, 0.0))
        / _safe_denom(pvph_s, _MIN_LOW_LEVEL_WIND)
    )
    zriw = pri_s * (1.0 - zalfa) / (1.0 + zalfa * zsqr) ** 2
    zdel = (
        4.0 / zsqr / _CRITICAL_RICHARDSON
        + 1.0 / _CRITICAL_RICHARDSON ** 2
        + 4.0 / _CRITICAL_RICHARDSON
    )
    zb = 1.0 / _CRITICAL_RICHARDSON + 2.0 / zsqr
    zalpha = 0.5 * (-zb + jnp.sqrt(jnp.maximum(zdel, 0.0)))
    zdz2n = (pvph_s * zalpha) ** 2 / pstab_safe
    ptau_capped = znorm * zdz2n * zoro
    ptau_new_uncapped = jnp.where(
        zriw < _CRITICAL_RICHARDSON, ptau_capped, ptau_init_s
    )

    h_sat = _lev_idx(nlev, horiz_ndim) + 1  # 1..nlev
    cap_h = jnp.where(h_sat <= kcrit_bc, 0.0, ptau_new_uncapped)
    seed = jnp.where(
        h_sat < kkcrith_bc,
        cap_h,
        jnp.where(h_sat == kkcrith_bc, ptau_init_s, jnp.finfo(ptau_init.dtype).max),
    )
    cum_min = lax.cummin(seed, axis=0, reverse=True)
    cum_min_below = jnp.concatenate([cum_min[1:], cum_min[-1:]], axis=0)
    ptau_sat_s = jnp.where(
        h_sat < kkcrith_bc,
        jnp.where(cum_min_below < _MIN_ANISOTROPY, 0.0, cum_min),
        ptau_init_s,
    )
    ptau_after_sat = jnp.concatenate([ptau_init[:1], ptau_sat_s], axis=0)

    # ----- Reorganization at low level + model top (lines 1242-1291) -------
    ztau_kc = _pick_level(ptau_after_sat, kkcrith)
    ztau_top = ptau_after_sat[ntop - 1]   # ntop-1 = 0-based equivalent
    ztau_sfc = ptau_after_sat[nlev]
    interp2 = (
        ztau_sfc
        + (paphm1 - paphm1_sfc) / _safe_denom(paphm1_kc - paphm1_sfc, 1.0)
        * (ztau_kc - ztau_sfc)
    )
    ptau_final = jnp.where(
        (full_idx > kkcrith_bc) & (full_idx < nlev),
        interp2, ptau_after_sat,
    )
    ptau_final = jnp.where(full_idx < ntop - 1, ztau_top, ptau_final)
    return ptau_final


# ---------------------------------------------------------------------------
# Energy-conserving cap + frictional heating (orodrag lines 481-494)
# ---------------------------------------------------------------------------

def _energy_conserving_cap(u, v, dudt, dvdt, dt):
    """Cap the drag so it never accelerates the wind; return the heating.

    ECHAM (``mo_ssortns.f90::orodrag``) forms ``zdis = 0.5*(|u|² - |u*|²)`` for
    the updated wind ``u* = u + dt*a`` and, ``IF (zdis < 0)``, scales ``u*`` back
    to the old speed ``|u|``; the drag may turn or slow the wind but never
    accelerate it. That is ``u* -> u* * min(1, |u|/|u*|)``, written here with no
    ``IF``/``where`` at all, for two reasons that both come from ``zdis`` being
    a difference of nearly equal squares:

    * ``gain = 0.5*(|u*|² - |u|²)`` is formed from the increment
      (``|u*|² - |u|² = Δu (2u + Δu) + Δv (2v + Δv)``), so it is exactly 0
      wherever the drag is exactly 0, in any evaluation order. Formed as a
      difference of squares it is rounding noise of either sign (about 1e-5
      m²/s² at 30 m/s in float32) at every level the scheme does not touch.
    * a branch test on that noise, consumed in more than one place, is
      evaluated independently at each: XLA:CPU duplicates the cheap
      elementwise producer of ``zdis`` into every consumer fusion and LLVM
      contracts multiply-adds differently in each copy (an operand with a
      second use cannot be fused), so a ``speed2_new = where(zdis < 0, ...,
      1)`` guard and the select that applies the rescale can disagree. The
      rescale then runs on a quotient built from the guard's placeholder
      denominator, i.e. ``zred`` equal to the wind speed, and a level the drag
      did not touch is accelerated by up to 2.5 m/s² (the CPU-only blow-up of
      #981). A select whose arms differ by orders of magnitude at its own
      threshold is not safe to drive from a quantity that is zero in exact
      arithmetic.

    ``max(gain, 0)`` is continuous, and the quotient below is ``|u|²/|u*|²``
    when the wind gained energy and ``|u|²/|u|² = 1`` otherwise, so every
    evaluation of it, whichever side of the threshold rounding puts it on, is
    within round-off of the other.

    The quotient's denominator is ``|u|² + 2*gain_pos >= |u|²``, so it is never
    the updated wind's own square, which is exactly zero at an exactly calm
    level and — when the blocked-flow drag stops the wind in one step with
    ``orography_std`` just above ``_MIN_OROG_STD`` — nearly zero: the
    ``1/den**2`` derivative of a quotient over that square is infinite in
    float32 (#663). The denominator is still floored, at ``_MIN_SPEED2``,
    because ``0/0`` at ``|u|² = 0`` (calm level, no drag) is NaN and JAX's
    division derivative forms ``den**-2``, which only stays inside float32's
    range above ~1e-19; ``u*`` is also ~0 there, so the correction vanishes.

    Args:
        u, v: Wind before the drag [m/s].
        dudt, dvdt: Drag tendency before the cap [m/s²].
        dt: Physics time step [s].

    Returns:
        ``(dudt, dvdt, dissipation)``: the capped tendencies [m/s²] and the
        frictional heating ``zdis/dt`` [W/kg], which is the kinetic energy lost
        where the wind slowed and zero where it was capped back to its old
        speed (ECHAM's recomputed ``zdis`` is round-off there).

    """
    zdu = dt * dudt
    zdv = dt * dvdt
    gain = 0.5 * (zdu * (2.0 * u + zdu) + zdv * (2.0 * v + zdv))
    gain_pos = jnp.maximum(gain, 0.0)
    speed2_old = u ** 2 + v ** 2
    zred = jnp.sqrt(jnp.maximum(
        speed2_old / jnp.maximum(speed2_old + 2.0 * gain_pos, _MIN_SPEED2),
        1.0e-30))
    # u*·zred - u == dt*a + u*·(zred - 1), so the tendency is unchanged (exactly,
    # not to round-off) wherever zred == 1.
    dudt = dudt + (u + zdu) * (zred - 1.0) / dt
    dvdt = dvdt + (v + zdv) * (zred - 1.0) / dt
    return dudt, dvdt, jnp.maximum(-gain, 0.0) / dt


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
    ztemp = -(ptau[1:] - ptau[:-1]) / (
        _safe_denom(pvph_sfc, _MIN_LOW_LEVEL_WIND) * pmair)
    pdmod_safe = _safe_denom(pdmod, _MIN_ANISOTROPY)
    zdudt_wave = (pulow * pd1 - pvlow * pd2) * ztemp / pdmod_safe
    zdvdt_wave = (pvlow * pd1 + pulow * pd2) * ztemp / pdmod_safe
    # Overshoot guard (line 423-429). Floor the sum-of-squares INSIDE the
    # sqrt: at the (common) zero-wind / zero-drag state ``sqrt(0)`` has an
    # infinite derivative, so ``sqrt(u²+v²)`` — even where a downstream
    # ``maximum`` or ``where`` masks it — leaves a NaN cotangent (issue #558).
    zforc = jnp.sqrt(jnp.maximum(zdudt_wave ** 2 + zdvdt_wave ** 2, 1.0e-30))
    ztend = jnp.sqrt(jnp.maximum(pum1 ** 2 + pvm1 ** 2, 1.0e-30)) / pdtime
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
    zconb = 2.0 * pdtime * gkwake * psig / (4.0 * _safe_denom(pstd, _MIN_OROG_STD))
    zabsv = jnp.sqrt(jnp.maximum(pum1 ** 2 + pvm1 ** 2, 1.0e-30)) / 2.0
    cos_psi = jnp.cos(ppsi_full)
    sin_psi = jnp.sin(ppsi_full)
    zzd1 = zb * cos_psi ** 2 + zc * sin_psi ** 2
    ratio = ((cos_psi ** 2 + pgam_safe * sin_psi ** 2)
             / (pgam_safe * cos_psi ** 2 + sin_psi ** 2))
    zbet = (jnp.maximum(0.0, 2.0 - 1.0 / ratio)
            * zconb * pzdep * zzd1 * zabsv)
    block_du = -pum1 / pdtime * zbet / (1.0 + zbet)
    block_dv = -pvm1 / pdtime * zbet / (1.0 + zbet)
    use_block = (
        (gkwake != 0.0)
        & (_lev_idx(nlev, pum1.ndim - 1) > jnp.expand_dims(kkenvh, axis=0))
    )
    zdudt = jnp.where(use_block, block_du, zdudt_wave)
    zdvdt = jnp.where(use_block, block_dv, zdvdt_wave)

    # Energy dissipation and the energy-conserving cap (lines 481-494).
    zdudt, zdvdt, pdis = _energy_conserving_cap(
        pum1, pvm1, zdudt, zdvdt, pdtime)
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
    config: SSOParameters,
    *,
    nktopg: int,
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
        config: tunable :class:`SSOParameters`.
        nktopg: 1-based level index (top = 1) that the tops of the
            blocked-flow and low-level averaging layers are raised to at
            least (Python int, static at trace time). A property of the
            vertical grid — compute it with :func:`echam_nktopg`; there is
            no grid-free default (``1`` would stretch the "low-level" layer
            over the whole column).
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

    # Activation mask (ssodrag's ``itest``). No land-fraction factor, as in
    # ``mo_ssortns.f90::ssodrag``, which takes no land mask: the descriptors
    # are statistics over the whole grid cell with its ocean part entering as
    # zero elevation (``jcm.data.mirror.sso``, as in ECHAM's boundary files),
    # so a coastal cell's smaller ocean-diluted ``orostd``/``orosig``/peak
    # already carry its land fraction. Scaling the drag by it again would
    # count the ocean part twice.
    zero = jnp.zeros_like(drag_u)
    drag_u = jnp.where(active, drag_u, zero)
    drag_v = jnp.where(active, drag_v, zero)
    dissipation = jnp.where(active, dissipation, zero)

    u_stress = jnp.sum(drag_u * layer_mass, axis=0)
    v_stress = jnp.sum(drag_v * layer_mass, axis=0)
    dissip_total = jnp.sum(dissipation * layer_mass, axis=0)

    return (
        SSOTendencies(dudt=drag_u, dvdt=drag_v, dissip=dissipation),
        SSOState(u_stress=u_stress, v_stress=v_stress,
                 dissip_total=dissip_total),
    )


# ---------------------------------------------------------------------------
# Composable physics term wrapper
# ---------------------------------------------------------------------------

from typing import ClassVar  # noqa: E402

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
    ``orogam``, ``orothe``, ``oropic``, ``oroval``) from
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
        self._nktopg: int | None = None

    def cache_coords(self, coords) -> None:
        """Derive ``nktopg`` from the vertical grid (``sugwd``).

        Handles ``HybridCoordinates`` (``a`` in Pa, ``b``) and
        ``SigmaCoordinates`` (``a = 0``, ``b = sigma``), both top-first.
        """
        from dinosaur.hybrid_coordinates import HybridCoordinates

        vertical = coords.vertical
        if isinstance(vertical, HybridCoordinates):
            a_half, b_half = vertical.a_boundaries, vertical.b_boundaries
        else:
            b_half = jnp.asarray(vertical.boundaries)
            a_half = jnp.zeros_like(b_half)
        self._nktopg = echam_nktopg(a_half, b_half)

    def __call__(
        self,
        state: PhysicsState,
        diagnostics: dict,
        forcing: ForcingData,
        terrain: TerrainData,
    ) -> tuple[PhysicsTendency, dict]:
        """Compute u/v/T tendencies from Lott-Miller SSO."""
        if self._nktopg is None:
            raise RuntimeError(
                "LottMillerSso needs cache_coords(coords) before it is "
                "called: nktopg is derived from the vertical grid.")
        nktopg = self._nktopg
        # Flatten any trailing horizontal axes into a single column axis so
        # both grid-layout and column-vectorized hosts execute natively in
        # level-major ``(nlev, ncols)`` order without transposing.
        horiz = state.temperature.shape[1:]
        nlev = state.temperature.shape[0]

        def _cols(x):
            """Flatten the horizontal of a level-major field to one axis."""
            return x.reshape(x.shape[0], -1)

        dt = diagnostics["_dt_seconds"]
        params = self.params.get_value()

        pressure_full = _cols(diagnostics["pressure_full"])
        pressure_half = _cols(diagnostics["pressure_half"])
        height_full = _cols(diagnostics["height_full"])
        temperature = _cols(state.temperature)
        u_wind = _cols(state.u_wind)
        v_wind = _cols(state.v_wind)
        ncols = temperature.shape[1]

        layer_mass = (
            (pressure_half[1:, :] - pressure_half[:-1, :])
            / _physical_constants.grav
        )
        # Coriolis is consumed only by the unported mountain-lift branch.
        coriolis = jnp.zeros((ncols,), dtype=temperature.dtype)
        orog = terrain.orog.reshape(-1)

        tend, _state = sso_drag(
            jnp.asarray(dt),
            coriolis,
            height_full,
            orog,
            pressure_half,
            pressure_full,
            layer_mass,
            temperature,
            u_wind,
            v_wind,
            orog,
            terrain.orostd.reshape(-1),
            terrain.orosig.reshape(-1),
            terrain.orogam.reshape(-1),
            terrain.orothe.reshape(-1),
            terrain.oropic.reshape(-1),
            terrain.oroval.reshape(-1),
            params,
            nktopg=nktopg,
            ntop=1,
        )

        dt_temperature = tend.dissip / _physical_constants.cpd

        def _grid(x):
            return x.reshape((nlev,) + horiz)

        return PhysicsTendency(
            u_wind=_grid(tend.dudt),
            v_wind=_grid(tend.dvdt),
            temperature=_grid(dt_temperature),
            specific_humidity=jnp.zeros_like(state.specific_humidity),
            tracers={},
        ), diagnostics
