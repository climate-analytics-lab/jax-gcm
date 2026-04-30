"""Hines (1997) Doppler-spread spectral non-orographic gravity-wave drag.

Faithful JAX port of ECHAM ``mo_gw_hines.f90`` (atm_phy_echam version,
2326 lines). Reference: Hines (1997), J. Atmos. Solar-Terr. Phys.

The port targets the production-default control flow:

- ``naz = 8``  azimuths
- ``slope = 1``  spectral slope
- ``lheatcal = .TRUE.``  compute heating + diffusion
- ``icutoff = 0``  no exponential damping above ``alt_cutoff``
- ``lfront = lozpr = lrmscon_lat = .FALSE.``  no frontal/precip/lat sources
- ``sigsqmcw = 0``  orographic-wave coupling off (``sigmatm = sigalpmc = 0``
  throughout the column, so the ``f2mfac`` correction vanishes and
  ``n_over_m = f2 * sigma_t``)

The four other-slope code paths in the Fortran (``slope=1.5``, ``slope=2``,
and the orographic-coupling correction) are not implemented; passing
``slope != 1.0`` to ``HinesParameters.default`` will trigger an assertion at
trace time.

Subroutine map (Fortran → JAX helper):

- ``gw_hines``      → :func:`hines_gwd` (entry point)
- ``hines_extro``   → ``_hines_extro`` (internal driver)
- ``hines_wavnum``  → ``_hines_wavnum`` (level-by-level sweep, lax.scan)
- ``hines_wind``    → ``_hines_wind`` (8-azimuth projection)
- ``hines_flux``    → ``_hines_flux`` (momentum flux + drag)
- ``hines_heat``    → ``_hines_heat`` (heating + diffusion)
- ``hines_sigma``   → ``_hines_sigma`` (azimuthal/total RMS)
- ``hines_intgrl``  → ``_hines_intgrl`` (i_alpha integral, slope=1)
- ``vert_smooth``   → ``_vert_smooth`` (1-2-1 passes)
- ``hines_exp``     → not ported (icutoff=0 in production)

Brunt-Vaisala calculation in :func:`hines_gwd` mirrors lines 286-309 of
``mo_gw_hines.f90``: pressure-coordinate finite difference of T/Pi^kappa
plus a single-pass log-pressure smoothing.
"""
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import tree_math

# ICON physical constants — match values in mo_physical_constants.f90.
_GRAV = 9.80665      # m/s^2
_RD = 287.04         # J/(kg K)
_CPD = 1004.64       # J/(kg K)
_COS45 = 0.7071067811865476


@tree_math.struct
class HinesParameters:
    """Parameters for the Hines (1997) Doppler-spread non-orographic GWD scheme.

    Field-by-field provenance:

    Namelist (``mo_echam_gwd_config``):
        ``rmscon``, ``emiss_lev``, ``kstar``, ``m_min``, ``lheatcal``.

    Module-level constants (``mo_gw_hines.f90`` lines 58-72):
        ``naz``, ``slope``, ``f1``, ``f2``, ``f3``, ``f5``, ``f6``,
        ``icutoff``, ``alt_cutoff``, ``smco``, ``nsmax``.
    """

    rmscon: jnp.ndarray
    emiss_lev: jnp.ndarray
    kstar: jnp.ndarray
    m_min: jnp.ndarray
    lheatcal: jnp.ndarray
    naz: jnp.ndarray
    slope: jnp.ndarray
    f1: jnp.ndarray
    f2: jnp.ndarray
    f3: jnp.ndarray
    f5: jnp.ndarray
    f6: jnp.ndarray
    icutoff: jnp.ndarray
    alt_cutoff: jnp.ndarray
    smco: jnp.ndarray
    nsmax: jnp.ndarray

    @classmethod
    def default(
        cls,
        rmscon: float = 1.0,
        emiss_lev: int = 10,
        kstar: float = 5e-5,
        m_min: float = 1e-4,
        lheatcal: bool = True,
        naz: int = 8,
        slope: float = 1.0,
        f1: float = 1.5,
        f2: float = 0.3,
        f3: float = 1.0,
        f5: float = 1.0,
        f6: float = 0.5,
        icutoff: int = 0,
        alt_cutoff: float = 105e3,
        smco: float = 2.0,
        nsmax: int = 5,
    ) -> "HinesParameters":
        # Let JAX pick the default dtype — tests/production run at f32 by
        # default; the Fortran-comparison harness enables x64 explicitly.
        return cls(
            rmscon=jnp.asarray(rmscon),
            emiss_lev=jnp.asarray(emiss_lev),
            kstar=jnp.asarray(kstar),
            m_min=jnp.asarray(m_min),
            lheatcal=jnp.asarray(1.0 if lheatcal else 0.0),
            naz=jnp.asarray(naz),
            slope=jnp.asarray(slope),
            f1=jnp.asarray(f1),
            f2=jnp.asarray(f2),
            f3=jnp.asarray(f3),
            f5=jnp.asarray(f5),
            f6=jnp.asarray(f6),
            icutoff=jnp.asarray(icutoff),
            alt_cutoff=jnp.asarray(alt_cutoff),
            smco=jnp.asarray(smco),
            nsmax=jnp.asarray(nsmax),
        )


class HinesState(NamedTuple):
    """Diagnostic outputs from the Hines scheme."""

    flux_u: jnp.ndarray       # zonal momentum flux profile (Pa)
    flux_v: jnp.ndarray       # meridional momentum flux profile (Pa)
    diffco: jnp.ndarray       # vertical diffusion coefficient (m^2/s)


class HinesTendencies(NamedTuple):
    """Tendencies from the Hines scheme."""

    dudt: jnp.ndarray         # m/s^2
    dvdt: jnp.ndarray         # m/s^2
    dissip: jnp.ndarray       # energy dissipation rate (W/kg = J/(s*kg))


# ---------------------------------------------------------------------------
# Helper: 8-azimuth wind projection
# ---------------------------------------------------------------------------

def _hines_wind_8(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Project ``(u, v)`` onto 8 azimuths.

    Mirrors ``hines_wind`` (lines 1053-1080) for the ``naz=8`` branch.
    Output ordering: 1=E, 2=NE, 3=N, 4=NW, 5=W, 6=SW, 7=S, 8=SE.
    Returns shape ``(nlev, 8)``.
    """
    umin = 0.001
    u = jnp.where(jnp.abs(u) < umin, jnp.copysign(umin, u), u)
    v = jnp.where(jnp.abs(v) < umin, jnp.copysign(umin, v), v)
    vpu = v + u
    vpu = jnp.where(jnp.abs(vpu) < umin, jnp.copysign(umin, vpu), vpu)
    vmu = v - u
    vmu = jnp.where(jnp.abs(vmu) < umin, jnp.copysign(umin, vmu), vmu)

    a1 = u
    a2 = _COS45 * vpu
    a3 = v
    a4 = _COS45 * vmu
    return jnp.stack([a1, a2, a3, a4, -a1, -a2, -a3, -a4], axis=-1)


# ---------------------------------------------------------------------------
# Helper: total + per-azimuth RMS wind from sigsqh_alpha (8-azimuth case)
# ---------------------------------------------------------------------------

def _sigma_8(sigsqh_alpha: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute (sigma_t, sigma_alpha) from per-azimuth squared RMS at one level.

    Mirrors ``hines_sigma`` (lines 1492-1528) for ``naz=8``. Input shape is
    ``(8,)``; returns ``sigma_t`` scalar and ``sigma_alpha`` of shape ``(8,)``.
    Uses the ECHAM convention where azimuths 5..8 mirror 1..4 in the output.
    """
    s = sigsqh_alpha
    sum_odd = (s[0] + s[2] + s[4] + s[6]) * 0.5
    sum_even = (s[1] + s[3] + s[5] + s[7]) * 0.5
    sa1 = jnp.sqrt(s[0] + s[4] + sum_even)
    sa2 = jnp.sqrt(s[1] + s[5] + sum_odd)
    sa3 = jnp.sqrt(s[2] + s[6] + sum_even)
    sa4 = jnp.sqrt(s[3] + s[7] + sum_odd)
    sigma_alpha = jnp.stack([sa1, sa2, sa3, sa4, sa1, sa2, sa3, sa4])
    sigma_t = jnp.sqrt(jnp.sum(s))
    return sigma_t, sigma_alpha


# ---------------------------------------------------------------------------
# Helper: Hines integral i_alpha for slope = 1
# ---------------------------------------------------------------------------

def _i_alpha_slope1(v_alpha: jnp.ndarray, m_alpha: jnp.ndarray,
                    bvfb: jnp.ndarray, m_min: jnp.ndarray) -> jnp.ndarray:
    """Compute the Hines integral i_alpha at one level for ``slope=1``.

    Mirrors ``hines_intgrl`` lines 1629-1716. Vectorised over azimuth (last
    axis). Returns the integral of (1 - q*m')/(1 - q*m_alpha) - 1 weighted
    appropriately; positive by construction. Uses the analytic form
    everywhere except near ``|qm| < qm_min`` or ``|q_alpha| < q_min`` where a
    4-term Taylor expansion is used (Horner-evaluated, matching the Fortran
    line 1706-1710).

    The result is forced to zero where ``m_alpha <= m_min`` (the do_alpha
    mask in the Fortran).
    """
    q_min = 1.0
    qm_min = 0.01

    rbvfb = 1.0 / bvfb
    q_alpha = v_alpha * rbvfb
    qm = q_alpha * m_alpha
    qmm = q_alpha * m_min

    # Analytic branch (slope=1, line 1683): -(log(1-qm) - (1-qm) - log(1-qmm) + (1-qmm)) / q_alpha^2
    safe = jnp.maximum(jnp.abs(q_alpha), 1e-30)
    inv_q2 = 1.0 / (q_alpha * q_alpha + 1e-300)
    one_m_qm = 1.0 - qm
    one_m_qmm = 1.0 - qmm
    ana = -(jnp.log(jnp.maximum(one_m_qm, 1e-300))
            - one_m_qm
            - jnp.log(jnp.maximum(one_m_qmm, 1e-300))
            + one_m_qmm) * inv_q2

    # Taylor branch (line 1706, Horner-form): qm^2 (1/2 + qm(1/3 + qm(1/4 + qm/5))) - same with qmm.
    one_third = 1.0 / 3.0
    poly_qm = qm * qm * (0.5 + qm * (one_third + qm * (0.25 + qm * 0.2)))
    poly_qmm = qmm * qmm * (0.5 + qmm * (one_third + qmm * (0.25 + qmm * 0.2)))
    taylor = jnp.where(
        jnp.abs(q_alpha) < 1e-30,
        0.5 * (m_alpha * m_alpha - m_min * m_min),
        (poly_qm - poly_qmm) * inv_q2,
    )

    # Pick branch by condition (Fortran lines 1660-1662).
    use_taylor = (jnp.abs(qm) < qm_min) | (jnp.abs(q_alpha) < q_min)
    i_alpha = jnp.where(use_taylor, taylor, ana)

    # Round-off guard (line 1687) and m_alpha<=m_min mask (line 1644).
    i_alpha = jnp.maximum(i_alpha, 0.0)
    i_alpha = jnp.where(m_alpha <= m_min, 0.0, i_alpha)
    return i_alpha


# ---------------------------------------------------------------------------
# Helper: 1-2-1 vertical smoother, applied N times, interior only
# ---------------------------------------------------------------------------

def _vert_smooth(arr: jnp.ndarray, coeff: float, nsmooth: int,
                 lev1: int, lev2: int) -> jnp.ndarray:
    """Apply the ``vert_smooth`` 1-coeff-1 smoother ``nsmooth`` times.

    Mirrors lines 2125-2142 of ``mo_gw_hines.f90``: only indices
    ``lev1+1 .. lev2-1`` are averaged; ``lev1`` and ``lev2`` stay fixed, and
    indices outside ``[lev1, lev2]`` are not touched. Index 0 = top.
    """
    sum_wts = coeff + 2.0

    def one_pass(a):
        # Smooth only the interior (lev1+1 .. lev2-1).
        idxs = jnp.arange(lev1 + 1, lev2)
        prev_vals = a[idxs - 1]
        cur_vals = a[idxs]
        next_vals = a[idxs + 1]
        smoothed = (prev_vals + coeff * cur_vals + next_vals) / sum_wts
        return a.at[idxs].set(smoothed)

    def body(_, a):
        return one_pass(a)

    return lax.fori_loop(0, nsmooth, body, arr)


# ---------------------------------------------------------------------------
# Helper: Brunt-Vaisala frequency from T profile (column)
# ---------------------------------------------------------------------------

def _brunt_vaisala(t: jnp.ndarray, p_full: jnp.ndarray, p_sfc: jnp.ndarray
                   ) -> jnp.ndarray:
    """Compute Brunt-Vaisala frequency at full levels from T profile.

    Direct port of lines 286-309 in ``mo_gw_hines.f90``. The Fortran works in
    sigma = p/p_sfc and uses a finite difference of T/exner against sigma to
    estimate dT/dz, then converts to N. Index 0 = top.
    """
    rgocp = _RD / _CPD
    sgj = p_full / p_sfc
    shxkj = sgj ** rgocp

    # Layer-centred dT/dsigma (lines 289-291); jk=0 left at jk=1 value.
    dttdsf = (t[1:] / shxkj[1:] - t[:-1] / shxkj[:-1]) / (sgj[1:] - sgj[:-1])
    dttdsf = jnp.minimum(dttdsf, -5.0 / sgj[1:])
    dttdsf = dttdsf * shxkj[1:]
    bvf2 = -dttdsf * sgj[1:] / _RD
    bvf2 = jnp.maximum(bvf2, 0.0)
    bvfreq_inner = jnp.sqrt(bvf2) * _GRAV / t[1:]

    # Pad index 0 with index 1 (line 302).
    bvfreq = jnp.concatenate([bvfreq_inner[:1], bvfreq_inner])

    # Single-pass log-sigma smoothing (lines 304-309).
    def smooth_step(carry, k):
        prev = carry
        ratio = 5.0 * jnp.log(sgj[k] / sgj[k - 1])
        cur = (prev + ratio * bvfreq[k]) / (1.0 + ratio)
        return cur, cur

    _, smoothed = lax.scan(smooth_step, bvfreq[0],
                           jnp.arange(1, bvfreq.shape[0]))
    return jnp.concatenate([bvfreq[:1], smoothed])


# ---------------------------------------------------------------------------
# Main column algorithm (slope=1, naz=8, lheatcal=true)
# ---------------------------------------------------------------------------

def _hines_extro_column(
    bvfreq: jnp.ndarray,         # (nlev,)
    density: jnp.ndarray,        # (nlev,)
    mair: jnp.ndarray,           # (nlev,)
    uhs: jnp.ndarray,            # (nlev,) wind relative to launch
    vhs: jnp.ndarray,            # (nlev,)
    rmswind: jnp.ndarray,        # scalar
    config: HinesParameters,
    levbot: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Column-mode Hines drag algorithm.

    Returns ``(drag_u, drag_v, heat, diffco, flux_u, flux_v)`` each of shape
    ``(nlev,)``. All work done assuming ``slope=1, naz=8``.
    """
    nlev = bvfreq.shape[0]
    naz = 8
    f1 = config.f1
    f2 = config.f2
    f3 = config.f3
    f5 = config.f5
    f6 = config.f6
    kstar = config.kstar
    m_min = config.m_min
    visc_min = 1.0e-10

    # Visc. molecular profile (line 325); constant in this scheme.
    visc_mol = jnp.full((nlev,), 1.5e-5)

    bvfb = bvfreq[levbot]
    densb = density[levbot]

    # Project winds onto 8 azimuths (full column).
    v_alpha = jax.vmap(_hines_wind_8)(uhs, vhs)   # (nlev, 8)

    # --- Initial conditions at the launch level (lines 794-826) -----------
    sqr_rms_wind = rmswind * rmswind
    anis = jnp.full((naz,), 1.0 / naz)
    sigsqh_launch = anis * sqr_rms_wind                                # (8,)

    sigma_t_launch, sigma_alpha_launch = _sigma_8(sigsqh_launch)
    mmsq = m_min * m_min
    m_alpha_launch = bvfb / (f1 * sigma_alpha_launch + f2 * sigma_t_launch)
    ak_alpha = 2.0 * sigsqh_launch / (m_alpha_launch ** 2 - mmsq)
    mmin_alpha_launch = m_alpha_launch                                  # (8,)

    # --- Initial array buffers, indexed top..bottom -----------------------
    m_alpha = jnp.full((nlev, naz), m_min)
    m_alpha = m_alpha.at[levbot].set(m_alpha_launch)
    sigma_alpha = jnp.zeros((nlev, naz))
    sigma_alpha = sigma_alpha.at[levbot].set(sigma_alpha_launch)
    sigsqh_alpha = jnp.zeros((nlev, naz))
    sigsqh_alpha = sigsqh_alpha.at[levbot].set(sigsqh_launch)
    sigma_t = jnp.zeros((nlev,))
    sigma_t = sigma_t.at[levbot].set(sigma_t_launch)

    # --- Sweep upward from levbot-1 down to index 0 ------------------------
    # Process level l using state at lbelow = l + 1.
    # The scan iterates over step = 0 .. levbot-1; level = levbot-1 - step.
    # Note: we keep mmin_alpha as a per-step running minimum.

    def step(carry, idx):
        m_alpha, sigma_t, sigma_alpha, sigsqh_alpha, mmin_alpha, do_alpha = carry
        l = idx
        lbelow = l + 1

        # n_over_m (line 881): with sigmatm=0 throughout, f2mfac=0 →
        #   n_over_m = f2 * sigma_t[lbelow]
        n_over_m_turb = f2 * sigma_t[lbelow]
        # m_sub_m_mol cubic-root branch (line 894).
        visc = jnp.maximum(visc_mol[l], visc_min)
        m_sub_m_mol = jnp.cbrt(bvfreq[l] * kstar / visc) / f3
        # Equivalent rewrite of "if m_sub_m_turb >= m_sub_m_mol" using m_sub_m_turb = bvfreq/n_over_m:
        n_over_m = jnp.where(
            bvfreq[l] / n_over_m_turb >= m_sub_m_mol,
            bvfreq[l] / m_sub_m_mol,
            n_over_m_turb,
        )

        # m_trial (line 923) — sigalpmc=0, so omitted.
        m_trial = bvfb / (f1 * sigma_alpha[lbelow] + n_over_m + v_alpha[l])
        m_trial = jnp.where(m_trial <= 0.0, mmin_alpha, m_trial)
        m_trial = jnp.minimum(m_trial, mmin_alpha)
        m_trial = jnp.maximum(m_trial, m_min)
        m_alpha_l = jnp.where(do_alpha, m_trial, m_min)
        mmin_alpha_new = jnp.minimum(mmin_alpha, m_alpha_l)

        # Hines integral (line 943).
        i_alpha = _i_alpha_slope1(v_alpha[l], m_alpha_l, bvfb, m_min)
        # do_alpha turns false where m_alpha <= m_min (line 1645).
        do_alpha_new = do_alpha & (m_alpha_l > m_min)

        # New variances (lines 952-961).
        sigfac = (densb / density[l]) * (bvfreq[l] / bvfb)
        sigsqh_new = sigfac * ak_alpha * i_alpha
        sigma_t_new, sigma_alpha_new = _sigma_8(sigsqh_new)

        m_alpha = m_alpha.at[l].set(m_alpha_l)
        sigsqh_alpha = sigsqh_alpha.at[l].set(sigsqh_new)
        sigma_t = sigma_t.at[l].set(sigma_t_new)
        sigma_alpha = sigma_alpha.at[l].set(sigma_alpha_new)

        new_carry = (m_alpha, sigma_t, sigma_alpha, sigsqh_alpha,
                     mmin_alpha_new, do_alpha_new)
        return new_carry, None

    init_do_alpha = jnp.full((naz,), True)
    indices = jnp.arange(levbot - 1, -1, -1)  # levbot-1, levbot-2, ..., 0
    init = (m_alpha, sigma_t, sigma_alpha, sigsqh_alpha,
            mmin_alpha_launch, init_do_alpha)
    (m_alpha, sigma_t, sigma_alpha, sigsqh_alpha, _, _), _ = lax.scan(
        step, init, indices,
    )

    # losigma_t mask (line 970); equivalent to sigma_t > eps.
    losigma_t = sigma_t > 1e-30

    # --- Vertical smoothing (lines 572-589) -------------------------------
    # Fortran range is lev1=0 (top) to lev2=levbot inclusive; only the
    # interior 1..levbot-1 is averaged.
    nsmax = int(config.nsmax)
    smco = float(config.smco)
    if nsmax > 0:
        m_alpha = jnp.stack(
            [_vert_smooth(m_alpha[:, n], smco, nsmax, 0, levbot)
             for n in range(naz)],
            axis=-1,
        )
        sigma_t = _vert_smooth(sigma_t, smco, nsmax, 0, levbot)

    # --- Flux + drag (lines 1086-1303), slope=1, naz=8 --------------------
    k_alpha = jnp.full((naz,), kstar)
    ak_k_alpha = ak_alpha * k_alpha
    flux_per_az = ak_k_alpha[None, :] * (m_alpha - m_min) * densb
    flux_u = (flux_per_az[:, 0] - flux_per_az[:, 4]
              + _COS45 * (flux_per_az[:, 1] - flux_per_az[:, 3]
                          - flux_per_az[:, 5] + flux_per_az[:, 7]))
    flux_v = (flux_per_az[:, 2] - flux_per_az[:, 6]
              + _COS45 * (flux_per_az[:, 1] + flux_per_az[:, 3]
                          - flux_per_az[:, 5] - flux_per_az[:, 7]))

    # Drag at intermediate levels: -d(flux)/dmair  (line 1273).
    # lev1=0 (top): one-sided forward (line 1288): drag = flux/mair.
    # No level below levbot is processed (since lev2 = levbot, lev2p>nlevs branch skipped).
    drag_u_int = -(flux_u[:-1] - flux_u[1:]) / mair[1:]
    drag_v_int = -(flux_v[:-1] - flux_v[1:]) / mair[1:]
    drag_u_top = flux_u[0] / mair[0]
    drag_v_top = flux_v[0] / mair[0]
    drag_u = jnp.concatenate([jnp.array([drag_u_top]), drag_u_int])
    drag_v = jnp.concatenate([jnp.array([drag_v_top]), drag_v_int])

    # Below levbot, drag_u/drag_v stay at their initialised zero value.
    below_launch = jnp.arange(nlev) > levbot
    drag_u = jnp.where(below_launch, 0.0, drag_u)
    drag_v = jnp.where(below_launch, 0.0, drag_v)

    # --- Heating + diffusion (lines 1374-1424), if lheatcal --------------
    visc = jnp.maximum(visc_mol, visc_min)
    m_sub_m_turb = bvfreq / (f2 * jnp.maximum(sigma_t, 1e-30))
    m_sub_m_mol = jnp.cbrt(bvfreq * kstar / visc) / f3
    m_sub_m = jnp.minimum(m_sub_m_turb, m_sub_m_mol)

    # dfdz at intermediate levels (line 1382). For lev1 (top), the Fortran
    # uses ``mair(i,l)`` where ``l`` is the leftover loop variable from the
    # prior ``DO l = lev1p, lev2`` loop. With gfortran's standard behaviour
    # ``l = lev2 + lincr = levbot + 1`` after the loop ends, so the actual
    # value is ``mair[levbot + 1]`` (1-based) = ``mair[levbot]`` (0-based,
    # since 1-based indexing offset shifts by one too). ``m_sub_m`` and
    # ``sigma_alpha`` ARE re-evaluated at lev1 (lines 1390-1393). We mirror
    # the leftover-loop-variable bug bit-for-bit since it's part of the
    # reference output.
    factor = f1 * sigma_alpha + (bvfreq / jnp.maximum(m_sub_m, 1e-30))[:, None]
    dfdz_int = (flux_per_az[:-1] - flux_per_az[1:]) / mair[1:, None] * factor[1:]
    # In 0-based indexing, levbot is the launch level and levbot+1 is the
    # next level down — same as the post-loop l value in 1-based Fortran.
    dfdz_top = -flux_per_az[0] / mair[levbot + 1] * factor[0]
    dfdz = jnp.concatenate([dfdz_top[None, :], dfdz_int], axis=0)

    heatng = -f5 * jnp.sum(dfdz, axis=-1)
    heat = jnp.where(losigma_t, heatng, 0.0)
    # Avoid NaN cube root for non-positive heatng.
    safe_heat = jnp.maximum(heatng, 0.0)
    diffco = jnp.where(
        losigma_t & (heatng > 0.0),
        f6 * jnp.cbrt(safe_heat) / jnp.maximum(m_sub_m, 1e-30) ** (4.0 / 3.0),
        0.0,
    )
    # Apply lheatcal switch.
    heat = jnp.where(config.lheatcal > 0.5, heat, 0.0)
    diffco = jnp.where(config.lheatcal > 0.5, diffco, 0.0)

    return drag_u, drag_v, heat, diffco, flux_u, flux_v


def hines_gwd(
    paphm1: jnp.ndarray,
    papm1: jnp.ndarray,
    pzh: jnp.ndarray,
    prho: jnp.ndarray,
    pmair: jnp.ndarray,
    ptm1: jnp.ndarray,
    pum1: jnp.ndarray,
    pvm1: jnp.ndarray,
    config: HinesParameters,
) -> Tuple[HinesTendencies, HinesState]:
    """Compute Hines GWD tendencies for a single column.

    Mirrors the entry point ``gw_hines`` in ``mo_gw_hines.f90`` line 76. All
    arrays use the ECHAM convention: index 0 = top, index ``nlev-1`` =
    surface. ``paphm1`` and ``pzh`` are on half levels (length ``nlev+1``);
    the rest are on full levels (length ``nlev``).
    """
    nlev = pum1.shape[0]
    p_sfc = paphm1[-1]

    bvfreq = _brunt_vaisala(ptm1, papm1, p_sfc)

    # levbot is a Python int so the scan loop length is static. Fortran uses
    # 1-based ``levbot = nlev - emiss_lev``; subtract one for 0-based.
    levbot = nlev - int(config.emiss_lev) - 1
    levbot_arr = jnp.asarray(levbot)

    uhs = pum1 - pum1[levbot_arr]
    vhs = pvm1 - pvm1[levbot_arr]
    # Above the launch level the relative wind is what propagates upward;
    # below the launch level, gw_hines zeros out the drag (we mask later).
    # The Fortran loops only over jk=1..levbot for uhs/vhs (line 343-346);
    # below that the array is uninitialised but is never read by hines_extro
    # which uses lev2 = levbot. We zero below for clarity.
    below_launch = jnp.arange(nlev) > levbot
    uhs = jnp.where(below_launch, 0.0, uhs)
    vhs = jnp.where(below_launch, 0.0, vhs)

    rmswind = config.rmscon  # constant per-column in production path

    drag_u, drag_v, heat, diffco, flux_u, flux_v = _hines_extro_column(
        bvfreq, prho, pmair, uhs, vhs, rmswind, config, levbot,
    )

    return (
        HinesTendencies(dudt=drag_u, dvdt=drag_v, dissip=heat),
        HinesState(flux_u=flux_u, flux_v=flux_v, diffco=diffco),
    )
