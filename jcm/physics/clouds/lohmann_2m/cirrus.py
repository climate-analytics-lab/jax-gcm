"""Kaercher-Lohmann cirrus ice nucleation (jax-gcm#1017 task 2 part 2a).

JAX port of ECHAM6.3-HAM2.3 r7492 ``mo_cirrus.f90``: ``SCRHOM``, ``PISAT``,
``TAUG`` (closed-form helpers) and ``XFRZMSTR``/``XFRZHOM``/``XICEHOM`` (the
column-level nucleation calculation ``nic_cirrus = 2`` needs). Produces
``mo_cloud_micro_2m.f90``'s ``ZNICEX``/``ZRI`` -- the number and mean radius
of newly homogeneously-frozen ice crystals -- from one level's temperature,
pressure, updraft and ice supersaturation, plus the aerosol number available
to freeze.

**Only the reachable code is ported.** ``XFRZHET``/``XICEHET`` (heterogeneous
freezing below 235 K) and every aerosol-size-effect branch of
``XICEHOM``/``XICEHET`` are UNREACHABLE in any supported HAM configuration:

* ``lhetfreeze`` -- the switch ``ham_IN_setup`` forwards as ``XFRZMSTR``'s
  ``LHET`` -- is an ``em_error`` ("not currently supported") unless ECHAM is
  compiled with ``-DWITH_LHET`` (``mo_ham.f90:616-622``), which the reference
  build is not, so ``ld_het`` is always ``.FALSE.`` and ``XFRZHET`` is never
  called.
* ``nosize`` -- the flag ``mo_cloud_micro_2m.f90`` passes as ``XFRZMSTR``'s
  ``NOSIZE`` -- is a compile-time ``PARAMETER .true.``
  (``mo_cloud_micro_2m.f90:446``), so ``XICEHOM``'s (and ``XICEHET``'s)
  size-dependent, lognormal-bin-integration branches never run either.

Porting ``XFRZHET``/the size-dependent branches would be ~350 lines of code
with zero path to ever being exercised through this wiring; this module
ports only ``XICEHOM``'s ``NOSIZE`` branch (``mo_cirrus.f90:1038-1048``) and
documents the omission here rather than leaving a stub. If ECHAM-HAM is ever
rebuilt with ``-DWITH_LHET`` and ``ham_IN_setup`` changes accordingly, this
module would need the heterogeneous branch added; nothing here assumes
``ld_het`` is a jcm *choice* -- it is HAM's own.

Because ``NOSIZE`` is always true, the aerosol radius (``ZAPRX``) and
geometric standard deviation (``ZAPSIGX``) XICEHOM would otherwise read never
affect the result (confirmed: its ``IF (NOSIZE)`` branch uses only the
SUMMED aerosol number, never ``R``/``SIG`` individually) -- this module's
functions therefore do not take them as arguments at all.

Units follow the Fortran's own CGS convention internally (cm, cm/s, 1/cm3,
hPa, g); :func:`karcher_lohmann_frz` converts at the boundary from jcm's SI
fields, the same pattern ``deposition_freezing.py`` and ``assembly.py`` use
for ``paprx``/``prwetai`` elsewhere in this scheme. ``XFRZMSTR`` itself
already converts its two OUTPUTS back to SI (``ZRI=ZRI1*1e-2``,
``ZNICEX=ZNICE*1e6``), so only the inputs need conversion here.

Differentiability: whenever a cell is gated out (the pre-check fails, or no
freezing temperature is found within 120 search steps -- both ordinary,
frequent outcomes, not edge cases), its inputs are first replaced by a
fixed, regular, interior point (T=210 K, susati=0.5, updraft=50 cm/s,
aerosol=1e8 cm-3) before the whole pipeline runs, so neither AD mode ever
differentiates a division by a near-zero ``CTOT``, ``COOLR`` or ``CTAU``
on a path whose result the final ``jnp.where`` discards anyway -- the
project's double-``where`` convention (``JAX_gotchas.md``), applied once at
the top of the pipeline rather than at each individual division.
"""
from __future__ import annotations

import jax.numpy as jnp

from ..lohmann_2m_params import CloudParams2M

# --- mo_cirrus.f90's own CGS physical constants (literals, not jcm's SI
# constants module -- this is a faithfulness port of the Fortran's exact
# numbers, including its deliberately truncated PI/THIRD/SQ31 literals,
# which are NOT full double-precision pi/third/1-over-root-3). -------------
_RHOICE = 0.925        # g/cm3                  mo_cirrus.f90:272
_PI = 3.1415927        # truncated literal       mo_cirrus.f90:273
_ALPHA = 0.5
_THIRD = 0.3333333     # truncated literal       mo_cirrus.f90:275
_RGAS = 8.3145e7
_BK = 1.3807e-16
_CPAIR = 1.00467e7
_HEAT = 2830.3e7
_AVOG = 6.02213e23
_GRAV = 981.0          # cm/s2
_SVOL = 3.23e-23
_XMW = 2.992e-23
_XWW = 18.016
_XWA = 28.966
_TWOPI = 6.2831853
_SIX1 = 0.166666666    # truncated literal       mo_cirrus.f90:190
_SQ31 = 0.577350269    # truncated literal       mo_cirrus.f90:189
_KMAX = 120
_VOLF = _PI * _RHOICE / 0.75


def _scrhom(t: jnp.ndarray) -> jnp.ndarray:
    """``SCRHOM`` (mo_cirrus.f90:15-36): critical ice saturation ratio for
    homogeneous freezing, at 0.25 um particle size.
    """
    temp = jnp.clip(t, 170.0, 240.0)
    return 2.418 - temp / 245.68


def _pisat(t: jnp.ndarray) -> jnp.ndarray:
    """``PISAT`` (mo_cirrus.f90:40-61): vapour pressure over ice [hPa]."""
    return 0.01 * 10.0 ** (12.537 - 2663.5 / t)


def _taug(b: jnp.ndarray, y: jnp.ndarray, x0: jnp.ndarray) -> jnp.ndarray:
    """``TAUG`` (mo_cirrus.f90:180-207): dimensionless growth time scale.

    ``y <= x0`` returns 0 in the Fortran (an early ``RETURN``); here that is
    the discarded branch of the outer ``jnp.where``, computed on a SAFE
    ``x = max(y, x0 + 1e-3)`` so ``log``/``atan`` never see ``x <= x0`` (which
    would not itself be singular, but keeps the two branches' arguments in
    the same well-behaved range the real search uses).
    """
    active = y > x0
    x_safe = jnp.where(active, jnp.minimum(y, 0.999), jnp.minimum(x0 + 1.0e-3, 0.999))
    f1 = _SIX1 * jnp.log((1.0 + x_safe + x_safe * x_safe) / (1.0 - x_safe) ** 2)
    f10 = _SIX1 * jnp.log((1.0 + x0 + x0 * x0) / (1.0 - x0) ** 2)
    f2 = _SQ31 * jnp.arctan(_SQ31 * (1.0 + 2.0 * x_safe))
    f20 = _SQ31 * jnp.arctan(_SQ31 * (1.0 + 2.0 * x0))
    taug = (b + 1.0) * (f1 - f10) + (b - 1.0) * (f2 - f20)
    return jnp.where(active, taug, 0.0)


def _xicehom_nosize(
    phi: jnp.ndarray, tau: jnp.ndarray, b1: jnp.ndarray, b2: jnp.ndarray,
    ctot: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``XICEHOM``'s ``IF (NOSIZE)`` branch (mo_cirrus.f90:1038-1048).

    ``YK`` (computed in the Fortran but never read again in this branch) is
    omitted. ``ctot`` here is ``XFRZHOM``'s ``CCR`` (``C*PWCR/PW``), not the
    caller's raw aerosol number -- the scaling XICEHOM actually sees.
    """
    ci = jnp.minimum(
        _SVOL * (b2 / (_TWOPI * b1)) ** 1.5 * phi / jnp.sqrt(tau), ctot)
    xmihat = _XMW * _PI * phi * tau / 6.0
    rihat = (xmihat / (_VOLF * ci)) ** _THIRD
    return ci, rihat


def _xfrzhom(
    ctot: jnp.ndarray, p_hpa: jnp.ndarray, v_cms: jnp.ndarray,
    coolr: jnp.ndarray, pw: jnp.ndarray, scr: jnp.ndarray, temp: jnp.ndarray,
    pice: jnp.ndarray, pwcr: jnp.ndarray, dt: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``XFRZHOM`` (mo_cirrus.f90:804-967), ``NOSIZE`` branch only.

    Returns ``(si, ci, ri)`` -- ``CI`` [1/cm3], ``RI`` [cm] -- matching the
    Fortran's own ``(SI, CI, RI)`` INTENT(out) order. The Fortran's ``T``
    argument (as opposed to ``TEMP``) is declared but never read anywhere in
    this subroutine's body, so it is not a parameter here.
    """
    pcr = p_hpa * pwcr / pw
    ccr = ctot * pwcr / pw

    ctau = jnp.maximum(2260.0 - 10.0 * temp, 100.0)   # NOSIZE branch, :919
    dlnjdt = jnp.abs(4.37 - 0.03 * temp)
    tau = 1.0 / (ctau * dlnjdt * coolr)

    diffc = 4.0122e-3 * temp ** 1.94 / pcr
    bkt = _BK * temp
    vth = jnp.sqrt(8.0 * bkt / (_PI * _XMW))
    cisat = 1.0e3 * pice / bkt
    theta = _HEAT * _XWW / (_RGAS * temp)
    a1 = (theta / _CPAIR - _XWA / _RGAS) * (_GRAV / temp)
    a2 = 1.0 / cisat
    a3 = 0.001 * _XWW ** 2 * _HEAT ** 2 / (_AVOG * _XWA * pcr * temp * _CPAIR)
    b1 = _SVOL * 0.25 * _ALPHA * vth * cisat * (scr - 1.0)
    b2 = 0.25 * _ALPHA * vth / diffc

    phi = v_cms * a1 * scr / (a2 + a3 * scr)
    ci, rihat = _xicehom_nosize(phi, tau, b1, b2, ccr)

    xmi0 = _VOLF * ci * rihat ** 3
    xmimax = xmi0 + _XMW * cisat * (scr - 1.0)
    rimax = (xmimax / (_VOLF * ci)) ** _THIRD

    xmisat = _XMW * cisat
    tgrow = 0.75 / (_PI * diffc * ci * rimax)
    zf = tgrow / dt
    xmfp = 3.0 * diffc / vth
    beta = xmfp / (0.75 * _ALPHA * rimax)
    x0 = rihat / rimax

    # ``DO X = 1.0, X0, -0.01; IF (Z<=1) EXIT`` (:627-632, :954-959):
    # evaluate the whole fixed 101-point grid at once and take the first
    # (largest) X that satisfies Z<=1 AND X>=X0 -- a vectorised equivalent
    # of the Fortran's descending search, not an approximation of it. If no
    # grid point satisfies it (the Fortran loop would then fall through
    # with X just below X0), fall back to X0 itself, matching that outcome.
    x_grid = 1.0 - 0.01 * jnp.arange(101, dtype=xmi0.dtype)       # (101,)
    x_grid_b = jnp.broadcast_to(x_grid.reshape((101,) + (1,) * x0.ndim),
                                 (101,) + x0.shape)
    z = zf[None] * _taug(beta[None], x_grid_b, x0[None])
    hit = (z <= 1.0) & (x_grid_b >= x0[None])
    any_hit = jnp.any(hit, axis=0)
    first = jnp.argmax(hit, axis=0)
    x_found = jnp.take_along_axis(x_grid_b, first[None], axis=0)[0]
    x = jnp.where(any_hit, x_found, x0)

    ri = x * rimax
    xmi = _VOLF * ci * ri ** 3
    si = scr - (xmi - xmi0) / xmisat
    return si, ci, ri


def xfrzmstr(
    ice_supersaturation: jnp.ndarray,   # susati = S_ice - 1 [1]
    updraft: jnp.ndarray,               # [m/s]
    aerosol_number: jnp.ndarray,        # apn [1/cm3] (already the depleted,
                                        # per-mode-summed quantity the
                                        # orchestrator passes -- see module
                                        # docstring; nfrzmod=1 always)
    temperature: jnp.ndarray,           # [K]
    pressure: jnp.ndarray,              # [Pa]
    dt: jnp.ndarray,                    # timestep [s]
    params: CloudParams2M,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``XFRZMSTR`` (mo_cirrus.f90:211-441): homogeneous cirrus nucleation.

    JAX port of the per-level gating/search loop ECHAM's
    ``mo_cloud_micro_2m.f90`` calls once per level inside its own
    ``nic_cirrus == 2`` loop (:1030-1038) -- here vectorised over whatever
    leading shape the inputs carry (one level's worth of columns, or a
    single column, per this scheme's broadcasting convention).

    Three stages, exactly as the Fortran: (1) a cheap pre-check -- would the
    air, cooling adiabatically over ``dt``, reach saturation ratio
    ``SCRHOM`` at its coldest (``TMIN``)? (2) if so, find the temperature
    (at ``T`` itself, or by stepping down through up to 120 discrete
    sub-steps) at which the ice saturation ratio first reaches ``SCRHOM``;
    (3) run :func:`_xfrzhom` there. A cell failing (1), or never finding a
    crossing in (2), returns exactly zero for both outputs (``ZRI``/
    ``ZNICEX`` stay at their Fortran zero-initialisation, :300-301) --
    this is the common case, not a fallback.

    Returns:
        ``(ri, nicex)``: mean radius [m] and number [1/m3] of newly
        homogeneously frozen ice crystals (``ZRI``, ``ZNICEX``).

    """
    # XFRZMSTR's own ``TMELT`` argument is declared but never read anywhere
    # in its body (mo_cirrus.f90:211-441); not threaded through here either.
    zthomi = params.cthomi
    zmin = params.eps
    t = temperature
    verv = updraft * 100.0       # m/s -> cm/s
    apn = aerosol_number
    si = ice_supersaturation + 1.0

    gate_pre = (
        (ice_supersaturation > 0.0) & (t < zthomi) & (verv > zmin)
        & (apn > 0.0) & (verv > 0.0)
    )

    # Double-``where``: substitute a safe, regular interior point wherever
    # this cell is gated out, so TAU/CTOT/CI/RIMAX below never divide by
    # (near-)zero on a branch the final select discards (module docstring).
    t_s = jnp.where(gate_pre, t, 210.0)
    verv_s = jnp.where(gate_pre, verv, 50.0)
    apn_s = jnp.where(gate_pre, apn, 1.0e8)
    si_s = jnp.where(gate_pre, si, 1.5)
    p_hpa_s = jnp.where(gate_pre, pressure * 0.01, 250.0)

    coolr = _GRAV * verv_s / _CPAIR
    tmin = jnp.maximum(t_s - coolr * dt, 170.0)
    scr_tmin = _scrhom(tmin)
    pw0 = si_s * _pisat(t_s)
    pwcr_tmin = pw0 * (tmin / t_s) ** 3.5
    pisat_tmin = _pisat(tmin)
    precheck_ok = (pwcr_tmin / pisat_tmin) >= scr_tmin
    gate1 = gate_pre & precheck_ok

    # --- freezing-temperature search (:339-387) ---------------------------
    scr_t = _scrhom(t_s)
    immediate = si_s >= scr_t

    k = jnp.arange(1, _KMAX + 2, dtype=t_s.dtype)             # (121,) K=1..121
    sl = (t_s - 170.0) / _KMAX
    k_b = k.reshape((_KMAX + 1,) + (1,) * t_s.ndim)
    temp_k = t_s[None] - sl[None] * (k_b - 1.0)
    scr_k = _scrhom(temp_k)
    pice_k = _pisat(temp_k)
    pwcr_k = pw0[None] * (temp_k / t_s[None]) ** 3.5
    hit_k = (pwcr_k / pice_k) >= scr_k
    any_hit = jnp.any(hit_k, axis=0)
    first_k = jnp.argmax(hit_k, axis=0)
    temp_at_k = jnp.take_along_axis(temp_k, first_k[None], axis=0)[0]
    scr_at_k = jnp.take_along_axis(scr_k, first_k[None], axis=0)[0]
    pice_at_k = jnp.take_along_axis(pice_k, first_k[None], axis=0)[0]
    pwcr_at_k = jnp.take_along_axis(pwcr_k, first_k[None], axis=0)[0]

    found = immediate | any_hit
    # On ``found``, dispatch exactly as :389-422: the immediate branch
    # reassigns PW=SCR*PICE; the K-search branch keeps the stage-1 PW
    # (``pw0``) unchanged and only recomputes PWCR/SCR/PICE at the found K.
    pice_t = _pisat(t_s)
    temp_f = jnp.where(immediate, t_s, temp_at_k)
    scr_f = jnp.where(immediate, scr_t, scr_at_k)
    pice_f = jnp.where(immediate, pice_t, pice_at_k)
    pw_f = jnp.where(immediate, scr_t * pice_t, pw0)
    pwcr_f = jnp.where(immediate, pw_f, pwcr_at_k)

    _si_out, ci, ri = _xfrzhom(
        apn_s, p_hpa_s, verv_s, coolr, pw_f, scr_f, temp_f, pice_f, pwcr_f, dt)

    final = gate1 & found
    ri_m = jnp.where(final, ri * 1.0e-2, 0.0)
    nicex_m3 = jnp.where(final, ci * 1.0e6, 0.0)
    return ri_m, nicex_m3
