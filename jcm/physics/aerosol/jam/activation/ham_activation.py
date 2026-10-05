"""ECHAM6.3-HAM2.3 M7 aerosol activation: Koehler A/B, ARG, updraft, Lin-Leaitch.

Faithful port of the r7492 routines (file:line as extracted into the
standalone harness -- see ``jcm/data/test/echam_cloud_reference/
ham_activ_{README.md,provenance.json}``):

* :func:`koehler_ab` -- ``mo_ham_activ.f90::ham_activ_koehler_ab`` (463-594)
* :func:`ham_arg` -- ``mo_ham_activ.f90::ham_activ_abdulrazzak_ghan`` (65-375)
* :func:`ham_logtail` -- ``mo_ham_tools.f90::ham_m7_logtail`` (203-320) with
  ``mo_ham_m7.f90::m7_cumulative_normal`` (75-326) as the closed-form normal
  tail (``erf``)
* :func:`ham_updraft` -- ``mo_activ.f90::activ_updraft`` (75-152) with
  ``aero_activ_updraft_sigma``/``aero_activ_updraft_pdf`` (154-238)
* :func:`lin_leaitch` -- ``mo_ham_activ.f90::ham_avail_activ_lin_leaitch``
  (606-732) + ``mo_activ.f90::activ_lin_leaitch`` (242-330)

This is HAM's OWN activation formulation, distinct from CAM's in
``activation/arg.py``: the Koehler ``A`` uses the *local* temperature (not a
fixed reference state) and HAM's own moist-air-corrected thermal
conductivity, and ``B`` comes from each mode's electrolyte species
(``nion``/osmotic coefficient) rather than a precomputed kappa. The ARG
size-dependent shape coefficients (``f``, ``g``, exponent 3/2) are
nevertheless the same closed form as ``arg.py``'s ``"arg2000"`` variant
(Abdul-Razzak & Ghan 2000) -- HAM and CAM share that part of the scheme.

Reference physical constants are HAM's own literal values
(``mo_physical_constants.f90``), not ``jcm.constants``: the Fortran reference
this module is held to (rtol <= 1e-12 in
``ham_activation_reference_test.py``) was compiled against those exact
literals, which differ from ``jcm.constants`` in the last 1-2 digits (e.g.
``argas`` 8.314472 here vs ``jcm.constants.r_universal`` 8.314462). Mixing in
the live, overridable ``jcm.constants`` singleton would both break that
tolerance and make the comparison track unrelated calibration changes. This
mirrors ``activation/arg.py``'s own ``_T_REF``/``_P_REF`` (CAM's literals,
not ``jcm.constants.tmelt``).

Units and axes throughout: mode axis first, i.e. ``(n_modes, *cell)`` for
every per-mode array (number in m^-3, dry/wet radius in m, mass mixing ratio
kg/kg); ``ham_updraft``'s velocity-bin axis is first, ``(n_w, *cell)``, by
the same convention (the bin axis is this function's "class" axis). Every
function is pure and broadcasting-native: no branch on ``ndim`` or unpacked
horizontal shape, so the identical code runs on a single column or a whole
``(nlev, ncols)``/``(nlev, ix, il)`` grid.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import erf

from jcm.physics.aerosol.jam.population import ModalAerosolSpec

# EPSILON(1.0_dp) -- the empty-class / "no activation" threshold throughout
# the reference routines (e.g. mo_ham_activ.f90:193).
ZEPS = 2.220446049250313e-16

# ---------------------------------------------------------------------------
# HAM's own physical constants (mo_physical_constants.f90), kept local and
# distinct from jcm.constants -- see the module docstring.
# ---------------------------------------------------------------------------
_RGAS = 8.314472          # [J/K/mol] mo_physical_constants.f90:60 (argas)
_AMW = 18.0154e-3         # [kg/mol]  mo_physical_constants.f90:75 (amw, g/mol * 1e-3)
_AMD = 28.970e-3          # [kg/mol]  mo_physical_constants.f90:80 (amd, g/mol * 1e-3)
_GRAV = 9.80665           # [m/s2]    mo_physical_constants.f90:96
_CPD = 1004.64            # [J/K/kg]  mo_physical_constants.f90:108
_RHOH2O = 1000.0          # [kg/m3]   mo_physical_constants.f90:125
_ALV = 2.5008e6           # [J/kg]    mo_physical_constants.f90:144
_TMELT = 273.15           # [K]       mo_physical_constants.f90:147
_CTHOMI = _TMELT - 35.0   # [K]       mo_echam_cloud_params.f90:54
# Surface tension of H2O, HAM's own fixed value (distinct from CAM's
# ``surften``/jcm's ``surface_tension_water``) -- mo_ham_activ.f90:176,511.
_ZSTEN = 75.0e-3          # [J/m2]
# Curvature-parameter prefactor (zafac, mo_ham_activ.f90:532): A = _A_FACTOR/T.
_A_FACTOR = 2.0 * _ZSTEN * _AMW / (_RHOH2O * _RGAS)

# Species electrolyte properties (mo_ham_species.f90 new_species() calls):
# (nion, osmotic coefficient). Non-electrolyte species (nion absent) default
# to (0, 0.0) in the real code too (mo_species.f90:202, ``speclist(:)%
# nion = 0``), so ``bc``/``oc``/``du`` are listed only for self-documentation.
HAM_ELECTROLYTE = {
    "so4": (2, 1.0),   # mo_ham_species.f90:294-311
    "bc": (0, 0.0),    # mo_ham_species.f90:343-355 (not lelectrolyte)
    "oc": (0, 0.0),    # mo_ham_species.f90:357-377
    "ss": (2, 1.0),    # mo_ham_species.f90:382-398
    "du": (0, 0.0),    # mo_ham_species.f90:404-416
}

# Lin & Leaitch (1997) empirical constants (mo_activ.f90:270-271).
_LL_C2 = 2.3e-10   # [m4 s-1]
_LL_C3 = 1.27      # [1]
# Lower size cut-offs of the instrument used by ham_avail_activ_lin_leaitch
# (mo_ham_activ.f90:622,626): stratiform 0.03 um, convective 0.02 um radius.
LL_CRCUT_STRAT = 0.03e-6
LL_CRCUT_CONV = 0.02e-6

# West et al. (2013) updraft PDF: mo_activ.f90 activ_initialize's
# ``SELECT CASE(ABS(nactivpdf)) CASE(1): nw = 20`` -- the default bin count
# when the PDF option is switched on (nactivpdf /= 0).
PDF_DEFAULT_BINS = 20
_W_SIGMA_MIN = 0.1   # [m/s] mo_activ.f90:69


def mode_col(x, ndim_cell):
    """Reshape a static per-mode 1-D array to broadcast against
    ``(n_modes, *cell)`` with ``ndim_cell = len(cell)``, without the caller
    having to know the cell rank in advance (arg_term.py's callers instead
    pre-reshape to a fixed rank; this module's functions serve column,
    vectorized-column and whole-grid hosts alike, so the rank is read off
    another argument instead -- see each function's first few lines).
    """
    return jnp.reshape(x, (-1,) + (1,) * ndim_cell)


def _safe(mask, num, den):
    """``num/den`` where ``mask``, ``0`` elsewhere, with a benign (1.0)
    denominator substituted off-mask so neither the forward value nor its
    gradient ever sees a true division by zero (the pattern used throughout
    ``ice_nucleation/ham_freezing.py``).
    """
    return jnp.where(mask, num / jnp.where(mask, den, 1.0), 0.0)


def koehler_ab(spec: ModalAerosolSpec, mass: dict, temperature: jnp.ndarray):
    """Koehler curvature parameter A [m] and hygroscopicity parameter B [-].

    Port of ``ham_activ_koehler_ab`` (mo_ham_activ.f90:463-594). Both are
    computed only for modes the population marks ``can_activate`` (HAM's
    ``sizeclass%lactivation``, dod #377) -- zero elsewhere, including a
    soluble-but-non-activating mode such as M7's nucleation mode NS. Within
    an activating mode, only *electrolyte* species (``nion > 0``) contribute
    to B's sums -- a non-electrolyte species' mass still counts in the mode's
    total mass (the ``massfrac`` denominator) but not in either running sum
    (mo_ham_activ.f90:562, the same ``IF (nion > 0 ...)`` gates both).

    Args:
        spec: the M7 population (mode ``species``/``can_activate``; species
            ``density`` [kg/m3] and ``molar_mass`` [kg/mol]).
        mass: ``(species_token, mode_short) -> mass mixing ratio [kg/kg]``
            array, broadcastable to ``temperature``'s shape; a missing key is
            treated as an all-zero field (the harness term supplies every
            member pair from the operator-split tracer view).
        temperature: ``(*cell)`` [K].

    Returns:
        ``(a_coef, b_coef)``, both ``(n_modes, *cell)``, in ``spec.modes``
        order.

    """
    zero = jnp.zeros_like(temperature)
    a_list, b_list = [], []
    for mode in spec.modes:
        if not mode.can_activate:
            a_list.append(zero)
            b_list.append(zero)
            continue
        masssum = zero
        for sp in mode.species:
            masssum = masssum + mass.get((sp, mode.short), 0.0)
        has_mass = masssum > ZEPS
        sumtop = zero
        sumbot = zero
        for sp in mode.species:
            nion, osm = HAM_ELECTROLYTE.get(sp, (0, 0.0))
            if nion <= 0:
                continue
            props = spec.species_props(sp)
            m_sp = mass.get((sp, mode.short), 0.0)
            massfrac = _safe(has_mass, m_sp, masssum)
            sumtop = sumtop + m_sp * nion * osm * massfrac / props.molar_mass
            sumbot = sumbot + m_sp / props.density
        ok = sumbot > ZEPS
        b_list.append(_safe(ok, _AMW * sumtop, _RHOH2O * sumbot))
        a_list.append(jnp.where(ok, _A_FACTOR / temperature, 0.0))
    return jnp.stack(a_list, axis=0), jnp.stack(b_list, axis=0)


def ham_logtail(count_median_radius, cutoff_radius, sigmaln, mass_factor=1.0):
    """Compute the number (or mass) fraction of a log-normal class above ``cutoff_radius``.

    Port of ``ham_m7_logtail`` (mo_ham_tools.f90:203-320) with
    ``m7_cumulative_normal`` (mo_ham_m7.f90:75-326) as the closed-form
    complementary normal CDF, ``0.5*(1-erf(z/sqrt(2)))``. ``mass_factor`` is
    HAM's ``cmedr2mmedr`` (count-to-mass-median-radius ratio), 1.0 for the
    number fraction every caller in this module needs; kept as a parameter
    so the function is a complete port of the (``ld_numb`` False) mass
    branch too, not because any current caller uses it.

    All three branches of mo_ham_tools.f90:295-315 are reproduced:
    ``cutoff_radius`` and ``count_median_radius`` both above EPSILON gives
    the normal-tail formula; a (numerically) zero ``cutoff_radius`` with a
    positive ``count_median_radius`` gives 1 (everything is "above" an
    infinitesimal threshold); a (numerically) empty class
    (``count_median_radius <= EPSILON``) gives 0 regardless of the cutoff.
    """
    cmr = count_median_radius * mass_factor
    has_class = count_median_radius > ZEPS
    above_cutoff = cutoff_radius > ZEPS
    normal_branch = has_class & above_cutoff
    safe_cmr = jnp.where(normal_branch, cmr, 1.0)
    safe_r = jnp.where(normal_branch, cutoff_radius, 1.0)
    zt = (jnp.log(safe_r) - jnp.log(safe_cmr)) / sigmaln
    tail = 0.5 * (1.0 - erf(zt / jnp.sqrt(2.0)))
    # has_class & ~above_cutoff -> 1.0 (mo_ham_tools.f90:307-309);
    # ~has_class -> 0.0 (mo_ham_tools.f90:311-313, the empty-class default).
    return jnp.where(normal_branch, tail, jnp.where(has_class, 1.0, 0.0))


def ham_updraft(tke, omega, air_density, w_min, fact_tke, n_pdf_bins=None):
    """Stratiform activation updraft velocity bins and their PDF weights.

    Port of ``activ_updraft`` (mo_activ.f90:75-152) with
    ``aero_activ_updraft_sigma``/``aero_activ_updraft_pdf`` (154-238).
    ``n_pdf_bins=None`` is HAM's ``nactivpdf = 0`` (the reference preset): a
    single characteristic updraft ``MAX(w_min, w_large + w_turb)`` with
    ``pwpdf = 1`` (the probability is irrelevant, as the Fortran comment at
    mo_activ.f90:140-142 notes -- it cancels out of the weighted mean in
    :func:`ham_arg`). An integer ``n_pdf_bins`` (HAM's ``nactivpdf != 0``,
    :data:`PDF_DEFAULT_BINS` = 20 is the West et al. 2013 default) instead
    returns that many Gaussian-PDF bins spanning ``[0, 4*sigma_w]`` -- note
    the bins are **not** centred on ``w_large`` and can sit entirely in its
    tail if ``|w_large|`` is large relative to ``sigma_w`` (a genuine Fortran
    behaviour, not a port artifact: see ``ham_activation_reference_test.py``
    and the harness provenance notes on the ``w_min``-binding cell).

    ``n_pdf_bins`` is a **static** Python int/None (the scheme's bin count is
    fixed at compose time, like ``nactivpdf`` itself), never a traced value.

    Args:
        tke: ``(*cell)`` turbulent kinetic energy [m2/s2] (``ptkem1``).
        omega: ``(*cell)`` large-scale vertical velocity [Pa/s] (``pvervel``).
        air_density: ``(*cell)`` [kg/m3].
        w_min: minimum characteristic updraft [m/s] (HAM's module default is
            0.0 and is never overridden by the reference namelist, but is
            exposed here as the differentiable tunable the term carries).
        fact_tke: turbulent-velocity prefactor (HAM: 0.7 for ARG, 1.33 for
            Lin & Leaitch -- mo_cloud_utils.f90:77, mo_activ.f90:120-125).
        n_pdf_bins: ``None`` for the single-updraft path, else the PDF bin
            count.

    Returns:
        ``(w, pwpdf)``, both ``(n_w, *cell)`` with ``n_w = 1`` or
        ``n_pdf_bins``.

    """
    # sqrt(0) has an infinite derivative: a laminar column (TKE == 0) is a
    # real, differentiated input (not a degenerate one to special-case away
    # at the call site), so both sqrt(TKE)-based quantities below use the
    # double-``jnp.where`` guard (substituting a safe operand off the
    # selected branch) rather than a bare ``jnp.sqrt(jnp.maximum(tke, 0))``
    # -- the same pattern ``arg_term.py``'s own TKE update already applies
    # for exactly this input.
    tke_nonneg = jnp.maximum(tke, 0.0)
    tke_pos = tke_nonneg > 0.0
    sqrt_tke = jnp.where(tke_pos, jnp.sqrt(jnp.where(tke_pos, tke_nonneg, 1.0)), 0.0)

    w_large = -omega / (_GRAV * air_density)
    w_turb = fact_tke * sqrt_tke
    if n_pdf_bins is None:
        w = jnp.maximum(w_min, w_large + w_turb)[jnp.newaxis, ...]
        return w, jnp.ones_like(w)
    sqrt_23tke = jnp.where(
        tke_pos, jnp.sqrt(jnp.where(tke_pos, (2.0 / 3.0) * tke_nonneg, 1.0)), 0.0)
    sigma_w = jnp.maximum(_W_SIGMA_MIN, sqrt_23tke)
    bin_width = 4.0 * sigma_w / n_pdf_bins
    # Bin centres (jw - 0.5)*width for the Fortran's 1-indexed jw = 1..nw
    # (mo_activ.f90:223); jw below is 0-indexed, so the same centre is
    # (jw + 1 - 0.5) = (jw + 0.5).
    jw = jnp.arange(n_pdf_bins).reshape((n_pdf_bins,) + (1,) * tke.ndim)
    w = (jw + 0.5) * bin_width[jnp.newaxis, ...]
    # West et al. (2013)'s continuous Gaussian density at each bin centre
    # (mo_activ.f90:225-228); not normalised to sum to 1 over the finite bin
    # set -- that normalisation happens via the weighted mean in ham_arg
    # (ham_activ_abdulrazzak_ghan's zfracn_top/zfracn_bot, mo_ham_activ.f90:
    # 357-360), exactly as the real code relies on it.
    pwpdf = (1.0 / jnp.sqrt(2.0 * jnp.pi)) / sigma_w[jnp.newaxis, ...] * jnp.exp(
        -((w - w_large[jnp.newaxis, ...]) ** 2) / (2.0 * sigma_w[jnp.newaxis, ...] ** 2)
    )
    return w, pwpdf


def ham_arg(r_dry, number_vol, a_coef, b_coef, can_activate, sigma_g,
            updraft, pwpdf, temperature, pressure, specific_humidity, esw):
    """HAM's Abdul-Razzak & Ghan (2000) activation on an M7-shaped population.

    Port of ``ham_activ_abdulrazzak_ghan`` (mo_ham_activ.f90:65-375). Reads
    the Koehler A/B (:func:`koehler_ab`) and the dry radius/number as given
    (HAM's M7 core computes these; this harness takes them as inputs, as the
    real routine does too -- ``prdry``/``pxtm1`` are both ``INTENT(IN)``).
    Only modes with ``can_activate`` contribute (KS, AS, CS for M7 -- the
    same mask :func:`koehler_ab` uses, so a non-activating mode's A/B/Sm/rc
    are consistently zero/dummy through both functions).

    Args:
        r_dry, number_vol, a_coef, b_coef: ``(n_modes, *cell)`` dry radius
            [m], number [m^-3], Koehler A [m] / B [-].
        can_activate: ``(n_modes,)`` bool, HAM's ``sizeclass%lactivation``.
        sigma_g: ``(n_modes,)`` geometric standard deviation [-].
        updraft, pwpdf: ``(n_w, *cell)`` from :func:`ham_updraft`.
        temperature, pressure, specific_humidity, esw: ``(*cell)`` [K], [Pa],
            [kg/kg], saturation vapour pressure [Pa] (``pesw``, an exogenous
            input to the real routine too -- it is never computed inside
            ``ham_activ_abdulrazzak_ghan``).

    Returns:
        ``cdncact`` ``(*cell)`` total activated CDNC [m^-3]; ``number_frac``,
        ``activated_number``, ``sm`` all ``(n_modes, *cell)`` (per-mode
        activated fraction [-], activated number [m^-3], critical
        supersaturation [-]); ``smax`` ``(n_w, *cell)`` maximum
        supersaturation per bin; ``rc`` ``(n_modes, n_w, *cell)`` critical
        radius [m] per mode and bin.

    """
    ndim_cell = temperature.ndim
    can_activate = mode_col(can_activate, ndim_cell).astype(bool)
    ln_sigma = mode_col(jnp.log(sigma_g), ndim_cell)
    f_co = 0.5 * jnp.exp(2.5 * ln_sigma ** 2)
    g_co = 1.0 + 0.25 * ln_sigma

    t, p, q = temperature, pressure, specific_humidity

    # Water-vapour diffusivity and the moist-air-corrected thermal
    # conductivity (mo_ham_activ.f90:242-258) -- distinct from CAM's
    # dry-air-only ndrop.F90 form in activation/arg.py.
    zdif = 0.211e-4 * (t / _TMELT) ** 1.94 * (101325.0 / p)
    zxv = q * (_AMD / _AMW)
    zka = (5.69 + 0.017 * (t - 273.15)) * 1.0e-5
    zkv = (3.78 + 0.020 * (t - 273.15)) * 1.0e-5
    zk = zka * (1.0 - (1.17 - 1.02 * zkv / zka) * zxv) * 418.68

    zgrowth = 1.0 / (
        (_RHOH2O * _RGAS * t) / (esw * zdif * _AMW)
        + (_ALV * _RHOH2O / (zk * t)) * (_ALV * _AMW / (_RGAS * t) - 1.0)
    )
    zalpha = (_GRAV * _AMW * _ALV) / (_CPD * _RGAS * t ** 2) - (_GRAV * _AMD) / (_RGAS * t)
    zgamma = (_RGAS * t) / (esw * _AMW) + (_AMW * _ALV ** 2) / (_CPD * p * _AMD * t)

    aw_over_g = zalpha[jnp.newaxis, ...] * updraft / zgrowth[jnp.newaxis, ...]   # (n_w,*cell)

    # Outer gate (mo_ham_activ.f90:230-232,312-314): a single-bin run
    # (n_w == 1) additionally requires that bin's updraft to exceed EPSILON;
    # a PDF run (n_w > 1) does not gate on w at all. Both require q and T
    # above EPSILON/cthomi. Sm itself is computed only inside this gate
    # (mo_ham_activ.f90:270-281 sit inside the same IF as zalpha/zgamma
    # above) -- outside it, zsm keeps its zero initial value
    # (mo_ham_activ.f90:185), which is what the per-mode ``sc`` output must
    # show too (the `cold_high_alt`/`w_min`-binding reference cells both
    # exercise this: a nonzero A/B/r_dry with the gate failed still reports
    # Sm = 0).
    n_w = updraft.shape[0]
    outer_ok = ((n_w > 1) | (updraft[0] > ZEPS)) & (q > ZEPS) & (t > _CTHOMI)

    # Per-mode critical supersaturation Sm (mo_ham_activ.f90:280-281),
    # nw-independent: mask1 is the outer gate plus the same "zn>eps,
    # rdry>1e-9, a>eps, b>eps" gate the Fortran uses for both Sm and the
    # smax summation below.
    mask1 = outer_ok[jnp.newaxis, ...] & (number_vol > ZEPS) & (r_dry > 1.0e-9) \
        & (a_coef > ZEPS) & (b_coef > ZEPS)
    sm = jnp.where(
        mask1,
        2.0 / jnp.sqrt(jnp.where(mask1, b_coef, 1.0))
        * (a_coef / jnp.where(mask1, 3.0 * r_dry, 1.0)) ** 1.5,
        0.0,
    )

    # zxi, zeta per (mode, bin) -- mo_ham_activ.f90:285-290. ``sqrt`` at
    # exactly 0 (a zero updraft, e.g. the w_min-binding reference cell) has
    # an infinite derivative; the double-``jnp.where`` substitutes a safe
    # operand off the branch that is selected there, so the gradient sees
    # ``maximum``'s own (finite, zero) one-sided derivative instead of
    # ``sqrt``'s (the same hardening pattern ``ice_nucleation/
    # ham_freezing.py`` and ``arg_term.py``'s TKE update use).
    aw_bcast = aw_over_g[jnp.newaxis, ...]
    aw_pos = aw_bcast > 0.0
    sqrt_aw = jnp.where(aw_pos, jnp.sqrt(jnp.where(aw_pos, aw_bcast, 1.0)), 0.0)
    zxi = (2.0 / 3.0) * a_coef[:, jnp.newaxis, ...] * sqrt_aw
    safe_n = jnp.where(mask1, number_vol, 1.0)
    zeta = aw_over_g[jnp.newaxis, ...] ** 1.5 / (
        2.0 * jnp.pi * _RHOH2O * zgamma[jnp.newaxis, ...] * safe_n[:, jnp.newaxis, ...]
    )

    # zsum (mo_ham_activ.f90:292-299): accumulated only where mask1 AND
    # can_activate AND the bin's own updraft exceeds EPSILON.
    sm_safe = jnp.where(mask1, sm, 1.0)
    pw_ok = updraft > ZEPS   # (n_w,*cell), broadcasts against (n_modes,n_w,*cell)
    term = f_co[:, jnp.newaxis, ...] * (zxi / jnp.where(zeta > ZEPS, zeta, 1.0)) ** 1.5 \
        + g_co[:, jnp.newaxis, ...] * (
            sm_safe[:, jnp.newaxis, ...] ** 2
            / jnp.where(zeta + 3.0 * zxi > ZEPS, zeta + 3.0 * zxi, 1.0)
        ) ** 0.75
    contribution = jnp.where(
        mask1[:, jnp.newaxis, ...] & can_activate[:, jnp.newaxis, ...] & pw_ok[jnp.newaxis, ...],
        term / sm_safe[:, jnp.newaxis, ...] ** 2,
        0.0,
    )
    inv_smax2 = jnp.sum(contribution, axis=0)   # (n_w,*cell)

    smax_ok = outer_ok[jnp.newaxis, ...] & (inv_smax2 > ZEPS)
    smax = jnp.where(smax_ok, 1.0 / jnp.sqrt(jnp.where(smax_ok, inv_smax2, 1.0)), 0.0)

    # Critical radius per (mode, bin) (mo_ham_activ.f90:335-345); the
    # un-activated default is 1 m (mo_ham_activ.f90:188), which
    # ham_logtail's own number-fraction formula naturally sends to ~0 for any
    # real aerosol radius -- reproduced exactly rather than shortcut, so the
    # masked branch matches the Fortran bit-for-bit, not just numerically.
    mask2 = smax_ok & (sm[:, jnp.newaxis, ...] > ZEPS) \
        & mask1[:, jnp.newaxis, ...]
    smax_safe = jnp.where(mask2, smax[jnp.newaxis, ...], 1.0)
    rc = jnp.where(
        mask2,
        r_dry[:, jnp.newaxis, ...] * (sm_safe[:, jnp.newaxis, ...] / smax_safe) ** (2.0 / 3.0),
        1.0,
    )

    fracn = ham_logtail(
        r_dry[:, jnp.newaxis, ...], rc, ln_sigma[:, jnp.newaxis, ...],
    )   # (n_modes, n_w, *cell)
    top = jnp.sum(fracn * pwpdf[jnp.newaxis, ...], axis=1)
    bot = jnp.sum(pwpdf, axis=0)
    bot_ok = bot > ZEPS
    number_frac = jnp.where(
        can_activate, jnp.where(bot_ok, top / jnp.where(bot_ok, bot, 1.0), 0.0), 0.0,
    )
    activated_number = number_frac * number_vol
    cdncact = jnp.sum(activated_number, axis=0)
    return cdncact, number_frac, activated_number, sm, smax, rc


def _lin_leaitch_available(number_vol, can_activate, wet_radius, sigmaln, crcut):
    """Available number [m^-3] above the instrument cut-off ``crcut``.

    Port of ``ham_avail_activ_lin_leaitch`` (mo_ham_activ.f90:606-732) for
    ``nham_subm = HAM_M7``. The per-mode cut factor ``cfracn`` is 1 for the
    four *soluble* M7 modes and 0 for the three insoluble ones
    (mo_ham_activ.f90:644), but both the ``ham_m7_logtail`` call and the
    final number sum additionally sit inside
    ``IF (sizeclass(jclass)%lactivation)`` (mo_ham_activ.f90:661,711) -- the
    same mask :func:`ham_arg` uses as ``can_activate``. For M7 that mask is
    already zero on every mode where ``cfracn`` is zero (the three insoluble
    modes) and, unlike ``cfracn``, is *also* zero on the soluble nucleation
    mode (NS): ``cfracn`` alone would include NS, but ``lactivation`` gates
    it out first, so NS never contributes. The net per-mode mask is
    therefore exactly ``can_activate`` (KS, AS, CS), not ``soluble``.
    """
    frac = ham_logtail(wet_radius, crcut, sigmaln)
    return jnp.sum(jnp.where(can_activate, number_vol * frac, 0.0), axis=0)


def _lin_leaitch_activate(available_number, updraft):
    """Activated CDNC [m^-3] from available number and updraft.

    Port of ``activ_lin_leaitch`` (mo_activ.f90:242-330), Lin & Leaitch
    (1997): ``pcdncact = 0.1e6*(1e-6*Nmax)**c3`` with
    ``Nmax = na*w/(w + c2*na)``, both gated on ``w > EPSILON`` and
    ``na > EPSILON`` (mo_activ.f90:304,320).
    """
    ok = (updraft > ZEPS) & (available_number > ZEPS)
    denom = updraft + _LL_C2 * available_number
    n_max = jnp.where(ok, available_number * updraft / jnp.where(ok, denom, 1.0), 0.0)
    out = 0.1e6 * (1.0e-6 * jnp.where(ok, n_max, 1.0)) ** _LL_C3
    return jnp.where(ok, out, 0.0)


def lin_leaitch(number_vol, can_activate, wet_radius, sigma_g, updraft):
    """Lin & Leaitch (1997) activation, stratiform and convective cuts.

    Port of ``ham_avail_activ_lin_leaitch`` (mo_ham_activ.f90:606-732) +
    ``activ_lin_leaitch`` (mo_activ.f90:242-330). Unlike :func:`ham_arg`,
    Lin & Leaitch never uses the updraft PDF (HAM's namelist enforces
    ``nw = 1`` for ``ncd_activ = 1``), so ``updraft`` is the single
    characteristic updraft from ``ham_updraft(..., n_pdf_bins=None)``,
    shape ``(*cell)`` (the caller squeezes out :func:`ham_updraft`'s
    length-1 bin axis) -- not ``(n_w, *cell)``.

    Args:
        number_vol: ``(n_modes, *cell)`` number [m^-3].
        can_activate: ``(n_modes,)`` bool, HAM's ``sizeclass%lactivation``
            -- the same mask :func:`ham_arg` uses (see
            :func:`_lin_leaitch_available`'s docstring for why this scheme's
            own ``cfracn`` factor reduces to it, not to ``soluble``).
        wet_radius: ``(n_modes, *cell)`` [m] (``rwet`` -- Lin & Leaitch reads
            the *wet*, not dry, radius: mo_ham_activ.f90:613,664-675).
        sigma_g: ``(n_modes,)`` geometric standard deviation [-].
        updraft: ``(*cell)`` [m/s], the single-bin stratiform updraft.

    Returns:
        ``(available_strat, available_conv, cdncact_strat, cdncact_conv)``,
        all ``(*cell)``.

    """
    ndim_cell = updraft.ndim
    can_activate_col = mode_col(can_activate, ndim_cell).astype(bool)
    sigmaln = mode_col(jnp.log(sigma_g), ndim_cell)
    na = _lin_leaitch_available(number_vol, can_activate_col, wet_radius, sigmaln, LL_CRCUT_STRAT)
    na_cv = _lin_leaitch_available(number_vol, can_activate_col, wet_radius, sigmaln, LL_CRCUT_CONV)
    cdncact = _lin_leaitch_activate(na, updraft)
    cdncact_cv = _lin_leaitch_activate(na_cv, updraft)
    return na, na_cv, cdncact, cdncact_cv
