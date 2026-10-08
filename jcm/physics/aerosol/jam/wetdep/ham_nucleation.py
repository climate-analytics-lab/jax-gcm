"""ECHAM-HAM's own in-cloud nucleation scavenging: ``ic_scav_nuc`` (#1017 follow-up A).

Stratiform aerosol-size-dependent nucleation scavenging (``kscavICtype=3``,
the branch ``nwetdep=3`` selects -- see ``ham_setscav``, mo_ham_wetdep.f90:
1388-1407) computes a per-mode, per-phase (water/ice), per-tracer-phase
(number/mass) activated/scavenged FRACTION by INVERTING the mode's own
lognormal tail at the critical radius that reproduces the ACTUAL in-cloud
droplet/crystal number this step, then reading the SAME tail forward for the
tracer in question. This replaces jcm's existing implicit treatment (the
activation scheme's OWN activated fraction, ``_jam_activation``) for the
``"ham"`` wetdep scheme only (named ``"ham_nuc_bc"`` while this and
follow-up B landed separately); ``"jcm"``/``"ham_below_cloud"`` keep the
implicit treatment unchanged -- see ``wetdep_term.py``'s module docstring.

ECHAM input -> jcm source
-------------------------
======================  =========================================  ==========================================================
ECHAM (``mo_ham_wetdep.f90`` unless noted)                          jcm source
======================  =========================================  ==========================================================
``pxtp1c_sav(idt_cdnc)``/   in-cloud CDNC/ICNC mixing ratio           ``state.tracers["qnc"]``/``["qni"]`` * ``clouds.
``pxtp1c_sav(idt_icnc)``    (grid-mean * cloud_fraction, "untouched   cloud_fraction`` -- ``WetScavenging`` runs AFTER the
                            by wetdep" since ic_scav_nuc runs          2M microphysics term each step, so these ARE this
                            before THIS term's own removal) (:518,     step's post-microphysics, pre-wetdep tracer values,
                            :560-562)                                 the same "untouched by wetdep" property.
``pxtp1c_sav(idt_nks/       in-cloud KS/AS/CS NUMBER mixing ratio      ``state.tracers[number_name(mode.short)]`` *
nas/ncs)``                 (same convention as CDNC/ICNC above)       ``clouds.cloud_fraction``, ``mode.short in
                                                                       ("ks","as","cs")``
``na``                      WATER phase only: total activated/         ARG: ``diagnostics["activated_cdnc"]`` (confirmed
                            available aerosol number [m-3]             identical -- ``ham_activ_diag_abdulrazzak_ghan_
                            (mo_activ.f90, module var)                 strat``, mo_ham_activ.f90:409-416, sums ``pnact``
                                                                        over activating modes, exactly ``HamActivation``'s
                                                                        own ``cdncact``). Lin-Leaitch: recomputed here via
                                                                        :func:`lin_leaitch_available_number` from
                                                                        ``ham_logtail``/``LL_CRCUT_STRAT`` -- NOT published
                                                                        by ``HamActivation`` (its own ``na`` local is
                                                                        discarded after computing ``cdncact``'s ratio), so
                                                                        reusing the shared primitive rather than a new
                                                                        diagnostic avoids touching that term.
``frac(kmod)``              WATER phase only: per-mode activated      ARG: ``_jam_activation.number_frac[mode]`` (confirmed
                            fraction (mo_ham_streams.f90)              identical -- the SAME ``ham_activ_diag_..._strat``
                                                                        loop sets ``frac(jclass)=pfracn(jclass)``, HAM's own
                                                                        ``ham_arg``/jcm's ``number_frac`` output, with NO
                                                                        further scaling). Lin-Leaitch: recomputed here as
                                                                        the RAW per-mode cutoff tail fraction (``zfracn``,
                                                                        mo_ham_activ.f90:655-662) -- distinct from
                                                                        ``HamActivation``'s PUBLISHED Lin-Leaitch
                                                                        ``number_frac``, which additionally scales by
                                                                        ``cdncact/na`` (that term's own flagged, undocumented-
                                                                        in-Fortran decomposition for the cloud-borne
                                                                        exchange consumer -- a DIFFERENT consumer with a
                                                                        different contract than ``ic_scav_nuc``'s).
``rdry``/``rwet``           per-mode dry/wet radius, selected by       ``aer.r_dry``/``aer.r_wet`` (the M7 state the
                            ``ncd_activ`` (mo_ham_wetdep.f90:          activation scheme itself reads) -- the SAME
                            710-714)                                  selection HamActivation makes internally.
======================  =========================================  ==========================================================

``ham_m7_invertlogtail`` (mo_ham_tools.f90:325-419) is a new port: a
closed-form (Winitzki 2008-style rational) approximation to the inverse
error function, not a root-find, so it is exact-formula faithful rather than
an iterative approximation of one.
"""
from __future__ import annotations

import jax.numpy as jnp

from jcm.physics.aerosol.jam.activation.ham_activation import (
    LL_CRCUT_STRAT,
    ZEPS,
    ham_logtail,
    mode_col,
)

# mo_ham_wetdep.f90:74: REAL(dp), PARAMETER :: zeps_mass = 1.e-30_dp -- the
# "is there measurably any condensate/hydrometeor here at all" gate used
# throughout ic_scav_nuc; distinct from ZEPS (EPSILON(1d0)), which guards
# the divisions below.
ZEPS_MASS = 1.0e-30

# mo_math_constants.f90's pi, read directly rather than importing a whole
# constants module for one literal (mo_ham_tools.f90:372's own za_rcp uses
# the same value HAM does, not jcm.constants, matching ham_activation.py's
# own stance on keeping this reference held to HAM's exact literals).
_PI = 3.14159265358979323846


def _safe(mask, num, den):
    """``num/den`` where ``mask``, ``0`` elsewhere (see ham_activation.py)."""
    return jnp.where(mask, num / jnp.where(mask, den, 1.0), 0.0)


def ham_m7_invertlogtail(count_median_radius, xie, sigmaln):
    """Critical radius whose lognormal tail contains the fraction implied by ``xie``.

    Faithful port of ``ham_m7_invertlogtail`` (mo_ham_tools.f90:325-419):
    inverts ``Tail = N/2 - N/2*erf[ln(R/Rg)/(sqrt(2)*ln(sigma))]`` for R using
    the closed-form rational approximation to ``erf^-1`` quoted in the
    subroutine's own header comment (mo_ham_tools.f90:356-363), rather than a
    numerical root-find.

    Args:
        count_median_radius: the mode's count median radius [m] (``pcmr`` --
            ``rdry``/``rwet`` as selected by the caller).
        xie: the inverse-erf argument (``pxie``), already the ``1 -
            2*fraction`` transform ic_scav_nuc computes.
        sigmaln: ``ln(sigma_g)`` for this mode (``sigmaln(kmod)``).

    Returns:
        The critical radius [m] (``pcritrad``), shaped like the broadcast of
        the inputs.

    """
    a_rcp = (3.0 * _PI * (4.0 - _PI)) / (8.0 * (_PI - 3.0))
    pre_fact = jnp.sqrt(2.0) * sigmaln

    small = jnp.abs(xie) < 1.0
    huge_tail = xie >= 1.0
    negative_small = small & (xie < 0.0)

    x2 = jnp.where(small, xie * xie, 0.0)
    log1mx2 = jnp.log(jnp.where(small, 1.0 - x2, 1.0))  # x2 < 1 strictly inside `small`
    b = 2.0 / _PI * a_rcp + 0.5 * log1mx2
    c = log1mx2 * a_rcp
    y = jnp.sqrt(jnp.where(small, -b + jnp.sqrt(b * b - c), 1.0))
    y = jnp.where(negative_small, -y, y)

    critical_radius = jnp.exp(pre_fact * y) * count_median_radius
    critical_radius = jnp.where(small, critical_radius, 0.0)
    # "Minimal scavenging of the mode by using an artificial large critical
    # radius" (mo_ham_tools.f90:417-418) -- xie >= 1 means the implied tail
    # is empty/negative, so nothing is scavenged.
    return jnp.where(huge_tail, 500.0e-6, critical_radius)


def water_phase_xie(cdnc_incloud, air_density, na, frac):
    """``zxie`` for the water phase (mo_ham_wetdep.f90:749-755,777-778).

    Args:
        cdnc_incloud: in-cloud CDNC mixing ratio (grid-mean * cloud_fraction)
            [kg^-1] (``zxtp1c(idt_cdnc)``).
        air_density: [kg/m3] (``prhop1``).
        na: total activated/available aerosol number [m-3] (see the module
            docstring's ``na`` row).
        frac: this mode's activated fraction [-] (see the module docstring's
            ``frac(kmod)`` row).

    """
    ok = (cdnc_incloud > ZEPS_MASS) & (na > ZEPS)
    ratio = _safe(ok, cdnc_incloud * air_density * frac, na)
    return jnp.where(ok, 1.0 - 2.0 * jnp.clip(ratio, 0.0, 1.0), 1.0)


def ice_phase_xie(icnc_incloud, n_ks, n_as, n_cs, mode_short):
    """``zxie`` for the ice phase, M7's size-ordered depletion (mo_ham_wetdep.f90:757-778).

    The coarse-soluble (CS) mode is assumed to use up ICNC first, then
    accumulation-soluble (AS) the remainder, then Aitken-soluble (KS) what
    is left -- a literal transcription of the three hard-coded ``IF (kmod ==
    ...)`` branches, not a loop (the Fortran itself special-cases each mode
    by its M7 index rather than expressing a general recurrence).

    Args:
        icnc_incloud, n_ks, n_as, n_cs: in-cloud ICNC and the three
            activating modes' own NUMBER mixing ratios (grid-mean *
            cloud_fraction, all the same convention) [kg^-1].
        mode_short: ``"ks"``, ``"as"`` or ``"cs"`` (static Python str).

    """
    if mode_short == "cs":
        ratio = _safe(n_cs > ZEPS, icnc_incloud, n_cs)
    elif mode_short == "as":
        numer = jnp.maximum(icnc_incloud - n_cs, 0.0)
        ratio = _safe(n_as > ZEPS, numer, n_as)
    elif mode_short == "ks":
        numer = jnp.maximum(icnc_incloud - n_cs - n_as, 0.0)
        ratio = _safe(n_ks > ZEPS, numer, n_ks)
    else:
        raise ValueError(f"ice_phase_xie is only defined for ks/as/cs, got {mode_short!r}")
    ratio = jnp.clip(ratio, 0.0, 1.0)
    ok = icnc_incloud > ZEPS_MASS
    return jnp.where(ok, 1.0 - 2.0 * ratio, 1.0)


def nucleation_scavenged_fraction(xie, radius, sigmaln, mass_factor):
    """One phase's ``(number_frac, mass_frac)`` for one mode.

    ``ic_scav_nuc``'s own two-call pattern (mo_ham_wetdep.f90:779-794): a
    single ``rcritrad`` inversion feeds TWO forward ``ham_logtail`` reads
    (number ``mass_factor=1``, mass ``mass_factor=cmedr2mmedr``) — mirroring
    exactly how ``rcritrad`` is cached once per (mode, phase) and reused for
    both tracer phases in the reference. Clipped to [0, 1]
    (mo_ham_wetdep.f90:582-584, ``get_icscavfrac``'s final confinement).
    """
    rcritrad = ham_m7_invertlogtail(radius, xie, sigmaln)
    frac_number = jnp.clip(ham_logtail(radius, rcritrad, sigmaln), 0.0, 1.0)
    frac_mass = jnp.clip(
        ham_logtail(radius, rcritrad, sigmaln, mass_factor=mass_factor), 0.0, 1.0)
    return frac_number, frac_mass


def lin_leaitch_available_number(number_vol, can_activate, wet_radius, sigma_g):
    """Lin & Leaitch's ``na`` (mo_ham_activ.f90:649,705-716): available number
    [m-3] above the stratiform instrument cutoff, summed over activating modes.

    Reuses :func:`ham_activation.ham_logtail`/``LL_CRCUT_STRAT`` rather than
    re-deriving them (the same primitives ``HamActivation``'s own
    ``lin_leaitch`` branch calls for its OWN, different, published ``na``
    row of ``_lin_leaitch_available`` -- that function returns the SAME
    quantity but is private to ``ham_activation.py``; this is a thin,
    intentional duplicate of its three-line body rather than exporting a
    private helper, since the two call sites want it for unrelated
    purposes).
    """
    ndim_cell = wet_radius.ndim - 1
    ln_sigma = mode_col(jnp.log(sigma_g), ndim_cell)
    cut_frac = ham_logtail(wet_radius, jnp.full_like(wet_radius, LL_CRCUT_STRAT), ln_sigma)
    can_activate_col = mode_col(can_activate, ndim_cell).astype(bool)
    return jnp.sum(jnp.where(can_activate_col, number_vol * cut_frac, 0.0), axis=0), cut_frac
