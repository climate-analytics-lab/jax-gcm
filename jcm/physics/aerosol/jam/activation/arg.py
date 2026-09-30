"""Abdul-Razzak & Ghan (2000) modal aerosol activation.

Closed-form maximum-supersaturation activation for a log-normal modal
population — a faithful port of CAM's ``ndrop.F90`` (Abdul-Razzak & Ghan
2000) fed MAM4 modal properties (volume-weighted κ mixing per mode; a single
sqrt(2·TKE/3) characteristic updraft). This is deliberately the κ-Köhler CAM
form, not HAMMOZ ``mo_ham_activ::ham_activ_abdulrazzak_ghan``, which builds a
van't Hoff Köhler ``B`` coefficient and uses 0.7·sqrt(TKE); the CAM form is
correct as written for MAM4 κ inputs.
Two variants of the size-dependent shape coefficients are provided:

* ``"arg2000"`` — the original Abdul-Razzak & Ghan (2000) coefficients.
* ``"ghosh2025"`` — the 3-coefficient revision of Ghosh et al. (2025,
  *Geosci. Model Dev.* 18, 4899) that removes the activated-fraction bias
  at narrow/broad mode widths and the unphysical suppression of activation
  in polluted (kinetically limited) conditions. Code-trivial drop-in: it
  only changes the ``f``, ``g`` and the ``(ζ/η)`` exponent ``p``.

Everything here is closed-form and differentiable (``erf`` is JAX-native);
the variant is chosen at compose time so the ζ/η branch in ``ghosh2025``
is the only data-dependent switch, handled with ``jnp.where``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import erf

# Constants are read through the module alias and never bound with
# ``from jcm.constants import grav``: a from-import captures the float at
# import time, so a later ``set_constants`` override (another planet, a
# sensitivity study, gradient calibration) would silently never reach this
# scheme while the dynamics used the new value (#772).
import jcm.constants as c

# Ghosh et al. (2025) σ_acc validity range.
_SIGMA_ACC_LO = 1.4
_SIGMA_ACC_HI = 2.1

# CAM ``ndrop.F90``'s reference state for its droplet-growth fits: its local
# ``tmelt = 273._r8`` (NOT the 273.15 K melting point in ``jcm.constants``) and
# ``p0 = 1013.25e2_r8``. Part of the fits' definition, so kept with the scheme.
_T_REF = 273.0
_P_REF = 101325.0
# Pruppacher & Klett (13.18) slope of the dry-air conductivity fit,
# ``k = (5.69 + 0.017·(T − T_ref))·1e-5 cal/cm/s/K``, as a fraction of the
# 273 K value ``c.air_thermal_conductivity``.
_KA_SLOPE = 0.017 / 5.69


def vapor_diffusivity(temperature: jnp.ndarray, pressure: jnp.ndarray) -> jnp.ndarray:
    """Water-vapour diffusivity in air [m²/s] at the local T, p.

    CAM ``ndrop.F90::activate_modal``
    ``diff0 = 0.211e-4·(p0/pres)·(tair/tmelt)**1.94`` (Pruppacher & Klett
    13.3; the same form as ECHAM-HAM ``mo_ham_activ`` ``zdif``). At 500 hPa
    and 260 K it is ~1.8x the sea-level value.
    """
    return (c.vapor_diffusivity * (_P_REF / pressure)
            * (temperature / _T_REF) ** 1.94)


def air_thermal_conductivity(temperature: jnp.ndarray) -> jnp.ndarray:
    """Thermal conductivity of (dry) air [W/m/K] at the local T.

    CAM ``ndrop.F90::activate_modal``
    ``conduct0 = (5.69 + 0.017·(tair − tmelt))·4.186e2·1e-5`` (Pruppacher &
    Klett 13.18). CAM takes the dry-air value; ECHAM-HAM's ``zk`` adds a
    moist-air correction, which the CAM form this scheme ports omits.
    """
    return c.air_thermal_conductivity * (1.0 + _KA_SLOPE * (temperature - _T_REF))


def _saturation_vapor_pressure(temperature: jnp.ndarray) -> jnp.ndarray:
    """Saturation vapour pressure over liquid water [Pa].

    Magnus-Tetens form with the empirical WMO/Alduchov & Eskridge (1996)
    coefficients: ``es = 611.2·exp(17.62·t_c / (t_c + 243.12))`` with
    ``t_c`` in °C (here ``t_c + 243.12 = T − 30.03`` for ``T`` in K). The
    three numbers are the standard empirical Magnus fit, not derivable from
    fundamental constants.
    """
    t_c = temperature - 273.15
    return 611.2 * jnp.exp(17.62 * t_c / (temperature - 30.03))


def _shape_coefficients(
    ln_sigma: jnp.ndarray,
    zeta_over_eta: jnp.ndarray,
    sigma_acc: float,
    variant: str,
):
    """Return ``(f, g, p)`` for the chosen ARG variant.

    ``f``/``g`` multiply the two ARG terms; ``p`` is the exponent on
    ``ζ/η`` (3/2 in the original scheme). For ``ghosh2025`` the three are
    functions of the accumulation-mode width ``sigma_acc`` (applied to all
    modes, per the paper), and ``p`` switches in the kinetically limited
    ``ζ/η > 1`` regime.

    The ``variant`` branch is a *compile-time static* dispatch — ``variant``
    is a plain Python string fixed at compose time, so this remains fully
    jittable (the branch is resolved during tracing, not at run time). Only
    the ``jnp.where`` on ``ζ/η`` is a traced, data-dependent switch.
    """
    if variant == "arg2000":
        f = 0.5 * jnp.exp(2.5 * ln_sigma ** 2)
        g = 1.0 + 0.25 * ln_sigma
        p = jnp.asarray(1.5)
        return f, g, p
    if variant == "ghosh2025":
        # NOTE: these coefficients are *reconstructed* by fitting the
        # closed forms to Ghosh et al. (2025) Table 3 (σ_acc, f, g, p):
        #   (1.4, 0.0109, 0.6608, 0.0462) … (2.1, 0.0172, 0.4368, 0.7226).
        # They reproduce that table (f,g exact-ish; p to ~0.01) but were
        # NOT taken from the paper's equation text — verify against the
        # published PDF before any scientific use. Gated off by default.
        s = min(max(sigma_acc, _SIGMA_ACC_LO), _SIGMA_ACC_HI)
        f = 0.004377 * jnp.exp(0.6517 * s)
        g = 1.1088 - 0.32 * s
        p_lim = -3.4966 + 3.5734 * s - 0.74488 * s ** 2
        # Kinetic-limit branch: p = p_lim for ζ/η ≤ 1, else 1.5 (smooth where).
        p = jnp.where(zeta_over_eta <= 1.0, p_lim, 1.5)
        return jnp.asarray(f), jnp.asarray(g), p
    raise ValueError(f"Unknown ARG variant {variant!r}.")


def arg_activation(
    r_dry: jnp.ndarray,        # (M, nlev, ncols) number-mode dry radius [m]
    kappa: jnp.ndarray,        # (M, nlev, ncols) hygroscopicity κ [-]
    number_vol: jnp.ndarray,   # (M, nlev, ncols) number concentration [m^-3]
    sigma_g: jnp.ndarray,      # (M, 1, 1) geometric std dev per mode [-]
    can_activate: jnp.ndarray, # (M, 1, 1) 0/1 mask
    updraft: jnp.ndarray,      # (nlev, ncols) updraft velocity [m/s]
    temperature: jnp.ndarray,  # (nlev, ncols) [K]
    pressure: jnp.ndarray,     # (nlev, ncols) [Pa]
    sigma_acc: float,
    *,
    variant: str = "arg2000",
):
    """ARG closed-form activation.

    Returns ``(activated_cdnc, activated_fraction, s_max, number_frac,
    mass_frac)``:
      * ``activated_cdnc``  (nlev, ncols) total activated number [m^-3]
      * ``activated_fraction`` (nlev, ncols) number-weighted fraction [-]
      * ``s_max`` (nlev, ncols) maximum supersaturation [-]
      * ``number_frac`` (M, nlev, ncols) per-mode activated number fraction,
        zero for non-activatable modes
      * ``mass_frac`` (M, nlev, ncols) per-mode activated mass fraction
        (the log-normal number erf shifted by ``3·lnσ/√2``), same masking
    """
    # Floor the modal number at 0: spectral advection of the aerosol-number
    # tracers leaves small NEGATIVE number on the near-zero cold-start field
    # (Gibbs ringing). A negative ``number_vol`` makes the number-weighted
    # ``n_total`` negative, collapsing ``activated_fraction = n_act / n_total``
    # to a ±huge value (and the per-cell ``activated_cdnc`` negative). The cloud
    # masks the negative CDNC via the SPA floor and wet deposition clamps the
    # rate, so the model stays finite — but the activated fraction handed to wet
    # scavenging is garbage. Flooring here makes every derived quantity physical:
    # n_act, n_total ≥ 0 ⇒ activated_fraction ∈ [0, 1], activated_cdnc ≥ 0. Same
    # ringing root cause as the JAM-optics number floor (#543).
    number_vol = jnp.maximum(number_vol, 0.0)
    # Bound here, per call, rather than at import: the short aliases keep the
    # ARG formulae below readable while still reading the live constants (see
    # the import note). _RGAS is the *universal* gas constant (J/mol/K),
    # distinct from the per-mass dry-air constant.
    _LV, _CP, _G = c.alhc, c.cpd, c.grav
    _MA, _MW, _RGAS, _RHOW = c.m_air, c.m_water, c.r_universal, c.rhow
    _SIGMA_W, _TINY = c.surface_tension_water, c.tiny
    t = temperature
    p = pressure
    es = _saturation_vapor_pressure(t)
    w = jnp.maximum(updraft, 1.0e-3)
    # Local-state transport coefficients (CAM diff0/conduct0): the constant
    # sea-level values under-state Dv aloft (1/p) and so over-state the growth
    # resistance, biasing activation high by ~4 % at 900 hPa to ~20 % at
    # 500 hPa (#679).
    dv = vapor_diffusivity(t, p)
    ka = air_thermal_conductivity(t)

    # Kelvin coefficient A [m] and condensation growth coefficient G [m²/s].
    # A is CAM's ``aten = 2·mwh2o·surften/(r_universal·tmelt·rhoh2o)``: the
    # surface tension and the temperature are both CAM's fixed reference
    # values, so A — and every mode's critical supersaturation — is the same
    # at every level, exactly as in ``ndrop.F90``.
    a_kelvin = 2.0 * _SIGMA_W * _MW / (_RHOW * _RGAS * _T_REF)
    g_growth = 1.0 / (
        (_RHOW * _RGAS * t) / (es * dv * _MW)
        + (_LV * _RHOW / (ka * t)) * (_LV * _MW / (_RGAS * t) - 1.0)
    )

    alpha = (_G * _MW * _LV) / (_CP * _RGAS * t ** 2) - (_G * _MA) / (_RGAS * t)
    gamma = (_RGAS * t) / (es * _MW) + (_MW * _LV ** 2) / (_CP * p * _MA * t)

    aw_over_g = alpha * w / g_growth                     # (nlev, ncols)
    zeta = (2.0 / 3.0) * a_kelvin * jnp.sqrt(aw_over_g)  # (nlev, ncols)

    # Per-mode critical supersaturation Sm_i = sqrt(4 A³ / (27 κ r³)).
    kappa_s = jnp.maximum(kappa, 1.0e-10)
    r_s = jnp.maximum(r_dry, 1.0e-10)
    sm = jnp.sqrt(
        4.0 * a_kelvin ** 3 / (27.0 * kappa_s * r_s ** 3)
    )                                                    # (M, nlev, ncols)

    n_s = jnp.maximum(number_vol, _TINY)
    eta = (aw_over_g ** 1.5) / (2.0 * jnp.pi * _RHOW * gamma * n_s)

    ln_sigma = jnp.log(sigma_g)                          # (M, 1, 1)
    f_co, g_co, p_exp = _shape_coefficients(
        ln_sigma, zeta / jnp.maximum(eta, _TINY), sigma_acc, variant,
    )

    # 1/Smax² = Σ_i (mask_i / Sm_i²)[ f (ζ/η_i)^p + g (Sm_i²/(η_i+3ζ))^{3/4} ]
    term = (
        f_co * (zeta / jnp.maximum(eta, _TINY)) ** p_exp
        + g_co * (sm ** 2 / (eta + 3.0 * zeta)) ** 0.75
    )
    inv_smax2 = jnp.sum(can_activate * term / sm ** 2, axis=0)
    s_max = 1.0 / jnp.sqrt(jnp.maximum(inv_smax2, _TINY))

    # Activated fraction per mode and total. The mass fraction is the same
    # log-normal integral evaluated against the mass distribution, whose
    # median is ``exp(3·ln²σ)`` above the number median — the standard ARG
    # ``u_m = u − 3·lnσ/√2`` shift.
    u = (2.0 * jnp.log(sm / s_max)) / (3.0 * jnp.sqrt(2.0) * ln_sigma)
    f_act = 0.5 * (1.0 - erf(u))
    f_mass = 0.5 * (1.0 - erf(u - 3.0 * ln_sigma / jnp.sqrt(2.0)))
    n_act = jnp.sum(can_activate * number_vol * f_act, axis=0)

    n_total = jnp.sum(can_activate * number_vol, axis=0)
    activated_fraction = n_act / jnp.maximum(n_total, _TINY)
    return n_act, activated_fraction, s_max, can_activate * f_act, (
        can_activate * f_mass
    )
