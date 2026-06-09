"""Abdul-Razzak & Ghan (2000) modal aerosol activation.

Closed-form maximum-supersaturation activation for a log-normal modal
population (mirrors HAMMOZ ``mo_ham_activ::ham_activ_abdulrazzak_ghan``).
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

# All physical constants are sourced from jcm.constants.PhysicalConstants
# (short local aliases keep the formulae readable). _RGAS is the *universal*
# gas constant (J/mol/K), distinct from the per-mass dry-air constant.
from jcm.constants import air_thermal_conductivity as _KA
from jcm.constants import alhc as _LV
from jcm.constants import cpd as _CP
from jcm.constants import grav as _G
from jcm.constants import m_air as _MA
from jcm.constants import m_water as _MW
from jcm.constants import r_universal as _RGAS
from jcm.constants import rhow as _RHOW
from jcm.constants import surface_tension_water as _SIGMA_W
from jcm.constants import tiny as _TINY
from jcm.constants import vapor_diffusivity as _DV

# Ghosh et al. (2025) σ_acc validity range.
_SIGMA_ACC_LO = 1.4
_SIGMA_ACC_HI = 2.1


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

    Returns ``(activated_cdnc, activated_fraction, s_max)``:
      * ``activated_cdnc``  (nlev, ncols) total activated number [m^-3]
      * ``activated_fraction`` (nlev, ncols) number-weighted fraction [-]
      * ``s_max`` (nlev, ncols) maximum supersaturation [-]
    """
    t = temperature
    p = pressure
    es = _saturation_vapor_pressure(t)
    w = jnp.maximum(updraft, 1.0e-3)

    # Kelvin coefficient A [m] and condensation growth coefficient G [m²/s].
    a_kelvin = 2.0 * _SIGMA_W * _MW / (_RHOW * _RGAS * t)
    g_growth = 1.0 / (
        (_RHOW * _RGAS * t) / (es * _DV * _MW)
        + (_LV * _RHOW / (_KA * t)) * (_LV * _MW / (_RGAS * t) - 1.0)
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

    # Activated fraction per mode and total.
    u = (2.0 * jnp.log(sm / s_max)) / (3.0 * jnp.sqrt(2.0) * ln_sigma)
    f_act = 0.5 * (1.0 - erf(u))
    n_act = jnp.sum(can_activate * number_vol * f_act, axis=0)

    n_total = jnp.sum(can_activate * number_vol, axis=0)
    activated_fraction = n_act / jnp.maximum(n_total, _TINY)
    return n_act, activated_fraction, s_max
