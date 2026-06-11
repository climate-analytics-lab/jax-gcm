"""Lohmann & Diehl (2006) number-based heterogeneous freezing (#494).

The classic ECHAM-HAM scheme, adapted to a diagnostic frozen-number form:

* **Immersion** — Lohmann & Diehl (2006), as in HAMMOZ
  ``mo_cloud_micro_2m.f90``: the immersion freezing of soluble dust/BC scales
  with the species coefficients ``a_du = 32.3`` (montmorillonite) ≫
  ``a_bc = 2.91e-3``, the temperature factor ``exp(T_melt − T)``, and the
  cooling rate (ascent). Here the frozen *fraction* of the soluble IN over the
  step is ``1 − exp(−a·exp(T_melt−T)·|cooling|·Δt·C)``, capped by the IN number.
  The relative dust≫BC, colder-is-more, and ascent dependences are L&D's; the
  absolute magnitude is set by the calibration constant ``C`` and the
  differentiable ``scale`` (the diagnostic form drops HAMMOZ's per-droplet
  volume factor).
* **Deposition** — Meyers et al. (1992) condensation/deposition number,
  ``n_dep = 10³·exp(−0.639 + 0.1296·100·(S_ice−1))`` [m⁻³], active where
  ice-supersaturated, capped by the insoluble dust/BC number.

Returns the heterogeneous ice-crystal number [m⁻³].
"""

from __future__ import annotations

import jax.numpy as jnp

from jcm.physics.aerosol.jam.ice_nucleation.params import IceNucleationParameters

_TMELT = 273.15
# Lohmann & Diehl (2006) immersion coefficients (HAMMOZ).
_A_IMM_DU = 32.3        # montmorillonite
_A_IMM_BC = 2.91e-3     # black carbon
# Calibration bringing the diagnostic frozen fraction into a physical range
# (absorbs HAMMOZ's per-droplet liquid-volume factor that this diagnostic form
# drops). Chosen so dust immersion ramps from ~0 near −10 °C to near-complete by
# ~−35 °C at typical cooling rates; tune together with the ``scale`` param.
_LD_CALIB = 1.0e-10
# Meyers et al. (1992) deposition number coefficients (per litre → ×1e3 = m⁻³).
_MEYERS_A = -0.639
_MEYERS_B = 0.1296


def _immersion_fraction(coeff, temperature, cooling, dt, scale):
    """Frozen fraction of an immersion-IN population (L&D dependences)."""
    temp_factor = jnp.exp(jnp.clip(_TMELT - temperature, 0.0, 60.0))
    rate = scale * _LD_CALIB * coeff * temp_factor * jnp.maximum(cooling, 0.0)
    return 1.0 - jnp.exp(-jnp.clip(rate * dt, 0.0, 50.0))


def lohmann_diehl_inp(pops, temperature, s_ice, cooling, dt,
                      params: IceNucleationParameters):
    """Heterogeneous INP number [m⁻³] (immersion + deposition)."""
    t = temperature
    cold = t < _TMELT

    # --- Immersion: soluble dust + BC, L&D temperature/cooling dependence ---
    f_du = _immersion_fraction(_A_IMM_DU, t, cooling, dt, params.scale)
    f_bc = _immersion_fraction(_A_IMM_BC, t, cooling, dt, params.scale)
    inp_imm = jnp.where(
        cold, pops["du_number_sol"] * f_du + pops["bc_number_sol"] * f_bc, 0.0
    )

    # --- Deposition: Meyers (1992) on insoluble IN, ice-supersaturated ---
    si_excess = jnp.maximum(s_ice - 1.0, 0.0)
    n_meyers = params.deposition_scale * 1.0e3 * jnp.exp(
        _MEYERS_A + _MEYERS_B * jnp.clip(100.0 * si_excess, 0.0, 100.0)
    )
    insol_number = pops["du_number_insol"] + pops["bc_number_insol"]
    inp_dep = jnp.where(si_excess > 0.0, jnp.minimum(n_meyers, insol_number), 0.0)

    return params.scale * (inp_imm + inp_dep)
