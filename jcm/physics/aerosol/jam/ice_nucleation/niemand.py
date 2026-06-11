"""Niemand-family singular (active-site) heterogeneous freezing (#494).

Surface-area-based immersion + deposition INP, the modern "ns" approach:

* **Immersion** — Niemand et al. (2012) desert-dust active-site density
  ``ns_imm(T) = exp(−0.517·(T−273.15) + 8.934)`` [m⁻²] (valid ~237–261 K). The
  immersion-frozen number is the active-site count capped by the available
  particle number: ``INP = min(ns·A_sol, N_sol)``. BC contributes with a
  reduced efficiency ``bc_efficiency``.
* **Deposition** — a simplified, calibratable active-site density that rises
  monotonically with ice supersaturation (Ullrich-2017-style), active on the
  insoluble (bare) dust/BC where ``S_ice > 1`` and ``T`` is cold:
  ``ns_dep = deposition_scale · ns_dep0 · max(S_ice−1, 0)``. Coefficients are
  exposed for calibration rather than claiming exact published values.

Both contributions are diagnostic and smooth (differentiable). Returns the
heterogeneous ice-crystal number [m⁻³].
"""

from __future__ import annotations

import jax.numpy as jnp

from jcm.physics.aerosol.jam.ice_nucleation.params import IceNucleationParameters

# Niemand (2012) immersion active-site density coefficients.
_NS_IMM_A = -0.517
_NS_IMM_B = 8.934
# Simplified deposition: reference active-site density [m⁻²] per unit S_ice
# excess, gated below this temperature (cirrus/cold mixed-phase).
_NS_DEP0 = 1.0e9
_T_DEP_MAX = 260.0   # K — deposition only colder than this
_T_IMM_MAX = 273.15  # K — immersion only below freezing


def ns_imm_niemand(temperature: jnp.ndarray) -> jnp.ndarray:
    """Niemand (2012) immersion active-site density [m⁻²]."""
    return jnp.exp(_NS_IMM_A * (temperature - 273.15) + _NS_IMM_B)


def niemand_inp(pops, temperature, s_ice, params: IceNucleationParameters):
    """Immersion and deposition INP numbers [m⁻³] as ``(immersion, deposition)``."""
    t = temperature
    cold = t < _T_IMM_MAX

    # --- Immersion (soluble dust + BC), active-site count capped by number ---
    ns_imm = ns_imm_niemand(t)
    sites_du = ns_imm * pops["du_area_sol"]
    sites_bc = params.bc_efficiency * ns_imm * pops["bc_area_sol"]
    inp_imm_du = jnp.minimum(sites_du, pops["du_number_sol"])
    inp_imm_bc = jnp.minimum(sites_bc, pops["bc_number_sol"])
    inp_imm = jnp.where(cold, inp_imm_du + inp_imm_bc, 0.0)

    # --- Deposition (insoluble dust + BC), ice-supersaturated and cold ---
    si_excess = jnp.maximum(s_ice - 1.0, 0.0)
    ns_dep = params.deposition_scale * _NS_DEP0 * si_excess
    sites_du_dep = ns_dep * pops["du_area_insol"]
    sites_bc_dep = params.bc_efficiency * ns_dep * pops["bc_area_insol"]
    inp_dep_du = jnp.minimum(sites_du_dep, pops["du_number_insol"])
    inp_dep_bc = jnp.minimum(sites_bc_dep, pops["bc_number_insol"])
    active_dep = (si_excess > 0.0) & (t < _T_DEP_MAX)
    inp_dep = jnp.where(active_dep, inp_dep_du + inp_dep_bc, 0.0)

    return params.scale * inp_imm, params.scale * inp_dep
